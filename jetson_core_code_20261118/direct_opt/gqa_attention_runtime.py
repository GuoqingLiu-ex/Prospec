"""Strict, opt-in Target tree GQA backends before HF's repeat_kv.

No model/GPU work runs on import. cudnn selects ONLY CUDNN_ATTENTION; flex
compiles the general FlexAttention kernel (group 28/4=7 is incompatible with
its short-query decoding kernel). Neither catches errors to run another backend.
The original additive mask and already-rotated Q/K pass through unchanged.
Fused arithmetic is not guaranteed bitwise equal to the existing math backend.

The policy patches the Qwen attention registry locally, then restores the exact
prior entry. Only the specified Target's language self-attention modules inside
an active ring verification are selected. Prefill, vision and Draft pass through.
Capture mode calls the original attention and optionally saves real operands;
its CPU transfers invalidate performance/memory readings from that request.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import torch


def tensor_description(x):
    return dict(shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype),
                device=str(x.device), contiguous=x.is_contiguous())


def validate_attention(query, key, value, mask):
    tensors=(query,key,value,mask)
    if any(not isinstance(x,torch.Tensor) for x in tensors):
        raise ValueError('Q/K/V and explicit additive mask tensors are required')
    if any(x.ndim!=4 for x in tensors) or query.shape[0]!=1:
        raise ValueError('batch-one four-dimensional tensors required')
    if key.shape!=value.shape or key.shape[0]!=1 or query.shape[-1]!=key.shape[-1]:
        raise ValueError('Q/K/V shape mismatch')
    if min(query.shape[1:])<1 or min(key.shape[1:])<1 or query.shape[1]%key.shape[1]:
        raise ValueError('positive sequence/head sizes and integral GQA group required')
    if mask.shape!=(1,1,query.shape[-2],key.shape[-2]):
        raise ValueError('additive tree mask must be [1,1,Q,K] at actual attention width')
    if any(x.device!=query.device for x in tensors) or any(x.dtype!=query.dtype for x in (key,value)):
        raise ValueError('Q/K/V must share dtype; all operands must share device')
    if mask.dtype not in (query.dtype,torch.float32):
        raise ValueError('additive mask must use Q dtype or the native FP32 mask dtype')
    if not query.is_floating_point() or any(x.requires_grad for x in tensors):
        raise ValueError('floating-point inference tensors required')
    if any(x.stride(-1)!=1 for x in tensors):
        raise ValueError('last dimension must be contiguous; no hidden input packing is performed')


def _flex_with_bias(query,key,value,mask,scale):
    from torch.nn.attention.flex_attention import flex_attention
    # Batch/head mask dimensions are both one by the validated contract.
    # Slice them outside score_mod: mixing a Python integer with scalar tensor
    # indices inside the higher-order graph triggers scalar extraction in 2.8.
    bias_matrix=mask[0,0]
    def score_mod(score,b,h,q,k):
        # Triton may evaluate padded tile indices. Clamp the lookup before
        # applying the validity test; never read past the original mask.
        qi=q.clamp(0,mask.shape[-2]-1)
        ki=k.clamp(0,mask.shape[-1]-1)
        bias=bias_matrix[qi,ki]
        valid=(q<mask.shape[-2]) & (k<mask.shape[-1])
        return torch.where(valid,score+bias,float('-inf'))
    return flex_attention(query,key,value,score_mod=score_mod,scale=scale,
                          enable_gqa=True,return_lse=False,
                          kernel_options={'FORCE_USE_FLEX_ATTENTION':True})


class GQAAttentionBackend:
    def __init__(self,backend):
        if backend not in ('cudnn','flex'):
            raise ValueError('choose cudnn or flex explicitly; no automatic fallback')
        self.backend=backend
        self._compiled=None
        self.attempted=0
        self.completed=0
        self.failures=[]
        self.shapes={}
        self.compile_wrapper_prepare_wall_s=0.0

    def __call__(self,query,key,value,mask,scale=None):
        self.attempted+=1
        signature=str((tuple(query.shape),tuple(key.shape),tuple(key.stride()),str(query.dtype)))
        self.shapes[signature]=self.shapes.get(signature,0)+1
        try:
            validate_attention(query,key,value,mask)
            if query.device.type!='cuda' or query.dtype not in (torch.float16,torch.bfloat16):
                raise ValueError('fused candidate requires CUDA FP16/BF16; CPU is validation-only')
            if scale is None:scale=query.shape[-1]**-.5
            if not isinstance(scale,(float,int)) or not math.isfinite(scale) or scale<=0:
                raise ValueError('positive finite scalar attention scale required')
            if self.backend=='cudnn':
                from torch.nn.attention import sdpa_kernel,SDPBackend
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    output=torch.nn.functional.scaled_dot_product_attention(query,key,value,
                        attn_mask=mask,dropout_p=0.0,is_causal=False,scale=float(scale),enable_gqa=True)
            else:
                if self._compiled is None:
                    start=time.perf_counter()
                    self._compiled=torch.compile(_flex_with_bias,fullgraph=True,dynamic=True)
                    self.compile_wrapper_prepare_wall_s=time.perf_counter()-start
                # First actual invocation includes compilation. The outer probe
                # records its synchronized cost; no warm/cold success is hidden.
                output=self._compiled(query,key,value,mask,float(scale))
            if output.shape!=query.shape or output.dtype!=query.dtype or output.device!=query.device:
                raise RuntimeError('backend output shape/dtype/device differs from the input contract')
            self.completed+=1
            return output
        except Exception as exc:
            self.failures.append(dict(type=type(exc).__name__,message=str(exc),shape=signature))
            raise RuntimeError(f'{self.backend} GQA failed; no fallback was executed: {exc}') from exc

    def metadata(self):
        return dict(requested_backend=self.backend,backend_constraint=('CUDNN_ATTENTION only' if self.backend=='cudnn' else
                    'compiled FlexAttention; FORCE_USE_FLEX_ATTENTION=True'),
                    attempted_calls=self.attempted,completed_calls=self.completed,failures=list(self.failures),
                    shapes=dict(self.shapes),fallback_calls=0,actual_kernel_trace_required=True,
                    wrapper_compile_prepare_wall_s=self.compile_wrapper_prepare_wall_s,
                    compile_timing_note='Wrapper construction only; actual first-call compilation must be timed externally',
                    explicit_repeat_kv=False,quality_or_speed_claim=False)


@contextmanager
def gqa_attention_policy(target,backend='cudnn',*,capture=None,interface=None,state_provider=None,executor=None):
    """Context surrounding a real request; nested registry policies rejected.

    Pass a shared executor to reuse Flex compilation across request contexts.
    Executor counters are cumulative; use snapshots/deltas per request.
    """
    if backend not in ('cudnn','flex','capture'):
        raise ValueError('backend must be cudnn, flex or explicit native capture')
    modules=tuple(layer.self_attn for layer in target.model.language_model.layers)
    module_ids={id(m) for m in modules}
    if not modules or len(module_ids)!=len(modules):raise ValueError('unique Target self-attention modules required')
    if interface is None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
        interface=ALL_ATTENTION_FUNCTIONS
    if not hasattr(interface,'_local_mapping'):
        raise TypeError('HF AttentionInterface with local mapping is required')
    previous=interface['sdpa']
    if getattr(previous,'_strict_gqa_policy',False):raise RuntimeError('nested GQA policies unsupported')
    had_local='sdpa' in interface._local_mapping
    local_value=interface._local_mapping.get('sdpa')
    if state_provider is None:
        import ring_cache_runtime
        state_provider=lambda:ring_cache_runtime._ACTIVE
    if executor is not None and (not isinstance(executor,GQAAttentionBackend) or executor.backend!=backend):
        raise ValueError('shared executor must match the explicitly selected backend')
    executor=executor if executor is not None else (None if backend=='capture' else GQAAttentionBackend(backend))
    record=dict(backend=backend,selected_calls=0,passthrough_calls=0,capture_invalidates_timing=bool(capture),
                fallback_calls=0,failures=[])

    def attention(module,query,key,value,attention_mask,dropout=0.0,scaling=None,
                  is_causal=None,position_bias=None,**kwargs):
        state=state_provider()
        selected=(id(module) in module_ids and state is not None and
                  state.target is target and state.in_forward)
        if not selected:
            record['passthrough_calls']+=1
            return previous(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,
                            is_causal=is_causal,position_bias=position_bias,**kwargs)
        record['selected_calls']+=1
        try:
            validate_attention(query,key,value,attention_mask)
            if dropout!=0.0 or position_bias is not None or kwargs.get('output_attentions',False):
                raise ValueError('tree GQA requires no dropout/position bias/output attentions')
            # The additive tree mask is authoritative. Never substitute a
            # physical-slot causal triangle; is_causal defaults on HF modules.
            if backend=='capture':
                output,weights=previous(module,query,key,value,attention_mask,dropout=dropout,
                    scaling=scaling,is_causal=is_causal,position_bias=position_bias,**kwargs)
            else:
                output=executor(query,key,value,attention_mask,scaling).transpose(1,2).contiguous()
                weights=None
            if capture is not None:
                capture(module,query,key,value,attention_mask,output,scaling,
                        dict(backend=backend,logical_base=state.pending['logical_base'] if state.pending else None))
            return output,weights
        except Exception as exc:
            record['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
            raise
    attention._strict_gqa_policy=True
    interface['sdpa']=attention
    try:
        yield record
    finally:
        if executor is not None:record['executor']=executor.metadata()
        if had_local:interface['sdpa']=local_value
        else:del interface['sdpa']


class AttentionCapture:
    """Bounded real-input capture, deliberately outside any timing claim."""
    def __init__(self,directory,*,layer=0,rounds=(0,),limit=3):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.layer=layer;self.rounds=set(rounds);self.limit=limit
        self.calls={};self.records=[]

    def __call__(self,module,q,k,v,mask,output,scale,context):
        if context['backend']!='capture':raise ValueError('reference captures must use original native attention')
        index=int(module.layer_idx)
        ordinal=self.calls.get(index,0);self.calls[index]=ordinal+1
        if index!=self.layer or ordinal not in self.rounds or len(self.records)>=self.limit:return
        original=dict(query=q,key=k,value=v,mask=mask,native_output=output)
        description={name:tensor_description(t) for name,t in original.items()}
        filename=f'attention_{len(self.records):03d}.pt'
        sample={name:t.detach().to('cpu').contiguous() for name,t in original.items()}
        meta=dict(file=filename,layer=index,layer_call_index=ordinal,scale=scale,context=context,
                  tensors=description,timing_valid=False,memory_valid=False)
        torch.save(dict(tensors=sample,metadata=meta),self.directory/filename)
        self.records.append(meta)
        (self.directory/'summary.json').write_text(json.dumps(dict(schema='real_tree_attention_v1',
            captures=self.records,capture_calls=self.calls,reference='HF native math; original 4-head KV captured before repeat',
            timing_valid=False,memory_valid=False),indent=2)+'\n')
