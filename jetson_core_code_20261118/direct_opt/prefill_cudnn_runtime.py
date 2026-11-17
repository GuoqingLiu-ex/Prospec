"""Opt-in Target language prefill causal cuDNN GQA; no model work on import.

Only one empty-cache, batch-one, unpadded complete prompt is selected. Vision,
Draft and tree/decode retain their existing attention path. Real attention must
have mask=None, Q=K=prompt, FP16 28/4 heads and D128; otherwise abort. No fallback.
Capture mode keeps native math and is diagnostic, invalidating timing/memory.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

import ring_cache_runtime as bridge_runtime
from gqa_attention_runtime import tensor_description


def validate_prefill_attention(query,key,value,mask,prompt_tokens):
    if mask is not None:raise ValueError('language prefill requires an actual None attention mask')
    if not all(isinstance(x,torch.Tensor) for x in (query,key,value)):
        raise ValueError('actual Q/K/V tensors required')
    if query.shape!=(1,28,prompt_tokens,128) or key.shape!=(1,4,prompt_tokens,128) or value.shape!=key.shape:
        raise ValueError('complete-prompt causal Qwen 28Q/4KV/D128 shapes required')
    if (prompt_tokens<2 or any(x.dtype!=torch.float16 or x.device!=query.device or
            x.requires_grad or x.stride(-1)!=1 for x in (query,key,value))):
        raise ValueError('same-device FP16 inference tensors with contiguous last axis required')


class CausalCudnnExecutor:
    backend='cudnn'
    def __init__(self):self.attempted=0;self.completed=0;self.failures=[]
    def __call__(self,query,key,value,scale):
        self.attempted+=1
        try:
            validate_prefill_attention(query,key,value,None,query.shape[-2])
            if query.device.type!='cuda':raise ValueError('cuDNN prefill requires CUDA; no CPU fallback')
            if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale) or scale<=0:
                raise ValueError('positive finite captured attention scale required')
            from torch.nn.attention import sdpa_kernel,SDPBackend
            # This point is inside the frozen Target's outer MATH context.
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                result=torch.nn.functional.scaled_dot_product_attention(query,key,value,
                    attn_mask=None,is_causal=True,dropout_p=0.0,scale=float(scale),enable_gqa=True)
            if result.shape!=query.shape or result.dtype!=query.dtype or result.device!=query.device:
                raise RuntimeError('cuDNN prefill output contract mismatch')
            self.completed+=1
            return result
        except Exception as exc:
            self.failures.append(dict(type=type(exc).__name__,message=str(exc)))
            raise
    def metadata(self):
        return dict(backend='CUDNN_ATTENTION only',attempted_calls=self.attempted,completed_calls=self.completed,
                    failures=list(self.failures),fallback_calls=0,explicit_repeat_kv=False,
                    scope='Target language complete-prompt prefill only; actual trace required')


class PrefillAttentionCapture:
    def __init__(self,directory,*,layers=(0,),limit=None):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        if any(type(i) is not int or i<0 for i in layers):raise ValueError('nonnegative capture layer indices required')
        self.layers=set(layers);self.limit=len(self.layers) if limit is None else limit
        if type(self.limit) is not int or self.limit<1:raise ValueError('positive capture limit required')
        self.records=[]
    def __call__(self,module,q,k,v,mask,output,scale,context):
        if context['mode']!='capture' or mask is not None:raise ValueError('native causal prefill capture only')
        if context['layer_index'] not in self.layers or len(self.records)>=self.limit:return
        tensors={name:x.detach().cpu().clone() for name,x in
                 [('query',q),('key',k),('value',v),('native_output',output)]}
        metadata=dict(schema='real_language_prefill_attention_v1',mask=None,is_causal=True,enable_gqa=True,
            scale=scale,context=dict(context),timing_valid=False,memory_valid=False,
            tensors={name:tensor_description(x) for name,x in
                     [('query',q),('key',k),('value',v),('native_output',output)]})
        filename=f'prefill_attention_{len(self.records):03d}.pt'
        torch.save(dict(tensors=tensors,metadata=metadata),self.directory/filename)
        self.records.append(dict(file=filename,metadata=metadata))
        (self.directory/'summary.json').write_text(json.dumps(dict(schema='real_language_prefill_attention_v1',
            diagnostic_only=True,captures=self.records),indent=2)+'\n')


@contextmanager
def prefill_cudnn_policy(target,*,mode='cudnn',capture=None,interface=None,executor=None):
    """Wrap one native bridge prefill; can surround an inner tree/KV policy.

    The inner KV runtime captures this _prefill and calls it before conversion.
    Its registry must pass through non-tree calls, as current visual KV does.
    Native-position tensors are inspected once, passed through without editing.
    """
    if mode not in ('cudnn','capture'):raise ValueError('choose cudnn or explicit native capture')
    if mode=='cudnn' and capture is not None:raise ValueError('capture requires native capture mode')
    if interface is None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
        interface=ALL_ATTENTION_FUNCTIONS
    if not hasattr(interface,'_local_mapping'):raise TypeError('HF attention interface required')
    previous=interface['sdpa'];local=interface._local_mapping.get('sdpa');had_local='sdpa' in interface._local_mapping
    if getattr(previous,'_prefill_cudnn_policy',False):raise RuntimeError('nested prefill policies unsupported')
    modules=[layer.self_attn for layer in target.model.language_model.layers]
    indices={id(module):i for i,module in enumerate(modules)}
    if len(modules)!=28 or len(indices)!=28:raise ValueError('first recipe requires 28 unique Target language layers')
    if any(not getattr(m,'is_causal',False) or getattr(m,'sliding_window',None) is not None for m in modules):
        raise ValueError('full causal language attention required; no sliding windows')
    executor=executor if executor is not None else (CausalCudnnExecutor() if mode=='cudnn' else None)
    if mode=='cudnn' and getattr(executor,'backend',None)!='cudnn':raise ValueError('explicit cuDNN executor required')
    bridge=bridge_runtime._bridge();old_prefill=bridge._prefill;old_forward=target.forward
    old_attribute=target.__dict__.get('forward',bridge_runtime._MISSING)
    record=dict(mode=mode,prefill_calls=0,target_forward_calls=0,selected_calls=0,completed_calls=0,
                layer_calls=[0]*28,failures=[],fallback_calls=0,active=False,finalized=False,
                positions_modified=False,kv_quantized=False,scope='Target language native prefill only',
                capture_invalidates_timing_and_memory=mode=='capture')

    def forward(*a,**kw):
        if not record['active']:return old_forward(*a,**kw)
        record['target_forward_calls']+=1
        if a or record['target_forward_calls']!=1 or kw.get('past_key_values') is not None:
            raise RuntimeError('one keyword-only empty-cache Target prefill required')
        ids,pos=kw.get('input_ids'),kw.get('position_ids');p=record['prompt_tokens']
        if (ids is not record['_input_ids'] or not isinstance(pos,torch.Tensor) or
                pos.shape!=(4,1,p) or pos.dtype!=torch.long or pos.device!=ids.device or
                not kw.get('use_cache') or kw.get('logits_to_keep')!=1):
            raise RuntimeError('native complete-prompt IDs/4-axis positions/cache/logits contract changed')
        if not torch.equal(pos[0,0],torch.arange(p,device=pos.device)):
            raise RuntimeError('physical text positions must cover the full original prompt in order')
        record['native_forward']=dict(empty_cache=True,use_cache=True,logits_to_keep=1,
            input_shape=list(ids.shape),position_shape=list(pos.shape),position_dtype=str(pos.dtype),
            physical_positions_start=0,physical_positions_end=p-1,position_tensor_passed_unchanged=True)
        return old_forward(*a,**kw)

    def prefill(received,values,counts,**kw):
        if received is not target:return old_prefill(received,values,counts,**kw)
        if record['prefill_calls'] or record['active']:raise RuntimeError('only one native prefill per policy context')
        ids=values.get('input_ids');mask=values.get('attention_mask')
        if (not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.long
                or ids.shape[1]<2 or values.get('past_key_values') is not None):
            raise ValueError('batch-one complete unpadded prompt with empty cache required')
        if mask is not None and (mask.shape!=ids.shape or not bool(mask.all())):
            raise ValueError('padding or explicit attention bias is unsupported in prefill')
        record.update(prefill_calls=1,prompt_tokens=ids.shape[1],active=True,_input_ids=ids,
                      original_padding_mask_present=mask is not None,original_padding_mask_all_visible=True)
        try:
            result=old_prefill(received,values,counts,**kw)
            if record['target_forward_calls']!=1 or record['layer_calls']!=[1]*28 or record['completed_calls']!=28:
                raise RuntimeError('causal prefill must cover every Target language layer exactly once')
            cache=result[0].past_key_values
            lengths=[layer.get_seq_length() for layer in cache.layers]
            if len(lengths)!=28 or any(length!=ids.shape[1] for length in lengths):
                raise RuntimeError('native prefill cache must retain every prompt token in all layers')
            record['native_cache_tokens_per_layer']=ids.shape[1]
            return result
        except Exception as exc:
            record['failures'].append(dict(stage='prefill',type=type(exc).__name__,message=str(exc)))
            raise
        finally:record['active']=False;record.pop('_input_ids',None)

    def attention(module,query,key,value,attention_mask,dropout=0.0,scaling=None,
                  is_causal=None,position_bias=None,**kwargs):
        if id(module) not in indices or not record['active']:
            return previous(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,
                            is_causal=is_causal,position_bias=position_bias,**kwargs)
        index=indices[id(module)];record['selected_calls']+=1
        try:
            validate_prefill_attention(query,key,value,attention_mask,record['prompt_tokens'])
            if (record['layer_calls'][index] or dropout!=0 or position_bias is not None or is_causal is False
                    or kwargs.get('sliding_window') is not None or kwargs.get('output_attentions',False)):
                raise ValueError('causal prefill selected duplicate layer or nonstandard attention semantics')
            if mode=='capture':
                output,weights=previous(module,query,key,value,None,dropout=0.0,scaling=scaling,
                                        is_causal=True,position_bias=None,**kwargs)
            else:
                output=executor(query,key,value,scaling).transpose(1,2).contiguous();weights=None
            if output.shape!=(1,record['prompt_tokens'],28,128) or output.dtype!=query.dtype:
                raise RuntimeError('language prefill output layout/dtype mismatch')
            if capture is not None:
                capture(module,query,key,value,None,output,scaling,
                    dict(mode=mode,layer_index=index,prompt_tokens=record['prompt_tokens'],empty_cache=True))
            record['layer_calls'][index]+=1;record['completed_calls']+=1
            return output,weights
        except Exception as exc:
            record['failures'].append(dict(stage='attention',layer=index,type=type(exc).__name__,message=str(exc)))
            raise

    attention._prefill_cudnn_policy=True
    bridge._prefill=prefill;target.forward=forward;interface['sdpa']=attention
    try:
        yield record
    finally:
        record['active']=False;record.pop('_input_ids',None);record['finalized']=True
        if executor is not None:record['executor']=executor.metadata()
        bridge._prefill=old_prefill;bridge_runtime._restore_attribute(target,'forward',old_attribute)
        if had_local:interface['sdpa']=local
        else:del interface['sdpa']
