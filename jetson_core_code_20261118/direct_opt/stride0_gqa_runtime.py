"""Scoped Target tree math policy with unchanged native DynamicCache.

An observed native bridge prefill establishes cache identity. Only subsequent
explicit verification forwards using that cache select the 28 Target language
attention modules. Short prefill, vision, Draft, projections, RoPE and cache
operations remain native. The stride-zero primitive is a numerical execution
candidate, not a bitwise guarantee or a new GQA algorithm.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import torch

_MISSING=object()
METHOD='stride0_four_group_fp32_math'


@contextmanager
def math_precision_scope():
    previous=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    try:yield precision_flags()
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous[1]
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous[2])


def precision_flags():
    return dict(allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        reduced_precision_math_sdp=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))


def _check_precision():
    flags=precision_flags()
    if flags!=dict(allow_tf32=False,float32_matmul_precision='highest',reduced_precision_math_sdp=False):
        raise RuntimeError('tree math comparison requires highest FP32 precision, TF32=False and math reduction=False')


class Stride0AttentionPolicy:
    def __init__(self,target,mode,bridge,previous_attention):
        if mode not in ('native','stride0'):raise ValueError('choose native or stride0 explicitly')
        _check_precision()
        if getattr(target,'_vispec_precision',{}).get('target_sdpa')!='math':
            raise ValueError('requires the frozen native Target math loader')
        modules=[layer.self_attn for layer in target.model.language_model.layers]
        self.indices={id(module):i for i,module in enumerate(modules)}
        if len(modules)!=28 or len(self.indices)!=28:
            raise ValueError('28 unique Target language attention layers required')
        if any(getattr(m,'sliding_window',None) is not None for m in modules):
            raise ValueError('full attention only; sliding-window layers unsupported')
        self.target,self.mode,self.bridge=target,mode,bridge
        self.original_forward=target.forward
        self.original_prefill=bridge._prefill
        self.original_length=bridge._length
        self.original_compact=bridge._compact
        self.previous_attention=previous_attention
        self.cache=None;self.pending=None;self.in_prefill=False;self.failed=False
        self.prefill_ids=None
        self.record=dict(mode=mode,scope='Target tree attention only',target_cache_policy='native_full_dynamic_cache',
            new_kv_policy_installed=False,cache_functions_changed=False,prefill_calls=0,prefill_forward_calls=0,
            prefill_layer_calls=[0]*28,verify_forward_calls=0,verify_query_tokens=0,
            selected_calls=0,completed_calls=0,layer_calls=[0]*28,candidate_calls=0,native_tree_calls=0,
            unrelated_passthrough_calls=0,rounds=[],failures=[],fallback_calls=0,finalized=False,
            representative_receipt=None,logical_copy_totals={},precision=precision_flags(),
            original_mask_reused=True,position_ids_passed_unchanged=True,rotary_reapplied=False,
            debug_intermediates=False,primitive_receipt_computation_in_request=True,
            receipt_aggregation_in_request=True,actual_dram_measured=False,quality_pass=None,
            logical_copy_totals_scope='Counterfactual HF repeat then math at candidate actual shapes; excludes DRAM and cross-arm trajectory differences',
            numerical_equivalence_claim=False)

    def _fail(self,stage,exc):
        self.failed=True
        self.record['failures'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))

    def prefill(self,target,values,counts,**kwargs):
        if target is not self.target:return self.original_prefill(target,values,counts,**kwargs)
        if self.failed or self.record['prefill_calls'] or self.in_prefill or self.pending is not None:
            raise RuntimeError('one native prefill per policy context required')
        ids=values.get('input_ids')
        if not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1:
            raise ValueError('native prefill requires nonempty batch-one IDs')
        self.prefill_ids=ids;self.in_prefill=True;self.record['prefill_calls']+=1
        try:
            result=self.original_prefill(target,values,counts,**kwargs)
            cache=result[0].past_key_values
            if (self.record['prefill_forward_calls']!=1 or self.record['prefill_layer_calls']!=[1]*28
                    or len(cache.layers)!=28 or self.original_length(cache)!=ids.shape[1]):
                raise RuntimeError('native prefill did not cover all 28 layers and full prompt exactly once')
            self.cache=cache
            self.record.update(prompt_tokens=int(ids.shape[1]),native_rope_delta=int(result[1]),
                native_cache_type=type(cache).__module__+'.'+type(cache).__qualname__,
                native_cache_layer_types=[type(layer).__module__+'.'+type(layer).__qualname__ for layer in cache.layers])
            return result
        except BaseException as exc:
            self._fail('prefill',exc);raise
        finally:self.in_prefill=False;self.prefill_ids=None

    def forward(self,*args,**kwargs):
        if self.failed:raise RuntimeError('tree attention request previously failed')
        if self.in_prefill:
            self.record['prefill_forward_calls']+=1
            if (args or self.record['prefill_forward_calls']!=1 or kwargs.get('past_key_values') is not None
                    or kwargs.get('input_ids') is not self.prefill_ids):
                raise RuntimeError('native prefill must be one unchanged empty-cache forward')
            return self.original_forward(*args,**kwargs)
        try:
            _check_precision()
            cache=kwargs.get('past_key_values')
            if args or cache is None or cache is not self.cache or self.pending is not None:
                raise RuntimeError('verification requires the observed prefill cache and no nested forward')
            ids,mask,pos=(kwargs.get(name) for name in ('input_ids','attention_mask','position_ids'))
            if (not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.long
                    or not isinstance(mask,torch.Tensor) or not isinstance(pos,torch.Tensor)):
                raise ValueError('explicit tree IDs, FP32 mask and original positions required')
            base=self.original_length(cache);nodes=int(ids.shape[1])
            if (not 1<=nodes<=30 or mask.shape!=(1,1,nodes,base+nodes) or mask.dtype!=torch.float32
                    or pos.shape!=(3,1,nodes) or pos.dtype!=torch.long
                    or mask.device!=ids.device or pos.device!=ids.device
                    or kwargs.get('use_cache') is not True or kwargs.get('logits_to_keep')!=0
                    or kwargs.get('return_dict') is not True
                    or kwargs.get('inputs_embeds') is not None or kwargs.get('pixel_values') is not None
                    or kwargs.get('pixel_values_videos') is not None):
                raise ValueError('native tree shape/mask/position/cache/logits contract differs')
            self.pending=dict(base=base,nodes=nodes,mask=mask,positions=pos,layer_calls=[0]*28)
            self.record['verify_forward_calls']+=1
            output=self.original_forward(**kwargs)
            if (self.pending['layer_calls']!=[1]*28 or output.past_key_values is not cache
                    or self.original_length(cache)!=base+nodes):
                raise RuntimeError('verification must execute every layer once and append the complete tree')
            self.record['verify_query_tokens']+=nodes
            self.record['rounds'].append(dict(cache_before=base,tree_nodes=nodes,cache_after=base+nodes,
                layer_calls=list(self.pending['layer_calls']),mask_shape=list(mask.shape),
                mask_dtype=str(mask.dtype),position_shape=list(pos.shape),position_dtype=str(pos.dtype)))
            return output
        except BaseException as exc:
            self._fail('forward',exc);raise
        finally:self.pending=None

    def attention(self,module,query,key,value,attention_mask,dropout=0.0,scaling=None,
                  is_causal=None,position_bias=None,**kwargs):
        index=self.indices.get(id(module))
        def native():
            return self.previous_attention(module,query,key,value,attention_mask,dropout=dropout,
                scaling=scaling,is_causal=is_causal,position_bias=position_bias,**kwargs)
        if index is None:
            self.record['unrelated_passthrough_calls']+=1
            return native()
        if self.in_prefill:
            self.record['prefill_layer_calls'][index]+=1
            if self.record['prefill_layer_calls'][index]!=1:raise RuntimeError('duplicate native prefill layer')
            return native()
        self.record['selected_calls']+=1
        try:
            pending=self.pending
            if self.failed or pending is None:raise RuntimeError('Target attention outside recognized prefill/tree forward')
            n,t=pending['base']+pending['nodes'],pending['nodes']
            if (query.shape!=(1,28,t,128) or key.shape!=(1,4,n,128) or value.shape!=key.shape
                    or any(x.dtype!=torch.float16 for x in (query,key,value))
                    or any(x.device!=query.device for x in (key,value,attention_mask))
                    or attention_mask is not pending['mask'] or attention_mask.stride(-1)!=1
                    or any(x.requires_grad or x.stride(-1)!=1 for x in (query,key,value,attention_mask))
                    or pending['layer_calls'][index] or dropout!=0.0 or position_bias is not None
                    or kwargs.get('sliding_window') is not None or kwargs.get('output_attentions',False)):
                raise ValueError('tree layer operands, original mask identity or attention options changed')
            if self.mode=='native':
                output,weights=native();self.record['native_tree_calls']+=1
            else:
                from stride0_gqa_math import stride0_gqa_math
                output,receipt,debug=stride0_gqa_math(query,key,value,attention_mask,scaling,
                                                   debug_intermediates=False)
                if debug is not None:raise RuntimeError('debug tensors are forbidden in full requests')
                weights=None;self.record['candidate_calls']+=1
                if self.record['representative_receipt'] is None:self.record['representative_receipt']=receipt
                # All receipt construction and aggregation stay in the call.
                # Only scalar totals and one tensor-free representative persist.
                for name,value_bytes in receipt['logical_copies']['removed_explicit_component_bytes'].items():
                    totals=self.record['logical_copy_totals'];totals[name]=totals.get(name,0)+value_bytes
            if output.shape!=(1,t,28,128) or output.dtype!=torch.float16 or output.device!=query.device:
                raise RuntimeError('tree attention output shape/dtype/device differs from HF contract')
            pending['layer_calls'][index]+=1
            self.record['layer_calls'][index]+=1;self.record['completed_calls']+=1
            return output,weights
        except BaseException as exc:
            self._fail('attention',exc);raise

    def validate_result(self,result):
        _check_precision()
        c=result.forward_counts;r=self.record;rounds=[x for x in result.rounds if x.get('kind')=='tree']
        expected=int(c['target_verify_calls'])
        if (self.failed or self.pending is not None or self.in_prefill or r['failures']
                or r['prefill_calls']!=c['target_prefill_calls']
                or r['verify_forward_calls']!=expected or len(r['rounds'])!=expected or len(rounds)!=expected
                or r['verify_query_tokens']!=c['target_verify_query_tokens']
                or r['selected_calls']!=28*expected or r['completed_calls']!=28*expected
                or r['layer_calls']!=[expected]*28
                or r['candidate_calls']!=(28*expected if self.mode=='stride0' else 0)
                or r['native_tree_calls']!=(28*expected if self.mode=='native' else 0)
                or self.bridge._length is not self.original_length or self.bridge._compact is not self.original_compact):
            raise RuntimeError('tree attention policy did not cover exactly all native forwards/layers without cache changes')
        for observed,source in zip(r['rounds'],rounds):
            if observed['cache_before']!=source['target_cache_before'] or observed['tree_nodes']!=source['tree_nodes']:
                raise RuntimeError('attention coverage differs from actual verification rounds')
        r['result_coverage_validated']=True

    def summary(self):return copy.deepcopy(self.record)


@contextmanager
def stride0_attention_policy(target,mode='stride0',*,interface=None,bridge=None):
    if interface is None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
        interface=ALL_ATTENTION_FUNCTIONS
    if bridge is None:import hf517_vispec_adapter as bridge
    if not hasattr(interface,'_local_mapping'):raise TypeError('HF local attention registry required')
    previous=interface['sdpa']
    if getattr(previous,'_stride0_tree_policy',False) or getattr(previous,'_strict_gqa_policy',False):
        raise RuntimeError('nested tree attention policies unsupported')
    policy=Stride0AttentionPolicy(target,mode,bridge,previous)
    had_local='sdpa' in interface._local_mapping;local=interface._local_mapping.get('sdpa')
    old_forward_attribute=target.__dict__.get('forward',_MISSING)
    def attention(*a,**kw):return policy.attention(*a,**kw)
    attention._stride0_tree_policy=True
    bridge._prefill=policy.prefill;target.forward=policy.forward;interface['sdpa']=attention
    try:yield policy
    finally:
        bridge._prefill=policy.original_prefill
        if old_forward_attribute is _MISSING:delattr(target,'forward')
        else:target.forward=old_forward_attribute
        if had_local:interface['sdpa']=local
        else:del interface['sdpa']
        policy.cache=None;policy.pending=None;policy.prefill_ids=None;policy.in_prefill=False
        policy.record['finalized']=True
