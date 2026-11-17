"""Isolated two-BMM Target tree policy; native control is the frozen policy.

The new class reuses explicit prefill/cache/forward lifecycle checks, not an
old primitive global. Only grouped tree attention and result accounting differ.
Full DynamicCache, original FP32 tree mask and already rotated K stay intact.
One tensor-free primitive receipt is retained; subsequent calls request none.
"""
from __future__ import annotations

from contextlib import contextmanager
import torch

import stride0_gqa_runtime as base
from stride0_gqa_runtime import math_precision_scope, precision_flags
from grouped_gqa_math import grouped_gqa_math

METHOD='grouped_two_bmm_fp32_math'
_MISSING=object()


class GroupedAttentionPolicy(base.Stride0AttentionPolicy):
    def __init__(self,target,bridge,previous_attention):
        # Reuse only initialization and the unchanged prefill/forward lifecycle.
        # Every tree dispatch and result validation below uses the actual mode.
        super().__init__(target,'native',bridge,previous_attention)
        self.mode='grouped'
        self._kv_elements=0;self._query_pack_bytes=0;self._query_pack_calls=0
        self.record.update(mode='grouped',implementation=METHOD,
            primitive_receipt_calls=0,ordinary_primitive_record_receipt=False,
            maximum_representative_receipts_per_request=1,
            first_tree_call_receipt_in_request=True,
            primitive_receipt_computation_in_request='first tree layer only',
            receipt_aggregation_in_request='scalar shape/copy counts on every tree layer',
            summary_dictionary_prepared_in_runtime_finalization=True,
            query_layout_contract='contiguous or native dense token-major [1,28,T,128]',
            logical_copy_totals_scope='Candidate actual shapes versus HF repeat then math; '
                'KV-only removed copies and added Q pack are separately reported. '
                'Net explicit-copy difference excludes GEMM reads, BLAS workspace and actual DRAM.')

    def attention(self,module,query,key,value,attention_mask,dropout=0.0,scaling=None,
                  is_causal=None,position_bias=None,**kwargs):
        index=self.indices.get(id(module))
        if index is None or self.in_prefill:
            return super().attention(module,query,key,value,attention_mask,dropout=dropout,
                scaling=scaling,is_causal=is_causal,position_bias=position_bias,**kwargs)
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
            # These two dense layouts have a known preserve-format FP32 cast.
            # The guard makes scalar Q-pack accounting exact without inspecting
            # newly allocated tensors or constructing a receipt every layer.
            query_contiguous=query.is_contiguous()
            if not query_contiguous and query.stride()[-3:]!=(128,28*128,1):
                raise ValueError('unsupported query layout for exact lightweight Q-pack accounting')
            first=self.record['representative_receipt'] is None
            output,receipt,debug=grouped_gqa_math(query,key,value,attention_mask,scaling,
                record_receipt=first,debug_intermediates=False)
            if debug is not None:raise RuntimeError('debug tensors are forbidden in full requests')
            if first:
                if receipt is None or receipt['query_pack_copy']!=(not query_contiguous):
                    raise RuntimeError('first primitive receipt disagrees with supported query layout')
                self.record['representative_receipt']=receipt
                self.record['primitive_receipt_calls']+=1
            elif receipt is not None:raise RuntimeError('ordinary primitive path unexpectedly constructed a receipt')
            if output.shape!=(1,t,28,128) or output.dtype!=torch.float16 or output.device!=query.device:
                raise RuntimeError('tree attention output shape/dtype/device differs from HF contract')
            self._kv_elements+=4*n*128
            if not query_contiguous:
                self._query_pack_bytes+=4*28*t*128
                self._query_pack_calls+=1
            pending['layer_calls'][index]+=1
            self.record['layer_calls'][index]+=1;self.record['completed_calls']+=1
            self.record['candidate_calls']+=1
            return output,None
        except BaseException as exc:
            self._fail('attention',exc);raise

    def validate_result(self,result):
        base._check_precision()
        c=result.forward_counts;r=self.record;rounds=[x for x in result.rounds if x.get('kind')=='tree']
        expected=int(c['target_verify_calls'])
        if (self.failed or self.pending is not None or self.in_prefill or r['failures']
                or r['prefill_calls']!=c['target_prefill_calls']
                or r['verify_forward_calls']!=expected or len(r['rounds'])!=expected or len(rounds)!=expected
                or r['verify_query_tokens']!=c['target_verify_query_tokens']
                or r['selected_calls']!=28*expected or r['completed_calls']!=28*expected
                or r['layer_calls']!=[expected]*28 or r['candidate_calls']!=28*expected
                or r['native_tree_calls']!=0 or r['primitive_receipt_calls']!=int(expected>0)
                or self.bridge._length is not self.original_length or self.bridge._compact is not self.original_compact):
            raise RuntimeError('grouped attention policy did not cover exactly all native forwards/layers without cache changes')
        for observed,source in zip(r['rounds'],rounds):
            if observed['cache_before']!=source['target_cache_before'] or observed['tree_nodes']!=source['tree_nodes']:
                raise RuntimeError('attention coverage differs from actual verification rounds')
        r['result_coverage_validated']=True

    def summary(self):
        result=super().summary();e=self._kv_elements;q=self._query_pack_bytes
        removed=dict(read_bytes=76*e,write_bytes=100*e,read_plus_write_bytes=176*e)
        added=dict(read_bytes=q,write_bytes=q,read_plus_write_bytes=2*q)
        result['logical_copy_totals']=dict(kv_only_removed_explicit_bytes=removed,
            added_query_pack_bytes=added,query_pack_calls=self._query_pack_calls,
            net_removed_explicit_bytes={key:removed[key]-added[key] for key in removed},
            source_read_assumption='one logical source read per produced element; not physical bus traffic',
            actual_dram_measured=False,blas_internal_copy_measured=False,
            attention_mac_reduction=0,kv_resident_capacity_reduction=0)
        return result


@contextmanager
def grouped_attention_policy(target,mode='grouped',*,interface=None,bridge=None):
    if mode=='native':
        # Exactly the existing native control, including its host checks.
        with base.stride0_attention_policy(target,'native',interface=interface,bridge=bridge) as policy:
            yield policy
        return
    if mode!='grouped':raise ValueError('choose native or grouped explicitly')
    if interface is None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
        interface=ALL_ATTENTION_FUNCTIONS
    if bridge is None:import hf517_vispec_adapter as bridge
    if not hasattr(interface,'_local_mapping'):raise TypeError('HF local attention registry required')
    previous=interface['sdpa']
    if any(getattr(previous,key,False) for key in ('_stride0_tree_policy','_strict_gqa_policy','_grouped_tree_policy')):
        raise RuntimeError('nested tree attention policies unsupported')
    policy=GroupedAttentionPolicy(target,bridge,previous)
    had_local='sdpa' in interface._local_mapping;local=interface._local_mapping.get('sdpa')
    old_forward_attribute=target.__dict__.get('forward',_MISSING)
    def attention(*a,**kw):return policy.attention(*a,**kw)
    attention._grouped_tree_policy=True
    attention._stride0_tree_policy=True  # makes the frozen policy reject cross-mode nesting too
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
