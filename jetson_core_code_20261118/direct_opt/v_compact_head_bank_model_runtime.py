"""Independent compact mixed-head value request policy.

Frozen row-head pack; no tile8. Explicit CPU reference injection only. Native
prefill sources remain alive until aggregate compact-copy status completes.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import time
from types import MethodType
import weakref

import torch
import v_bank_model_runtime as parent
from v_bank_model_runtime import _sid, _bytes, _check_tensor, _restore_attribute, _MISSING
import stride0_gqa_runtime as lifecycle
from stride0_gqa_runtime import math_precision_scope, precision_flags
from grouped_compact_head_bank_math import grouped_compact_head_bank_math
import v_head_scale_cuda_ops as row_ops
import v_compact_head_bank_cuda_ops as compact_cuda

VARIANT='full_k_compact_mixed_head_bank14_grouped_math'
BUDGET_HEADS=14
_ACTIVE=None
_MARKERS=parent._MARKERS+('_v_compact_head_bank_model_policy',)


class _ResultStatus:
    """Inherited result lifecycle's single collective check validates the global compact owner."""
    def __init__(self,policy):self.policy=policy
    def check_status(self,banks,*,stage):
        p=self.policy
        return p.compact_ops.check_compact_status(p.owner,banks,stage=stage)


class CompactHeadBankModelPolicy(parent.VBankModelPolicy):
    def __init__(self,target,bridge,previous,reference_ops=None):
        super().__init__(target,bridge,previous,'int8_visual',reference_ops)
        self.head_ops=row_ops if reference_ops is None else reference_ops.head
        self.compact_ops=compact_cuda if reference_ops is None else reference_ops.compact
        self.ops=_ResultStatus(self)
        self.owner=None;self._original_values=[];self._score_slab=None;self._score_ref=None
        self.old_bank_refs=[];self.old_quant_refs=[]
        self.record.update(variant=VARIANT,target_cache_policy=VARIANT,compact_budget_heads=BUDGET_HEADS,
            conversion_boundary='after native Target prefill, before Draft adaptor; score from frozen row-head pack',
            selection_rule='stable GPU descending zero-aware group-scale proxy; layer*4+head tie break',
            scale_proxy_is_task_risk=False,head_pack_variant='validated_row',compact_ready_status_calls=0,
            row_head_pack_calls=0,compact_head_copy_calls=0,score_zero_initializations=0,
            source_full_V_stacked=False,all_original_V_retained_until_copy_ready=True,
            steady_fill_payload_reduction_vs_previous_overlay=False,
            global_visual_bank_counted_once=True,
            compact_fill_payload_scope='all 28 layers per completed verification; fixed14 budget, no per-layer route readback',
            score_source_tensor_refs_alive=None)

    def _storage(self):
        # Failure can leave uninstalled pack roots in traceback frames: never
        # promote a partial inventory to complete request-owned capacity.
        original_roots={};original_objects=set()
        for ref in self.old_value_refs:
            value=ref()
            if value is not None:
                original_roots[_sid(value)]=value.untyped_storage().nbytes()
                original_objects.add(id(value))
        for value in self._original_values:
            original_roots[_sid(value)]=value.untyped_storage().nbytes();original_objects.add(id(value))
        old_banks=sum(ref() is not None for ref in self.old_bank_refs)
        old_quant=sum(ref() is not None for ref in self.old_quant_refs)
        transient=dict(original_full_V_known_backing_retained=bool(original_roots),
            original_value_tensor_refs_alive=len(original_objects),
            transient_retained_original_V_known_backing_bytes=sum(original_roots.values()),
            old_bank_tensor_refs_alive=old_banks,old_code_scale_tensor_refs_alive=old_quant,
            transient_score_slab_known_backing_bytes=0 if self._score_slab is None else self._score_slab.untyped_storage().nbytes())
        if (len(self.stores)!=28 or self.owner is None or original_roots or old_banks or old_quant
                or self._score_slab is not None):
            return dict(transient,partial=True,capacity_status='unavailable',
                unavailable_reason='conversion/installation/release incomplete; traceback-held temporary roots may be unaccounted',
                layer_count=len(self.stores),expected_layer_count=28,
                owned_plus_shared_device_bytes=None,unique_layer_backing_bytes=None,
                allocator_reserved_is_separate=True,
                scope='partial diagnostic inventory, not complete capacity; phase peaks remain separate')
        self.owner.validate_metadata()
        known={};local_roots=[];layers=[]
        for i,item in enumerate(self.stores):
            item.check()
            if item.bank.owner is not self.owner or item.bank.layer_idx!=i:
                raise RuntimeError('layer view belongs to another owner or index')
            entries={}
            for name,tensor in [('keys',item.keys16),('text_values',item.text_values16),('error_flag',item.bank.error_flag)]:
                entries[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),
                    logical_bytes=_bytes(tensor),backing_bytes=tensor.untyped_storage().nbytes())
                if tensor.numel():local_roots.append(tensor)
            layers.append(entries)
        ids=[_sid(t) for t in local_roots]
        if len(ids)!=len(set(ids)):raise RuntimeError('K/text/flags must have independent layer backing')
        for t in local_roots:known[_sid(t)]=t.untyped_storage().nbytes()
        local_bytes=sum(known.values())
        globals_=[self.owner.codes,self.owner.scales,self.owner.original_fp16,self.owner.head_route,self.route.prefix_route]
        global_ids=[_sid(t) for t in globals_ if t.numel()]
        if len(global_ids)!=len(set(global_ids)) or set(global_ids).intersection(ids):
            raise RuntimeError('global compact payload/route aliases K/text/flags')
        for t in globals_:
            if t.numel():known[_sid(t)]=t.untyped_storage().nbytes()
        return dict(transient,partial=False,capacity_status='complete',layer_count=28,layers=layers,
            unique_layer_backing_bytes=local_bytes,compact_bank=self.owner.storage_receipt(),
            global_visual_bank_counted_once=True,shared_device_route_bytes=_bytes(self.route.prefix_route),
            owned_plus_shared_device_bytes=sum(known.values()),route_counted_once=True,
            host_partition_route_bytes=self.partition.prefix_route.untyped_storage().nbytes(),
            allocator_reserved_is_separate=True,
            scope='unique global visual payload plus per-layer full K/text/flags and shared route; no allocator/DRAM inference')

    def _fail(self,stage,exc):
        super()._fail(stage,exc)
        if self.owner is not None:
            self.owner.failed=True;self.owner.validated=False

    def prefill(self,target,values,counts,**kwargs):
        if target is not self.target:return self.original_prefill(target,values,counts,**kwargs)
        try:
            self.device=values['input_ids'].device
            if self.reference_only and self.device.type!='cpu':raise ValueError('reference_ops are CPU-only')
            if not self.reference_only and self.device.type!='cuda':raise ValueError('overlay model requires CUDA; no fallback')
            # Bypass only parent's immediate V conversion/release, never native prefill guards.
            output=lifecycle.Stride0AttentionPolicy.prefill(self,target,values,counts,**kwargs)
            self._phase('prefill');self.current_phase='conversion';start=time.perf_counter()
            self.partition,visual_counts=parent._partition_from_values(target,values)
            self.route=self.head_ops.prepare_partition_route(self.partition,self.device)
            self._io('route_upload',4*self.route.P,4*self.route.P)
            self.record.update(visual_counts=visual_counts,partition=self.partition.receipt(),route=self.route.receipt())
            originals=[layer.values for layer in self.cache.layers]
            self._original_values=originals
            original_ids=set()
            for layer in self.cache.layers:
                _check_tensor(layer.keys,self.device,self.route.P,'prefill K')
                _check_tensor(layer.values,self.device,self.route.P,'prefill V')
                if _sid(layer.keys)==_sid(layer.values):raise ValueError('prefill K/V share backing')
                if _sid(layer.values) in original_ids:raise ValueError('original layers share V backing')
                original_ids.add(_sid(layer.values));self.old_value_refs.append(weakref.ref(layer.values))
            with torch.inference_mode(False):
                self._score_slab=torch.zeros((28,4),dtype=torch.float32,device=self.device)
            self._score_ref=weakref.ref(self._score_slab)
            self.record['score_zero_initializations']+=1
            self._io('score_slab_zero',0,28*4*4)
            banks=[];texts=[]
            # Ownership stays in the 28 original layer tensors until all copies are ready.
            for index in range(28):
                bank,text=self.head_ops.pack_v_with_head_scale(originals[index],self.route,self._score_slab[index])
                if bank.route is not self.route or bank.mode!='int8_visual':raise RuntimeError('head pack route/mode mismatch')
                bank.validate_metadata();_check_tensor(text,self.device,self.route.R,'packed text V')
                roots=[text,bank.visual_values,bank.scales,bank.error_flag]
                ids=[_sid(x) for x in roots if x.numel()]
                if len(ids)!=len(set(ids)) or original_ids.intersection(ids):raise RuntimeError('pack roots alias original V/each other')
                banks.append(bank);texts.append(text);self.record['row_head_pack_calls']+=1
                self.old_bank_refs.append(weakref.ref(bank))
                self.old_quant_refs.extend((weakref.ref(bank.visual_values),weakref.ref(bank.scales)))
                self._io('prefill_pack',2*4*self.route.P*128,_bytes(bank.visual_values)+_bytes(bank.scales)+_bytes(text))
            packed_roots=[x for b,t in zip(banks,texts) for x in (b.visual_values,b.scales,b.error_flag,t) if x.numel()]
            packed_ids=[_sid(x) for x in packed_roots]
            key_ids={_sid(layer.keys) for layer in self.cache.layers}
            if (len(set(packed_ids))!=len(packed_ids) or key_ids.intersection(packed_ids)
                    or _sid(self._score_slab) in packed_ids):
                raise RuntimeError('cross-layer pack/text roots alias each other, K or score slab')
            del packed_roots,packed_ids
            self.record['setup_status']=self.head_ops.validate_pack_status(banks);self.record['setup_status_calls']+=1
            if any(not b.validated or b.failed for b in banks):raise RuntimeError('not all head packs validated')
            self.owner=self.compact_ops.prepare_compact_head_bank(self._score_slab,self.route,budget_heads=BUDGET_HEADS)
            for index,bank in enumerate(banks):
                self.compact_ops.copy_compact_head_layer_(self.owner,index,originals[index],bank)
                self.record['compact_head_copy_calls']+=1
            self.record['compact_ready_status']=self.compact_ops.validate_compact_ready(self.owner,banks)
            self.record['compact_ready_status_calls']+=1
            if not self.owner.validated or self.owner.failed:raise RuntimeError('compact bank not ready')
            # This aggregate status boundary has completed queued source copies.
            self.record['compact_preparation']=copy.deepcopy(self.owner.preparation)
            read=BUDGET_HEADS*self.route.M*128*2
            self._io('original_head_overlay_copy',read,read)
            self.io['original_head_overlay_copy']['calls']=28
            quant=(112-BUDGET_HEADS)*self.route.M*144
            self._io('quantized_code_scale_compaction',quant,quant)
            self.io['quantized_code_scale_compaction']['calls']=28
            self.record['conversion_payload_source_calls']=28
            self.record['conversion_payload_call_scope']='28 fused source calls, not sum of two logical branch ledgers'
            self.record['row_head_pack_plan']=row_ops.pack_head_scale_launch_plan(self.route.P,self.route.M)
            for index,(layer,text) in enumerate(zip(self.cache.layers,texts)):
                view=self.owner.layer_view(index)
                owned=[layer.keys,text,view.error_flag,self.owner.codes,self.owner.scales,
                       self.owner.original_fp16,self.owner.head_route]
                ids=[_sid(x) for x in owned if x.numel()]
                if len(set(ids))!=len(ids) or original_ids.intersection(ids):
                    raise RuntimeError('compact installed roots alias full V/K/text/global bank')
                self.layer_restore.append((layer,layer.__dict__.get('update',_MISSING)))
                layer.values=text
                self.stores.append(parent._Layer(layer,view,self.route,self.mode))
                def update(instance,keys,values,*a,_index=index,**kw):
                    return self._update(_index,instance,keys,values,a,kw)
                layer.update=MethodType(update,layer)
            # Loop locals and validation containers must not keep old pack roots.
            del owned,ids,bank,text,banks,texts,roots
            self._original_values.clear();del originals
            self._score_slab=None
            if (any(ref() is not None for ref in self.old_value_refs) or self._score_ref() is not None
                    or any(ref() is not None for ref in self.old_bank_refs+self.old_quant_refs)):
                raise RuntimeError('original V, old V8 bank/payload or scores retained after compact ready')
            self.record['score_source_tensor_refs_alive']=0
            self.record['old_bank_tensor_refs_alive']=0
            self.record['old_code_scale_tensor_refs_alive']=0
            if self.original_length(self.cache)!=self.route.P:raise RuntimeError('K semantic length changed')
            self.record['storage_after_conversion']=self._storage()
            self.record['conversion_wall_s']=time.perf_counter()-start
            self._phase('conversion');self.current_phase='decode'
            return output
        except BaseException as exc:
            self._fail('compact_head_prefill',exc);raise

    def forward(self,*args,**kwargs):
        if self.in_prefill:return super().forward(*args,**kwargs)
        output=super().forward(*args,**kwargs)
        # Every layer executed once; total protected count is known without reading route values.
        self.after_verified_round(self.plan)
        return output

    def after_verified_round(self,plan):
        S=plan.semantic_before+plan.nodes;M=self.route.M;k=BUDGET_HEADS
        reads=k*M*128*2+(112-k)*M*(128+4*4)+28*4*(S-M)*128*2
        writes=28*4*S*128*4
        self._io('value_operand_fill',reads,writes)
        self.io['value_operand_fill']['calls']+=27
        avoided=self.record.setdefault('counterfactual_protected_V8_scale_read_bytes_avoided',0)
        self.record['counterfactual_protected_V8_scale_read_bytes_avoided']=avoided+k*M*(128+4*4)
        # Literal route loads in compact fill; flag/atomic traffic remains excluded.
        self._io('compact_fill_route_logical_reads',28*(4*self.route.P*4+4*((S+7)//8)*4),0)
        self.io['compact_fill_route_logical_reads']['calls']+=27

    def attention(self,module,query,key,value,attention_mask,dropout=0.0,scaling=None,is_causal=None,position_bias=None,**kwargs):
        index=self.indices.get(id(module))
        if index is None or self.in_prefill:
            return super().attention(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,
                is_causal=is_causal,position_bias=position_bias,**kwargs)
        self.record['selected_calls']+=1
        try:
            if self.failed or not self.in_forward or self.pending is None or self.plan is None:
                raise RuntimeError('attention outside active Target verification')
            item=self.stores[index];item.check(self.plan,active=True)
            if (self.update_counts[index]!=1 or self.pending['layer_calls'][index] or key is not item.keys16
                    or value is not item.text_values16 or attention_mask is not self.pending['mask']
                    or query.shape!=(1,28,self.plan.nodes,128) or dropout!=0.0 or position_bias is not None
                    or is_causal not in (None,False) or set(kwargs)-{'position_ids','sliding_window','output_attentions','use_cache'}
                    or kwargs.get('position_ids') is not None or kwargs.get('sliding_window') is not None
                    or kwargs.get('output_attentions',False) is not False
                    or ('use_cache' in kwargs and kwargs['use_cache'] is not True)):
                raise ValueError('attention operands/options do not belong to the updated V-bank layer')
            first=self.record['representative_receipt'] is None
            output,receipt,debug=grouped_compact_head_bank_math(query,key,item.bank,value,attention_mask,scaling,
                owner=self.owner,layer_idx=index,
                fill_impl=self.compact_ops.fill_compact_value_fp32_ if self.reference_only else None,
                record_receipt=first,debug_intermediates=False)
            if debug is not None or (first and receipt is None) or (not first and receipt is not None):
                raise RuntimeError('unexpected primitive receipt/debug contract')
            if first:self.record['representative_receipt']=receipt;self.record['primitive_receipt_calls']+=1
            if output.shape!=(1,self.plan.nodes,28,128) or output.dtype!=torch.float16 or not output.is_contiguous():
                raise RuntimeError('attention output differs from HF layout')
            self.pending['layer_calls'][index]+=1
            self.record['layer_calls'][index]+=1;self.record['completed_calls']+=1;self.record['candidate_calls']+=1
            self.record['fill_calls']+=1
            return output,None
        except BaseException as exc:
            self._fail('attention',exc);raise

    def validate_result(self,result):
        if (self.owner is None or not self.owner.validated or self.owner.failed
                or self.record['row_head_pack_calls']!=28 or self.record['compact_head_copy_calls']!=28
                or self.record['compact_ready_status_calls']!=1 or self._original_values or self._score_slab is not None
                or any(ref() is not None for ref in self.old_bank_refs+self.old_quant_refs)):
            exc=RuntimeError('incomplete compact setup or retained conversion inputs')
            self._fail('compact_head_result',exc);raise exc
        return super().validate_result(result)

    def release(self):
        try:super().release()
        finally:
            self._original_values.clear();self._score_slab=None;self.owner=None

    def summary(self):
        result=super().summary()
        result['score_tensor_refs_alive']=int(self._score_ref is not None and self._score_ref() is not None)
        result['old_bank_tensor_refs_alive']=sum(ref() is not None for ref in self.old_bank_refs)
        result['old_code_scale_tensor_refs_alive']=sum(ref() is not None for ref in self.old_quant_refs)
        result['logical_io_scope']+='; protected-head V8/scales excluded from fill read payload; avoided component is counterfactual, not executed IO'
        return result


@contextmanager
def compact_head_bank_model_policy(target,*,interface=None,bridge=None,reference_ops=None):
    global _ACTIVE
    import ring_cache_runtime as cache_lifecycle
    if _ACTIVE is not None or parent._ACTIVE is not None or cache_lifecycle._ACTIVE is not None:
        raise RuntimeError('nested/concurrent compact/cache policy unsupported')
    if interface is None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
        interface=ALL_ATTENTION_FUNCTIONS
    if bridge is None:
        from ring_cache_runtime import _bridge
        bridge=_bridge()
    if not hasattr(interface,'_local_mapping'):raise TypeError('local HF attention registry required')
    previous=interface['sdpa']
    if any(getattr(previous,k,False) for k in _MARKERS):raise RuntimeError('nested attention policy unsupported')
    if getattr(bridge._prefill,'_v_bank_model_policy',False):raise RuntimeError('nested bridge policy unsupported')
    state=CompactHeadBankModelPolicy(target,bridge,previous,reference_ops)
    had_local='sdpa' in interface._local_mapping;old_local=interface._local_mapping.get('sdpa')
    old_forward=target.__dict__.get('forward',_MISSING)
    def attention(*args,**kwargs):return state.attention(*args,**kwargs)
    for marker in _MARKERS:setattr(attention,marker,True)
    def restore_registry():
        if had_local:interface['sdpa']=old_local
        elif 'sdpa' in interface._local_mapping:del interface['sdpa']
    with ExitStack() as stack:
        stack.callback(setattr,sys_module(),'_ACTIVE',None)
        stack.callback(setattr,cache_lifecycle,'_ACTIVE',None)
        stack.callback(state.release)
        stack.callback(restore_registry)
        stack.callback(_restore_attribute,target,'forward',old_forward)
        stack.callback(setattr,bridge,'_compact',state.original_compact)
        stack.callback(setattr,bridge,'_prefill',state.original_prefill)
        _ACTIVE=state;cache_lifecycle._ACTIVE=state
        bridge._prefill=state.prefill;bridge._compact=state.compact
        target.forward=state.forward;interface['sdpa']=attention
        try:yield state
        except BaseException as exc:
            state._fail('context',exc)
            try:state.finish()
            except BaseException as final_exc:state._fail('finish_after_failure',final_exc)
            raise
        else:state.finish()


def sys_module():
    import sys
    return sys.modules[__name__]
