"""Independent physical original-FP16 head overlay request policy.

Frozen row-head pack; no tile8. Explicit CPU reference injection only. Native
prefill sources remain alive until aggregate overlay-copy status completes.
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
from grouped_head_overlay_math import grouped_head_overlay_math
import v_head_scale_cuda_ops as row_ops
import v_head_overlay_cuda_ops as overlay_cuda

VARIANT='full_k_v8_physical_head_overlay14_grouped_math'
BUDGET_HEADS=14
_ACTIVE=None
_MARKERS=parent._MARKERS+('_v_head_overlay_model_policy',)


class _ResultStatus:
    """Inherited result lifecycle's single collective check validates overlay too."""
    def __init__(self,policy):self.policy=policy
    def check_status(self,banks,*,stage):
        p=self.policy
        return p.overlay_ops.check_overlay_status(p.overlay,banks,stage=stage)


class HeadOverlayModelPolicy(parent.VBankModelPolicy):
    def __init__(self,target,bridge,previous,reference_ops=None):
        super().__init__(target,bridge,previous,'int8_visual',reference_ops)
        self.head_ops=row_ops if reference_ops is None else reference_ops.head
        self.overlay_ops=overlay_cuda if reference_ops is None else reference_ops.overlay
        self.ops=_ResultStatus(self)
        self.overlay=None;self._original_values=[];self._score_slab=None;self._score_ref=None
        self.record.update(variant=VARIANT,target_cache_policy=VARIANT,overlay_budget_heads=BUDGET_HEADS,
            conversion_boundary='after native Target prefill, before Draft adaptor; score from frozen row-head pack',
            selection_rule='stable GPU descending zero-aware group-scale proxy; layer*4+head tie break',
            scale_proxy_is_task_risk=False,head_pack_variant='validated_row',overlay_ready_status_calls=0,
            row_head_pack_calls=0,original_head_copy_calls=0,score_zero_initializations=0,
            source_full_V_stacked=False,all_original_V_retained_until_copy_ready=True,
            overlay_fill_payload_scope='all 28 layers per completed verification; fixed14 budget, no per-layer route readback',
            score_source_tensor_refs_alive=None)

    def _storage(self):
        if len(self.stores)!=28:
            # Conversion can fail while banks/texts still live in a traceback,
            # before they are all installed in self.stores.  The installed
            # subset is not a complete capacity receipt.  Report only original
            # V roots that are directly observable, without retaining them.
            originals={_sid(t):t.untyped_storage().nbytes() for t in self._original_values}
            original_objects={id(t) for t in self._original_values}
            for ref in self.old_value_refs:
                value=ref()
                if value is not None:
                    originals[_sid(value)]=value.untyped_storage().nbytes()
                    original_objects.add(id(value))
            if self.cache is not None:
                for layer in self.cache.layers[len(self.stores):]:
                    value=layer.values
                    if value is not None:
                        originals[_sid(value)]=value.untyped_storage().nbytes()
                        original_objects.add(id(value))
            return dict(partial=True,capacity_status='unavailable',
                unavailable_reason='overlay conversion has not installed all 28 layers; temporary or traceback-held roots are not fully inventoried',
                layer_count=len(self.stores),expected_layer_count=28,
                owned_plus_shared_device_bytes=None,unique_layer_backing_bytes=None,
                original_full_V_known_backing_retained=bool(originals),
                original_value_tensor_refs_alive=len(original_objects),
                transient_retained_original_V_known_backing_bytes=sum(originals.values()),
                transient_score_slab_known_backing_bytes=0 if self._score_slab is None else self._score_slab.untyped_storage().nbytes(),
                overlay_available=self.overlay is not None,allocator_reserved_is_separate=True,
                scope='partial failure diagnostics only; original V bytes are observed roots, not complete owned capacity; allocator phase peaks remain separate')
        info=super()._storage()
        if self.overlay is None:
            return dict(info,overlay_available=False)
        self.overlay.validate_metadata()
        roots=[self.overlay.head_route]+([self.overlay.values] if self.overlay.values.numel() else [])
        known={_sid(t):t.untyped_storage().nbytes() for item in self.stores for t in item.tensors() if t.numel()}
        if self.route is not None:known[_sid(self.route.prefix_route)]=self.route.prefix_route.untyped_storage().nbytes()
        if any(_sid(t) in known for t in roots):raise RuntimeError('overlay roots alias K/text/base/prefix roots')
        extra={_sid(t):t.untyped_storage().nbytes() for t in roots}
        info.update(overlay_available=True,overlay_storage=self.overlay.storage_receipt(),
            unique_overlay_head_route_backing_bytes=sum(extra.values()),
            owned_plus_shared_device_bytes=sum(known.values())+sum(extra.values()),
            transient_score_slab_known_backing_bytes=0 if self._score_slab is None else self._score_slab.untyped_storage().nbytes(),
            transient_retained_original_V_known_backing_bytes=sum({_sid(t):t.untyped_storage().nbytes() for t in self._original_values}.values()),
            original_full_V_known_backing_retained=bool(self._original_values))
        return info

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
            self.overlay=self.overlay_ops.prepare_head_overlay(self._score_slab,self.route,budget_heads=BUDGET_HEADS)
            for index,bank in enumerate(banks):
                self.overlay_ops.copy_head_overlay_layer_(self.overlay,index,originals[index],bank)
                self.record['original_head_copy_calls']+=1
            self.record['overlay_ready_status']=self.overlay_ops.validate_overlay_ready(self.overlay,banks)
            self.record['overlay_ready_status_calls']+=1
            if not self.overlay.validated or self.overlay.failed:raise RuntimeError('physical overlay not ready')
            # This aggregate status boundary has completed queued source copies.
            self.record['overlay_preparation']=copy.deepcopy(self.overlay.preparation)
            read=BUDGET_HEADS*self.route.M*128*2
            self._io('original_head_overlay_copy',read,read)
            self.io['original_head_overlay_copy']['calls']=28
            self.record['row_head_pack_plan']=row_ops.pack_head_scale_launch_plan(self.route.P,self.route.M)
            for index,(layer,bank,text) in enumerate(zip(self.cache.layers,banks,texts)):
                owned=[layer.keys,text,bank.visual_values,bank.scales,bank.error_flag,self.overlay.head_route,self.overlay.values]
                ids=[_sid(x) for x in owned if x.numel()]
                if len(set(ids))!=len(ids) or original_ids.intersection(ids):raise RuntimeError('installed overlay roots alias full V/K/text/base')
                self.layer_restore.append((layer,layer.__dict__.get('update',_MISSING)))
                layer.values=text
                self.stores.append(parent._Layer(layer,bank,self.route,self.mode))
                def update(instance,keys,values,*a,_index=index,**kw):
                    return self._update(_index,instance,keys,values,a,kw)
                layer.update=MethodType(update,layer)
            del owned,ids,bank,text,banks,texts
            self._original_values.clear();del originals
            self._score_slab=None
            if any(ref() is not None for ref in self.old_value_refs) or self._score_ref() is not None:
                raise RuntimeError('original V or scores retained after ready and installation')
            self.record['score_source_tensor_refs_alive']=0
            if self.original_length(self.cache)!=self.route.P:raise RuntimeError('K semantic length changed')
            self.record['storage_after_conversion']=self._storage()
            self.record['conversion_wall_s']=time.perf_counter()-start
            self._phase('conversion');self.current_phase='decode'
            return output
        except BaseException as exc:
            self._fail('head_overlay_prefill',exc);raise

    def forward(self,*args,**kwargs):
        if self.in_prefill:return super().forward(*args,**kwargs)
        output=super().forward(*args,**kwargs)
        # Every layer executed once; total protected count is known without reading route values.
        S=self.plan.semantic_before+self.plan.nodes;M=self.route.M;k=BUDGET_HEADS
        reads=k*M*128*2+(112-k)*M*(128+4*4)+28*4*(S-M)*128*2
        writes=28*4*S*128*4
        self._io('value_operand_fill',reads,writes)
        self.io['value_operand_fill']['calls']+=27
        avoided=self.record.setdefault('counterfactual_protected_V8_scale_read_bytes_avoided',0)
        self.record['counterfactual_protected_V8_scale_read_bytes_avoided']=avoided+k*M*(128+4*4)
        # Literal route loads in frozen overlay fill; flag/atomic traffic remains excluded.
        self._io('overlay_fill_route_logical_reads',28*(4*self.route.P*4+4*((S+7)//8)*4),0)
        self.io['overlay_fill_route_logical_reads']['calls']+=27
        return output

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
            output,receipt,debug=grouped_head_overlay_math(query,key,item.bank,value,attention_mask,scaling,
                overlay=self.overlay,layer_idx=index,
                fill_impl=self.overlay_ops.fill_overlay_value_fp32_ if self.reference_only else None,
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
        if (self.overlay is None or not self.overlay.validated or self.overlay.failed
                or self.record['row_head_pack_calls']!=28 or self.record['original_head_copy_calls']!=28
                or self.record['overlay_ready_status_calls']!=1 or self._original_values or self._score_slab is not None):
            exc=RuntimeError('incomplete overlay setup or retained conversion inputs')
            self._fail('overlay_result',exc);raise exc
        return super().validate_result(result)

    def release(self):
        try:super().release()
        finally:
            self._original_values.clear();self._score_slab=None;self.overlay=None

    def summary(self):
        result=super().summary()
        result['score_tensor_refs_alive']=int(self._score_ref is not None and self._score_ref() is not None)
        result['logical_io_scope']+='; protected-head V8/scales excluded from fill read payload; avoided component is counterfactual, not executed IO'
        return result


@contextmanager
def head_overlay_model_policy(target,*,interface=None,bridge=None,reference_ops=None):
    global _ACTIVE
    import ring_cache_runtime as cache_lifecycle
    if _ACTIVE is not None or parent._ACTIVE is not None or cache_lifecycle._ACTIVE is not None:
        raise RuntimeError('nested/concurrent head overlay/cache policy unsupported')
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
    state=HeadOverlayModelPolicy(target,bridge,previous,reference_ops)
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
