"""Independent active-KV/cuDNN candidate with one hard FP16 mask per tree.

Native prefill/vision/Draft are untouched. The controlled tree/history boolean
layout directly produces {-inf, 0} FP16, once per Target verification; all layers
consume that exact Tensor. No per-layer GPU-to-CPU mask inspection or conversion.
The original hard-mask and mRoPE guards still run once per round and are timed.
Fused arithmetic/FP16 hard-mask representation require actual output validation.
"""
from __future__ import annotations

from contextlib import contextmanager
import time

import torch

import ring_cache_runtime as base
from active_ring_runtime import ActiveRingCacheState
from gqa_attention_runtime import GQAAttentionBackend

ATTENTION_VARIANT='cudnn_hard_fp16_shared_round'


class CudnnHardMaskState(ActiveRingCacheState):
    def __init__(self,target,bridge,recent_capacity,tree_capacity,executor):
        super().__init__(target,bridge,'protected_prompt',recent_capacity,tree_capacity)
        modules=[layer.self_attn for layer in target.model.language_model.layers]
        self.attention_indices={id(m):i for i,m in enumerate(modules)}
        if not modules or len(modules)!=len(self.attention_indices):raise ValueError('unique Target attention modules required')
        self.executor=executor;self.prepared_mask=None;self.attention_counts=[]
        self.selected_calls=0;self.completed_forwards=0
        self.mask_preparations=0;self.mask_clears=0;self.mask_prepare_records=[];self.attention_failures=[]

    def prefill(self,*args,**kwargs):
        result=super().prefill(*args,**kwargs)
        if args and args[0] is self.target and len(self.stores)!=len(self.attention_indices):
            raise RuntimeError('Target/cache layer count mismatch')
        return result

    def forward(self,*args,**kwargs):
        cache=kwargs.get('past_key_values')
        if self.cache is None or cache is not self.cache:return self.original_forward(*args,**kwargs)
        if self.failed or self.pending is not None or self.in_forward or self.prepared_mask is not None:
            raise RuntimeError('previous tree must be committed and mask cleared before verification')
        if args:raise RuntimeError('explicit keyword verification arguments required')
        started=time.perf_counter()
        mask,pos,ids=(kwargs.get(k) for k in ('attention_mask','position_ids','input_ids'))
        logical_base=self.length(cache)
        if (not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.long
                or not isinstance(mask,torch.Tensor) or mask.ndim!=4 or mask.dtype!=torch.float32
                or not isinstance(pos,torch.Tensor) or pos.dtype!=torch.long):
            raise RuntimeError('batch-one IDs, original FP32 hard mask and explicit long positions required')
        nodes=ids.shape[1]
        if (nodes<1 or nodes>self.tree_capacity or mask.shape!=(1,1,nodes,logical_base+nodes)
                or pos.shape!=(3,1,nodes) or any(x.device!=self.device for x in (ids,mask,pos))
                or mask.requires_grad or pos.requires_grad):
            raise RuntimeError('logical mask/tree/position shape or device mismatch')
        history={s.history_length for s in self.stores};capacity={s.physical_capacity for s in self.stores}
        if len(history)!=1 or len(capacity)!=1:raise RuntimeError('layers disagree on active KV layout')
        # Existing history guard scans history once. The additional arbitrary-
        # bias rejection scans only the small tree square, never all K per layer.
        tree=mask[...,-nodes:]
        tree_is_hard=(tree==0)|(tree==torch.finfo(torch.float32).min)|torch.isneginf(tree)
        invalid=(mask[...,:logical_base]!=0).any() | (~tree_is_hard).any()
        if bool(invalid):raise ValueError('only zero history and hard tree-mask values are supported; arbitrary bias rejected')
        from ring_kv_cache import prepare_tree
        plan=prepare_tree(tree==0)
        expected=(logical_base+self.delta+plan.depth_tensor).view(1,1,nodes).expand(3,1,nodes)
        if not torch.equal(pos,expected):raise RuntimeError('mRoPE positions disagree with semantic base and tree depth')
        allowed=self.stores[0].build_attention_mask(plan)
        # Construct directly from the validated history/ancestor layout. This
        # is not a general FP32-mask cast and cannot round arbitrary bias.
        prepared=torch.full(allowed.shape,float('-inf'),dtype=torch.float16,device=self.device)
        prepared.masked_fill_(allowed,0)
        self.prepared_mask=prepared;self.mask_preparations+=1
        self.mask_prepare_records.append(dict(round_index=self.completed_forwards,
            shape=list(prepared.shape),bytes=prepared.numel()*prepared.element_size(),
            host_wall_s=time.perf_counter()-started,
            host_wall_scope='includes round guards/host checks and enqueue; excludes pending CUDA tail; entire work is inside request timing'))
        self.pending=dict(logical_base=logical_base,nodes=nodes,semantic_rope_base=logical_base+self.delta,
            variant='active_width',attention_variant=ATTENTION_VARIANT,
            active_attention_key_tokens=prepared.shape[-1],
            reserved_physical_capacity=self.stores[0].physical_capacity,
            excluded_padding_key_tokens=self.stores[0].physical_capacity-prepared.shape[-1],
            shared_fp16_mask_bytes=prepared.numel()*prepared.element_size(),mask_preparations=1)
        self.plan=plan;self.update_counts=[0]*len(self.stores);self.attention_counts=[0]*len(self.stores)
        self.in_forward=True
        try:
            result=self.original_forward(**dict(kwargs,attention_mask=prepared))
            if (any(x!=1 for x in self.update_counts) or any(x!=1 for x in self.attention_counts)
                    or any(s.active_tree_nodes!=nodes for s in self.stores)):
                raise RuntimeError('each Target layer must update once and attend once with the shared mask')
            if self.length(cache)!=logical_base+nodes:raise RuntimeError('post-forward semantic cache length mismatch')
            self.completed_forwards+=1
            return result
        except BaseException:
            self.failed=True;raise
        finally:
            self.in_forward=False;self.prepared_mask=None;self.mask_clears+=1

    def attention(self,module,query,key,value,mask,*,dropout=0.0,scaling=None,position_bias=None,**kwargs):
        index=self.attention_indices[id(module)]
        if (not self.in_forward or self.prepared_mask is None or mask is not self.prepared_mask
                or self.update_counts[index]!=1 or self.attention_counts[index]):
            raise RuntimeError('attention must use this round shared mask exactly once after its layer update')
        if dropout!=0.0 or position_bias is not None or kwargs.get('output_attentions',False):
            raise ValueError('inference attention without additional bias or requested weights required')
        self.selected_calls+=1
        try:
            output=self.executor(query,key,value,mask,scaling)
            if output.shape!=query.shape or output.dtype!=query.dtype or output.device!=query.device:
                raise RuntimeError('cuDNN output contract mismatch')
            self.attention_counts[index]+=1
            return output.transpose(1,2).contiguous(),None
        except Exception as exc:
            self.attention_failures.append(dict(type=type(exc).__name__,message=str(exc)))
            raise

    def compact(self,cache,selected,logical_base,tree_nodes):
        if cache is self.cache and any(s.recent_count+selected.numel()>s.recent_capacity for s in self.stores):
            raise RuntimeError('fixed text capacity exceeded; this experiment never evicts')
        return super().compact(cache,selected,logical_base,tree_nodes)

    def release(self):
        try:super().release()
        finally:self.prepared_mask=None

    def summary(self):
        result=super().summary()
        result.update(attention_variant=ATTENTION_VARIANT,tree_attention_backend='cudnn',
            selected_attention_calls=self.selected_calls,completed_target_forwards=self.completed_forwards,
            target_attention_layer_count=len(self.attention_indices),mask_preparations=self.mask_preparations,
            mask_clears=self.mask_clears,mask_retained=self.prepared_mask is not None,
            mask_prepare_records=list(self.mask_prepare_records),attention_failures=list(self.attention_failures),
            executor=self.executor.metadata(),fallback_calls=0,target_history_eviction=False,
            mask_scope='one FP16 {-inf,0} Tensor per verification shared by every Target layer; no layer-level data scan/cast',
            source_mask_guard_scope='one history check plus tree-square hard-value check per round; all checks counted in request',
            precision_claim='FP16 mask and cuDNN arithmetic need independent actual-output validation; no lossless claim')
        return result


@contextmanager
def cudnn_hard_mask_policy(target,*,recent_capacity=512,tree_capacity=30,interface=None,executor=None):
    if base._ACTIVE is not None:raise RuntimeError('nested/concurrent KV policies unsupported')
    if any(type(v) is not int or v<1 for v in (recent_capacity,tree_capacity)):
        raise ValueError('positive fixed capacities required')
    if interface is None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
        interface=ALL_ATTENTION_FUNCTIONS
    if not hasattr(interface,'_local_mapping'):raise TypeError('HF attention interface required')
    previous=interface['sdpa'];had_local='sdpa' in interface._local_mapping;local=interface._local_mapping.get('sdpa')
    if any(getattr(previous,name,False) for name in ('_strict_gqa_policy','_mixed_visual_policy','_hard_mask_policy')):
        raise RuntimeError('nested custom attention policies unsupported')
    executor=executor if executor is not None else GQAAttentionBackend('cudnn')
    if executor.backend!='cudnn':raise ValueError('only explicit cuDNN backend supported')
    bridge=base._bridge();state=CudnnHardMaskState(target,bridge,recent_capacity,tree_capacity,executor)
    original_forward=target.__dict__.get('forward',base._MISSING)
    def attention(module,query,key,value,attention_mask,**kwargs):
        if id(module) in state.attention_indices and base._ACTIVE is state and state.in_forward:
            return state.attention(module,query,key,value,attention_mask,**kwargs)
        return previous(module,query,key,value,attention_mask,**kwargs)
    attention._hard_mask_policy=True
    base._ACTIVE=state
    try:
        bridge._prefill=state.prefill;bridge._length=state.length;bridge._compact=state.compact
        target.forward=state.forward;interface['sdpa']=attention
        try:yield state
        except BaseException:state.failed=True;raise
    finally:
        try:state.finish()
        finally:
            bridge._prefill=state.original_prefill;bridge._length=state.original_length;bridge._compact=state.original_compact
            base._restore_attribute(target,'forward',original_forward)
            if had_local:interface['sdpa']=local
            else:del interface['sdpa']
            try:state.release()
            finally:base._ACTIVE=None
