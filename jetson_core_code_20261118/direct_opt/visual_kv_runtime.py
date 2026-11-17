"""Experimental no-eviction visual INT8 / text FP16 Target KV runtime.

Native prefill and Draft stay unchanged. All visual positions are retained in a
separate bank; original nonvisual prompt + generated text + tree stay FP16.
The custom kernel uses one softmax over both banks and original mask columns.
No fallback, full-cache dequantization, RoPE adjustment or token eviction.
CPU tests must inject explicit reference pack/attention functions; the production
runner exposes no reference switch. All conversion/checks/map creation are inside
request timing. Quantized and FP16-visual controls use the identical kernel.
"""
from __future__ import annotations

from contextlib import contextmanager
import operator
from pathlib import Path
import sys
from types import MethodType
import weakref

import torch

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from direct_opt.visual_kv_int8 import partition_prefix,prepare_fp16_positions
from direct_opt.triton_mixed_attention import pack_visual_cuda,mixed_attention
from active_ring_kv_cache import ActiveRingKVLayer
import ring_cache_runtime as base_runtime

VARIANT='segmented_visual_no_eviction'


def identify_visual_positions(target,values):
    ids=values['input_ids']
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.long:
        raise ValueError('batch-one int64 prompt IDs required')
    config=target.config
    merge=int(config.vision_config.spatial_merge_size)
    if merge<1:raise ValueError('invalid spatial merge size')
    # One prompt-ID host transfer, once per request; counted in conversion.
    host_ids=ids[0].tolist();selected=[];counts={}
    for kind,field in [('video','video_token_id'),('image','image_token_id')]:
        token=getattr(config,field,None)
        if token is None:continue
        positions=[i for i,t in enumerate(host_ids) if t==int(token)]
        grid=values.get(kind+'_grid_thw')
        expected=0
        if grid is not None:
            rows=grid.tolist() if isinstance(grid,torch.Tensor) else grid
            for t,h,w in rows:
                if min(t,h,w)<1 or h%merge or w%merge:raise ValueError('invalid visual grid')
                expected+=int(t*h*w//(merge*merge))
        if len(positions)!=expected:raise ValueError(f'{kind} prompt IDs ({len(positions)}) disagree with grid ({expected})')
        selected.extend(positions);counts[kind]=len(positions)
    if len(set(selected))!=len(selected):raise ValueError('image and video token IDs overlap')
    selected.sort()
    return partition_prefix(len(host_ids),selected),counts


class SegmentedLayer(ActiveRingKVLayer):
    @classmethod
    @torch.inference_mode()
    def convert(cls,keys,values,partition,*,quantized,max_generated,tree_capacity,pack):
        visual=pack(keys,values,partition,quantized=quantized)
        positions=torch.tensor(partition.nonvisual,device=keys.device,dtype=torch.long)
        text_k=keys.index_select(-2,positions);text_v=values.index_select(-2,positions)
        invalid=(~torch.isfinite(text_k).all()) | (~torch.isfinite(text_v).all())
        self=super().from_prefill(text_k,text_v,mode='protected_prompt',
                                 recent_capacity=max_generated,tree_capacity=tree_capacity)
        self.visual=visual;self.partition=partition;self.original_prompt=partition.prefix_tokens
        self.next_position=partition.prefix_tokens
        self.slot_positions[:self.prefix_length].copy_(positions)
        self.initial_text_gather_payload_bytes=text_k.numel()*text_k.element_size()*2
        for k in ('read_bytes','write_bytes'):
            self.payload_copy_totals[k]+=self.initial_text_gather_payload_bytes
        self.payload_copy_totals['total_bytes']+=2*self.initial_text_gather_payload_bytes
        self.payload_copy_totals['metadata_update_write_bytes']+=positions.numel()*8
        self.conversion_invalid=invalid | (visual.nonfinite_flag.ne(0) if visual.nonfinite_flag is not None else False)
        return self

    @torch.inference_mode()
    def build_attention_mask(self,plan):
        self._check_plan(plan)
        if self.active_tree_nodes and self._plan is not plan:raise RuntimeError('different pending tree')
        mask=torch.ones((1,1,plan.nodes,self.next_position+plan.nodes),dtype=torch.bool,device=self.keys.device)
        mask[...,self.next_position:].copy_(plan.mask)
        return mask

    @torch.inference_mode()
    def commit_prepared(self,decision):
        # Reject before inherited gather/write; fixed text capacity is a limit,
        # never an eviction rule. An EOS-clipped path remains in original order.
        from ring_kv_cache import RingCommitPlan
        if not isinstance(decision,RingCommitPlan):raise TypeError('RingCommitPlan required')
        if self.recent_count+decision.accepted_tokens>self.recent_capacity:
            raise RuntimeError('generated KV exceeds fixed capacity; no eviction permitted')
        return super().commit_prepared(decision)

    def metadata(self):
        text=super().metadata();visual=self.visual.storage_report()
        return dict(variant=VARIANT,mode='int8' if self.visual.quantized else 'fp16_visual',
                    original_prompt_tokens=self.original_prompt,nonvisual_prompt_tokens=self.prefix_length,
                    visual_tokens=self.visual.logical_tokens,committed_tokens=self.recent_count,
                    next_position=self.next_position,active_tree_nodes=self.active_tree_nodes,
                    fp16_active_tokens=self.attention_length(),max_generated=self.recent_capacity,
                    visual_bank=visual,text_backing=text,
                    visual_unique_backing_bytes=visual['unique_backing_bytes'],
                    text_unique_backing_bytes=text['owned_storage_bytes'],
                    total_unique_backing_bytes=visual['unique_backing_bytes']+text['owned_storage_bytes'],
                    payload_copy_totals=dict(self.payload_copy_totals),operation_counts=dict(self.operation_counts),
                    initial_text_gather_rw_bytes=2*self.initial_text_gather_payload_bytes,
                    visual_conversion_logical_read_bytes=self.visual.logical_tokens*self.keys.shape[0]*self.keys.shape[1]*128*2*2,
                    visual_conversion_padded_payload_and_scale_write_bytes=visual['kv_payload_bytes']+visual['scale_bytes'],
                    visual_conversion_position_write_bytes=visual['position_bytes'],
                    payload_copy_totals_scope='FP16 text gather/initialization/append/commit only; visual pack separately recorded; excludes finite-check reductions and allocator effects',
                    eviction=False,full_cache_dequantization=False,rotary_reapplied=False,
                    original_position_semantics_preserved=True,
                    scope='one layer: bank plus FP16 text/tree storage; per-round shared position map separate; logical traffic is not measured DRAM')


class VisualKVState(base_runtime.RingCacheState):
    def __init__(self,target,bridge,mode,max_generated,tree_capacity,pack,attention):
        super().__init__(target,bridge,mode,max_generated,tree_capacity)
        self.pack_impl=pack;self.attention_impl=attention
        self.partition=None;self.visual_counts={};self.position_map=None;self.logical_mask=None
        self.attention_counts=[];self.old_kv_refs=[]
        self.mixed_attention_calls=0;self.completed_target_forwards=0
        modules=tuple(layer.self_attn for layer in target.model.language_model.layers)
        self.attention_indices={id(m):i for i,m in enumerate(modules)}
        if not modules or len(self.attention_indices)!=len(modules):raise ValueError('unique Target attention layers required')

    def prefill(self,target,values,counts,**kwargs):
        if target is not self.target:return self.original_prefill(target,values,counts,**kwargs)
        if self.prefill_calls:raise RuntimeError('one prefill required per context')
        self.prefill_calls+=1;self.device=values['input_ids'].device
        out,delta,features=self.original_prefill(target,values,counts,**kwargs)
        self._close_phase('prefill');self.current_phase='conversion'
        self.cache=out.past_key_values;self.prompt=int(values['input_ids'].shape[1]);self.delta=int(delta)
        if self.original_length(self.cache)!=self.prompt:raise RuntimeError('native prefill does not cover full prompt')
        if len(self.cache.layers)!=len(self.attention_indices):raise RuntimeError('Target attention/cache layer count mismatch')
        self.partition,self.visual_counts=identify_visual_positions(target,values)
        self.bytes_per_token=0
        invalid=[]
        for index,layer in enumerate(self.cache.layers):
            k,v=layer.keys,layer.values
            if (k.shape!=v.shape or k.ndim!=4 or k.shape[0]!=1 or k.shape[1]!=4 or
                k.shape[-1]!=128 or k.shape[-2]!=self.prompt or k.dtype!=torch.float16 or
                v.dtype!=k.dtype or k.device!=self.device or v.device!=self.device):
                raise RuntimeError('first configuration requires full-prefill FP16 [1,4,P,128] KV')
            self.old_kv_refs.extend((weakref.ref(k),weakref.ref(v)))
            store=SegmentedLayer.convert(k,v,self.partition,quantized=self.mode=='int8',
                max_generated=self.recent_capacity,tree_capacity=self.tree_capacity,pack=self.pack_impl)
            self.stores.append(store);invalid.append(store.conversion_invalid)
            saved={name:layer.__dict__.get(name,base_runtime._MISSING) for name in ('update','get_seq_length','get_mask_sizes')}
            self.layer_restore.append((layer,saved));layer.keys,layer.values=store.keys,store.values
            self.bytes_per_token+=2*k.shape[1]*k.shape[-1]*k.element_size()
            del k,v
            def update(instance,keys,values,*a,_index=index,**kw):return self._update(_index,keys,values,a,kw)
            def length(instance,_index=index):
                s=self.stores[_index];return s.next_position+s.active_tree_nodes
            def mask_sizes(instance,query_length,_index=index):
                q=operator.index(query_length)
                if isinstance(query_length,bool) or q<1 or q>self.tree_capacity:raise ValueError('invalid tree query length')
                return self.stores[_index].next_position+q,0
            layer.update=MethodType(update,layer);layer.get_seq_length=MethodType(length,layer)
            layer.get_mask_sizes=MethodType(mask_sizes,layer)
        # Exactly one aggregate nonfinite host check after all visual packs.
        if bool(torch.stack(invalid).any().item()):raise RuntimeError('nonfinite KV found during conversion')
        for s in self.stores:s.conversion_invalid=None
        if any(ref() is not None for ref in self.old_kv_refs):raise RuntimeError('original full FP16 KV Tensor still retained')
        self.prefill_metadata=self._cache_metadata()
        self._close_phase('conversion');self.current_phase='decode'
        return out,delta,features

    def forward(self,*args,**kwargs):
        own=self.cache is not None and kwargs.get('past_key_values') is self.cache
        if own:
            ids=kwargs.get('input_ids')
            if not isinstance(ids,torch.Tensor) or ids.ndim!=2:raise RuntimeError('explicit tree IDs required')
            nodes=int(ids.shape[1])
            if nodes<1 or nodes>self.tree_capacity:raise RuntimeError('tree size exceeds fixed capacity')
            committed={s.recent_count for s in self.stores}
            if len(committed)!=1:raise RuntimeError('segmented layers disagree on committed length')
            self.position_map=prepare_fp16_positions(self.partition,next(iter(committed))+nodes,self.device)
            self.logical_mask=kwargs.get('attention_mask')
            self.attention_counts=[0]*len(self.stores)
        result=super().forward(*args,**kwargs)
        if own:
            if any(x!=1 for x in self.attention_counts):raise RuntimeError('every Target layer must use mixed attention exactly once')
            self.completed_target_forwards+=1
            self.pending.update(variant=VARIANT,visual_mode=self.mode,visual_tokens=len(self.partition.visual),
                fp16_attention_tokens=self.stores[0].attention_length(),
                logical_attention_tokens=self.position_map.total_tokens,
                shared_position_map_bytes=self.position_map.indices.numel()*8)
        return result

    def attention(self,module,query,key,value,mask,*,dropout=0.0,scaling=None,position_bias=None,**kwargs):
        index=self.attention_indices[id(module)];store=self.stores[index]
        if (not self.in_forward or self.plan is None or self.position_map is None or
            self.update_counts[index]!=1 or self.attention_counts[index]):raise RuntimeError('mixed attention outside unique completed layer update')
        if dropout!=0.0 or position_bias is not None or kwargs.get('output_attentions',False):
            raise ValueError('mixed attention requires inference without extra position bias')
        if query.shape[1]!=28 or query.shape[-2]!=self.plan.nodes:raise ValueError('first mixed kernel requires Qwen 28 query heads')
        expected=store.attention_view()
        for actual,needed in zip((key,value),expected):
            if actual.shape!=needed.shape or actual.stride()!=needed.stride() or actual.data_ptr()!=needed.data_ptr():
                raise RuntimeError('attention must receive current FP16 text/tree view')
        # Parent validates/rebuilds a layout mask. The kernel consumes the exact
        # original additive values, indexed by original token-column receipts.
        if mask.shape!=self.logical_mask.shape:raise RuntimeError('logical mask width changed in Target')
        result=self.attention_impl(query,store.visual,key,value,self.position_map,self.logical_mask,scale=scaling)
        if result.shape!=query.shape or result.dtype!=query.dtype:raise RuntimeError('mixed attention output shape/dtype mismatch')
        self.attention_counts[index]+=1
        self.mixed_attention_calls+=1
        return result.transpose(1,2).contiguous(),None

    def compact(self,cache,selected,base,tree_nodes):
        result=super().compact(cache,selected,base,tree_nodes)
        if cache is self.cache:self.position_map=None;self.logical_mask=None
        return result

    def _cache_metadata(self):
        if self.cache is None:return None
        layers=[s.metadata() for s in self.stores]
        return dict(variant=VARIANT,mode=self.mode,semantic_length=self.length(self.cache),layer_count=len(layers),
                    first_layer=layers[0] if layers else None,
                    allocated_kv_storage_bytes=sum(x['visual_bank']['kv_payload_bytes']+x['text_backing']['allocated_kv_storage_bytes'] for x in layers),
                    owned_storage_bytes=sum(x['total_unique_backing_bytes'] for x in layers),
                    shared_position_map_bytes=0 if self.position_map is None else self.position_map.indices.numel()*8)

    def release(self):
        try:super().release()
        finally:self.position_map=None;self.logical_mask=None

    def summary(self):
        result=super().summary()
        result.update(variant=VARIANT,visual_mode=self.mode,eviction=False,quantized=self.mode=='int8',
            visual_counts=dict(self.visual_counts),max_generated=self.recent_capacity,
            kernel='triton_mixed_attention.mixed_attention; BM32 BN64; one online softmax over both banks',
            fallback_calls=0,original_full_fp16_kv_tensor_refs_alive=sum(ref() is not None for ref in self.old_kv_refs),
            mixed_attention_calls=self.mixed_attention_calls,
            completed_target_forwards=self.completed_target_forwards,
            target_attention_layer_count=len(self.attention_indices),
            precision_control='fp16_visual uses identical kernel/traversal without quantization',
            cache_after_context='request-local segmented cache invalidated',
            scope='All original token positions retained; visual bank only changes storage precision; logical copy bytes are not DRAM')
        return result


@contextmanager
def visual_kv_policy(target,mode='int8',*,max_generated=512,tree_capacity=30,interface=None,
                     pack_impl=None,attention_impl=None):
    if mode not in ('int8','fp16_visual'):raise ValueError('choose int8 or fp16_visual explicitly')
    if any(type(x) is not int or x<1 for x in (max_generated,tree_capacity)):raise ValueError('positive fixed capacities required')
    if base_runtime._ACTIVE is not None:raise RuntimeError('nested/concurrent KV policies unsupported')
    if interface is None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
        interface=ALL_ATTENTION_FUNCTIONS
    if not hasattr(interface,'_local_mapping'):raise TypeError('HF attention interface required')
    previous_attention=interface['sdpa'];had_local='sdpa' in interface._local_mapping
    local=interface._local_mapping.get('sdpa')
    if getattr(previous_attention,'_strict_gqa_policy',False) or getattr(previous_attention,'_mixed_visual_policy',False):
        raise RuntimeError('nested custom attention policies unsupported')
    bridge=base_runtime._bridge()
    state=VisualKVState(target,bridge,mode,max_generated,tree_capacity,pack_impl or pack_visual_cuda,attention_impl or mixed_attention)
    original_forward=target.__dict__.get('forward',base_runtime._MISSING)
    def attention(module,query,key,value,attention_mask,**kwargs):
        if id(module) in state.attention_indices and base_runtime._ACTIVE is state and state.in_forward:
            return state.attention(module,query,key,value,attention_mask,**kwargs)
        return previous_attention(module,query,key,value,attention_mask,**kwargs)
    attention._mixed_visual_policy=True
    base_runtime._ACTIVE=state
    try:
        bridge._prefill=state.prefill;bridge._length=state.length;bridge._compact=state.compact
        target.forward=state.forward;interface['sdpa']=attention
        try:yield state
        except BaseException:state.failed=True;raise
    finally:
        try:state.finish()
        finally:
            bridge._prefill=state.original_prefill;bridge._length=state.original_length;bridge._compact=state.original_compact
            base_runtime._restore_attribute(target,'forward',original_forward)
            if had_local:interface['sdpa']=local
            else:del interface['sdpa']
            try:state.release()
            finally:base_runtime._ACTIVE=None
