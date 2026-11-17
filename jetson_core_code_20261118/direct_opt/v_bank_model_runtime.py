"""Request-local full-K / visual-V-bank Target integration, no model runner.

Production always uses frozen CUDA pack/fill. CPU tests must explicitly inject
reference_ops. Full native prefill, Draft, original mRoPE/mask and greedy tree
selection remain untouched. A converted cache is invalidated on context exit.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
import time
from types import MethodType
import weakref

import torch
import stride0_gqa_runtime as lifecycle
from stride0_gqa_runtime import math_precision_scope, precision_flags
from partitioned_v_bank import VisualPartition, prepare_tree, prepare_commit
from grouped_v_bank_math import grouped_v_bank_math
import v_bank_cuda_ops as cuda_ops

VARIANT = 'full_k_partitioned_visual_v_grouped_math'
MODES = ('fp16_partition','int8_visual')
_ACTIVE = None
_MISSING = object()
_MARKERS = ('_stride0_tree_policy','_strict_gqa_policy','_grouped_tree_policy',
            '_grouped_no_weights_tree_policy','_mixed_visual_policy','_v_bank_model_policy')


def _sid(tensor):
    return str(tensor.device), tensor.untyped_storage().data_ptr()


def _bytes(tensor):
    return tensor.numel()*tensor.element_size()


def _restore_attribute(obj,name,old):
    if old is _MISSING: obj.__dict__.pop(name,None)
    else: setattr(obj,name,old)


def _check_tensor(tensor,device,tokens,name):
    if (not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided
            or tensor.shape!=(1,4,tokens,128) or tensor.dtype!=torch.float16
            or tensor.device!=device or tensor.requires_grad or tensor.stride(-1)!=1
            or any(v<0 for v in tensor.stride())):
        raise ValueError(name+' must be FP16 [1,4,tokens,128] on the request device')


def _partition_from_values(target,values):
    """Same real-token/grid rules as identify_visual_positions; one prompt host copy."""
    ids=values.get('input_ids')
    if not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.long or ids.shape[1]<1:
        raise ValueError('batch-one nonempty int64 prompt required')
    host=ids.detach().cpu()[0].tolist()
    config=target.config
    merge=getattr(config.vision_config,'spatial_merge_size',None)
    if type(merge) is not int or merge<1:raise ValueError('positive spatial merge size required')
    positions=[];counts={};grids={};token_ids={}
    for kind in ('video','image'):
        token=getattr(config,kind+'_token_id',None)
        grid=values.get(kind+'_grid_thw')
        rows=grid.detach().cpu().tolist() if isinstance(grid,torch.Tensor) else grid
        if token is None:
            if rows is not None:raise ValueError('visual grid has no configured token ID')
            continue
        if type(token) is not int or token<0:raise ValueError('invalid visual token ID')
        selected=[i for i,value in enumerate(host) if value==token]
        expected=0
        for row in rows or []:
            if len(row)!=3 or any(type(v) is not int or v<1 for v in row):raise ValueError('invalid visual grid row')
            t,h,w=row
            if h%merge or w%merge:raise ValueError('visual grid disagrees with spatial merge')
            expected+=t*h*w//(merge*merge)
        if len(selected)!=expected:raise ValueError(kind+' token IDs disagree with actual grid')
        positions.extend(selected);counts[kind]=len(selected);grids[kind]=rows;token_ids[kind]=token
    if len(set(positions))!=len(positions):raise ValueError('visual token IDs overlap')
    provenance=dict(input_ids_sha256=hashlib.sha256(json.dumps(host).encode()).hexdigest(),
        visual_token_ids=token_ids,grid_thw=grids,spatial_merge_size=merge,visual_counts=counts,
        identification='actual input IDs matched to configured image/video IDs and grid; no prefix-length inference')
    return VisualPartition.prepare(len(host),positions,provenance=provenance),counts


class _Layer:
    def __init__(self,layer,bank,route,mode):
        self.layer,self.bank,self.route,self.mode=layer,bank,route,mode
        self.keys16,self.text_values16=layer.keys,layer.values
        self.committed=0;self.active_plan=None

    def check(self,plan=None,*,active=False):
        self.bank.validate_metadata()
        if self.bank.failed:raise RuntimeError('V bank is poisoned')
        nodes=self.active_plan.nodes if self.active_plan is not None else 0
        length=self.route.P+self.committed+nodes
        _check_tensor(self.keys16,self.bank.visual_values.device,length,'stored K')
        _check_tensor(self.text_values16,self.bank.visual_values.device,length-self.route.M,'stored compact V')
        if self.layer.keys is not self.keys16 or self.layer.values is not self.text_values16:
            raise RuntimeError('cache owning references changed outside this request policy')
        if plan is not None and (plan.partition is not self.route.partition or plan.committed_tokens!=self.committed
                                or (active and self.active_plan is not plan)
                                or (not active and self.active_plan is not None)):
            raise RuntimeError('layer tree/commit state disagrees')

    def append(self,keys,values,plan):
        self.check(plan)
        device=self.keys16.device
        _check_tensor(keys,device,plan.nodes,'new K');_check_tensor(values,device,plan.nodes,'new V')
        if _sid(keys)==_sid(values):raise ValueError('new K/V must have separate backing')
        # Full K uses exactly native cat; compact V uses the same operation.
        k=torch.cat((self.keys16,keys),dim=-2)
        v=torch.cat((self.text_values16,values),dim=-2)
        self.keys16,self.text_values16=k,v
        self.layer.keys,self.layer.values=k,v
        self.active_plan=plan
        return k,v

    def commit(self,decision,ki,vi):
        self.check(decision.tree_plan,active=True)
        k=self.keys16.index_select(-2,ki)
        v=self.text_values16.index_select(-2,vi)
        self.keys16,self.text_values16=k,v
        self.layer.keys,self.layer.values=k,v
        self.committed+=len(decision.selected);self.active_plan=None
        return k,v

    def tensors(self):
        return [x for x in (self.keys16,self.text_values16,self.bank.visual_values,
                            self.bank.scales,self.bank.error_flag) if x is not None]


class VBankModelPolicy(lifecycle.Stride0AttentionPolicy):
    def __init__(self,target,bridge,previous,mode,reference_ops):
        super().__init__(target,'native',bridge,previous)
        if mode not in MODES:raise ValueError('choose fp16_partition or int8_visual')
        if target.training:raise ValueError('eval Target required; training unsupported')
        self.mode=mode;self.ops=cuda_ops if reference_ops is None else reference_ops
        self.reference_only=reference_ops is not None
        self.stores=[];self.layer_restore=[];self.old_value_refs=[]
        self.partition=None;self.route=None;self.device=None;self.committed=0
        self.in_forward=False;self.plan=None;self.update_counts=[];self.current_phase='prefill'
        self.final_storage=None;self._finished=False
        self.memory={name:None for name in ('prefill','conversion','decode','full_request_peak_allocated_bytes','full_request_peak_reserved_bytes')}
        self.io={};self.record.update(variant=VARIANT,mode=mode,target_cache_policy=VARIANT,
            new_kv_policy_installed=True,cache_functions_changed=True,original_length_unchanged=True,
            full_k_native_cat_index_select=True,full_fp16_value_reconstruction=False,
            tree_value_operand='one directly filled FP32 [1,4,S,128], no FP16 full V',
            native_unused_probability_cast_preserved=True,reference_only=self.reference_only,
            setup_status_calls=0,result_status_calls=0,update_calls=0,commit_calls=0,fill_calls=0,
            primitive_receipt_calls=0,conversion_wall_s=0.0,round_guard_wall_s=0.0,commit_wall_s=0.0,
            model_GPU_output_equivalence_verified=False,quality_pass=None,actual_dram_measured=False,
            cache_after_context='layers cleared and request invalidated; cannot resume converted cache',
            status_scope='aggregate pack/fill device errors; not a proof of all model arithmetic being finite')
        self.record['conversion_boundary']='immediately after native Target prefill, before Draft adaptor; fixed mode only'

    def _peak(self):
        if self.device is None or self.device.type!='cuda':
            return dict(peak_allocated_bytes=None,peak_reserved_bytes=None,allocated_at_boundary_bytes=None,
                        reserved_at_boundary_bytes=None,note='CPU reference; CUDA memory unavailable')
        torch.cuda.synchronize(self.device)
        return dict(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)),
            allocated_at_boundary_bytes=int(torch.cuda.memory_allocated(self.device)),
            reserved_at_boundary_bytes=int(torch.cuda.memory_reserved(self.device)))

    def _phase(self,name):
        self.memory[name]=self._peak()
        if self.device is not None and self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)

    def _io(self,name,read,write):
        item=self.io.setdefault(name,dict(calls=0,read_bytes=0,write_bytes=0,read_plus_write_bytes=0))
        item['calls']+=1;item['read_bytes']+=int(read);item['write_bytes']+=int(write)
        item['read_plus_write_bytes']+=int(read+write)

    def _storage(self):
        known={};per_layer=[]
        for item in self.stores:
            item.check()
            entries={}
            for name,tensor in [('keys',item.keys16),('text_values',item.text_values16),
                                ('visual_values',item.bank.visual_values),('scales',item.bank.scales),('error_flag',item.bank.error_flag)]:
                if tensor is None:continue
                entries[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),
                    logical_bytes=_bytes(tensor),backing_bytes=tensor.untyped_storage().nbytes())
                if tensor.numel():known[_sid(tensor)]=tensor.untyped_storage().nbytes()
            per_layer.append(entries)
        route_bytes=0 if self.route is None else self.route.prefix_route.untyped_storage().nbytes()
        return dict(layer_count=len(self.stores),layers=per_layer,unique_layer_backing_bytes=sum(known.values()),
            shared_device_route_bytes=route_bytes,owned_plus_shared_device_bytes=sum(known.values())+route_bytes,
            route_counted_once=True,host_partition_route_bytes=0 if self.partition is None else self.partition.prefix_route.untyped_storage().nbytes(),
            original_value_tensor_refs_alive=sum(ref() is not None for ref in self.old_value_refs),
            original_full_V_known_backing_retained=False,allocator_reserved_is_separate=True,
            scope='known live roots; bytes do not independently prove absence of external untracked aliases')

    def prefill(self,target,values,counts,**kwargs):
        if target is not self.target:return self.original_prefill(target,values,counts,**kwargs)
        try:
            self.device=values['input_ids'].device
            if self.reference_only and self.device.type!='cpu':raise ValueError('reference_ops are CPU-only')
            if not self.reference_only and self.device.type!='cuda':raise ValueError('default model V-bank path requires CUDA; no fallback')
            output=super().prefill(target,values,counts,**kwargs)
            self._phase('prefill');self.current_phase='conversion';start=time.perf_counter()
            self.partition,visual_counts=_partition_from_values(target,values)
            self.route=self.ops.prepare_partition_route(self.partition,self.device)
            self._io('route_upload',4*self.route.P,4*self.route.P)
            self.record.update(visual_counts=visual_counts,partition=self.partition.receipt(),route=self.route.receipt())
            for index,layer in enumerate(self.cache.layers):
                _check_tensor(layer.keys,self.device,self.route.P,'prefill K')
                _check_tensor(layer.values,self.device,self.route.P,'prefill V')
                if _sid(layer.keys)==_sid(layer.values):raise ValueError('prefill K/V share backing')
                old=weakref.ref(layer.values);old_storage=_sid(layer.values)
                bank,text=self.ops.pack_v_cuda(layer.values,self.route,mode=self.mode)
                if bank.route is not self.route or bank.mode!=self.mode:raise RuntimeError('pack returned another route/mode')
                bank.validate_metadata();_check_tensor(text,self.device,self.route.R,'packed text V')
                owned=[layer.keys,text,bank.visual_values,bank.error_flag]+([] if bank.scales is None else [bank.scales])
                ids=[_sid(x) for x in owned if x.numel()]
                if len(set(ids))!=len(ids) or old_storage in ids:raise RuntimeError('partition output retains or aliases full V backing')
                self.old_value_refs.append(old)
                self.layer_restore.append((layer,layer.__dict__.get('update',_MISSING)))
                layer.values=text
                store=_Layer(layer,bank,self.route,self.mode);self.stores.append(store)
                def update(instance,keys,values,*a,_index=index,**kw):
                    return self._update(_index,instance,keys,values,a,kw)
                layer.update=MethodType(update,layer)
                del owned,ids,text,bank,store
                if old() is not None:raise RuntimeError('original full V tensor still referenced after installation')
                self._io('prefill_pack',2*4*self.route.P*128,
                    self.stores[-1].bank.visual_values.numel()*self.stores[-1].bank.visual_values.element_size()
                    +(0 if self.stores[-1].bank.scales is None else _bytes(self.stores[-1].bank.scales))+2*4*self.route.R*128)
            banks=[item.bank for item in self.stores]
            self.record['setup_status']=self.ops.validate_pack_status(banks);self.record['setup_status_calls']+=1
            if any(not b.validated or b.failed for b in banks):raise RuntimeError('setup aggregate status did not validate every bank')
            if self.original_length(self.cache)!=self.route.P:raise RuntimeError('K semantic length changed during partition')
            self.record['storage_after_conversion']=self._storage()
            self.record['conversion_wall_s']=time.perf_counter()-start
            self._phase('conversion');self.current_phase='decode'
            return output
        except BaseException as exc:
            self._fail('partition_prefill',exc);raise

    def _update(self,index,layer,keys,values,args,kwargs):
        try:
            if (self.failed or not self.in_forward or self.pending is None or self.plan is None
                    or args or kwargs or self.stores[index].layer is not layer or self.update_counts[index]):
                raise RuntimeError('layer update outside its unique active tree call')
            result=self.stores[index].append(keys,values,self.plan)
            self.update_counts[index]+=1;self.record['update_calls']+=1
            self._io('key_cat',_bytes(result[0]),_bytes(result[0]))
            self._io('compact_value_cat',_bytes(result[1]),_bytes(result[1]))
            return result
        except BaseException as exc:
            self._fail('update',exc);raise

    def forward(self,*args,**kwargs):
        if self.in_prefill:return super().forward(*args,**kwargs)
        try:
            if self.failed or self._finished or self.pending is not None or self.in_forward:
                raise RuntimeError('request failed/finished or previous tree not committed')
            lifecycle._check_precision()
            cache=kwargs.get('past_key_values')
            if args or cache is None or cache is not self.cache:raise RuntimeError('unknown or missing request cache')
            ids,mask,pos=(kwargs.get(k) for k in ('input_ids','attention_mask','position_ids'))
            if not isinstance(ids,torch.Tensor) or ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.long:
                raise ValueError('explicit tree IDs required')
            t=ids.shape[1];base=self.original_length(cache)
            if (not 1<=t<=30 or not isinstance(mask,torch.Tensor) or mask.shape!=(1,1,t,base+t)
                    or mask.dtype!=torch.float32 or mask.device!=self.device
                    or not isinstance(pos,torch.Tensor) or pos.shape!=(3,1,t) or pos.dtype!=torch.long or pos.device!=self.device
                    or ids.device!=self.device or kwargs.get('use_cache') is not True
                    or kwargs.get('logits_to_keep')!=0 or kwargs.get('return_dict') is not True
                    or any(kwargs.get(k) is not None for k in ('inputs_embeds','pixel_values','pixel_values_videos'))
                    or set(kwargs)-{'input_ids','past_key_values','attention_mask','position_ids','use_cache','logits_to_keep','return_dict'}):
                raise ValueError('full native tree IDs/mask/positions/kwargs contract differs')
            start=time.perf_counter()
            tree=mask[0,0,:,-t:].detach().cpu()
            if bool((mask[...,:base]!=0).any().item()) or not bool(((tree==0)|(tree==torch.finfo(torch.float32).min)).all()):
                raise ValueError('requires original hard tree mask and fully visible history')
            plan=prepare_tree(self.partition,self.committed,tree==0)
            expected=(base+self.record['native_rope_delta']+torch.tensor([len(a)-1 for a in plan.ancestors])).view(1,1,t).expand(3,1,t)
            if not torch.equal(pos.detach().cpu(),expected):raise ValueError('absolute mRoPE positions disagree with actual ancestor depth')
            if plan.semantic_before!=base:raise RuntimeError('committed K length disagrees with partition')
            for item in self.stores:item.check(plan)
            self.record['round_guard_wall_s']+=time.perf_counter()-start
            self._io('round_guard_host_payload',4*t*t+8*3*t,4*t*t+8*3*t)
            self.plan=plan;self.update_counts=[0]*28
            self.pending=dict(base=base,nodes=t,mask=mask,positions=pos,layer_calls=[0]*28)
            self.in_forward=True;self.record['verify_forward_calls']+=1
            output=self.original_forward(**kwargs)
            if (self.update_counts!=[1]*28 or self.pending['layer_calls']!=[1]*28
                    or output.past_key_values is not cache or self.original_length(cache)!=base+t):
                raise RuntimeError('every Target layer must update and attend once on the full tree')
            self.record['verify_query_tokens']+=t
            self.record['rounds'].append(dict(cache_before=base,tree_nodes=t,cache_after=base+t,
                compact_value_tokens=self.route.R+self.committed+t,layer_calls=list(self.pending['layer_calls']),
                update_counts=list(self.update_counts),selected_tree_indices=None))
            return output
        except BaseException as exc:
            self._fail('forward',exc);raise
        finally:self.in_forward=False

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
            output,receipt,debug=grouped_v_bank_math(query,key,item.bank,value,attention_mask,scaling,
                fill_impl=self.ops.fill_value_fp32_ if self.reference_only else None,
                record_receipt=first,debug_intermediates=False)
            if debug is not None or (first and receipt is None) or (not first and receipt is not None):
                raise RuntimeError('unexpected primitive receipt/debug contract')
            if first:self.record['representative_receipt']=receipt;self.record['primitive_receipt_calls']+=1
            if output.shape!=(1,self.plan.nodes,28,128) or output.dtype!=torch.float16 or not output.is_contiguous():
                raise RuntimeError('attention output differs from HF layout')
            self.pending['layer_calls'][index]+=1
            self.record['layer_calls'][index]+=1;self.record['completed_calls']+=1;self.record['candidate_calls']+=1
            self.record['fill_calls']+=1
            self._io('value_operand_fill',_bytes(item.bank.visual_values)+(0 if item.bank.scales is None else _bytes(item.bank.scales))+_bytes(value),
                     4*4*key.shape[-2]*128)
            return output,None
        except BaseException as exc:
            self._fail('attention',exc);raise

    def compact(self,cache,selected,base,tree_nodes):
        start=time.perf_counter()
        try:
            if cache is not self.cache:raise RuntimeError('unknown cache cannot use request compact')
            if self.failed or self.in_forward or self.pending is None or self.plan is None:
                raise RuntimeError('commit outside completed tree')
            if (base!=self.plan.semantic_before or tree_nodes!=self.plan.nodes or self.update_counts!=[1]*28
                    or self.pending['layer_calls']!=[1]*28 or self.original_length(cache)!=base+tree_nodes
                    or not isinstance(selected,torch.Tensor) or selected.ndim!=1 or selected.dtype!=torch.long
                    or selected.device!=self.device):raise RuntimeError('invalid shared acceptance boundary')
            # One accepted host list and one pair of indices shared by every layer.
            positions=tuple(selected.detach().cpu().tolist())
            decision=prepare_commit(self.plan,positions)
            ki=decision.key_indices.to(device=self.device);vi=decision.text_indices.to(device=self.device)
            for item in self.stores:item.check(self.plan,active=True)
            for item in self.stores:
                k,v=item.commit(decision,ki,vi)
                self._io('key_compact',_bytes(k),_bytes(k));self._io('compact_value_compact',_bytes(v),_bytes(v))
            self.committed+=len(positions)
            if self.original_length(cache)!=self.route.P+self.committed:raise RuntimeError('K compact semantic length failed')
            self.record['rounds'][-1].update(selected_tree_indices=list(positions),cache_after_commit=base+len(positions),
                compact_value_after_commit=self.route.R+self.committed)
            self.record['commit_calls']+=1
            self._io('shared_commit_index_upload',8*(ki.numel()+vi.numel()),8*(ki.numel()+vi.numel()))
            self.plan=None;self.pending=None;self.update_counts=[]
        except BaseException as exc:
            self._fail('compact',exc);raise
        finally:self.record['commit_wall_s']+=time.perf_counter()-start

    def validate_result(self,result):
        try:
            lifecycle._check_precision()
            counts=result.forward_counts;rounds=[r for r in result.rounds if r.get('kind')=='tree'];n=counts['target_verify_calls']
            if (self.failed or self.pending is not None or self.in_forward or self._finished or self.record['failures']
                    or counts['target_prefill_calls']!=1 or self.record['prefill_calls']!=1 or self.record['setup_status_calls']!=1
                    or self.record['verify_forward_calls']!=n or len(rounds)!=n or len(self.record['rounds'])!=n
                    or self.record['commit_calls']!=n or self.record['verify_query_tokens']!=counts['target_verify_query_tokens']
                    or self.record['layer_calls']!=[n]*28 or self.record['update_calls']!=28*n
                    or self.record['selected_calls']!=28*n or self.record['completed_calls']!=28*n or self.record['fill_calls']!=28*n
                    or self.record['primitive_receipt_calls']!=int(n>0) or self.record['result_status_calls']):
                raise RuntimeError('incomplete V-bank request lifecycle/coverage')
            if counts['target_compaction_calls']!=n:raise RuntimeError('compaction counts differ')
            if counts['target_compaction_selected_tokens']!=self.committed:
                raise RuntimeError('accepted-token counts differ')
            if result.target_cache_length!=self.original_length(self.cache):
                raise RuntimeError('result final K semantic length differs')
            for observed,actual in zip(self.record['rounds'],rounds):
                if (observed['cache_before']!=actual['target_cache_before'] or observed['tree_nodes']!=actual['tree_nodes']
                        or observed['selected_tree_indices']!=actual['selected_tree_indices']
                        or observed['cache_after_commit']!=actual['target_cache_after']):
                    raise RuntimeError('V-bank receipts disagree with actual accepted path')
            if self.bridge._length is not self.original_length:raise RuntimeError('original bridge length changed')
            self.record['result_status']=self.ops.check_status([item.bank for item in self.stores],stage='model_result_boundary')
            self.record['result_status_calls']+=1;self.record['result_coverage_validated']=True
            self.record['storage_at_result']=self._storage()
        except BaseException as exc:
            self._fail('result',exc);raise

    def finish(self):
        if self._finished:return
        try:
            if self.cache is not None:
                if not self.failed and not self.record.get('result_coverage_validated'):
                    raise RuntimeError('context cannot accept output before final status/result validation')
                if self.failed:
                    # A partial update/commit can leave unequal layer lengths;
                    # preserve the original error rather than replacing it with
                    # a secondary full-state validation failure during cleanup.
                    try:self.final_storage=self._storage()
                    except BaseException as exc:
                        self.final_storage=dict(partial=True,unavailable_reason=repr(exc))
                else:self.final_storage=self._storage()
            self.memory[self.current_phase]=self._peak()
            for key in ('allocated','reserved'):
                vals=[v['peak_'+key+'_bytes'] for v in self.memory.values() if isinstance(v,dict) and v.get('peak_'+key+'_bytes') is not None]
                self.memory['full_request_peak_'+key+'_bytes']=max(vals) if vals else None
        except BaseException as exc:
            self._fail('finish',exc);raise
        finally:self._finished=True

    def release(self):
        # Methods restore before destroying ownership; even a poison/partial install is unusable.
        errors=[]
        for layer,previous in reversed(self.layer_restore):
            try:_restore_attribute(layer,'update',previous)
            except BaseException as exc:errors.append(exc)
        if self.cache is not None:
            for layer in self.cache.layers:
                layer.keys=None;layer.values=None
                if hasattr(layer,'is_initialized'):layer.is_initialized=False
            self.cache.layers.clear()
            self.cache._v_bank_invalidated=True
        self.stores.clear();self.layer_restore.clear();self.cache=None;self.pending=None
        self.plan=None;self.route=None;self.partition=None;self.prefill_ids=None;self.in_prefill=False
        self.record['finalized']=True
        if errors:raise RuntimeError('layer method restoration failed') from errors[0]

    def summary(self):
        result=copy.deepcopy(self.record)
        result.update(memory_phase_peaks=copy.deepcopy(self.memory),final_storage=copy.deepcopy(self.final_storage),
            logical_io=copy.deepcopy(self.io),actual_dram_measured=False,reference_only=self.reference_only,
            logical_io_scope='ideal payload reads/writes; route/index and round guard transfers separate; '
                'scale lanes, metadata/error flags, CPU reference temporaries, scalar reductions, allocator/BLAS excluded',
            original_value_tensor_refs_alive=sum(ref() is not None for ref in self.old_value_refs))
        return result


@contextmanager
def v_bank_model_policy(target,mode='fp16_partition',*,interface=None,bridge=None,reference_ops=None):
    global _ACTIVE
    if _ACTIVE is not None:raise RuntimeError('nested/concurrent V-bank policy unsupported')
    import ring_cache_runtime as cache_lifecycle
    if cache_lifecycle._ACTIVE is not None:raise RuntimeError('nested cache policy unsupported')
    if mode not in MODES:raise ValueError('unsupported V-bank mode')
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
    state=VBankModelPolicy(target,bridge,previous,mode,reference_ops)
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
        try:
            yield state
        except BaseException as exc:
            state._fail('context',exc)
            try:state.finish()
            except BaseException as final_exc:state._fail('finish_after_failure',final_exc)
            raise
        else:state.finish()


def sys_module():
    import sys
    return sys.modules[__name__]
