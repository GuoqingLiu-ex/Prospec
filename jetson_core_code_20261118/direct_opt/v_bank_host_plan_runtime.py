"""Request-local reuse of author CPU tree/retrieve, preserving frozen V-bank math.

Only the registered host_tree_joint producer may bind mask/position receipts.
Tensor identity/layout/version checks do not authenticate arbitrary raw CUDA
mutation. Inference tensors require the producer's exclusive same-stream use.
No GPU metadata reconstruction, no device error suppression, no CPU fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from types import MethodType

import torch
import v_bank_model_runtime as frozen
from partitioned_v_bank import prepare_tree

_BASE_POLICY = frozen.v_bank_model_policy
METHOD = 'author_cpu_tree_reuse_v1'
_ACTIVE = False


def _snapshot(x):
    if not isinstance(x,torch.Tensor):raise TypeError('actual tensor required')
    try:version=x._version
    except RuntimeError:
        if not torch.is_inference(x):raise
        version=None
    return (id(x),x.data_ptr(),str(x.device),x.dtype,tuple(x.shape),tuple(x.stride()),
            x.storage_offset(),version)


def _same(x,snapshot):
    if _snapshot(x)!=snapshot:raise RuntimeError('bound tensor identity/layout/version changed')


@dataclass(frozen=True)
class HostCommit:
    tree_plan: object
    selected: tuple


class HostTreeBroker:
    def __init__(self,state):
        self.state=state;self.current=None
        self.record=dict(method=METHOD,captured_trees=0,bound_forwards=0,consumed_forwards=0,
            accepted_paths=0,committed_paths=0,avoided_tree_position_array_D2H_calls=0,
            avoided_history_scalar_D2H_calls=0,avoided_commit_index_H2D_calls=0,
            gpu_index_api_calls={'arange':0,'add':0,'cat':0},
            gpu_index_payload_bytes_sum=0,gpu_index_payload_bytes_peak=0,
            gpu_index_construction_live_upper_bound_bytes=0,
            host_immutable_tree_retrieve_elements_peak=0,
            retained_existing_device_tensor_backing_bytes_peak=0,
            bound_tensors_are_existing_objects_not_new_device_copies=True,
            host_element_count_excludes_Python_object_overhead=True,
            no_new_persistent_device_metadata=True,no_GPU_metadata_readback=True,
            inference_tensor_mutation_contract='exclusive fixed producer and same stream; raw writes not detected',
            actual_kernel_count_measured=False,actual_DRAM_measured=False,
            scope='source-derived logical/API ledger, not profiler or hardware counter',rounds=[])

    def capture(self,tree,cache,base,delta):
        s=self.state
        if (self.current is not None or cache is not s.cache or s.pending is not None
                or base!=s.original_length(cache) or delta!=s.record['native_rope_delta']):
            raise RuntimeError('stale/nested host tree or wrong request boundary')
        tokens,retrieve,mask,depths=tree
        if (not isinstance(mask,torch.Tensor) or mask.device.type!='cpu' or mask.ndim!=4
                or mask.shape[:2]!=(1,1) or mask.shape[-1]!=mask.shape[-2]
                or mask.dtype not in (torch.float32,torch.bool)
                or not isinstance(retrieve,torch.Tensor) or retrieve.device.type!='cpu'
                or retrieve.dtype!=torch.long or retrieve.ndim!=2):
            raise ValueError('author tree mask/retrieve must already be CPU; no readback fallback')
        n=mask.shape[-1]
        if not 1<=n<=30 or tokens.shape!=(1,n) or tokens.device!=s.device or depths.shape!=(n,):
            raise ValueError('author tree token/depth shape/device differs')
        rows=mask[0,0].tolist()
        if any(v not in (0,1,False,True) for row in rows for v in row):
            raise ValueError('author CPU mask must contain only hard zero/one')
        plan=prepare_tree(s.partition,s.committed,torch.tensor(rows,dtype=torch.bool))
        paths=tuple(tuple(row) for row in retrieve.tolist())
        if not paths or not paths[0] or any(len(row)!=len(paths[0]) for row in paths):
            raise ValueError('rectangular nonempty author retrieve required')
        for row in paths:
            path=tuple(i for i in row if i!=-1)
            if row!=path+(-1,)*(len(row)-len(path)):
                raise ValueError('retrieve padding must be a trailing suffix')
            self._validate_path(plan,path)
        self.current=dict(plan=plan,paths=paths,cache=cache,base=base,delta=delta,nodes=n,
            tokens=tokens,token_snapshot=_snapshot(tokens),stage='captured')
        r=self.record;r['captured_trees']+=1
        r['host_immutable_tree_retrieve_elements_peak']=max(r['host_immutable_tree_retrieve_elements_peak'],
            sum(map(len,plan.ancestors))+sum(map(len,paths)))

    @staticmethod
    def _validate_path(plan,positions):
        if (not positions or positions[0]!=0 or any(type(i) is not int or not 0<=i<plan.nodes for i in positions)
                or len(set(positions))!=len(positions)):
            raise ValueError('accepted/retrieve path must start at root and have unique valid indices')
        seen=set()
        for node in positions:
            seen.add(node)
            if set(plan.ancestors[node])!=seen:raise ValueError('path must preserve complete ancestors')

    def bind_forward(self,cache,tokens,mask,pos,retrieve,base,delta):
        c=self.current
        if (c is None or c['stage']!='captured' or cache is not c['cache'] or tokens is not c['tokens']
                or (base,delta)!=(c['base'],c['delta'])):
            raise RuntimeError('forward binding differs from current author tree')
        _same(tokens,c['token_snapshot'])
        if (retrieve.shape!=(len(c['paths']),len(c['paths'][0])) or retrieve.dtype!=torch.long
                or retrieve.device!=tokens.device):raise ValueError('uploaded retrieve metadata differs')
        n=c['nodes']
        if (mask.shape!=(1,1,n,base+n) or mask.dtype!=torch.float32 or mask.device!=tokens.device
                or pos.shape!=(3,1,n) or pos.dtype!=torch.long or pos.device!=tokens.device):
            raise ValueError('constructed mask/absolute position metadata differs')
        c.update(mask=mask,positions=pos,retrieve=retrieve,mask_snapshot=_snapshot(mask),
            position_snapshot=_snapshot(pos),retrieve_snapshot=_snapshot(retrieve),stage='bound')
        roots={(str(x.device),x.untyped_storage().data_ptr()):x.untyped_storage().nbytes()
               for x in (tokens,mask,pos,retrieve)}
        self.record['retained_existing_device_tensor_backing_bytes_peak']=max(
            self.record['retained_existing_device_tensor_backing_bytes_peak'],sum(roots.values()))
        self.record['bound_forwards']+=1

    def consume_forward(self,cache,ids,mask,pos,base,nodes):
        c=self.current
        if (c is None or c['stage']!='bound' or cache is not c['cache'] or ids is not c['tokens']
                or mask is not c['mask'] or pos is not c['positions'] or (base,nodes)!=(c['base'],c['nodes'])):
            raise RuntimeError('missing/stale/foreign actual forward receipt')
        for x,k in [(ids,'token'),(mask,'mask'),(pos,'position'),(c['retrieve'],'retrieve')]:_same(x,c[k+'_snapshot'])
        c['stage']='forward';self.record['consumed_forwards']+=1
        self.record['avoided_tree_position_array_D2H_calls']+=2
        self.record['avoided_history_scalar_D2H_calls']+=1
        return c['plan']

    def bind_acceptance(self,best,keep,selected):
        c=self.current
        if c is None or c['stage']!='forward':raise RuntimeError('acceptance without unique completed forward')
        if (type(best) is not int or not 0<=best<len(c['paths']) or type(keep) is not int
                or not 1<=keep<=len(c['paths'][best])):raise ValueError('host posterior/clip selection invalid')
        positions=c['paths'][best][:keep];self._validate_path(c['plan'],positions)
        retrieve=c['retrieve'];_same(retrieve,c['retrieve_snapshot'])
        if (selected.dtype!=torch.long or selected.device!=retrieve.device or selected.shape!=(keep,)
                or selected.untyped_storage().data_ptr()!=retrieve.untyped_storage().data_ptr()
                or selected.storage_offset()!=retrieve.storage_offset()+best*retrieve.stride(0)
                or selected.stride()!=(retrieve.stride(1),)):
            raise ValueError('selected must be original retrieve[best,:keep] view')
        c.update(selected=selected,selected_snapshot=_snapshot(selected),accepted=positions,stage='accepted')
        self.record['accepted_paths']+=1
        return positions

    def consume_commit(self,cache,selected,base,nodes):
        c=self.current
        if (c is None or c['stage']!='accepted' or cache is not c['cache'] or selected is not c['selected']
                or (base,nodes)!=(c['base'],c['nodes'])):raise RuntimeError('commit not bound to accepted tuple')
        _same(selected,c['selected_snapshot']);c['stage']='committing'
        return HostCommit(c['plan'],c['accepted'])

    def committed(self,ki,vi):
        c=self.current
        if c is None or c['stage']!='committing':raise RuntimeError('unexpected commit completion')
        n=len(c['accepted']);payload=8*(ki.numel()+vi.numel());r=self.record
        r['committed_paths']+=1;r['avoided_commit_index_H2D_calls']+=2
        for k in r['gpu_index_api_calls']:r['gpu_index_api_calls'][k]+=2
        r['gpu_index_payload_bytes_sum']+=payload;r['gpu_index_payload_bytes_peak']=max(r['gpu_index_payload_bytes_peak'],payload)
        # Both final index tensors plus the currently built arange and offset vector.
        r['gpu_index_construction_live_upper_bound_bytes']=max(r['gpu_index_construction_live_upper_bound_bytes'],2*payload)
        r['rounds'].append(dict(base=c['base'],nodes=c['nodes'],accepted=list(c['accepted']),
            key_index_elements=ki.numel(),text_index_elements=vi.numel(),logical_index_bytes=payload))
        self.current=None

    def validate_result(self,result):
        n=result.forward_counts['target_verify_calls']
        if self.current is not None or any(self.record[k]!=n for k in
            ('captured_trees','bound_forwards','consumed_forwards','accepted_paths','committed_paths')):
            raise RuntimeError('incomplete host-plan coverage')
        actual=[r for r in result.rounds if r.get('kind')=='tree']
        if len(actual)!=n:raise RuntimeError('result round coverage mismatch')
        for r,a in zip(self.record['rounds'],actual):
            if (r['base']!=a['target_cache_before'] or r['nodes']!=a['tree_nodes']
                    or r['accepted']!=a['selected_tree_indices']):raise RuntimeError('host accepted receipt/result differs')
        self.record['result_coverage_validated']=True

    def release(self):self.current=None


class HostPlanPolicy:
    def __init__(self,state):self.state=state;self.record=state.record;self.broker=HostTreeBroker(state)
    def validate_result(self,result):
        self.state.validate_result(result);self.broker.validate_result(result)
        result.precision=dict(getattr(result,'precision',{}),target_v_bank_metadata_path=METHOD)
    def summary(self):
        info=self.state.summary();info['host_tree_reuse']=dict(self.broker.record)
        return info


@contextmanager
def v_bank_host_plan_policy(target,mode='int8_visual',*,interface=None,bridge=None,reference_ops=None):
    global _ACTIVE
    if _ACTIVE:raise RuntimeError('nested host-plan policy')
    _ACTIVE=True
    try:
        with _BASE_POLICY(target,mode,interface=interface,bridge=bridge,reference_ops=reference_ops) as state:
            policy=HostPlanPolicy(state);state._host_tree_broker=policy.broker
            old_forward,old_compact=target.forward,state.bridge._compact
            state.forward=MethodType(_forward,state);state.compact=MethodType(_compact,state)
            target.forward=state.forward;state.bridge._compact=state.compact
            try:yield policy
            finally:
                target.forward=old_forward;state.bridge._compact=old_compact
                policy.broker.release();del state._host_tree_broker
                del state.forward;del state.compact
    finally:_ACTIVE=False


def _forward(self,*args,**kwargs):
    if self.in_prefill:return frozen.lifecycle.Stride0AttentionPolicy.forward(self,*args,**kwargs)
    try:
        if self.failed or self._finished or self.pending is not None or self.in_forward:
            raise RuntimeError('request failed/finished or previous tree not committed')
        frozen.lifecycle._check_precision()
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
        plan=self._host_tree_broker.consume_forward(cache,ids,mask,pos,base,t)
        if plan.semantic_before!=base:raise RuntimeError('committed K length disagrees with partition')
        for item in self.stores:item.check(plan)
        self.record['round_guard_wall_s']+=time.perf_counter()-start
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


def _compact(self,cache,selected,base,tree_nodes):
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
        decision=self._host_tree_broker.consume_commit(cache,selected,base,tree_nodes)
        positions=decision.selected
        tbase=self.route.R+self.committed
        ki=torch.cat((torch.arange(base,device=self.device),selected+base))
        vi=torch.cat((torch.arange(tbase,device=self.device),selected+tbase))
        for item in self.stores:item.check(self.plan,active=True)
        for item in self.stores:
            k,v=item.commit(decision,ki,vi)
            self._io('key_compact',frozen._bytes(k),frozen._bytes(k));self._io('compact_value_compact',frozen._bytes(v),frozen._bytes(v))
        self.committed+=len(positions)
        if self.original_length(cache)!=self.route.P+self.committed:raise RuntimeError('K compact semantic length failed')
        self.record['rounds'][-1].update(selected_tree_indices=list(positions),cache_after_commit=base+len(positions),
            compact_value_after_commit=self.route.R+self.committed)
        self.record['commit_calls']+=1
        self._host_tree_broker.committed(ki,vi)
        self.plan=None;self.pending=None;self.update_counts=[]
    except BaseException as exc:
        self._fail('compact',exc);raise
    finally:self.record['commit_wall_s']+=time.perf_counter()-start
