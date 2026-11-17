"""CPU-only reference for full FP16 K plus partitioned visual V storage.

No HF/runtime hooks, CUDA path, eviction, capacity reservation or fallback.
from_prefill keeps the original K object but owns independent visual/text V
allocations; it only weakly references the original V tensor. Setup validates
finite inputs. All conversions/checks are real work to account for in a future
runtime. No throughput, allocator-peak, SRAM or actual-DRAM claim is made.

Owned banks and shared plans are immutable to callers. Mutating their contents,
even through an alias, invalidates the contract. Normal versioned tensors are
created even inside inference_mode, permitting metadata-only mutation guards.
fill_value_fp32_reference_ never reads tensor data on the host for validation;
it writes a supplied contiguous FP32 operand, using at most one 32-value FP32
conversion temporary. It does not retain that operand or restore full FP16 V.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import weakref

import torch
from ring_kv_cache import prepare_tree as _validate_tree

D, HKV, GROUP = 128, 4, 32
MODES = ('fp16_partition', 'int8_visual')


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name+' must be a host integer >= '+str(minimum))
    return value


def _size(tensor):
    return tensor.numel()*tensor.element_size()


def _storage_id(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


def _source(tensor, name, tokens=None):
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type!='cpu'
            or tensor.layout!=torch.strided or tensor.dtype!=torch.float16
            or tensor.ndim!=4 or tensor.shape[:2]!=(1,HKV) or tensor.shape[-1]!=D
            or tensor.requires_grad):
        raise ValueError(name+' requires CPU_reference_only FP16 [1,4,S,128], no gradients')
    if (tensor.stride(-1)!=1 or tensor.stride(-2)!=D
            or tensor.stride(1)<tensor.shape[2]*D):
        raise ValueError(name+' must have native nonoverlapping head/row/channel layout')
    if tokens is not None and tensor.shape[2]!=tokens:
        raise ValueError(name+' logical length differs')
    # Explicit CPU setup/append validation, never hidden GPU synchronization.
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(name+' contains nonfinite values')


def _pair(keys, values, tokens):
    _source(keys,'keys',tokens);_source(values,'values',tokens)
    if _storage_id(keys)==_storage_id(values):
        raise ValueError('K and V must have independent backing storage')


@dataclass(frozen=True, eq=False, init=False)
class VisualPartition:
    prefix_tokens: int
    visual: tuple[int, ...]
    nonvisual: tuple[int, ...]
    host_route: tuple[int, ...]
    prefix_route: torch.Tensor = field(repr=False)
    provenance_json: str
    _route_version: int = field(repr=False)

    def __init__(self,*args,**kwargs):raise TypeError('use VisualPartition.prepare')

    @classmethod
    def prepare(cls, prefix_tokens, visual_positions, *, provenance=None, device='cpu'):
        _integer(prefix_tokens,'prefix_tokens',1)
        if torch.device(device).type!='cpu':
            raise ValueError('CPU_reference_only; no GPU fallback')
        if isinstance(visual_positions,torch.Tensor):
            raise TypeError('visual_positions must be already prepared host integers')
        visual=tuple(visual_positions)
        if not visual or any(type(i) is not int for i in visual):
            raise ValueError('at least one real visual host index required')
        if len(set(visual))!=len(visual) or any(i<0 or i>=prefix_tokens for i in visual):
            raise ValueError('duplicate/out-of-range visual index')
        visual=tuple(sorted(visual));selected=set(visual)
        nonvisual=tuple(i for i in range(prefix_tokens) if i not in selected)
        route=[None]*prefix_tokens
        for row,pos in enumerate(visual):route[pos]=row
        for row,pos in enumerate(nonvisual):route[pos]=-row-1
        encoded=json.dumps(provenance or {},sort_keys=True,allow_nan=False)
        with torch.inference_mode(False):
            tensor=torch.tensor(route,dtype=torch.int32,device='cpu')
        result=object.__new__(cls)
        for name,value in dict(prefix_tokens=prefix_tokens,visual=visual,nonvisual=nonvisual,host_route=tuple(route),
                               prefix_route=tensor,provenance_json=encoded,_route_version=tensor._version).items():
            object.__setattr__(result,name,value)
        return result

    def validate_metadata(self):
        if self.prefix_route.dtype!=torch.int32 or self.prefix_route.device.type!='cpu' or \
                self.prefix_route.shape!=(self.prefix_tokens,) or self.prefix_route._version!=self._route_version:
            raise ValueError('shared prefix route changed')

    def receipt(self):
        self.validate_metadata()
        return dict(prefix_tokens=self.prefix_tokens,visual_tokens=len(self.visual),
            nonvisual_prompt_tokens=len(self.nonvisual),route_dtype='int32',
            shared_route_bytes=self.prefix_route.untyped_storage().nbytes(),
            route_sha256=hashlib.sha256(json.dumps(self.host_route).encode()).hexdigest(),
            provenance=json.loads(self.provenance_json),count_once_per_request=True)


@dataclass(frozen=True, eq=False, init=False)
class VTreePlan:
    partition: VisualPartition
    committed_tokens: int
    nodes: int
    ancestors: tuple
    ancestor_provenance: str

    def __init__(self,*args,**kwargs):raise TypeError('use prepare_tree')

    @property
    def semantic_before(self):return self.partition.prefix_tokens+self.committed_tokens


@dataclass(frozen=True, eq=False, init=False)
class VCommitPlan:
    tree_plan: VTreePlan
    selected: tuple[int, ...]
    key_indices: torch.Tensor = field(repr=False)
    text_indices: torch.Tensor = field(repr=False)
    _versions: tuple = field(repr=False)

    def __init__(self,*args,**kwargs):raise TypeError('use prepare_commit')

    def validate_metadata(self):
        if (self.key_indices.device.type!='cpu' or self.text_indices.device.type!='cpu'
                or self.key_indices.dtype!=torch.long or self.text_indices.dtype!=torch.long
                or (self.key_indices._version,self.text_indices._version)!=self._versions):
            raise ValueError('shared commit indices changed')


def prepare_tree(partition, committed_tokens, ancestor_mask):
    if not isinstance(partition,VisualPartition):raise TypeError('VisualPartition required')
    partition.validate_metadata();_integer(committed_tokens,'committed_tokens')
    if (not isinstance(ancestor_mask,torch.Tensor) or ancestor_mask.device.type!='cpu'
            or ancestor_mask.ndim not in (2,4) or ancestor_mask.shape[-1] not in range(1,31)):
        raise ValueError('CPU tree mask with 1..30 nodes required')
    validated=_validate_tree(ancestor_mask)  # small CPU setup once, never in fill
    provenance=hashlib.sha256(json.dumps(validated.ancestors).encode()).hexdigest()
    result=object.__new__(VTreePlan)
    for name,value in dict(partition=partition,committed_tokens=committed_tokens,nodes=validated.nodes,
                           ancestors=validated.ancestors,ancestor_provenance=provenance).items():
        object.__setattr__(result,name,value)
    return result


def prepare_commit(plan, selected):
    if not isinstance(plan,VTreePlan):raise TypeError('VTreePlan required')
    if isinstance(selected,torch.Tensor):raise TypeError('selected must be already validated host integers')
    positions=tuple(selected)
    if (not positions or any(type(i) is not int for i in positions) or positions[0]!=0
            or len(set(positions))!=len(positions) or any(i<0 or i>=plan.nodes for i in positions)):
        raise ValueError('selected must start at root with unique valid tree indices')
    path=set()
    for node in positions:
        path.add(node)
        if set(plan.ancestors[node])!=path:raise ValueError('selected must preserve a complete ancestor path')
    kbase=plan.semantic_before;tbase=len(plan.partition.nonvisual)+plan.committed_tokens
    with torch.inference_mode(False):
        ki=torch.tensor((*range(kbase),*(kbase+i for i in positions)),dtype=torch.long)
        vi=torch.tensor((*range(tbase),*(tbase+i for i in positions)),dtype=torch.long)
    result=object.__new__(VCommitPlan)
    for name,value in dict(tree_plan=plan,selected=positions,key_indices=ki,text_indices=vi,
                           _versions=(ki._version,vi._version)).items():
        object.__setattr__(result,name,value)
    return result


class VOnlyLayerStore:
    @classmethod
    @torch.no_grad()
    def from_prefill(cls, keys, values, partition, *, mode='fp16_partition'):
        if not isinstance(partition,VisualPartition):raise TypeError('VisualPartition required')
        if mode not in MODES:raise ValueError('unsupported V storage mode')
        partition.validate_metadata();_pair(keys,values,partition.prefix_tokens)
        self=cls();self.partition=partition;self.mode=mode;self.keys16=keys
        self.committed_tokens=0;self.active_plan=None
        self._source_value_ref=weakref.ref(values)
        self._source_value_storage=_storage_id(values)
        self._source_value_backing_bytes=values.untyped_storage().nbytes()
        with torch.inference_mode(False):
            visual_index=torch.tensor(partition.visual,dtype=torch.long)
            text_index=torch.tensor(partition.nonvisual,dtype=torch.long)
            self.text_values16=values.index_select(2,text_index)
            if mode=='fp16_partition':
                self.visual_values=values.index_select(2,visual_index);self.scales=None
            else:
                # CPU reference packs one group at a time, without a full visual
                # FP16/FP32 gather. No scale is inferred across different tokens.
                self.visual_values=torch.empty((1,HKV,len(partition.visual),D),dtype=torch.int8)
                self.scales=torch.empty((1,HKV,len(partition.visual),D//GROUP),dtype=torch.float32)
                for row,column in enumerate(partition.visual):
                    for h in range(HKV):
                        for g in range(D//GROUP):
                            sl=slice(g*GROUP,(g+1)*GROUP)
                            group=values[0,h,column,sl].float()
                            maximum=group.abs().amax()
                            scale=torch.where(maximum==0,torch.ones_like(maximum),maximum/127.)
                            code=torch.round(group/scale).clamp_(-127,127).to(torch.int8)
                            self.visual_values[0,h,row,sl].copy_(code)
                            self.scales[0,h,row,g].copy_(scale)
        self._bank_versions=(self.visual_values._version,None if self.scales is None else self.scales._version)
        self.validate_quantization_reference()
        if self._source_value_storage in {_storage_id(x) for x in self._owned()}:
            raise RuntimeError('old full V backing retained by store')
        visual_elements=HKV*len(partition.visual)*D;text_elements=HKV*len(partition.nonvisual)*D
        self.counts=dict(prefill=1,append_tree=0,commit=0,fill=0)
        self.logical_io={};self.logical_io_totals=dict(read_bytes=0,write_bytes=0,total_bytes=0)
        self._record('prefill_partition',2*(visual_elements+text_elements),
            _size(self.visual_values)+(0 if self.scales is None else _size(self.scales))+2*text_elements,
            scope='payload movement only; finite scans, route/index metadata and quantization temporaries excluded')
        return self

    def _owned(self):
        return [x for x in (self.keys16,self.visual_values,self.scales,self.text_values16) if x is not None]

    def _record(self,operation,read,write,**extra):
        self.logical_io[operation]=dict(read_bytes=read,write_bytes=write,total_bytes=read+write,**extra)
        self.logical_io_totals['read_bytes']+=read
        self.logical_io_totals['write_bytes']+=write
        self.logical_io_totals['total_bytes']+=read+write

    def _check_banks(self):
        self.partition.validate_metadata()
        versions=(self.visual_values._version,None if self.scales is None else self.scales._version)
        if versions!=self._bank_versions:raise ValueError('immutable visual values/scales changed')
        expected=self.partition.prefix_tokens+self.committed_tokens+(self.active_plan.nodes if self.active_plan else 0)
        if self.keys16.shape!=(1,HKV,expected,D) or self.text_values16.shape!=(1,HKV,expected-len(self.partition.visual),D):
            raise ValueError('K/text-V semantic lengths disagree')

    def validate_quantization_reference(self):
        """Explicit CPU diagnostic; never called by fill or used as a GPU fallback."""
        if self.scales is not None:
            if (self.scales.device.type!='cpu' or self.scales.dtype!=torch.float32
                    or self.scales.shape!=(1,HKV,len(self.partition.visual),D//GROUP)
                    or not bool(torch.isfinite(self.scales).all()) or not bool((self.scales>0).all())):
                raise ValueError('quantization scales must be finite positive FP32')

    @torch.no_grad()
    def append_tree(self, keys, values, plan):
        self._check_banks()
        if (not isinstance(plan,VTreePlan) or plan.partition is not self.partition
                or plan.committed_tokens!=self.committed_tokens or self.active_plan is not None):
            raise ValueError('wrong semantic-before, partition, or duplicate pending tree')
        _pair(keys,values,plan.nodes)
        # Both allocations finish before installing either new tensor.
        k=torch.cat((self.keys16,keys),dim=2)
        v=torch.cat((self.text_values16,values),dim=2)
        self.keys16,self.text_values16,self.active_plan=k,v,plan
        self.counts['append_tree']+=1
        payload=_size(k)+_size(v)
        self._record('append_tree',payload,payload,tree_nodes=plan.nodes,
                     scope='K + compact-V cat payload; input finite-validation scans excluded')
        return k,v

    @torch.no_grad()
    def commit(self, decision):
        self._check_banks()
        if not isinstance(decision,VCommitPlan) or decision.tree_plan is not self.active_plan or self.active_plan is None:
            raise ValueError('commit belongs to a different or already committed tree')
        decision.validate_metadata()
        k=self.keys16.index_select(2,decision.key_indices)
        v=self.text_values16.index_select(2,decision.text_indices)
        self.keys16,self.text_values16=k,v
        self.committed_tokens+=len(decision.selected);self.active_plan=None;self.counts['commit']+=1
        payload=_size(k)+_size(v)
        self._record('commit',payload,payload,accepted_tokens=len(decision.selected),
                     scope='K + compact-V index_select payload; selected order retained')
        return k,v

    @torch.no_grad()
    def fill_value_fp32_reference_(self, out, plan=None):
        """CPU-only fill; caller owns/reclaims out. No host tensor-value checks."""
        self._check_banks()
        if plan is not self.active_plan:raise ValueError('fill requires exact current tree plan, or None without tree')
        if (not isinstance(out,torch.Tensor) or out.device.type!='cpu' or out.dtype!=torch.float32
                or out.shape!=self.keys16.shape or not out.is_contiguous() or out.requires_grad):
            raise ValueError('CPU_reference_only contiguous FP32 [1,4,S,128] output required')
        if _storage_id(out) in {_storage_id(x) for x in self._owned()} or _storage_id(out)==_storage_id(self.partition.prefix_route):
            raise ValueError('operand must not alias store or route storage')
        p=self.partition.prefix_tokens;r=len(self.partition.nonvisual)
        for column in range(out.shape[2]):
            route=self.partition.host_route[column] if column<p else -(r+column-p)-1
            for h in range(HKV):
                if route<0:
                    out[0,h,column].copy_(self.text_values16[0,h,-route-1])
                elif self.scales is None:
                    out[0,h,column].copy_(self.visual_values[0,h,route])
                else:
                    for g in range(D//GROUP):
                        sl=slice(g*GROUP,(g+1)*GROUP)
                        temporary=self.visual_values[0,h,route,sl].float()
                        temporary.mul_(self.scales[0,h,route,g])
                        out[0,h,column,sl].copy_(temporary)
                        del temporary
        self.counts['fill']+=1
        read=_size(self.visual_values)+(0 if self.scales is None else _size(self.scales))+_size(self.text_values16)
        self._record('fill',read,_size(out),
            reference_int8_scratch_logical_read_plus_write_bytes=(16*self.visual_values.numel() if self.scales is not None else 0),
            reference_max_explicit_fp32_conversion_temporary_bytes=GROUP*4 if self.scales is not None else 0,
            scope='ideal source-once + operand write payload; INT8 reference scratch traffic is separate, no actual DRAM')
        return out

    def storage_receipt(self):
        self._check_banks()
        entries={};unique={}
        for name in ('keys16','visual_values','scales','text_values16'):
            tensor=getattr(self,name)
            if tensor is None:continue
            storage=tensor.untyped_storage();identity=_storage_id(tensor)
            if storage.nbytes():unique[identity]=storage.nbytes()
            entries[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),stride=list(tensor.stride()),
                logical_bytes=_size(tensor),backing_bytes=storage.nbytes(),storage_pointer=storage.data_ptr())
        return dict(mode=self.mode,CPU_reference_only=True,no_GPU_fallback=True,
            logical_tokens=self.keys16.shape[2],committed_tokens=self.committed_tokens,
            active_tree_nodes=self.active_plan.nodes if self.active_plan else 0,
            banks=entries,unique_store_backing_bytes=sum(unique.values()),shared_route=self.partition.receipt(),
            shared_route_included_in_store_total=False,counts=dict(self.counts),last_logical_io=dict(self.logical_io),
            ideal_payload_io_totals=dict(self.logical_io_totals),
            original_value_tensor_alive=self._source_value_ref() is not None,
            original_value_storage_excluded_at_install=True,
            known_store_retains_original_value_backing=(self._source_value_ref() is not None and self._source_value_storage in unique),
            source_pointer_reuse_possible_after_release=True,
            original_value_backing_bytes=self._source_value_backing_bytes,
            full_fp16_value_reconstruction=False,FP32_operand_retained=False,
            actual_DRAM_measured=False,allocator_reserved_measured=False,
            scope='known store roots only; external caller aliases may keep old V alive; shared route count once per request')
