"""Compact mixed visual V storage; independent of the frozen per-layer bank.

Setup consumes validated full V8 banks plus original FP16 V, with no new
quantization. Global signed routes partition all 112 heads. Views retain only
the global owner and layer index. Ready releases all old-bank weak references;
only the original 28 persistent error flags survive. No CUDA/CPU fallback.
Raw CUDA writes do not advance tensor versions: exclusive ownership and stream
ordering remain caller obligations. Ordinary metadata checks do not scan values.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import weakref
import torch
import v_bank_cuda_ops as base
from v_head_overlay_cuda_ops import _budget, _layer, _version, _source_snapshot, validate_selection_metadata

LAYERS, HEADS, D, TILE, UNITS = 28, 4, 128, 8, 112
SCHEMA = 'compact_mixed_visual_v_heads_v1'
COPY_DYNAMIC = ('L','K','P','M','R','SH','SN','SD')
FILL_DYNAMIC = ('L','K','P','M','R','S','NF','TH','TN','TD','OH','ON','OD')
_KERNELS = None


def capacity_receipt(M, budget_heads=14):
    base._positive(M,'M'); _budget(budget_heads)
    codes=(UNITS-budget_heads)*M*D; scales=(UNITS-budget_heads)*M*16
    original=budget_heads*M*D*2; full=UNITS*M*D*2
    return dict(schema=SCHEMA,visual_tokens=M,budget_heads=budget_heads,
        compact_codes_bytes=codes,compact_scales_bytes=scales,original_FP16_bytes=original,
        head_route_bytes=448,persistent_visual_payload_bytes=codes+scales+original+448,
        saving_vs_full_visual_FP16_bytes=full-codes-scales-original-448,
        saving_vs_full_V8_plus_overlay_bytes=budget_heads*M*144,
        old_code_scale_compaction_read_bytes=codes+scales,
        new_code_scale_compaction_write_bytes=codes+scales,
        original_FP16_copy_read_bytes=original,original_FP16_copy_write_bytes=original,
        unchanged_existing_error_flag_bytes=112,extra_persistent_error_flag_bytes=0,
        scope='logical visual payload plus signed head route; flags/shared prefix route separate',
        conversion_peak_measured=False,actual_DRAM_measured=False,
        steady_fill_payload_reduction_vs_full_V8_overlay=False)


def _roots(owner):
    return [x for x in (owner.codes,owner.scales,owner.original_fp16,owner.head_route) if x.numel()]


def _unique_bytes(tensors):
    return sum({base._sid(x):x.untyped_storage().nbytes() for x in tensors if x.numel()}.values())


def _poison(owner):
    owner.failed=True; owner.validated=False


@dataclass(eq=False,init=False)
class CompactHeadBank:
    route: base.PartitionRoute
    budget_heads: int
    codes: torch.Tensor=field(repr=False)
    scales: torch.Tensor=field(repr=False)
    original_fp16: torch.Tensor=field(repr=False)
    head_route: torch.Tensor=field(repr=False)
    error_flags: list=field(repr=False)
    _versions: tuple
    _bank_refs: list=field(repr=False)
    _completed: list
    validated: bool
    failed: bool
    preparation: dict

    def __init__(self,*args,**kwargs):raise TypeError('use prepare_compact_head_bank')

    def validate_metadata(self):
        self.route.validate_metadata();_budget(self.budget_heads)
        device=self.route.prefix_route.device;M=self.route.M;K=self.budget_heads
        for tensor,name,shape,dtype in (
            (self.codes,'compact codes',(UNITS-K,M,D),torch.int8),
            (self.scales,'compact scales',(UNITS-K,M,4),torch.float32),
            (self.original_fp16,'original FP16',(K,M,D),torch.float16),
            (self.head_route,'signed head route',(LAYERS,HEADS),torch.int32)):
            base._tensor_meta(tensor,name,shape,dtype,device,True)
        if tuple(_version(x) for x in (self.codes,self.scales,self.original_fp16,self.head_route))!=self._versions:
            raise ValueError('immutable compact payload or route changed')
        if len(self.error_flags)!=LAYERS or len(self._completed)!=LAYERS or len(self._bank_refs)!=LAYERS:
            raise ValueError('invalid layer metadata')
        flags=[]
        for i,f in enumerate(self.error_flags):
            if f is not None:
                base._tensor_meta(f,'persistent flag',(),torch.int32,device,True);flags.append(f)
            if self._completed[i]!=(f is not None):raise ValueError('completion/flag mismatch')
        roots=_roots(self)+[self.route.prefix_route]+flags
        if len({base._sid(x) for x in roots})!=len(roots):raise ValueError('compact payload/route/flag backing alias')
        if self.validated and not all(self._completed):raise ValueError('ready requires 28 layers')

    def layer_view(self,layer):
        _layer(layer);self.validate_metadata()
        if self.failed or not self.validated:raise RuntimeError('ready compact owner required')
        return CompactLayerView(self,layer)

    def storage_receipt(self):
        self.validate_metadata();flags=[f for f in self.error_flags if f is not None]
        payload=_unique_bytes(_roots(self));flagbytes=_unique_bytes(flags)
        return dict(**capacity_receipt(self.route.M,self.budget_heads),
            actual_visual_payload_route_unique_backing_bytes=payload,
            actual_existing_error_flags_unique_backing_bytes=flagbytes,
            actual_owned_unique_backing_bytes=_unique_bytes(_roots(self)+flags),
            shared_prefix_route_unique_backing_bytes=base._bytes(self.route.prefix_route),
            completed_layer_copies=sum(self._completed),validated=self.validated,failed=self.failed,
            old_bank_weak_refs_present=sum(r is not None for r in self._bank_refs),
            owns_old_banks=False,owns_original_sources=False,owns_scores_or_sort=False,
            actual_backing_scope='whole unique backing including offsets; global payload counted once',
            preparation=dict(self.preparation))


@dataclass(eq=False,frozen=True)
class CompactLayerView:
    owner: CompactHeadBank=field(repr=False)
    layer_idx: int
    @property
    def route(self):return self.owner.route
    @property
    def mode(self):return 'int8_visual'
    @property
    def visual_values(self):return self.owner.codes
    @property
    def scales(self):return self.owner.scales
    @property
    def error_flag(self):return self.owner.error_flags[self.layer_idx]
    @property
    def validated(self):return self.owner.validated
    @property
    def failed(self):return self.owner.failed
    def validate_metadata(self):
        _layer(self.layer_idx);self.owner.validate_metadata()
        if self.error_flag is None:raise ValueError('layer not copied')
    def storage_receipt(self):
        return dict(schema=SCHEMA,layer_idx=self.layer_idx,global_owner_storage_counted_here=False,
            global_owner_receipt_required=True)


@torch.no_grad()
def prepare_compact_head_bank(scores,route,*,budget_heads=14):
    before=validate_selection_metadata(scores,route,budget_heads);base._require_cuda(scores)
    K=budget_heads
    with torch.inference_mode(False):
        order=torch.argsort(scores.flatten(),descending=True,stable=True)
        hi=order[:K];lo=order[K:]
        hs=torch.arange(K,dtype=torch.int32,device=scores.device)
        ls=-(torch.arange(UNITS-K,dtype=torch.int32,device=scores.device)+1)
        route_heads=torch.empty(UNITS,dtype=torch.int32,device=scores.device)
        route_heads.scatter_(0,hi,hs);route_heads.scatter_(0,lo,ls)
        valid=torch.isfinite(scores).all()&(scores>=0).all()
        # Unique slot IDs checked at every one of 112 sorted positions prove
        # both subsets disjoint and bijective; no second sort is needed.
        valid=valid&(route_heads.gather(0,hi)==hs).all()&(route_heads.gather(0,lo)==ls).all()
        if not bool(valid.item()):raise ValueError('invalid scores or non-bijective signed head route')
        codes=torch.empty((UNITS-K,route.M,D),dtype=torch.int8,device=scores.device)
        scales=torch.empty((UNITS-K,route.M,4),dtype=torch.float32,device=scores.device)
        original=torch.empty((K,route.M,D),dtype=torch.float16,device=scores.device)
        route_heads=route_heads.view(LAYERS,HEADS)
    if validate_selection_metadata(scores,route,K)!=before:raise ValueError('scores changed during selection')
    owner=base._construct(CompactHeadBank,route=route,budget_heads=K,codes=codes,scales=scales,
        original_fp16=original,head_route=route_heads,
        _versions=tuple(_version(x) for x in (codes,scales,original,route_heads)),
        error_flags=[None]*LAYERS,_bank_refs=[None]*LAYERS,_completed=[False]*LAYERS,
        validated=False,failed=False,preparation=dict(stable_GPU_argsort_calls=1,
            route_scatter_calls=2,explicit_host_value_checks=1,
            setup_permutation_check_sort_calls=0,
            tie_break='ascending layer*4+KV_head',custom_conversion_launches=0,
            inference_source_layer_count=0,versioned_source_layer_count=0,
            no_requantization=True,original_sources_stacked=False,
            host_check_scope='finite nonnegative scores and both subset permutation/bijection',
            cost_scope='sort/scatter/validation/allocations are additional, not free; no measured kernel count'))
    owner.validate_metadata()
    if any(base._sid(x)==base._sid(scores) for x in _roots(owner)):raise ValueError('allocated bank aliases scores')
    return owner


def _get_kernels():
    global _KERNELS
    if _KERNELS is not None:return _KERNELS
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:raise RuntimeError('Triton compact bank unavailable; no fallback') from exc

    @triton.jit(do_not_specialize=COPY_DYNAMIC)
    def _copy(SRC,OLD,OLD_SCALE,PREFIX,HEAD_ROUTE,CODES,SCALES,FP16,ERROR,
              L,K,P,M,R,SH,SN,SD,BLOCK:tl.constexpr):
        h=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
        d=tl.arange(0,128);g=tl.arange(0,4)
        inside=j<P;ri=tl.load(PREFIX+j,mask=inside,other=-1).to(tl.int64)
        visual=inside&(ri>=0)&(ri<M)
        ti=-ri-1;text=inside&(ri<0)&(ti>=0)&(ti<R)
        slot=tl.load(HEAD_ROUTE+L*4+h).to(tl.int64)
        low=-slot-1;hi_ok=(slot>=0)&(slot<K);lo_ok=(slot<0)&(low>=0)&(low<112-K)
        if tl.sum((inside&(~(visual|text))).to(tl.int32),0)>0:tl.atomic_or(ERROR,32)
        if ~(hi_ok|lo_ok):tl.atomic_or(ERROR,32)
        if hi_ok:
            raw=tl.load(SRC+h.to(tl.int64)*SH+j[:,None].to(tl.int64)*SN+d[None,:]*SD,mask=visual[:,None],other=0.)
            value=raw.to(tl.float32)
            bad=visual[:,None]&((value!=value)|(tl.abs(value)==float('inf')))
            if tl.sum(tl.sum(bad.to(tl.int32),1),0)>0:tl.atomic_or(ERROR,32)
            tl.store(FP16+(slot*M+ri[:,None])*128+d[None,:],raw,mask=visual[:,None])
        elif lo_ok:
            code=tl.load(OLD+(h.to(tl.int64)*M+ri[:,None])*128+d[None,:],mask=visual[:,None],other=0)
            scale=tl.load(OLD_SCALE+(h.to(tl.int64)*M+ri[:,None])*4+g[None,:],mask=visual[:,None],other=1.)
            badc=visual[:,None]&(code==-128)
            bads=visual[:,None]&((scale!=scale)|(tl.abs(scale)==float('inf'))|(scale<=0))
            if (tl.sum(tl.sum(badc.to(tl.int32),1),0)+tl.sum(tl.sum(bads.to(tl.int32),1),0))>0:tl.atomic_or(ERROR,32)
            tl.store(CODES+(low*M+ri[:,None])*128+d[None,:],code,mask=visual[:,None])
            tl.store(SCALES+(low*M+ri[:,None])*4+g[None,:],scale,mask=visual[:,None])

    @triton.jit(do_not_specialize=FILL_DYNAMIC)
    def _fill(CODES,SCALES,FP16,TEXT,PREFIX,HEAD_ROUTE,ERROR,OUT,
              L,K,P,M,R,S,NF,TH,TN,TD,OH,ON,OD,BLOCK:tl.constexpr):
        h=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK);d=tl.arange(0,128)
        inside=j<S;prefix=inside&(j<P)
        ri=tl.load(PREFIX+j,mask=prefix,other=-1).to(tl.int64)
        visual=prefix&(ri>=0)&(ri<M)
        ti=tl.where(prefix,-ri-1,R+j-P).to(tl.int64)
        text=(prefix&(ri<0)&(ti>=0)&(ti<R))|(inside&(~prefix)&(ti>=R)&(ti<NF))
        slot=tl.load(HEAD_ROUTE+L*4+h).to(tl.int64);low=-slot-1
        hi_ok=(slot>=0)&(slot<K);lo_ok=(slot<0)&(low>=0)&(low<112-K)
        valid=tl.full((BLOCK,128),True,tl.int1)
        if hi_ok:
            value=tl.load(FP16+(slot*M+ri[:,None])*128+d[None,:],mask=visual[:,None],other=0.).to(tl.float32)
        else:
            code=tl.load(CODES+(low*M+ri[:,None])*128+d[None,:],mask=visual[:,None]&lo_ok,other=0)
            scale=tl.load(SCALES+(low*M+ri[:,None])*4+d[None,:]//32,mask=visual[:,None]&lo_ok,other=1.)
            valid=(code!=-128)&(scale==scale)&(tl.abs(scale)!=float('inf'))&(scale>0)
            value=code.to(tl.float32)*scale
        tv=tl.load(TEXT+h.to(tl.int64)*TH+ti[:,None]*TN+d[None,:]*TD,mask=text[:,None],other=0.).to(tl.float32)
        value=tl.where(visual[:,None],value,tv)
        bad=inside[:,None]&((value!=value)|(tl.abs(value)==float('inf'))|(~valid)|(~(hi_ok|lo_ok))|(~(visual|text))[:,None])
        if tl.sum(tl.sum(bad.to(tl.int32),1),0)>0:tl.atomic_or(ERROR,32)
        prior=tl.load(ERROR);value=tl.where(bad|(prior!=0),float('nan'),value)
        tl.store(OUT+h.to(tl.int64)*OH+j[:,None].to(tl.int64)*ON+d[None,:]*OD,value,mask=inside[:,None])
    _KERNELS=(_copy,_fill)
    return _KERNELS


@torch.no_grad()
def copy_compact_head_layer_(owner,layer,source,oldbank):
    owner.validate_metadata();_layer(layer)
    if owner.failed or owner.validated or owner._completed[layer]:raise RuntimeError('copy each layer once before ready')
    if not isinstance(oldbank,base.CudaVisualVBank):raise TypeError('frozen validated V8 bank required')
    oldbank.validate_metadata()
    if oldbank.route is not owner.route or oldbank.mode!='int8_visual' or not oldbank.validated or oldbank.failed:
        raise ValueError('same-route validated V8 source bank required')
    if any(r is not None and r() is oldbank for r in owner._bank_refs):raise ValueError('bank reused across layers')
    base.validate_pack_metadata(source,owner.route,'int8_visual')
    incoming=[oldbank.visual_values,oldbank.scales,oldbank.error_flag]
    roots=_roots(owner)+[owner.route.prefix_route]+[f for f in owner.error_flags if f is not None]
    if any(base._sid(x)==base._sid(y) for x in incoming for y in roots):raise ValueError('old bank aliases owner or another flag')
    if any(base._sid(source)==base._sid(x) for x in incoming+roots):raise ValueError('original source aliases bank')
    base._require_cuda(source);before=_source_snapshot(source);copy,_=_get_kernels()
    try:
        copy[(HEADS,math.ceil(owner.route.P/TILE))](source,oldbank.visual_values,oldbank.scales,
            owner.route.prefix_route,owner.head_route,owner.codes,owner.scales,owner.original_fp16,oldbank.error_flag,
            layer,owner.budget_heads,owner.route.P,owner.route.M,owner.route.R,*source.stride()[1:],
            BLOCK=TILE,num_warps=4,num_stages=1,enable_fp_fusion=False)
        if _source_snapshot(source)!=before:raise ValueError('source modified during copy')
        owner.validate_metadata();owner._bank_refs[layer]=weakref.ref(oldbank)
        owner.error_flags[layer]=oldbank.error_flag;owner._completed[layer]=True
        owner.preparation['custom_conversion_launches']+=1
        owner.preparation['inference_source_layer_count' if before['inference_tensor'] else 'versioned_source_layer_count']+=1
    except BaseException:
        _poison(owner);oldbank.failed=True;oldbank.validated=False;raise
    return owner


def _flags_status(owner,stage):
    try:
        bad=torch.stack(owner.error_flags).ne(0).any()
        if bool(bad.item()):raise RuntimeError('persistent compact bank device error at '+str(stage))
    except BaseException:_poison(owner);raise
    return dict(stage=stage,layer_count=LAYERS,aggregate_host_reads=1,all_flags_zero=True,flags_reset=False)


def validate_compact_ready(owner,banks):
    owner.validate_metadata();banks=tuple(banks)
    if owner.failed or owner.validated:raise RuntimeError('fresh healthy compact setup required')
    if len(banks)!=LAYERS or not all(owner._completed):raise RuntimeError('all 28 copies required')
    if len({id(b) for b in banks})!=LAYERS:raise ValueError('distinct ordered banks required')
    roots=[]
    for l,b in enumerate(banks):
        b.validate_metadata()
        if owner._bank_refs[l]() is not b or owner.error_flags[l] is not b.error_flag:
            raise ValueError('bank/flag identity or order mismatch')
        if b.route is not owner.route or b.mode!='int8_visual' or b.failed or not b.validated:raise ValueError('old pack no longer valid')
        roots.extend((b.visual_values,b.scales,b.error_flag))
    if len({base._sid(x) for x in roots})!=len(roots):raise ValueError('old layer backings alias')
    receipt=_flags_status(owner,'after_compaction_before_fill')
    owner._bank_refs=[None]*LAYERS;owner.validated=True
    return receipt


def check_compact_status(owner,views=None,*,stage):
    owner.validate_metadata()
    if owner.failed or not owner.validated:raise RuntimeError('healthy ready compact bank required')
    if views is not None:
        views=tuple(views)
        if len(views)!=LAYERS or any(type(v) is not CompactLayerView or v.owner is not owner or v.layer_idx!=i for i,v in enumerate(views)):
            raise ValueError('28 same-owner ordered views required')
    return _flags_status(owner,stage)


def _row_layout(x,name,shape,dtype,device):
    base._tensor_meta(x,name,shape,dtype,device)
    if x.stride(-1)!=1 or x.stride(-2)<D or x.stride(1)<max(shape[2],1)*x.stride(-2):
        raise ValueError(name+' requires positive disjoint row/head strides')


def validate_fill_metadata(out,view,text,owner=None,layer_idx=None):
    if type(view) is not CompactLayerView:raise TypeError('independent CompactLayerView required')
    view.validate_metadata();o=view.owner
    if owner is not None and owner is not o:raise ValueError('wrong owner')
    if layer_idx is not None and layer_idx!=view.layer_idx:raise ValueError('wrong layer')
    if o.failed or not o.validated:raise RuntimeError('healthy ready owner required')
    if not isinstance(out,torch.Tensor) or out.ndim!=4:raise ValueError('FP32 operand shape required')
    S=out.shape[2];base._positive(S,'S')
    if S<o.route.P:raise ValueError('operand shorter than complete prefix')
    NF=o.route.R+S-o.route.P;device=o.route.prefix_route.device
    _row_layout(out,'out',(1,HEADS,S,D),torch.float32,device)
    _row_layout(text,'text',(1,HEADS,NF,D),torch.float16,device)
    roots=_roots(o)+[o.route.prefix_route]+o.error_flags
    if any(base._sid(x)==base._sid(y) for x in (out,text) if x.numel() for y in roots):raise ValueError('text/out aliases immutable bank')
    if text.numel() and base._sid(out)==base._sid(text):raise ValueError('text aliases output')
    return S


@torch.no_grad()
def fill_compact_value_fp32_(out,view,text,owner=None,layer_idx=None):
    S=validate_fill_metadata(out,view,text,owner,layer_idx);o=view.owner;base._require_cuda(out)
    _,fill=_get_kernels()
    try:
        fill[(HEADS,math.ceil(S/TILE))](o.codes,o.scales,o.original_fp16,text,o.route.prefix_route,
            o.head_route,view.error_flag,out,view.layer_idx,o.budget_heads,o.route.P,o.route.M,o.route.R,
            S,text.shape[2],*text.stride()[1:],*out.stride()[1:],BLOCK=TILE,num_warps=4,num_stages=1,enable_fp_fusion=False)
    except BaseException:_poison(o);raise
    return out


@torch.no_grad()
def compact_value_fp32_reference(codes,scales,original_fp16,text,partition,head_route,layer_idx):
    """Allocating CPU-only oracle. Not a production fallback or model policy."""
    _layer(layer_idx);partition.validate_metadata();M=len(partition.visual);K=original_fp16.shape[0];_budget(K)
    device=torch.device('cpu')
    for x,n,shape,dtype in ((codes,'codes',(UNITS-K,M,D),torch.int8),(scales,'scales',(UNITS-K,M,4),torch.float32),
                          (original_fp16,'fp16',(K,M,D),torch.float16),(head_route,'route',(28,4),torch.int32)):
        base._tensor_meta(x,n,shape,dtype,device,True)
    rv=head_route.flatten().tolist()
    if sorted(x for x in rv if x>=0)!=list(range(K)) or sorted(-int(x)-1 for x in rv if x<0)!=list(range(UNITS-K)):
        raise ValueError('signed route is not a full two-subset bijection')
    R=len(partition.nonvisual);P=partition.prefix_tokens
    NF=text.shape[2] if isinstance(text,torch.Tensor) and text.ndim==4 else -1
    base._tensor_meta(text,'text',(1,4,NF,D),torch.float16,device)
    if NF<R or not torch.isfinite(text).all():raise ValueError('invalid CPU text')
    out=torch.empty((1,4,P+NF-R,D),dtype=torch.float32)
    for h in range(4):
        slot=int(head_route[layer_idx,h])
        if slot>=0:
            value=original_fp16[slot].float()
        else:
            q=-slot-1;s=scales[q]
            if not torch.isfinite(s).all() or (s<=0).any() or (codes[q]==-128).any():raise ValueError('invalid quantized payload')
            value=(codes[q].float().reshape(M,4,32)*s[...,None]).reshape(M,D)
        if not torch.isfinite(value).all():raise ValueError('nonfinite value')
        out[0,h,list(partition.visual)]=value
        out[0,h,list(partition.nonvisual)]=text[0,h,:R].float()
        out[0,h,P:]=text[0,h,R:].float()
    return out
