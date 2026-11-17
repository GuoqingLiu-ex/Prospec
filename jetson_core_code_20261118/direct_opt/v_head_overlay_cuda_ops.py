"""Independent full-V8 plus original-FP16 head overlay primitives.

No model integration, automatic selector policy, CPU execution fallback, or GPU
evidence. Only prepare performs one explicit value readback. Caller keeps each
actual original V alive until its copy completes; no 28-layer V stack is made.
Raw CUDA writes do not increment Tensor versions: exclusive ownership, stream
ordering, same-source provenance and final status validation remain mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import weakref

import torch
import v_bank_cuda_ops as base

LAYERS, HEADS, D, GROUP, TILE = 28, 4, 128, 32, 8
UNITS = LAYERS * HEADS
SCHEMA = 'physical_visual_v_head_overlay_v1'
COPY_DYNAMIC = ('L', 'K', 'P', 'M', 'R', 'SH', 'SN', 'SD')
FILL_DYNAMIC = ('L', 'K') + base.FILL_DYNAMIC
_KERNELS = None


def _budget(k):
    if type(k) is not int or not 0 <= k <= UNITS:
        raise ValueError('explicit budget_heads integer in [0,112] required')


def _layer(layer):
    if type(layer) is not int or not 0 <= layer < LAYERS:
        raise ValueError('layer_idx must be integer in [0,28)')


def _version(x):
    try:
        return x._version
    except RuntimeError as exc:
        raise ValueError('allocate immutable overlay tensors under inference_mode(False)') from exc


def _source_snapshot(x):
    """Native prefill may produce inference tensors; never clone to add versioning.

    Metadata can detect replacement/layout changes, not source data mutation.
    Versioned source tensors retain their additional ordinary-mutation guard.
    """
    try:
        version=x._version
    except RuntimeError:
        if not torch.is_inference(x): raise
        version=None
    return dict(identity=id(x),data_ptr=x.data_ptr(),storage=base._sid(x),
        shape=tuple(x.shape),stride=tuple(x.stride()),offset=x.storage_offset(),
        dtype=str(x.dtype),device=str(x.device),version=version,
        inference_tensor=torch.is_inference(x))


def capacity_receipt(M, budget_heads=14):
    base._positive(M, 'M'); _budget(budget_heads)
    codes = UNITS * M * D
    scales = UNITS * M * (D // GROUP) * 4
    values = budget_heads * M * D * 2
    original = UNITS * M * D * 2
    return dict(schema=SCHEMA, visual_tokens=M, budget_heads=budget_heads,
        original_visual_FP16_bytes=original, complete_V8_codes_bytes=codes,
        complete_V8_scales_bytes=scales, original_FP16_overlay_bytes=values,
        head_route_bytes=UNITS*4, extra_persistent_error_flag_bytes=0,
        persistent_visual_payload_bytes=codes+scales+values+UNITS*4,
        saving_vs_full_visual_FP16_bytes=original-codes-scales-values-UNITS*4,
        duplicated_selected_V8_payload_bytes=budget_heads*M*(D+D//GROUP*4),
        caller_score_slab_temporary_bytes=UNITS*4,
        full_argsort_indices_temporary_bytes=UNITS*8,
        selected_indices_is_view_of_full_sort=True,
        retained_original_visual_payload_until_global_selection_bytes=original,
        source_copy_read_bytes=values, overlay_copy_write_bytes=values,
        measured_conversion_peak=False, actual_DRAM_measured=False,
        exclusions=('unchanged full K/text/prefix route/existing bank flags',
            'allocator rounding/reserve', 'sort/route/check workspaces and CUDA backend temporaries'),
        scope='complete V8 base retained; no replacement-format mixed-precision assumption')


def copy_launch_plan(P, M, budget_heads=14):
    base.pack_launch_plan(P, M, 'int8_visual'); _budget(budget_heads)
    return dict(grid=(HEADS, math.ceil(P/TILE)), num_warps=4, num_stages=1,
        constexpr={'BLOCK': TILE}, runtime_scalars=COPY_DYNAMIC,
        custom_copy_launches_per_layer=1, scheduled_layers=LAYERS,
        launch_scope='copy custom kernel only, excludes sort/route/flags/status',
        all_layers_selected_source_read_bytes=budget_heads*M*D*2,
        all_layers_overlay_write_bytes=budget_heads*M*D*2,
        unselected_head_source_V_read=False, copies_original_FP16_bits=True,
        original_full_V_stacked=False, actual_GPU_compilation_verified=False)


def fill_launch_plan(P, M, S, protected_heads):
    base.fill_launch_plan(P, M, S, 'int8_visual')
    if type(protected_heads) is not int or not 0 <= protected_heads <= HEADS:
        raise ValueError('explicit known protected_heads in [0,4] required for logical accounting')
    return dict(grid=(HEADS, math.ceil(S/TILE)), custom_fill_launches=1,
        runtime_scalars=FILL_DYNAMIC, constexpr={'BLOCK': TILE},
        protected_original_FP16_read_bytes=protected_heads*M*D*2,
        unprotected_V8_read_bytes=(HEADS-protected_heads)*M*D,
        unprotected_scale_unique_read_bytes=(HEADS-protected_heads)*M*(D//GROUP)*4,
        text_read_bytes=HEADS*(S-M)*D*2, operand_write_bytes=HEADS*S*D*4,
        protected_head_V8_or_scale_load=False,
        scope='ideal source-once payload, not actual DRAM/scale-lane loads/route reads/kernel instructions')


def validate_selection_metadata(scores, route, budget_heads=14):
    _budget(budget_heads)
    if not isinstance(route, base.PartitionRoute):
        raise TypeError('frozen shared PartitionRoute required')
    route.validate_metadata()
    base._tensor_meta(scores, 'head scores', (LAYERS, HEADS), torch.float32,
                      route.prefix_route.device, True)
    if base._sid(scores) == base._sid(route.prefix_route):
        raise ValueError('score slab aliases prompt route')
    return dict(version=_version(scores), data_ptr=scores.data_ptr(),
        storage=base._sid(scores), shape=list(scores.shape),
        backing_bytes=scores.untyped_storage().nbytes(), budget_heads=budget_heads)


@dataclass(eq=False, init=False)
class CudaHeadOverlay:
    route: base.PartitionRoute
    budget_heads: int
    values: torch.Tensor = field(repr=False)
    head_route: torch.Tensor = field(repr=False)
    _versions: tuple
    _bank_refs: list = field(repr=False)
    validated: bool
    failed: bool
    preparation: dict

    def __init__(self, *args, **kwargs):
        raise TypeError('use prepare_head_overlay')

    def validate_metadata(self):
        self.route.validate_metadata(); _budget(self.budget_heads)
        device = self.route.prefix_route.device
        base._tensor_meta(self.values, 'original FP16 overlay',
            (self.budget_heads, self.route.M, D), torch.float16, device, True)
        base._tensor_meta(self.head_route, 'head route', (LAYERS, HEADS), torch.int32, device, True)
        if (_version(self.values), _version(self.head_route)) != self._versions:
            raise ValueError('immutable overlay values/head route changed')
        roots = [self.head_route, self.route.prefix_route]
        if self.values.numel(): roots.append(self.values)
        if len({base._sid(x) for x in roots}) != len(roots):
            raise ValueError('overlay/route backing alias')
        if len(self._bank_refs) != LAYERS:
            raise ValueError('invalid layer completion metadata')

    def storage_receipt(self):
        self.validate_metadata()
        roots = [self.head_route] + ([self.values] if self.values.numel() else [])
        live_banks=[ref() if ref is not None else None for ref in self._bank_refs]
        live_base=[x for bank in live_banks if bank is not None
                   for x in (bank.visual_values,bank.scales)]
        base_storage={base._sid(x):x.untyped_storage().nbytes() for x in live_base}
        all_storage=dict(base_storage)
        all_storage.update({base._sid(x):x.untyped_storage().nbytes() for x in roots})
        return dict(**capacity_receipt(self.route.M, self.budget_heads),
            actual_overlay_route_unique_backing_bytes=sum(
                x.untyped_storage().nbytes() for x in roots),
            caller_live_layer_banks=len([b for b in live_banks if b is not None]),
            actual_caller_live_V8_base_unique_backing_bytes=sum(base_storage.values()),
            actual_live_visual_base_overlay_route_unique_backing_bytes=sum(all_storage.values()),
            actual_backing_scope='currently live caller-owned V8 codes/scales plus owned overlay/head route; '
                'entire storage counted including view offsets; existing flags/prefix route excluded',
            completed_layer_copies=sum(x is not None for x in self._bank_refs),
            owns_original_source=False, owns_score_or_sort_buffers=False,
            owns_FP32_operand=False, validated=self.validated, failed=self.failed,
            raw_CUDA_mutations_not_detectable_by_tensor_version=True,
            source_history_is_caller_contract=True, preparation=self.preparation)


@torch.no_grad()
def prepare_head_overlay(scores, route, *, budget_heads=14):
    """One GPU stable sort plus one explicit score/slot value-check boundary.

    The supplied score slab must already have passed the contributing packs'
    persistent flag validation. This function cannot authenticate that history.
    It keeps neither scores nor the full argsort index storage after returning.
    """
    snapshot = validate_selection_metadata(scores, route, budget_heads)
    base._require_cuda(scores)
    with torch.inference_mode(False):
        order = torch.argsort(scores.flatten(), descending=True, stable=True)
        selected = order[:budget_heads]
        head_route = torch.full((UNITS,), -1, dtype=torch.int32, device=scores.device)
        slots = torch.arange(budget_heads, dtype=torch.int32, device=scores.device)
        head_route.scatter_(0, selected, slots)
        # argsort is a permutation; gather==slots additionally checks distinct
        # selected slots. Unselected entries retain the initialized sentinel.
        finite_nonnegative = torch.isfinite(scores).all() & (scores >= 0).all()
        bijective_slots = (head_route.gather(0, selected) == slots).all()
        if not bool((finite_nonnegative & bijective_slots).item()):
            raise ValueError('invalid score values or non-bijective selected slots')
        values = torch.empty((budget_heads, route.M, D), dtype=torch.float16, device=scores.device)
        head_route = head_route.view(LAYERS, HEADS)
    if validate_selection_metadata(scores, route, budget_heads) != snapshot:
        raise ValueError('scores changed while preparing stable selection')
    if any(x.numel() and base._sid(x) in (base._sid(scores), base._sid(route.prefix_route))
           for x in (values, head_route)):
        raise RuntimeError('allocated overlay/head route aliases input')
    overlay = base._construct(CudaHeadOverlay, route=route, budget_heads=budget_heads,
        values=values, head_route=head_route, _versions=(_version(values), _version(head_route)),
        _bank_refs=[None]*LAYERS, validated=False, failed=False,
        preparation=dict(stable_descending_GPU_argsort_calls=1,
            tie_break='ascending layer*4+KV_head', explicit_host_value_checks=1,
            check_scope='finite nonnegative scores and generated slot bijection',
            score_pack_status_is_caller_precondition=True,
            operations=('argsort', 'full route sentinel', 'arange slots', 'scatter_',
                'isfinite/ge/all', 'gather/eq/all', 'item', 'empty overlay'),
            kernel_launch_count_measured=False, score_slab_retained=False,
            inference_source_layer_count=0,versioned_source_layer_count=0,
            inference_source_guard='identity/layout only; caller guarantees original source provenance, '
                'exclusive ownership and same-stream no-mutation until copy completes; no source clone',
            actual_GPU_numerics_verified=False))
    overlay.validate_metadata()
    return overlay


def _bank_contract(overlay, bank, layer_idx):
    if not isinstance(overlay, CudaHeadOverlay): raise TypeError('CudaHeadOverlay required')
    overlay.validate_metadata(); _layer(layer_idx)
    if overlay.failed: raise RuntimeError('overlay failed; no reuse')
    if not isinstance(bank, base.CudaVisualVBank): raise TypeError('frozen V8 bank required')
    bank.validate_metadata()
    if bank.mode != 'int8_visual' or bank.route is not overlay.route:
        raise ValueError('same immutable shared route and int8_visual bank required')
    if bank.failed or not bank.validated: raise RuntimeError('original V8 pack status must pass first')
    roots = [bank.visual_values, bank.scales, bank.error_flag]
    for x in (overlay.values, overlay.head_route):
        if x.numel() and any(base._sid(x) == base._sid(y) for y in roots):
            raise ValueError('overlay storage aliases base/scale/flag')


def _get_kernels():
    global _KERNELS
    if _KERNELS is not None: return _KERNELS
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError('Triton overlay primitives unavailable; no fallback') from exc

    @triton.jit(do_not_specialize=COPY_DYNAMIC)
    def _copy(SRC, PREFIX, HEAD_ROUTE, OVERLAY, ERROR, L, K, P, M, R, SH, SN, SD,
              BLOCK: tl.constexpr):
        h=tl.program_id(0)
        slot=tl.load(HEAD_ROUTE+L*4+h).to(tl.int64)
        if (slot < -1) | (slot >= K): tl.atomic_or(ERROR,16)
        if (slot >= 0) & (slot < K):
            j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
            d=tl.arange(0,128)
            inside=j<P
            ri=tl.load(PREFIX+j,mask=inside,other=-1).to(tl.int64)
            visual=inside&(ri>=0)&(ri<M)
            text=inside&(ri<0)&(-ri-1<R)
            if tl.sum((inside&(~(visual|text))).to(tl.int32),0)>0: tl.atomic_or(ERROR,16)
            src=tl.load(SRC+h.to(tl.int64)*SH+j[:,None].to(tl.int64)*SN+d[None,:]*SD,
                        mask=visual[:,None],other=0.)
            fp=src.to(tl.float32)
            bad=visual[:,None]&((fp!=fp)|(tl.abs(fp)==float('inf')))
            if tl.sum(tl.sum(bad.to(tl.int32),1),0)>0: tl.atomic_or(ERROR,16)
            # Store the original FP16 tensor, not fp or a dequantized value.
            tl.store(OVERLAY+(slot*M+ri[:,None])*128+d[None,:],src,mask=visual[:,None])

    @triton.jit(do_not_specialize=FILL_DYNAMIC)
    def _fill(VIS,SCALES,TEXT,PREFIX,HEAD_ROUTE,OVERLAY,ERROR,OUT,
              L,K,P,M,R,S,NF,TH,TN,TD,BLOCK:tl.constexpr):
        h=tl.program_id(0)
        j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
        d=tl.arange(0,128)
        inside=j<S; prefix=inside&(j<P)
        ri=tl.load(PREFIX+j,mask=prefix,other=-1).to(tl.int64)
        visual=prefix&(ri>=0)&(ri<M)
        ti=tl.where(prefix,-ri-1,R+j-P).to(tl.int64)
        text=(prefix&(ri<0)&(ti>=0)&(ti<R))|(inside&(~prefix)&(ti>=R)&(ti<NF))
        slot=tl.load(HEAD_ROUTE+L*4+h).to(tl.int64)
        slot_ok=(slot>=-1)&(slot<K)
        # Scalar CTA-uniform branch: protected heads have no V8/scale loads.
        if (slot>=0)&(slot<K):
            vis=tl.load(OVERLAY+(slot*M+ri[:,None])*128+d[None,:],
                mask=visual[:,None],other=0.).to(tl.float32)
            scale_ok=tl.full((BLOCK,128),True,tl.int1)
        else:
            vis=tl.load(VIS+(h.to(tl.int64)*M+ri[:,None])*128+d[None,:],
                mask=visual[:,None]&(slot==-1),other=0).to(tl.float32)
            scales=tl.load(SCALES+(h.to(tl.int64)*M+ri[:,None])*4+d[None,:]//32,
                mask=visual[:,None]&(slot==-1),other=1.)
            scale_ok=(scales==scales)&(tl.abs(scales)!=float('inf'))&(scales>0)
            vis=vis*scales
        tv=tl.load(TEXT+h.to(tl.int64)*TH+ti[:,None]*TN+d[None,:]*TD,
            mask=text[:,None],other=0.).to(tl.float32)
        value=tl.where(visual[:,None],vis,tv)
        finite=(value==value)&(tl.abs(value)!=float('inf'))
        bad=inside[:,None]&((~finite)|(~scale_ok)|(~slot_ok)|(~(visual|text))[:,None])
        if tl.sum(tl.sum(bad.to(tl.int32),1),0)>0: tl.atomic_or(ERROR,16)
        prior=tl.load(ERROR)
        value=tl.where(bad|(prior!=0),float('nan'),value)
        tl.store(OUT+(h.to(tl.int64)*S+j[:,None])*128+d[None,:],value,mask=inside[:,None])

    _KERNELS=(_copy,_fill)
    return _KERNELS


@torch.no_grad()
def copy_head_overlay_layer_(overlay, layer_idx, original_full_V, bank):
    _bank_contract(overlay, bank, layer_idx)
    if overlay.validated or overlay._bank_refs[layer_idx] is not None:
        raise RuntimeError('each layer copy exactly once before overlay ready')
    if any(ref is not None and ref() is bank for ref in overlay._bank_refs):
        raise ValueError('same bank cannot supply two layers')
    base.validate_pack_metadata(original_full_V, overlay.route, 'int8_visual')
    roots=[overlay.values,overlay.head_route,bank.visual_values,bank.scales,bank.error_flag]
    if any(x.numel() and base._sid(original_full_V)==base._sid(x) for x in roots):
        raise ValueError('actual original V aliases destination/base/flags')
    base._require_cuda(original_full_V)
    before=_source_snapshot(original_full_V)
    copy,_=_get_kernels(); plan=copy_launch_plan(overlay.route.P,overlay.route.M,overlay.budget_heads)
    try:
        copy[plan['grid']](original_full_V,overlay.route.prefix_route,overlay.head_route,overlay.values,
            bank.error_flag,layer_idx,overlay.budget_heads,overlay.route.P,overlay.route.M,overlay.route.R,
            *original_full_V.stride()[1:],BLOCK=TILE,num_warps=4,num_stages=1,enable_fp_fusion=False)
        if _source_snapshot(original_full_V)!=before: raise ValueError('source modified during layer copy')
        overlay.validate_metadata()
        overlay._bank_refs[layer_idx]=weakref.ref(bank)
        counter='inference_source_layer_count' if before['inference_tensor'] else 'versioned_source_layer_count'
        overlay.preparation[counter]+=1
    except BaseException:
        overlay.failed=True;overlay.validated=False;bank.failed=True;bank.validated=False
        raise
    return overlay


def _validate_banks(overlay,banks):
    overlay.validate_metadata()
    if overlay.failed: raise RuntimeError('overlay failed; no reuse')
    if len(banks)!=LAYERS or any(r is None for r in overlay._bank_refs):
        raise RuntimeError('all 28 layer copies required')
    if len({id(b) for b in banks})!=LAYERS: raise ValueError('28 distinct layer banks required')
    owned=[x for bank in banks for x in (bank.visual_values,bank.scales,bank.error_flag)]
    if len({base._sid(x) for x in owned})!=len(owned):
        raise ValueError('layer base/scale/error backings must be independent')
    for layer,bank in enumerate(banks):
        _bank_contract(overlay,bank,layer)
        if overlay._bank_refs[layer]() is not bank:
            raise ValueError('bank identity/order differs from layer copy')


def check_overlay_status(overlay,banks,*,stage):
    """Mandatory explicit collective boundary; never called by hot fill."""
    banks=tuple(banks);_validate_banks(overlay,banks)
    try:
        return base.check_status(banks,stage=stage)
    except BaseException:
        overlay.failed=True;overlay.validated=False
        raise


def validate_overlay_ready(overlay,banks):
    if overlay.validated: raise RuntimeError('overlay setup already finalized')
    receipt=check_overlay_status(overlay,banks,stage='after_original_head_copy_before_fill')
    overlay.validated=True
    return receipt


@torch.no_grad()
def fill_overlay_value_fp32_(out,bank,text,overlay,layer_idx):
    _bank_contract(overlay,bank,layer_idx)
    if not overlay.validated or overlay._bank_refs[layer_idx]() is not bank:
        raise RuntimeError('matching copied bank and finalized overlay required')
    S=base.validate_fill_metadata(out,bank,text)
    if any(x.numel() and base._sid(out)==base._sid(x) for x in (overlay.values,overlay.head_route)):
        raise ValueError('caller operand aliases overlay/head route')
    if text.numel() and any(x.numel() and base._sid(text)==base._sid(x)
                           for x in (overlay.values,overlay.head_route)):
        raise ValueError('text aliases immutable overlay/head route')
    base._require_cuda(out); _,fill=_get_kernels()
    try:
        fill[(HEADS,math.ceil(S/TILE))](bank.visual_values,bank.scales,text,bank.route.prefix_route,
            overlay.head_route,overlay.values,bank.error_flag,out,layer_idx,overlay.budget_heads,
            bank.route.P,bank.route.M,bank.route.R,S,text.shape[2],*text.stride()[1:],
            BLOCK=TILE,num_warps=4,num_stages=1,enable_fp_fusion=False)
    except BaseException:
        overlay.failed=True;overlay.validated=False;bank.failed=True;bank.validated=False
        raise
    return out


@torch.no_grad()
def overlay_value_fp32_reference(codes,scales,text,partition,head_route,overlay,layer_idx):
    """Explicit allocating CPU oracle; never selected by a CUDA wrapper."""
    _layer(layer_idx);partition.validate_metadata()
    M=len(partition.visual);R=len(partition.nonvisual);P=partition.prefix_tokens
    k=overlay.shape[0];_budget(k)
    device=torch.device('cpu')
    base._tensor_meta(codes,'CPU codes',(1,HEADS,M,D),torch.int8,device,True)
    base._tensor_meta(scales,'CPU scales',(1,HEADS,M,4),torch.float32,device,True)
    base._tensor_meta(head_route,'CPU head route',(LAYERS,HEADS),torch.int32,device,True)
    base._tensor_meta(overlay,'CPU overlay',(k,M,D),torch.float16,device,True)
    nf=text.shape[2] if isinstance(text,torch.Tensor) and text.ndim==4 else -1
    base._tensor_meta(text,'CPU text',(1,HEADS,nf,D),torch.float16,device)
    if nf<R: raise ValueError('CPU text shorter than prompt')
    rv=head_route.flatten().tolist()
    if any(s < -1 or s>=k for s in rv) or sorted(s for s in rv if s>=0)!=list(range(k)):
        raise ValueError('CPU head route not bijective')
    if not bool(torch.isfinite(text).all()) or not bool(torch.isfinite(overlay).all()):
        raise ValueError('nonfinite CPU source')
    S=P+nf-R;out=torch.empty((1,HEADS,S,D),dtype=torch.float32)
    for h in range(HEADS):
        slot=int(head_route[layer_idx,h])
        if slot>=0:
            visual=overlay[slot].float()
        else:
            if not bool(torch.isfinite(scales[0,h]).all()) or bool((scales[0,h]<=0).any()):
                raise ValueError('invalid unprotected scales')
            visual=(codes[0,h].float().reshape(M,4,32)*scales[0,h,...,None]).reshape(M,D)
        out[0,h,list(partition.visual)]=visual
        out[0,h,list(partition.nonvisual)]=text[0,h,:R].float()
        out[0,h,P:]=text[0,h,R:].float()
    return out
