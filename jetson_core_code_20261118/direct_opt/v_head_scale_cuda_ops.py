"""Independent INT8 pack accumulating four zero-aware FP32 scale-risk scores.

The caller zero-initializes one versioned [28,4] request slab once, passes a
contiguous layer slice [4], and keeps it live through stream completion. Pack
never clears it: atomic_max accumulates across all visual rows and any repeated
calls. A valid finite prior maximum is supported; metadata cannot prove a fresh
zero initialization. Scores are only acceptable after frozen aggregate status.

Original code/scale/text quantization and fill contracts are unchanged. One
source-row load feeds both quantization and the proxy. No source/code rescan,
CPU readback, new bank ownership of the score buffer, selector or model hook.
This is a scale proxy, not actual reconstruction error or a QA guarantee.
"""
from __future__ import annotations

import torch
import v_bank_cuda_ops as base

SCHEMA = 'visual_v_pack_head_scale_risk_v1'
MODE = 'int8_visual'
PACK_DYNAMIC = base.PACK_DYNAMIC
_KERNELS = None

prepare_partition_route = base.prepare_partition_route
validate_pack_status = base.validate_pack_status
check_status = base.check_status
fill_value_fp32_ = base.fill_value_fp32_


def head_score_metadata(scores):
    """No value read; support a [4] slice of the single caller-owned slab."""
    if (not isinstance(scores, torch.Tensor) or scores.layout != torch.strided
            or tuple(scores.shape) != (base.HEADS,) or scores.dtype != torch.float32
            or scores.device.type not in ('cpu','cuda') or scores.requires_grad
            or not scores.is_contiguous()):
        raise ValueError('head scores require contiguous nongrad FP32 [4] on CPU metadata or CUDA')
    try:
        version = scores._version
    except RuntimeError as exc:
        raise ValueError('allocate head score slab under inference_mode(False) for version tracking') from exc
    return dict(schema=SCHEMA, shape=list(scores.shape), stride=list(scores.stride()),
        dtype=str(scores.dtype), device=str(scores.device), version=version,
        data_ptr=scores.data_ptr(), storage_data_ptr=scores.untyped_storage().data_ptr(),
        storage_offset=scores.storage_offset(), payload_bytes=4*base.HEADS,
        backing_bytes=scores.untyped_storage().nbytes(),
        backing_scope='caller request slab; count backing once, not once per layer',
        initial_values_read_or_zero_verified=False)


def validate_head_score_metadata(scores, snapshot):
    if head_score_metadata(scores) != snapshot:
        raise ValueError('score identity/layout/version differs from recorded snapshot')


def pack_head_scale_launch_plan(P, M, mode=MODE):
    if mode != MODE:
        raise ValueError('head-scale pack is int8_visual only; use frozen pack for FP16')
    original = base.pack_launch_plan(P, M, MODE)
    visual_rows = base.HEADS*M
    return dict(original, schema=SCHEMA, mode=MODE, constexpr={},
        original_bank_text_write_bytes=original['output_payload_write_bytes'],
        head_score_payload_bytes=4*base.HEADS, request_28_layer_score_slab_bytes=28*4*base.HEADS,
        score_initialization_performed_by_pack=False, caller_zero_initialization_once_per_request=True,
        initialization_requires_stream_order_before_pack=True,
        score_semantics='atomic_max accumulation; each valid visual/head row contributes its max valid scale/2',
        score_proxy_rule='max(where(original group maxabs>0, FP32 scale, 0))/2',
        extra_atomic_max_operations=visual_rows, atomic_operand_bytes=4,
        naive_atomic_read_plus_write_byte_model=visual_rows*8,
        atomic_byte_model_scope='one logical FP32 read-modify-write per visual/head row; not actual transfers or DRAM',
        additional_score_read_for_guard=False,
        invalid_prior_score_flag_bit=8,
        invalid_prior_score_check='returned atomic old value must be finite and nonnegative; finite nonzero priors are valid accumulations',
        score_source_or_code_reread=False, score_host_readback_performed=False,
        score_d2h_payload_if_collected_bytes=4*base.HEADS,
        strict_error_bound=False, semantic_or_task_risk=False,
        logical_extra_per_visual_head_row=dict(group_zero_masks=4, masked_scale_values=4,
            max_reduction_input_elements=4, FP32_half_scaling=1, FP32_atomic_max=1),
        kernel_launch_scope='one replacement custom_pack_with_head_scale_kernel_only',
        extra_custom_kernel_launches_vs_frozen_pack=0, actual_GPU_numerics_verified=False,
        additional_costs_unmeasured=original['additional_costs_unmeasured']+(
            'caller request score slab allocation/one zero initialization and its stream ordering',
            '4-scale masking/reduction and FP32 atomic_max contention among M CTAs per head',
            'returned atomic-value guard, register work, compiler instructions and occupancy',
            'caller aggregate score D2H/synchronization or device ranking; neither is performed here'),
        scope='logical plan only, not actual GPU kernel count, DRAM, latency, QA or deployed selector')


def validate_pack_head_scale_metadata(values, route, scores, *, mode=MODE):
    if mode != MODE:
        raise ValueError('head-scale pack only supports int8_visual')
    base.validate_pack_metadata(values,route,MODE)
    snapshot = head_score_metadata(scores)
    if scores.device != values.device:
        raise ValueError('head scores must share the source device')
    if base._sid(scores) in (base._sid(values),base._sid(route.prefix_route)):
        raise ValueError('head scores alias source V or shared route backing')
    return dict(pack_head_scale_launch_plan(route.P,route.M,mode),head_scores=snapshot)


def _get_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice
    except ImportError as exc:
        raise RuntimeError('Triton head-scale pack unavailable; no fallback') from exc

    @triton.jit(do_not_specialize=PACK_DYNAMIC)
    def _pack_head_scale(SRC,ROUTE,VIS,SCALES,TEXT,ERROR,HEAD_SCORES,P,M,R,SH,SN,SD):
        row=tl.program_id(0)
        j=row%P;h=row//P
        ri=tl.load(ROUTE+j).to(tl.int64)
        visual=(ri>=0)&(ri<M);text=(ri<0)&(-ri-1<R)
        valid_route=visual|text
        d=tl.arange(0,128)
        source=tl.load(SRC+h.to(tl.int64)*SH+j.to(tl.int64)*SN+d*SD)
        x=source.to(tl.float32)
        finite=(x==x)&(tl.abs(x)!=float('inf'))
        if not valid_route:tl.atomic_or(ERROR,1)
        if tl.sum((~finite).to(tl.int32),0)>0:tl.atomic_or(ERROR,2)
        if visual:
            safe=tl.where(finite,x,0.)
            grouped=tl.reshape(safe,(4,32))
            maximum=tl.max(tl.abs(grouped),1)
            scale=tl.where(maximum==0,1.,tl.div_rn(maximum,127.))
            codes=libdevice.nearbyint(tl.div_rn(grouped,scale[:,None]))
            codes=tl.minimum(tl.maximum(codes,-127.),127.).to(tl.int8)
            tl.store(VIS+(h.to(tl.int64)*M+ri)*128+d,tl.reshape(codes,(128,)))
            tl.store(SCALES+(h.to(tl.int64)*M+ri)*4+tl.arange(0,4),scale)
            risk=tl.max(tl.where(maximum>0,scale,0.),0)*0.5
            old=tl.atomic_max(HEAD_SCORES+h,risk)
            old_ok=(old==old)&(tl.abs(old)!=float('inf'))&(old>=0)
            if not old_ok:tl.atomic_or(ERROR,8)
        if text:
            tl.store(TEXT+(h.to(tl.int64)*R-ri-1)*128+d,source)

    _KERNELS = (_pack_head_scale,)
    return _KERNELS


def _require_cuda(values):
    base._require_cuda(values)


@torch.no_grad()
def pack_v_with_head_scale(values, route, head_scores_layer, *, mode=MODE):
    """Return frozen bank,text and atomically accumulate caller's score slice.

    No score initialization or host value read occurs here. Caller owns source
    and scores through stream completion, validates all pack flags together,
    then may read/rank scores. Invalid source/route/prior score rejects the
    entire result even if the final scores look finite. Failure never clears a
    flag. Raw CUDA writes are not tracked by PyTorch _version: ownership and
    stream ordering remain required; snapshots only catch ordinary mutation.
    """
    plan=validate_pack_head_scale_metadata(values,route,head_scores_layer,mode=mode)
    _require_cuda(values)
    (pack,)=_get_kernels()
    with torch.inference_mode(False):
        visual=torch.empty((1,base.HEADS,route.M,base.D),device=values.device,dtype=torch.int8)
        scales=torch.empty((1,base.HEADS,route.M,base.D//base.GROUP),device=values.device,dtype=torch.float32)
        text=torch.empty((1,base.HEADS,route.R,base.D),device=values.device,dtype=torch.float16)
        flag=torch.zeros((),device=values.device,dtype=torch.int32)
    outputs=[visual,scales,text,flag]
    forbidden={base._sid(values),base._sid(route.prefix_route),base._sid(head_scores_layer)}
    if any(x.numel() and base._sid(x) in forbidden for x in outputs):
        raise RuntimeError('head-scale pack outputs alias source/route/score output')
    nonempty=[x for x in outputs if x.numel()]
    if len({base._sid(x) for x in nonempty})!=len(nonempty):
        raise RuntimeError('head-scale pack output backings alias')
    pack[plan['grid']](values,route.prefix_route,visual,scales,text,flag,head_scores_layer,
        route.P,route.M,route.R,*values.stride()[1:],num_warps=4,num_stages=1,enable_fp_fusion=False)
    bank=base._construct(base.CudaVisualVBank,route=route,mode=MODE,visual_values=visual,scales=scales,
        error_flag=flag,_versions=(visual._version,scales._version),validated=False,failed=False)
    bank.validate_metadata()
    base._tensor_meta(text,'compact prompt text V',(1,base.HEADS,route.R,base.D),torch.float16,values.device,True)
    validate_head_score_metadata(head_scores_layer,plan['head_scores'])
    return bank,text


def head_scale_risk_fp32_reference(visual_values, *, initial_scores=None):
    """Allocating pure CPU oracle over compact visual rows; not a fallback.

    Quantization uses the frozen FP32 div/RNE oracle. Group maxabs supplies the
    zero sentinel mask. Optional finite/nonnegative priors verify max-accumulation;
    neither inputs nor priors are modified. No actual-error or QA bound is used.
    """
    from visual_v_quant_risk_reference import quantize_visual_v_reference
    packed=quantize_visual_v_reference(visual_values)
    maximum=visual_values.float().reshape(1,base.HEADS,visual_values.shape[2],4,32).abs().amax(-1)
    scores=torch.where(maximum>0,packed.scales,torch.zeros_like(packed.scales)).amax(dim=(-2,-1))[0]*0.5
    if initial_scores is not None:
        meta=head_score_metadata(initial_scores)
        if meta['device']!='cpu' or not bool(torch.isfinite(initial_scores).all()) or bool((initial_scores<0).any()):
            raise ValueError('CPU prior scores must be finite and nonnegative')
        scores=torch.maximum(scores,initial_scores)
    return scores.contiguous()
