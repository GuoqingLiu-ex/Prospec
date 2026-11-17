"""Independent INT8 V pack with one FP32 reconstruction-error scalar per row.

A diagnostic extension only: fixed group32 quantization, no selector/model hooks.
The source V is loaded once per native-row CTA; code*scale, subtraction, abs and
max reuse those registers. Caller owns contiguous versioned error_out [4,M].
Existing bank, fill and explicit aggregate status contracts are reused unchanged.
No CUDA/JIT initialization at import, CPU fallback, full reconstructed V output,
or source/error-array ownership is added to the bank. GPU compilation/numerics
remain to be validated separately. FP32 rounded e is not a strict FP64 bound.
"""
from __future__ import annotations

import torch
import v_bank_cuda_ops as base

SCHEMA = 'visual_v_pack_fp32_max_error_v1'
MODE = 'int8_visual'
PACK_DYNAMIC = base.PACK_DYNAMIC
_KERNELS = None

# Reuse the exact frozen lifecycle/consumer APIs, not replacement kernels.
prepare_partition_route = base.prepare_partition_route
validate_pack_status = base.validate_pack_status
check_status = base.check_status
fill_value_fp32_ = base.fill_value_fp32_


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError as exc:
        raise ValueError('error output must be allocated under inference_mode(False) for version tracking') from exc


def error_output_metadata(error_out):
    """Value-free snapshot; caller keeps this buffer alive and exclusively owned."""
    if (not isinstance(error_out, torch.Tensor) or error_out.layout != torch.strided
            or error_out.ndim != 2 or error_out.shape[0] != base.HEADS or error_out.shape[1] < 1
            or error_out.dtype != torch.float32 or error_out.device.type not in ('cpu','cuda')
            or error_out.requires_grad or not error_out.is_contiguous()):
        raise ValueError('error output requires contiguous nongrad FP32 [4,M>=1] on CPU metadata or CUDA')
    return dict(schema=SCHEMA, shape=list(error_out.shape), stride=list(error_out.stride()),
        dtype=str(error_out.dtype), device=str(error_out.device), version=_version(error_out),
        data_ptr=error_out.data_ptr(), storage_data_ptr=error_out.untyped_storage().data_ptr(),
        storage_offset=error_out.storage_offset(), payload_bytes=error_out.numel()*4,
        backing_bytes=error_out.untyped_storage().nbytes(),
        backing_scope='caller-owned; a layer slice may share a larger all-layer diagnostic backing; do not sum backing per layer')


def validate_error_output_metadata(error_out, snapshot):
    if error_output_metadata(error_out) != snapshot:
        raise ValueError('error output identity/layout/version differs from recorded snapshot')


def pack_error_launch_plan(P, M, mode=MODE):
    if mode != MODE:
        raise ValueError('error diagnostic is fixed int8_visual; use frozen pack for FP16 control')
    original = base.pack_launch_plan(P, M, MODE)
    diagnostic = 4*base.HEADS*M
    return dict(original, schema=SCHEMA, mode=MODE, constexpr={},
        output_payload_write_bytes=original['output_payload_write_bytes']+diagnostic,
        original_bank_text_write_bytes=original['output_payload_write_bytes'],
        diagnostic_output_write_bytes=diagnostic, diagnostic_payload_bytes=diagnostic,
        diagnostic_error_flag_logical_read_bytes=4*base.HEADS*M,
        diagnostic_error_flag_read_scope='one prior-flag load per visual/head CTA; logical repeated reads, not DRAM',
        diagnostic_d2h_payload_if_collected_bytes=diagnostic, diagnostic_d2h_performed=False,
        source_reread_for_error=False, full_reconstructed_value_output=False,
        diagnostic_math='independent FP32 code*scale, FP32 subtraction, abs, max128; FP fusion disabled',
        strict_error_bound=False, source_zero_sign_preserved_by_int8=False,
        logical_extra_per_visual_head_row=dict(int8_to_fp32_elements=128,multiply_elements=128,
            subtract_elements=128,abs_elements=128,max_reduction_input_elements=128),
        kernel_launch_scope='one replacement custom_pack_with_error_kernel_only',
        extra_custom_kernel_launches_vs_frozen_pack=0, actual_GPU_numerics_verified=False,
        additional_costs_unmeasured=original['additional_costs_unmeasured']+(
            'caller diagnostic allocation/backing and metadata guards',
            '128-element reconstruction/error/reduction register work; compiler instructions, spills and occupancy',
            'optional caller aggregate error D2H and synchronization; not performed by pack'),
        scope='logical payload and declared arithmetic only; not measured kernels, DRAM, performance or task risk')


def validate_pack_error_metadata(values, route, error_out, *, mode=MODE):
    if mode != MODE:
        raise ValueError('error diagnostic only supports int8_visual')
    base.validate_pack_metadata(values, route, MODE)
    snapshot = error_output_metadata(error_out)
    if error_out.shape != (base.HEADS,route.M) or error_out.device != values.device:
        raise ValueError('error output shape/device must match this layer visual route')
    if base._sid(error_out) in (base._sid(values),base._sid(route.prefix_route)):
        raise ValueError('error output aliases source V or shared route backing')
    return dict(pack_error_launch_plan(route.P,route.M,mode),error_output=snapshot)


def _get_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice
    except ImportError as exc:
        raise RuntimeError('Triton diagnostic pack unavailable; no fallback') from exc

    @triton.jit(do_not_specialize=PACK_DYNAMIC)
    def _pack_error(SRC,ROUTE,VIS,SCALES,TEXT,ERROR,EOUT,P,M,R,SH,SN,SD):
        row=tl.program_id(0)
        j=row%P;h=row//P
        ri=tl.load(ROUTE+j).to(tl.int64)
        visual=(ri>=0)&(ri<M);text=(ri<0)&(-ri-1<R)
        valid_route=visual|text
        d=tl.arange(0,128)
        source=tl.load(SRC+h.to(tl.int64)*SH+j.to(tl.int64)*SN+d*SD)
        x=source.to(tl.float32)
        finite=(x==x)&(tl.abs(x)!=float('inf'))
        bad_source=tl.sum((~finite).to(tl.int32),0)>0
        if not valid_route:tl.atomic_or(ERROR,1)
        if bad_source:tl.atomic_or(ERROR,2)
        if visual:
            safe=tl.where(finite,x,0.)
            grouped=tl.reshape(safe,(4,32))
            maximum=tl.max(tl.abs(grouped),1)
            scale=tl.where(maximum==0,1.,tl.div_rn(maximum,127.))
            codes=libdevice.nearbyint(tl.div_rn(grouped,scale[:,None]))
            codes=tl.minimum(tl.maximum(codes,-127.),127.).to(tl.int8)
            tl.store(VIS+(h.to(tl.int64)*M+ri)*128+d,tl.reshape(codes,(128,)))
            tl.store(SCALES+(h.to(tl.int64)*M+ri)*4+tl.arange(0,4),scale)
            # Independent FP32 instructions; launcher disables fusion. No SRC reload.
            represented=codes.to(tl.float32)*scale[:,None]
            delta=tl.reshape(represented,(128,))-x
            e=tl.max(tl.abs(delta),0)
            prior=tl.load(ERROR)
            e=tl.where(bad_source|(prior!=0),float('nan'),e)
            tl.store(EOUT+h.to(tl.int64)*M+ri,e)
        if text:
            tl.store(TEXT+(h.to(tl.int64)*R-ri-1)*128+d,source)

    _KERNELS = (_pack_error,)
    return _KERNELS


def _require_cuda(values):
    base._require_cuda(values)


@torch.no_grad()
def pack_v_with_error(values, route, error_out_layer, *, mode=MODE):
    """Return frozen (CudaVisualVBank,text), write caller e without owning it.

    Caller must validate_pack_status([all layer banks]) before accepting the
    diagnostics/filling; a single flagged row rejects the complete result. No
    host tensor read occurs here. Inference-mode callers allocate error_out with
    inference_mode(False). Version snapshots cannot detect arbitrary .data/raw
    device-pointer mutation; exclusive immutable ownership remains required.
    """
    plan=validate_pack_error_metadata(values,route,error_out_layer,mode=mode)
    _require_cuda(values)
    (pack,)=_get_kernels()
    with torch.inference_mode(False):
        visual=torch.empty((1,base.HEADS,route.M,base.D),device=values.device,dtype=torch.int8)
        scales=torch.empty((1,base.HEADS,route.M,base.D//base.GROUP),device=values.device,dtype=torch.float32)
        text=torch.empty((1,base.HEADS,route.R,base.D),device=values.device,dtype=torch.float16)
        flag=torch.zeros((),device=values.device,dtype=torch.int32)
    outputs=[visual,scales,text,flag]
    forbidden={base._sid(values),base._sid(route.prefix_route),base._sid(error_out_layer)}
    if any(x.numel() and base._sid(x) in forbidden for x in outputs):
        raise RuntimeError('diagnostic pack bank/text outputs alias source/route/error output')
    nonempty=[x for x in outputs if x.numel()]
    if len({base._sid(x) for x in nonempty})!=len(nonempty):
        raise RuntimeError('diagnostic pack output backings alias')
    pack[plan['grid']](values,route.prefix_route,visual,scales,text,flag,error_out_layer,
        route.P,route.M,route.R,*values.stride()[1:],num_warps=4,num_stages=1,enable_fp_fusion=False)
    # Same frozen bank type and fields. It owns neither source nor error output.
    bank=base._construct(base.CudaVisualVBank,route=route,mode=MODE,visual_values=visual,scales=scales,
        error_flag=flag,_versions=(visual._version,scales._version),validated=False,failed=False)
    bank.validate_metadata()
    base._tensor_meta(text,'compact prompt text V',(1,base.HEADS,route.R,base.D),torch.float16,values.device,True)
    validate_error_output_metadata(error_out_layer,plan['error_output'])
    return bank,text


def quantization_error_fp32_reference(visual_values):
    """Explicit CPU oracle; full temporary tensors, never execution fallback.

    Inputs are already gathered compact visual rows. Compared with the earlier
    FP64 analysis, this measures after separately rounded FP32 multiplication
    and subtraction. It is not guaranteed to round upward.
    """
    from visual_v_quant_risk_reference import quantize_visual_v_reference
    packed=quantize_visual_v_reference(visual_values)
    return (packed.represented-visual_values.float()).abs().amax(-1)[0].contiguous()
