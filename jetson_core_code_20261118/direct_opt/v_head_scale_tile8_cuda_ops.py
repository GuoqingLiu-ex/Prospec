"""Eight source rows/head/CTA INT8 pack with one visual-risk atomic per tile.

This is an independent candidate, not a replacement for frozen head-scale ops.
Per-row quantization preserves the FP32 maxabs/div_rn/nearbyint sequence. A CTA
reduces its visual rows' original scale/2 risks before atomic_max. All source
rows, including text, are read once; no global score scratch or second pack
launch is introduced. Register pressure, occupancy and GPU equality are untested
here. The CPU oracle and logical operation budget are not GPU execution proof.
"""
from __future__ import annotations

import torch
import v_bank_cuda_ops as base
import v_head_scale_cuda_ops as frozen

SCHEMA = 'visual_v_pack_head_scale_tile8_v1'
MODE = 'int8_visual'
ROWS_PER_CTA = 8
PACK_DYNAMIC = base.PACK_DYNAMIC
_KERNELS = None

prepare_partition_route = base.prepare_partition_route
validate_pack_status = base.validate_pack_status
check_status = base.check_status
fill_value_fp32_ = base.fill_value_fp32_
head_score_metadata = frozen.head_score_metadata
validate_head_score_metadata = frozen.validate_head_score_metadata


def pack_head_scale_tile8_launch_plan(P, M, mode=MODE, *, visual_positions=None):
    """Logical work only. Exact occupied tiles use an already-owned host tuple.

    M alone does not determine occupied source tiles: visual rows may be mixed
    with text. Never retrieve CUDA route contents to refine this budget.
    """
    original = frozen.pack_head_scale_launch_plan(P, M, mode)
    tiles = (P - 1) // ROWS_PER_CTA + 1
    occupied = None
    if visual_positions is not None:
        if (not isinstance(visual_positions, tuple) or len(visual_positions) != M
                or any(type(p) is not int or not 0 <= p < P for p in visual_positions)
                or any(a >= b for a, b in zip(visual_positions, visual_positions[1:]))):
            raise ValueError('visual_positions must be the sorted unique prepared host tuple of M positions')
        occupied = len({p // ROWS_PER_CTA for p in visual_positions})
    atomics = None if occupied is None else base.HEADS * occupied
    return dict(original, schema=SCHEMA, grid=(base.HEADS * tiles,),
        rows_per_CTA=ROWS_PER_CTA, source_tiles_per_head=tiles,
        masked_tail_rows_per_head=tiles * ROWS_PER_CTA - P,
        token_padding=0, padding_scope='masked lanes only; no padded output bank or source read',
        visual_occupied_tiles_per_head=occupied,
        visual_occupied_tiles_per_head_bounds=((M + 7) // 8, min(M, tiles)),
        original_head_scale_atomic_max_operations=base.HEADS * M,
        extra_atomic_max_operations=atomics,
        atomic_max_operations_bounds=(4 * ((M + 7) // 8), 4 * min(M, tiles)),
        atomic_count_source='prepared CPU partition positions' if occupied is not None else 'bounds only; positions absent',
        naive_atomic_read_plus_write_byte_model=None if atomics is None else atomics * 8,
        atomic_byte_model_scope='one logical FP32 RMW per visual-occupied tile/head, not DRAM transactions or bytes',
        score_semantics='max of original per-row visual scale/2 risks in tile, then atomic_max accumulation',
        logical_extra_per_visual_head_row=dict(group_zero_masks=4, masked_scale_values=4,
            max_reduction_input_elements=4, FP32_half_scaling=1),
        logical_tile_score_reduction_input_elements=8,
        mixed_tile_quantization='visual-occupied CTA computes masked quantization on all 8 lanes; text results are discarded',
        explicit_tile_value_elements=8 * 128, explicit_tile_FP32_values_bytes=8 * 128 * 4,
        tile_residency_measured=False, global_score_scratch_bytes=0,
        kernel_launch_scope='one replacement tile8 custom pack kernel; output/flag allocation operations excluded',
        additional_costs_unmeasured=base.pack_launch_plan(P, M, MODE)['additional_costs_unmeasured'] + (
            'caller score slab allocation/zero initialization and stream ordering',
            '8-row live values, row reductions, tile max and register pressure/occupancy/spills',
            'quantization arithmetic on nonvisual or padding lanes of a mixed visual tile',
            'FP32 atomic contention among visual-occupied tiles per head and returned-old guard',
            'caller aggregate status/score readback or selection, not performed in pack'),
        scope='logical operation reduction candidate, not measured transfers, GPU time, residency or QA')


def validate_pack_head_scale_tile8_metadata(values, route, scores, *, mode=MODE):
    original = frozen.validate_pack_head_scale_metadata(values, route, scores, mode=mode)
    return dict(pack_head_scale_tile8_launch_plan(route.P, route.M, mode,
        visual_positions=route.partition.visual), head_scores=original['head_scores'])


def _get_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    try:
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice
    except ImportError as exc:
        raise RuntimeError('Triton tile8 head-scale pack unavailable; no fallback') from exc

    @triton.jit(do_not_specialize=PACK_DYNAMIC)
    def _pack_head_scale_tile8(SRC,ROUTE,VIS,SCALES,TEXT,ERROR,HEAD_SCORES,P,M,R,SH,SN,SD):
        blocks=(P-1)//8+1
        pid=tl.program_id(0)
        h=pid//blocks
        j=(pid%blocks)*8+tl.arange(0,8)
        valid=j<P
        ri=tl.load(ROUTE+j,mask=valid,other=0).to(tl.int64)
        visual=valid&(ri>=0)&(ri<M)
        text=valid&(ri<0)&(-ri-1<R)
        valid_route=visual|text
        d=tl.arange(0,128)
        source=tl.load(SRC+h.to(tl.int64)*SH+j.to(tl.int64)[:,None]*SN+d[None,:]*SD,
            mask=valid[:,None],other=0)
        x=source.to(tl.float32)
        finite=(x==x)&(tl.abs(x)!=float('inf'))
        if tl.sum((valid&(~valid_route)).to(tl.int32),0)>0:tl.atomic_or(ERROR,1)
        if tl.sum(tl.sum((valid[:,None]&(~finite)).to(tl.int32),1),0)>0:tl.atomic_or(ERROR,2)
        if tl.sum(visual.to(tl.int32),0)>0:
            safe=tl.where(finite,x,0.)
            grouped=tl.reshape(safe,(8,4,32))
            maximum=tl.max(tl.abs(grouped),2)
            scale=tl.where(maximum==0,1.,tl.div_rn(maximum,127.))
            codes=libdevice.nearbyint(tl.div_rn(grouped,scale[:,:,None]))
            codes=tl.minimum(tl.maximum(codes,-127.),127.).to(tl.int8)
            tl.store(VIS+(h.to(tl.int64)*M+ri[:,None])*128+d[None,:],
                tl.reshape(codes,(8,128)),mask=visual[:,None])
            tl.store(SCALES+(h.to(tl.int64)*M+ri[:,None])*4+tl.arange(0,4)[None,:],
                scale,mask=visual[:,None])
            row_risk=tl.max(tl.where(maximum>0,scale,0.),1)*0.5
            risk=tl.max(tl.where(visual,row_risk,0.),0)
            old=tl.atomic_max(HEAD_SCORES+h,risk)
            old_ok=(old==old)&(tl.abs(old)!=float('inf'))&(old>=0)
            if not old_ok:tl.atomic_or(ERROR,8)
        tl.store(TEXT+(h.to(tl.int64)*R-ri[:,None]-1)*128+d[None,:],source,mask=text[:,None])

    _KERNELS = (_pack_head_scale_tile8,)
    return _KERNELS


def _require_cuda(values):
    base._require_cuda(values)


@torch.no_grad()
def pack_v_with_head_scale_tile8(values, route, head_scores_layer, *, mode=MODE):
    """Return frozen bank/text. Caller owns scores until stream completion.

    No host value validation or fallback. Frozen aggregate flag validation is
    mandatory before consuming scores or bank; invalid prior score uses bit 8.
    Pack never resets scores and bank never retains source or score references.
    """
    plan=validate_pack_head_scale_tile8_metadata(values,route,head_scores_layer,mode=mode)
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
        raise RuntimeError('tile8 head-scale pack outputs alias source/route/score output')
    nonempty=[x for x in outputs if x.numel()]
    if len({base._sid(x) for x in nonempty})!=len(nonempty):
        raise RuntimeError('tile8 head-scale pack output backings alias')
    pack[plan['grid']](values,route.prefix_route,visual,scales,text,flag,head_scores_layer,
        route.P,route.M,route.R,*values.stride()[1:],num_warps=4,num_stages=1,enable_fp_fusion=False)
    bank=base._construct(base.CudaVisualVBank,route=route,mode=MODE,visual_values=visual,scales=scales,
        error_flag=flag,_versions=(visual._version,scales._version),validated=False,failed=False)
    bank.validate_metadata()
    base._tensor_meta(text,'compact prompt text V',(1,base.HEADS,route.R,base.D),torch.float16,values.device,True)
    validate_head_score_metadata(head_scores_layer,plan['head_scores'])
    return bank,text


def tile8_cpu_reference(values, route, *, initial_scores=None):
    """Allocating CPU reference only; returns codes, scales, text, head scores.

    This models tile grouping using the frozen per-row quantization oracle. It
    does not run or emulate Triton instructions, and rejects invalid source.
    """
    from visual_v_quant_risk_reference import quantize_visual_v_reference
    base.validate_pack_metadata(values,route,MODE)
    if values.device.type != 'cpu':
        raise ValueError('tile8 reference is CPU only, never a CUDA fallback')
    if not bool(torch.isfinite(values).all()):
        raise ValueError('reference rejects nonfinite source, including text')
    visual=values[:,:,list(route.partition.visual)]
    packed=quantize_visual_v_reference(visual)
    text=values[:,:,list(route.partition.nonvisual)].contiguous()
    scores=torch.zeros(4,dtype=torch.float32)
    if initial_scores is not None:
        frozen.head_scale_risk_fp32_reference(visual,initial_scores=initial_scores)
        scores=initial_scores.clone()
    maximum=visual.float().reshape(1,4,route.M,4,32).abs().amax(-1)
    row_risk=torch.where(maximum>0,packed.scales,torch.zeros_like(packed.scales)).amax(-1)[0]*0.5
    occupied={p//8 for p in route.partition.visual}
    for tile in sorted(occupied):
        rows=[i for i,p in enumerate(route.partition.visual) if p//8==tile]
        scores=torch.maximum(scores,row_risk[:,rows].amax(-1))
    return packed.codes,packed.scales,text,scores.contiguous()
