"""Prototype mixed INT8-visual / FP16-text tree attention.

No HF patching, no fallback, no automatic CUDA initialization on import.
Both banks participate in ONE online softmax. K/V retain original RoPE values;
positions map columns of the supplied original additive/bool tree mask.
FP16 control uses this same kernel and traversal. Fusion arithmetic differs
from native FP32 math SDPA; this module makes no token-equivalence claim.
"""
from __future__ import annotations
import math
import torch
try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = tl = None

from .visual_kv_int8 import VisualKVBank, KVPartition, KVPositionMap, padded_tokens, validate_source

# These values change with prompt/confirmed-text length. Keeping them dynamic
# also covers narrow views whose leading strides reflect fixed backing capacity.
# do_not_specialize prevents Triton's integer equality/alignment specializations.
MIXED_RUNTIME_SCALARS = tuple(f"{prefix}{axis}" for prefix in "QVWKFM" for axis in range(4)) + ("NV", "NP", "NF")


if triton is not None:
    @triton.jit
    def _pack(SRC, POS, OUT, SCALE, INVALID,
              S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
              H: tl.constexpr, N: tl.constexpr, NP: tl.constexpr,
              QUANT: tl.constexpr):
        # One 128-channel token/head per program; scale per 32 channels.
        row = tl.program_id(0)
        n = row % NP
        bh = row // NP
        b, h = bh // H, bh % H
        d = tl.arange(0, 128)
        original = tl.load(POS + n, n < N, other=0)
        x = tl.load(SRC + b*S0 + h*S1 + original*S2 + d*S3,
                    n < N, other=0).to(tl.float32)
        finite = (x == x) & (tl.abs(x) != float("inf"))
        if tl.sum((~finite).to(tl.int32), 0) != 0:
            tl.atomic_or(INVALID, 1)
        if QUANT:
            xg = tl.reshape(x, (4, 32))
            a = tl.max(tl.abs(xg), 1)
            scale = tl.where(a == 0, 1.0, tl.div_rn(a, 127.0))
            q = libdevice.nearbyint(tl.div_rn(xg, scale[:, None]))
            q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
            tl.store(OUT + row*128 + d, tl.reshape(q, (128,)))
            tl.store(SCALE + row*4 + tl.arange(0, 4), scale)
        else:
            tl.store(OUT + row*128 + d, x.to(tl.float16))

    @triton.jit(do_not_specialize=MIXED_RUNTIME_SCALARS)
    def _mixed(Q, VK, VV, KS, VS, VP, FK, FV, FP, MASK, OUT,
               Q0,Q1,Q2,Q3,
               V0,V1,V2,V3,
               W0,W1,W2,W3,
               K0,K1,K2,K3,
               F0,F1,F2,F3,
               M0,M1,M2,M3,
               MB:tl.constexpr,MH:tl.constexpr,HQ:tl.constexpr,HKV:tl.constexpr,
               T:tl.constexpr,NV,NP,NF,
               QUANT:tl.constexpr,BOOL_MASK:tl.constexpr,SM_SCALE:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr):
        bh = tl.program_id(0)
        b, h = bh // HKV, bh % HKV
        rows = tl.program_id(1)*BM + tl.arange(0, BM)
        group = HQ // HKV
        qi, qh = rows // group, h*group + rows % group
        d = tl.arange(0, 128)
        q = tl.load(Q+b*Q0+qh[:,None]*Q1+qi[:,None]*Q2+d[None,:]*Q3,
                    qi[:,None] < T, other=0)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.zeros((BM,), tl.float32)
        accumulator = tl.zeros((BM,128), tl.float32)
        mask_b = 0 if MB == 1 else b
        mask_h = tl.full((BM,), 0, tl.int32) if MH == 1 else qh
        # Each segment is contiguous storage but uses original semantic mask columns.
        for segment in tl.static_range(2):
            end = NV if segment == 0 else NF
            for start in range(0, end, BN):
                n = start + tl.arange(0, BN)
                if segment == 0:
                    original = tl.load(VP+n, n < NV, other=0)
                    k = tl.load(VK+b*V0+h*V1+n[:,None]*V2+d[None,:]*V3,
                                n[:,None] < NV, other=0)
                    v = tl.load(VV+b*W0+h*W1+n[:,None]*W2+d[None,:]*W3,
                                n[:,None] < NV, other=0)
                    if QUANT:
                        scale_offset = ((b*HKV+h)*NP+n[:,None])*4+d[None,:]//32
                        sk = tl.load(KS+scale_offset, n[:,None] < NV, other=1)
                        sv = tl.load(VS+scale_offset, n[:,None] < NV, other=1)
                        k = (k.to(tl.float32)*sk).to(tl.float16)
                        v = (v.to(tl.float32)*sv).to(tl.float16)
                else:
                    original = tl.load(FP+n, n < NF, other=0)
                    k = tl.load(FK+b*K0+h*K1+n[:,None]*K2+d[None,:]*K3,
                                n[:,None] < NF, other=0)
                    v = tl.load(FV+b*F0+h*F1+n[:,None]*F2+d[None,:]*F3,
                                n[:,None] < NF, other=0)
                valid = (qi[:,None] < T) & (n[None,:] < end)
                bias = tl.load(MASK+mask_b*M0+mask_h[:,None]*M1+qi[:,None]*M2+original[None,:]*M3,
                               valid, other=0)
                scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SM_SCALE
                if BOOL_MASK:
                    scores = tl.where(valid & bias, scores, -float("inf"))
                else:
                    scores = tl.where(valid, scores+bias.to(tl.float32), -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                alpha = tl.where(maximum == -float("inf"), 0.0, tl.exp(maximum-new_max))
                p = tl.where(new_max[:,None] == -float("inf"), 0.0,
                             tl.exp(scores-new_max[:,None]))
                denominator = denominator*alpha + tl.sum(p, 1)
                accumulator = accumulator*alpha[:,None]
                accumulator = tl.dot(p.to(tl.float16), v, accumulator, input_precision="ieee")
                maximum = new_max
        output = accumulator / tl.where(denominator == 0, 1.0, denominator)[:,None]
        tl.store(OUT+((b*HQ+qh[:,None])*T+qi[:,None])*128+d[None,:],
                 output.to(tl.float16), qi[:,None] < T)


def _require_cuda(tensor):
    if triton is None:
        raise RuntimeError("Triton is not installed; no CPU or SDPA fallback")
    if tensor.device.type != "cuda":
        raise ValueError("Triton execution requires CUDA; CPU tests use explicit references")


def pack_visual_cuda(keys, values, partition: KVPartition, *, quantized=True) -> VisualKVBank:
    validate_source(keys, values, partition)
    _require_cuda(keys)
    b,h,_,d = keys.shape
    n = len(partition.visual)
    npad = padded_tokens(n)
    positions = torch.tensor(partition.visual, dtype=torch.long, device=keys.device)
    dtype = torch.int8 if quantized else torch.float16
    outk,outv = [torch.empty((b,h,npad,d),device=keys.device,dtype=dtype) for _ in range(2)]
    scales = [torch.empty((b,h,npad,4),device=keys.device,dtype=torch.float32) for _ in range(2)] if quantized else [None,None]
    flag = torch.zeros((),device=keys.device,dtype=torch.int32)
    if npad:
        for source, output, scale in zip((keys,values),(outk,outv),scales):
            _pack[(b*h*npad,)](source,positions,output,scale if quantized else output,flag,
                               *source.stride(),h,n,npad,quantized,num_warps=4)
    bank = VisualKVBank(outk,outv,positions,n,*scales,flag,partition.visual)
    bank.validate_metadata()
    return bank


def validate_attention_metadata(query, visual, fp16_k, fp16_v, fp16_positions, mask):
    visual.validate_metadata()
    if query.ndim != 4 or query.dtype != torch.float16 or query.shape[-1] != 128:
        raise ValueError("query must be FP16 [B,Hq,T,128]")
    b,hq,t,d = query.shape
    if fp16_k.ndim != 4 or fp16_k.shape != fp16_v.shape or fp16_k.shape[:2] != visual.keys.shape[:2]:
        raise ValueError("FP16 bank must match visual batch and KV heads")
    if fp16_k.shape[0] != b or fp16_k.shape[-1] != d or hq % fp16_k.shape[1] or t < 1:
        raise ValueError("incompatible GQA/query dimensions")
    if fp16_k.dtype != torch.float16 or fp16_v.dtype != torch.float16:
        raise ValueError("text/tree bank must stay FP16")
    if not isinstance(fp16_positions, KVPositionMap):
        raise TypeError("FP16 mapping requires a setup-created KVPositionMap")
    map_tensor = fp16_positions.indices
    if map_tensor.dtype != torch.long or tuple(map_tensor.shape) != (fp16_k.shape[2],) or map_tensor.stride() != (1,):
        raise ValueError("FP16 positions must be int64 [Nactive]")
    if mask.ndim != 4 or mask.shape[0] not in (1,b) or mask.shape[1] not in (1,hq) or mask.shape[2] != t:
        raise ValueError("mask must be [B|1,Hq|1,T,Noriginal]")
    if mask.dtype not in (torch.bool,torch.float16,torch.float32):
        raise ValueError("unsupported mask dtype")
    if (len(fp16_positions.host_indices) != fp16_k.shape[2] or
        fp16_positions.total_tokens != mask.shape[-1] or
        fp16_positions.visual_indices != visual.host_positions):
        raise ValueError("two banks must cover every original mask column exactly once")
    tensors = [visual.keys,fp16_k,fp16_v,map_tensor,mask]
    if any(x.device != query.device for x in tensors):
        raise ValueError("all tensors must share the query device")
    if query.device.type == "cpu":
        if tuple(visual.positions.tolist()) != visual.host_positions or tuple(map_tensor.tolist()) != fp16_positions.host_indices:
            raise ValueError("CPU mapping tensor differs from its setup receipt")
    # CUDA map tensors are immutable setup products; reading them back here would
    # add a per-layer host synchronization. Setup receipts validate full coverage.


def mixed_attention(query, visual: VisualKVBank, fp16_k, fp16_v, fp16_positions,
                    mask, *, scale=None, block_m=32, block_n=64):
    validate_attention_metadata(query,visual,fp16_k,fp16_v,fp16_positions,mask)
    _require_cuda(query)
    if block_m != 32 or block_n != 64:
        raise ValueError("first configuration is frozen at block_m=32, block_n=64")
    scale = 1.0/math.sqrt(128) if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("finite positive scale required")
    b,hq,t,_ = query.shape
    hkv = fp16_k.shape[1]
    output = torch.empty(query.shape,device=query.device,dtype=torch.float16)
    _mixed[(b*hkv,triton.cdiv(t*(hq//hkv),block_m))](
        query,visual.keys,visual.values,
        visual.key_scales if visual.quantized else visual.keys,
        visual.value_scales if visual.quantized else visual.values,visual.positions,
        fp16_k,fp16_v,fp16_positions.indices,mask,output,
        *query.stride(),*visual.keys.stride(),*visual.values.stride(),
        *fp16_k.stride(),*fp16_v.stride(),*mask.stride(),
        mask.shape[0],mask.shape[1],hq,hkv,t,visual.logical_tokens,visual.keys.shape[2],fp16_k.shape[2],
        visual.quantized,mask.dtype==torch.bool,scale,block_m,block_n,
        # SM120 has 101376 B shared memory per block. The two-bank kernel with
        # two stages requested 106496 B; one stage keeps this tile size viable.
        num_warps=4,num_stages=1)
    return output
