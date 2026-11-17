"""Grouped FP32 math with an explicit physical head overlay; unused cast preserved.

Default fill is the frozen CUDA primitive. CPU reference injection is explicit;
there is no CPU/GPU fallback, FP16 full-V reconstruction, or second full V32 bank.
No model lifecycle, GPU execution, numerical or hardware benefit is implied here.
"""
from __future__ import annotations

import math
import torch
from stride0_gqa_math import D, GROUP, HKV, HQ, _description

METHOD = 'grouped_two_bmm_head_overlay_fp32_math'


@torch.no_grad()
def grouped_head_overlay_math(query, full_key, bank, compact_value, mask, scale=None, *,
                        overlay, layer_idx, fill_impl=None, record_receipt=False, debug_intermediates=False):
    if type(record_receipt) is not bool or type(debug_intermediates) is not bool:
        raise TypeError('receipt/debug flags must be bool')
    tensors = (query, full_key, compact_value, mask)
    if not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in tensors):
        raise ValueError('Q/full K/compact V/mask must be four-dimensional tensors')
    t, s = query.shape[-2], full_key.shape[-2]
    bank.validate_metadata()
    route = bank.route
    if (query.shape != (1, HQ, t, D) or not 1 <= t <= 30
            or full_key.shape != (1, HKV, s, D) or s < route.P
            or compact_value.shape != (1, HKV, s-route.M, D)
            or mask.shape != (1, 1, t, s)
            or any(x.dtype != torch.float16 for x in tensors[:3]) or mask.dtype != torch.float32
            or any(x.device != query.device for x in tensors)
            or query.device != bank.visual_values.device
            or any(x.requires_grad or x.stride(-1) != 1 or any(v < 0 for v in x.stride()) for x in tensors)):
        raise ValueError('partitioned V grouped operand contract differs')
    if query.device.type not in ('cpu', 'cuda'):
        raise ValueError('CPU reference or CUDA required')
    if type(layer_idx) is not int or not 0 <= layer_idx < 28:
        raise ValueError('explicit layer index [0,28) required')
    if fill_impl is None:
        if query.device.type != 'cuda':
            raise ValueError('default V fill requires CUDA; explicitly inject CPU reference')
        from v_head_overlay_cuda_ops import fill_overlay_value_fp32_
        fill_impl = fill_overlay_value_fp32_
    elif query.device.type != 'cpu':
        raise ValueError('custom fill injection is CPU reference only; CUDA uses frozen overlay fill')
    if bank.failed or not bank.validated:
        raise RuntimeError('bank requires successful setup status and cannot be poisoned')
    scale = D ** -.5 if scale is None else scale
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite scalar scale required')
    if torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
        raise RuntimeError('reduced precision math SDPA must be disabled')
    if query.device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError('TF32 must be disabled by caller')
    factor = math.sqrt(scale)
    receipt = None
    if record_receipt:
        receipt = dict(implementation=METHOD, mode=bank.mode, original_mask_object=True,
            overlay_layer=layer_idx, overlay_budget_heads=overlay.budget_heads,
            fp32_operand_count=1, full_fp16_value_reconstruction=False,
            native_unused_probability_cast_preserved=True, debug_intermediates=debug_intermediates,
            qk_bmm_calls=1, pv_bmm_calls=1, attention_major_macs=2*HQ*t*s*D,
            shape=dict(query_tokens=t, logical_key_tokens=s, compact_value_tokens=s-route.M),
            input={name:_description(x) for name,x in zip(('query','key','compact_value','mask'),tensors)},
            query_pack_copy=not query.is_contiguous(), output_operand_retained=debug_intermediates,
            actual_dram_measured=False, model_GPU_output_equivalence_verified=False)
    debug = {} if debug_intermediates else None
    with torch.autocast(device_type=query.device.type, enabled=False):
        q32 = query.to(torch.float32)
        q_scaled = q32 * factor
        if debug is not None: debug.update(q32=q32, q_scaled=q_scaled)
        del q32
        grouped_query = q_scaled.contiguous().view(HKV, GROUP*t, D)
        if debug is not None: debug['grouped_query'] = grouped_query
        del q_scaled
        k32 = full_key.to(torch.float32)
        k_scaled_t = k32.transpose(-2,-1) * factor
        if debug is not None: debug.update(k32=k32, k_scaled_transposed=k_scaled_t)
        del k32
        grouped_key = k_scaled_t[0]
        scores = torch.bmm(grouped_query, grouped_key).view(1,HQ,t,s)
        if debug is not None: debug['qk_scores'] = scores.clone()
        del grouped_query, grouped_key, k_scaled_t
        scores.add_(mask)
        probabilities = torch.ops.aten._safe_softmax.default(scores,-1,None)
        if debug is not None: debug.update(masked_scores=scores, probabilities=probabilities)
        del scores
        # One destination in the same logical order and contiguous shape as value.float().
        v32 = torch.empty((1,HKV,s,D), device=query.device, dtype=torch.float32)
        if fill_impl(v32, bank, compact_value, overlay, layer_idx) is not v32:
            raise RuntimeError('fill must return exactly the caller-owned FP32 operand')
        grouped_probabilities = probabilities.view(HKV,GROUP*t,s)
        grouped_value = v32[0]
        pv = torch.bmm(grouped_probabilities, grouped_value).view(1,HQ,t,D)
        if debug is not None: debug.update(v32=v32, pv_fp32=pv)
        del grouped_probabilities, grouped_value, v32
        output_bhtd = pv.to(query.dtype)
        del pv
        unused_weights = probabilities.to(query.dtype)
        if debug is not None: debug.update(output_bhtd_fp16=output_bhtd, native_unused_weights_fp16=unused_weights)
        del unused_weights, probabilities
        output = output_bhtd.transpose(1,2).contiguous()
    return output, receipt, debug
