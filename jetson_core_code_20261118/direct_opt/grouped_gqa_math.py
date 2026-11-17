"""Two-BMM FP32 GQA candidate, with seven query heads folded into rows.

This uses the frozen stride0 candidate's Torch2.8 math order: separate FP32
sqrt(scale) multiplications, the original additive mask, full-shape safe
softmax and the otherwise unused FP16 probability cast. Only small Q is
packed; K/V keep four unique heads. GEMM shapes change, so GPU bitwise output
and allocated-peak equivalence require measurement. No fused-kernel, SRAM,
DRAM, throughput or new GQA algorithm claim is made.
"""
from __future__ import annotations

import math
import torch

from stride0_gqa_math import D, GROUP, HKV, HQ, _description, _validate, logical_copy_accounting


@torch.no_grad()
def grouped_gqa_math(query, key, value, mask, scale=None, *, record_receipt=False,
                     debug_intermediates=False):
    """Return (HF [1,T,28,128] FP16 output, optional receipt, optional debug).

    Q/K/V are already RoPE-rotated FP16, mask is the original FP32 additive
    tensor, and 1<=T<=30. No data-dependent host read, fallback, autocast or
    global precision mutation occurs. Default execution builds no storage
    dictionaries. Debug retains tensors and is unsuitable for ordinary peak
    or timing measurements; receipt-only diagnostics also add Python work.
    """
    if type(record_receipt) is not bool:
        raise TypeError('record_receipt must be bool')
    scale = _validate(query, key, value, mask, scale, debug_intermediates)
    t, n = query.shape[-2], key.shape[-2]
    factor = math.sqrt(scale)
    receipt = None
    if record_receipt:
        receipt = dict(implementation='grouped_two_bmm_fp32_math', torch_version=torch.__version__,
            scale=scale, sqrt_scale=factor,
            input={name: _description(x) for name, x in
                   (('query', query), ('key', key), ('value', value), ('mask', mask))},
            allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            reduced_precision_math_sdp=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()),
            qk_bmm_calls=1, pv_bmm_calls=1, native_bmm_calls=2, old_stride0_bmm_calls=8,
            head_mapping='row g*(7*T)+h*T+t uses KV g and Q head 7*g+h',
            explicit_repeat_kv=False, unique_kv_heads=HKV, mask_same_object=True,
            mask_values_transformed=False, rotary_reapplied=False,
            native_unused_probability_cast_preserved=True, debug_intermediates=debug_intermediates,
            debug_extra_copy_bytes=0, intermediate_tensors={},
            logical_copies=logical_copy_accounting(t, n), bitwise_equivalence_claim=False,
            logical_copy_comparison_scope='logical_copies removed table is KV-only; subtract the separately '
                'reported added Q-pack read/write before describing a net explicit-copy difference. '
                'Neither value counts GEMM reads or actual DRAM.',
            actual_internal_copy_measured=False, actual_dram_measured=False,
            ordinary_peak_vs_stride0_verified=False,
            receipt_scope='Python tensor layouts and explicit copy formulas; not simultaneous allocator peak, '
                          'SRAM or DRAM. BLAS packing/workspace and the ordinary GPU peak require a probe.')
    debug = {} if debug_intermediates else None
    with torch.autocast(device_type=query.device.type, enabled=False):
        q32 = query.to(torch.float32)
        q_scaled = q32 * factor
        if receipt is not None:
            receipt['intermediate_tensors'].update(q32=_description(q32), q_scaled=_description(q_scaled))
        if debug is not None:
            debug.update(q32=q32, q_scaled=q_scaled)
        del q32
        # Actual Q is [1,28,T,128] with stride (...,128,3584,1).
        # A contiguous logical head-major pack is necessary before folding
        # the head and token axes; a storage reinterpretation would mix them.
        needs_q_pack = not q_scaled.is_contiguous()
        grouped_query = q_scaled.contiguous().view(HKV, GROUP * t, D)
        if receipt is not None:
            q_bytes = q_scaled.numel() * q_scaled.element_size()
            receipt.update(query_pack_copy=needs_q_pack,
                query_pack_logical_read_bytes=q_bytes if needs_q_pack else 0,
                query_pack_logical_write_bytes=q_bytes if needs_q_pack else 0)
            receipt['intermediate_tensors']['grouped_query'] = _description(grouped_query)
        if debug is not None:
            debug['grouped_query'] = grouped_query
        del q_scaled

        k32 = key.to(torch.float32)
        k_scaled_t = k32.transpose(-2, -1) * factor
        if receipt is not None:
            receipt['intermediate_tensors'].update(k32=_description(k32),
                                                   k_scaled_transposed=_description(k_scaled_t))
        if debug is not None:
            debug.update(k32=k32, k_scaled_transposed=k_scaled_t)
        del k32
        grouped_key = k_scaled_t[0]
        scores = torch.bmm(grouped_query, grouped_key).view(1, HQ, t, n)
        if receipt is not None:
            receipt['intermediate_tensors']['grouped_key'] = _description(grouped_key)
            receipt['intermediate_tensors']['scores'] = _description(scores)
        if debug is not None:
            debug['qk_scores'] = scores.clone()
            if receipt is not None:
                receipt['debug_extra_copy_bytes'] = scores.numel() * scores.element_size()
        del grouped_query, grouped_key, k_scaled_t

        scores.add_(mask)
        probabilities = torch.ops.aten._safe_softmax.default(scores, -1, None)
        if receipt is not None:
            receipt['intermediate_tensors']['probabilities'] = _description(probabilities)
        if debug is not None:
            debug.update(masked_scores=scores, probabilities=probabilities)
        del scores

        # V is needed only after softmax. Deferring this cast and releasing
        # Q/K temporaries avoids keeping all those banks live together.
        v32 = value.to(torch.float32)
        grouped_probabilities = probabilities.view(HKV, GROUP * t, n)
        grouped_value = v32[0]
        pv = torch.bmm(grouped_probabilities, grouped_value).view(1, HQ, t, D)
        if receipt is not None:
            receipt['intermediate_tensors'].update(v32=_description(v32),
                grouped_probabilities=_description(grouped_probabilities),
                grouped_value=_description(grouped_value), pv_fp32=_description(pv))
        if debug is not None:
            debug.update(v32=v32, pv_fp32=pv)
        del grouped_probabilities, grouped_value, v32
        output_bhtd = pv.to(query.dtype)
        del pv
        # Preserve the native math operator's discarded second output cast.
        unused_weights = probabilities.to(query.dtype)
        if receipt is not None:
            receipt['intermediate_tensors'].update(output_bhtd_fp16=_description(output_bhtd),
                native_unused_weights_fp16=_description(unused_weights))
        if debug is not None:
            debug.update(output_bhtd_fp16=output_bhtd, native_unused_weights_fp16=unused_weights)
        del unused_weights, probabilities
        output = output_bhtd.transpose(1, 2).contiguous()
        if receipt is not None:
            copied = output.untyped_storage().data_ptr() != output_bhtd.untyped_storage().data_ptr()
            receipt.update(output=_description(output), hf_output_layout_copy=copied,
                hf_output_layout_copy_read_plus_write_bytes=
                    2 * output.numel() * output.element_size() if copied else 0)
    return output, receipt, debug
