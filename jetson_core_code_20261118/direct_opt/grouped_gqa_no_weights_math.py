"""Exact-output candidate that does not materialize discarded FP16 weights.

The frozen grouped primitive returns only an output to the HF policy. Its
FP16 probability cast is a discarded diagnostic inherited from native math.
This separate candidate preserves all live matrix/softmax operations and their
order. It has no runtime integration and no GPU/DRAM/performance result yet.
"""
from __future__ import annotations

import math
import torch

from stride0_gqa_math import D, GROUP, HKV, HQ, _description, _validate


@torch.no_grad()
def grouped_gqa_no_weights_math(query, key, value, mask, scale=None, *,
                                record_receipt=False, debug_intermediates=False):
    """Return HF output, optional scalar receipt and optional live intermediates.

    Debug retains tensors and invalidates ordinary memory/timing comparisons.
    No discarded FP16 probability tensor is produced, even in debug mode.
    Callers requesting attention weights must reject this output-only primitive.
    """
    if type(record_receipt) is not bool:
        raise TypeError('record_receipt must be bool')
    scale = _validate(query, key, value, mask, scale, debug_intermediates)
    t, n = query.shape[-2], key.shape[-2]
    factor = math.sqrt(scale)
    receipt = None
    if record_receipt:
        count = HQ * t * n
        receipt = dict(implementation='grouped_two_bmm_no_discarded_weights',
            scale=scale, sqrt_scale=factor, torch_version=torch.__version__,
            input={name: _description(x) for name, x in
                   (('query', query), ('key', key), ('value', value), ('mask', mask))},
            qk_bmm_calls=1, pv_bmm_calls=1, explicit_repeat_kv=False,
            original_mask_object=True, mask_values_transformed=False,
            native_unused_probability_cast_preserved=False,
            discarded_probability_materialization=False,
            removed_discarded_probability_copy=dict(elements=count,
                read_bytes=4*count, write_bytes=2*count,
                read_plus_write_bytes=6*count,
                scope='Logical element payload, not physical DRAM or kernel count'),
            kv_resident_capacity_reduction=0, attention_major_mac_reduction=0,
            actual_dram_measured=False, actual_kernel_count_measured=False,
            ordinary_GPU_output_equivalence_verified=False,
            debug_intermediates=debug_intermediates)
    debug = {} if debug_intermediates else None
    with torch.autocast(device_type=query.device.type, enabled=False):
        q32 = query.to(torch.float32)
        q_scaled = q32 * factor
        if debug is not None:
            debug.update(q32=q32, q_scaled=q_scaled)
        del q32
        grouped_query = q_scaled.contiguous().view(HKV, GROUP * t, D)
        if debug is not None:
            debug['grouped_query'] = grouped_query
        del q_scaled

        k32 = key.to(torch.float32)
        k_scaled_t = k32.transpose(-2, -1) * factor
        if debug is not None:
            debug.update(k32=k32, k_scaled_transposed=k_scaled_t)
        del k32
        grouped_key = k_scaled_t[0]
        scores = torch.bmm(grouped_query, grouped_key).view(1, HQ, t, n)
        if debug is not None:
            debug['qk_scores'] = scores.clone()
        del grouped_query, grouped_key, k_scaled_t

        scores.add_(mask)
        probabilities = torch.ops.aten._safe_softmax.default(scores, -1, None)
        if debug is not None:
            debug.update(masked_scores=scores, probabilities=probabilities)
        del scores

        v32 = value.to(torch.float32)
        grouped_probabilities = probabilities.view(HKV, GROUP * t, n)
        grouped_value = v32[0]
        pv = torch.bmm(grouped_probabilities, grouped_value).view(1, HQ, t, D)
        if debug is not None:
            debug.update(v32=v32, pv_fp32=pv)
        del grouped_probabilities, grouped_value, v32
        output_bhtd = pv.to(query.dtype)
        del pv
        if debug is not None:
            debug['output_bhtd_fp16'] = output_bhtd
        # Output has already been computed. No attention-weight result is used.
        del probabilities
        output = output_bhtd.transpose(1, 2).contiguous()
    return output, receipt, debug
