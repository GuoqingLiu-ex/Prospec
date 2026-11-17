"""Training-free local visual compressors with complete donor coverage.

The input order is flattened (time, height, width). Full 2x2 spatial cells
never cross a temporal patch group. Tokens in an odd final row or column are
kept as singletons. Budgeted methods have a statically known output size.
The returned group sizes are integer donor multiplicities; the caller decides
whether to apply log(group_size) correction in its frozen attention adaptor.

All content-dependent ranking stays on the input device. The threshold gate
reads one scalar eligible-cell count to allocate its variable-length output;
this synchronization belongs inside the measured compression cost. No CPU
sort or data-dependent nonzero operation is used.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import torch


METHODS = ("attention", "difference", "response")
SUPPORTED_METHODS = METHODS + ("budgeted_difference", "difference_gate")


def _validate(x, grid_thw, ratio, method, accumulation_dtype):
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("x must be a floating point tensor of shape [N, D]")
    if not x.is_floating_point() or x.shape[1] <= 0:
        raise ValueError("x must have a floating point dtype and positive feature dimension")
    if isinstance(grid_thw, torch.Tensor):
        raise TypeError("Pass grid_thw as CPU integer metadata, not a device tensor")
    if len(grid_thw) != 3 or any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in grid_thw):
        raise ValueError("grid_thw must contain three positive integers (T, H, W)")
    grid = tuple(int(v) for v in grid_thw)
    if math.prod(grid) != x.shape[0]:
        raise ValueError("x.shape[0] must equal T * H * W")
    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("ratio must be a finite host number in (0, 1]")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulation_dtype must be torch.float32 or torch.float64")
    return grid, float(ratio)


def _check_signal(value, name, x, ndim):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or value.shape[0] != x.shape[0]:
        raise ValueError(f"{name} must have shape [N]" if ndim == 1 else f"{name} must have shape [N, HQ]")
    if value.device != x.device:
        raise ValueError(f"{name} and x must be on the same device")
    if not value.is_floating_point() or (ndim == 2 and value.shape[1] == 0):
        raise ValueError(f"{name} must be a floating point tensor with nonempty dimensions")


def _topology(x, grid):
    """Build fixed-size index tensors from host shape metadata only."""
    t, h, w = grid
    hh, ww = (h // 2) * 2, (w // 2) * 2
    ids = torch.arange(x.shape[0], device=x.device).reshape(t, h, w)
    cells = ids[:, :hh, :ww].reshape(t, h // 2, 2, w // 2, 2)
    cells = cells.permute(0, 1, 3, 2, 4).reshape(-1, 4)
    edges = []
    if h % 2:
        edges.append(ids[:, -1, :].reshape(-1))
    if w % 2:
        edges.append(ids[:, :hh, -1].reshape(-1))
    boundary = torch.cat(edges) if edges else torch.empty(0, dtype=torch.long, device=x.device)
    return cells, boundary


def _summary(x, grid, ratio, method, cells, boundary, actual_n, target_n, identity, accumulation_dtype):
    return {
        "method": method, "grid_thw": list(grid), "requested_ratio": ratio,
        "original_tokens": x.shape[0], "actual_tokens": actual_n,
        "actual_ratio": actual_n / x.shape[0], "target_tokens": target_n,
        "budget_error_tokens": actual_n - target_n, "full_2x2_cells": cells,
        "boundary_singletons": boundary, "boundary_policy": "keep_singletons",
        "identity": identity, "aggregation_dtype": str(accumulation_dtype),
        "output_dtype": str(x.dtype), "group_sizes_are_donor_counts": True,
        "group_size_correction_applied": False,
        "output_order": "interior cell groups followed by boundary singletons; identity preserves input order",
    }


def _donor_map(x, cell_ids, boundary_ids, interior_groups, interior_output_n):
    result = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    result.scatter_(0, cell_ids.reshape(-1), interior_groups.reshape(-1))
    result.scatter_(0, boundary_ids, torch.arange(boundary_ids.shape[0], device=x.device) + interior_output_n)
    return result


def _relative_difference(accumulator, cell_mean):
    energy = accumulator.square().mean(dim=(1, 2))
    variance = (energy - cell_mean.square().mean(dim=1)).clamp_min(0)
    return variance / energy.clamp_min(torch.finfo(accumulator.dtype).eps)


def cell_difference_scores(
    x: torch.Tensor,
    grid_thw: Sequence[int],
    *,
    accumulation_dtype: torch.dtype = torch.float32,
):
    """Return device-resident relative variance for each complete 2x2 cell.

    Scores follow (time, cell row, cell column) order. Odd boundary donors
    are excluded because they remain singletons under every method. This
    helper uses the exact score used by difference_gate and the budgeted
    baseline; threshold calibration and tensor serialization belong outside
    the request's measured path. A thin grid returns an empty tensor.
    """
    grid, _ = _validate(x, grid_thw, 1.0, "difference", accumulation_dtype)
    cell_ids, _ = _topology(x, grid)
    if cell_ids.shape[0] == 0:
        return torch.empty(0, dtype=accumulation_dtype, device=x.device)
    accumulator = x[cell_ids].to(dtype=accumulation_dtype)
    return _relative_difference(accumulator, accumulator.mean(dim=1))


def compress_tokens(
    x: torch.Tensor,
    grid_thw: Sequence[int],
    ratio: float = 0.5,
    method: str = "difference",
    *,
    saliency: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
    difference_threshold: float | None = None,
    accumulation_dtype: torch.dtype = torch.float32,
    return_donor_map: bool = False,
):
    """Return ``(tokens[M,D], group_sizes[M], summary)``.

    attention:
        Within each full cell, keep ceil(4*ratio)-1 high-saliency tokens and
        average the rest. ratio=.5 gives top1+pool3; .75 gives top2+pool2.
        Supply saliency[N], or scores[N,HQ] interpreted as frozen query/head
        logits. Logits become saliency by softmax over N and mean over HQ.
    difference / budgeted_difference:
        Rank full cells by relative FP32 feature variance. Keep every token
        in the highest-ranked K cells, pool each remaining cell into one.
        The compatibility alias "difference" is recorded as
        "budgeted_difference" in summary.
    difference_gate:
        Pool only cells whose relative variance is <= difference_threshold.
        ratio sets the maximum compression: output size is at least
        ceil(N*ratio). If more cells qualify than the budget permits, pool
        those with the smallest scores. Every high-difference cell remains
        intact. Supplying difference_threshold to "difference" selects
        this gate too. The gate reads one scalar count from the device;
        summary reports host_scalar_syncs=1 when this evaluation occurs.
    response:
        Same cells and exactly the same output budget as difference, ranked
        by within-cell variance of supplied query responses scores[N,HQ],
        averaged over HQ. Pre-softmax, scaled attention logits are preferred.

    difference/response use nearest feasible whole-cell budget, with ties
    favoring the larger budget. Odd boundaries are always singleton donors.
    Ratios below the all-pooled minimum are rounded up; .5/.75 are preferred.
    Thin grids and budgets rounded to no compression return x unchanged.

    Input features/signals must be finite. Finite-value checks are not run
    here because a host-side branch on their results would synchronize CUDA.
    Optional donor_to_group and retained indices remain DEVICE tensors in
    summary; serialize them only outside the measured path. The default
    summary contains ordinary shape/configuration metadata only.
    """
    grid, ratio = _validate(x, grid_thw, ratio, method, accumulation_dtype)
    if difference_threshold is not None:
        if (isinstance(difference_threshold, bool) or not isinstance(difference_threshold, Real)
                or not math.isfinite(difference_threshold) or difference_threshold < 0):
            raise ValueError("difference_threshold must be a finite nonnegative host number")
        if method not in ("difference", "difference_gate"):
            raise ValueError("difference_threshold applies only to difference or difference_gate")
        method = "difference_gate"
        difference_threshold = float(difference_threshold)
    elif method == "difference_gate":
        raise ValueError("difference_gate requires difference_threshold")
    if method == "difference":
        method = "budgeted_difference"
    t, h, w = grid
    cell_count = t * (h // 2) * (w // 2)
    boundary_count = x.shape[0] - 4 * cell_count
    target_n = math.ceil(x.shape[0] * ratio)
    outputs_per_cell = math.ceil(4 * ratio)
    cell_ids = boundary_ids = cell_x = accumulator = cell_mean = cell_scores = None
    gate_metadata = {"host_scalar_syncs": 0, "dynamic_output_shape": False}
    if method == "attention":
        actual_n = boundary_count + outputs_per_cell * cell_count
        keep_cells = None
    elif method == "difference_gate":
        max_pool_cells = min(cell_count, max(0, (x.shape[0] - target_n) // 3))
        eligible_cells = None
        pool_cells = 0
        if max_pool_cells:
            cell_ids, boundary_ids = _topology(x, grid)
            cell_x = x[cell_ids]
            accumulator = cell_x.to(dtype=accumulation_dtype)
            cell_mean = accumulator.mean(dim=1)
            cell_scores = _relative_difference(accumulator, cell_mean)
            # Deliberate single host scalar read: allocating the true output
            # shape requires it. Include this call in wall-time measurements.
            eligible_cells = int((cell_scores <= difference_threshold).sum().item())
            pool_cells = min(eligible_cells, max_pool_cells)
            gate_metadata.update(host_scalar_syncs=1, dynamic_output_shape=True)
        keep_cells = cell_count - pool_cells
        actual_n = x.shape[0] - 3 * pool_cells
        gate_metadata.update(difference_threshold=difference_threshold,
                             eligible_full_cells=eligible_cells,
                             max_pool_full_cells=max_pool_cells,
                             gate_evaluated=bool(max_pool_cells),
                             budget_policy="ratio_is_minimum_keep_fraction")
    else:
        extra_budget = target_n - boundary_count - cell_count
        keep_cells = max(0, min(cell_count, (extra_budget + 1) // 3))
        actual_n = boundary_count + cell_count + 3 * keep_cells
    identity = actual_n == x.shape[0]
    summary = _summary(x, grid, ratio, method, cell_count, boundary_count,
                       actual_n, target_n, identity, accumulation_dtype)
    summary.update(gate_metadata)
    if identity:
        group_sizes = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
        summary["score_source"] = "relative_feature_variance" if cell_scores is not None else "unused_identity"
        if method == "difference_gate":
            summary.update(retained_full_cells=cell_count, pooled_full_cells=0,
                           budget_quantum_tokens=3)
        if return_donor_map:
            summary["donor_to_group"] = torch.arange(x.shape[0], device=x.device)
        return x, group_sizes, summary

    if cell_ids is None:
        cell_ids, boundary_ids = _topology(x, grid)
        cell_x = x[cell_ids]
        accumulator = cell_x.to(dtype=accumulation_dtype)
    boundary_x = x[boundary_ids]

    if method == "attention":
        keep_tokens = outputs_per_cell - 1
        if keep_tokens:
            if saliency is not None:
                _check_signal(saliency, "saliency", x, 1)
                token_saliency = saliency.to(dtype=accumulation_dtype)
                summary["score_source"] = "supplied_saliency"
            else:
                _check_signal(scores, "scores", x, 2)
                token_saliency = scores.to(dtype=accumulation_dtype).softmax(dim=0).mean(dim=1)
                summary["score_source"] = "mean_query_head_attention"
            # Sorting four entries per row is fixed-size and stays on device.
            retained_local = torch.argsort(token_saliency[cell_ids], dim=1,
                                            descending=True, stable=True)[:, :keep_tokens]
            retained = cell_x.gather(1, retained_local.unsqueeze(-1).expand(-1, -1, x.shape[1]))
        else:
            retained_local = torch.empty((cell_count, 0), dtype=torch.long, device=x.device)
            retained = cell_x[:, :0]
            summary["score_source"] = "unused_all_pool"
        pooled_mask = torch.ones((cell_count, 4), dtype=torch.bool, device=x.device)
        pooled_mask.scatter_(1, retained_local, False)
        pooled = (accumulator * pooled_mask.unsqueeze(-1)).sum(dim=1) / (4 - keep_tokens)
        interior = torch.cat((retained, pooled.to(x.dtype).unsqueeze(1)), dim=1).reshape(-1, x.shape[1])
        cell_sizes = torch.ones((cell_count, outputs_per_cell), dtype=torch.long, device=x.device)
        cell_sizes[:, -1] = 4 - keep_tokens
        group_sizes = torch.cat((cell_sizes.reshape(-1), torch.ones(boundary_count, dtype=torch.long, device=x.device)))
        summary.update(retained_tokens_per_full_cell=keep_tokens,
                       retained_full_cells=0, pooled_full_cells=cell_count)
        if return_donor_map:
            local_groups = torch.full((cell_count, 4), keep_tokens, dtype=torch.long, device=x.device)
            local_groups.scatter_(1, retained_local, torch.arange(keep_tokens, device=x.device).expand(cell_count, -1))
            groups = torch.arange(cell_count, device=x.device).unsqueeze(1) * outputs_per_cell + local_groups
            summary["donor_to_group"] = _donor_map(x, cell_ids, boundary_ids, groups, interior.shape[0])
            summary["retained_token_indices"] = cell_ids.gather(1, retained_local).reshape(-1)
    else:
        if cell_mean is None:
            cell_mean = accumulator.mean(dim=1)
        if keep_cells:
            if method in ("budgeted_difference", "difference_gate"):
                if cell_scores is None:
                    cell_scores = _relative_difference(accumulator, cell_mean)
                summary["score_source"] = "relative_feature_variance"
            else:
                _check_signal(scores, "scores", x, 2)
                responses = scores.to(dtype=accumulation_dtype)[cell_ids]
                cell_scores = responses.var(dim=1, unbiased=False).mean(dim=1)
                summary["score_source"] = "query_response_variance"
            retained_cells = torch.argsort(cell_scores, descending=True, stable=True)[:keep_cells]
        else:
            retained_cells = torch.empty(0, dtype=torch.long, device=x.device)
            summary["score_source"] = "relative_feature_variance" if cell_scores is not None else "unused_all_pool"
        # One base output per cell. Selected cells replace their pooled base
        # with donor 0 and put donors 1/2/3 into a fixed-size extra tensor.
        base = cell_mean.to(x.dtype).index_copy(0, retained_cells, cell_x[retained_cells, 0])
        extras = cell_x[retained_cells, 1:].reshape(-1, x.shape[1])
        interior = torch.cat((base, extras), dim=0)
        base_sizes = torch.full((cell_count,), 4, dtype=torch.long, device=x.device)
        base_sizes.index_fill_(0, retained_cells, 1)
        group_sizes = torch.cat((base_sizes, torch.ones(3 * keep_cells + boundary_count, dtype=torch.long, device=x.device)))
        summary.update(retained_full_cells=keep_cells, pooled_full_cells=cell_count - keep_cells,
                       budget_quantum_tokens=3)
        if return_donor_map:
            groups = torch.arange(cell_count, device=x.device).unsqueeze(1).expand(-1, 4).clone()
            groups[retained_cells, 1:] = cell_count + torch.arange(3 * keep_cells, device=x.device).reshape(-1, 3)
            summary["donor_to_group"] = _donor_map(x, cell_ids, boundary_ids, groups, interior.shape[0])
            summary["retained_cell_ids"] = retained_cells
            summary["retained_token_indices"] = cell_ids[retained_cells].reshape(-1)
    return torch.cat((interior, boundary_x), dim=0), group_sizes, summary


def attention_compress(x, grid_thw, ratio=0.5, **kwargs):
    return compress_tokens(x, grid_thw, ratio, "attention", **kwargs)


def difference_compress(x, grid_thw, ratio=0.5, **kwargs):
    return compress_tokens(x, grid_thw, ratio, "difference", **kwargs)


def difference_gate_compress(x, grid_thw, ratio=0.5, *, difference_threshold, **kwargs):
    return compress_tokens(x, grid_thw, ratio, "difference_gate",
                           difference_threshold=difference_threshold, **kwargs)


def response_compress(x, grid_thw, ratio=0.5, **kwargs):
    return compress_tokens(x, grid_thw, ratio, "response", **kwargs)
