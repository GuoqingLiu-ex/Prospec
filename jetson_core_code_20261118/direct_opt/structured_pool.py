"""Fixed spatial mean pooling before the frozen Draft visual adaptor.

The input grid is AFTER Qwen's spatial merger: (T,H/s,W/s), s=2 in the current
model. T already counts temporal patches and is never divided again here.

Supported cells are horizontal (1,2), vertical (2,1), or square (2,2). Complete cells
are pooled independently inside each temporal patch. An incomplete last row or
column is retained as individual tokens. Every donor appears exactly once.
There is no score, sort, similarity, threshold, data-dependent shape or CPU sync
inside forward. This is a lossy baseline, not a claim of improved acceptance.

Default native avg_pool2d uses a channels-last view of the original contiguous
tokens. Float32 mean is an explicit alternative for numerical comparison. All
casts, pooling, boundary concatenation, and adaptor work remain inside forward.
Only shape metadata and constant group multiplicities are prepared beforehand.
"""
from __future__ import annotations

import math
from numbers import Integral
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class StructuredMeanPool(nn.Module):
    """[N,D]/[B,N,D] -> (same-rank pooled tokens, group_sizes[M]).

    Call prepare_for(x) before the first CUDA forward, then retain this instance.
    group_sizes is a constant buffer; do not modify it. CPU inputs work directly.
    Output order per temporal patch is pooled cells, bottom singleton row, then
    right singleton column (excluding the bottom corner already retained).
    """

    def __init__(self, grid_thw, *, block_hw=(1, 2), mean_precision="native"):
        super().__init__()
        if (not isinstance(grid_thw, (tuple, list)) or len(grid_thw) != 3
                or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in grid_thw)):
            raise ValueError("grid_thw must be three positive host integers AFTER spatial merger")
        if tuple(block_hw) not in ((1, 2), (2, 1), (2, 2)):
            raise ValueError("block_hw must be (1,2), (2,1), or (2,2)")
        if mean_precision not in ("native", "float32"):
            raise ValueError("mean_precision must be native or float32")
        self.grid_thw = tuple(int(v) for v in grid_thw)
        self.block_hw = tuple(block_hw)
        self.mean_precision = mean_precision
        self.t, self.h, self.w = self.grid_thw
        self.bh, self.bw = self.block_hw
        self.used_h = (self.h // self.bh) * self.bh
        self.used_w = (self.w // self.bw) * self.bw
        self.cells_per_time = (self.h // self.bh) * (self.w // self.bw)
        self.bottom_per_time = (self.h - self.used_h) * self.w
        self.right_per_time = self.used_h * (self.w - self.used_w)
        self.original_tokens = math.prod(self.grid_thw)
        self.identity = self.cells_per_time == 0
        if self.identity:
            sizes = torch.ones(self.original_tokens, dtype=torch.long)
            self.outputs_per_time = self.h * self.w
        else:
            per_time = torch.cat((
                torch.full((self.cells_per_time,), self.bh * self.bw, dtype=torch.long),
                torch.ones(self.bottom_per_time + self.right_per_time, dtype=torch.long),
            ))
            sizes = per_time.repeat(self.t)
            self.outputs_per_time = per_time.numel()
        self.output_tokens = self.t * self.outputs_per_time
        assert sizes.numel() == self.output_tokens and int(sizes.sum()) == self.original_tokens
        self.uniform_groups = self.identity or (self.bottom_per_time + self.right_per_time == 0)
        self.register_buffer("group_sizes", sizes, persistent=False)
        self.prepare_count = 0
        self.prepare_time_ms = 0.0

    def _shape(self, x: Tensor):
        if not isinstance(x, Tensor) or x.ndim not in (2, 3) or not x.is_floating_point():
            raise ValueError("x must be floating [N,D] or [B,N,D]")
        if x.shape[-2] != self.original_tokens or min(x.shape) < 1:
            raise ValueError("input token count must match the post-merger grid")
        return (1 if x.ndim == 2 else x.shape[0]), x.shape[-1]

    @torch.no_grad()
    def prepare_for(self, x: Tensor) -> dict[str, Any]:
        """Move constant multiplicities to the input device; no feature caching."""
        self._shape(x)
        _sync(x.device)
        started = time.perf_counter()
        self.group_sizes = self.group_sizes.to(device=x.device)
        _sync(x.device)
        self.prepare_time_ms = (time.perf_counter() - started) * 1000
        self.prepare_count += 1
        return self.metadata()

    def metadata(self) -> dict[str, Any]:
        return {"method": "structured_mean", "grid_after_spatial_merge": list(self.grid_thw),
                "block_hw": list(self.block_hw), "mean_precision": self.mean_precision,
                "original_tokens": self.original_tokens, "kept_tokens": self.output_tokens,
                "actual_keep_ratio": self.output_tokens / self.original_tokens,
                "pooled_cells": self.t * self.cells_per_time,
                "singleton_boundary_tokens": (self.original_tokens if self.identity else
                                               self.t * (self.bottom_per_time + self.right_per_time)),
                "uniform_group_sizes": self.uniform_groups, "identity": self.identity,
                "cache_bytes": self.group_sizes.numel() * self.group_sizes.element_size(),
                "prepare_count": self.prepare_count, "prepare_time_ms": self.prepare_time_ms,
                "host_content_syncs_in_forward": 0,
                "ordering": "within each temporal patch: pooled cells, bottom row, right column"}

    @torch.no_grad()
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, width = self._shape(x)
        if self.group_sizes.device != x.device:
            raise RuntimeError("call pool.prepare_for(x) before moving to request timing")
        if self.identity:
            return x, self.group_sizes
        pixels = x.reshape(batch, self.t, self.h, self.w, width)
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.mean_precision == "native":
                # [BT,D,H,W] with channels-last strides when x is contiguous.
                image = pixels.reshape(batch * self.t, self.h, self.w, width).permute(0, 3, 1, 2)
                pooled = F.avg_pool2d(image, kernel_size=self.block_hw, stride=self.block_hw,
                                      padding=0, ceil_mode=False, count_include_pad=False)
                pooled = pooled.permute(0, 2, 3, 1).reshape(batch, self.t, self.cells_per_time, width)
            else:
                # Reduce spatial subdimensions directly. This avoids first
                # materializing a full float32 copy of all input features.
                interior = pixels[:, :, :self.used_h, :self.used_w, :]
                cells = interior.reshape(batch, self.t, self.used_h // self.bh, self.bh,
                                         self.used_w // self.bw, self.bw, width)
                pooled = cells.mean(dim=(3, 5), dtype=torch.float32).to(x.dtype)
                pooled = pooled.reshape(batch, self.t, self.cells_per_time, width)
            parts = [pooled]
            if self.bottom_per_time:
                parts.append(pixels[:, :, self.used_h:, :, :].reshape(batch, self.t, self.bottom_per_time, width))
            if self.right_per_time:
                parts.append(pixels[:, :, :self.used_h, self.used_w:, :].reshape(batch, self.t, self.right_per_time, width))
            # Avoid even the concatenation when all cells are complete.
            result = pooled if len(parts) == 1 else torch.cat(parts, dim=2)
            shape = ((self.output_tokens, width) if x.ndim == 2 else
                     (batch, self.output_tokens, width))
            return result.reshape(shape), self.group_sizes


class StructuredPoolAdaptor(nn.Module):
    """One timed/capturable call: fixed mean pooling + existing frozen adaptor.

    The wrapped adaptor must implement adaptor(x[B,N,D], group_sizes=None).
    Uniform multiplicities add only a constant to each softmax row, so their
    correction can be cancelled exactly in real arithmetic. Nonuniform groups
    always pass actual sizes to the adaptor for log(group_size) correction.
    """

    def __init__(self, adaptor: nn.Module, grid_thw, *, block_hw=(1, 2),
                 mean_precision="native", elide_uniform_group_bias=True):
        super().__init__()
        self.adaptor = adaptor
        self.pool = StructuredMeanPool(grid_thw, block_hw=block_hw, mean_precision=mean_precision)
        self.elide_uniform_group_bias = bool(elide_uniform_group_bias)

    def prepare_for(self, x: Tensor) -> dict[str, Any]:
        self.pool.prepare_for(x)
        return self.metadata()

    def metadata(self) -> dict[str, Any]:
        skip_constant = self.elide_uniform_group_bias and self.pool.uniform_groups
        return {**self.pool.metadata(),
                "group_size_correction": "constant_cancelled_in_softmax" if skip_constant else "explicit_log_group_size",
                "forward_scope": "pool, dtype conversion, boundary concatenation, group-size correction and full adaptor",
                "target_visual_information": "unchanged; install only at Draft imadpt"}

    @torch.no_grad()
    def forward(self, x: Tensor, group_sizes: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("StructuredPoolAdaptor expects [B,N,D]")
        if group_sizes is not None:
            raise ValueError("input must contain original post-merger tokens, not already aggregated groups")
        pooled, sizes = self.pool(x)
        if self.elide_uniform_group_bias and self.pool.uniform_groups:
            sizes = None
        return self.adaptor(pooled, group_sizes=sizes)
