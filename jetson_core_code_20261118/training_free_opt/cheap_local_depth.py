"""Same 0/1/2 local actions with less per-request Python and fewer allocations.

Does not change thresholds, window grouping, or the inherited block bypass.
window_origins stay on the policy object and are emitted in summary(), not
copied into the timed request record.
"""
from __future__ import annotations

import torch

from local_depth import LocalDepthPolicy


class CheapLocalDepthPolicy(LocalDepthPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._after_sums = None

    def _means_into(self, hidden: torch.Tensor) -> torch.Tensor:
        layout = self._layout
        n, dim = layout.window_lengths.numel(), hidden.shape[-1]
        buf = self._after_sums
        if buf is None or buf.shape != (n, dim) or buf.device != hidden.device or buf.dtype != torch.float32:
            buf = hidden.new_zeros((n, dim), dtype=torch.float32)
            self._after_sums = buf
        else:
            buf.zero_()
        buf.index_add_(0, self._token_windows, hidden.float())
        return buf / layout.window_lengths[:, None]

    def _start(self, hidden_states, grid_thw):
        thresholds = self.thresholds
        if thresholds is None:
            self.thresholds = {}
        try:
            super(LocalDepthPolicy, self)._start(hidden_states, grid_thw)
        finally:
            self.thresholds = thresholds
        self._current.update(
            policy="local_window_statistics_v1_cheap",
            collect_only=self.collect_only,
            protect_first=self.protect_first,
            windows_per_time=list(self._windows_per_time),
        )

    def _gate(self, before_mean, after):
        after_mean = self._means_into(after)
        before_norm = torch.linalg.vector_norm(before_mean, dim=-1)
        after_norm = torch.linalg.vector_norm(after_mean, dim=-1)
        update_windows = torch.linalg.vector_norm(after_mean - before_mean, dim=-1) / before_norm.clamp_min(1e-6)
        previous_mean = after_mean[self._previous_windows]
        previous_norm = after_norm[self._previous_windows]
        temporal_windows = torch.linalg.vector_norm(after_mean - previous_mean, dim=-1) / torch.maximum(
            after_norm, previous_norm).clamp_min(1e-6)
        finite_windows = torch.isfinite(update_windows) & torch.isfinite(temporal_windows)
        update = self._per_time_max(update_windows.masked_fill(~finite_windows, float("inf")))
        temporal = self._per_time_max(temporal_windows.masked_fill(~finite_windows, float("inf")))
        finite = torch.isfinite(update) & torch.isfinite(temporal)
        self._current["actions"] = self._decide(update, temporal).masked_fill(~finite, 0).detach()
        self._current["features"] = {
            "max_window_relative_update": update.detach(),
            "max_window_temporal_change": temporal.detach(),
            "finite": finite.detach(), "anchor": self._layout.anchors,
        }

    def summary(self):
        result = super().summary()
        result.update(
            policy="local_window_statistics_v1_cheap",
            window_origins=list(self._window_origins or []),
            window_origin_fields=["clip", "temporal_block", "patch_y", "patch_x"],
        )
        return result
