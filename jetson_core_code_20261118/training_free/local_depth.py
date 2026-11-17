"""Training-free, window-local statistics for Qwen2.5-VL visual depth.

The decision uses only block 27's input/output. Blocks 28/29 are the only
candidates; native attention, RoPE, packing, and output handling are inherited
from DepthPolicy. A temporal block is a tubelet (two frames for this model).

Windows are the *native* fixed spatial attention windows, including smaller
edge windows. Transformers 5.17 get_vision_window_index orders nonempty windows
by clip, time, window row, window column. permute_input_for_window_attn keeps
each spatial merge unit together. This order persists through all blocks.
Thus DepthPolicy's nonempty window_lengths define contiguous summary groups,
and the same window ordinal in adjacent times denotes the same spatial area.

These are representation heuristics, not estimates or guarantees of QA risk.
Window means reduce cancellation *between* windows, but can still hide changes
inside a window. No learned parameters or additional model forward is used.

Use collect_only=True without thresholds to collect full-depth dev statistics.
Calibrate on distinct development videos, then pass frozen thresholds. Include
prepare(), feature reductions and inherited packing in the request interval;
call summary() only after stopping that interval. This is not sync-free CUDA:
the parent policy uses dynamic nonzero and the native math-SDPA path may sync.
"""
from __future__ import annotations

import math

import torch

try:
    from ..vision_depth import DepthPolicy
except ImportError:
    from vision_depth import DepthPolicy


THRESHOLD_KEYS = {"update1", "update2", "temporal1", "temporal2"}


def _validate_thresholds(thresholds):
    if set(thresholds) != THRESHOLD_KEYS:
        raise ValueError(f"Expected exactly these thresholds: {sorted(THRESHOLD_KEYS)}")
    values = {key: float(value) for key, value in thresholds.items()}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Local-statistics thresholds must be finite and nonnegative")
    if values["update2"] > values["update1"] or values["temporal2"] > values["temporal1"]:
        raise ValueError("Skip-two thresholds must not exceed skip-one thresholds")
    return values


class LocalDepthPolicy(DepthPolicy):
    """One decision point; reuse the parent's real block-bypass implementation."""

    def __init__(self, target, thresholds=None, anchor_stride=4, collect_only=False,
                 protect_first=True, skip_layers=(28, 29), prefix_layer=27):
        if tuple(skip_layers) != (28, 29) or prefix_layer != 27:
            raise ValueError("This experiment fixes prefix 27 and skip layers 28/29")
        if int(anchor_stride) != anchor_stride or anchor_stride < 0:
            raise ValueError("anchor_stride must be a nonnegative integer")
        if thresholds is None and not collect_only:
            raise ValueError("Supply calibrated thresholds or use collect_only=True")
        super().__init__(target, mode="adaptive", anchor_stride=anchor_stride,
                         skip_layers=skip_layers, prefix_layer=prefix_layer)
        # Parent _start only records this dictionary. _decide/_gate below give
        # its new keys their own meaning; parent cosine thresholds are unused.
        self.thresholds = None if thresholds is None else _validate_thresholds(thresholds)
        self.collect_only = bool(collect_only)
        self.protect_first = bool(protect_first)
        self._local_layout = None
        self._token_windows = None
        self._previous_windows = None
        self._window_origins = None
        self._windows_per_time = None

    def prepare(self, grid_thw, device=None):
        super().prepare(grid_thw, device)
        layout = self._layout
        if self._local_layout is layout:
            return self
        merge = int(self.visual.spatial_merge_size)
        span = int(self.visual.window_size) // merge // int(self.visual.patch_size)
        previous, origins, counts, anchors = [], [], [], []
        for clip, (times, height, width) in enumerate(layout.grid):
            spatial = [(y * merge, x * merge)
                       for y in range(0, height // merge, span)
                       for x in range(0, width // merge, span)]
            count = len(spatial)
            for time in range(times):
                start = len(previous)
                previous.extend(start + i if time == 0 else start + i - count
                                for i in range(count))
                origins.extend((clip, time, y, x) for y, x in spatial)
                counts.append(count)
                anchors.append((self.protect_first and time == 0) or
                               (time > 0 and self.anchor_stride > 0 and
                                time % self.anchor_stride == 0))
        if len(previous) != layout.window_lengths.numel():
            raise ValueError("Native window metadata and local summary groups disagree")
        device = layout.window_lengths.device
        self._token_windows = torch.repeat_interleave(
            torch.arange(len(previous), device=device), layout.window_lengths.long(),
            output_size=layout.token_count)
        self._previous_windows = torch.tensor(previous, device=device, dtype=torch.long)
        layout.anchors = torch.tensor(anchors, device=device, dtype=torch.bool)
        self._window_origins = origins
        self._windows_per_time = counts
        self._local_layout = layout
        return self

    def _start(self, hidden_states, grid_thw):
        # The parent records dict(thresholds). Empty is intentional in collect
        # mode; scoring never reads thresholds unless actions are requested.
        thresholds = self.thresholds
        if thresholds is None:
            self.thresholds = {}
        try:
            super()._start(hidden_states, grid_thw)
        finally:
            self.thresholds = thresholds
        self._current.update(
            policy="local_window_statistics_v1", collect_only=self.collect_only,
            protect_first=self.protect_first, windows_per_time=list(self._windows_per_time),
            window_origins=list(self._window_origins),
            window_origin_fields=["clip", "temporal_block", "patch_y", "patch_x"])

    def _means(self, hidden):
        """Override *both* prefix summaries: return [native_windows, channels]."""
        layout = self._layout
        if hidden.ndim != 2 or hidden.shape[0] != layout.token_count:
            raise ValueError("Expected all native window-ordered visual tokens")
        sums = torch.zeros((layout.window_lengths.numel(), hidden.shape[-1]),
                           dtype=torch.float32, device=hidden.device)
        sums.index_add_(0, self._token_windows, hidden.float())
        return sums / layout.window_lengths[:, None]

    def _per_time_max(self, window_scores):
        result = window_scores.new_full((self._layout.time_count,), -float("inf"))
        return result.scatter_reduce_(0, self._layout.window_owners, window_scores,
                                      reduce="amax", include_self=True)

    def _decide(self, update, temporal):
        if self.collect_only:
            return torch.zeros_like(update, dtype=torch.long)
        q = self.thresholds
        one = (update <= q["update1"]) & (temporal <= q["temporal1"])
        two = (update <= q["update2"]) & (temporal <= q["temporal2"])
        return (one.long() + two.long()).masked_fill(self._layout.anchors, 0)

    def _gate(self, before_mean, after):
        after_mean = self._means(after)
        before_norm = torch.linalg.vector_norm(before_mean, dim=-1)
        after_norm = torch.linalg.vector_norm(after_mean, dim=-1)
        update_windows = torch.linalg.vector_norm(after_mean - before_mean, dim=-1) / before_norm.clamp_min(1e-6)
        previous_mean = after_mean[self._previous_windows]
        previous_norm = after_norm[self._previous_windows]
        # Symmetric relative temporal distance includes direction and magnitude.
        # The first time of each clip compares with itself, and is protected by
        # default. No correspondence is formed across clip boundaries.
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

    def _selected(self, layer, original, hidden_states, args, kwargs):
        if self.collect_only:
            return original(hidden_states, *args, **kwargs)
        return super()._selected(layer, original, hidden_states, args, kwargs)

    def summary(self):
        result = super().summary()
        result.update(policy="local_window_statistics_v1", trained=False,
                      collect_only=self.collect_only,
                      quality_status="representation_heuristic_not_QA_risk")
        # Strict JSON has no NaN/Infinity. Preserve invalidity through `finite`
        # and emit null scores; invalid time blocks always take action zero.
        for record in result["calls"]:
            for key in ("max_window_relative_update", "max_window_temporal_change"):
                if key in record["features"]:
                    record["features"][key] = [value if math.isfinite(value) else None
                                                for value in record["features"][key]]
        return result


def local_depth_policy(target, thresholds=None, **kwargs):
    return LocalDepthPolicy(target, thresholds=thresholds, **kwargs)


def calibrate_local_thresholds(records):
    """Four quantile thresholds from >=2 distinct development videos.

    Input: one {video_id, features} per video, with features from one full-depth
    collect_only call. Repeated videos are rejected to avoid overweighting one
    video via multiple questions. Only finite, nonanchor temporal blocks count.
    q75 permits skip-one; q50 permits skip-two, with both statistics required.
    These marginal quantiles do not promise a particular joint skip budget or
    task accuracy. Return value is directly accepted by LocalDepthPolicy.
    """
    videos, contributing, updates, temporals = set(), set(), [], []
    for record in records:
        video = str(record["video_id"])
        if video in videos:
            raise ValueError("Supply one feature record per distinct development video")
        videos.add(video)
        features = record["features"]
        a = features["max_window_relative_update"]
        b = features["max_window_temporal_change"]
        anchors = features["anchor"]
        if len(a) != len(b) or len(a) != len(anchors):
            raise ValueError("Per-time scores and anchors must have matching lengths")
        for update, temporal, anchor in zip(a, b, anchors):
            if anchor or update is None or temporal is None:
                continue
            if math.isfinite(update) and math.isfinite(temporal):
                if update < 0 or temporal < 0:
                    raise ValueError("Relative distances cannot be negative")
                updates.append(float(update))
                temporals.append(float(temporal))
                contributing.add(video)
    if len(contributing) < 2:
        raise ValueError("Need finite nonanchor scores from >=2 distinct development videos")

    def quantile(values, q):
        values = sorted(values)
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {"update1": quantile(updates, .75), "update2": quantile(updates, .5),
            "temporal1": quantile(temporals, .75), "temporal2": quantile(temporals, .5)}
