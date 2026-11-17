"""One-decision, two-block execution for training-free Qwen visual depth.

The local-window rule is unchanged in real arithmetic: prefix 27 predicts
actions 0/1/2 for blocks 28/29. This is a scheduling candidate, not a claim of
improved QA accuracy. Both layers remain optional only in local attention.

At the single decision point, two scores per temporal block move to the CPU.
The CPU chooses the action and groups time blocks in action order 0,1,2.
Block 28 runs on the action-0 prefix; block 29 on the action-0/1 prefix.
Native order is restored before block 30. Each block is invoked at most once.
There is no per-token Python loop or dynamic CUDA nonzero. Native attention
may still synchronize. All-zero decisions call both original blocks directly.

The default window reduction is a contiguous segment mean. Its reduction
order can differ from index_add; use summary_backend='index_add' for the
original statistics path. Frozen thresholds and real QA checks are required.

Usage (include construction/prepare in the timed wall interval)::
    with direct_depth_policy(target, thresholds=frozen_thresholds) as policy:
        policy.prepare(packed['video_grid_thw'], device=target.model.visual.device)
        result = target(**packed)
    stats = policy.summary()  # outside timing

For full-depth development statistics, use collect_only=True with no thresholds.
calibrate_local_thresholds accepts these feature records without conversion.
"""
from __future__ import annotations

from itertools import accumulate
import math

import torch

try:
    from ..training_free.local_depth import LocalDepthPolicy, calibrate_local_thresholds
    from ..vision_depth import DepthPolicy
except ImportError:
    from local_depth import LocalDepthPolicy, calibrate_local_thresholds
    from vision_depth import DepthPolicy


class DirectDepthPolicy(LocalDepthPolicy):
    def __init__(self, target, thresholds=None, *, anchor_stride=4,
                 collect_only=False, protect_first=True, summary_backend="segment",
                 skip_layers=(28, 29), prefix_layer=27):
        super().__init__(target, thresholds=thresholds, anchor_stride=anchor_stride,
                         collect_only=collect_only, protect_first=protect_first,
                         skip_layers=skip_layers, prefix_layer=prefix_layer)
        if summary_backend not in {"segment", "index_add"}:
            raise ValueError("summary_backend must be segment or index_add")
        self.summary_backend = summary_backend
        self._direct_layout = None
        self._route = None

    def prepare(self, grid_thw, device=None):
        super().prepare(grid_thw, device)
        if self._direct_layout is self._layout:
            return self
        self._direct_layout = self._layout
        span = self.visual.window_size // self.visual.spatial_merge_size // self.visual.patch_size
        merge = self.visual.spatial_merge_size
        time_lengths, windows, anchors = [], [], []
        for times, height, width in self._layout.grid:
            hh, ww = height // merge, width // merge
            lengths = tuple(min(span, hh-y) * min(span, ww-x) * merge**2
                            for y in range(0, hh, span) for x in range(0, ww, span))
            for time in range(times):
                time_lengths.append(height * width)
                windows.append(lengths)
                anchors.append((self.protect_first and time == 0) or
                               (time > 0 and self.anchor_stride > 0 and time % self.anchor_stride == 0))
        self._time_lengths = tuple(time_lengths)
        self._time_offsets = (0, *accumulate(time_lengths))
        self._time_windows = tuple(windows)
        self._cpu_anchors = tuple(anchors)
        # Metadata was validated against the native layout at preparation time.
        self._segment_lengths = self._layout.window_lengths.long()
        return self

    def _start(self, hidden_states, grid_thw):
        thresholds = self.thresholds
        if thresholds is None:
            self.thresholds = {}
        try:
            DepthPolicy._start(self, hidden_states, grid_thw)
        finally:
            self.thresholds = thresholds
        self._route = None
        self._current.update(
            policy="direct_local_depth_v1", statistics_backend=self.summary_backend,
            collect_only=self.collect_only, protect_first=self.protect_first,
            windows_per_time=self._windows_per_time, window_origins=self._window_origins,
            window_origin_fields=["clip", "temporal_block", "patch_y", "patch_x"],
            decision_host_transfers=0, time_permutations=0, route="pending")

    def _means(self, hidden):
        if self.summary_backend == "index_add":
            return super()._means(hidden)
        if hidden.ndim != 2 or hidden.shape[0] != self._layout.token_count:
            raise ValueError("Expected all native window-ordered visual tokens")
        # unsafe avoids revalidating GPU lengths through host synchronization;
        # preparation supplies positive native lengths whose sum is token_count.
        return torch.segment_reduce(hidden.float(), "mean", lengths=self._segment_lengths,
                                    axis=0, unsafe=True)

    def _actions_from_scores(self, scores):
        q = self.thresholds
        actions = []
        for (update, temporal), anchor in zip(scores, self._cpu_anchors):
            if anchor or not (math.isfinite(update) and math.isfinite(temporal)):
                actions.append(0)
            else:
                one = update <= q["update1"] and temporal <= q["temporal1"]
                two = update <= q["update2"] and temporal <= q["temporal2"]
                actions.append(int(one) + int(two))
        return tuple(actions)

    def _gate(self, before_mean, after):
        after_mean = self._means(after)
        before_norm = torch.linalg.vector_norm(before_mean, dim=-1)
        after_norm = torch.linalg.vector_norm(after_mean, dim=-1)
        update = torch.linalg.vector_norm(after_mean-before_mean, dim=-1) / before_norm.clamp_min(1e-6)
        previous = after_mean[self._previous_windows]
        temporal = torch.linalg.vector_norm(after_mean-previous, dim=-1) / torch.maximum(
            after_norm, after_norm[self._previous_windows]).clamp_min(1e-6)
        # Reduce both statistics together; no read of candidate/future layers.
        window_scores = torch.stack((update, temporal), dim=1)
        finite_windows = torch.isfinite(window_scores).all(dim=1)
        window_scores = window_scores.masked_fill(~finite_windows[:, None], float("inf"))
        scores = window_scores.new_full((self._layout.time_count, 2), -float("inf"))
        scores.scatter_reduce_(0, self._layout.window_owners[:, None].expand(-1, 2),
                               window_scores, reduce="amax", include_self=True)
        if self.collect_only:
            actions = (0,) * self._layout.time_count
        else:
            cpu_scores = scores.detach().cpu().tolist()  # one decision synchronization
            self._current["decision_host_transfers"] = int(after.is_cuda)
            actions = self._actions_from_scores(cpu_scores)
        self._current["actions"] = actions
        self._current["features"] = {
            "max_window_relative_update": scores[:, 0].detach(),
            "max_window_temporal_change": scores[:, 1].detach(),
            "finite": torch.isfinite(scores).all(dim=1).detach(),
            "anchor": self._layout.anchors,
        }
        self._route = self._make_route(actions, after.device)

    def _make_route(self, actions, device):
        if len(actions) != len(self._time_lengths) or any(a not in (0, 1, 2) for a in actions):
            raise ValueError("Require one 0/1/2 action per time block")
        unique = set(actions)
        if len(unique) == 1:
            self._current["route"] = f"uniform{actions[0]}"
            return {"uniform": actions[0]}
        order = tuple(t for action in (0, 1, 2) for t, a in enumerate(actions) if a == action)
        inverse = tuple(order.index(t) for t in range(len(order)))
        lengths = tuple(self._time_lengths[t] for t in order)
        n0 = sum(n for n, t in zip(lengths, order) if actions[t] == 0)
        n01 = sum(n for n, t in zip(lengths, order) if actions[t] < 2)
        win0 = sum(len(self._time_windows[t]) for t in order if actions[t] == 0)
        kept_windows = [n for t in order if actions[t] < 2 for n in self._time_windows[t]]
        cu01 = torch.tensor((0, *accumulate(kept_windows)), dtype=torch.int32, device=device)
        equal_lengths = len(set(self._time_lengths)) == 1
        reordered = order != tuple(range(len(order)))
        indices = (torch.tensor((order, inverse), dtype=torch.long, device=device)
                   if reordered and equal_lengths else None)
        self._current["route"] = "grouped012"
        self._current["time_permutations"] = 2 if reordered else 0
        return dict(uniform=None, order=order, inverse=inverse, lengths=lengths,
                    reordered=reordered, indices=indices, n0=n0, n01=n01,
                    cu0=cu01[:win0+1], cu01=cu01,
                    grouped_offsets=(0, *accumulate(lengths)), positions=None)

    def _permute(self, tensor, inverse=False):
        route = self._route
        if not route["reordered"]:
            return tensor
        order = route["inverse"] if inverse else route["order"]
        if route["indices"] is not None:
            n = len(order)
            return tensor.reshape(n, self._time_lengths[0], *tensor.shape[1:]).index_select(
                0, route["indices"][int(inverse)]).reshape_as(tensor)
        # Multi-clip fallback: variable-sized time blocks remain contiguous.
        offsets = route["grouped_offsets"] if inverse else self._time_offsets
        return torch.cat([tensor[offsets[t]:offsets[t+1]] for t in order], dim=0)

    def _selected(self, layer, original, hidden_states, args, kwargs):
        route = self._route
        if route is None:
            raise RuntimeError("Prefix decision was not executed")
        rank = self.skip_layers.index(layer) + 1
        uniform = route["uniform"]
        if uniform is not None:
            if uniform >= rank:
                self._current["executed_token_blocks"] -= hidden_states.shape[0]
                self._current["executed_tokens_per_layer"][layer] = 0
                return hidden_states
            return original(hidden_states, *args, **kwargs)
        if args:
            raise TypeError("Expected native Qwen cu_seqlens as keyword")
        if rank == 1:
            positions = kwargs.get("position_embeddings")
            if positions is None or len(positions) != 2:
                raise ValueError("Expected Qwen cosine/sine position_embeddings")
            route["positions"] = tuple(self._permute(p) for p in positions)
            # Own the storage before writing updated prefixes. Input/recorded
            # tensors are not mutated even when the route order is identity.
            hidden_states = self._permute(hidden_states)
            if not route["reordered"]:
                hidden_states = hidden_states.clone()
        elif route["positions"] is None:
            raise RuntimeError("Adjacent candidate blocks must execute in order")
        kept = route["n0"] if rank == 1 else route["n01"]
        self._current["executed_token_blocks"] -= hidden_states.shape[0] - kept
        self._current["executed_tokens_per_layer"][layer] = kept
        if kept:
            selected = dict(kwargs, cu_seqlens=route["cu0"] if rank == 1 else route["cu01"],
                            position_embeddings=tuple(p[:kept] for p in route["positions"]))
            updated = original(hidden_states[:kept], **selected)
            hidden_states[:kept].copy_(updated)
        return self._permute(hidden_states, inverse=True) if rank == 2 else hidden_states

    def summary(self):
        result = super().summary()
        result.update(policy="direct_local_depth_v1", trained=False,
                      statistics_backend=self.summary_backend,
                      quality_status="same_local_rule_QA_improvement_unproven")
        return result


def direct_depth_policy(target, thresholds=None, mode="adaptive", **kwargs):
    """Context-manager interface matching the full-request runner."""
    if mode == "adaptive":
        return DirectDepthPolicy(target, thresholds=thresholds, **kwargs)
    if mode in {"fixed0", "fixed1", "fixed2"}:
        for key in ("collect_only", "protect_first", "summary_backend"):
            kwargs.pop(key, None)
        return DepthPolicy(target, mode=mode, skip_layers=kwargs.pop("skip_layers", (28, 29)),
                           prefix_layer=kwargs.pop("prefix_layer", 27), **kwargs)
    raise ValueError("mode must be adaptive, fixed0, fixed1, or fixed2")
