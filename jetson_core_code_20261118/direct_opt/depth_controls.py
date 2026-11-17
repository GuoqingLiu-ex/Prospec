"""Same-budget allocation controls that retain the actual depth controller.

Run the same prefix statistics and 0/1/2 gate as DirectDepthPolicy, then assign
its per-clip budget uniformly or randomly among finite, unprotected tubelets.
First tubelets and periodic anchors always retain full depth. Random allocation
preserves the per-clip action histogram; uniform allocation spreads the budget
as evenly as possible (all eligible blocks get one before any gets two).

Both candidate layers 28/29 use identical local-attention matrix shapes within
a clip. Preserving each clip's sum of skipped depth therefore preserves its
theoretical matrix MACs, including when different clips have different sizes.
Packing, kernel efficiency, controller cost, and quality can still differ.
This is an allocation ablation, not a free or input-independent baseline.

Include construction, prepare(), statistics, and allocation in wall timing.
Call summary() after timing. Use a video ID as allocation_key, and freeze seed
across repeats; separate seeds are separate controls, not timing repetitions.
"""
from __future__ import annotations

import hashlib
import math
import random

try:
    from .depth import DirectDepthPolicy
except ImportError:
    from depth import DirectDepthPolicy


def allocate_actions(actions, clip_lengths, eligible, *, allocation="uniform",
                     seed=20261002, allocation_key=None):
    """Pure CPU allocator; match calibrate_depth's documented control rules."""
    actions, clip_lengths, eligible = tuple(actions), tuple(clip_lengths), tuple(eligible)
    if allocation not in {"uniform", "random"}:
        raise ValueError("allocation must be uniform or random")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if allocation == "random" and (not isinstance(allocation_key, str) or not allocation_key):
        raise ValueError("Random control requires a stable nonempty allocation_key, such as video_id")
    if (not clip_lengths or any(type(n) is not int or n <= 0 for n in clip_lengths) or
            sum(clip_lengths) != len(actions) or len(eligible) != len(actions)):
        raise ValueError("Clip lengths, actions, and eligibility disagree")
    if any(type(a) is not int or a not in (0, 1, 2) for a in actions):
        raise ValueError("Actions must be integers 0/1/2")
    if any(type(ok) is not bool or (not ok and a != 0) for a, ok in zip(actions, eligible)):
        raise ValueError("Protected or invalid tubelets must have zero adaptive action")
    result, offset = [0] * len(actions), 0
    for clip, length in enumerate(clip_lengths):
        if eligible[offset]:
            raise ValueError("The first tubelet of every clip must be protected")
        indices = [i for i in range(offset, offset + length) if eligible[i]]
        offset += length
        if not indices:
            continue
        values = [actions[i] for i in indices]
        if allocation == "uniform":
            base, remainder = divmod(sum(values), len(indices))
            values = [base] * len(indices)
            for j in range(remainder):
                values[((2 * j + 1) * len(indices)) // (2 * remainder)] += 1
        else:
            key = f"{seed}:{allocation_key}:{clip}".encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
            rng.shuffle(values)
        for i, value in zip(indices, values):
            result[i] = value
    return tuple(result)


class BudgetMatchedDepthPolicy(DirectDepthPolicy):
    """Reuse the real gate and all grouped execution; change only allocation."""

    def __init__(self, target, thresholds=None, *, allocation="uniform", seed=20261002,
                 allocation_key=None, **kwargs):
        if kwargs.get("protect_first", True) is not True:
            raise ValueError("Budget controls always protect each clip's first tubelet")
        # Validate allocator settings before installing any forward hooks.
        allocate_actions((0,), (1,), (False,), allocation=allocation, seed=seed,
                         allocation_key=allocation_key)
        super().__init__(target, thresholds=thresholds, **kwargs)
        self.allocation = allocation
        self.allocation_seed = seed
        self.allocation_key = allocation_key

    def _start(self, hidden_states, grid_thw):
        super()._start(hidden_states, grid_thw)
        self._current.update(policy="direct_depth_budget_control_v1", allocation=self.allocation,
                             allocation_seed=self.allocation_seed,
                             allocation_key=self.allocation_key,
                             adaptive_actions=None, allocated_actions=None,
                             gate_retained=True, allocation_applied=False)

    def _record_allocation(self, adaptive, allocated, eligible):
        clips, offset = [], 0
        for times, height, width in self._layout.grid:
            a = sum(adaptive[offset:offset + times])
            b = sum(allocated[offset:offset + times])
            if a != b:
                raise AssertionError("Allocation changed a clip's skipped-depth budget")
            clips.append(dict(time_blocks=times, eligible_time_blocks=(
                                  sum(eligible[offset:offset + times]) if eligible is not None else None),
                              adaptive_skipped_depth=a, allocated_skipped_depth=b,
                              skipped_token_blocks=a * height * width))
            offset += times
        self._current.update(adaptive_actions=adaptive, allocated_actions=allocated,
                             allocation_changed=adaptive != allocated, allocation_clips=clips,
                             per_clip_budget_preserved=True,
                             skipped_token_blocks=sum(c["skipped_token_blocks"] for c in clips))

    def _actions_from_scores(self, scores):
        adaptive = super()._actions_from_scores(scores)
        eligible = tuple(not anchor and math.isfinite(update) and math.isfinite(temporal)
                         for (update, temporal), anchor in zip(scores, self._cpu_anchors))
        allocated = allocate_actions(adaptive, [row[0] for row in self._layout.grid], eligible,
                                     allocation=self.allocation, seed=self.allocation_seed,
                                     allocation_key=self.allocation_key)
        self._record_allocation(adaptive, allocated, eligible)
        self._current["allocation_applied"] = True
        return allocated

    def _gate(self, before_mean, after):
        super()._gate(before_mean, after)
        if self.collect_only:
            # Direct collection bypasses the CPU decision; do not add a host
            # transfer merely to populate diagnostic fields.
            actions = self._current["actions"]
            self._record_allocation(actions, actions, None)

    def summary(self):
        result = super().summary()
        result.update(policy="direct_depth_budget_control_v1", allocation=self.allocation,
                      allocation_seed=self.allocation_seed, allocation_key=self.allocation_key,
                      control_scope="Same online gate and per-clip budget; allocation ablation",
                      quality_status="Allocation_quality_unvalidated",
                      cost_scope="Per-clip local-layer matrix MACs matched; latency/packing may differ")
        return result


def depth_control_policy(target, thresholds=None, mode="adaptive", **kwargs):
    """Context-manager interface; pass allocation='uniform' or 'random'."""
    if mode != "adaptive":
        raise ValueError("Budget controls require the adaptive gate")
    return BudgetMatchedDepthPolicy(target, thresholds=thresholds, **kwargs)
