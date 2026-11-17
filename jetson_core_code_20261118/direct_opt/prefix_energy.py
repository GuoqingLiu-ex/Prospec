"""Full-depth prefix27 diagnostics; no new gate, thresholds, or bypass actions.

Statistics are a deliberately unfused FP32 reference. Holding the native block
input by reference is permitted only after audit_native_prefix verifies the
inspected Transformers source. Unit tests can request an explicit clone.
Neither collection cost nor signal correlation establishes deployed gate speed
or QA-risk prediction.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import torch

try:
    from ..training_free.local_depth import LocalDepthPolicy
except ImportError:
    from local_depth import LocalDepthPolicy

AUDITED_MODEL_SHA256 = "507cb2c383ead0e0e22b16ec9c8af2eb82703843cd5849112715751d990ff15e"
PRIMARY_SIGNAL = "max_window_residual_energy"
SECONDARY_SIGNAL = "max_token_relative_update"
CORRELATION_SIGNALS = (PRIMARY_SIGNAL, SECONDARY_SIGNAL, "window_mean_update_max")
SIGNAL_DEFINITIONS = {
    PRIMARY_SIGNAL: "max_window sqrt(sum_token ||after-before||^2 / clamp(sum_token ||before||^2, eps^2))",
    SECONDARY_SIGNAL: "max_token ||after-before|| / clamp(||before||, eps)",
    "window_mean_update_max": "max_window ||mean(after)-mean(before)|| / clamp(||mean(before)||, eps)",
}


def audit_native_prefix(visual):
    """Verify the exact source inspected before retaining a native input alias."""
    block = visual.blocks[27]
    if type(block).__name__ != "Qwen2_5_VLVisionBlock" or "forward" in block.__dict__:
        raise ValueError("Expected the unpatched native Qwen2.5-VL vision block")
    path = Path(inspect.getsourcefile(type(block))).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != AUDITED_MODEL_SHA256 or type(block.norm1).__name__ != "Qwen2_5_VLRMSNorm":
        raise ValueError("Native source differs from the inspected non-inplace implementation")
    if "forward" in block.norm1.__dict__:
        raise ValueError("Prefix input normalization was patched")
    return dict(path=str(path), sha256=digest, block_forward_line=355, rmsnorm_forward_line=72,
                input_snapshot="reference", verified_no_inplace_input_update=True,
                evidence="RMSNorm forms new tensors; block updates use hidden_states = hidden_states + ...")


def energy_features(before, after, window_lengths, window_owners, time_lengths,
                    previous_windows, eps=1e-6):
    """Return token/window/time tensors, preserving the native window ordering."""
    if (before.ndim != 2 or before.shape != after.shape or not before.numel() or
            not math.isfinite(eps) or eps <= 0):
        raise ValueError("Expected matching nonempty [tokens,channels] and a positive epsilon")
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("Nonfinite prefix representation")
    b, a = before.float(), after.float()
    delta = a - b
    residual = delta.square().sum(-1)
    input_energy = b.square().sum(-1)
    token_ratio = torch.sqrt(residual / input_energy.clamp_min(eps**2))
    wl, tl = window_lengths.long(), time_lengths.long()
    if (wl.ndim != 1 or tl.ndim != 1 or not wl.numel() or not tl.numel() or
            not torch.equal(wl, window_lengths) or not torch.equal(tl, time_lengths) or
            bool((wl <= 0).any()) or bool((tl <= 0).any()) or
            int(wl.sum()) != before.shape[0] or int(tl.sum()) != before.shape[0]):
        raise ValueError("Segment lengths do not cover the input")
    if (window_owners.shape != wl.shape or previous_windows.shape != wl.shape or
            window_owners.dtype != torch.long or previous_windows.dtype != torch.long or
            bool((window_owners < 0).any()) or bool((window_owners >= tl.numel()).any()) or
            bool((previous_windows < 0).any()) or bool((previous_windows >= wl.numel()).any())):
        raise ValueError("Invalid window owners or temporal correspondence")
    # Windows and temporal blocks are contiguous in the same native ordering.
    owner_lengths = torch.zeros_like(tl).scatter_add_(0, window_owners, wl)
    if not torch.equal(owner_lengths, tl) or bool((window_owners[1:] < window_owners[:-1]).any()):
        raise ValueError("Window ownership disagrees with temporal segments")
    window_residual = torch.segment_reduce(residual, "sum", lengths=wl)
    window_input = torch.segment_reduce(input_energy, "sum", lengths=wl)
    window_ratio = torch.sqrt(window_residual / window_input.clamp_min(eps**2))
    bm = torch.segment_reduce(b, "mean", lengths=wl)
    am = torch.segment_reduce(a, "mean", lengths=wl)
    mean_update = torch.linalg.vector_norm(am-bm, dim=-1) / torch.linalg.vector_norm(bm, dim=-1).clamp_min(eps)
    previous = am[previous_windows]
    temporal = torch.linalg.vector_norm(am-previous, dim=-1) / torch.maximum(
        torch.linalg.vector_norm(am, dim=-1), torch.linalg.vector_norm(previous, dim=-1)).clamp_min(eps)
    # Energy decomposition: sum ||delta_i||² = n||mean(delta)||² + centered energy.
    coherent_fraction = (wl * (am-bm).square().sum(-1) / window_residual.clamp_min(eps**2)).clamp(0., 1.)
    cancellation = torch.where(window_residual > eps**2, 1-coherent_fraction, torch.zeros_like(coherent_fraction))

    def window_max(values):
        out = values.new_full((tl.numel(),), -float("inf"))
        return out.scatter_reduce_(0, window_owners, values, reduce="amax", include_self=True)

    time_residual = torch.segment_reduce(residual, "sum", lengths=tl)
    time_input = torch.segment_reduce(input_energy, "sum", lengths=tl)
    token_peak = torch.segment_reduce(residual, "max", lengths=tl)
    max_window = window_max(window_ratio)
    max_token = torch.segment_reduce(token_ratio, "max", lengths=tl)
    return {
        "token": {"residual_energy": residual, "input_energy": input_energy,
                  "relative_residual": token_ratio, "residual_rms": torch.sqrt(residual / b.shape[1]),
                  "input_norm_clamped": input_energy < eps**2},
        "window": {"residual_energy": window_residual, "input_energy": window_input,
                   "energy_ratio": window_ratio, "mean_relative_update": mean_update,
                   "temporal_relative_change": temporal, "cancellation_fraction": cancellation},
        "time": {PRIMARY_SIGNAL: max_window, SECONDARY_SIGNAL: max_token,
                 "window_energy_max": max_window, "token_ratio_max": max_token,
                 "residual_energy_ratio": torch.sqrt(time_residual / time_input.clamp_min(eps**2)),
                 "max_token_energy_fraction": token_peak / time_residual.clamp_min(eps**2),
                 "max_window_energy_fraction": window_max(window_residual) / time_residual.clamp_min(eps**2),
                 "window_mean_update_max": window_max(mean_update),
                 "window_temporal_change_max": window_max(temporal),
                 "window_cancellation_max": window_max(cancellation)},
    }


class PrefixEnergyCollector(LocalDepthPolicy):
    def __init__(self, target, *, input_snapshot="clone", anchor_stride=4, eps=1e-6):
        super().__init__(target, collect_only=True, anchor_stride=anchor_stride)
        if input_snapshot not in {"reference", "clone"}:
            raise ValueError("input_snapshot must be reference or clone")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.source_audit = audit_native_prefix(self.visual) if input_snapshot == "reference" else None
        self.input_snapshot, self.eps = input_snapshot, eps

    def _means(self, hidden):
        # Parent prefix hook passes this value straight to our _gate. In this
        # collector it is an input snapshot, not the old window mean.
        return hidden.detach() if self.input_snapshot == "reference" else hidden.detach().clone()

    def _gate(self, before, after):
        layout = self._layout
        self._current["features"] = energy_features(before, after, layout.window_lengths,
            layout.window_owners, layout.time_lengths, self._previous_windows, self.eps)
        self._current["actions"] = (0,) * layout.time_count
        self._current.update(policy="prefix27_energy_diagnostic", input_snapshot=self.input_snapshot,
                             norm_clamp_min=self.eps, source_audit=self.source_audit,
                             time_lengths=layout.time_lengths, window_lengths=layout.window_lengths,
                             window_owners=layout.window_owners, anchors=layout.anchors,
                             time_origins=[(clip, time) for clip, (times, _, _) in enumerate(layout.grid)
                                           for time in range(times)])

    def summary(self):
        result = super().summary()
        result.update(policy="prefix27_energy_diagnostic", diagnostic_only=True,
                      signal_definitions=SIGNAL_DEFINITIONS,
                      ordering="Native nonempty spatial-window ordering; time blocks remain contiguous",
                      quality_status="No QA-risk prediction; all visual blocks executed",
                      timing_status="Unfused FP32 diagnostic; no deployed gate cost claim")
        return result


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, start = [0.] * len(values), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for j in range(start, end):
            result[order[j]] = (start + end - 1) / 2
        start = end
    return result


def spearman(x, y):
    if any(not math.isfinite(v) for v in [*x, *y]):
        raise ValueError("Spearman inputs must be finite")
    if len(x) != len(y) or len(x) < 3:
        return None
    a, b = _ranks(x), _ranks(y)
    am, bm = sum(a)/len(a), sum(b)/len(b)
    numerator = sum((u-am)*(v-bm) for u, v in zip(a, b))
    denominator = math.sqrt(sum((u-am)**2 for u in a) * sum((v-bm)**2 for v in b))
    return numerator/denominator if denominator else None


def compare_pair_errors(prefix_records, pair_rows):
    """Descriptive within-video association only; never select a cutoff here."""
    refs = {}
    for row in pair_rows:
        if row["skip_layers"] == [28, 29] and row["fixed_depth"] in (1, 2):
            key = (str(row["video_id"]), row["fixed_depth"])
            if key in refs:
                raise ValueError("Duplicate pair12 video/depth observation")
            refs[key] = row
    observations = []
    videos = [str(r["video_id"]) for r in prefix_records]
    if len(videos) != len(set(videos)) or set(refs) != {(v, d) for v in videos for d in (1, 2)}:
        raise ValueError("Require the same distinct videos with both fixed-depth observations")
    for prefix in prefix_records:
        if len(prefix["stats"]["calls"]) != 1:
            raise ValueError("Require one visual call per video")
        call = prefix["stats"]["calls"][0]
        if any(call["actions"]) or not call.get("completed", False):
            raise ValueError("Prefix signals must come from full-depth collection")
        origins = [(clip, time) for clip, (times, _, _) in enumerate(prefix["video_grid_thw"])
                   for time in range(times)]
        if (len(call["anchors"]) != len(call["actions"]) or len(origins) != len(call["actions"]) or
                [list(v) for v in call["time_origins"]] != [list(v) for v in origins]):
            raise ValueError("Temporal metadata differs from prefix count/order")
        indices = [i for i, anchor in enumerate(call["anchors"]) if not anchor]
        for depth in (1, 2):
            row = refs[(str(prefix["video_id"]), depth)]
            if row["video_grid_thw"] != prefix["video_grid_thw"] or row["timestamps"] != prefix["timestamps"]:
                raise ValueError("Pair12 and prefix capture input grids/timestamps differ")
            errors = row["features_vs_native"]["temporal_blocks"]
            if (len(errors) != len(call["actions"]) or
                    [(e["clip"], e["temporal_block"]) for e in errors] != origins or
                    any(not math.isfinite(e["relative_l2"]) or e["relative_l2"] < 0 for e in errors)):
                raise ValueError("Temporal feature errors differ from prefix count/order or are invalid")
            target = [errors[i]["relative_l2"] for i in indices]
            for signal in CORRELATION_SIGNALS:
                if len(call["features"]["time"][signal]) != len(origins):
                    raise ValueError("Signal length differs from temporal count")
                values = [call["features"]["time"][signal][i] for i in indices]
                observations.append(dict(video_id=str(prefix["video_id"]), fixed_depth=depth,
                    signal=signal, eligible_time_blocks=len(indices),
                    spearman=spearman(values, target), values=values, final_feature_errors=target))
    summaries = []
    for signal in CORRELATION_SIGNALS:
        for depth in (1, 2):
            valid = [r["spearman"] for r in observations if r["signal"] == signal and
                     r["fixed_depth"] == depth and r["spearman"] is not None]
            selected = [r for r in observations if r["signal"] == signal and r["fixed_depth"] == depth]
            summaries.append(dict(signal=signal, fixed_depth=depth, valid_videos=len(valid),
                positive_videos=sum(v > 0 for v in valid),
                macro_spearman=sum(valid)/len(valid) if valid else None,
                pooled_spearman=spearman([v for r in selected for v in r["values"]],
                                         [v for r in selected for v in r["final_feature_errors"]]),
                pooled_block_count=sum(r["eligible_time_blocks"] for r in selected)))
    primary = [r for r in summaries if r["signal"] == PRIMARY_SIGNAL]
    passed = all(r["valid_videos"] > 0 and r["macro_spearman"] > 0 and
                 r["positive_videos"] * 3 >= r["valid_videos"] * 2 for r in primary)
    return dict(observations=observations, summaries=summaries, thresholds_fitted=False,
                signal_definitions=SIGNAL_DEFINITIONS,
                registered_screen={"primary_signal": PRIMARY_SIGNAL,
                    "criteria": {"require_positive_macro_spearman_for_both_depths": True,
                                 "minimum_positive_fraction_of_sortable_videos_each_depth": 2/3},
                    "per_depth_checks": [{"fixed_depth": r["fixed_depth"],
                        "positive_macro_spearman": r["valid_videos"] > 0 and r["macro_spearman"] > 0,
                        "enough_positive_videos": r["valid_videos"] > 0 and
                            r["positive_videos"] * 3 >= r["valid_videos"] * 2} for r in primary],
                    "passes_descriptive_screen": passed,
                    "next_step": "Actual mixed-allocation validation is permitted" if passed else
                                 "Stop this signal branch; do not fit new thresholds"},
                interpretation="Descriptive association with feature error, not causal QA risk or gate acceptance",
                pooled_limitation="Descriptive only; temporal blocks are clustered by video and are not independent samples",
                exclusions="Anchors excluded; fewer than three eligible blocks or constant ranks yield null correlation")
