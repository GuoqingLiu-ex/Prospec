"""One fixed RMS-normalized prefix27 statistic; full-depth diagnosis only.

Use the model's actual merger.ln_q module and weights, in its native dtype.
RMSNorm does not subtract a mean. Applying it at prefix27 is a hypothesis about
a feature-error proxy, not an assertion that later residual layers erase scale
changes, and not a QA-risk predictor. No candidate Transformer block is probed.
"""
from __future__ import annotations

import hashlib
import math

import torch

try:
    from .prefix_energy import PrefixEnergyCollector, PRIMARY_SIGNAL as RAW_SIGNAL, compare_pair_errors, spearman
except ImportError:
    from prefix_energy import PrefixEnergyCollector, PRIMARY_SIGNAL as RAW_SIGNAL, compare_pair_errors, spearman

PRIMARY_SIGNAL = "max_window_merger_rmsnorm_relative_update"
SIGNAL_DEFINITION = ("max_window ||R(after27)-R(before27)||_F / "
                     "clamp(||R(before27)||_F,1e-6), R=actual visual.merger.ln_q")
SCOPE = "Final-feature-L2 proxy diagnosis only; no QA guarantee, deployment gate or latency claim"


def normalized_window_features(before, after, window_lengths, window_owners, time_lengths, eps=1e-6):
    """FP32 reference reductions of actual native-normalization outputs."""
    if (before.ndim != 2 or before.shape != after.shape or not before.numel() or
            not math.isfinite(eps) or eps <= 0 or
            not torch.isfinite(before).all() or not torch.isfinite(after).all()):
        raise ValueError("Expected matching finite normalization outputs and positive epsilon")
    wl, tl = window_lengths.long(), time_lengths.long()
    if (wl.ndim != 1 or tl.ndim != 1 or not wl.numel() or not tl.numel() or
            not torch.equal(wl, window_lengths) or not torch.equal(tl, time_lengths) or
            bool((wl <= 0).any()) or bool((tl <= 0).any()) or
            int(wl.sum()) != len(before) or int(tl.sum()) != len(before) or
            window_owners.dtype != torch.long or window_owners.shape != wl.shape or
            bool((window_owners < 0).any()) or bool((window_owners >= len(tl)).any()) or
            bool((window_owners[1:] < window_owners[:-1]).any()) or
            not torch.equal(torch.zeros_like(tl).scatter_add_(0, window_owners, wl), tl)):
        raise ValueError("Normalized feature layout differs from complete native windows")
    b, a = before.float(), after.float()
    residual = torch.segment_reduce((a-b).square().sum(-1), "sum", lengths=wl)
    reference = torch.segment_reduce(b.square().sum(-1), "sum", lengths=wl)
    scores = torch.sqrt(residual/reference.clamp_min(eps**2))
    maximum = scores.new_full((len(tl),), -float("inf"))
    maximum.scatter_reduce_(0, window_owners, scores, reduce="amax", include_self=True)
    return dict(window=dict(residual_energy=residual, reference_energy=reference, relative_l2=scores),
                time={PRIMARY_SIGNAL: maximum})


def rmsnorm_metadata(norm):
    if (type(norm).__name__ != "Qwen2_5_VLRMSNorm" or norm.weight.ndim != 1 or
            not math.isfinite(norm.variance_epsilon) or norm.variance_epsilon <= 0):
        raise ValueError("Require the actual native Qwen merger RMSNorm, not LayerNorm")
    weight = norm.weight.detach().cpu().contiguous()
    return dict(module="visual.merger.ln_q", class_name=type(norm).__name__,
                variance_epsilon=norm.variance_epsilon, weight_shape=list(weight.shape),
                weight_dtype=str(weight.dtype),
                weight_sha256=hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
                mean_subtraction=False, weights_modified=False,
                output_precision="Actual module/native dtype, then FP32 score accumulation")


class NormalizedPrefixCollector(PrefixEnergyCollector):
    """Retain original prefix statistics and add exactly one normalized signal."""
    def __init__(self, target):
        super().__init__(target, input_snapshot="clone", anchor_stride=4, eps=1e-6)
        self.normalizer = self.visual.merger.ln_q
        self.normalization_metadata = rmsnorm_metadata(self.normalizer)

    def _gate(self, before, after):
        super()._gate(before, after)
        # Separate outputs are temporary. Neither native stream is replaced.
        normalized_before = self.normalizer(before)
        normalized_after = self.normalizer(after)
        layout = self._layout
        self._current["normalized_prefix_features"] = normalized_window_features(
            normalized_before, normalized_after, layout.window_lengths,
            layout.window_owners, layout.time_lengths, self.eps)
        self._current.update(policy="normalized_prefix27_diagnostic", normalization=self.normalization_metadata,
            diagnostic_extra_transformer_calls=0, diagnostic_extra_rmsnorm_calls=2,
            normalized_input_dtype=str(before.dtype), normalized_output_dtype=str(normalized_before.dtype),
            two_norm_input_payload_bytes=2*before.numel()*before.element_size(),
            cost_note="Two extra full-feature normalization reads plus output writes and FP32 reduction scans; "
                      "payload bytes are a minimum input volume, not measured memory traffic. Original diagnostic statistics also remain.",
            proxy_scope=SCOPE)

    def summary(self):
        result = super().summary()
        result.update(policy="normalized_prefix27_diagnostic", normalization=self.normalization_metadata,
                      primary_signal=PRIMARY_SIGNAL, signal_definition=SIGNAL_DEFINITION,
                      raw_control_signal=RAW_SIGNAL, thresholds_fitted=False,
                      quality_status=SCOPE,
                      timing_status="Unfused diagnostic with two extra full-feature RMSNorm calls; no claim of negligible overhead")
        return result


def compare_normalized_pair_errors(records, pair_rows, raw_association):
    """Same fixed population and screen; no additional aggregation candidates."""
    raw_result = compare_pair_errors(records, pair_rows)  # Reuse grid/time/action alignment validation.
    current = {(str(r["video_id"]), r["fixed_depth"]): r for r in raw_result["observations"]
               if r["signal"] == RAW_SIGNAL}
    baseline = {}
    for row in raw_association["observations"]:
        if row["signal"] == RAW_SIGNAL:
            key = (str(row["video_id"]), row["fixed_depth"])
            if key in baseline:
                raise ValueError("Duplicate original prefix association")
            baseline[key] = row
    if set(current) != set(baseline):
        raise ValueError("Original prefix association video/depth coverage differs")
    observations, control = [], []
    for record in records:
        call = record["stats"]["calls"][0]
        values = call["normalized_prefix_features"]["time"][PRIMARY_SIGNAL]
        if (len(values) != len(call["actions"]) or
                any(not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError("Invalid normalized signal or temporal coverage")
        indices = [i for i, anchor in enumerate(call["anchors"]) if not anchor]
        selected = [values[i] for i in indices]
        for depth in (1, 2):
            key = (str(record["video_id"]), depth)
            raw, old = current[key], baseline[key]
            if (raw["final_feature_errors"] != old["final_feature_errors"] or
                    len(raw["values"]) != len(old["values"]) or
                    any(not math.isfinite(v) for v in old["values"])):
                raise ValueError("Original prefix reference errors or eligible blocks differ")
            rho = spearman(selected, raw["final_feature_errors"])
            observations.append(dict(video_id=key[0], fixed_depth=depth, signal=PRIMARY_SIGNAL,
                values=selected, final_feature_errors=raw["final_feature_errors"], spearman=rho,
                eligible_time_blocks=len(indices), reference_video_sortable=old["spearman"] is not None,
                original_raw_spearman=old["spearman"], current_raw_spearman=raw["spearman"],
                paired_delta_spearman=rho-old["spearman"] if rho is not None and old["spearman"] is not None else None))
            control.append(dict(video_id=key[0], fixed_depth=depth,
                max_absolute_raw_signal_difference=max((abs(a-b) for a,b in zip(raw["values"],old["values"])), default=0.),
                current_raw_values=raw["values"], original_raw_values=old["values"],
                current_raw_spearman=raw["spearman"], original_raw_spearman=old["spearman"]))
    summaries = []
    for depth in (1, 2):
        rows = [r for r in observations if r["fixed_depth"] == depth]
        population = [r for r in rows if r["reference_video_sortable"]]
        valid = [r["spearman"] for r in population if r["spearman"] is not None]
        delta = [r["paired_delta_spearman"] for r in population if r["paired_delta_spearman"] is not None]
        summaries.append(dict(fixed_depth=depth, signal=PRIMARY_SIGNAL,
            fixed_reference_video_denominator=len(population), sortable_videos=len(valid),
            positive_videos=sum(v > 0 for v in valid), macro_spearman=sum(valid)/len(valid) if valid else None,
            unrankable_reference_video_ids=[r["video_id"] for r in population if r["spearman"] is None],
            paired_delta_macro=sum(delta)/len(delta) if delta else None, paired_delta_videos=len(delta),
            pooled_spearman=spearman([v for r in rows for v in r["values"]],
                                     [v for r in rows for v in r["final_feature_errors"]]),
            pooled_blocks=sum(r["eligible_time_blocks"] for r in rows)))
    checks = [dict(fixed_depth=r["fixed_depth"],
                   positive_macro_spearman=r["macro_spearman"] is not None and r["macro_spearman"] > 0,
                   enough_positive_videos=r["fixed_reference_video_denominator"] > 0 and
                       r["positive_videos"]*3 >= 2*r["fixed_reference_video_denominator"])
              for r in summaries]
    passed = all(r["positive_macro_spearman"] and r["enough_positive_videos"] for r in checks)
    return dict(observations=observations, summaries=summaries, raw_control=control,
        normalization="Actual merger RMSNorm with learned weights; no mean subtraction",
        signal_definition=SIGNAL_DEFINITION, thresholds_fitted=False, aggregators_searched=False,
        registered_screen=dict(primary_signal=PRIMARY_SIGNAL, per_depth_checks=checks,
            criteria="Both depths: macro Spearman > 0 and >=2/3 positive in original sortable-video population; new constant scores remain in denominator",
            passes_descriptive_screen=passed,
            next_step="Only further task-quality and hardware validation could support a gate" if passed else
                      "Stop this normalization-signal branch; do not change aggregation or fit thresholds"),
        interpretation=SCOPE,
        pooled_limitation="Temporal blocks cluster within videos; pooled block counts are not independent samples",
        exclusions="First and periodic anchor4 timeblocks excluded; fewer than three eligible blocks or constant ranks yield null")
