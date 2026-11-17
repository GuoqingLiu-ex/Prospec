"""Full-depth diagnosis using two predeclared complete-window scouts.

At prefix27, select one complete native window per temporal block by geometric
center or greatest normalized prefix residual energy. On each packed selection
execute B28(x), B29(B28(x)), and B29(x). These correspond to action0, action1
(skip28), and action2 (identity). Original full-window execution is retained;
scout outputs never replace model activations. There is no deployed gate here.
"""
from __future__ import annotations

from contextlib import ExitStack
from itertools import accumulate
import math

import torch

try:
    from ..training_free.local_depth import LocalDepthPolicy
    from ..compute_cost import calculate_vision_cost
except ImportError:
    from local_depth import LocalDepthPolicy
    from compute_cost import calculate_vision_cost
try:
    from .prefix_energy import spearman
    from .scout_precision import packed_downproj_accumulation
except ImportError:
    from prefix_energy import spearman
    from scout_precision import packed_downproj_accumulation

STRATEGIES = ("geometric_center", "prefix_energy_max")
DEFAULT_RTOL = 5e-3
DEFAULT_ATOL = 5e-3
DEFAULT_RELATIVE_L2 = 1e-3
VALIDATION_KEYS = ("native_block28_validation", "native_block29_validation",
                   "counterfactual_block29_validation")
DIAGNOSTIC_SCOPE = ("Candidate-signal diagnosis only; original numerical checks remain unchanged. "
                    "Full-window reference scores are diagnostic, not a deployed gate or QA guarantee.")


def choose_windows(grid, origins, lengths, owners, energy):
    """Stable row-major ties; geometry uses actual clipped window centers."""
    if not (len(origins) == len(lengths) == len(owners) == len(energy)):
        raise ValueError("Window metadata lengths differ")
    if any(not math.isfinite(v) or v < 0 for v in energy):
        raise ValueError("Prefix window energies must be finite and nonnegative")
    times = sum(row[0] for row in grid)
    groups = [[i for i, owner in enumerate(owners) if owner == time] for time in range(times)]
    if any(not group for group in groups):
        raise ValueError("Every temporal block must have a complete native window")
    # Window starts determine each actual edge rectangle. Only spatially aligned
    # native windows from LocalDepthPolicy are supplied; no token subsampling.
    shapes = []
    for clip, time, y, x in origins:
        _, height, width = grid[clip]
        same = [p for p in origins if p[0] == clip and p[1] == time]
        next_y = min([p[2] for p in same if p[2] > y] + [height])
        next_x = min([p[3] for p in same if p[3] > x] + [width])
        shapes.append((next_y-y, next_x-x))
    if any(h*w != length for (h, w), length in zip(shapes, lengths)):
        raise ValueError("Native window rectangles and token lengths disagree")
    result = {name: [] for name in STRATEGIES}
    for group in groups:
        def distance(index):
            clip, _, y, x = origins[index]
            _, height, width = grid[clip]
            h, w = shapes[index]
            return (y+h/2-height/2)**2 + (x+w/2-width/2)**2
        result["geometric_center"].append(min(group, key=lambda i: (distance(i), i)))
        result["prefix_energy_max"].append(max(group, key=lambda i: (energy[i], -i)))
    return result, shapes


def pack_windows(hidden, position_embeddings, window_lengths, selected_windows, cu_dtype=torch.int32):
    """Gather complete contiguous native windows and their original RoPE rows."""
    if (hidden.ndim != 2 or sum(window_lengths) != hidden.shape[0] or
            not window_lengths or any(type(n) is not int or n <= 0 for n in window_lengths)):
        raise ValueError("Invalid native hidden/window dimensions")
    if (len(set(selected_windows)) != len(selected_windows) or not selected_windows or
            any(type(i) is not int or i < 0 or i >= len(window_lengths) for i in selected_windows)):
        raise ValueError("Require distinct valid selected windows")
    if position_embeddings is None or len(position_embeddings) != 2 or any(
            p.shape[0] != hidden.shape[0] for p in position_embeddings):
        raise ValueError("Expected original per-token cosine/sine rows")
    offsets = [0, *accumulate(window_lengths)]
    indices = torch.cat([torch.arange(offsets[i], offsets[i+1], device=hidden.device)
                         for i in selected_windows])
    lengths = [window_lengths[i] for i in selected_windows]
    cumulative = torch.tensor([0, *accumulate(lengths)], device=hidden.device, dtype=cu_dtype)
    positions = tuple(p.index_select(0, indices) for p in position_embeddings)
    return hidden.index_select(0, indices), positions, cumulative, indices, lengths


def window_errors(candidate, reference, lengths, eps=1e-12):
    """FP64 CPU per-window diagnostics; one selected window represents one time."""
    a, b = (x.detach().to(device="cpu", dtype=torch.float64) for x in (candidate, reference))
    if (a.ndim != 2 or a.shape != b.shape or sum(lengths) != len(a) or
            any(n <= 0 for n in lengths) or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise ValueError("Invalid or nonfinite window outputs")
    result, start = [], 0
    for length in lengths:
        x, y = a[start:start+length], b[start:start+length]
        delta = x-y
        norms = torch.linalg.vector_norm(y, dim=-1)
        token_error = torch.linalg.vector_norm(delta, dim=-1)
        result.append(dict(relative_l2=float(torch.linalg.vector_norm(delta) /
                                              torch.linalg.vector_norm(y).clamp_min(eps)),
                           max_token_relative_l2=float((token_error/norms.clamp_min(eps)).max()),
                           max_token_absolute_l2=float(token_error.max()),
                           max_absolute_element_error=float(delta.abs().max()),
                           zero_reference_token_norms=int((norms < eps).sum())))
        start += length
    return result


def validate_native_branch(probe, reference, lengths, *, rtol, atol, relative_l2_limit):
    diagnostics = window_errors(probe, reference, lengths)
    a, b = (x.detach().to(device="cpu", dtype=torch.float64) for x in (probe, reference))
    start = 0
    for row, length in zip(diagnostics, lengths):
        row["allclose"] = bool(torch.allclose(a[start:start+length], b[start:start+length],
                                             rtol=rtol, atol=atol))
        row["passed"] = row["allclose"] and row["relative_l2"] <= relative_l2_limit
        start += length
    return diagnostics


def validate_packed_layout(hidden, positions, lengths, selected, packed):
    """Structural errors remain fatal even when numerical mismatches may continue."""
    x, pos, cu, indices, packed_lengths = packed
    offsets = [0, *accumulate(lengths)]
    expected_indices = torch.cat([torch.arange(offsets[i], offsets[i+1], device=hidden.device)
                                  for i in selected])
    expected_lengths = [lengths[i] for i in selected]
    if (packed_lengths != expected_lengths or not torch.equal(indices, expected_indices) or
            cu.tolist() != [0, *accumulate(expected_lengths)] or
            not torch.equal(x, hidden.index_select(0, expected_indices)) or
            len(pos) != len(positions) or any(not torch.equal(p, original.index_select(0, expected_indices))
                                             for p, original in zip(pos, positions))):
        raise ValueError("Packed window layout, activations or RoPE rows differ from native selection")


def compare_action_scores(packed, native, anchors, origins):
    """Actual local-score discrepancies, ranks and safe halves, excluding anchors."""
    eligible = [i for i, anchor in enumerate(anchors) if not anchor]
    result = {}
    for action in ("1", "2"):
        if len(packed[action]) != len(origins) or len(native[action]) != len(origins):
            raise ValueError("Local score comparison does not cover all time blocks")
        result[action] = {}
        for signal in ("relative_l2", "max_token_relative_l2"):
            p = [row[signal] for row in packed[action]]
            n = [row[signal] for row in native[action]]
            if any(not math.isfinite(v) or v < 0 for v in p+n):
                raise ValueError("Nonfinite or negative local action score")
            delta = [abs(a-b) for a, b in zip(p, n)]
            p_order = sorted(eligible, key=lambda i: (p[i], i))
            n_order = sorted(eligible, key=lambda i: (n[i], i))
            sortable = len(eligible) >= 2 and len({p[i] for i in eligible}) > 1 and len({n[i] for i in eligible}) > 1
            half = len(eligible)//2
            safe_p, safe_n = p_order[:half], n_order[:half]
            sign = lambda x: (x > 0)-(x < 0)
            pairs = [(sign(p[i]-p[j]), sign(n[i]-n[j])) for at, i in enumerate(eligible)
                     for j in eligible[at+1:]]
            result[action][signal] = dict(packed_values=p, selected_native_values=n,
                absolute_score_errors=delta, max_absolute_score_error=max(delta),
                mean_absolute_score_error=sum(delta)/len(delta), eligible_indices=eligible,
                eligible_spearman=spearman([p[i] for i in eligible], [n[i] for i in eligible]),
                pairwise_comparisons=len(pairs), pairwise_order_changes=sum(a != b for a, b in pairs),
                pairwise_direction_reversals=sum(a*b < 0 for a, b in pairs),
                pairwise_tie_changes=sum((a == 0) != (b == 0) for a, b in pairs),
                packed_ascending_time_blocks=[list(origins[i]) for i in p_order],
                selected_native_ascending_time_blocks=[list(origins[i]) for i in n_order],
                packed_safest_half_time_blocks=[list(origins[i]) for i in safe_p] if sortable else [],
                selected_native_safest_half_time_blocks=[list(origins[i]) for i in safe_n] if sortable else [],
                safest_half_sets_equal=set(safe_p) == set(safe_n) if sortable else None,
                safe_half_comparable=sortable,
                tie_rule="Temporal index ascending; half=floor(n/2). Constant scores or fewer than 2 eligible blocks give no safe-half comparison.",
                scope=DIAGNOSTIC_SCOPE)
    return result


def scout_mac_accounting(grid, config, selected_lengths, anchors):
    """Dense-matrix counts and optimistic savings bound, never a speed estimate."""
    native = calculate_vision_cost(grid, config, skip_layers=(28, 29))
    spec, blocks = native["vision_config"], native["temporal_blocks"]
    if len(selected_lengths) != len(blocks) or len(anchors) != len(blocks):
        raise ValueError("Selected windows must cover all temporal blocks")
    d, f = spec["hidden_size"], spec["intermediate_size"]
    def local(lengths):
        return sum(lengths)*(4*d*d+3*d*f)+2*d*sum(n*n for n in lengths)
    per_time = []
    for block, length, anchor in zip(blocks, selected_lengths, anchors):
        if length not in block["window_lengths"]:
            raise ValueError("Selected length is not a complete native window")
        whole, sampled = local(block["window_lengths"]), local([length])
        per_time.append(dict(local_layer_macs=whole, selected_window_layer_macs=sampled,
                             window_mac_fraction=sampled/whole, window_token_fraction=length/block["tokens"],
                             diagnostic_extra_three_calls_macs=3*sampled, anchor=bool(anchor)))
    full = native["full"]["vision_major_matrix_macs"]
    fixed1 = full-sum(row["local_layer_macs"] for row in per_time)
    # Optimistic deployed case only: do not probe protected anchors and send
    # every eligible time block to action2. No probe output is reusable then.
    best = full-sum(2*r["local_layer_macs"]-3*r["selected_window_layer_macs"]
                   for r in per_time if not r["anchor"])
    extra = sum(r["diagnostic_extra_three_calls_macs"] for r in per_time)
    return dict(per_time=per_time, native_vision_major_macs=full,
                diagnostic_extra_three_calls_macs=extra,
                diagnostic_extra_fraction_of_native=extra/full,
                fixed1_vision_major_macs=fixed1,
                ideal_all_eligible_skip2_vision_major_macs=best,
                ideal_saved_macs_upper_bound_vs_fixed1=fixed1-best,
                ideal_saved_fraction_upper_bound_vs_fixed1=(fixed1-best)/fixed1,
                convention="1 MAC = 2 FLOPs", excluded=native["excluded"],
                limitation="Ideal matrix-operation bound only; excludes statistics, packing, synchronization and kernel efficiency; no latency claim")


class NativeWindowScout(LocalDepthPolicy):
    """Side-effect-free full-depth collector with genuine three-call probes."""
    def __init__(self, target, *, anchor_stride=4, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
                 relative_l2_limit=DEFAULT_RELATIVE_L2, packed_downproj_fp32_accum=False,
                 diagnostic_allow_numerical_mismatch=False):
        super().__init__(target, collect_only=True, anchor_stride=anchor_stride)
        if type(packed_downproj_fp32_accum) is not bool:
            raise ValueError("packed_downproj_fp32_accum must be boolean")
        self.packed_downproj_fp32_accum = packed_downproj_fp32_accum
        if type(diagnostic_allow_numerical_mismatch) is not bool:
            raise ValueError("diagnostic_allow_numerical_mismatch must be boolean")
        self.diagnostic_allow_numerical_mismatch = diagnostic_allow_numerical_mismatch
        for value in (rtol, atol, relative_l2_limit):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Numerical validation tolerances must be finite and nonnegative")
        self.rtol, self.atol, self.relative_l2_limit = rtol, atol, relative_l2_limit
        self._probe_buffers = {}
        self._full_prefix_x = None
        self._full_counterfactual_y1 = None

    def __enter__(self):
        self._native29 = self.visual.blocks[29].forward
        return super().__enter__()

    def __exit__(self, *args):
        self._probe_buffers.clear()
        self._full_prefix_x = self._full_counterfactual_y1 = None
        return super().__exit__(*args)

    def _means(self, hidden):
        return hidden.detach().clone()

    def _gate(self, before, after):
        b, a = before.float(), after.float()
        lengths = self._layout.window_lengths.long()
        numerator = torch.segment_reduce((a-b).square().sum(-1), "sum", lengths=lengths)
        denominator = torch.segment_reduce(b.square().sum(-1), "sum", lengths=lengths)
        energy = torch.sqrt(numerator/denominator.clamp_min(1e-12)).detach().cpu().tolist()
        self._lengths = lengths.cpu().tolist()
        owners = self._layout.window_owners.cpu().tolist()
        self._selections, shapes = choose_windows(self._layout.grid, self._window_origins,
                                                  self._lengths, owners, energy)
        self._probe_buffers = {}
        self._current.update(actions=(0,)*self._layout.time_count,
            policy="native_window_scout_diagnostic", anchors=self._layout.anchors,
            packed_downproj_fp32_accum=self.packed_downproj_fp32_accum,
            diagnostic_allow_numerical_mismatch=self.diagnostic_allow_numerical_mismatch,
            strict_validation_passed=None, numerical_validation_failures=[], validation_scope=DIAGNOSTIC_SCOPE,
            time_origins=[(clip, time) for clip, (times, _, _) in enumerate(self._layout.grid)
                          for time in range(times)],
            prefix_window_energy=energy,
            prefix_window_input_energy=denominator.detach().cpu().tolist(),
            window_lengths=self._lengths, window_owners=owners,
            prefix_energy_definition="sqrt(sum_token ||prefix27_after-before||^2 / clamp(sum_token ||before||^2,1e-12))",
            scout={}, validation_tolerances=dict(rtol=self.rtol, atol=self.atol,
                                                 relative_l2_limit=self.relative_l2_limit))
        for strategy, windows in self._selections.items():
            self._current["scout"][strategy] = dict(selected_windows=[dict(
                global_window_index=i, clip=self._window_origins[i][0],
                temporal_block=self._window_origins[i][1], patch_y=self._window_origins[i][2],
                patch_x=self._window_origins[i][3], patch_height=shapes[i][0], patch_width=shapes[i][1],
                tokens=self._lengths[i], area_fraction=self._lengths[i]/int(self._layout.time_lengths[time]),
                prefix_window_energy=energy[i]) for time, i in enumerate(windows)],
                actual_extra_block_calls=0)

    def _record_validation(self, strategy, key, rows, message):
        self._current["scout"][strategy][key] = rows
        if len(rows) != self._layout.time_count:
            raise ValueError("Numerical validation coverage differs from temporal layout")
        failed = [i for i, row in enumerate(rows) if not row["passed"]]
        if failed:
            self._current["strict_validation_passed"] = False
            self._current["numerical_validation_failures"].append(
                dict(strategy=strategy, check=key, failed_temporal_indices=failed))
            if not self.diagnostic_allow_numerical_mismatch:
                raise ValueError(message)

    def _selected(self, layer, original, hidden_states, args, kwargs):
        if layer not in (28, 29):
            raise ValueError("Scout only supports the declared local blocks28/29")
        if args:
            raise TypeError("Expected native Qwen block metadata as keywords")
        expected_cu = torch.tensor([0, *accumulate(self._lengths)], device=hidden_states.device,
                                   dtype=kwargs["cu_seqlens"].dtype)
        if not torch.equal(expected_cu, kwargs["cu_seqlens"]):
            raise ValueError("Candidate block is not using the expected complete native windows")
        if layer == 28:
            # One full-window counterfactual for *diagnosis only*. This is an
            # additional entire local layer, separate from any deployed scout.
            self._full_prefix_x = hidden_states.detach().clone()
            counter_flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            self._full_counterfactual_y1 = self._native29(self._full_prefix_x.clone(), **kwargs)
            self._current["full_window_counterfactual"] = dict(path="B29(x)",
                actual_extra_block_calls=1, tokens=len(hidden_states), diagnostic_only=True,
                reduction_flag_before=counter_flag,
                reduction_flag_after=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
            if self._current["full_window_counterfactual"]["reduction_flag_after"] != counter_flag:
                raise AssertionError("Full counterfactual changed the reduction flag")
            for strategy, windows in self._selections.items():
                x, pos, cu, indices, lengths = pack_windows(hidden_states, kwargs.get("position_embeddings"),
                                                           self._lengths, windows, expected_cu.dtype)
                validate_packed_layout(hidden_states, kwargs["position_embeddings"], self._lengths, windows,
                                       (x, pos, cu, indices, lengths))
                packed_kwargs = dict(kwargs, cu_seqlens=cu, position_embeddings=pos)
                # Original max_seqlen is a valid upper bound for this subset.
                record = self._current["scout"][strategy]
                precision = dict(enabled=self.packed_downproj_fp32_accum,
                    global_flag_before_probes=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                record["packed_precision"] = precision
                try:
                    with ExitStack() as stack:
                        if self.packed_downproj_fp32_accum:
                            for block_number in (28, 29):
                                precision[f"block{block_number}"] = stack.enter_context(
                                    packed_downproj_accumulation(self.visual.blocks[block_number], label=f"block{block_number}"))
                        z28 = original(x.clone(), **packed_kwargs)
                        record["actual_extra_block_calls"] += 1
                        y0 = self._native29(z28.clone(), **packed_kwargs)
                        record["actual_extra_block_calls"] += 1
                        y1 = self._native29(x.clone(), **packed_kwargs)
                        record["actual_extra_block_calls"] += 1
                finally:
                    precision["global_flag_after_probes"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
                if (record["actual_extra_block_calls"] != 3 or
                        precision["global_flag_before_probes"] != precision["global_flag_after_probes"]):
                    raise AssertionError("Unexpected packed call count or failed reduction-flag restoration")
                if self.packed_downproj_fp32_accum:
                    for block_number, expected_calls in ((28, 1), (29, 2)):
                        trace = precision[f"block{block_number}"]
                        if (len(trace["down_proj_calls"]) != expected_calls or not trace["forward_restored"] or
                                trace["entry_flag"] != trace["exit_flag"] or any(
                                    row["flag_before"] != row["flag_after"] or row["flag_during"] is not False
                                    for row in trace["down_proj_calls"])):
                            raise AssertionError("Unexpected packed down_proj calls or failed forward restoration")
                record.update(packed_tokens=len(x), packed_cu_seqlens=cu,
                    action_paths={"0": "B29(B28(x))", "1": "B29(x)", "2": "x"},
                    action_errors={"1": window_errors(y1, y0, lengths),
                                   "2": window_errors(x, y0, lengths)})
                self._probe_buffers[strategy] = dict(indices=indices, lengths=lengths, z28=z28, y0=y0)
                counter_reference = self._full_counterfactual_y1.index_select(0, indices)
                counter_check = validate_native_branch(y1, counter_reference, lengths, rtol=self.rtol,
                    atol=self.atol, relative_l2_limit=self.relative_l2_limit)
                self._record_validation(strategy, "counterfactual_block29_validation", counter_check,
                    f"{strategy} packed B29(x) disagrees with full-window counterfactual")
        main_flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        output = original(hidden_states, **kwargs)
        self._current.setdefault("native_block_reduction_flags", {})[str(layer)] = dict(
            before=main_flag, after=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        if self._current["native_block_reduction_flags"][str(layer)]["after"] != main_flag:
            raise AssertionError("Native main path changed the reduction flag")
        for strategy, buffer in self._probe_buffers.items():
            reference = output.index_select(0, buffer["indices"])
            probe = buffer["z28"] if layer == 28 else buffer["y0"]
            diagnostics = validate_native_branch(probe, reference, buffer["lengths"], rtol=self.rtol,
                                                 atol=self.atol, relative_l2_limit=self.relative_l2_limit)
            self._record_validation(strategy, f"native_block{layer}_validation", diagnostics,
                f"{strategy} packed scout disagrees with native block{layer} reference")
        if layer == 29:
            full_errors = {"1": window_errors(self._full_counterfactual_y1, output, self._lengths),
                           "2": window_errors(self._full_prefix_x, output, self._lengths)}
            self._current["full_window_action_errors"] = full_errors
            owners = self._current["window_owners"]
            for strategy, selected in self._selections.items():
                unselected = {}
                for action, errors in full_errors.items():
                    unselected[action] = []
                    for time, window in enumerate(selected):
                        remaining = [i for i, owner in enumerate(owners) if owner == time and i != window]
                        worst = max(remaining, key=lambda i: (errors[i]["relative_l2"], -i)) if remaining else None
                        unselected[action].append(dict(unselected_windows=len(remaining),
                            worst_unselected_window_index=worst,
                            max_window_relative_l2=errors[worst]["relative_l2"] if worst is not None else None,
                            max_token_relative_l2=max(errors[i]["max_token_relative_l2"] for i in remaining)
                                                  if remaining else None))
                record = self._current["scout"][strategy]
                record["unselected_action_errors"] = unselected
                record["selected_native_action_errors"] = {
                    action: [errors[i] for i in selected] for action, errors in full_errors.items()}
                record["packed_vs_selected_native_scores"] = compare_action_scores(
                    record["action_errors"], record["selected_native_action_errors"],
                    self._current["anchors"], self._current["time_origins"])
                record["strict_validation_passed"] = all(
                    row["passed"] for key in VALIDATION_KEYS for row in record[key])
            self._current["strict_validation_passed"] = all(
                row["strict_validation_passed"] for row in self._current["scout"].values())
            self._probe_buffers.clear()
            self._full_prefix_x = self._full_counterfactual_y1 = None
        return output

    def summary(self):
        result = super().summary()
        result.update(policy="native_window_scout_diagnostic", diagnostic_only=True,
                      strategies=list(STRATEGIES),
                      packed_downproj_fp32_accum=self.packed_downproj_fp32_accum,
                      diagnostic_allow_numerical_mismatch=self.diagnostic_allow_numerical_mismatch,
                      strict_validation_passed=bool(result["calls"]) and all(
                          call.get("completed") and call.get("strict_validation_passed") is True
                          for call in result["calls"]), validation_scope=DIAGNOSTIC_SCOPE,
                      quality_status="Local candidate-layer response is not QA risk",
                      timing_status="Three real block probes per selection, one full B29(x) diagnostic and complete native vision; no timing claim",
                      token_block_accounting="Parent counts the unchanged main path only; extra scout calls are recorded separately")
        return result


def associate_scout_errors(records, pair_rows, prefix_association, *, diagnostic_allow_numerical_mismatch=False):
    """All predeclared variants; no threshold fitting or winner selection."""
    if type(diagnostic_allow_numerical_mismatch) is not bool:
        raise ValueError("diagnostic_allow_numerical_mismatch must be boolean")
    refs = {}
    for row in pair_rows:
        if row["skip_layers"] == [28, 29] and row["fixed_depth"] in (1, 2):
            key = (str(row["video_id"]), row["fixed_depth"])
            if key in refs:
                raise ValueError("Duplicate pair12 observation")
            refs[key] = row
    videos = [str(row["video_id"]) for row in records]
    if len(set(videos)) != len(videos) or set(refs) != {(v, d) for v in videos for d in (1, 2)}:
        raise ValueError("Require matching unique videos and both fixed-depth references")
    baselines = {}
    for row in prefix_association["observations"]:
        if row["signal"] == "max_window_residual_energy":
            key = (str(row["video_id"]), row["fixed_depth"])
            if key in baselines:
                raise ValueError("Duplicate prefix baseline association")
            baselines[key] = row
    if set(baselines) != set(refs):
        raise ValueError("Prefix baseline video/depth coverage differs from scout")
    observations, blind_spots, numerical_failures = [], [], []
    for record in records:
        if len(record["stats"]["calls"]) != 1:
            raise ValueError("Require one full-depth visual call per video")
        call = record["stats"]["calls"][0]
        origins = [(clip, time) for clip, (times, _, _) in enumerate(record["video_grid_thw"])
                   for time in range(times)]
        if (not call["completed"] or any(call["actions"]) or len(call["anchors"]) != len(origins) or
                len(call["actions"]) != len(origins) or [list(v) for v in origins] != call["time_origins"]):
            raise ValueError("Invalid full-depth temporal metadata")
        eligible = [i for i, anchor in enumerate(call["anchors"]) if not anchor]
        for depth in (1, 2):
            reference = refs[(str(record["video_id"]), depth)]
            errors = reference["features_vs_native"]["temporal_blocks"]
            if (reference["video_grid_thw"] != record["video_grid_thw"] or
                    reference["timestamps"] != record["timestamps"] or
                    [(e["clip"], e["temporal_block"]) for e in errors] != origins or
                    any(not math.isfinite(e["relative_l2"]) or e["relative_l2"] < 0 for e in errors)):
                raise ValueError("Pair12 errors and scout inputs are not aligned")
            target = [errors[i]["relative_l2"] for i in eligible]
            baseline = baselines[(str(record["video_id"]), depth)]
            if baseline["final_feature_errors"] != target:
                raise ValueError("Prefix baseline uses different final errors or eligible time blocks")
            for strategy in STRATEGIES:
                scout = call["scout"][strategy]
                if any(len(scout[key]) != len(origins) for key in VALIDATION_KEYS):
                    raise ValueError("Native scout validation coverage differs from layout")
                strict_passed = all(row["passed"] for key in VALIDATION_KEYS for row in scout[key])
                if not strict_passed:
                    if not diagnostic_allow_numerical_mismatch:
                        raise ValueError("Native scout validation failed")
                    if (call.get("diagnostic_allow_numerical_mismatch") is not True or
                            call.get("strict_validation_passed") is not False or
                            "selected_native_action_errors" not in scout or
                            "packed_vs_selected_native_scores" not in scout):
                        raise ValueError("Failed numerical rows require an explicitly completed diagnostic collection")
                    if depth == 1:
                        numerical_failures.append(dict(video_id=str(record["video_id"]), strategy=strategy,
                            failed_checks={key: [i for i, row in enumerate(scout[key]) if not row["passed"]]
                                           for key in VALIDATION_KEYS}))
                scores = scout["action_errors"][str(depth)]
                native_scores = scout.get("selected_native_action_errors", {}).get(str(depth))
                unselected = scout["unselected_action_errors"][str(depth)]
                selected = scout["selected_windows"]
                if (len(scores) != len(origins) or len(unselected) != len(origins) or
                        [(row["clip"], row["temporal_block"]) for row in selected] != origins or
                        (native_scores is not None and len(native_scores) != len(origins))):
                    raise ValueError("Scout errors do not cover all time blocks")
                if any(not math.isfinite(row[signal]) or row[signal] < 0
                       for group in (scores, native_scores or []) for row in group
                       for signal in ("relative_l2", "max_token_relative_l2")) or any(
                           row["max_window_relative_l2"] is not None and
                           (not math.isfinite(row["max_window_relative_l2"]) or row["max_window_relative_l2"] < 0)
                           for row in unselected):
                    raise ValueError("Nonfinite or negative scout action error")
                for signal in ("relative_l2", "max_token_relative_l2"):
                    values = [scores[i][signal] for i in eligible]
                    native_values = [native_scores[i][signal] for i in eligible] if native_scores is not None else None
                    rho = spearman(values, target)
                    unselected_indices = [i for i in eligible if unselected[i]["max_window_relative_l2"] is not None]
                    remaining_target = [unselected[i]["max_window_relative_l2"] for i in unselected_indices]
                    remaining_scores = [scores[i][signal] for i in unselected_indices]
                    baseline_rho = baseline["spearman"]
                    observations.append(dict(video_id=str(record["video_id"]), strategy=strategy,
                        fixed_depth=depth, signal=signal, eligible_time_blocks=len(eligible),
                        strict_validation_passed=strict_passed, scope=DIAGNOSTIC_SCOPE,
                        values=values, final_feature_errors=target, spearman=rho,
                        selected_native_values=native_values,
                        selected_native_spearman=spearman(native_values, target) if native_values is not None else None,
                        prefix_energy_spearman=baseline_rho,
                        reference_video_sortable=baseline_rho is not None,
                        paired_delta_spearman=rho-baseline_rho if rho is not None and baseline_rho is not None else None,
                        unselected_eligible_time_blocks=len(unselected_indices),
                        unselected_values=remaining_scores, unselected_window_errors=remaining_target,
                        unselected_spearman=spearman(remaining_scores, remaining_target)))
                # Primary scout energy only. No safety claim: identify whether
                # the largest unseen error lies in the superficially safe half.
                valid = [i for i in eligible if unselected[i]["max_window_relative_l2"] is not None]
                primary = [scores[i]["relative_l2"] for i in valid]
                rankable = len(valid) >= 3 and len(set(primary)) > 1
                worst = max(valid, key=lambda i: (unselected[i]["max_window_relative_l2"], -i)) if valid else None
                unseen_values = [unselected[i]["max_window_relative_l2"] for i in valid]
                unseen_rankable = len(valid) >= 3 and len(set(unseen_values)) > 1
                all_worst = [i for i in valid if unselected[i]["max_window_relative_l2"] == max(unseen_values)] if valid else []
                safe = sorted(valid, key=lambda i: (scores[i]["relative_l2"], i))[:len(valid)//2] if rankable else []
                blind_spots.append(dict(video_id=str(record["video_id"]), strategy=strategy, fixed_depth=depth,
                    strict_validation_passed=strict_passed,
                    eligible_time_blocks=len(valid), scout_sortable=rankable, unselected_error_sortable=unseen_rankable,
                    worst_unselected_temporal_block=list(origins[worst]) if worst is not None else None,
                    worst_unselected_temporal_blocks=[list(origins[i]) for i in all_worst],
                    worst_unselected_error=unselected[worst]["max_window_relative_l2"] if worst is not None else None,
                    safest_half_temporal_blocks=[list(origins[i]) for i in safe],
                    worst_is_in_safest_half=any(i in safe for i in all_worst) if rankable and unseen_rankable else None,
                    tie_rule="All tied maxima retained; flag means ANY maximum in safe half. Scout ties use temporal index ascending; half size floor(n/2). Constant scout or unseen errors give null flag."))
    summaries = []
    for strategy in STRATEGIES:
        for depth in (1, 2):
            for signal in ("relative_l2", "max_token_relative_l2"):
                selected = [r for r in observations if (r["strategy"], r["fixed_depth"], r["signal"]) ==
                            (strategy, depth, signal)]
                fixed_population = [r for r in selected if r["reference_video_sortable"]]
                valid = [r["spearman"] for r in fixed_population if r["spearman"] is not None]
                delta = [r["paired_delta_spearman"] for r in fixed_population if r["paired_delta_spearman"] is not None]
                unseen = [r["unselected_spearman"] for r in fixed_population if r["unselected_spearman"] is not None]
                native = [r["selected_native_spearman"] for r in fixed_population
                          if r["selected_native_spearman"] is not None]
                summaries.append(dict(strategy=strategy, fixed_depth=depth, signal=signal,
                    strict_validation_passed=all(r["strict_validation_passed"] for r in selected),
                    numerical_mismatch_videos=[r["video_id"] for r in selected if not r["strict_validation_passed"]],
                    selected_native_macro_spearman=sum(native)/len(native) if native else None,
                    selected_native_sortable_videos=len(native),
                    sortable_videos=len(valid), positive_videos=sum(v > 0 for v in valid),
                    fixed_reference_video_denominator=len(fixed_population),
                    positive_fraction_fixed_denominator=sum(v > 0 for v in valid)/len(fixed_population) if fixed_population else None,
                    unrankable_reference_video_ids=[r["video_id"] for r in fixed_population if r["spearman"] is None],
                    macro_spearman=sum(valid)/len(valid) if valid else None,
                    paired_delta_macro=sum(delta)/len(delta) if delta else None,
                    paired_delta_videos=len(delta),
                    paired_delta_limitation="Complete pairs only; missing scout ranks stay in the fixed positive-rate denominator",
                    unselected_macro_spearman=sum(unseen)/len(unseen) if unseen else None,
                    unselected_sortable_videos=len(unseen),
                    unselected_positive_videos=sum(v > 0 for v in unseen),
                    pooled_spearman=spearman([v for r in selected for v in r["values"]],
                                             [v for r in selected for v in r["final_feature_errors"]]),
                    pooled_blocks=sum(r["eligible_time_blocks"] for r in selected)))
    screens = []
    for strategy in STRATEGIES:
        primary = [r for r in summaries if r["strategy"] == strategy and r["signal"] == "relative_l2"]
        passed = all(r["macro_spearman"] is not None and r["macro_spearman"] > 0 and
                     r["positive_videos"] * 3 >= 2*r["fixed_reference_video_denominator"] for r in primary)
        screens.append(dict(strategy=strategy, both_depths_positive_macro_and_two_thirds_fixed_population=passed,
                            interpretation="Descriptive screen only; no winner selected and no QA acceptance"))
    return dict(observations=observations, summaries=summaries, blind_spots=blind_spots,
                diagnostic_allow_numerical_mismatch=diagnostic_allow_numerical_mismatch,
                strict_validation_passed=not numerical_failures,
                numerical_validation_failures=numerical_failures, scope=DIAGNOSTIC_SCOPE,
                per_strategy_screens=screens, thresholds_fitted=False,
                strategies_selected_by_result=False,
                interpretation="Descriptive local-to-final feature association only; no QA or runtime claim",
                exclusions="First and periodic anchor4 blocks excluded; fewer than 3 eligible blocks or constant ranks yield null",
                pooled_limitation="Time blocks cluster within videos; pooled counts are not independent sample size")
