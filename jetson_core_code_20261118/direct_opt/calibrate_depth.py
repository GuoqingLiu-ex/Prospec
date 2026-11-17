"""Freeze training-free depth thresholds from disjoint development prefixes.

No model, torch, answer labels, or evaluation output is used. The three rules
are declared below before inspecting evaluation quality. Marginal quantiles
describe representation/budget heuristics; they do not estimate QA error.

Example:
  python calibrate_depth.py --records DEV/prefix_records.json \
    --metadata DEV/results.jsonl --vision-config DEV/vision_config.json \
    --out NEW_DIRECTORY

Each candidate JSON contains `thresholds` accepted by direct_depth_policy and
run_depth_bench. Anchor stride is a separate runtime argument. `report.json`
contains predicted development actions for strides 4 and 0, exact dense-matrix
MAC counts when grid/config are supplied, and budget-matched control schedules.
All candidate thresholds are fitted once to finite non-first blocks; periodic
anchors are imposed only during application, so the anchor ablation does not
also change thresholds. This differs from fitting only recorded nonanchors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compute_cost import calculate_vision_cost

SCORE_KEYS = ("max_window_relative_update", "max_window_temporal_change")
SKIP_LAYERS = (28, 29)
PREFIX_LAYER = 27
BUDGET_GRID = tuple((min(1., (i + 8) / 40), i / 40) for i in range(20, 41))
DEFAULT_SEEDS = (20261002, 20261003, 20261004)


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        result = json.loads(path.read_text())
        if isinstance(result, dict):
            result = result["records"]
    if not isinstance(result, list) or not result:
        raise ValueError("Require a nonempty record list")
    return result


def _extract(record):
    record = dict(record)
    if "stats" in record:
        calls = record["stats"]["calls"]
        if len(calls) != 1:
            raise ValueError("Require one visual call per development observation")
        call = calls[0]
        if (call.get("prefix_layer", PREFIX_LAYER) != PREFIX_LAYER or
                tuple(call.get("skip_layers", SKIP_LAYERS)) != SKIP_LAYERS):
            raise ValueError("Expected prefix 27 and skip layers 28/29")
        if call.get("completed") is False or any(call.get("actions", [])):
            raise ValueError("Use completed full-depth development observations")
        record["features"] = call["features"]
        record["video_grid_thw"] = call["grid_thw"]
    return record


def normalize_records(records, metadata=None):
    """Validate scores and optionally join true grids, without reading answers."""
    by_request = {}
    for row in metadata or []:
        row = _extract(row)
        key = row["request_id"]
        if key in by_request:
            raise ValueError("Repeated metadata request ID")
        by_request[key] = row
    result, videos, contributors = [], set(), set()
    for original in records:
        row = _extract(original)
        if row.get("video_id") is None:
            raise ValueError("Each record must identify its source video")
        video = str(row["video_id"])
        if video in videos:
            raise ValueError("Use one observation per distinct development video")
        videos.add(video)
        request = row.get("request_id", video)
        if metadata is not None:
            if request not in by_request:
                raise ValueError(f"Missing grid metadata for {request}")
            other = by_request[request]
            if str(other["video_id"]) != video or other["features"] != row["features"]:
                raise ValueError("Metadata and prefix records disagree")
            row["video_grid_thw"] = other["video_grid_thw"]
        f = row["features"]
        update, temporal = (f[key] for key in SCORE_KEYS)
        size = len(update)
        if not size or len(temporal) != size:
            raise ValueError("Require nonempty equally sized per-time scores")
        for key in ("finite", "anchor"):
            if key in f and (len(f[key]) != size or any(type(x) is not bool for x in f[key])):
                raise ValueError(f"Invalid {key} vector")
        finite = []
        for i, (u, t) in enumerate(zip(update, temporal)):
            for value in (u, t):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ValueError("Scores must be numeric or null")
                if value is not None and math.isfinite(value) and value < 0:
                    raise ValueError("Relative distances cannot be negative")
            finite.append(f.get("finite", [True] * size)[i] and
                          all(v is not None and math.isfinite(v) for v in (u, t)))
        grid = row.get("video_grid_thw", row.get("grid_thw"))
        if grid is not None:
            if not isinstance(grid, list) or not grid:
                raise ValueError("Expected a nonempty grid list")
            for shape in grid:
                if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
                    raise ValueError("Grid dimensions must be positive integers")
            lengths = [shape[0] for shape in grid]
        else:
            lengths = [size]
        if sum(lengths) != size:
            raise ValueError("Grid temporal count and feature length disagree")
        starts, offset = [], 0
        for length in lengths:
            starts.append(offset)
            offset += length
        first = set(starts)
        if any(finite[i] and i not in first for i in range(size)):
            contributors.add(video)
        result.append(dict(video_id=video, request_id=request, update=update, temporal=temporal,
                           finite=finite, clip_lengths=lengths, first=starts,
                           grid_thw=grid, clip_boundaries_verified=grid is not None))
    if len(contributors) < 2:
        raise ValueError("Need finite non-first observations from >=2 development videos")
    return result


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def fit_thresholds(records, q1, q2):
    if not 0 <= q2 <= q1 <= 1:
        raise ValueError("Expected 0 <= q2 <= q1 <= 1")
    eligible = [(r["update"][i], r["temporal"][i]) for r in records
                for i in range(len(r["update"])) if r["finite"][i] and i not in r["first"]]
    if not eligible:
        raise ValueError("No eligible scores")
    update, temporal = zip(*eligible)
    return {"update1": quantile(update, q1), "update2": quantile(update, q2),
            "temporal1": quantile(temporal, q1), "temporal2": quantile(temporal, q2)}


def eligibility(record, anchor_stride):
    if type(anchor_stride) is not int or anchor_stride < 0:
        raise ValueError("anchor_stride must be a nonnegative integer")
    result = []
    for start, length in zip(record["first"], record["clip_lengths"]):
        result.extend(record["finite"][start + i] and i > 0 and
                      (anchor_stride == 0 or i % anchor_stride != 0) for i in range(length))
    return result


def decide(record, thresholds, anchor_stride):
    q = thresholds
    return [int(u <= q["update1"] and t <= q["temporal1"]) +
            int(u <= q["update2"] and t <= q["temporal2"]) if allowed else 0
            for u, t, allowed in zip(record["update"], record["temporal"],
                                     eligibility(record, anchor_stride))]


def _metrics(record, actions, config=None):
    count = len(actions)
    out = dict(actions=actions, histogram={str(i): actions.count(i) for i in range(3)},
               skipped_depth=sum(actions), mean_skipped_depth=sum(actions) / count)
    if record["grid_thw"] is not None:
        weights = [h * w for t, h, w in record["grid_thw"] for _ in range(t)]
        out["skipped_token_blocks"] = sum(w * a for w, a in zip(weights, actions))
        out["original_token_blocks"] = 32 * sum(weights)
        out["token_block_reduction"] = out["skipped_token_blocks"] / out["original_token_blocks"]
    else:
        out["token_block_reduction"] = None
    if config is not None and record["grid_thw"] is not None:
        cost = calculate_vision_cost(record["grid_thw"], config, actions, SKIP_LAYERS)
        out.update(full_vision_major_matrix_macs=cost["full"]["vision_major_matrix_macs"],
                   executed_vision_major_matrix_macs=cost["executed"]["vision_major_matrix_macs"],
                   saved_macs=cost["saved_macs"],
                   vision_major_matrix_mac_reduction=cost["vision_major_matrix_mac_reduction"])
    return out


def matched_controls(record, actions, anchor_stride, seeds=DEFAULT_SEEDS):
    """Match each clip's budget; never shuffle across different spatial grids."""
    allowed = eligibility(record, anchor_stride)
    uniform = [0] * len(actions)
    random_controls = {str(seed): [0] * len(actions) for seed in seeds}
    for clip, (start, length) in enumerate(zip(record["first"], record["clip_lengths"])):
        indices = [i for i in range(start, start + length) if allowed[i]]
        if not indices:
            continue
        budget = sum(actions[i] for i in indices)
        base, remainder = divmod(budget, len(indices))
        for i in indices:
            uniform[i] = base
        # Even spacing, independent of score rank; floor gives unique positions.
        for j in range(remainder):
            uniform[indices[((2 * j + 1) * len(indices)) // (2 * remainder)]] += 1
        for seed in seeds:
            key = f"{seed}:{record['video_id']}:{clip}".encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
            values = [actions[i] for i in indices]
            rng.shuffle(values)
            for i, action in zip(indices, values):
                random_controls[str(seed)][i] = action
    return {"uniform_budget": uniform, **{f"random_{k}": v for k, v in random_controls.items()}}


def summarize(rows):
    count = sum(len(r["actions"]) for r in rows)
    out = dict(videos=len(rows), temporal_blocks=count,
               histogram={str(i): sum(r["histogram"][str(i)] for r in rows) for i in range(3)},
               macro_mean_skipped_depth=sum(r["mean_skipped_depth"] for r in rows) / len(rows),
               pooled_mean_skipped_depth=sum(r["skipped_depth"] for r in rows) / count)
    if all("original_token_blocks" in r for r in rows):
        out["pooled_token_block_reduction"] = (sum(r["skipped_token_blocks"] for r in rows) /
                                                sum(r["original_token_blocks"] for r in rows))
    if all("saved_macs" in r for r in rows):
        full = sum(r["full_vision_major_matrix_macs"] for r in rows)
        saved = sum(r["saved_macs"] for r in rows)
        out.update(full_vision_major_matrix_macs=full, saved_macs=saved,
                   pooled_vision_major_matrix_mac_reduction=saved / full,
                   macro_vision_major_matrix_mac_reduction=sum(
                       r["vision_major_matrix_mac_reduction"] for r in rows) / len(rows))
    return out


def evaluate(records, thresholds, anchor_stride, config=None, controls=True, seeds=DEFAULT_SEEDS):
    rows, control_rows = [], {}
    for record in records:
        actions = decide(record, thresholds, anchor_stride)
        allowed = eligibility(record, anchor_stride)
        row = dict(video_id=record["video_id"], request_id=record["request_id"],
                   grid_thw=record["grid_thw"], **_metrics(record, actions, config))
        row.update(eligible_time_blocks=sum(allowed),
                   protected_or_invalid_time_blocks=len(actions) - sum(allowed),
                   max_mean_skipped_depth=2 * sum(allowed) / len(actions),
                   exceeds_unprotected_fixed1_depth=row["mean_skipped_depth"] > 1)
        rows.append(row)
        if controls:
            row["matched_controls"] = {}
            for name, control in matched_controls(record, actions, anchor_stride, seeds).items():
                stats = _metrics(record, control, config)
                stats["differs_from_adaptive"] = control != actions
                if "saved_macs" in stats:
                    if stats["saved_macs"] != row["saved_macs"]:
                        raise AssertionError("Budget control changed theoretical matrix MACs")
                    stats["same_matrix_macs_verified"] = True
                row["matched_controls"][name] = stats
                control_rows.setdefault(name, []).append(stats)
    summary = summarize(rows)
    summary["macro_max_mean_skipped_depth"] = sum(r["max_mean_skipped_depth"] for r in rows) / len(rows)
    eligible = sum(r["eligible_time_blocks"] for r in rows)
    summary["fraction_of_allowed_depth_budget_used"] = (sum(r["skipped_depth"] for r in rows) /
                                                       (2 * eligible) if eligible else 0.)
    summary["eligible_fraction_action2"] = summary["histogram"]["2"] / eligible if eligible else 0.
    summary["videos_exceeding_fixed1_depth"] = sum(r["exceeds_unprotected_fixed1_depth"] for r in rows)
    return dict(anchor_stride=anchor_stride, protect_first=True, summary=summary, records=rows,
                matched_control_summaries={name: {**summarize(values),
                    "videos_with_different_actions": sum(r["differs_from_adaptive"] for r in values)}
                    for name, values in control_rows.items()})


def generate_candidates(records, config=None, target_depth=1.10, seeds=DEFAULT_SEEDS):
    if not math.isfinite(target_depth) or not 1 < target_depth < 2:
        raise ValueError("Budget target must be between one and two skipped layers")
    if config is not None:
        spec = config.get("vision_config", config)
        if spec["depth"] != 32 or any(k in spec["fullatt_block_indexes"] for k in SKIP_LAYERS):
            raise ValueError("This calibration fixes the 32-layer tower and local layers 28/29")
    search, selected = [], None
    for q1, q2 in BUDGET_GRID:
        thresholds = fit_thresholds(records, q1, q2)
        summary = evaluate(records, thresholds, 4, controls=False)["summary"]
        search.append(dict(q1=q1, q2=q2, macro_mean_skipped_depth=summary["macro_mean_skipped_depth"],
                           histogram=summary["histogram"]))
        if selected is None and summary["macro_mean_skipped_depth"] >= target_depth:
            selected = (q1, q2)
    attainable = selected is not None
    # Report the boundary when unattainable, but do not recommend it silently.
    selected = selected if attainable else BUDGET_GRID[-1]
    definitions = [
        ("reference_q75_q50", .75, .50, "Marginal q75 permits one and q50 permits two"),
        ("envelope_q100_q90", 1., .90, "Observed dev envelope permits one; q90 permits two"),
        ("budget_first_ge110", *selected,
         "First predeclared quantile-grid point reaching dev macro depth target with stride4"),
    ]
    candidates = []
    for name, q1, q2, rule in definitions:
        thresholds = fit_thresholds(records, q1, q2)
        candidates.append(dict(name=name, rule=rule, thresholds=thresholds,
                               quantiles=dict(skip1=q1, skip2=q2),
                               anchor_scenarios={str(stride): evaluate(records, thresholds, stride, config,
                                                                      seeds=seeds) for stride in (4, 0)}))
    duplicates = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1:]:
            for stride in ("4", "0"):
                if [r["actions"] for r in left["anchor_scenarios"][stride]["records"]] == [
                        r["actions"] for r in right["anchor_scenarios"][stride]["records"]]:
                    duplicates.append(dict(candidates=[left["name"], right["name"]], anchor_stride=int(stride)))
    baselines = {}
    for depth in (1, 2):
        rows = [dict(video_id=r["video_id"], **_metrics(r, [depth] * len(r["update"]), config)) for r in records]
        baselines[f"fixed{depth}_all_time_blocks"] = dict(summary=summarize(rows), records=rows)
    return dict(schema_version=1, no_training=True, answer_labels_used=False,
                prefix_layer=PREFIX_LAYER, skip_layers=list(SKIP_LAYERS),
                video_ids=[r["video_id"] for r in records],
                pooling="All finite non-first blocks across distinct dev videos; periodic anchors not excluded",
                clip_boundaries_verified=all(r["clip_boundaries_verified"] for r in records),
                quantile_method="linear", random_control_seeds=list(seeds),
                semantics="Representation/budget heuristic; does not predict or guarantee QA correctness",
                cost_scope="Visual dense matrix MACs only; 1 MAC = 2 FLOPs; gate/packing/memory/LLM excluded",
                control_scope="Per-clip adaptive budget, then uniform or random allocation; not an input-independent baseline. Include controller cost if using online. Random keeps histogram; uniform may redistribute 0/1/2.",
                budget_selection=dict(target_macro_depth=target_depth, selection_anchor_stride=4,
                                      target_attainable=attainable, search=search,
                                      selected_quantiles=dict(skip1=selected[0], skip2=selected[1])),
                recommended_candidate="budget_first_ge110" if attainable else None,
                recommended_anchor_stride=4, candidates=candidates,
                identical_development_action_scenarios=duplicates, baselines=baselines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--metadata", help="Matching full-depth results JSONL supplies exact grids")
    parser.add_argument("--vision-config", help="Required with grids for exact matrix MAC accounting")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--target-depth", type=float, default=1.10)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error("Output directory exists; preserve frozen calibration and use a new directory")
    records = normalize_records(read_records(args.records),
                                read_records(args.metadata) if args.metadata else None)
    config = json.loads(Path(args.vision_config).read_text()) if args.vision_config else None
    report = generate_candidates(records, config, target_depth=args.target_depth)
    source_paths = [args.records, args.metadata, args.vision_config, __file__]
    report["sources"] = [{"path": str(Path(path).resolve()),
                           "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                          for path in source_paths if path]
    out.mkdir(parents=True)
    for candidate in report["candidates"]:
        frozen = {key: value for key, value in candidate.items() if key != "anchor_scenarios"}
        frozen.update(prefix_layer=PREFIX_LAYER, skip_layers=list(SKIP_LAYERS), protect_first=True,
                      recommended_anchor_stride=4, no_training=True, answer_labels_used=False,
                      source_video_ids=report["video_ids"], sources=report["sources"],
                      semantics=report["semantics"])
        (out / f"{candidate['name']}.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out.resolve()), "recommended_candidate": report["recommended_candidate"],
                      "summaries": {c["name"]: {s: a["summary"] for s, a in c["anchor_scenarios"].items()}
                                    for c in report["candidates"]}}, indent=2))


if __name__ == "__main__":
    main()
