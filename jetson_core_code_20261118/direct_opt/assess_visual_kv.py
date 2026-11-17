"""Offline receipts, length and storage audit; never imports Torch or runs a model.

One formal row per request is required after optional arm selection. Missing
requests remain visible; duplicates/unknown phases are errors. Text and token
agreement do not constitute a correctness review of the video description.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from summarize_dram_campaign import validate_output

VARIANT = "segmented_visual_no_eviction"
EOS_IDS = (151645, 151643)
LENGTH_LIMIT = Decimal("1.10")
WORK_FIELDS = ("target_prefill_calls", "target_prefill_query_tokens", "target_verify_calls",
               "target_verify_query_tokens", "draft_tree_calls", "draft_model_forward_calls",
               "draft_decoder_query_tokens", "target_compaction_calls")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(directory, arm=None):
    directory = Path(directory)
    status_path, results_path = directory/"status.json", directory/"results.jsonl"
    status = json.loads(status_path.read_text())
    rows, warmups, other_arms = {}, 0, 0
    for line, text in enumerate(results_path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if arm is not None and row.get("arm") != arm:
            other_arms += 1
            continue
        if row.get("phase") == "warmup":
            warmups += 1
            continue
        if row.get("phase") != "measure":
            raise ValueError(f"{results_path}:{line}: missing/unsupported formal phase")
        key = row.get("request_id")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{results_path}:{line}: missing request_id")
        if key in rows:
            raise ValueError(f"{results_path}:{line}: duplicate formal request {key}; select one arm/run explicitly")
        rows[key] = row
    if not rows:
        raise ValueError(f"{results_path}: no formal rows after selection")
    return {"status": status, "rows": rows, "source": {
        "directory": str(directory.resolve()), "status_sha256": sha256(status_path),
        "results_sha256": sha256(results_path), "arm_filter": arm,
        "warmups_excluded": warmups, "other_arm_rows_excluded": other_arms}}


def is_count(value):
    return type(value) is int and value >= 0


def generation(row, status=None):
    result = row.get("result", {})
    ids = row.get("output_token_ids")
    valid_ids = isinstance(ids, list) and bool(ids) and all(is_count(x) for x in ids)
    n = len(ids) if valid_ids else None
    counts = result.get("forward_counts", {})
    verifies = counts.get("target_verify_calls")
    trace_error = None
    try:
        validate_output(row, status or {"args": {"max_new_tokens": None}})
        base = result["prompt_tokens"]
        trees = [r for r in result["rounds"] if r["kind"] == "tree"]
        if counts["target_prefill_calls"] != 1 or counts["target_prefill_query_tokens"] != base:
            raise ValueError("prefill counts disagree with full prompt")
        if counts["target_compaction_calls"] != len(trees) or counts["target_compaction_selected_tokens"] != sum(len(r["submitted"]) for r in trees):
            raise ValueError("compaction counts disagree with accepted path lengths")
        for part in result["rounds"]:
            if part["kind"] == "tree":
                selected, submitted, nodes = part["selected_tree_indices"], part["submitted"], part["tree_nodes"]
                if (not selected or selected[0] != 0 or len(selected) != len(submitted) or
                    len(set(selected)) != len(selected) or any(not is_count(x) or x >= nodes for x in selected)):
                    raise ValueError("invalid accepted tree indices")
                if part["target_cache_before"] != base or part["target_cache_after"] != base+len(selected):
                    raise ValueError("accepted-path semantic cache boundary mismatch")
                base += len(selected)
            elif part["kind"] == "terminal_target_token":
                if part is not result["rounds"][-1] or len(part["submitted"]) != 1 or part["target_cache_after"] != base:
                    raise ValueError("invalid terminal token record")
            else:
                raise ValueError("unknown generation round kind")
        if result["target_cache_length"] != base:
            raise ValueError("final semantic cache length mismatch")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        trace_error = str(exc)
    tree_records = [r for r in result.get("rounds", []) if isinstance(r,dict) and r.get("kind") == "tree"]
    checks = {
        "valid_token_ids": valid_ids,
        "duplicate_token_fields_agree": valid_ids and result.get("token_ids") == ids,
        "length_fields_agree": n is not None and row.get("num_tokens") == n and result.get("output_length") == n,
        "stop_reason_fields_agree": row.get("stop_reason") == result.get("stop_reason") and row.get("stop_reason") in ("eos", "limit"),
        "truncated_field_consistent": type(row.get("truncated")) is bool and row["truncated"] == (row.get("stop_reason") == "limit"),
        "text_present": isinstance(row.get("text"), str),
        "verify_receipts_agree": is_count(verifies) and row.get("verify_rounds") == verifies and len(tree_records) == verifies,
        "saved_trace_consistent": trace_error is None,
    }
    eos_positions = [i for i, token in enumerate(ids) if token in EOS_IDS] if valid_ids else []
    natural = valid_ids and row.get("stop_reason") == "eos" and row.get("truncated") is False and eos_positions == [n-1]
    return {"num_tokens_including_eos": n, "token_ids": ids, "text": row.get("text"),
            "stop_reason": row.get("stop_reason"), "truncated": row.get("truncated"),
            "eos_positions": eos_positions, "natural_eos": bool(natural),
            "checks": checks, "receipt_valid": all(checks.values()),
            "trace_validation_error": trace_error,
            "forward_counts": {k: counts.get(k) for k in WORK_FIELDS},
            "precision": result.get("precision", {})}


def kv_summary(row, candidate):
    return row.get("adaptor_preparation", {}).get("visual_kv" if candidate else "ring_kv", {})


def storage_snapshot(snapshot, candidate):
    if not isinstance(snapshot, dict):
        return {"reported_bytes": None, "valid": False, "reason": "missing snapshot"}
    if candidate:
        owned, shared = snapshot.get("owned_storage_bytes"), snapshot.get("shared_position_map_bytes")
        valid = is_count(owned) and is_count(shared)
        return {"reported_bytes": owned+shared if valid else None, "valid": valid,
                "owned_storage_bytes": owned, "shared_position_map_bytes": shared,
                "scope": "Target segmented owned backing including scales/metadata plus shared position map; excludes model/attention temporaries/allocator reserve"}
    amount = snapshot.get("allocated_kv_storage_bytes")
    return {"reported_bytes": amount if is_count(amount) else None, "valid": is_count(amount),
            "scope": "Native full FP16 Target K/V backing only; baseline did not record metadata ownership; not interchangeable with candidate payload-only field"}


def storage_evidence(info, candidate):
    points = [("prefill", info.get("prefill_cache")), ("final", info.get("final_cache"))]
    for i, row in enumerate(info.get("rounds", [])):
        points.extend((f"round_{i}_{side}", row.get(f"cache_{side}_commit")) for side in ("before", "after"))
    evaluated = [(name, storage_snapshot(point, candidate)) for name, point in points]
    valid = [(name, value) for name, value in evaluated if value["valid"]]
    largest = max(valid, key=lambda pair: pair[1]["reported_bytes"]) if valid else None
    return {"prefill": evaluated[0][1], "final": evaluated[1][1],
            "maximum_observed_snapshot": dict(snapshot=largest[0], **largest[1]) if largest else None,
            "snapshot_count": len(evaluated), "all_snapshots_valid": bool(evaluated) and len(valid) == len(evaluated),
            "invalid_snapshots": [name for name, value in evaluated if not value["valid"]],
            "scope": "Observed persistent-storage snapshots, not an allocator peak or measured DRAM traffic. Candidate prefill_cache is captured after conversion; native prefill peak is preserved separately."}


def memory_evidence(row, info):
    phases = info.get("memory_phase_peaks", {})
    checks, whole = {}, {}
    for kind in ("allocated", "reserved"):
        name = f"peak_{kind}_bytes"
        values = [phases.get(p, {}).get(name) for p in ("prefill", "conversion", "decode")]
        outer = row.get(name)
        phase_whole = phases.get("full_request_"+name)
        summary_whole = info.get("full_request_"+name)
        complete = all(is_count(x) for x in values) and is_count(outer) and is_count(phase_whole) and is_count(summary_whole)
        checks[kind] = complete and phase_whole == summary_whole == max(values) and outer >= phase_whole
        whole[kind] = {"row_peak_bytes": outer, "phase_max_bytes": phase_whole, "runtime_summary_peak_bytes": summary_whole}
    return {"phases": {p: phases.get(p) for p in ("prefill", "conversion", "decode")},
            "whole_request": whole, "consistent_complete_peaks": all(checks.values()), "checks": checks,
            "memory_scope": row.get("memory_scope"), "timing_scope": info.get("timing_scope")}


def candidate_receipt(row, status, mode):
    info = kv_summary(row, True)
    counts = row.get("result", {}).get("forward_counts", {})
    verifies = counts.get("target_verify_calls")
    precision = row.get("result", {}).get("precision", {})
    layers = info.get("target_attention_layer_count")
    checks = {
        "status_variant_and_mode": status.get("kv_variant") == VARIANT and status.get("visual_kv_mode") == mode,
        "runtime_variant_and_mode": info.get("variant") == VARIANT and info.get("visual_mode") == info.get("mode") == mode,
        "finalized_without_failure": info.get("finalized") is True and info.get("failed") is False,
        "no_eviction": info.get("eviction") is False and status.get("target_history_eviction") is False and status.get("draft_history_eviction") is False,
        "quantization_mode": info.get("quantized") is (mode == "int8") and status.get("visual_quantized") is (mode == "int8"),
        "no_fallback_or_retained_full_fp16": info.get("fallback_calls") == 0 and info.get("original_full_fp16_kv_tensor_refs_alive") == 0,
        "all_verifications_covered": is_count(verifies) and info.get("completed_target_forwards") == info.get("completed_verifications") == verifies and len(info.get("rounds", [])) == verifies,
        "all_target_layers_covered": layers == 28 and is_count(verifies) and info.get("mixed_attention_calls") == verifies*28,
        "native_prefill_once": info.get("prefill_calls") == counts.get("target_prefill_calls") == 1,
        "precision_backend": precision.get("target_sdpa") == "math_prefill+triton_mixed_tree" and precision.get("target_verify_attention_backend") == "triton_mixed" and precision.get("target_prefill_attention_backend") == "math",
        "precision_storage": precision.get("target_visual_kv_storage") == ("int8_group32_fp32scale" if mode == "int8" else "fp16"),
        "draft_backend_unchanged": precision.get("draft_attention_backend") == "unchanged",
    }
    return {"passed": all(checks.values()), "checks": checks, "mode": mode,
            "mixed_attention_calls": info.get("mixed_attention_calls"), "target_attention_layer_count": layers,
            "completed_target_forwards": info.get("completed_target_forwards"),
            "precision": precision, "scope": info.get("scope")}


def compare_runs(baseline, candidate, mode):
    br, cr = baseline["rows"], candidate["rows"]
    config_checks = {name: baseline["status"].get(name) is not None and baseline["status"].get(name) == candidate["status"].get(name)
                     for name in ("target", "draft")}
    limits = [run["status"].get("args", {}).get("max_new_tokens") for run in (baseline,candidate)]
    config_checks["same_generation_limit"] = is_count(limits[0]) and limits[0] > 0 and limits[0] == limits[1]
    config_checks["both_runs_completed"] = baseline["status"].get("status") == candidate["status"].get("status") == "completed"
    records = []
    for key in sorted(br.keys() & cr.keys()):
        b, c = br[key], cr[key]
        bg, cg = generation(b,baseline["status"]), generation(c,candidate["status"])
        bn, cn = bg["num_tokens_including_eos"], cg["num_tokens_including_eos"]
        ratio = cn/bn if bn and cn is not None else None
        length_ok = bn is not None and cn is not None and Decimal(cn) <= LENGTH_LIMIT*Decimal(bn)
        same = bg["token_ids"] == cg["token_ids"] if bg["checks"]["valid_token_ids"] and cg["checks"]["valid_token_ids"] else None
        first = next((i for i, (x,y) in enumerate(zip(bg["token_ids"],cg["token_ids"])) if x != y), None) if same is not None else None
        if same is False and first is None:
            first = min(bn,cn)
        receipt = candidate_receipt(c, candidate["status"], mode)
        bi, ci = kv_summary(b, False), kv_summary(c, True)
        baseline_full = bi.get("mode") == "full" and bi.get("failed") is False and bi.get("finalized") is True
        bs, cs = storage_evidence(bi, False), storage_evidence(ci, True)
        bm, cm = memory_evidence(b, bi), memory_evidence(c, ci)
        inputs = {name: b.get(name) is not None and b.get(name) == c.get(name)
                  for name in ("input_shapes", "video_grid_thw", "timestamps")}
        inputs["prompt_token_count"] = b.get("result", {}).get("prompt_tokens") is not None and b["result"]["prompt_tokens"] == c.get("result", {}).get("prompt_tokens")
        records.append({"request_id": key, "baseline_arm": b.get("arm"), "candidate_arm": c.get("arm"),
            "baseline_repeat": b.get("repeat"), "candidate_repeat": c.get("repeat"),
            "baseline": bg, "candidate": cg, "token_ids_identical": same, "first_token_difference": first,
            "length_ratio_including_eos": ratio, "within_1_10_length_limit": length_ok,
            "observed_input_metadata_equal": inputs, "candidate_receipt": receipt,
            "baseline_full_cache_receipt": baseline_full,
            "storage": {"baseline": bs, "candidate": cs}, "memory": {"baseline": bm, "candidate": cm},
            "raw_timing": {side: {k: row.get(k) for k in ("e2e_wall_s", "valid_timing", "timing_reasons", "runtime_finalize_wall_s")}
                           for side,row in (("baseline",b),("candidate",c))},
            "mechanical_checks_passed": all((baseline_full,all(config_checks.values()),bg["receipt_valid"],cg["receipt_valid"],bg["natural_eos"],cg["natural_eos"],length_ok,receipt["passed"],cs["all_snapshots_valid"],bm["consistent_complete_peaks"],cm["consistent_complete_peaks"],all(inputs.values()))),
            "quality_pass": None, "quality_review_status": "manual video-evidence review required; baseline is not gold"})
    missing, extra = sorted(br.keys()-cr.keys()), sorted(cr.keys()-br.keys())
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {Path(__file__).name:sha256(__file__), "summarize_dram_campaign.py":sha256(Path(__file__).with_name("summarize_dram_campaign.py"))}, "expected_candidate_mode": mode,
        "length_limit": str(LENGTH_LIMIT), "length_unit": "output token IDs including EOS", "eos_ids": list(EOS_IDS),
        "sources": {"baseline": baseline["source"], "candidate": candidate["source"]},
        "run_status": {side: run["status"].get("status") for side,run in (("baseline",baseline),("candidate",candidate))},
        "run_configuration_checks": config_checks,
        "run_configuration": {side: {key: run["status"].get(key) for key in ("target", "draft", "args", "source_sha256")}
                              for side,run in (("baseline",baseline),("candidate",candidate))},
        "matching": {"baseline_formal_requests": len(br), "candidate_formal_requests": len(cr), "matched": len(records),
                     "missing_candidate_requests": missing, "extra_candidate_requests": extra, "complete": not missing and not extra},
        "summary": {"token_identical": sum(r["token_ids_identical"] is True for r in records),
                    "natural_eos_candidate": sum(r["candidate"]["natural_eos"] for r in records),
                    "length_pass": sum(r["within_1_10_length_limit"] for r in records),
                    "receipt_pass": sum(r["candidate_receipt"]["passed"] for r in records),
                    "mechanical_pass": sum(r["mechanical_checks_passed"] for r in records), "quality_pass": None},
        "scope": "Offline mechanical audit only. No automatic task-quality acceptance, speed pass, total MAC or DRAM estimate. Saved rows do not contain exact input token IDs; observed input metadata agreement is partial evidence.",
        "requests": records}


def render_markdown(report):
    lines = ["# Visual KV offline evidence", "", report["scope"], "",
        f"Mode: `{report['expected_candidate_mode']}`. Matched {report['matching']['matched']}/{report['matching']['baseline_formal_requests']}. Length ceiling: 1.10 × baseline, including EOS.", "",
        "| Request | Tokens B→C | Verify B→C | Ratio | Same IDs | EOS C | Length | Receipt | Mechanical |",
        "|---|---:|---:|---:|---|---|---|---|---|"]
    for r in report["requests"]:
        ratio = "missing" if r["length_ratio_including_eos"] is None else f"{r['length_ratio_including_eos']:.4f}"
        lines.append(f"| {r['request_id']} | {r['baseline']['num_tokens_including_eos']}→{r['candidate']['num_tokens_including_eos']} | {r['baseline']['forward_counts']['target_verify_calls']}→{r['candidate']['forward_counts']['target_verify_calls']} | {ratio} | {r['token_ids_identical']} | {r['candidate']['natural_eos']} | {r['within_1_10_length_limit']} | {r['candidate_receipt']['passed']} | {r['mechanical_checks_passed']} |")
    lines += ["", "Missing candidate requests: "+", ".join(report["matching"]["missing_candidate_requests"]),
              "", "Extra candidate requests: "+", ".join(report["matching"]["extra_candidate_requests"]), "",
              "Candidate capacity uses owned storage plus shared position map, including scales. Baseline full-cache capacity covers FP16 K/V backing only. Three-phase peaks and whole-request peaks are preserved separately in evidence.json.", "",
              "See review_texts.md for paired answers. Key facts, actions, hallucinations and omissions require video evidence; neither brevity nor matching baseline proves correctness."]
    review = ["# Paired texts for manual video review", "", "Labels are known; this is not a blind review. Baseline is not gold. Quality labels remain unset."]
    for r in report["requests"]:
        review += ["", "## "+r["request_id"], "", "### Baseline", "", r["baseline"]["text"] or "[missing text]", "",
                   "### Candidate", "", r["candidate"]["text"] or "[missing text]", "",
                   "Manual review: core facts/actions retained; new unsupported claims; omissions; repetition; evidence limitations. **Pending.**"]
    return "\n".join(lines)+"\n", "\n".join(review)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=("int8", "fp16_visual"), required=True)
    parser.add_argument("--baseline-arm")
    parser.add_argument("--candidate-arm")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        baseline = load_run(args.baseline, args.baseline_arm)
        candidate = load_run(args.candidate, args.candidate_arm)
        report = compare_runs(baseline, candidate, args.expected_mode)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    markdown, review = render_markdown(report)
    (args.out/"evidence.md").write_text(markdown)
    (args.out/"review_texts.md").write_text(review)
    print(json.dumps({"out": str(args.out.resolve()), "matching": report["matching"], "summary": report["summary"]}))
    return report


if __name__ == "__main__":
    main()
