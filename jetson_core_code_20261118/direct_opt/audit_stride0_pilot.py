"""Offline audit of sequential native/stride0 full-request pilots; no Torch imports."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from assess_visual_kv import generation
from audit_native_delta_abba import aggregate, hardware_evidence

FLAGS = {"allow_tf32": False, "float32_matmul_precision": "highest",
         "reduced_precision_math_sdp": False}
EXTRA_COUNTS = {"target_tree_attention_policy_calls", "target_stride0_attention_calls"}
SOURCES = ("direct_opt/run_joint.py", "direct_opt/run_stride0_gqa_layout.py",
           "direct_opt/stride0_gqa_runtime.py", "direct_opt/stride0_gqa_math.py",
           "vendor/hf517_vispec_adapter.py", "vendor/vispec_joint_adapter.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def common_counts(row):
    return {k: v for k, v in row["result"]["forward_counts"].items() if k not in EXTRA_COUNTS}


def compare(a, b):
    checks = {key: a[key] == b[key] for key in
              ("output_token_ids", "text", "input_shapes", "video_grid_thw", "timestamps")}
    checks.update({"result."+key: a["result"][key] == b["result"][key] for key in
                   ("rounds", "prompt_tokens", "target_cache_length", "budget_trace", "visual_context")})
    checks["common_forward_counts"] = common_counts(a) == common_counts(b)
    return checks


def coverage(row, mode):
    info = row["adaptor_preparation"]["tree_attention"]
    counts = row["result"]["forward_counts"]
    trees = [r for r in row["result"]["rounds"] if r["kind"] == "tree"]
    n = len(trees)
    candidate = mode == "stride0"
    rounds = info["rounds"]
    round_ok = len(rounds) == n and all(
        r["cache_before"] == t["target_cache_before"] and
        r["tree_nodes"] == t["tree_nodes"] and
        r["cache_after"] == r["cache_before"]+r["tree_nodes"] and
        r["layer_calls"] == [1]*28 and
        r["mask_shape"] == [1, 1, r["tree_nodes"], r["cache_after"]] and
        r["position_shape"] == [3, 1, r["tree_nodes"]] and
        r["mask_dtype"] == "torch.float32" and r["position_dtype"] == "torch.int64"
        for r, t in zip(rounds, trees))
    precision = {"target": "float16", "output_head": "float32", "draft": "float16",
        "draft_sdpa": "auto", "target_sdpa": "math_prefill+stride0_fp32_math_tree" if candidate else "math",
        "target_prefill_attention_backend": "math",
        "target_verify_attention_backend": "stride0_four_group_fp32_math" if candidate else "math",
        "target_tree_attention_mode": mode, "target_tree_math_precision": FLAGS,
        "target_cache_policy": "native_full_dynamic_cache", "draft_attention_backend": "unchanged",
        "tree_attention_bitwise_equivalence_claim": False}
    e = sum(28*4*r["cache_after"]*128 for r in rounds)
    expected_copy = {"read_bytes": 76*e, "write_bytes": 100*e, "read_plus_write_bytes": 176*e} if candidate else {}
    checks = {
        "mode": info["mode"] == mode == row["adaptor_preparation"]["tree_attention_mode"],
        "finalized_no_failure": info["finalized"] is True and info["failures"] == [] and info["fallback_calls"] == 0,
        "coverage_validated": info["result_coverage_validated"] is True,
        "native_prefill_once": info["prefill_calls"] == info["prefill_forward_calls"] == 1 and info["prefill_layer_calls"] == [1]*28,
        "all_verifications": info["verify_forward_calls"] == counts["target_verify_calls"] == n and info["verify_query_tokens"] == counts["target_verify_query_tokens"],
        "all_layers": info["layer_calls"] == [n]*28 and info["selected_calls"] == info["completed_calls"] == 28*n == counts["target_tree_attention_policy_calls"],
        "mode_calls": info["candidate_calls"] == counts["target_stride0_attention_calls"] == (28*n if candidate else 0) and info["native_tree_calls"] == (0 if candidate else 28*n),
        "every_round_shape_and_precommit_boundary": round_ok,
        "full_native_cache": info["target_cache_policy"] == "native_full_dynamic_cache" and info["new_kv_policy_installed"] is False and info["cache_functions_changed"] is False and info["native_cache_type"] == "transformers.cache_utils.DynamicCache" and info["native_cache_layer_types"] == ["transformers.cache_utils.DynamicLayer"]*28,
        "mask_and_rotary": info["original_mask_reused"] is True and info["position_ids_passed_unchanged"] is True and info["rotary_reapplied"] is False,
        "expected_precision": row["result"]["precision"] == precision and info["precision"] == FLAGS,
        "no_debug_or_dram_claim": info["debug_intermediates"] is False and info["actual_dram_measured"] is False,
        "representative_receipt": (info["representative_receipt"] is not None) == candidate,
        "logical_copy_formula": info["logical_copy_totals"] == expected_copy,
    }
    return {"passed": all(checks.values()), "checks": checks,
        "verify_calls": n, "attention_layer_calls": info["completed_calls"],
        "candidate_calls": info["candidate_calls"], "native_tree_calls": info["native_tree_calls"],
        "logical_copy_totals": info["logical_copy_totals"],
        "logical_copy_scope": info["logical_copy_totals_scope"], "precision": precision}


def read_run(path, mode, known_graphics):
    path = Path(path)
    status = json.loads((path/"status.json").read_text())
    rows = [json.loads(x) for x in (path/"results.jsonl").read_text().splitlines() if x.strip()]
    requests = json.loads((path/"requests.json").read_text())
    schedule = json.loads((path/"schedule.json").read_text())
    terminal = json.loads(Path(str(path)+".exit.json").read_text())
    key = lambda r: tuple(r[k] for k in ("request_id", "arm", "phase", "repeat"))
    check(status["status"] == "completed" and terminal["returncode"] == 0, "Run not successfully terminal")
    check(len(rows) == status["completed_cells"] == status["total_cells"] == 3 and len(requests) == 1, "Pilot must have 3 cells / one request")
    check(len(set(map(key, rows))) == 3 and list(map(key, rows)) == list(map(key, schedule)), "Duplicate or changed schedule")
    check([(r["phase"], r["repeat"]) for r in rows] == [("warmup", 0), ("measure", 0), ("measure", 1)], "Unexpected pilot order")
    check({r["request_id"] for r in rows} == {requests[0]["request_id"]}, "Request mismatch")
    check({r["arm"] for r in rows} == {"memory_shared_embedding_compiled_fullkv_tree_"+mode+"_math"}, "Wrong arm")
    check(status["args"]["warmup"] == 1 and status["args"]["repeats"] == 2 and status["args"]["max_new_tokens"] == 512, "Wrong decode recipe")
    check(status["tree_attention_mode"] == mode and status["memory_layout"] == "shared_embedding" and status["deployment_adaptor"] == "compiled", "Wrong deployment")
    details = []
    for row in rows:
        gen = generation(row, status)
        cov = coverage(row, mode)
        components = row["generation_wall_s"]+row["output_decode_wall_s"]+row["runtime_finalize_wall_s"]
        times = ("e2e_wall_s", "generation_wall_s", "model_wall_s", "prepare_wall_s", "policy_prepare_wall_s", "output_decode_wall_s", "runtime_finalize_wall_s")
        timing = {"finite_nonnegative": all(math.isfinite(row[k]) and row[k] >= 0 for k in times),
            "finalization_in_model": math.isclose(row["model_wall_s"], components, abs_tol=1e-6),
            "e2e_includes_all_phases": math.isclose(row["e2e_wall_s"], components+row["prepare_wall_s"]+row["policy_prepare_wall_s"], abs_tol=1e-3)}
        check(row["scheme"]["depth"] == "fixed0" and row["scheme"]["compression"] is None and row["scheme"]["ratio"] == 1.0 and row["capture_enabled"] is False, "Changed visual recipe")
        details.append({"phase": row["phase"], "repeat": row["repeat"], "num_tokens": row["num_tokens"], "verify_rounds": row["verify_rounds"],
            "generation_checks": gen["checks"], "receipt_valid": gen["receipt_valid"], "natural_eos": gen["natural_eos"],
            "trace_validation_error": gen["trace_validation_error"], "coverage": cov, "timing_checks": timing,
            "timing": {k: row[k] for k in times}, "peak_allocated_bytes": row["peak_allocated_bytes"], "peak_reserved_bytes": row["peak_reserved_bytes"],
            "hardware": hardware_evidence(row, status["pid"], known_graphics)})
    return {"status": status, "rows": rows, "requests": requests, "terminal": terminal,
        "details": details, "aggregate": aggregate(rows[1:]), "directory": str(path.resolve()),
        "hashes": {name: digest(path/name) for name in ("status.json", "schedule.json", "requests.json", "results.jsonl")},
        "terminal_sha256": digest(str(path)+".exit.json")}


def audit(native, candidate, reference, known_graphics=()):
    runs = [read_run(native, "native", known_graphics), read_run(candidate, "stride0", known_graphics)]
    a, b = runs
    config = {k: a["status"][k] == b["status"][k] for k in ("target", "draft", "torch", "transformers", "device_name", "request_sha256")}
    config.update({"args."+k: a["status"]["args"][k] == b["status"]["args"][k] for k in ("max_new_tokens", "seed", "physical_gpu")})
    config.update({"source."+k: bool(a["status"]["source_sha256"].get(k)) and a["status"]["source_sha256"].get(k) == b["status"]["source_sha256"].get(k) for k in SOURCES})
    config["same_manifest"] = a["requests"] == b["requests"]
    config["distinct_nonoverlapping_processes"] = a["status"]["pid"] != b["status"]["pid"] and a["terminal"]["ended"] <= b["terminal"]["started"]
    pairs = [{"phase": x["phase"], "repeat": x["repeat"], "checks": compare(x, y)} for x, y in zip(a["rows"], b["rows"])]
    within = {mode: [compare(run["rows"][0], r) for r in run["rows"][1:]] for mode, run in zip(("native", "stride0"), runs)}
    reference = Path(reference)
    oldstatus = json.loads((reference/"status.json").read_text())
    old = [json.loads(x) for x in (reference/"results.jsonl").read_text().splitlines() if x.strip()]
    old = [r for r in old if r["phase"] == "measure" and r["request_id"] == a["requests"][0]["request_id"]]
    check(len(old) == 1 and oldstatus["status"] == "completed" and oldstatus["memory_layout"] == "shared_embedding" and oldstatus["deployment_adaptor"] == "compiled", "Ambiguous/wrong historical compiled reference")
    lineage = {"directory": str(reference.resolve()), "results_sha256": digest(reference/"results.jsonl"),
        "same_models": all(oldstatus[k] == a["status"][k] for k in ("target", "draft")),
        "checks_per_native_cell": [compare(old[0], r) for r in a["rows"]],
        "scope": "Historical output/trajectory check only. No historical timing comparison."}
    arms = {mode: run["aggregate"] for mode, run in zip(("native", "stride0"), runs)}
    delta = {phase: {"throughput_change_pct": 100*(arms["stride0"][phase]["tokens_per_second"]/arms["native"][phase]["tokens_per_second"]-1),
             "latency_change_pct": 100*(arms["stride0"][phase]["total_seconds"]/arms["native"][phase]["total_seconds"]-1)} for phase in ("e2e", "generation")}
    all_details = [d for run in runs for d in run["details"]]
    formal = [d for d in all_details if d["phase"] == "measure"]
    hw = [d["hardware"] for d in formal]
    mechanical = all(config.values()) and all(all(p["checks"].values()) for p in pairs)
    mechanical &= all(all(c.values()) for checks in within.values() for c in checks)
    mechanical &= lineage["same_models"] and all(all(c.values()) for c in lineage["checks_per_native_cell"])
    mechanical &= all(d["receipt_valid"] and d["natural_eos"] and d["coverage"]["passed"] and all(d["timing_checks"].values()) for d in all_details)
    return {"schema": "stride0_sequential_pilot_audit_v1", "tool_sha256": digest(__file__),
        "helper_sha256": {f: digest(Path(__file__).with_name(f)) for f in ("assess_visual_kv.py", "audit_native_delta_abba.py", "summarize_dram_campaign.py")},
        "mechanical_exact_comparison_pass": bool(mechanical), "configuration_checks": config,
        "cross_arm_cells": pairs, "within_arm_repeat_checks": within, "historical_compiled_reference": lineage,
        "common_forward_counts_per_request": common_counts(a["rows"][0]), "formal_aggregate": arms, "change": delta,
        "hardware_formal": {"raw_valid_timing_counts": dict(Counter(str(h["valid_timing"]) for h in hw)),
            "foreign_pids": sorted({p for h in hw for p in h["foreign_pids"]}),
            "foreign_except_supplied_graphics": sorted({p for h in hw for p in h["foreign_except_known_graphics"]}),
            "supplied_known_graphics_pids": list(known_graphics), "process_queries": sum(h["process_query_count"] for h in hw),
            "query_failures": sum(h["process_query_failures"] for h in hw), "all_visibility_complete": all(h["process_visibility_complete"] for h in hw),
            "errors": [e for h in hw for e in h["errors"]]},
        "runs": [{"mode": mode, **{k: run[k] for k in ("directory", "hashes", "terminal", "terminal_sha256", "details")},
                  "process": {k: run["status"][k] for k in ("pid", "started_at", "completed_at", "load_seconds", "experimental_setup_seconds")},
                  "source_sha256": {k: run["status"]["source_sha256"][k] for k in SOURCES}} for mode, run in zip(("native", "stride0"), runs)],
        "quality_pass": None, "actual_dram_measured": False,
        "limitations": ["One request; two measured repeats per arm in sequential independent processes. Exploratory, not stable or isolated throughput.",
            "Raw valid_timing flags preserved. Sampled process visibility cannot exclude work between samples.",
            "Model loading and one-time installation excluded. Full request includes policy finalization and final synchronization.",
            "Whole-request allocator peaks include prefill; no phase-level peak or persistent-KV reduction is established here.",
            "Logical avoided copies are counterfactual tensor operations, not measured DRAM; attention MAC and full KV capacity are unchanged.",
            "Exact output and trajectory reproduction on this request is not a general numerical or task-quality guarantee."]}


def markdown(report):
    if not report["mechanical_exact_comparison_pass"]:
        return ("# Stride0 单请求完整链路审计\n\n机械核对失败。请检查 audit.json 中逐条检查结果；"
                "此报告不宣称输出、轨迹或执行回执通过。\n")
    a = report["formal_aggregate"]
    sample = report["runs"][0]["details"][0]
    hw = report["hardware_formal"]
    lines = ["# Stride0 单请求完整链路审计", "", f"机械核对通过：**{report['mechanical_exact_comparison_pass']}**。每臂 1 次预热、2 次正式请求；预热不计速度。", "",
        "| 指标 | native 同布局 | stride0 |", "|---|---:|---:|"]
    for label, phase in (("完整请求", "e2e"), ("生成阶段", "generation")):
        lines += [f"| {label}平均秒 | {a['native'][phase]['total_seconds']/2:.6f} | {a['stride0'][phase]['total_seconds']/2:.6f} |",
                  f"| {label} tokens/s | {a['native'][phase]['tokens_per_second']:.6f} | {a['stride0'][phase]['tokens_per_second']:.6f} |"]
    lines += ["", f"完整请求吞吐变化 **{report['change']['e2e']['throughput_change_pct']:+.3f}%**，耗时变化 **{report['change']['e2e']['latency_change_pct']:+.3f}%**；生成阶段吞吐变化 **{report['change']['generation']['throughput_change_pct']:+.3f}%**。这不是整请求吞吐减半。", "",
        f"6 个请求均为 {sample['num_tokens']} tokens、自然 EOS、{sample['verify_rounds']} 次验证。两臂对应 3 cells 的全部输出 IDs、完整 rounds、输入元数据及原有 forward counts 完全一致；同臂重复和历史 shared+compiled 输出/轨迹也一致。历史速度未参与比较。", "",
        f"每个请求 prefill 的 28 层各调用 1 次；验证的 28 层各调用 {sample['verify_rounds']} 次，共 {sample['coverage']['attention_layer_calls']} 次。policy 正常 finalized、无 fallback；模式和精度回执符合 native/stride0 的预定差异。", ""]
    for run in report["runs"]:
        d = [x for x in run["details"] if x["phase"] == "measure"]
        lines.append(f"- {run['mode']} 正式整请求 peak allocated：{[x['peak_allocated_bytes'] for x in d]} B；peak reserved：{[x['peak_reserved_bytes'] for x in d]} B。")
    peaks = [[x['peak_allocated_bytes'] for x in r['details'] if x['phase'] == 'measure'] for r in report['runs']]
    copies = report['runs'][1]['details'][0]['coverage']['logical_copy_totals']
    lines += ["", f"两臂正式整请求 peak allocated 相同：{peaks[0] == peaks[1]}。模块临时显存下降不能等同整请求峰值下降。", "",
        f"正式 valid_timing 原值计数：{hw['raw_valid_timing_counts']}；{hw['process_queries']} 次采样查询观察到外部 PID {hw['foreign_pids']}；查询失败 {hw['query_failures']} 次。已由调用者确认的图形进程 PID 为 {hw['supplied_known_graphics_pids']}。两个进程按 native→stride0 先后执行，仅支持本次探索性观察。", "",
        f"候选每请求按真实形状计算的避免逻辑写入为 {copies['write_bytes']:,} B，逻辑读写为 {copies['read_plus_write_bytes']:,} B；这是 HF repeat→math 的张量操作反事实，不是实测 DRAM。矩阵 MAC 与完整 KV 驻留容量未变。", "",
        "## 来源", ""]
    lines += [f"- `{r['directory']}`" for r in report["runs"]]
    lines += ["", "## 限制", ""]+["- "+x for x in report["limitations"]]
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--compiled-reference", required=True, type=Path)
    p.add_argument("--known-graphics-pid", action="append", type=int, default=[])
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    report = audit(args.native, args.candidate, args.compiled_reference, args.known_graphics_pid)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.out/"audit.md").write_text(markdown(report))
    print(json.dumps({k: report[k] for k in ("mechanical_exact_comparison_pass", "formal_aggregate", "change", "hardware_formal", "tool_sha256")}, indent=2))


if __name__ == "__main__":
    main()
