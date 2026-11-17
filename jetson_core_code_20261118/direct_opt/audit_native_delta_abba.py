"""Offline audit of four independent deployments in native/delta/delta/native order.

Reads saved JSON only. Generation and complete-request throughput have separate
denominators; neither includes process startup or loading. No Torch imports.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from assess_visual_kv import generation, memory_evidence
from audit_delta_commit_pairs import DELTA, commit_receipt, digest, require

NATIVE = "memory_native_native"
RUNS = (("00_native", "native"), ("01_delta", "delta"),
        ("02_delta", "delta"), ("03_native", "native"))
ROW_FIELDS = ("output_token_ids", "text", "input_shapes", "video_grid_thw", "timestamps")
RESULT_FIELDS = ("rounds", "forward_counts", "precision", "prompt_tokens",
                 "target_cache_length", "budget_trace", "visual_context")


def equality(a, b):
    checks = {key: a[key] == b[key] for key in ROW_FIELDS}
    checks.update({"result."+key: a["result"][key] == b["result"][key]
                   for key in RESULT_FIELDS})
    return checks


def round_differences(a, b):
    native, delta = a["result"]["rounds"], b["result"]["rounds"]
    changes = []
    for index in range(max(len(native), len(delta))):
        left = native[index] if index < len(native) else None
        right = delta[index] if index < len(delta) else None
        if left == right:
            continue
        values = ({key: {"native": left.get(key), "delta": right.get(key)}
                   for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key)}
                  if left is not None and right is not None else {"record": {"native": left, "delta": right}})
        changes.append({"round_index": index, "fields": values})
    return {"changed_round_count": len(changes),
            "changed_field_counts": dict(Counter(key for change in changes for key in change["fields"])),
            "changes": changes}


def aggregate(rows):
    require(bool(rows), "No rows to aggregate")
    tokens = sum(row["num_tokens"] for row in rows)
    result = {"formal_count": len(rows), "output_tokens_including_eos": tokens,
              "verify_rounds": sum(row["verify_rounds"] for row in rows)}
    for phase in ("e2e", "generation"):
        times = [row[phase+"_wall_s"] for row in rows]
        result[phase] = {"total_seconds": sum(times), "tokens_per_second": tokens/sum(times),
                         "median_request_seconds": statistics.median(times)}
    result["phase_seconds"] = {key: sum(row[key] for row in rows) for key in
                               ("prepare_wall_s", "policy_prepare_wall_s",
                                "output_decode_wall_s", "runtime_finalize_wall_s")}
    return result


def improvement(native, delta):
    return {phase: 100*(delta[phase]["tokens_per_second"]/native[phase]["tokens_per_second"]-1)
            for phase in ("e2e", "generation")}


def hardware_evidence(row, own_pid, known_graphics):
    hardware = row["hardware"]
    foreign = sorted({item["pid"] for item in hardware["foreign_processes"]})
    return {"valid_timing": row["valid_timing"], "timing_reasons": row.get("timing_reasons"),
            "available": hardware.get("available"),
            "device": hardware.get("device"),
            "physical_device_index": hardware.get("physical_device_index"),
            "allowed_pids": hardware.get("allowed_pids"),
            "own_pid_allowlisted": own_pid in hardware.get("allowed_pids", []),
            "foreign_processes": hardware["foreign_processes"], "foreign_pids": foreign,
            "foreign_except_known_graphics": sorted(set(foreign)-set(known_graphics)),
            "process_query_count": hardware.get("process_query_count"),
            "process_query_failures": hardware.get("process_query_failures"),
            "process_visibility_complete": hardware.get("process_visibility_complete"),
            "process_interval_s": hardware.get("process_interval_s"),
            "errors": hardware.get("errors"), "telemetry_summary": hardware.get("telemetry_summary")}


def read_block(path, label, expected_requests, known_graphics):
    status = json.loads((path/"status.json").read_text())
    schedule = json.loads((path/"schedule.json").read_text())
    requests = json.loads((path/"requests.json").read_text())
    rows = [json.loads(line) for line in (path/"results.jsonl").read_text().splitlines() if line.strip()]
    require(status["status"] == "completed", f"{path.name}: incomplete run")
    args = status["args"]
    require(args["repeats"] == args["warmup"] == 1 and args["max_new_tokens"] == 512,
            f"{path.name}: expected one warmup, one formal repeat and 512-token budget")
    require(len(requests) == len({r["request_id"] for r in requests}) == expected_requests,
            f"{path.name}: wrong/duplicate manifest requests")
    key = lambda row: tuple(row[k] for k in ("request_id", "arm", "phase", "repeat"))
    require(len(rows) == len({key(row) for row in rows}), f"{path.name}: duplicate cells")
    require([key(row) for row in rows] == [key(cell) for cell in schedule],
            f"{path.name}: results differ from schedule")
    require(len(rows) == status["completed_cells"] == status["total_cells"] == expected_requests+1,
            f"{path.name}: missing/extra cells")
    arm = NATIVE if label == "native" else DELTA
    require({r["arm"] for r in rows} == {arm}, f"{path.name}: unexpected arm")
    formal = [r for r in rows if r["phase"] == "measure"]
    warm = [r for r in rows if r["phase"] == "warmup"]
    require(len(warm) == 1 and warm[0]["request_id"] == requests[0]["request_id"],
            f"{path.name}: warmup must be exactly the first manifest request")
    require(len(formal) == expected_requests and {r["repeat"] for r in formal} == {0},
            f"{path.name}: bad formal allocation")
    require({r["request_id"] for r in formal} == {r["request_id"] for r in requests},
            f"{path.name}: formal request coverage differs")
    require(status["memory_layout"] == ("native" if label == "native" else "shared_embedding")
            and status["deployment_adaptor"] == ("native" if label == "native" else "compiled"),
            f"{path.name}: wrong deployment recipe")
    if label == "delta":
        require(status["paired_active_baseline"] is False, "Candidate must be its own deployment")
    details = []
    for row in rows:
        gen = generation(row, status)
        prep = row["adaptor_preparation"]
        require(row["depth_mode"] == "fixed0" and row["visual_keep_ratio"] == 1.0
                and row["capture_enabled"] is False, "Visual compute/input path changed")
        require(row["result"]["precision"] == {"target": "float16", "output_head": "float32",
                    "target_sdpa": "math", "draft": "float16", "draft_sdpa": "auto"},
                "Unexpected model precision/attention backend")
        require(row["scheme"]["compression"] is None and row["scheme"]["ratio"] == 1.0,
                "This campaign must retain all visual tokens")
        require(("ring_kv" in prep) == (label == "delta"), "Native must not have a ring wrapper")
        for field in ("e2e_wall_s", "generation_wall_s", "model_wall_s", "prepare_wall_s",
                      "policy_prepare_wall_s", "output_decode_wall_s", "runtime_finalize_wall_s"):
            value = row[field]
            require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                    f"{path.name}: missing/invalid timing {field}")
        require(row["e2e_wall_s"] > 0 and row["generation_wall_s"] > 0, "Zero timing denominator")
        components = row["generation_wall_s"]+row["output_decode_wall_s"]+row["runtime_finalize_wall_s"]
        timing_checks = {
            "finalization_in_model_wall": math.isclose(row["model_wall_s"], components, abs_tol=1e-6),
            "complete_wall_covers_all_phases": math.isclose(row["e2e_wall_s"], components+
                row["prepare_wall_s"]+row["policy_prepare_wall_s"], abs_tol=1e-3)}
        receipt = commit_receipt(row) if label == "delta" else None
        memory = memory_evidence(row, prep["ring_kv"]) if label == "delta" else {
            "outer_peak_allocated_bytes": row["peak_allocated_bytes"],
            "outer_peak_reserved_bytes": row["peak_reserved_bytes"],
            "scope": "Native does not reset allocator peaks within this request"}
        details.append({"request_id": row["request_id"], "phase": row["phase"],
            "repeat": row["repeat"], "num_tokens": row["num_tokens"], "verify_rounds": row["verify_rounds"],
            "generation_checks": gen["checks"], "trace_validation_error": gen["trace_validation_error"],
            "receipt_valid": gen["receipt_valid"], "natural_eos": gen["natural_eos"],
            "stop_reason": row["stop_reason"], "truncated": row["truncated"],
            "e2e_wall_s": row["e2e_wall_s"], "generation_wall_s": row["generation_wall_s"],
            "runtime_finalize_wall_s": row["runtime_finalize_wall_s"], "timing_scope_checks": timing_checks,
            "hardware": hardware_evidence(row, status["pid"], known_graphics),
            "commit": receipt, "memory": memory})
    return {"name": path.name, "label": label, "status": status, "requests": requests,
            "formal": formal, "source": str(path.resolve()),
            "hashes": {name: digest(path/name) for name in
                       ("status.json", "results.jsonl", "requests.json", "schedule.json")},
            "details": details, "aggregate": aggregate(formal)}


def audit(root, expected_requests=12, known_graphics=()):
    blocks = [read_block(root/name, label, expected_requests, known_graphics) for name, label in RUNS]
    first = blocks[0]
    config_checks = {field: all(b["status"][field] == first["status"][field] for b in blocks)
                     for field in ("target", "draft", "torch", "transformers", "device_name")}
    config_checks.update({"args."+field: all(b["status"]["args"][field] == first["status"]["args"][field]
                         for b in blocks) for field in ("max_new_tokens", "physical_gpu", "seed")})
    config_checks["same_full_manifest"] = all(b["requests"] == first["requests"] for b in blocks)
    config_checks["four_distinct_processes"] = len({b["status"]["pid"] for b in blocks}) == 4
    config_checks["hf517_baseline"] = first["status"]["transformers"] == "5.17.0"
    # The common generation loop/bridge is the evidence for identical timing scope and algorithms.
    for source in ("direct_opt/run_joint.py", "vendor/hf517_vispec_adapter.py", "vendor/vispec_joint_adapter.py"):
        hashes = [b["status"]["source_sha256"].get(source) for b in blocks]
        config_checks["same_source."+source] = all(hashes) and len(set(hashes)) == 1
    ids = [r["request_id"] for r in first["requests"]]
    maps = [{r["request_id"]: r for r in b["formal"]} for b in blocks]
    require(all(set(m) == set(ids) for m in maps), "Cross-process request coverage differs")
    pairs, per_request = [], []
    for request_id in ids:
        four = [m[request_id] for m in maps]
        agreements = {blocks[i]["name"]: equality(four[0], four[i]) for i in range(1, 4)}
        native, delta = aggregate([four[0], four[3]]), aggregate([four[1], four[2]])
        per_request.append({"request_id": request_id, "comparison_with_00_native": agreements,
            "all_four_exact": all(all(c.values()) for c in agreements.values()),
            "output_lengths_in_run_order": [r["num_tokens"] for r in four],
            "native": native, "delta": delta, "throughput_change_pct": improvement(native, delta)})
        for label, ni, di in (("AB", 0, 1), ("BA", 3, 2)):
            nr, dr = four[ni], four[di]
            ng, dg = aggregate([nr]), aggregate([dr])
            pairs.append({"request_id": request_id, "block_pair": label,
                "first_arm": "native" if label == "AB" else "delta", "equality": equality(nr, dr),
                "round_differences": round_differences(nr, dr),
                "length_ratio": dr["num_tokens"]/nr["num_tokens"],
                "within_1_10": 10*dr["num_tokens"] <= 11*nr["num_tokens"],
                "native": ng, "delta": dg, "throughput_change_pct": improvement(ng, dg)})
    arms = {label: aggregate([r for b in blocks if b["label"] == label for r in b["formal"]])
            for label in ("native", "delta")}
    block_pairs = []
    for label, ni, di in (("AB", 0, 1), ("BA", 3, 2)):
        na, da = blocks[ni]["aggregate"], blocks[di]["aggregate"]
        block_pairs.append({"order": label, "native_block": blocks[ni]["name"],
            "delta_block": blocks[di]["name"], "native": na, "delta": da,
            "throughput_change_pct": improvement(na, da)})
    details = [dict(row, block=b["name"], arm=b["label"]) for b in blocks for row in b["details"]]
    formal = [r for r in details if r["phase"] == "measure"]
    hw = [r["hardware"] for r in formal]
    hardware = {"valid_timing_counts": dict(Counter(str(h["valid_timing"]) for h in hw)),
        "process_queries": sum(h["process_query_count"] for h in hw),
        "process_query_failures": sum(h["process_query_failures"] for h in hw),
        "all_visibility_complete": all(h["process_visibility_complete"] is True for h in hw),
        "all_own_pids_allowlisted": all(h["own_pid_allowlisted"] for h in hw),
        "foreign_pids": sorted({p for h in hw for p in h["foreign_pids"]}),
        "foreign_except_known_graphics": sorted({p for h in hw for p in h["foreign_except_known_graphics"]}),
        "known_graphics_pids_supplied": list(known_graphics)}
    valid = all(config_checks.values()) and all(r["all_four_exact"] for r in per_request)
    valid = valid and all(r["receipt_valid"] and r["natural_eos"] and all(r["timing_scope_checks"].values())
                          for r in details)
    valid = valid and all(r["memory"].get("consistent_complete_peaks", True) for r in formal)
    comparable = all(config_checks.values()) and all(
        all(c[field] for field in ("input_shapes", "video_grid_thw", "timestamps", "result.prompt_tokens",
                                  "result.precision", "result.budget_trace"))
        for item in per_request for c in item["comparison_with_00_native"].values())
    comparable = comparable and all(r["receipt_valid"] and all(r["timing_scope_checks"].values())
                                    for r in details)
    comparable = comparable and all(r["memory"].get("consistent_complete_peaks", True) for r in formal)
    equality_summary = {field: sum(all(c[field] for c in r["comparison_with_00_native"].values())
                                  for r in per_request)
                        for field in list(ROW_FIELDS)+["result."+f for f in RESULT_FIELDS]}
    summaries = [{k: b[k] for k in ("name", "label", "source", "hashes", "aggregate")} |
        {"process": {k: b["status"].get(k) for k in ("pid", "started_at", "completed_at", "load_seconds",
            "experimental_setup_seconds", "torch", "transformers", "device_name", "memory_layout",
            "deployment_adaptor")}, "args": b["status"]["args"],
         "recorded_source_sha256": b["status"]["source_sha256"]} for b in blocks]
    return {"schema": "native_delta_abba_offline_v1", "source": str(root.resolve()),
        "tool_sha256": digest(__file__), "helper_sha256": {name: digest(Path(__file__).with_name(name))
            for name in ("assess_visual_kv.py", "audit_delta_commit_pairs.py", "summarize_dram_campaign.py")},
        "mechanical_exact_comparison_pass": bool(valid), "configuration_checks": config_checks,
        "whole_method_comparison_receipts_pass": bool(comparable),
        "requests_equal_across_all_four_runs": equality_summary,
        "all_saved_generation_receipts_valid": all(r["receipt_valid"] for r in details),
        "all_saved_natural_eos": all(r["natural_eos"] for r in details),
        "aggregate": arms, "throughput_change_pct": improvement(arms["native"], arms["delta"]),
        "block_pairs": block_pairs, "per_request": per_request, "pairs": pairs,
        "blocks": summaries, "rows": details, "hardware": hardware,
        "quality_review_pass": None,
        "limits": [
            "Native means this project's HF 5.17 ViSpec compatibility path, FP16 Target/Draft and FP32 output head, Target math SDPA; not every possible upstream/fused ViSpec implementation.",
            "Candidate is shared embedding + compiled adaptor + active-width full-history KV + delta commit; aggregate difference does not isolate delta commit alone.",
            "Identical final tokens can coexist with different Draft tree selection/round boundaries. Such differences remain reported; this whole-method comparison need not satisfy the stricter full-round equality required for a commit-only ablation.",
            "Four independent loaded processes execute N/D/D/N. Adjacent blocks are compared AB and BA; requests are not interleaved between processes.",
            "Each process warms only its first manifest request. Model loading/installation and warmup are excluded; remaining per-request setup/compilation actually incurred is included.",
            "Primary e2e includes request preparation, policy preparation, generation, text decode and runtime finalization. Generation excludes these outer phases and includes visual encoding, language prefill and speculative generation; it is not decode-only throughput.",
            "Sequential single-request tokens/s, including EOS; not concurrent serving capacity or latency-to-first-token.",
            "Two runs per arm/video do not establish robust variance or unseen-video quality; exact text preserves any original errors.",
            "Input tensors/token values are not saved; manifest, shapes, grid, timestamps and prompt count are checked.",
            "Raw timing flags remain unchanged. Process polling, including only-known-Xorg observations, cannot establish strict isolation or remove thermal/order effects.",
            "Commit bytes are logical operator receipts, not measured DRAM. No whole-model MAC/IO inference is made."]}


def render(report):
    n, d = (report["aggregate"][label] for label in ("native", "delta"))
    lines = ["# 原版与当前方案 ABBA 吞吐审计", "",
        "4 个独立进程按原版／当前／当前／原版运行；每进程首题预热 1 次，正式各 12 题。", "",
        "| 指标 | 原版 | 当前方案 | 变化 |", "|---|---:|---:|---:|",
        f"| 正式请求 | {n['formal_count']} | {d['formal_count']} | — |",
        f"| 输出 token（含 EOS） | {n['output_tokens_including_eos']} | {d['output_tokens_including_eos']} | — |"]
    for phase, name in (("e2e", "完整请求"), ("generation", "生成调用")):
        lines += [f"| {name}总秒数 | {n[phase]['total_seconds']:.6f} | {d[phase]['total_seconds']:.6f} | — |",
            f"| {name} token/s | {n[phase]['tokens_per_second']:.5f} | {d[phase]['tokens_per_second']:.5f} | {report['throughput_change_pct'][phase]:+.4f}% |"]
    lines += ["", f"整体方案比较的输入、配置与内部回执核验：**{report['whole_method_comparison_receipts_pass']}**。"
        f"严格完整轨迹等价：**{report['mechanical_exact_comparison_pass']}**。"
        "检查全部输出 IDs、完整 rounds、forward counts、输入元数据、自然 EOS 和提交回执；不自动认定视频事实正确。", "",
        f"四次输出 IDs 一致 {report['requests_equal_across_all_four_runs']['output_token_ids']}/12；"
        f"完整 rounds 一致 {report['requests_equal_across_all_four_runs']['result.rounds']}/12；"
        f"forward counts 一致 {report['requests_equal_across_all_four_runs']['result.forward_counts']}/12。",
        f"全部保存轨迹内部校验通过={report['all_saved_generation_receipts_valid']}；"
        f"全部自然 EOS={report['all_saved_natural_eos']}。完整 round 差异逐项保存在 JSON 的 pairs 中。", "",
        "该结果可用于本次整体方案的吞吐比较；候选同时改变适配器与缓存维护，不能据此隔离 KV commit 的贡献。", "",
        "| 进程块顺序 | 完整请求吞吐变化 | 生成调用吞吐变化 |", "|---|---:|---:|"]
    for pair in report["block_pairs"]:
        p = pair["throughput_change_pct"]
        lines.append(f"| {pair['order']} | {p['e2e']:+.4f}% | {p['generation']:+.4f}% |")
    lines += ["", "| 视频 | 四次输出长度 | 全部等价 | 完整请求吞吐变化（两次合计） | 生成吞吐变化 |",
              "|---|---|---|---:|---:|"]
    for item in report["per_request"]:
        change = item["throughput_change_pct"]
        lines.append(f"| {item['request_id']} | {item['output_lengths_in_run_order']} | {item['all_four_exact']} | {change['e2e']:+.4f}% | {change['generation']:+.4f}% |")
    h = report["hardware"]
    lines += ["", "## 硬件与计时边界", "",
        f"正式阶段进程观察 {h['process_queries']} 次，失败 {h['process_query_failures']} 次；完整可见性 {h['all_visibility_complete']}。",
        f"foreign PID={h['foreign_pids']}，人工指定既知图形 PID 外={h['foreign_except_known_graphics']}。",
        f"原 valid_timing 分布：{h['valid_timing_counts']}；原样保留，不声明严格独占。",
        "完整请求包含候选 runtime finalization；生成调用指标不含此项，不能替代完整请求指标。", "",
        "## 范围限制", ""]
    lines += ["- "+limit for limit in report["limits"]]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("campaign", type=Path, help="Contains 00_native, 01_delta, 02_delta, 03_native")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--known-graphics-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    report = audit(args.campaign, known_graphics=args.known_graphics_pid)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (args.out/"audit.md").write_text(render(report))
    print(json.dumps({key: report[key] for key in
        ("mechanical_exact_comparison_pass", "aggregate", "throughput_change_pct", "hardware")}, indent=2))


if __name__ == "__main__":
    main()
