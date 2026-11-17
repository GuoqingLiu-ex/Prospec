"""Saved-result audit of paired delta versus implicit-layout deployments.

JSON/stdlib only: no models, SSH, Torch or GPU. Full generation trajectories
must match for the result to qualify as an isolated implementation comparison.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from assess_visual_kv import generation, memory_evidence
from audit_delta_commit_pairs import DELTA, digest, require
from audit_native_delta_abba import aggregate, equality, improvement, hardware_evidence, round_differences

BASE = DELTA
CANDIDATE = BASE + "_implicit_layout"
VARIANTS = {BASE: "active_width_delta_commit", CANDIDATE: "active_width_delta_commit_implicit_layout"}


def receipt(row):
    info = row["adaptor_preparation"]["ring_kv"]
    implicit = row["arm"] == CANDIDATE
    require(info["variant"] == VARIANTS[row["arm"]], "Incorrect runtime variant")
    require(info["finalized"] is True and info["failed"] is False, "Incomplete/failed runtime")
    require(info["mode"] == "protected_prompt" and info["recent_capacity"] == 512
            and info["tree_capacity"] == 30, "Incorrect cache capacities")
    require(info["attention_backend_changed"] is False and info["kv_quantized"] is False
            and info["target_history_eviction"] is False, "Changed attention/storage coverage")
    trees = [r for r in row["result"]["rounds"] if r["kind"] == "tree"]
    require(len(trees) == len(info["rounds"]) == info["completed_verifications"], "Uncovered verification")
    layers = info["final_layer_metadata"]
    count, per_token = len(layers), info["bytes_per_token_all_target_layers"]
    require(count == 28 and per_token == 57344, "Unexpected Target KV dimensions")
    copied = launches = accepted = moved = 0
    for tree, step in zip(trees, info["rounds"]):
        k = len(tree["selected_tree_indices"])
        d = sum(i != source for i, source in enumerate(tree["selected_tree_indices"]))
        require(step["logical_base"] == tree["target_cache_before"] and
                step["semantic_length_after"] == tree["target_cache_after"] and
                step["nodes"] == tree["tree_nodes"] and step["committed_tokens"] == k,
                "Runtime and generator boundaries differ")
        require(step["all_layer_commit_reports_equal"] is True, "Per-layer commit reports differ")
        commit, = step["layer_commit_reports"]
        expected = d*per_token//count
        require(commit["copy_read_bytes"] == commit["copy_write_bytes"] == expected
                and commit["copy_total_bytes"] == 2*expected, "Incorrect delta payload bytes")
        require(commit["payload_backend"] == "triton" and commit["moved_tokens"] == d
                and commit["payload_kernel_launches"] == int(d > 0)
                and commit["scratch_payload_peak_bytes"] == 0, "Wrong payload kernel/temporary receipt")
        require(commit["accepted_tokens"] == commit["stored_tokens"] == k
                and commit["evicted_recent_tokens"] == 0
                and commit["traffic_is_measured_dram"] is False, "Invalid commit retention/scope")
        require(commit["previous_scratch_start"] == tree["target_cache_before"]
                and commit["next_scratch_start"] == tree["target_cache_after"], "Not contiguous active-width commit")
        if implicit:
            require(step["caller_attention_mask_reused"] is True
                    and step["caller_position_ids_reused"] is True
                    and step["mask_rematerialization_bytes"] == 0
                    and commit["metadata_update_write_bytes"] == 0,
                    "Implicit round rematerializes mask/metadata or replaces positions")
        copied += 2*expected*count
        launches += commit["payload_kernel_launches"]*count
        accepted += k
        moved += d
    owned = kv = metadata_writes = 0
    for layer in layers:
        require(layer["next_position"] == row["result"]["target_cache_length"]
                and layer["recent_count"] == accepted <= 512 and layer["active_tree_nodes"] == 0,
                "Invalid final KV coverage")
        require(layer["operation_counts"] == {"prefill": 1, "append_tree": len(trees), "commit": len(trees)},
                "Unexpected layer operation coverage")
        require(layer["positions_are_sequence_offsets"] is True and layer["rotated_keys_modified"] is False,
                "Position/RoPE storage interpretation changed")
        totals = layer["payload_copy_totals"]
        each = (info["prompt_tokens"]+sum(t["tree_nodes"] for t in trees))*per_token//count + copied//(2*count)
        require(totals["read_bytes"] == totals["write_bytes"] == each and totals["total_bytes"] == 2*each,
                "Final prefill/append/commit ledger differs")
        layer_kv, layer_owned = layer["allocated_kv_storage_bytes"], layer["owned_storage_bytes"]
        if implicit:
            require(layer["metadata_tensor_storage_bytes"] == layer["metadata_update_write_bytes"] == 0
                    and totals["metadata_update_write_bytes"] == 0 and layer_owned == layer_kv,
                    "Implicit layer retains persistent slot metadata")
        else:
            require(layer_owned-layer_kv == 9*layer["physical_capacity"], "Unexpected delta slot metadata capacity")
            expected_metadata_writes = 9*(layer["physical_capacity"]+info["prompt_tokens"]+
                2*sum(t["tree_nodes"] for t in trees)+accepted)
            require(totals["metadata_update_write_bytes"] == expected_metadata_writes,
                    "Delta initialization/append/commit metadata ledger differs")
        owned += layer_owned
        kv += layer_kv
        metadata_writes += totals["metadata_update_write_bytes"]
    for snapshot in (info["prefill_cache"], info["final_cache"]):
        require(snapshot["owned_storage_bytes"] == owned and snapshot["allocated_kv_storage_bytes"] == kv,
                "Owned backing capacity changed/misreported")
    if implicit:
        require(info["metadata_tensor_storage_bytes"] == info["metadata_update_write_bytes"] == 0
                and info["per_layer_metadata_gpu_operations"] == 0
                and info["mask_rematerialization_bytes"] == 0
                and info["caller_attention_mask_reused"] is True
                and info["round_tree_path_mrope_validation_retained"] is True,
                "Incorrect implicit metadata/mask summary")
    return {"variant": info["variant"], "committed_tokens": accepted, "moved_tokens": moved,
        "already_in_place_tokens": accepted-moved, "copy_total_bytes": copied,
        "payload_kernel_launches": launches, "owned_storage_bytes": owned,
        "kv_backing_bytes": kv, "persistent_slot_metadata_bytes": owned-kv,
        "metadata_update_logical_write_bytes": metadata_writes,
        "mask_rematerialization_bytes": info.get("mask_rematerialization_bytes"),
        "shared_tree_plan_and_validation_temporaries_retained": True,
        "shared_temporary_peak_bytes_measured": None,
        "scope": "Persistent per-layer owned storage and logical maintenance receipts; shared temporary tensors and actual DRAM are not included"}


def audit(root, known_graphics=()):
    status = json.loads((root/"status.json").read_text())
    schedule = json.loads((root/"schedule.json").read_text())
    requests = json.loads((root/"requests.json").read_text())
    rows = [json.loads(line) for line in (root/"results.jsonl").read_text().splitlines() if line.strip()]
    require(status["status"] == "completed" and status["paired_delta_baseline"] is True, "Incomplete/unpaired campaign")
    require(status["args"]["repeats"] == 2 and status["args"]["warmup"] == 1
            and status["args"]["max_new_tokens"] == 512, "Unexpected experiment size")
    require(status["memory_layout"] == "shared_embedding" and status["deployment_adaptor"] == "compiled",
            "Wrong common deployment")
    ids = [r["request_id"] for r in requests]
    require(len(ids) == len(set(ids)) == 12, "Expected exactly 12 manifest requests")
    key = lambda r: tuple(r[k] for k in ("request_id", "arm", "phase", "repeat"))
    require(len(rows) == len({key(r) for r in rows}) == status["completed_cells"] == status["total_cells"] == 50,
            "Missing/duplicate campaign cells")
    require([key(r) for r in rows] == [key(r) for r in schedule], "Recorded order differs from schedule")
    require({r["arm"] for r in rows} == {BASE, CANDIDATE}, "Wrong campaign arms")
    formal = [r for r in rows if r["phase"] == "measure"]
    require(len(formal) == 48, "Expected 48 formal cells")
    for arm in (BASE, CANDIDATE):
        warm = [r for r in rows if r["phase"] == "warmup" and r["arm"] == arm]
        require(len(warm) == 1 and warm[0]["request_id"] == ids[0], "Wrong per-arm warmup")
        require({(r["request_id"], r["repeat"]) for r in formal if r["arm"] == arm}
                == {(rid, rep) for rid in ids for rep in (0, 1)}, "Unmatched request/repeat coverage")
    details = []
    for row in rows:
        gen = generation(row, status)
        require(row["depth_mode"] == "fixed0" and row["visual_keep_ratio"] == 1.0
                and row["capture_enabled"] is False, "Visual/input path changed")
        require(row["result"]["precision"] == {"target": "float16", "output_head": "float32",
            "target_sdpa": "math", "draft": "float16", "draft_sdpa": "auto"}, "Unexpected precision/backend")
        for field in ("e2e_wall_s", "generation_wall_s", "model_wall_s", "prepare_wall_s",
                      "policy_prepare_wall_s", "output_decode_wall_s", "runtime_finalize_wall_s"):
            require(math.isfinite(row[field]) and row[field] >= 0, "Missing/invalid timing")
        require(row["e2e_wall_s"] > 0 and row["generation_wall_s"] > 0, "Zero timing denominator")
        model = row["generation_wall_s"]+row["output_decode_wall_s"]+row["runtime_finalize_wall_s"]
        scope = math.isclose(row["model_wall_s"], model, abs_tol=1e-6) and math.isclose(
            row["e2e_wall_s"], model+row["prepare_wall_s"]+row["policy_prepare_wall_s"], abs_tol=1e-3)
        info = row["adaptor_preparation"]["ring_kv"]
        details.append({"request_id": row["request_id"], "arm": row["arm"], "phase": row["phase"],
            "repeat": row["repeat"], "num_tokens": row["num_tokens"], "verify_rounds": row["verify_rounds"],
            "receipt_valid": gen["receipt_valid"], "trace_validation_error": gen["trace_validation_error"],
            "natural_eos": gen["natural_eos"], "timing_scope_valid": scope,
            "e2e_wall_s": row["e2e_wall_s"], "generation_wall_s": row["generation_wall_s"],
            "runtime_finalize_wall_s": row["runtime_finalize_wall_s"], "cache": receipt(row),
            "memory": memory_evidence(row, info),
            "hardware": hardware_evidence(row, status["pid"], known_graphics)})
    pairs = []
    for rep in (0, 1):
        for rid in ids:
            pair = [r for r in formal if r["request_id"] == rid and r["repeat"] == rep]
            require(len(pair) == 2, "Wrong pair allocation")
            b, c = (next(r for r in pair if r["arm"] == arm) for arm in (BASE, CANDIDATE))
            position = [rows.index(r) for r in pair]
            require(position[1]-position[0] == 1, "Pair is not adjacent in randomized schedule")
            checks = equality(b, c)
            bs, cs = (dict(r["scheme"]) for r in (b, c))
            require(bs.pop("cache_layout") == "delta" and cs.pop("cache_layout") == "implicit", "Wrong per-row routing")
            checks["same_other_scheme_fields"] = bs == cs
            br, cr = receipt(b), receipt(c)
            for field in ("committed_tokens", "moved_tokens", "copy_total_bytes", "payload_kernel_launches", "kv_backing_bytes"):
                checks["same_"+field] = br[field] == cr[field]
            ba, ca = aggregate([b]), aggregate([c])
            pairs.append({"request_id": rid, "repeat": rep, "first_arm": pair[0]["arm"],
                "equality": checks, "isolated_exact": all(checks.values()), "round_differences": round_differences(b, c),
                "baseline": ba, "candidate": ca, "throughput_change_pct": improvement(ba, ca),
                "owned_storage_reduction_bytes": br["owned_storage_bytes"]-cr["owned_storage_bytes"]})
    summaries = {label: aggregate([r for r in formal if r["arm"] == arm])
                 for label, arm in (("baseline", BASE), ("candidate", CANDIDATE))}
    groups = []
    for field, values in (("repeat", (0, 1)), ("request_id", ids)):
        for value in values:
            a, b = (aggregate([r for r in formal if r[field] == value and r["arm"] == arm])
                    for arm in (BASE, CANDIDATE))
            groups.append({field: value, "baseline": a, "candidate": b, "throughput_change_pct": improvement(a, b)})
    selected = [r for r in details if r["phase"] == "measure"]
    hardware = [r["hardware"] for r in selected]
    structural = {}
    for label, arm in (("baseline", BASE), ("candidate", CANDIDATE)):
        subset = [r for r in selected if r["arm"] == arm]
        structural[label] = {
            "commit_logical_rw_bytes": sum(r["cache"]["copy_total_bytes"] for r in subset),
            "payload_kernel_launches": sum(r["cache"]["payload_kernel_launches"] for r in subset),
            "slot_metadata_logical_write_bytes": sum(r["cache"]["metadata_update_logical_write_bytes"] for r in subset),
            "persistent_slot_metadata_bytes_per_request_range": [min(r["cache"]["persistent_slot_metadata_bytes"] for r in subset),
                max(r["cache"]["persistent_slot_metadata_bytes"] for r in subset)],
            "maximum_whole_request_allocated_bytes": max(r["memory"]["whole_request"]["allocated"]["row_peak_bytes"] for r in subset)}
    structural["owned_storage_reduction_per_request_range_bytes"] = [
        min(p["owned_storage_reduction_bytes"] for p in pairs), max(p["owned_storage_reduction_bytes"] for p in pairs)]
    structural["whole_request_allocated_peak_equal_pairs"] = sum(
        len({r["memory"]["whole_request"]["allocated"]["row_peak_bytes"] for r in selected
             if r["request_id"] == p["request_id"] and r["repeat"] == p["repeat"]}) == 1 for p in pairs)
    structural["shared_temporary_peak_bytes_measured"] = None
    structural["scope"] = "Per-request persistent backing and cumulative logical writes across 24 formal requests; not DRAM or total temporary memory"
    exact = all(p["isolated_exact"] for p in pairs) and all(r["receipt_valid"] and r["natural_eos"]
        and r["timing_scope_valid"] and r["memory"]["consistent_complete_peaks"] for r in details)
    return {"schema": "implicit_delta_pair_offline_v1", "source": str(root.resolve()),
        "source_sha256": {n: digest(root/n) for n in ("status.json", "results.jsonl", "requests.json", "schedule.json")},
        "tool_sha256": digest(__file__), "helper_sha256": {n: digest(Path(__file__).with_name(n)) for n in
            ("audit_native_delta_abba.py", "audit_delta_commit_pairs.py", "assess_visual_kv.py", "summarize_dram_campaign.py")},
        "recorded_source_sha256": status["source_sha256"], "isolated_exact_comparison_pass": exact,
        "aggregate": summaries, "throughput_change_pct": improvement(summaries["baseline"], summaries["candidate"]),
        "structural_evidence": structural,
        "pairs": pairs, "groups": groups, "rows": details,
        "hardware": {"valid_timing_counts": dict(Counter(str(h["valid_timing"]) for h in hardware)),
            "queries": sum(h["process_query_count"] for h in hardware),
            "query_failures": sum(h["process_query_failures"] for h in hardware),
            "all_visibility_complete": all(h["process_visibility_complete"] is True for h in hardware),
            "foreign_pids": sorted({pid for h in hardware for pid in h["foreign_pids"]}),
            "foreign_except_known_graphics": sorted({pid for h in hardware for pid in h["foreign_except_known_graphics"]})},
        "limits": ["Two paired repeats per video; no robust uncertainty or held-out quality claim.",
            "All outputs/full rounds must match to call this an isolated implementation comparison.",
            "Whole request includes preparation, generation, text decoding and runtime finalization; model loading, installation and warmup are excluded.",
            "Generation includes visual encoding and prefill; its throughput excludes outer preparation and finalization.",
            "Sequential request throughput, not concurrent serving throughput.",
            "Slot metadata is zero per layer in the candidate, but shared tree plans, validation temporaries, and the existing caller mask remain; no claim of zero total metadata or temporary allocation.",
            "Persistent storage and logical writes are not DRAM traffic; payload copy/kernel equality is separately checked.",
            "Process observations do not prove strict isolation. Raw valid_timing flags are retained.",
            "Exact output agreement preserves baseline factual errors; it is not a new video-quality review."]}


def render(report):
    a, b = report["aggregate"]["baseline"], report["aggregate"]["candidate"]
    lines = ["# Implicit layout 与 delta 配对审计", "",
        f"严格隔离比较通过：**{report['isolated_exact_comparison_pass']}**。每臂 24 正式请求，另各 1 次首题预热。", "",
        "| 指标 | 旧 delta | Implicit | 变化 |", "|---|---:|---:|---:|",
        f"| 输出 token（含 EOS） | {a['output_tokens_including_eos']} | {b['output_tokens_including_eos']} | — |"]
    for phase, label in (("e2e", "完整请求"), ("generation", "生成调用")):
        lines.append(f"| {label} token/s | {a[phase]['tokens_per_second']:.5f} | {b[phase]['tokens_per_second']:.5f} | {report['throughput_change_pct'][phase]:+.4f}% |")
    lines += ["", "| 分组 | 完整请求吞吐变化 | 生成调用吞吐变化 |", "|---|---:|---:|"]
    for group in report["groups"]:
        label = group.get("request_id", "repeat "+str(group.get("repeat")))
        p = group["throughput_change_pct"]
        lines.append(f"| {label} | {p['e2e']:+.4f}% | {p['generation']:+.4f}% |")
    h = report["hardware"]
    structural = report["structural_evidence"]
    old, new = structural["baseline"], structural["candidate"]
    savings = structural["owned_storage_reduction_per_request_range_bytes"]
    lines += ["", f"硬件观察 {h['queries']} 次，失败 {h['query_failures']} 次，foreign PID={h['foreign_pids']}。",
        f"原 valid_timing：{h['valid_timing_counts']}；人工指定图形进程以外={h['foreign_except_known_graphics']}。",
        "每对完整 round 差异、存储容量、逐层逻辑写入和三阶段/整请求峰值见 JSON。", "",
        "## 元数据与 IO 结构", "",
        f"每请求持久 slot 元数据减少 {savings[0]:,}–{savings[1]:,} bytes（{savings[0]/2**20:.4f}–{savings[1]/2**20:.4f} MiB）；KV 主体驻留相同。",
        f"24 次正式请求的 slot 元数据逻辑写入累计 {old['slot_metadata_logical_write_bytes']:,} → {new['slot_metadata_logical_write_bytes']:,} bytes，含初始化、append 和 commit。",
        f"KV payload commit 逻辑读写两臂均 {old['commit_logical_rw_bytes']/2**20:.6f} MiB，payload kernel 均 {old['payload_kernel_launches']:,} 次。",
        f"整请求 allocated 峰值相同 {structural['whole_request_allocated_peak_equal_pairs']}/24 对；最大值两臂均 {old['maximum_whole_request_allocated_bytes']:,} bytes。",
        "仅逐层 slot 元数据与重建 mask 被移除；共享 tree plan、校验临时张量及调用方 mask 仍保留。", "",
        "上述结构减少可作为后续端侧验证的依据；没有据此推断实际 DRAM、能耗或端侧吞吐，也未跨批除以旧 native 结果声称超原版。", "",
        "## 范围限制", ""]
    lines.extend("- "+limit for limit in report["limits"])
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--known-graphics-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    result = audit(args.run, args.known_graphics_pid)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    (args.out/"audit.md").write_text(render(result))
    print(json.dumps({k: result[k] for k in ("isolated_exact_comparison_pass", "aggregate", "throughput_change_pct", "hardware")}, indent=2))


if __name__ == "__main__":
    main()
