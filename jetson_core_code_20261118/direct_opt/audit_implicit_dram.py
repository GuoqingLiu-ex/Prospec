"""Offline native-to-implicit generation DRAM audit of freshly collected NCU data.

Reads this campaign's native/ and implicit/ folders, or an explicitly supplied
failed-parent native campaign plus its provenance-bound implicit recovery.
No Torch, CUDA, SSH or profiler launch. Range times remain diagnostic.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics

from assess_visual_kv import generation
from audit_delta_commit_pairs import digest, require
from audit_native_delta_abba import equality, round_differences
from audit_implicit_pairs import receipt as implicit_receipt
from profile_implicit_kv_dram import bind_ncu, validate_workload
from recover_implicit_dram_campaign import EXPECTED_ERROR, validated_cpath

METRICS = ("read_bytes", "write_bytes", "total_bytes")


def option(command, name):
    require(command.count(name) == 1, "Missing/duplicate command option "+name)
    return command[command.index(name)+1]


def check_command(command, mode, gpu, request):
    expected = {"--config-file": "off", "--target-processes": "application-only",
        "--replay-mode": "app-range", "--cache-control": "none", "--clock-control": "none",
        "--pipeline-boost-state": "dynamic", "--metrics": "dram__bytes_read.sum,dram__bytes_write.sum",
        "--page": "raw", "--profile-mode": mode, "--profile-scope": "generation", "--profile-calls": "2",
        "--physical-gpu": str(gpu), "--request-id": request, "--warmup": "1", "--repeats": "2",
        "--max-new-tokens": "512"}
    for flag, value in expected.items():
        require(option(command, flag) == value, "Wrong profiling option "+flag)
    require("--csv" in command and "--disable-extra-suffixes" in command, "Missing raw CSV export options")
    require(not any(str(item).startswith(("--nvtx", "--launch-skip", "--launch-count", "--kernel-name"))
                    for item in command), "Unexpected kernel/range filtering")
    require(any(Path(item).name == "profile_implicit_kv_dram.py" for item in command), "Wrong profiling entry")


def csv_metadata(path):
    text = path.read_text()
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    records = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    units = [r for r in records if r["ID"] == ""]
    require(units, "Missing NCU unit row")
    for row in units:
        require(row["dram__bytes_read.sum"] == row["dram__bytes_write.sum"] == "byte", "DRAM must use byte units")
        require(row["profiler__replayer_passes"] == "pass", "Replay count units differ")
        if row.get("gpu__time_duration.sum"):
            require(row["gpu__time_duration.sum"] == "ns", "Range duration must use ns")
    actual = [r for r in records if r["ID"].isdigit()]
    require(len(actual) == 2, "Expected exactly two aggregate raw range records")
    require(all(r["Kernel Name"] == "range" and r["Device"] == "0" for r in actual),
            "Expected aggregate ranges on CUDA visible device zero")
    return {"devices": sorted({r["Device"] for r in actual}),
        "process_names": sorted({r["Process Name"] for r in actual}),
        "display_names": sorted({r.get("device__attribute_display_name") for r in actual}),
        "warnings": [line for line in lines if line.startswith("==WARNING==")],
        "metric_scope": "dram__bytes_read.sum and dram__bytes_write.sum, raw byte units, each generation range"}


def process_review(observations, own_pids, allowed_graphics):
    external, resident, errors = [], [], []
    own_seen = set()
    times = [r["time_unix"] for r in observations]
    require(times == sorted(times), "Nonmonotonic process observation times")
    for row in observations:
        compute = {p["pid"] for p in row.get("compute", [])}
        graphics = {p["pid"] for p in row.get("graphics", [])}
        own_seen.update((compute | graphics) & own_pids)
        foreign = (compute | (graphics-allowed_graphics))-own_pids
        known = (graphics & allowed_graphics)-own_pids
        if foreign:
            external.append({"time_unix": row["time_unix"], "pids": sorted(foreign)})
        if known:
            resident.append({"time_unix": row["time_unix"], "pids": sorted(known)})
        if any(k.endswith("_error") for k in row):
            errors.append(row)
    return {"observations": len(observations), "own_pids_observed": sorted(own_seen),
        "external": external, "resident_graphics": resident, "errors": errors,
        "max_poll_gap_seconds": max((b-a for a,b in zip(times,times[1:])), default=None),
        "strict_exclusive_gpu_use_proven": False, "background_subtracted": False}


def campaign_inputs(root, native_campaign=None):
    """Keep the original failed parent and both independent observation streams."""
    campaign = json.loads((root/"campaign.json").read_text())
    require(campaign["status"] == "completed" and campaign["scope"] == "generation", "Incomplete/wrong campaign")
    require(campaign["native_ring_full_instrumentation"] is False
            and campaign["background_subtracted"] is False, "Wrong baseline/background scope")
    if native_campaign is None:
        require(campaign["completed"] == ["native", "implicit"] and campaign["modes"] == ["native", "implicit"],
                "Recovery requires explicit --native-campaign; otherwise both fresh arms are required")
        return campaign, {mode: (root, campaign) for mode in ("native", "implicit")}, None
    native_campaign = Path(native_campaign)
    old = json.loads((native_campaign/"campaign.json").read_text())
    require(old["status"] == "failed" and old.get("error") == EXPECTED_ERROR
            and old["completed"] == [] and old["current"] == "native", "Wrong original terminal failure")
    require(not (native_campaign/"implicit").exists(), "Original campaign unexpectedly executed implicit")
    require(campaign["completed"] == campaign["modes"] == ["implicit"] and campaign["native_reexecuted"] is False,
            "Recovery must execute only implicit")
    reference_path = root/"native_reference.json"
    require(digest(reference_path) == campaign["recovery_native_reference_sha256"], "Native reference hash differs")
    reference = json.loads(reference_path.read_text())
    require(reference["native_reexecuted"] is False and reference["original_terminal_error"] == old["error"],
            "Native reference claims a different source outcome")
    require(reference["recovery_source_sha256"] == campaign["recovery_source_sha256"], "Recovery entry hashes differ")
    for name, expected in reference["source_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid provenance-relative filename")
        require(digest(native_campaign/relative) == expected, "Original native evidence changed: "+name)
    status = json.loads((native_campaign/"native/workload/status.json").read_text())
    cpath = validated_cpath(old["header_environment"]["CPATH"], status["profiler_effective_cpath"])
    require(reference["cpath_recovery"] == campaign["cpath_recovery"] == cpath, "CPATH recovery receipt differs")
    require(campaign["header_environment"]["CPATH"] == cpath["new_child_launch_cpath"], "Wrong recovered launch CPATH")
    require(old["header_environment"]["observed_header_sha256"] == campaign["header_environment"]["observed_header_sha256"],
            "Verified leading header files changed")
    for key in ("gpu", "gpu_uuid", "allowed_resident_graphics_pids", "protocol", "source_sha256"):
        require(old[key] == campaign[key], "Recovered campaign protocol/source differs: "+key)
    recovery = {"original_campaign": str(native_campaign.resolve()), "original_parent_status": old["status"],
        "original_parent_error": old["error"], "native_rerun": False, "reference": reference,
        "reference_sha256": digest(reference_path), "two_separate_campaigns": True}
    return campaign, {"native": (native_campaign, old), "implicit": (root, campaign)}, recovery


def audit(root, native_campaign=None):
    campaign, sources, recovery = campaign_inputs(root, native_campaign)
    request = campaign["protocol"]["request_id"]
    streams = {}
    for directory, _ in sources.values():
        name = str(directory.resolve())
        if name not in streams:
            streams[name] = [json.loads(line) for line in (directory/"gpu_observations.jsonl").read_text().splitlines() if line.strip()]
            require(bool(streams[name]), "Missing independent process observations")
    observations = sorted((r for stream in streams.values() for r in stream), key=lambda r:r["time_unix"])
    allowed = set(campaign["allowed_resident_graphics_pids"])
    arms = {}
    for mode in ("native", "implicit"):
        source, source_campaign = sources[mode]
        path = source/mode
        source_observations = streams[str(source.resolve())]
        status, rows, validation = validate_workload(path/"workload", mode, request)
        measured = [r for r in rows if r["phase"] == "measure"]
        command = json.loads((path/"command.json").read_text())
        launch = json.loads((path/"launch.json").read_text())
        exit_record = json.loads((path/"exit.json").read_text())
        pid = json.loads((path/"pid_receipt.json").read_text())
        require(exit_record["returncode"] == 0, "Profiler child did not exit cleanly")
        require(status["pid"] == pid["workload_pid"] == source_campaign["workload_pids"][mode], "Workload PID receipts differ")
        require(launch["started_unix_s"] == exit_record["started_unix_s"]
                and exit_record["finished_unix_s"] > launch["started_unix_s"], "Invalid child lifespan")
        require(launch["environment_overrides"]["CUDA_VISIBLE_DEVICES"] == str(campaign["gpu"]), "CUDA device remap differs")
        require(status["args"]["physical_gpu"] == campaign["gpu"] and status["transformers"] == "5.17.0",
                "Unexpected device or HF baseline version")
        check_command(command, mode, campaign["gpu"], request)
        metadata = csv_metadata(path/"metrics.csv")
        records = bind_ncu(status, rows, path/"metrics.csv")
        require(all(record["passes"] == 1 for record in records), "One hardware-counter pass required")
        if recovery is not None and mode == "native":
            saved = recovery["reference"]["native_bound"]
            require(saved["records"] == records and saved["validation"] == validation,
                    "Native reference differs from reconstructed original CSV/workload binding")
        else:
            saved_binding = json.loads((path/"range_binding.json").read_text())
            require(saved_binding["verified"] is True and saved_binding["records"] == records,
                    "Saved range binding differs from actual CSV/workload reconstruction")
        if recovery is not None:
            cpath = recovery["reference"]["cpath_recovery"]
            require(status["profiler_effective_cpath"] == cpath["native_effective"], "Effective child header search differs")
            require(launch["environment_overrides"]["CPATH"] == cpath[
                "verified_prefix" if mode == "native" else "new_child_launch_cpath"], "Unexpected per-child launch CPATH")
        for row in rows:
            gen = generation(row, status)
            require(gen["receipt_valid"] and gen["natural_eos"], "Incomplete trace/EOS")
            require(row["hardware"]["device"]["uuid"] == campaign["gpu_uuid"], "Wrong physical GPU UUID")
            require(row["valid_timing"] is False and row["valid_memory"] is False and row["valid_energy"] is False,
                    "Profiled values must remain diagnostic")
            if mode == "implicit":
                implicit_receipt(row)
        # This is a child-lifetime window, not an exact GPU-range time window.
        window = [r for r in source_observations if launch["started_unix_s"]-.5 <= r["time_unix"] <= exit_record["finished_unix_s"]+.5]
        require(window, "No process observations near child lifespan")
        binary = path/"dram.ncu-rep"
        files = ("command.json", "launch.json", "exit.json", "pid_receipt.json", "metrics.csv", "application.log",
                 "workload/status.json", "workload/results.jsonl",
                 "workload/schedule.json", "workload/requests.json", "workload/implicit_profile_validation.json")
        if not (recovery is not None and mode == "native"):
            files += ("range_binding.json", "workload_validation.json")
        arms[mode] = {"pid": status["pid"], "source": str(path.resolve()), "records": records, "csv": metadata,
            "workload_validation": validation, "status_source_sha256": status["source_sha256"],
            "protocol": {key: status[key] for key in ("target", "draft", "request_sha256", "torch", "transformers")},
            "generation_args": {key: status["args"][key] for key in ("max_new_tokens", "seed")},
            "source_sha256": {name: digest(path/name) for name in files},
            "binary_report": {"exists_locally": binary.is_file(), "bytes": binary.stat().st_size if binary.is_file() else None,
                              "scope": "CSV is independently parsed; proprietary binary is preserved for later NCU re-export"},
            "process_review": process_review(window, {status["pid"]}, allowed),
            "child_lifespan": exit_record,
            "bytes_per_generation": {key: {"median": statistics.median(r[key] for r in records),
                "min": min(r[key] for r in records), "max": max(r[key] for r in records),
                "sum_two_generations": sum(r[key] for r in records)} for key in METRICS},
            "_measured": measured}
    require(arms["native"]["pid"] != arms["implicit"]["pid"], "Expected separate deployment processes")
    require(arms["native"]["protocol"] == arms["implicit"]["protocol"]
            and arms["native"]["generation_args"] == arms["implicit"]["generation_args"], "Fresh model/input protocol differs")
    require(arms["native"]["child_lifespan"]["finished_unix_s"] <= arms["implicit"]["child_lifespan"]["started_unix_s"],
            "Child processes overlapped")
    pairs = []
    for index in range(2):
        left, right = (arms[mode]["_measured"][index] for mode in ("native", "implicit"))
        a, b = (arms[mode]["records"][index] for mode in ("native", "implicit"))
        require((left["request_id"],left["repeat"]) == (right["request_id"],right["repeat"]), "Different paired request/repeat")
        checks = equality(left, right)
        pairs.append({"repeat": index, "equality": checks,
            "strict_full_trajectory_equal": all(checks.values()), "round_differences": round_differences(left,right),
            "observed_reduction_pct": {key: 100*(1-b[key]/a[key]) if a[key] else None for key in METRICS}})
    for arm in arms.values():
        del arm["_measured"]
    whole = process_review(observations, {arm["pid"] for arm in arms.values()}, allowed)
    stream_ranges = [{"source": name, "first_observation_unix": values[0]["time_unix"],
                      "last_observation_unix": values[-1]["time_unix"], "count": len(values)}
                     for name, values in streams.items()]
    stream_ranges.sort(key=lambda item:item["first_observation_unix"])
    whole["observation_streams"] = stream_ranges
    whole["gaps_between_campaign_streams_seconds"] = [right["first_observation_unix"]-left["last_observation_unix"]
        for left,right in zip(stream_ranges,stream_ranges[1:])]
    whole["gap_scope"] = "A gap between independent campaign monitors is unobserved; do not interpret joined streams as continuous monitoring"
    comparison = {key: 100*(1-arms["implicit"]["bytes_per_generation"][key]["median"]/
                             arms["native"]["bytes_per_generation"][key]["median"]) for key in METRICS}
    return {"schema": "fresh_native_implicit_generation_dram_v1", "source": str(root.resolve()),
        "tool_sha256": digest(__file__), "source_sha256": {name: digest(root/name) for name in
            ("campaign.json", "gpu_observations.jsonl", "comparison.json")},
        "arms": arms, "pairs": pairs, "range_pid_one_pass_binding_verified": True,
        "recovery": recovery,
        "observed_reduction_pct_of_per_generation_medians": comparison,
        "strict_full_trajectory_equal": all(p["strict_full_trajectory_equal"] for p in pairs),
        "all_output_ids_equal": all(p["equality"]["output_token_ids"] for p in pairs),
        "process_review": whole, "campaign_monitor_error": {name: value.get("monitor_error")
            for name, (_,value) in sources.items()},
        "ordinary_latency_or_energy_evidence": False, "quality_review_pass": None,
        "limits": [
            "Only this campaign's fresh native and implicit runs are compared; historical ring/full/active counters are excluded.",
            "When recovery is explicit, the original failed-parent native directory is read unmodified and checked against native_reference hashes. Its successful child/ranges remain valid; a separate candidate campaign and monitor gap are disclosed.",
            "Each byte count is one aggregate whole-generation hardware counter range; sum read+write once, do not multiply by layers, kernels or replay passes. Two-range sum represents two requests, not one.",
            "Scope includes visual encoding, Target prefill, adaptor, candidate cache conversion, Draft/Target decoding and KV commits. It excludes model loading/installation, request preparation, tokenizer output decoding and context cleanup.",
            "Native uses this project's HF 5.17 compatibility implementation with math SDPA. Implicit includes shared embedding, compiled adaptor and cache changes; this is a whole-pipeline comparison, not an isolated metadata/commit attribution.",
            "Output agreement does not guarantee identical Draft tree paths. Full-round differences remain visible and prevent a strict trace-isolation claim.",
            "NCU instrumentation changes timing and memory behavior; range duration, request wall, allocator peaks and energy under NCU are diagnostic, not ordinary performance measurements.",
            "Caches/clocks are uncontrolled, as recorded in NCU warnings. Resident graphics work is not subtracted or proven excluded from DRAM counters.",
            "Process polling covers campaign and child lifespan, not exact start/end timestamps of GPU ranges; sub-poll work can be missed.",
            "One video, two measured repetitions, fixed native-then-implicit order; no claim of cross-video/Jetson generalization, total model MAC reduction or concurrent throughput."]}


def render(report):
    lines = ["# Fresh native → implicit：完整生成 DRAM 审计", "",
        "每臂 1 次非采样预热、2 次完整生成采样；分别核对实际工作进程 PID、采样单元与单 pass。", "",
        "| 每次生成的中位计数 | Native GiB | Implicit GiB | 观测减少 |", "|---|---:|---:|---:|"]
    for key in METRICS:
        a,b = (report["arms"][mode]["bytes_per_generation"][key]["median"] for mode in ("native","implicit"))
        lines.append(f"| {key} | {a/2**30:.6f} | {b/2**30:.6f} | {report['observed_reduction_pct_of_per_generation_medians'][key]:+.4f}% |")
    p = report["process_review"]
    lines += ["", f"输出 IDs 一致：{report['all_output_ids_equal']}；完整验证轨迹一致：{report['strict_full_trajectory_equal']}。",
        "全部原始计数、逐 range PID/pass、round 差异与输入检查见 JSON。", "",
        f"进程观察 {p['observations']} 次；外部非许可工作观察 {len(p['external'])} 次，错误 {len(p['errors'])} 次；已知图形进程观察 {len(p['resident_graphics'])} 次。",
        "未扣除背景；这不是严格独占或端侧实测结论。NCU 的耗时、显存峰值和能耗不作为普通运行性能。", "",
        "## 范围限制", ""]
    if report.get("recovery"):
        lines.insert(-2, "保留旧 native 父进程的 CPATH 检查失败记录；其子进程 exit 0 与两个原始计数区间已重新核验，未重跑 native。")
        lines.insert(-2, f"两段 campaign 监控之间的未观测间隔：{p['gaps_between_campaign_streams_seconds']} 秒。")
    lines += ["- "+limit for limit in report["limits"]]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--native-campaign", type=Path,
                        help="Explicit original failed-parent campaign retained by the implicit recovery")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.campaign, args.native_campaign)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    (args.out/"audit.md").write_text(render(report))
    print(json.dumps({key: report[key] for key in ("range_pid_one_pass_binding_verified", "all_output_ids_equal",
        "strict_full_trajectory_equal", "observed_reduction_pct_of_per_generation_medians")}, indent=2))


if __name__ == "__main__":
    main()
