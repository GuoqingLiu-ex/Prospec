"""Resume only the implicit child after the narrowly specified native CPATH postcheck failure.

The completed native workload/counters are validated and retained. A fresh
directory gets one implicit profiling process and a provenance-bound comparison.
No original campaign/runtime file or global environment is changed. Import and
argument parsing do not initialize CUDA; only main() launches the requested job.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import run_implicit_dram_campaign as frozen
from profile_implicit_kv_dram import bind_ncu, validate_workload
from summarize_dram_campaign import require

KNOWN_APPEND = (
    "/data1/lgq2026/jev_spec/assets/build/python_headers/extracted/usr/include/python3.12",
    "/data1/lgq2026/jev_spec/assets/build/python_headers/extracted/usr/include",
)
EXPECTED_ERROR = "ValueError('Child GPU/CPATH differs from recorded environment')"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validated_cpath(prefix, effective):
    first = prefix.split(":")
    require(len(first) == 3 and all(first) and len(set(first)) == 3, "Exact three-path prefix required")
    expected = first+list(KNOWN_APPEND)
    require(effective.split(":") == expected, "Only the exact known trailing loader append is recoverable")
    return {"verified_prefix": prefix, "native_effective": effective,
            "known_loader_append": list(KNOWN_APPEND), "prefix_order_preserved": True,
            "new_child_launch_cpath": effective,
            "scope": "Verified three paths keep precedence; only known original loader suffix retained"}


def validate_native(root, args, headers):
    root = Path(root)
    campaign = json.loads((root/"campaign.json").read_text())
    require(campaign["status"] == "failed" and campaign.get("error") == EXPECTED_ERROR,
            "Recovery requires the terminal exact CPATH postcheck failure")
    require(campaign["completed"] == [] and campaign["current"] == "native"
            and campaign["modes"] == ["native", "implicit"] and not (root/"implicit").exists(),
            "Recovery must not repeat or overwrite any candidate execution")
    require(campaign["gpu"] == args.gpu and campaign["protocol"]["request_id"] == args.request_id,
            "Recovery GPU/request differs from original campaign")
    require(campaign["header_environment"]["CPATH"] == headers["CPATH"], "Verified header prefix changed")
    require(campaign["header_environment"]["observed_header_sha256"] == headers["observed_header_sha256"],
            "Verified header files changed after native")
    require(campaign["allowed_resident_graphics_pids"] == args.allow_graphics_pid,
            "Recovery graphics allowlist differs")
    require(campaign.get("monitor_error") is None, "Native monitoring failed")
    native = root/"native"
    status, rows, validation = validate_workload(native/"workload", "native", args.request_id)
    exit_record = json.loads((native/"exit.json").read_text())
    launch = json.loads((native/"launch.json").read_text())
    pid = json.loads((native/"pid_receipt.json").read_text())
    require(exit_record["returncode"] == 0 and exit_record["finished_unix_s"] > exit_record["started_unix_s"],
            "Native child has not completed successfully")
    require(status["pid"] == pid["workload_pid"] == campaign["workload_pids"]["native"], "Native PID mismatch")
    require(status["args"]["physical_gpu"] == args.gpu, "The failure also involved a GPU mismatch")
    require(status["request_sha256"] == sha(args.requests), "Input manifest changed")
    require(launch["environment_overrides"]["CPATH"] == headers["CPATH"], "Native initial CPATH differs")
    require(launch["environment_overrides"]["CUDA_VISIBLE_DEVICES"] == str(args.gpu), "Native device mapping differs")
    require(all(r["hardware"]["device"]["uuid"] == campaign["gpu_uuid"] for r in rows), "Native GPU UUID mismatch")
    cpath = validated_cpath(headers["CPATH"], status["profiler_effective_cpath"])
    require(all(Path(path).is_dir() for path in KNOWN_APPEND), "Known loader suffix directories missing")
    command = json.loads((native/"command.json").read_text())
    expected_options = {"--target-processes": "application-only", "--replay-mode": "app-range",
        "--cache-control": "none", "--clock-control": "none", "--profile-mode": "native",
        "--profile-scope": "generation", "--profile-calls": "2", "--warmup": "1", "--repeats": "2",
        "--max-new-tokens": "512", "--request-id": args.request_id, "--physical-gpu": str(args.gpu),
        "--metrics": "dram__bytes_read.sum,dram__bytes_write.sum", "--page": "raw"}
    for flag, value in expected_options.items():
        require(command.count(flag) == 1 and command[command.index(flag)+1] == value,
                "Native profiler command differs: "+flag)
    require(command[0] == args.ncu, "Recovery NCU executable differs")
    records = bind_ncu(status, rows, native/"metrics.csv")
    require(validation["all_calls_same_output"] and validation["all_calls_same_rounds"]
            and validation["all_calls_natural_eos"], "Native repetitions are not stable")
    current = frozen.source_hashes()
    require(current == campaign["source_sha256"], "Frozen campaign/profile/runtime sources changed")
    protocol = {key: status[key] for key in ("target", "draft", "request_sha256")}
    protocol.update({key: status["args"][key] for key in ("seed", "max_new_tokens")})
    bound = dict(records=records, validation=validation, protocol=protocol)
    files = ("campaign.json", "gpu_observations.jsonl", "native/command.json", "native/launch.json",
             "native/exit.json", "native/pid_receipt.json", "native/metrics.csv", "native/application.log",
             "native/workload/status.json", "native/workload/results.jsonl", "native/workload/schedule.json",
             "native/workload/requests.json", "native/workload/implicit_profile_validation.json")
    provenance = {"source_campaign": str(root.resolve()), "source_native": str(native.resolve()),
        "source_sha256": {name: sha(root/name) for name in files}, "cpath_recovery": cpath,
        "original_terminal_error": campaign["error"], "native_reexecuted": False,
        "native_bound": bound, "native_process_review": {key: campaign.get(key) for key in
            ("gpu_uuid", "external_process_observations", "resident_graphics_observations", "observation_errors", "attribution")},
        "recovery_source_sha256": sha(__file__),
        "native_binary_report": {"path": str(native/"dram.ncu-rep"),
            "bytes": (native/"dram.ncu-rep").stat().st_size if (native/"dram.ncu-rep").is_file() else None},
        "scope": "Reuse completed native after a narrow parent postcheck failure; no old ring/full baseline"}
    return provenance, dict(headers, CPATH=cpath["new_child_launch_cpath"],
                            recovery_verified_prefix=headers["CPATH"], recovery_known_append=list(KNOWN_APPEND))


def run_recovery(args, native_campaign, *, make_backend, run_process=None, sleep=None):
    headers = frozen.header_environment(args.header_environment)
    provenance, effective_headers = validate_native(native_campaign, args, headers)
    out = Path(args.out)
    require(not out.exists(), "Recovery output must be fresh")
    originals = {name: getattr(frozen, name) for name in ("MODES", "header_environment", "compare_bound", "save")}

    def compare(arms):
        require(set(arms) == {"implicit"}, "Recovery may execute only implicit")
        result = originals["compare_bound"]({"native": provenance["native_bound"], "implicit": arms["implicit"]})
        result["native_reference"] = str(out/"native_reference.json")
        result["reused_native_after_cpath_postcheck"] = True
        return result

    def save(path, value):
        if Path(path).name == "campaign.json":
            reference = out/"native_reference.json"
            if not reference.exists():
                originals["save"](reference, provenance)
            value = dict(value, recovery_native_reference=str(reference), recovery_native_reference_sha256=sha(reference),
                native_reexecuted=False, recovery_source_sha256=sha(__file__),
                cpath_recovery=provenance["cpath_recovery"])
        originals["save"](path, value)

    try:
        frozen.MODES = ("implicit",)
        frozen.header_environment = lambda path: effective_headers
        frozen.compare_bound, frozen.save = compare, save
        kwargs = {"make_backend": make_backend}
        if run_process is not None:
            kwargs["run_process"] = run_process
        if sleep is not None:
            kwargs["sleep"] = sleep
        return frozen.run_campaign(args, **kwargs)
    finally:
        for name, value in originals.items():
            setattr(frozen, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, add_help=False)
    parser.add_argument("--native-campaign", required=True)
    parser.add_argument("--validate-only", action="store_true")
    first, rest = parser.parse_known_args(argv)
    args = frozen.parse_args(rest)
    if first.validate_only:
        provenance, headers = validate_native(first.native_campaign, args,
            frozen.header_environment(args.header_environment))
        print(json.dumps({"native_validated": True, "native_pid": provenance["native_bound"]["records"][0]["pid"],
            "ranges": len(provenance["native_bound"]["records"]), "cpath_recovery": provenance["cpath_recovery"],
            "implicit_launched": False}, indent=2))
        return provenance
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from hardware_metrics import _make_backend
    return run_recovery(args, first.native_campaign, make_backend=_make_backend)


if __name__ == "__main__":
    main()
