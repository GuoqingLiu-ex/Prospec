"""Sequential Nsight Compute campaign; never changes another GPU process.

This diagnostic driver only launches profile_ring_dram.py. The child keeps the
project's cooperative GPU lock. Raw NVML observations cover load and profiling.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import threading
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--scope", choices=("generation", "maintenance"), default="generation")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--only", nargs="+", choices=("native_full", "optimized_full", "protected512"))
    parser.add_argument("--allow-graphics-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=False, parents=True)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from hardware_metrics import _make_backend
    backend = _make_backend(args.gpu)
    stop = threading.Event()
    observations = []

    def observe():
        row = {"time_unix": time.time()}
        for key, call in (("compute", lambda: backend.processes("compute")),
                          ("graphics", lambda: backend.processes("graphics")),
                          ("utilization", backend.utilization)):
            try:
                row[key] = call()
            except Exception as exc:
                row[key + "_error"] = repr(exc)
        return row

    def watch():
        with (out / "gpu_observations.jsonl").open("w") as stream:
            while not stop.is_set():
                row = observe()
                observations.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                stop.wait(0.5)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    status = {"status": "starting", "scope": args.scope, "gpu": args.gpu,
              "gpu_uuid": backend.uuid(), "gpu_name": backend.name(),
              "driver": backend.driver_version(), "pid": os.getpid(), "completed": [],
              "allowed_resident_graphics_pids": args.allow_graphics_pid}
    save(out / "campaign.json", status)
    arms = [("native_full", "native", "native", "full"),
            ("optimized_full", "shared_embedding", "compiled", "full"),
            ("protected512", "shared_embedding", "compiled", "protected_prompt")]
    try:
        for name, memory, adaptor, policy in arms:
            if args.only and name not in args.only:
                continue
            for check in range(2):
                row = observe()
                if row.get("compute_error") or row.get("graphics_error"):
                    raise RuntimeError("Cannot verify GPU process list: " + str(row))
                graphics = [p for p in row["graphics"] if p["pid"] not in args.allow_graphics_pid]
                if row["compute"] or graphics or row.get("utilization", {}).get("gpu_util_pct", 100) != 0:
                    raise RuntimeError("GPU occupied before next arm; preserved existing work: " + str(row))
                if check == 0:
                    time.sleep(2)
            arm = out / name
            arm.mkdir()
            ncu = ["/usr/local/cuda/bin/ncu", "--config-file", "off",
                   "--target-processes", "application-only", "--replay-mode", "app-range",
                   "--cache-control", "none", "--clock-control", "none",
                   "--pipeline-boost-state", "dynamic", "--disable-extra-suffixes",
                   "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum",
                   "--export", str(arm / "dram"), "--csv", "--page", "raw",
                   "--log-file", str(arm / "metrics.csv")]
            if args.scope == "maintenance":
                ncu += ["--nvtx", "--nvtx-include", "KV_MAINT/"]
            workload = [sys.executable, str(here / "profile_ring_dram.py"),
                        "--profile-scope", args.scope,
                        "--ring-policy", policy, "--recent-capacity", "512",
                        "--memory-layout", memory, "--adaptor", adaptor,
                        "--out", str(arm / "workload"), "--physical-gpu", str(args.gpu),
                        "--requests", str(here.parents[1] / "coord/quality_requests.jsonl"),
                        "--request-id", "direct_quality_vdc_v_sPK-sbHIb3w",
                        "--warmup", "1", "--repeats", str(args.repeats),
                        "--max-new-tokens", "512"]
            cmd = ncu + workload
            save(arm / "command.json", cmd)
            status.update(status="running", current=name, updated_at=time.time())
            save(out / "campaign.json", status)
            print("START " + name, flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                       PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
            with (arm / "application.log").open("w") as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            save(arm / "exit.json", {"returncode": result.returncode, "time_unix": time.time()})
            if result.returncode:
                raise RuntimeError("NCU/workload failed for " + name)
            status["completed"].append(name)
            print("DONE " + name, flush=True)
        status.update(status="completed", current=None, updated_at=time.time())
    except Exception as exc:
        status.update(status="failed", error=repr(exc), updated_at=time.time())
        raise
    finally:
        stop.set()
        thread.join(timeout=5)
        backend.close()
        save(out / "campaign.json", status)


if __name__ == "__main__":
    main()
