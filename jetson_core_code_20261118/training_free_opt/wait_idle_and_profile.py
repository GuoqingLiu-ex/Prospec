"""Poll for an actually idle GPU, then run the first adaptor profile once.

Idle = no compute PID and leftover VRAM <= 64 MiB. Low utilization is not idle.
Does not stop or close other users' processes.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
import os

IDLE_MIB = 64
INTERVAL_S = 1800  # 30 minutes; do not poll more often unless the user changes this.
PYTHON = "/data1/lgq2026/jev_spec/envs/jev/bin/python"
FEATURES = "/data1/lgq2026/jev_spec/runs/training_free_visual_20261001/perf_features"
ROOT = Path("/data1/lgq2026/jev_spec/runs/cursor1_training_free_optimization_20261002")
CODE = ROOT / "code"
OUT = ROOT / "profile_gpu_round1"
HEARTBEAT = ROOT / "idle_waiter_heartbeat.json"


def gpu_state():
    q = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.used,memory.total,utilization.gpu,uuid",
         "--format=csv,noheader,nounits"], text=True)
    p = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
         "--format=csv,noheader"], text=True)
    gpus = {}
    for line in q.strip().splitlines():
        idx, name, cap, used, total, util, uid = [x.strip() for x in line.split(",", 6)]
        gpus[uid] = {
            "index": int(idx), "name": name, "compute_cap": cap,
            "used_mib": int(used), "total_mib": int(total),
            "leftover_mib": int(total) - int(used), "util_pct": int(util),
            "compute": [],
        }
    for line in p.strip().splitlines():
        if not line.strip():
            continue
        uid, pid, mem, name = [x.strip() for x in line.split(",", 3)]
        if uid in gpus:
            gpus[uid]["compute"].append({"pid": int(pid), "used_mib": mem, "name": name})
    cards = list(gpus.values())
    # Idle means no compute PID and only a small resident footprint (about 14 MiB
    # on these cards). used_mib is occupied memory, not free memory.
    idle = [g for g in cards if not g["compute"] and g["used_mib"] <= IDLE_MIB]
    return cards, idle


def write_heartbeat(payload):
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    if OUT.exists():
        write_heartbeat({"status": "already_started", "out": str(OUT)})
        return
    while True:
        cards, idle = gpu_state()
        write_heartbeat({
            "status": "waiting",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "interval_s": INTERVAL_S,
            "idle": [g["index"] for g in idle],
            "occupancy": [{
                "index": g["index"], "used_mib": g["used_mib"], "free_mib": g["leftover_mib"],
                "util_pct": g["util_pct"], "n_compute": len(g["compute"]),
                "owners": sorted({c["name"] for c in g["compute"]}),
            } for g in cards],
        })
        if idle:
            gpu = idle[0]["index"]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{CODE}:{CODE/'training_free'}:{CODE/'training_free_opt'}:{CODE/'vendor'}"
            cmd = [PYTHON, str(CODE / "training_free_opt" / "run_gpu_profile.py"),
                   "--features", FEATURES, "--out", str(OUT),
                   "--physical-gpu", str(gpu), "--repeats", "50", "--warmup", "8"]
            write_heartbeat({"status": "launching", "gpu": gpu, "cmd": cmd})
            subprocess.check_call(cmd, env=env)
            write_heartbeat({"status": "finished", "gpu": gpu, "out": str(OUT)})
            return
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
