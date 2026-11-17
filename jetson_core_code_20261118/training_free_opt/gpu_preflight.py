"""Report occupancy. Idle = no compute PID and leftover VRAM at or below the flock rule."""
from __future__ import annotations

import json
import subprocess


IDLE_MIB = 64


def main():
    q = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    p = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,used_memory,process_name",
         "--format=csv,noheader"],
        text=True,
    )
    gpus = []
    for line in q.strip().splitlines():
        idx, name, cap, used, total, util = [x.strip() for x in line.split(",")]
        leftover = int(total) - int(used)
        gpus.append({
            "index": int(idx), "name": name, "compute_cap": cap,
            "used_mib": int(used), "total_mib": int(total),
            "leftover_mib": leftover, "util_pct": int(util),
            "compute_pids": [],
        })
    # UUID mapping is incomplete here; attach all compute apps and let the caller inspect.
    compute = [line.strip() for line in p.strip().splitlines() if line.strip()]
    idle = [g for g in gpus if g["leftover_mib"] <= IDLE_MIB and not compute]
    # leftover-only is insufficient; attach compute rows for human/script pairing via nvidia-smi -q
    print(json.dumps({
        "idle_rule": "no compute PID and leftover VRAM <= 64 MiB; low util is not idle",
        "gpus": gpus,
        "compute_apps": compute,
        "possibly_idle_if_no_compute": [g["index"] for g in gpus if g["leftover_mib"] <= IDLE_MIB],
        "max_gpus_this_task": 2,
    }, indent=2))


if __name__ == "__main__":
    main()
