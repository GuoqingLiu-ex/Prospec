"""GPU timing of the original and cheap local policies on the same actions.

Hidden states are synthetic at the vision width. Grids are the captured
pre-merger video grids. This does not skip transformer blocks and is not a
long-answer quality result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "training_free"), str(ROOT / "training_free_opt"), str(ROOT / "vendor")]


def _vision(config_path: Path) -> dict:
    payload = json.loads(config_path.read_text())
    vision = payload.get("vision_config") or payload
    return {
        "spatial_merge_size": int(vision["spatial_merge_size"]),
        "patch_size": int(vision["patch_size"]),
        "window_size": int(vision["window_size"]),
        "hidden_size": int(vision["hidden_size"]),
        "temporal_patch_size": int(vision.get("temporal_patch_size", 2)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    spec = _vision(Path(args.config))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    from run_experiment import cooperative_lock
    from hardware_metrics import HardwareSampler
    from local_depth import LocalDepthPolicy
    from cheap_local_depth import CheapLocalDepthPolicy
    from test_local_depth import Visual, limits
    from run_gpu_profile import timed_pairs
    from visual_grid import single_video_grid

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    thresholds = limits(0.2, 0.1, 0.2, 0.1)
    with cooperative_lock(args.physical_gpu, out), torch.inference_mode():
        visual = Visual().cuda().eval()
        visual.spatial_merge_size = spec["spatial_merge_size"]
        visual.patch_size = spec["patch_size"]
        visual.window_size = spec["window_size"]
        visual.config.temporal_patch_size = spec["temporal_patch_size"]
        old = LocalDepthPolicy(visual, thresholds=thresholds)
        new = CheapLocalDepthPolicy(visual, thresholds=thresholds)
        payload = {
            "process_index": args.process_index,
            "vision": spec,
            "gpu_passed": False,
            "note": "Same 0/1/2 actions. Times the statistics gate, not a skipped visual block.",
            "cases": {},
            "timing": {},
        }
        names = sorted(p.stem for p in Path(args.features).glob("*.pt") if p.name != "adaptor.pt")
        sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.05, process_interval_s=0.2)
        sampler.start()
        for name in names:
            loaded = torch.load(Path(args.features) / f"{name}.pt", map_location="cpu", weights_only=True)
            raw = single_video_grid(loaded["grid_thw"])
            grid = torch.tensor([list(raw)], dtype=torch.long)
            hidden = torch.rand(raw[0] * raw[1] * raw[2], spec["hidden_size"], device="cuda") + 0.25
            after = hidden + torch.randn_like(hidden) * 0.02
            after[: max(8, hidden.shape[0] // 8)] += 0.8
            old.prepare(grid, device="cuda")
            new.prepare(grid, device="cuda")
            old._start(hidden, grid)
            new._start(hidden, grid)
            before = old._means(hidden)
            before_new = new._means(hidden)
            torch.cuda.synchronize()
            cold0 = time.perf_counter()
            new._gate(before_new, after)
            torch.cuda.synchronize()
            cold_ms = (time.perf_counter() - cold0) * 1000
            old._gate(before, after)
            if old._current["actions"].tolist() != new._current["actions"].tolist():
                raise RuntimeError(f"{name} cheap actions differ from the original policy")
            device = hidden.device
            payload["timing"][name] = timed_pairs(
                lambda: old._gate(before, after),
                lambda: new._gate(before_new, after),
                args.warmup, args.repeats, device,
            )
            if old._current["actions"].tolist() != new._current["actions"].tolist():
                raise RuntimeError(f"{name} actions diverged during timing")
            payload["cases"][name] = {
                "raw_grid": list(raw),
                "tokens": int(hidden.shape[0]),
                "hidden": spec["hidden_size"],
                "actions": old._current["actions"].tolist(),
                "cheap_first_gate_ms": cold_ms,
            }
            old._current = None
            new._current = None
        hardware = sampler.stop()
        foreign = []
        for item in hardware.get("foreign_processes") or []:
            try:
                comm = Path(f"/proc/{item['pid']}/comm").read_text().strip()
            except OSError:
                comm = ""
            foreign.append({**item, "comm": comm})
        payload["hardware"] = {
            "foreign_processes": foreign,
            "graphics_only": bool(foreign) and all(row.get("comm") == "Xorg" for row in foreign),
        }
        slim = {k: v for k, v in payload.items() if k != "timing"}
        slim["timing_summary"] = {
            key: {k: val for k, val in value.items() if k not in {"rows", "order"}}
            for key, value in payload["timing"].items()
        }
        (out / "cheap_depth.json").write_text(json.dumps(slim, indent=2) + "\n")
        (out / "timing_raw.json").write_text(json.dumps({
            key: {"order": value["order"], "rows": value["rows"]}
            for key, value in payload["timing"].items()
        }) + "\n")
        print(json.dumps({"process": args.process_index, "cases": list(payload["cases"]),
                          "graphics_only_foreign": payload["hardware"]["graphics_only"]}))


if __name__ == "__main__":
    main()
