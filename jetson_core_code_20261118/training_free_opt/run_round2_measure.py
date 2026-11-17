"""Formal pairs for the round-1 negative cached result. Profiler stays off.

Does not overwrite profile_gpu_round1. One process per output directory.
HardwareSampler covers the measured interval. Foreign compute marks the
group contaminated and keeps the rows.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_experiment import cooperative_lock
    from hardware_metrics import HardwareSampler
    from cached_factorized import CachedFactorizedAdaptor, run_full_token
    from factorized_adaptor import FactorizedAdaptor
    from run_gpu_profile import error_metrics, timed_pairs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    class Adaptor(nn.Module):
        def __init__(self, metadata):
            super().__init__()
            self.hidden_size = metadata["hidden_size"]
            self.num_heads = metadata["num_heads"]
            self.num_q = metadata["num_q"]
            self.head_dim = self.hidden_size // self.num_heads
            d = self.hidden_size
            state = metadata["state_dict"]
            self.q = nn.Parameter(torch.empty(self.num_q, self.num_heads, self.head_dim))
            self.k_proj = nn.Linear(d, d, bias="k_proj.bias" in state)
            self.v_proj = nn.Linear(d, d, bias="v_proj.bias" in state)
            self.o_proj = nn.Linear(d, d, bias="o_proj.bias" in state)
            self.to(device="cuda", dtype=state["q"].dtype)
            self.load_state_dict(state)

        def forward(self, x):
            b, n, d = x.shape
            q = self.q.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)
            k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
            return self.o_proj(y.transpose(1, 2).reshape(b, self.num_q, d))

    with cooperative_lock(args.physical_gpu, out), torch.inference_mode():
        metadata = torch.load(Path(args.features) / "adaptor.pt", map_location="cuda", weights_only=True)
        adaptor = Adaptor(metadata).eval()
        torch.cuda.synchronize()
        cold0 = time.perf_counter()
        old = FactorizedAdaptor(adaptor, precision="model")
        torch.cuda.synchronize()
        cold_old_ms = (time.perf_counter() - cold0) * 1000
        cold1 = time.perf_counter()
        checked = CachedFactorizedAdaptor(adaptor, precision="model", check_projections=True)
        torch.cuda.synchronize()
        cold_checked_ms = (time.perf_counter() - cold1) * 1000
        quiet = CachedFactorizedAdaptor(adaptor, precision="model", check_projections=False)
        payload = {
            "process_index": args.process_index,
            "physical_gpu": args.physical_gpu,
            "gpu_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "profiler": "off",
            "gpu_passed": False,
            "selected_path": "FactorizedAdaptor (existing reorder), not CachedFactorizedAdaptor",
            "cold_init": {
                "old_construct_ms": cold_old_ms,
                "old_prepare_count": old.prepare_count,
                "old_prepare_time_ms": old.prepare_time_ms,
                "checked_construct_ms": cold_checked_ms,
                "checked_prepare_count": checked.prepare_count,
                "checked_prepare_time_ms": checked.prepare_time_ms,
                "note": "construct_ms includes the first prepare and CUDA sync. prepare_count must be 1. This is still not a process-cold CUDA context.",
            },
            "errors": {},
            "timing": {},
        }
        feature_dir = Path(args.features)
        names = sorted(p.stem for p in feature_dir.glob("*.pt") if p.name != "adaptor.pt")
        sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.05, process_interval_s=0.2)
        sampler.start()
        for name in names:
            loaded = torch.load(feature_dir / f"{name}.pt", map_location="cuda", weights_only=True)
            grid = None
            if isinstance(loaded, dict):
                grid = loaded.get("grid_thw")
                x = loaded["hidden_states"] if "hidden_states" in loaded else loaded["x"]
            else:
                x = loaded
            y_old = old(x)
            y_checked = checked(x)
            y_quiet = quiet(x)
            y_selected, selected_name = run_full_token(adaptor, quiet, x)
            payload["errors"][name] = {
                "n": int(x.shape[1]),
                "selected": selected_name,
                "checked_vs_old": error_metrics(y_checked, y_old),
                "quiet_vs_old": error_metrics(y_quiet, y_old),
                "selected_vs_old": error_metrics(y_selected, y_old),
                "grid_thw": grid,
            }
            device = x.device
            payload["timing"][f"{name}_aa_native"] = timed_pairs(lambda: adaptor(x), lambda: adaptor(x), args.warmup, args.repeats, device)
            payload["timing"][f"{name}_aa_old"] = timed_pairs(lambda: old(x), lambda: old(x), args.warmup, args.repeats, device)
            payload["timing"][f"{name}_old_vs_checked"] = timed_pairs(lambda: old(x), lambda: checked(x), args.warmup, args.repeats, device)
            payload["timing"][f"{name}_old_vs_quiet"] = timed_pairs(lambda: old(x), lambda: quiet(x), args.warmup, args.repeats, device)
            payload["timing"][f"{name}_old_vs_selected"] = timed_pairs(lambda: old(x), lambda: run_full_token(adaptor, quiet, x)[0], args.warmup, args.repeats, device)
        hardware = sampler.stop()
        payload["hardware"] = {
            "scope": hardware.get("energy_attribution"),
            "foreign_processes": hardware.get("foreign_processes"),
            "process_visibility_complete": hardware.get("process_visibility_complete"),
            "duration_s": hardware.get("measurement_duration_s"),
            "contaminated": bool(hardware.get("foreign_processes")),
        }
        payload["energy_claim"] = "not attributed; device counter includes whatever else was on the card"
        slim = {k: v for k, v in payload.items() if k != "timing"}
        slim["timing_summary"] = {key: {k: val[k] for k in val if k != "rows" and k != "order"} for key, val in payload["timing"].items()}
        (out / "round2.json").write_text(json.dumps(slim, indent=2) + "\n")
        (out / "timing_raw.json").write_text(json.dumps(payload["timing"]) + "\n")
        print(json.dumps({"process": args.process_index, "contaminated": payload["hardware"]["contaminated"], "cold": payload["cold_init"], "summary_keys": list(slim["timing_summary"])}, indent=2))


if __name__ == "__main__":
    main()
