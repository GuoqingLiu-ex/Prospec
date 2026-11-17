"""GPU profile of native vs old reorder vs cached reorder on captured features.

Formal timing uses CUDA events + synchronized wall with profiler off.
Profile traces use 0.01% cutoff so the adaptor is visible.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "training_free"), str(ROOT / "training_free_opt"), str(ROOT / "vendor")]


def error_metrics(y, reference):
    import torch
    from torch.nn import functional as F
    candidate, expected = y.double(), reference.double()
    diff = candidate - expected
    return {
        "max_abs": diff.abs().max().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "relative_l2": (diff.norm() / expected.norm().clamp_min(1e-12)).item(),
        "cosine": F.cosine_similarity(candidate.flatten(), expected.flatten(), dim=0).item(),
    }


def timed_pairs(a_fn, b_fn, warmup, repeats, device):
    import torch
    for _ in range(warmup):
        a_fn()
        b_fn()
    torch.cuda.synchronize(device)
    rows = []
    start_a = torch.cuda.Event(enable_timing=True)
    end_a = torch.cuda.Event(enable_timing=True)
    start_b = torch.cuda.Event(enable_timing=True)
    end_b = torch.cuda.Event(enable_timing=True)
    order = []
    for i in range(repeats):
        first_a = (i % 2 == 0)
        order.append("AB" if first_a else "BA")
        pair = {}
        for name, fn, start, end in (
            (("a", a_fn, start_a, end_a) if first_a else ("b", b_fn, start_b, end_b)),
            (("b", b_fn, start_b, end_b) if first_a else ("a", a_fn, start_a, end_a)),
        ):
            torch.cuda.synchronize(device)
            wall0 = time.perf_counter()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            pair[f"{name}_wall_ms"] = (time.perf_counter() - wall0) * 1000
            pair[f"{name}_cuda_ms"] = start.elapsed_time(end)
        rows.append(pair)
    a_wall = [r["a_wall_ms"] for r in rows]
    b_wall = [r["b_wall_ms"] for r in rows]
    paired = [100.0 * (a - b) / a for a, b in zip(a_wall, b_wall)]
    return {
        "repeats": repeats,
        "warmup": warmup,
        "order": order,
        "a_median_wall_ms": statistics.median(a_wall),
        "b_median_wall_ms": statistics.median(b_wall),
        "a_p95_wall_ms": statistics.quantiles(a_wall, n=20)[18] if len(a_wall) >= 20 else max(a_wall),
        "b_p95_wall_ms": statistics.quantiles(b_wall, n=20)[18] if len(b_wall) >= 20 else max(b_wall),
        "paired_save_median_pct": statistics.median(paired),
        "paired_save_iqr_pct": statistics.quantiles(paired, n=4)[2] - statistics.quantiles(paired, n=4)[0] if len(paired) >= 4 else None,
        "rows": rows,
    }


def export_profile(fn, path, warmup=2, active=6):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(active):
            with record_function("adaptor_call"):
                fn()
        torch.cuda.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    keyed = {}
    cuda_keyed = {}
    for ev in prof.key_averages():
        keyed[ev.key] = keyed.get(ev.key, 0.0) + float(ev.self_cpu_time_total)
        cuda = getattr(ev, "self_device_time_total", None)
        if cuda is None:
            cuda = getattr(ev, "self_cuda_time_total", 0.0)
        cuda_keyed[ev.key] = cuda_keyed.get(ev.key, 0.0) + float(cuda or 0.0)
    cutoff = 0.0001
    def table(d, total):
        rows = []
        for name, us in sorted(d.items(), key=lambda item: -item[1]):
            share = us / total if total else 0.0
            if share < cutoff:
                continue
            rows.append({"name": name, "us": us, "share": share})
        return rows
    return {
        "cutoff": cutoff,
        "note": "0.01% cutoff so small adaptor ops remain visible",
        "kernel_or_cpu_ops": table(keyed, sum(keyed.values()) or 1.0),
        "device_ops": table(cuda_keyed, sum(cuda_keyed.values()) or 1.0),
        "chrome_trace": str(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--profile", type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_experiment import cooperative_lock
    from cached_factorized import CachedFactorizedAdaptor, dispatch_full_token
    from factorized_adaptor import FactorizedAdaptor

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
        old = FactorizedAdaptor(adaptor, precision="model")
        new = CachedFactorizedAdaptor(adaptor, precision="model")
        payload = {
            "physical_gpu": args.physical_gpu,
            "gpu_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "old_cache_bytes": old.cache_bytes,
            "new_cache_bytes": new.cache_bytes,
            "prepare_old": old.prepare(),
            "prepare_new": new.prepare(),
            "sizes": {},
            "errors": {},
            "timing": {},
            "profiles": {},
            "gpu_passed": False,
        }
        feature_dir = Path(args.features)
        names = sorted(p.stem for p in feature_dir.glob("*.pt") if p.name != "adaptor.pt")
        for name in names:
            x = torch.load(feature_dir / f"{name}.pt", map_location="cuda", weights_only=True)
            if isinstance(x, dict):
                x = x["hidden_states"] if "hidden_states" in x else x["x"]
            n = int(x.shape[1])
            y_native = adaptor(x)
            y_old = old(x)
            y_new = new(x)
            payload["errors"][name] = {
                "n": n,
                "dispatch": dispatch_full_token(n),
                "old_vs_native": error_metrics(y_old, y_native),
                "cached_vs_native": error_metrics(y_new, y_native),
                "cached_vs_old": error_metrics(y_new, y_old),
            }
            payload["timing"][f"{name}_native_vs_old"] = timed_pairs(
                lambda: adaptor(x), lambda: old(x), args.warmup, args.repeats, x.device
            )
            payload["timing"][f"{name}_old_vs_cached"] = timed_pairs(
                lambda: old(x), lambda: new(x), args.warmup, args.repeats, x.device
            )
            payload["sizes"][name] = n
            if args.profile and n in {468, 2016} or (args.profile and name == names[0]):
                payload["profiles"][f"{name}_native"] = export_profile(lambda: adaptor(x), out / f"trace_{name}_native.json")
                payload["profiles"][f"{name}_old"] = export_profile(lambda: old(x), out / f"trace_{name}_old.json")
                payload["profiles"][f"{name}_cached"] = export_profile(lambda: new(x), out / f"trace_{name}_cached.json")
        slim = {k: v for k, v in payload.items() if k != "timing"}
        slim["timing_summary"] = {
            key: {k: val[k] for k in val if k != "rows"}
            for key, val in payload["timing"].items()
        }
        (out / "profile_gpu.json").write_text(json.dumps(slim, indent=2) + "\n")
        (out / "timing_raw.json").write_text(json.dumps(payload["timing"]) + "\n")
        print(json.dumps(slim["timing_summary"], indent=2))


if __name__ == "__main__":
    main()
