"""Stage-separated adaptor profile. Cutoff 0.01% so the small adaptor is visible.

This is a PyTorch/ViSpec path. The SGLang live-capture helper is not used.
Formal timing must run this file with --profile 0.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "training_free"), str(ROOT / "training_free_opt")]


def _table(rows, total, cutoff=0.0001):
    out = []
    for name, us in sorted(rows.items(), key=lambda item: -item[1]):
        share = us / total if total else 0.0
        if share < cutoff:
            continue
        out.append({"name": name, "cpu_us": us, "share": share})
    return out


def profile_one(fn, warmup=3, active=8):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    for _ in range(warmup):
        fn()
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(active):
            with record_function("adaptor_call"):
                fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    keyed = defaultdict(float)
    cuda_keyed = defaultdict(float)
    for ev in prof.key_averages():
        keyed[ev.key] += float(ev.self_cpu_time_total)
        cuda = getattr(ev, "self_device_time_total", None)
        if cuda is None:
            cuda = getattr(ev, "self_cuda_time_total", 0.0)
        cuda_keyed[ev.key] += float(cuda or 0.0)
    cpu_total = sum(keyed.values()) or 1.0
    gpu_total = sum(cuda_keyed.values()) or 1.0
    return {
        "cutoff": 0.0001,
        "note": "0.01% cutoff (not the skill default 1%) so adaptor ops remain visible",
        "kernel_or_cpu_ops": _table(keyed, cpu_total),
        "device_ops": _table(cuda_keyed, gpu_total) if gpu_total > 1 else [],
        "cpu_total_us": cpu_total,
        "device_total_us": gpu_total if gpu_total > 1 else 0.0,
        "overlap_opportunity": [
            {"note": "single-trace conservative; mapping/formal pair needed for strong overlap claims"}
        ],
        "fuse_pattern": [
            {"pattern": "score linear + softmax + bmm(P,X) + bmm(V) + o_proj",
             "status": "still separate PyTorch ops in current implementation"}
        ],
    }


def synthetic_adaptor(device, dtype, n=2016, hidden=3584, heads=28, queries=2):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Adaptor(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden
            self.num_heads = heads
            self.num_q = queries
            self.head_dim = hidden // heads
            self.q = nn.Parameter(torch.randn(queries, heads, self.head_dim, device=device, dtype=dtype))
            self.k_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)
            self.v_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)
            self.o_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)

        def forward(self, x):
            b, length, _ = x.shape
            q = self.q.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)
            k = self.k_proj(x).reshape(b, length, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(b, length, self.num_heads, self.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
            return self.o_proj(y.transpose(1, 2).reshape(b, self.num_q, hidden))

    return Adaptor().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, default=-1)
    parser.add_argument("--n", type=int, default=2016)
    parser.add_argument("--profile", type=int, default=1)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from cached_factorized import CachedFactorizedAdaptor
    from factorized_adaptor import FactorizedAdaptor

    if args.physical_gpu >= 0:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and args.physical_gpu >= 0 else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if args.features:
        meta = torch.load(Path(args.features) / "adaptor.pt", map_location=device, weights_only=True)
        raise SystemExit("real-feature GPU path is selected only with --physical-gpu and a lock; use run_gpu_profile.py")
    adaptor = synthetic_adaptor(device, dtype, n=args.n)
    x = torch.randn(1, args.n, adaptor.hidden_size, device=device, dtype=dtype)
    old = FactorizedAdaptor(adaptor, precision="model")
    new = CachedFactorizedAdaptor(adaptor, precision="model")
    with torch.inference_mode():
        y0 = old(x)
        y1 = new(x)
    rel = (y0.double() - y1.double()).norm() / y0.double().norm().clamp_min(1e-12)
    payload = {
        "device": str(device),
        "n": args.n,
        "relative_l2_old_vs_cached": float(rel),
        "old_cache_bytes": old.cache_bytes,
        "new_cache_bytes": new.cache_bytes,
        "gpu_passed": False,
        "note": "synthetic CPU/local profile is not a hardware pass",
    }
    def _timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    payload["old_20_wall_ms"] = _timed(lambda: old(x))
    payload["cached_20_wall_ms"] = _timed(lambda: new(x))
    payload["wall_note"] = "20-call wall includes a device synchronize before and after; it is not a 50-pair GPU pass"
    if args.profile:
        payload["old_profile"] = profile_one(lambda: old(x))
        payload["cached_profile"] = profile_one(lambda: new(x))
    (out / "profile_synthetic.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k not in {"old_profile", "cached_profile"}}, indent=2))


if __name__ == "__main__":
    main()
