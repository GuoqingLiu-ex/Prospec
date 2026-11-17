"""Full score/select/merge/adaptor cost against the existing reorder baseline.

Uses each capture's own pre-merger grid. spatial_merge_size comes from the
model config. T is not divided. Does not overwrite earlier profile directories.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "training_free"), str(ROOT / "training_free_opt"), str(ROOT / "vendor")]


def _load_merge_size(config_path: Path) -> int:
    payload = json.loads(config_path.read_text())
    vision = payload.get("vision_config") or payload
    size = int(vision["spatial_merge_size"])
    if size < 1:
        raise ValueError("spatial_merge_size must be positive")
    return size


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
    merge_size = _load_merge_size(Path(args.config))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_experiment import cooperative_lock
    from hardware_metrics import HardwareSampler
    from factorized_adaptor import FactorizedAdaptor
    from compressors import compress_tokens
    from cheap_compressors import _TOPOLOGY_CACHE, compress_tokens_cached
    from run_gpu_profile import error_metrics, timed_pairs
    from visual_grid import assert_adaptor_tokens

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

    def full_path(tokens, grid, factor, method):
        def fn():
            scores = factor.score_tokens(tokens) if method in ("attention", "response") else None
            saliency = scores.softmax(dim=0).amax(dim=1) if method == "attention" else None
            compressed, sizes, _summary = compress_tokens(
                tokens, grid, 0.5, method=method, saliency=saliency, scores=scores,
            )
            return factor(compressed.unsqueeze(0), group_sizes=sizes)
        return fn

    def cached_path(tokens, grid, factor):
        def fn():
            compressed, sizes, _summary = compress_tokens_cached(tokens, grid, 0.5, method="difference")
            return factor(compressed.unsqueeze(0), group_sizes=sizes)
        return fn

    with cooperative_lock(args.physical_gpu, out), torch.inference_mode():
        metadata = torch.load(Path(args.features) / "adaptor.pt", map_location="cuda", weights_only=True)
        adaptor = Adaptor(metadata).eval()
        torch.cuda.synchronize()
        factor = FactorizedAdaptor(adaptor, precision="model")
        payload = {
            "process_index": args.process_index,
            "physical_gpu": args.physical_gpu,
            "spatial_merge_size": merge_size,
            "config": str(args.config),
            "profiler": "off",
            "gpu_passed": False,
            "baseline": "FactorizedAdaptor full-token reorder",
            "cases": {},
            "timing": {},
        }
        feature_dir = Path(args.features)
        names = sorted(p.stem for p in feature_dir.glob("*.pt") if p.name != "adaptor.pt")
        sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.05, process_interval_s=0.2)
        sampler.start()
        for name in names:
            loaded = torch.load(feature_dir / f"{name}.pt", map_location="cuda", weights_only=True)
            x = loaded["x"] if "x" in loaded else loaded["hidden_states"]
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError(f"{name} is not a single-sequence [1,N,D] capture")
            grid = assert_adaptor_tokens(int(x.shape[1]), loaded["grid_thw"], merge_size)
            tokens = x[0]
            reference = factor(x)
            case = {"n": int(x.shape[1]), "raw_grid": loaded["grid_thw"], "merged_grid": list(grid), "methods": {}}
            _TOPOLOGY_CACHE.clear()
            torch.cuda.synchronize()
            cold0 = time.perf_counter()
            cold_out = cached_path(tokens, grid, factor)()
            torch.cuda.synchronize()
            cold_ms = (time.perf_counter() - cold0) * 1000
            torch.cuda.synchronize()
            hot0 = time.perf_counter()
            cached_path(tokens, grid, factor)()
            torch.cuda.synchronize()
            hot_ms = (time.perf_counter() - hot0) * 1000
            case["topology_cache"] = {
                "first_call_ms": cold_ms,
                "second_call_ms": hot_ms,
                "cache_entries": len(_TOPOLOGY_CACHE),
                "note": "first call fills the 2x2 topology cache; scheme_context still calls uncached compress_tokens",
            }
            case["cached_vs_full_l2"] = error_metrics(cold_out, reference)["relative_l2"]
            for method in ("difference", "response", "attention"):
                path = full_path(tokens, grid, factor, method)
                compressed = path()
                kept = int(compress_tokens(
                    tokens, grid, 0.5, method=method,
                    saliency=(factor.score_tokens(tokens).softmax(dim=0).amax(dim=1) if method == "attention" else None),
                    scores=(factor.score_tokens(tokens) if method in ("attention", "response") else None),
                )[0].shape[0])
                case["methods"][method] = {
                    "kept_tokens": kept,
                    "relative_l2_vs_full_reorder": error_metrics(compressed, reference)["relative_l2"],
                }
                device = x.device
                payload["timing"][f"{name}_{method}"] = timed_pairs(
                    lambda xx=x: factor(xx), path, args.warmup, args.repeats, device,
                )
            payload["timing"][f"{name}_cached_difference"] = timed_pairs(
                lambda xx=x: factor(xx), cached_path(tokens, grid, factor), args.warmup, args.repeats, x.device,
            )
            payload["cases"][name] = case
        hardware = sampler.stop()
        foreign = []
        for item in hardware.get("foreign_processes") or []:
            comm = ""
            try:
                comm = Path(f"/proc/{item['pid']}/comm").read_text().strip()
            except OSError:
                comm = ""
            foreign.append({**item, "comm": comm})
        payload["hardware"] = {
            "foreign_processes": foreign,
            "graphics_only": bool(foreign) and all(row.get("comm") == "Xorg" for row in foreign),
            "duration_s": hardware.get("measurement_duration_s"),
        }
        slim = {k: v for k, v in payload.items() if k != "timing"}
        slim["timing_summary"] = {
            key: {k: val for k, val in value.items() if k not in {"rows", "order"}}
            for key, value in payload["timing"].items()
        }
        (out / "compress_cost.json").write_text(json.dumps(slim, indent=2) + "\n")
        (out / "timing_raw.json").write_text(json.dumps(payload["timing"]) + "\n")
        print(json.dumps({
            "process": args.process_index,
            "merge_size": merge_size,
            "graphics_only_foreign": payload["hardware"]["graphics_only"],
            "cases": {name: {"n": row["n"], "merged": row["merged_grid"]} for name, row in payload["cases"].items()},
        }))


if __name__ == "__main__":
    main()
