"""Exploratory native/cached/folded fusion microbench on real captured tensors.

Input is capture_visual_fusion.py's completed fusion_capture directory, including
weights.pt and summary.json. Native replay is checked against both saved outputs.
All variants coexist: memory records are module/setup allocations, not deployed
whole-model peaks. Shared-GPU timing never establishes an E2E or quality pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "direct_opt"), str(ROOT / "vendor")]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_native(weights, torch, device):
    """Construct the two original nn.Linear forwards from actual captured weights."""
    required = {"img_fc.weight", "img_fc.bias", "fc.weight", "fc.bias"}
    if set(weights) != required:
        raise ValueError("weights.pt must contain exactly the four captured fusion weight/bias keys")
    w = weights["img_fc.weight"]
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        raise ValueError("invalid captured img_fc weight")
    d = int(w.shape[0])
    for name, tensor in weights.items():
        if tensor is None and name.endswith("bias"):
            continue
        expected = (d,) if name.endswith("bias") else (d, 2*d)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected or tensor.dtype != w.dtype:
            raise ValueError(f"captured fusion weight mismatch: {name}")

    class NativePair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("img_fc", "fc"):
                layer = torch.nn.Linear(2*d, d, bias=weights[name + ".bias"] is not None,
                                        device="meta", dtype=w.dtype)
                layer.weight = torch.nn.Parameter(weights[name + ".weight"].to(device), requires_grad=False)
                if layer.bias is not None:
                    layer.bias = torch.nn.Parameter(weights[name + ".bias"].to(device), requires_grad=False)
                setattr(self, name, layer)
            self.hidden_size = d

        def image(self, hidden, global_visual):
            g = global_visual.reshape(1, d).expand_as(hidden)
            return self.img_fc(torch.cat((hidden, g), dim=-1))

        def forward(self, embed, hidden, global_visual):
            return self.fc(torch.cat((embed, self.image(hidden, global_visual)), dim=-1))

    return NativePair().eval()


def error_metrics(value, reference, torch):
    if value.shape != reference.shape:
        raise ValueError("reference output shape mismatch")
    a, b = value.double(), reference.double()
    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
    if not finite:
        return {"finite": False, "bitwise_equal": False, "relative_l2": None, "max_abs": None}
    diff = a - b
    return {"finite": True, "bitwise_equal": bool(value.dtype == reference.dtype and torch.equal(
                value.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))),
            "relative_l2": float((diff.norm() / b.norm().clamp_min(1e-12)).item()),
            "max_abs": float(diff.abs().max().item()) if value.numel() else 0.0}


def matrix_macs(n, d, img_bias):
    return {"native": 4*n*d*d,
            "cached_image_hot": 3*n*d*d, "cached_image_miss": 3*n*d*d + (d*d if n else 0),
            "folded_pair_hot": 2*n*d*d, "folded_pair_miss": 2*n*d*d + (d*d if n else 0),
            "new_global_projection_each_nonempty_g": d*d,
            "folded_setup_matrix_macs": 2*d*d*d,
            "folded_setup_optional_bias_projection_macs": d*d if img_bias else 0,
            "scope": "MACs of matrix products only; biases, broadcast additions, concat/copy/casts, allocation, checks and launches are not included"}


def measure_setup(factory, torch):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = time.perf_counter()
    module = factory()
    torch.cuda.synchronize()
    return module, {"synchronized_wall_ms": (time.perf_counter() - start)*1000,
                    "baseline_allocated_bytes": baseline,
                    "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline,
                    "resident_delta_allocated_bytes": torch.cuda.memory_allocated() - baseline}


def timed_pairs(a_fn, b_fn, torch, *, repeats, warmup, seed):
    for _ in range(warmup):
        a_fn()
        b_fn()
    torch.cuda.synchronize()
    a_first = ([True, False] * ((repeats + 1)//2))[:repeats]
    random.Random(seed).shuffle(a_first)
    events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for name in ("native", "candidate")}
    rows = []
    for index, first in enumerate(a_first):
        row = {"pair": index, "order": "AB" if first else "BA"}
        order = (("native", a_fn), ("candidate", b_fn))
        for name, fn in order if first else reversed(order):
            torch.cuda.synchronize()
            start_event, end_event = events[name]
            start = time.perf_counter()
            start_event.record()
            result = fn()
            end_event.record()
            torch.cuda.synchronize()
            row[name + "_wall_ms"] = (time.perf_counter() - start)*1000
            row[name + "_cuda_ms"] = start_event.elapsed_time(end_event)
            del result
        rows.append(row)
    saved = [100*(r["native_wall_ms"] - r["candidate_wall_ms"])/r["native_wall_ms"] for r in rows]
    return {"repeats": repeats, "warmup_pairs": warmup, "order_seed": seed,
            "native_median_wall_ms": statistics.median(r["native_wall_ms"] for r in rows),
            "candidate_median_wall_ms": statistics.median(r["candidate_wall_ms"] for r in rows),
            "native_median_cuda_ms": statistics.median(r["native_cuda_ms"] for r in rows),
            "candidate_median_cuda_ms": statistics.median(r["candidate_cuda_ms"] for r in rows),
            "paired_wall_save_median_pct": statistics.median(saved),
            "paired_wall_save_iqr_pct": (statistics.quantiles(saved, n=4)[2] - statistics.quantiles(saved, n=4)[0]) if repeats >= 4 else None,
            "paired_wall_reduction_median_ms": statistics.median(r["native_wall_ms"] - r["candidate_wall_ms"] for r in rows),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.physical_gpu < 0 or args.repeats < 1 or args.warmup < 1:
        parser.error("nonnegative GPU and positive repeats/warmup required")
    capture = Path(args.capture_dir).resolve()
    source = json.loads((capture / "summary.json").read_text())
    if source.get("status") != "completed" or source.get("pairing_errors") or not source.get("captures"):
        parser.error("require completed native capture with samples and no pairing errors")
    if len(source["captures"]) > 8:
        parser.error("capture exceeds the bounded eight-sample protocol")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    from frozen_visual_fusion import FrozenVisualFusion, _storage_info
    from run_experiment import cooperative_lock
    from hardware_metrics import HardwareSampler

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    payload = {"status": "preparing", "args": vars(args), "capture_dir": str(capture),
               "capture_summary": source, "source_sha256": {}, "setup": {}, "samples": {},
               "performance_pass": False, "e2e_quality_pass": False,
               "scope": "real-input module replay; all variants resident; paired profiler-off timing",
               "cache_protocol": {"miss": "reset_cache and forward both inside timed call; global projection included",
                                  "hot": "reset and one actual fill before warmup/timing; reuse identical g Tensor object",
                                  "cross_sample": "reset caches before each sample; no cached output reuse"},
               "timing_scope": "synchronized wall and CUDA-event spans; input H2D and validation outside; forward allocations/cats/copies included",
               "setup_scope": "native: captured weight transfer and module construction; candidates: frozen preparation atop resident native weights; no deployment peak claim"}
    raw = {}
    sampler = None
    try:
        with cooperative_lock(args.physical_gpu, out) as lock, torch.inference_mode():
            payload["lock"] = lock
            sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.05, process_interval_s=0.2)
            sampler.start()
            payload.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                           gpu_name=torch.cuda.get_device_name(0))
            for path in (capture / "summary.json", capture / "weights.pt", Path(__file__), ROOT / "direct_opt" / "frozen_visual_fusion.py"):
                payload["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            weights = torch.load(capture / "weights.pt", map_location="cpu", weights_only=True)
            native, prep = measure_setup(lambda: load_native(weights, torch, "cuda"), torch)
            prep["weights"] = _storage_info(list(native.named_parameters()))
            payload["setup"]["native"] = prep
            candidates = {}
            for mode in ("cached_image", "folded_pair"):
                candidates[mode], prep = measure_setup(lambda m=mode: FrozenVisualFusion(native.img_fc, native.fc, mode=m), torch)
                prep["module"] = candidates[mode].metadata()
                payload["setup"][mode] = prep
            for sample_index, entry in enumerate(source["captures"]):
                path = (capture / entry["file"]).resolve()
                if path.parent != capture:
                    raise ValueError("sample path must remain inside capture directory")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                names = ("embed", "hidden", "global_visual", "native_img_output", "native_fusion_output")
                tensors = {name: saved[name].to("cuda") for name in names}
                embed, hidden, g = (tensors[k] for k in ("embed", "hidden", "global_visual"))
                n, d = math.prod(embed.shape[:-1]), embed.shape[-1]
                if n != saved["row_count"] or d != native.hidden_size or embed.dtype != native.img_fc.weight.dtype:
                    raise ValueError("sample shape/dtype differs from captured metadata or weights")
                for module in candidates.values():
                    module.reset_cache()
                observed_image = native.image(hidden, g)
                observed = native(embed, hidden, g)
                result = {"metadata": {k: v for k, v in saved.items() if k not in names},
                          "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "matrix_macs": matrix_macs(n, d, native.img_fc.bias is not None),
                          "native_image_vs_saved": error_metrics(observed_image, tensors["native_img_output"], torch),
                          "native_fusion_vs_saved": error_metrics(observed, tensors["native_fusion_output"], torch),
                          "candidate_errors_vs_native_replay": {}, "timing": {}}
                payload["samples"][entry["file"]] = result
                if not result["native_image_vs_saved"]["finite"] or not result["native_fusion_vs_saved"]["finite"]:
                    raise RuntimeError("non-finite native replay or saved output")
                result["native_replay_bitwise"] = all(result[k]["bitwise_equal"] for k in ("native_image_vs_saved", "native_fusion_vs_saved"))
                if not result["native_replay_bitwise"]:
                    result["warning"] = "native replay differs from captured outputs; numerical provenance must be resolved before performance interpretation"
                for mode, module in candidates.items():
                    answer = module(embed, hidden, g)
                    result["candidate_errors_vs_native_replay"][mode] = error_metrics(answer, observed, torch)
                    if not result["candidate_errors_vs_native_replay"][mode]["finite"]:
                        raise RuntimeError(f"{mode} produced non-finite output")
                    del answer
                del observed_image, observed
                def a_fn():
                    return native(embed, hidden, g)
                cases = [("native_AA", a_fn)]
                for mode, module in candidates.items():
                    def miss(m=module):
                        m.reset_cache()
                        return m(embed, hidden, g)
                    cases.extend(((mode + "_miss", miss), (mode + "_hot", lambda m=module: m(embed, hidden, g))))
                for case_index, (label, b_fn) in enumerate(cases):
                    if label != "native_AA":
                        module = candidates[label.rsplit("_", 1)[0]]
                        module.reset_cache()
                        if label.endswith("_hot"):
                            warm = module(embed, hidden, g)
                            del warm
                        before_cache = module.cache_info()
                    sampler.mark(entry["file"] + ":" + label)
                    timing = timed_pairs(a_fn, b_fn, torch, repeats=args.repeats, warmup=args.warmup,
                                         seed=args.seed + sample_index*100 + case_index)
                    raw[entry["file"] + ":" + label] = timing["rows"]
                    result["timing"][label] = {k: v for k, v in timing.items() if k != "rows"}
                    if label != "native_AA":
                        after_cache = module.cache_info()
                        result["timing"][label]["cache_delta"] = {k: after_cache[k] - before_cache[k] for k in ("hits", "misses", "resets")}
                        expected = args.warmup + args.repeats if n else 0
                        actual = after_cache["misses" if label.endswith("_miss") else "hits"] - before_cache["misses" if label.endswith("_miss") else "hits"]
                        if actual != expected:
                            raise RuntimeError("cache benchmark did not execute the requested miss/hit path")
                print(json.dumps({"sample": entry["file"], "rows": n, "native_replay_bitwise": result["native_replay_bitwise"],
                      "candidate_errors": result["candidate_errors_vs_native_replay"],
                      "paired_wall_save_pct": {k: v["paired_wall_save_median_pct"] for k, v in result["timing"].items()}}), flush=True)
                for module in candidates.values():
                    module.reset_cache()
                del tensors, embed, hidden, g, saved
            payload["status"] = "completed"
    except BaseException as exc:
        payload.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if sampler is not None:
            hardware = sampler.stop()
            for process in hardware.get("foreign_processes", []):
                try:
                    process["comm"] = Path(f"/proc/{process['pid']}/comm").read_text().strip()
                except OSError:
                    process["comm"] = None
            save(out / "hardware.json", hardware)
            payload["shared_gpu"] = hardware.get("sharing") == "shared"
            payload["hardware_observation"] = {k: hardware.get(k) for k in (
                "sharing", "foreign_processes", "available", "process_visibility_complete", "process_query_failures")}
        payload["timing_interpretation"] = "exploratory module timing; no performance pass, E2E, throughput, energy or final-answer quality claim"
        save(out / "fusion.json", payload)
        save(out / "timing_raw.json", raw)


if __name__ == "__main__":
    main()
