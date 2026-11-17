"""Full visual-module validation over every explicitly supplied request.

Modes are native/fixed1/direct, with two warmups per mode/input and ten paired
measurement rounds by default. All requests in --requests are used unless
--request-id selects an explicit subset. Thresholds and anchor stride are
required and are never calibrated here; direct fixes prefix27 and layers28/29.

Full Target is resident, with no Draft or language decoding. Video IO,
preprocessing/upload, feature diagnostics, JSON and telemetry setup are outside
cell timing. Policy construction/installation, fresh prepare, gate, grouping,
visual forward, output materialization, completion wait and teardown are inside
CPU wall time. CUDA events cover the visual stream interval after preparation,
including host launch gaps; the wall interval is the primary net-cost measure.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "training_free"), str(ROOT / "vendor")]
MODES = ("native", "fixed1", "direct")



def load_requests(path, selected_ids=None):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r["request_id"] for r in records]
    if not records or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Require nonempty unique request IDs")
    if selected_ids:
        wanted = set(selected_ids)
        missing = wanted - set(ids)
        if missing:
            raise ValueError(f"Unknown request IDs: {sorted(missing)}")
        records = [r for r in records if r["request_id"] in wanted]
    return records


def validate_thresholds(q):
    if set(q) != {"update1", "update2", "temporal1", "temporal2"}:
        raise ValueError("Require frozen local update1/update2/temporal1/temporal2 thresholds")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in q.values()):
        raise ValueError("Thresholds must be finite and nonnegative")
    if q["update2"] > q["update1"] or q["temporal2"] > q["temporal1"]:
        raise ValueError("Skip-two thresholds must not exceed skip-one thresholds")


def make_policy_context(target, mode, thresholds, anchor_stride):
    if mode == "native":
        return nullcontext(None)
    if mode not in {"fixed1", "direct"}:
        raise ValueError("Expected native, fixed1, or direct")
    from depth import direct_depth_policy
    return direct_depth_policy(target, mode="adaptive" if mode == "direct" else "fixed1",
                               thresholds=thresholds, anchor_stride=anchor_stride,
                               skip_layers=(28, 29), prefix_layer=27)


def source_hashes():
    paths = [Path(__file__), HERE / "depth.py", ROOT / "vision_depth.py",
             ROOT / "training_free/local_depth.py", ROOT / "hardware_metrics.py",
             ROOT / "run_experiment.py", ROOT / "vendor/hf517_vispec_adapter.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_micro_schedule(request_ids, repeats, seed):
    """Two warmups per mode/input; paired measurement rounds in random order."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if len(set(request_ids)) != len(request_ids) or not request_ids:
        raise ValueError("Request IDs must be nonempty and unique")
    rng = random.Random(seed)
    schedule = []
    for request_id in request_ids:
        for repeat in range(2):
            modes = list(MODES)
            rng.shuffle(modes)
            schedule.extend(dict(request_id=request_id, mode=mode, phase="warmup", repeat=repeat)
                            for mode in modes)
    for repeat in range(repeats):
        ids = list(request_ids)
        rng.shuffle(ids)
        for request_id in ids:
            modes = list(MODES)
            rng.shuffle(modes)
            schedule.extend(dict(request_id=request_id, mode=mode, phase="measure", repeat=repeat)
                            for mode in modes)
    return schedule


def summarize(rows):
    measured = [r for r in rows if r["phase"] == "measure"]
    result = {}
    for request_id in sorted({r["request_id"] for r in measured}):
        selected = [r for r in measured if r["request_id"] == request_id]
        native = {r["repeat"]: r for r in selected if r["mode"] == "native"}
        fixed1 = {r["repeat"]: r for r in selected if r["mode"] == "fixed1"}
        modes = {}
        for mode in MODES:
            subset = [r for r in selected if r["mode"] == mode]
            details = {"measurements": len(subset)}
            for metric in ("vision_cuda_event_ms", "policy_and_vision_wall_ms"):
                values = [r[metric] for r in subset]
                paired = [1-r[metric]/native[r["repeat"]][metric] for r in subset]
                versus_fixed1 = [1-r[metric]/fixed1[r["repeat"]][metric] for r in subset]
                details[metric] = {
                    "mean": statistics.mean(values), "median": statistics.median(values),
                    "min": min(values), "max": max(values),
                    "mean_paired_fraction_saved": statistics.mean(paired),
                    "median_paired_fraction_saved": statistics.median(paired),
                    "mean_paired_fraction_saved_vs_fixed1": statistics.mean(versus_fixed1),
                    "median_paired_fraction_saved_vs_fixed1": statistics.median(versus_fixed1),
                }
            details["feature_diagnostic"] = {
                "relative_l2_max": max(r["features_vs_native"]["relative_l2"] for r in subset),
                "cosine_flat_min": min(r["features_vs_native"]["cosine_flat"] for r in subset),
                "interpretation": "Representation change only; not QA quality or correctness",
            }
            modes[mode] = details
        result[request_id] = modes
    return {"per_request": result,
            "scope": "Repeated visual module observations on shared GPU; not end-to-end gains",
            "units": {"time": "milliseconds", "saved_fraction": "positive means faster"},
            "cuda_event_note": "Elapsed GPU-stream interval, including host launch gaps; not sum of kernel times",
            "wall_note": "Includes policy installation/preparation and visual forward/completion; excludes input preparation and diagnostics",
            "statistics_note": "Repeats of the same input are not independent videos; no QA or population claim"}


def _feature_diagnostic(torch, actual, reference):
    # Offline scalar diagnostics only. Long FP32 dot/norm reductions previously
    # reported self-cosines slightly above one; preserve raw old results and use
    # FP64 accumulation for future diagnostics, outside every timing interval.
    actual = actual.detach().cpu().to(dtype=torch.float64)
    reference = reference.detach().cpu().to(dtype=torch.float64)
    if actual.shape != reference.shape:
        raise RuntimeError("Pooler output shape changed")
    a, b = actual.reshape(-1), reference.reshape(-1)
    an = torch.linalg.vector_norm(a)
    bn = torch.linalg.vector_norm(b)
    return {
        "shape": list(actual.shape),
        "relative_l2": float(torch.linalg.vector_norm(a-b)/bn.clamp_min(1e-12)),
        "cosine_flat": float((torch.dot(a, b)/(an*bn).clamp_min(1e-12)).clamp(-1., 1.)),
        "max_abs": float((a-b).abs().max()),
        "scalar_accumulation_dtype": "float64_cpu",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=range(8))
    parser.add_argument("--requests", required=True, help="Use every request in this explicit JSONL")
    parser.add_argument("--request-id", action="append", help="Optional exact subset; no two-request limit")
    parser.add_argument("--thresholds", required=True, help="Previously frozen threshold JSON; never fit in this script")
    parser.add_argument("--anchor-stride", type=int, required=True, help="Frozen anchor setting; first tubelet remains protected")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.anchor_stride < 0:
        parser.error("--anchor-stride must be nonnegative")
    try:
        requests = load_requests(args.requests, args.request_id)
    except ValueError as error:
        parser.error(str(error))
    threshold_record = json.loads(Path(args.thresholds).read_text())
    thresholds = threshold_record.get("thresholds", threshold_record)
    validate_thresholds(thresholds)
    skip_layers, prefix_layer = [28, 29], 27
    schedule = make_micro_schedule([r["request_id"] for r in requests], args.repeats, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    from run_experiment import cooperative_lock, timestamp, save_json, TARGET
    status = {
        "status": "starting", "started_at": timestamp(), "pid": os.getpid(),
        "args": vars(args), "target": TARGET, "thresholds": thresholds,
        "skip_layers": skip_layers, "prefix_layer": prefix_layer,
        "source_sha256": source_hashes(),
        "input_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for name, path in (("requests", args.requests), ("thresholds", args.thresholds))},
        "modes": list(MODES), "warmups_per_input_mode": 2,
        "total_cells": len(schedule), "completed_cells": 0,
        "timing_primary": "policy_and_vision_wall_ms",
        "scope": "vision_module_microbenchmark_shared_gpu",
        "memory_scope": "Full Target resident; no Draft loaded. Absolute memory is not comparable with full Target+Draft runs.",
        "inputs_scope": "Each video prepared and uploaded once, before all timed cells",
        "quality_scope": "No language model inference or QA evaluation; feature diagnostics only",
        "energy_scope": "One shared-device telemetry interval over warmups, measurements and untimed diagnostics; no per-cell or process energy claim",
    }
    save_json(out/"status.json", status)
    save_json(out/"schedule.json", schedule)
    save_json(out/"requests.json", requests)
    sampler = None
    rows = []
    try:
        with cooperative_lock(args.physical_gpu, out) as lock:
            status["lock"] = lock
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), TOKENIZERS_PARALLELISM="false",
                              OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            # The cooperative lock imports the legacy GPU-lock module and can
            # prepend its directory. Restore this frozen experiment's modules
            # before importing the current depth implementation and bridge.
            sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "training_free"), str(ROOT / "vendor")]
            import torch
            import cv2
            import transformers
            from torch.nn.attention import sdpa_kernel, SDPBackend
            import hf517_vispec_adapter as bridge
            from depth import direct_depth_policy  # preload before all timed cells
            from hardware_metrics import HardwareSampler
            bridge._native()
            from jev_spec.runtime import prepare_request
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            target = bridge.load_target(TARGET, "cuda:0", dtype="float16",
                                        output_head_dtype="float32", target_sdpa_backend="math")
            visual = target.model.visual
            prepared, references, input_metadata = {}, {}, {}
            with torch.inference_mode():
                for request in requests:
                    packed, sampled = prepare_request(target._hf517_processor, request, "cuda:0")
                    pixels = packed["pixel_values_videos"].to(dtype=visual.dtype)
                    grid = packed["video_grid_thw"]
                    prepared[request["request_id"]] = (pixels, grid)
                    input_metadata[request["request_id"]] = {
                        "video_grid_thw": grid.detach().cpu().tolist(), "pixel_shape": list(pixels.shape),
                        "pixel_dtype": str(pixels.dtype), "timestamps": list(sampled.timestamps),
                    }
                    # Calling visual directly bypasses Target.forward's math wrapper.
                    with sdpa_kernel(SDPBackend.MATH):
                        reference = visual(pixels, grid_thw=grid)
                    references[request["request_id"]] = reference.pooler_output.detach().float().cpu()
                    del reference, packed
            torch.cuda.synchronize()
            status.update(status="running", torch=torch.__version__, transformers=transformers.__version__,
                          device_name=torch.cuda.get_device_name(0), vision_dtype=str(visual.dtype),
                          vision_attn_implementation=visual.config._attn_implementation,
                          sdpa_backend="math_explicit_for_each_visual_call",
                          total_cells=len(schedule), completed_cells=0,
                          resident_allocated_bytes=torch.cuda.memory_allocated())
            save_json(out/"status.json", status)
            save_json(out/"inputs.json", input_metadata)
            save_json(out/"vision_config.json", visual.config.to_dict())
            sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=.1,
                                      process_interval_s=.5, known_shared=True)
            sampler.start()
            with (out/"results.jsonl").open("w") as writer, torch.inference_mode():
                for cell_index, cell in enumerate(schedule):
                    request_id, mode = cell["request_id"], cell["mode"]
                    pixels, grid = prepared[request_id]
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    sampler.mark(f"cell_{cell_index}_start")
                    wall_start = time.perf_counter()
                    # Fresh policy per cell: include installation and uncached preparation.
                    context = make_policy_context(target, mode, thresholds, args.anchor_stride)
                    with context as policy, sdpa_kernel(SDPBackend.MATH):
                        if policy is not None:
                            policy.prepare(grid, device=pixels.device)
                        start_event.record()
                        output = visual(pixels, grid_thw=grid)
                        end_event.record()
                        end_event.synchronize()
                    wall_ms = (time.perf_counter()-wall_start)*1000
                    event_ms = start_event.elapsed_time(end_event)
                    peak_allocated = torch.cuda.max_memory_allocated()
                    peak_reserved = torch.cuda.max_memory_reserved()
                    sampler.mark(f"cell_{cell_index}_timing_end")
                    # All tensor-to-host stats and diagnostics are outside both timers.
                    stats = policy.summary() if policy is not None else {
                        "mode": "native", "calls": [], "token_block_reduction": 0.0}
                    diagnostic = _feature_diagnostic(torch, output.pooler_output, references[request_id])
                    record = {
                        **cell, "cell_index": cell_index, "physical_gpu": args.physical_gpu,
                        "vision_cuda_event_ms": event_ms, "policy_and_vision_wall_ms": wall_ms,
                        "features_vs_native": diagnostic, "stats": stats,
                        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                        "video_grid_thw": input_metadata[request_id]["video_grid_thw"],
                        "timing_scope": "vision_only_inputs_prepared_once", "known_shared": True,
                    }
                    rows.append(record)
                    writer.write(json.dumps(record, ensure_ascii=False)+"\n")
                    writer.flush()
                    del output
                    status.update(completed_cells=cell_index+1, current=cell, updated_at=timestamp())
                    save_json(out/"status.json", status)
            hardware = sampler.stop()
            save_json(out/"hardware.json", hardware)
            summary = summarize(rows)
            save_json(out/"summary.json", summary)
            status.update(status="completed", completed_at=timestamp(), current=None)
            save_json(out/"status.json", status)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        if sampler is not None:
            save_json(out/"hardware.json", sampler.stop())
        status.update(status="failed", failed_at=timestamp(), error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        save_json(out/"status.json", status)
        raise


if __name__ == "__main__":
    main()
