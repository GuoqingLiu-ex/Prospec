"""One real adaptor forward inside CUDA profiler start/stop; no timing claims.

Run one arm per process under Nsight Compute with --profile-from-start off and
dram__bytes_read.sum,dram__bytes_write.sum. Sum the metric across all kernels in
the captured range. Keep the profiler's replay/cache policy with the results;
profiling traffic is not whole-request traffic or a process energy measurement.

Example workload arguments (the parent process supplies the ncu options):
  --features CAPTURE_DIR --feature SPORTS.pt --arm native --physical-gpu 0
  --native-source AUTHOR_REPO/vispec/model/cnets_ours.py --out NEW_DIR

The same command with --arm frozen_bmm profiles the full-token reassociation.
Setup, loading, warmup, output checks, and JSON writing stay outside the range.
No kernel filtering or launch limit should remove kernels from either arm.

With --verify-only, omit --arm and run without ncu. This separate mode checks
both outputs and collects five module incremental allocated-peak measurements
per arm. It never starts CUDA profiling and does not measure DRAM traffic.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "direct_opt"), str(ROOT / "vendor")]


def load_original_adaptor(source, metadata, torch):
    """Execute only the original ImgAdaptor class, avoiding unrelated imports."""
    source = Path(source)
    text = source.read_text()
    parsed = ast.parse(text, filename=str(source))
    matches = [node for node in parsed.body
               if isinstance(node, ast.ClassDef) and node.name == "ImgAdaptor"]
    if len(matches) != 1:
        raise ValueError("native source must contain exactly one ImgAdaptor class")
    namespace = {"torch": torch, "nn": torch.nn, "__name__": "original_img_adaptor"}
    module_ast = ast.Module(body=matches, type_ignores=[])
    exec(compile(module_ast, str(source), "exec"), namespace)
    state = metadata["state_dict"]
    if ("k_proj.bias" in state) != ("v_proj.bias" in state):
        raise ValueError("original ImgAdaptor requires matching K/V bias settings")
    config = SimpleNamespace(hidden_size=int(metadata["hidden_size"]),
                             num_attention_heads=int(metadata["num_heads"]),
                             qkv_bias="k_proj.bias" in state)
    adaptor = namespace["ImgAdaptor"](config, num_q=int(metadata["num_q"]))
    adaptor.to(dtype=state["q"].dtype)
    adaptor.load_state_dict(state, strict=True)
    return adaptor.eval().requires_grad_(False), {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "class_start_line": matches[0].lineno,
        "class_end_line": matches[0].end_lineno,
        "class_sha256": hashlib.sha256(ast.get_source_segment(text, matches[0]).encode()).hexdigest(),
    }


def load_feature(path, torch, expected_n):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        x = value["hidden_states"] if "hidden_states" in value else value["x"]
        grid = value.get("grid_thw", value.get("video_grid_thw"))
    else:
        x, grid = value, None
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 1:
        raise ValueError("capture must be a [1,N,D] Tensor")
    if int(x.shape[1]) != expected_n:
        raise ValueError(f"expected N={expected_n}, got {int(x.shape[1])}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("capture contains non-finite input")
    if isinstance(grid, torch.Tensor):
        grid = grid.tolist()
    return x, grid


def incremental_allocated_peak(module, x, torch, trials=5):
    """Same allocated-baseline/reset-peak protocol as the earlier module bench."""
    rows = []
    for trial in range(trials):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        answer = module(x)
        torch.cuda.synchronize()
        rows.append({"trial": trial, "baseline_bytes": baseline,
                     "incremental_peak_bytes": torch.cuda.max_memory_allocated() - baseline,
                     "output_bytes": answer.numel() * answer.element_size()})
        del answer
    return {"trials": rows,
            "median_incremental_peak_bytes": statistics.median(
                row["incremental_peak_bytes"] for row in rows)}


def verify_original_and_peak(native, bmm, x, torch, warmup, verify_compiled=False):
    """No CUDA-profiler API calls. Weights, cached directions, and input are resident."""
    for _ in range(warmup):
        a, b = native(x), bmm(x)
    torch.cuda.synchronize()
    del a, b
    reference, result = native(x), bmm(x)
    candidate, expected = result.double(), reference.double()
    diff = candidate - expected
    checks = {
        "output_shape": list(result.shape),
        "finite": bool(torch.isfinite(reference).all().item() and torch.isfinite(result).all().item()),
        "bitwise_equal": bool(torch.equal(result, reference)),
        "max_abs_vs_native": float(diff.abs().max().item()),
        "rmse_vs_native": float(diff.square().mean().sqrt().item()),
        "relative_l2_vs_native": float((diff.norm() / expected.norm().clamp_min(1e-12)).item()),
    }
    # Free validation outputs and FP64 temporaries before establishing each baseline.
    del reference, result, candidate, expected, diff
    if verify_compiled:
        from compiled_adaptor import CompiledQueryAdaptor
        compiled = CompiledQueryAdaptor.from_frozen(bmm)
        a, b = bmm(x), compiled(x)
        checks["compiled_vs_retained_bmm_bitwise_equal"] = bool(torch.equal(a, b))
        checks["compiled_state_metadata"] = compiled.metadata()
        if not checks["compiled_vs_retained_bmm_bitwise_equal"]:
            raise RuntimeError("Compiled deployment changed the BMM result")
        del a, b, compiled
    torch.cuda.synchronize()
    if not checks["finite"]:
        raise RuntimeError("non-finite native or BMM output")
    measurements = {name: incremental_allocated_peak(module, x, torch, trials=5)
                    for name, module in (("native", native), ("frozen_bmm", bmm))}
    return checks, {
        "measurement": "max_memory_allocated minus resident memory_allocated immediately before one forward",
        "scope": "module transient allocations and output; shared weights, input and BMM cache already resident",
        "units": "bytes", "trials_per_arm": 5,
        "includes_output": True, "resets_allocator_cache": False,
        "not_measured": ["whole-model peak", "memory_reserved peak", "DRAM traffic", "latency"],
        "results": measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Directory containing adaptor.pt")
    parser.add_argument("--feature", required=True, help="Capture filename or absolute path")
    parser.add_argument("--native-source", required=True, help="Actual author cnets_ours.py")
    parser.add_argument("--arm", choices=("native", "frozen_bmm"))
    parser.add_argument("--verify-only", action="store_true",
                        help="No profiler: compare both outputs and five allocated-peak trials per arm")
    parser.add_argument("--verify-compiled", action="store_true",
                        help="Within verify-only, also require compiled and retained BMM outputs to match bitwise")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--out", required=True, help="New output directory, never overwritten")
    parser.add_argument("--expected-n", type=int, default=2016)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not args.verify_only and args.arm is None:
        parser.error("--arm is required unless --verify-only is selected")
    if args.verify_only and args.arm is not None:
        parser.error("--verify-only compares both arms; omit --arm")
    if args.verify_compiled and not args.verify_only:
        parser.error("--verify-compiled requires --verify-only")
    if args.warmup < 1 or args.expected_n < 1 or args.physical_gpu < 0:
        parser.error("warmup/expected-n must be positive; physical-gpu must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    from compression import FrozenQueryAdaptor
    from run_experiment import cooperative_lock

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    feature_path = Path(args.feature)
    if not feature_path.is_absolute():
        feature_path = Path(args.features) / feature_path
    metadata_path = Path(args.features) / "adaptor.pt"
    status = {"status": "preparing", "arm": args.arm if not args.verify_only else "native_vs_frozen_bmm",
              "mode": "verify_only" if args.verify_only else "dram_profiler",
              "profiler_enabled": not args.verify_only, "physical_gpu": args.physical_gpu,
              "feature": str(feature_path.resolve()), "profiled_forward_calls": 0}
    status_path = out / "workload.json"
    try:
        with cooperative_lock(args.physical_gpu, out) as lock_info, torch.inference_mode():
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
            native, source_info = load_original_adaptor(args.native_source, metadata, torch)
            x, grid = load_feature(feature_path, torch, args.expected_n)
            if x.shape[-1] != native.hidden_size or x.dtype != native.q.dtype:
                raise ValueError("feature width/dtype differs from captured adaptor")
            native = native.cuda()
            x = x.cuda()
            bmm = FrozenQueryAdaptor(native, backend="bmm", precision="model").eval()
            if args.verify_only:
                status.update(status="verifying", input_shape=list(x.shape), input_stride=list(x.stride()),
                              dtype=str(x.dtype), grid_thw=grid, native_source=source_info,
                              native_state_load="strict=True; original class constructor and forward",
                              bmm_source_sha256=hashlib.sha256((ROOT / "direct_opt" / "compression.py").read_bytes()).hexdigest(),
                              torch_version=torch.__version__, cuda_version=torch.version.cuda,
                              gpu_name=torch.cuda.get_device_name(0), warmup_calls_per_arm=args.warmup,
                              prepare_time_ms=bmm.prepare_time_ms, cache_bytes=bmm.cache_bytes,
                              lock_initial_snapshot=lock_info, timing_valid=False,
                              timing_note="No latency measurement in verify-only mode.",
                              traffic_note="No DRAM counters collected; allocated bytes are not memory traffic.")
                status_path.write_text(json.dumps(status, indent=2) + "\n")
                checks, memory = verify_original_and_peak(native, bmm, x, torch, args.warmup,
                                                         verify_compiled=args.verify_compiled)
                status.update(checks)
                status.update(status="completed", incremental_module_memory=memory)
                print(json.dumps({key: status[key] for key in ["status", "arm", "input_shape", "profiled_forward_calls", "relative_l2_vs_native"]}))
                return
            module = native if args.arm == "native" else bmm
            # No profiler attachment, timers, or live hardware sampling in this workload.
            # The parent collects hardware state and all kernel DRAM counters.
            for _ in range(args.warmup):
                warmed = module(x)
            torch.cuda.synchronize()
            del warmed
            status.update(status="profiling", input_shape=list(x.shape), input_stride=list(x.stride()),
                          dtype=str(x.dtype), grid_thw=grid, native_source=source_info,
                          torch_version=torch.__version__, cuda_version=torch.version.cuda,
                          gpu_name=torch.cuda.get_device_name(0), warmup_calls=args.warmup,
                          prepare_time_ms=bmm.prepare_time_ms, cache_bytes=bmm.cache_bytes,
                          lock_initial_snapshot=lock_info,
                          scope="one adaptor forward; full captured visual input; parameters resident",
                          timing_valid=False,
                          timing_note="Use independent profiler-off paired timing, never NCU durations.",
                          traffic_note="Kernel DRAM counters under NCU replay/cache policy; not request-wide IO.")
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            # All earlier CUDA work is complete before profiling starts. Keep the
            # output live and finish its kernels before stopping the profiler.
            torch.cuda.profiler.start()
            try:
                torch.cuda.nvtx.range_push("adaptor_" + args.arm)
                try:
                    result = module(x)
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.nvtx.range_pop()
            finally:
                torch.cuda.profiler.stop()
            status["profiled_forward_calls"] = 1
            # Check the selected result against original only after capture ends.
            reference = native(x)
            candidate, expected = result.double(), reference.double()
            diff = candidate - expected
            status.update(status="completed", output_shape=list(result.shape),
                          finite=bool(torch.isfinite(result).all().item()),
                          max_abs_vs_native=float(diff.abs().max().item()),
                          relative_l2_vs_native=float((diff.norm() / expected.norm().clamp_min(1e-12)).item()))
            if not status["finite"]:
                raise RuntimeError("non-finite adaptor output")
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({key: status[key] for key in ["status", "arm", "input_shape", "profiled_forward_calls", "relative_l2_vs_native"]}))


if __name__ == "__main__":
    main()
