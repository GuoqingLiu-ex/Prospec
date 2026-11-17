"""Numerical diagnosis of the first failed dev scout; never deploys a gate.

Capture the actual prefix27 output for v_Ku76zinwWkg once. Replay native B29
with full inputs, the original packed central windows, and a same-M input that
zeros only unselected windows. Repeat full/packed to check reproducibility.
Optionally copy just B29 to FP32, preserving the values of its FP16 weights,
disable TF32, and compare full/packed. Original validation tolerances stay fixed.
An optional packed-only branch disables reduced-precision FP16 reduction only
inside down_proj.forward; tensor/weight dtypes and the full reference are kept.
Every additional call and its analytic matrix MAC count is diagnostic overhead.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from types import MethodType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULE_PATHS = [str(HERE), str(ROOT / "training_free"), str(ROOT), str(ROOT / "vendor")]
sys.path[:0] = MODULE_PATHS
FAILED_VIDEO = "v_Ku76zinwWkg"
TRACE_MODULES = ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
FP16_CASES = ("full", "full_repeat", "packed", "packed_repeat", "same_m_zero_padding")


def source_hashes():
    paths = [Path(__file__), HERE / "window_scout.py", HERE / "calibrate_pair12.py",
             ROOT / "vision_depth.py", ROOT / "training_free/local_depth.py",
             ROOT / "compute_cost.py", ROOT / "run_experiment.py", ROOT / "vendor/hf517_vispec_adapter.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def select_failed_request(path):
    from calibrate_pair12 import load_development_requests
    matches = [r for r in load_development_requests(path) if r["video_id"] == FAILED_VIDEO]
    if len(matches) != 1:
        raise ValueError("Require the exact first failed video in the frozen12dev input file")
    return matches[0]


def tensor_differences(actual, reference, full_token_indices, *, rtol=.005, atol=.005, examples=16):
    """FP64 error reporting, preserving exact mismatch and original allclose tests."""
    import torch
    if actual.ndim != 2 or actual.shape != reference.shape or len(actual) != len(full_token_indices):
        raise ValueError("Expected aligned selected-token matrices")
    a, b = (x.detach().to(device="cpu", dtype=torch.float64) for x in (actual, reference))
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite diagnostic activation")
    delta = (a-b).abs()
    tolerance = atol+rtol*b.abs()
    changed = delta != 0
    violated = delta > tolerance
    bad_coords = violated.nonzero(as_tuple=False)
    changed_coords = changed.nonzero(as_tuple=False)

    def example(row, channel):
        return dict(selected_row=row, full_token_index=int(full_token_indices[row]), channel=channel,
                    actual=float(a[row, channel]), reference=float(b[row, channel]),
                    absolute_error=float(delta[row, channel]), allowed_error=float(tolerance[row, channel]),
                    error_over_tolerance=float(delta[row, channel]/tolerance[row, channel]))

    worst = []
    if len(bad_coords):
        ratios = (delta/tolerance)[violated]
        order = torch.argsort(ratios, descending=True, stable=True)[:examples]
        worst = [example(*map(int, bad_coords[i].tolist())) for i in order.tolist()]
    first = example(*map(int, changed_coords[0].tolist())) if len(changed_coords) else None
    token_delta = torch.linalg.vector_norm(a-b, dim=-1)
    token_norm = torch.linalg.vector_norm(b, dim=-1).clamp_min(1e-12)
    return dict(shape=list(a.shape), total_elements=a.numel(), exact_mismatch_count=int(changed.sum()),
        allclose=bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
        allclose_violation_count=int(violated.sum()), relative_l2=float(torch.linalg.vector_norm(a-b) /
            torch.linalg.vector_norm(b).clamp_min(1e-12)), max_absolute_error=float(delta.max()),
        max_token_relative_l2=float((token_delta/token_norm).max()),
        first_difference=first, largest_tolerance_violations=worst,
        violation_order="Descending error/tolerance, stable original-coordinate order for ties",
        accumulation_dtype="float64_cpu", rtol=rtol, atol=atol)


def _metadata(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
                device=str(tensor.device), contiguous=tensor.is_contiguous())


def trace_block(block, hidden, kwargs, selected_indices=None):
    """Capture selected rows at seven modules while leaving their computation intact."""
    import torch
    traces, order, handles = {}, [], []

    def select(tensor):
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise ValueError("Unexpected module activation shape")
        selected = tensor if selected_indices is None else tensor.index_select(0, selected_indices)
        return selected.detach().cpu().clone()

    def hook(name):
        def capture(module, args, output):
            if name in traces:
                raise ValueError("Unexpected repeated module call inside a single B29 replay")
            traces[name] = dict(input=select(args[0]), output=select(output),
                                input_metadata=_metadata(args[0]), output_metadata=_metadata(output))
            order.append(name)
        return capture

    try:
        for name in TRACE_MODULES:
            handles.append(block.get_submodule(name).register_forward_hook(hook(name)))
        output = block(hidden.clone(), **kwargs)
        result = select(output)
    finally:
        for handle in handles:
            handle.remove()
    if order != list(TRACE_MODULES):
        raise ValueError(f"Unexpected native block module order: {order}")
    return dict(output=result, modules=traces, module_order=order,
                input_metadata=_metadata(hidden), output_metadata=_metadata(output))


def compare_traces(actual, reference, full_token_indices, selected_lengths):
    from window_scout import (validate_native_branch, DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_RELATIVE_L2)
    comparisons, first_difference, first_violation = {}, None, None
    for module in TRACE_MODULES:
        comparisons[module] = {}
        for boundary in ("input", "output"):
            row = tensor_differences(actual["modules"][module][boundary],
                reference["modules"][module][boundary], full_token_indices,
                rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
            row["actual_metadata"] = actual["modules"][module][f"{boundary}_metadata"]
            row["reference_metadata"] = reference["modules"][module][f"{boundary}_metadata"]
            comparisons[module][boundary] = row
            if first_difference is None and row["exact_mismatch_count"]:
                first_difference = dict(module=module, boundary=boundary, coordinate=row["first_difference"])
            if first_violation is None and row["allclose_violation_count"]:
                first_violation = dict(module=module, boundary=boundary,
                                       coordinate=row["largest_tolerance_violations"][0])
    output = tensor_differences(actual["output"], reference["output"], full_token_indices,
                                rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    windows = validate_native_branch(actual["output"], reference["output"], selected_lengths,
        rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, relative_l2_limit=DEFAULT_RELATIVE_L2)
    return dict(modules=comparisons, first_exact_difference=first_difference,
                first_allclose_violation=first_violation, output=output,
                output_window_original_validation=windows,
                original_validation_passed=all(r["passed"] for r in windows))


class _PrefixCaptured(Exception):
    pass


def capture_prefix27(visual, pixels, grid):
    """Run native layers0..27 once; abort after copying their true output/metadata."""
    capture = {}
    def hook(module, args, kwargs, output):
        capture["hidden"] = output.detach().clone()
        capture["kwargs"] = dict(kwargs)
        capture["kwargs"]["cu_seqlens"] = kwargs["cu_seqlens"].detach().clone()
        capture["kwargs"]["position_embeddings"] = tuple(t.detach().clone()
                                                          for t in kwargs["position_embeddings"])
        raise _PrefixCaptured()
    handle = visual.blocks[27].register_forward_hook(hook, with_kwargs=True)
    try:
        try:
            visual(pixels, grid_thw=grid)
        except _PrefixCaptured:
            pass
    finally:
        handle.remove()
    if not capture:
        raise ValueError("Native prefix27 capture did not execute")
    return capture["hidden"], capture["kwargs"]


@contextmanager
def no_tf32():
    import torch
    precision = torch.get_float32_matmul_precision()
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


@contextmanager
def packed_downproj_accumulation(block):
    """Scope the backend setting to down_proj only; restore on every exit path.

    This disables PyTorch's optional reduced-precision FP16 GEMM reduction. It
    does not cast the input/weights/output or claim a specific selected kernel.
    The diagnostic runner invokes this context only for its extra packed call.
    """
    import torch
    module = block.get_submodule("mlp.down_proj")
    existed = "forward" in module.__dict__
    saved = module.__dict__.get("forward")
    original = module.forward
    metadata = dict(scope="Only packed B29 mlp.down_proj.forward",
        entry_flag=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        requested_flag=False, tensor_dtype_changed=False, down_proj_calls=[])

    def forward(self, *args, **kwargs):
        old = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        call = dict(flag_before=old, requested_flag=False)
        metadata["down_proj_calls"].append(call)
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            call["flag_during"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            return original(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old
            call["flag_after"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction

    module.forward = MethodType(forward, module)
    try:
        yield metadata
    finally:
        if existed:
            module.forward = saved
        else:
            delattr(module, "forward")
        metadata["exit_flag"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        metadata["forward_restored"] = (module.__dict__.get("forward") is saved) if existed else "forward" not in module.__dict__


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, help="Frozen12dev requests.json; only failed video is used")
    parser.add_argument("--failure-stats", required=True, help="window_scout_dev12_r1/failed_call_stats.json")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--fp32-reference", action="store_true", help="Also copy only B29 to FP32 with TF32 disabled")
    parser.add_argument("--packed-downproj-fp32-accum", action="store_true",
                        help="One extra packed B29 call; disable reduced-precision FP16 reduction only inside down_proj")
    return parser


def main():
    args = build_parser().parse_args()
    request = select_failed_request(args.requests)
    previous = json.loads(Path(args.failure_stats).read_text())
    if len(previous["calls"]) != 1:
        raise ValueError("Expected one failed diagnostic call")
    previous_call = previous["calls"][0]
    from run_experiment import cooperative_lock, save_json, timestamp, TARGET
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status="starting", pid=os.getpid(), started_at=timestamp(), updated_at=timestamp(),
        args=vars(args), target=TARGET, video_id=FAILED_VIDEO, source_sha256=source_hashes(),
        input_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
                      (("requests", args.requests), ("failure_stats", args.failure_stats))},
        planned_fp16_block29_calls=5+int(args.packed_downproj_fp32_accum),
        planned_fp32_block29_calls=2 if args.fp32_reference else 0,
        completed_block29_calls=0, prefix_captures=0,
        scope="Numerical diagnosis only; no fitted tolerances, policy, QA or latency claims")
    save_json(out / "status.json", status)
    save_json(out / "request_visual_only.json", request)
    try:
        with cooperative_lock(args.physical_gpu, out) as lock:
            status["lock"] = lock
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), TOKENIZERS_PARALLELISM="false",
                              OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            sys.path[:0] = MODULE_PATHS
            import torch
            import cv2
            import transformers
            from torch.nn.attention import sdpa_kernel, SDPBackend
            import hf517_vispec_adapter as bridge
            from local_depth import LocalDepthPolicy
            from window_scout import (pack_windows, choose_windows, scout_mac_accounting,
                                      DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_RELATIVE_L2)
            from compute_cost import calculate_vision_cost
            bridge._native()
            from jev_spec.runtime import prepare_request
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            target = bridge.load_target(TARGET, "cuda:0", dtype="float16", output_head_dtype="float32",
                                        target_sdpa_backend="math")
            visual, block = target.model.visual, target.model.visual.blocks[29]
            if visual.training or 27 in visual.fullatt_block_indexes or 29 in visual.fullatt_block_indexes:
                raise ValueError("Expected inference and identical local-window metadata for blocks27/29")
            native_path = Path(inspect.getsourcefile(type(block))).resolve()
            status.update(status="running", updated_at=timestamp(), torch=torch.__version__,
                transformers=transformers.__version__, native_source={"path": str(native_path),
                    "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest()},
                original_validation_tolerances=dict(rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
                                                   relative_l2_limit=DEFAULT_RELATIVE_L2),
                fp16_backend_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    float32_matmul_precision=torch.get_float32_matmul_precision()),
                input_snapshot="clone", sdpa_backend="math_explicit")
            save_json(out / "status.json", status)
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                packed_request, sampled = prepare_request(target._hf517_processor, request, "cuda:0")
                grid = packed_request["video_grid_thw"]
                grid_list = grid.detach().cpu().tolist()
                if grid_list != previous_call["grid_thw"]:
                    raise ValueError("Current grid differs from the failed experiment")
                pixels = packed_request["pixel_values_videos"].to(dtype=visual.dtype)
                hidden, full_kwargs = capture_prefix27(visual, pixels, grid)
                status.update(prefix_captures=1, updated_at=timestamp())
                layout_policy = LocalDepthPolicy(target, collect_only=True)
                layout_policy.prepare(grid, device=hidden.device)
                layout = layout_policy._layout
                lengths = layout.window_lengths.cpu().tolist()
                choices, _ = choose_windows(layout.grid, layout_policy._window_origins, lengths,
                    layout.window_owners.cpu().tolist(), [0.]*len(lengths))
                selected = choices["geometric_center"]
                expected = [r["global_window_index"] for r in previous_call["scout"]["geometric_center"]["selected_windows"]]
                if selected != expected:
                    raise ValueError("Central windows differ from the failed experiment")
                small, positions, cu, indices, selected_lengths = pack_windows(hidden,
                    full_kwargs["position_embeddings"], lengths, selected, full_kwargs["cu_seqlens"].dtype)
                expected_cu = torch.tensor([0, *layout.window_lengths.cumsum(0).cpu().tolist()],
                    dtype=full_kwargs["cu_seqlens"].dtype, device=hidden.device)
                if not torch.equal(full_kwargs["cu_seqlens"], expected_cu):
                    raise ValueError("Captured prefix metadata differs from native windows")
                if not torch.equal(small, hidden.index_select(0, indices)) or not all(torch.equal(p, q.index_select(0, indices))
                        for p, q in zip(positions, full_kwargs["position_embeddings"])):
                    raise AssertionError("Packed inputs or RoPE differ before execution")
                small_kwargs = dict(full_kwargs, cu_seqlens=cu, position_embeddings=positions)
                same_m = torch.zeros_like(hidden).index_copy_(0, indices, small)
                full_indices = indices.cpu().tolist()
                save_json(out / "capture_metadata.json", dict(video_grid_thw=grid_list,
                    timestamps=list(sampled.timestamps), hidden=_metadata(hidden), selected_windows=selected,
                    selected_window_lengths=selected_lengths, selected_full_token_indices=full_indices,
                    full_cu_seqlens=full_kwargs["cu_seqlens"].cpu().tolist(), packed_cu_seqlens=cu.cpu().tolist(),
                    input_and_rope_exactly_aligned=True, same_m_selected_inputs_exactly_preserved=bool(
                        torch.equal(same_m.index_select(0, indices), small)),
                    prefix_execution="Native blocks0..27 once; blocks28..31 and merger are not executed during capture"))
                traces, report = {}, {"comparisons": {}, "decision": "Diagnostic evidence only; original thresholds unchanged"}
                for name in FP16_CASES:
                    status.update(current=name, updated_at=timestamp())
                    save_json(out / "status.json", status)
                    is_packed = name.startswith("packed")
                    x = small if is_packed else same_m if name == "same_m_zero_padding" else hidden
                    traces[name] = trace_block(block, x, small_kwargs if is_packed else full_kwargs,
                                               None if is_packed else indices)
                    status["completed_block29_calls"] += 1
                    reference_name = "packed" if name == "packed_repeat" else "full"
                    if name != "full":
                        report["comparisons"][f"{name}_vs_{reference_name}"] = compare_traces(
                            traces[name], traces[reference_name], full_indices, selected_lengths)
                        save_json(out / "numeric_report.json", report)
                    status.update(updated_at=timestamp())
                    save_json(out / "status.json", status)
                if args.packed_downproj_fp32_accum:
                    name = "packed_downproj_fp32_accum"
                    status.update(current=name, updated_at=timestamp())
                    save_json(out / "status.json", status)
                    with packed_downproj_accumulation(block) as flag_record:
                        traces[name] = trace_block(block, small, small_kwargs)
                    status["completed_block29_calls"] += 1
                    if len(flag_record["down_proj_calls"]) != 1 or not flag_record["forward_restored"]:
                        raise AssertionError("Expected one scoped down_proj call and a restored forward")
                    report["packed_downproj_fp32_accum"] = dict(flag_record=flag_record,
                        extra_block29_calls=1, packed_tokens=len(small), original_tolerances_unchanged=True,
                        semantics="FP16 input/weights/output unchanged; only the reduced-precision GEMM reduction option is disabled")
                    for reference_name in ("full", "packed"):
                        report["comparisons"][f"{name}_vs_{reference_name}"] = compare_traces(
                            traces[name], traces[reference_name], full_indices, selected_lengths)
                    save_json(out / "numeric_report.json", report)
                    status.update(updated_at=timestamp())
                    save_json(out / "status.json", status)
                if args.fp32_reference:
                    block32 = copy.deepcopy(block).float().eval()
                    kwargs32 = dict(full_kwargs, position_embeddings=tuple(p.float() for p in full_kwargs["position_embeddings"]))
                    small_kwargs32 = dict(small_kwargs, position_embeddings=tuple(p.float() for p in positions))
                    with no_tf32() as flags:
                        status["fp32_backend_flags"] = flags
                        status.update(current="fp32_full", updated_at=timestamp())
                        save_json(out / "status.json", status)
                        traces["fp32_full"] = trace_block(block32, hidden.float(), kwargs32, indices)
                        status["completed_block29_calls"] += 1
                        status.update(current="fp32_packed", updated_at=timestamp())
                        save_json(out / "status.json", status)
                        traces["fp32_packed"] = trace_block(block32, small.float(), small_kwargs32)
                        status["completed_block29_calls"] += 1
                    report["fp32_reference_semantics"] = "Original quantized input, RoPE and weights cast to FP32; no weight reload or upstream recomputation"
                    for a, b in (("fp32_packed", "fp32_full"), ("full", "fp32_full"), ("packed", "fp32_packed")):
                        report["comparisons"][f"{a}_vs_{b}"] = compare_traces(traces[a], traces[b], full_indices, selected_lengths)
                    del block32
                torch.cuda.synchronize()
                config = visual.config.to_dict()
                whole = calculate_vision_cost(grid_list, config, skip_layers=(28, 29))
                scout_cost = scout_mac_accounting(grid_list, config, selected_lengths, layout.anchors.cpu().tolist())
                full_block_macs = whole["full"]["layers"][29]["total_macs"]
                small_block_macs = sum(r["selected_window_layer_macs"] for r in scout_cost["per_time"])
                capture_macs = whole["full"]["patch_embedding_macs"]+sum(r["total_macs"] for r in whole["full"]["layers"][:28])
                report["diagnostic_cost"] = dict(prefix_capture_major_macs=capture_macs,
                    fp16_full_size_block_calls=3, fp16_packed_block_calls=2+int(args.packed_downproj_fp32_accum),
                    fp16_replay_major_macs=3*full_block_macs+(2+int(args.packed_downproj_fp32_accum))*small_block_macs,
                    packed_downproj_accum_extra_block_calls=int(args.packed_downproj_fp32_accum),
                    packed_downproj_accum_extra_major_macs=small_block_macs if args.packed_downproj_fp32_accum else 0,
                    fp32_full_size_block_calls=int(args.fp32_reference), fp32_packed_block_calls=int(args.fp32_reference),
                    fp32_replay_major_macs=(full_block_macs+small_block_macs) if args.fp32_reference else 0,
                    selected_window_fraction=small_block_macs/full_block_macs,
                    excluded=whole["excluded"], timing_status="No wall-time or deployment speed claims")
                report["interpretation"] = ["First differing module identifies where the trajectories separate; later modules can inherit that difference.",
                    "Same-M zero padding retains all native window boundaries and RoPE positions; only unselected windows change.",
                    "A stable same-M result with differing packed results supports a shape-dependent numerical explanation; inspect matched module inputs before attributing it to Linear.",
                    "FP32 reference is optional supporting evidence, not a reason to relax the original FP16 validation threshold."]
                save_json(out / "numeric_report.json", report)
                save_json(out / "vision_config.json", config)
                status.update(status="completed", current=None, completed_at=timestamp(), updated_at=timestamp())
                save_json(out / "status.json", status)
                print(json.dumps({"status": "completed", "out": str(out), "block29_calls": status["completed_block29_calls"]}))
    except BaseException as error:
        status.update(status="failed", failed_at=timestamp(), updated_at=timestamp(), error=repr(error),
                      traceback=traceback.format_exc())
        save_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
