"""Frozen-12dev, full-depth native-window scout diagnosis; no deployed gate.

Each visual forward adds three packed block calls per selection rule and one
full-window B29(x) counterfactual. All added computation is diagnostic. No QA
answers, fitted thresholds, timing claims or post-hoc choice of window rule.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULE_PATHS = [str(HERE), str(ROOT / "training_free"), str(ROOT), str(ROOT / "vendor")]
sys.path[:0] = MODULE_PATHS
from calibrate_pair12 import load_development_requests


def source_hashes():
    paths = [Path(__file__), HERE / "window_scout.py", HERE / "scout_precision.py", HERE / "prefix_energy.py",
             HERE / "calibrate_pair12.py", ROOT / "training_free/local_depth.py",
             ROOT / "vision_depth.py", ROOT / "compute_cost.py", ROOT / "run_experiment.py",
             ROOT / "vendor/hf517_vispec_adapter.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, help="Frozen depth_dev_collect/requests.json")
    parser.add_argument("--pair12-results", required=True, help="pair12 candidate_observations.jsonl")
    parser.add_argument("--prefix-association", required=True, help="Completed prefix_energy_dev12_r1/association.json")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--packed-downproj-fp32-accum", action="store_true",
                        help="Disable reduced-precision FP16 reduction only inside packed probe down_proj calls")
    parser.add_argument("--diagnostic-allow-numerical-mismatch", action="store_true",
                        help="Diagnostic only: retain failed numerical checks and continue; layout/nonfinite/restoration failures still abort")
    return parser


def main():
    args = build_parser().parse_args()
    requests = load_development_requests(args.requests)
    pair_rows = [json.loads(line) for line in Path(args.pair12_results).read_text().splitlines() if line.strip()]
    pair_rows = [r for r in pair_rows if r["skip_layers"] == [28, 29] and r["fixed_depth"] in (1, 2)]
    prefix_association = json.loads(Path(args.prefix_association).read_text())
    from run_experiment import cooperative_lock, save_json, timestamp, TARGET
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status="starting", pid=os.getpid(), started_at=timestamp(), updated_at=timestamp(),
        args=vars(args), target=TARGET, source_sha256=source_hashes(),
        input_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
                      (("requests", args.requests), ("pair12_results", args.pair12_results),
                       ("prefix_association", args.prefix_association))},
        development_provenance="Exact previously frozen depth_dev_collect videos reused as development before VDC12 selection",
        original_splits={r["video_id"]: r["original_split"] for r in requests},
        total_visual_forwards=12, completed_visual_forwards=0, completed_packed_scout_block_calls=0,
        completed_full_counterfactual_block_calls=0, no_training=True, answer_labels_used=False,
        packed_downproj_fp32_accum=args.packed_downproj_fp32_accum,
        diagnostic_allow_numerical_mismatch=args.diagnostic_allow_numerical_mismatch,
        strict_validation_passed=None, numerical_mismatch_videos=[],
        question_text_used=False, thresholds_fitted=False, all_main_path_visual_blocks_executed=True,
        scope="Local and final feature diagnosis only; no QA, speed or deployed policy claim")
    save_json(out / "status.json", status)
    save_json(out / "requests_visual_only.json", requests)
    records, policy = [], None
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
            from window_scout import (NativeWindowScout, STRATEGIES, scout_mac_accounting,
                                      associate_scout_errors, DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_RELATIVE_L2)
            from compute_cost import calculate_vision_cost
            bridge._native()
            from jev_spec.runtime import prepare_request
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            target = bridge.load_target(TARGET, "cuda:0", dtype="float16", output_head_dtype="float32",
                                        target_sdpa_backend="math")
            visual = target.model.visual
            if len(visual.blocks) != 32 or {28, 29} & set(visual.fullatt_block_indexes):
                raise ValueError("Expected the registered 32-layer architecture with local blocks28/29")
            config = visual.config.to_dict()
            save_json(out / "vision_config.json", config)
            native_path = Path(inspect.getsourcefile(type(visual.blocks[27]))).resolve()
            status.update(status="running", updated_at=timestamp(), torch=torch.__version__,
                transformers=transformers.__version__, device_name=torch.cuda.get_device_name(0),
                sdpa_backend="math_explicit", input_snapshot="clone", prefix_layer=27, candidate_layers=[28, 29],
                fp16_reduction_flag_outside_probes=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                strategies=list(STRATEGIES), anchor_stride=4,
                validation_tolerances=dict(rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, relative_l2_limit=DEFAULT_RELATIVE_L2),
                native_source={"path": str(native_path), "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest()},
                memory_scope="Full frozen Target resident; no Draft or language generation",
                numerical_diagnostics="FP64 CPU; probes use model FP16 math-SDPA")
            save_json(out / "status.json", status)
            with (out / "results.jsonl").open("w") as writer, torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for request in requests:
                    status.update(current=request["video_id"], updated_at=timestamp())
                    save_json(out / "status.json", status)
                    packed, sampled = prepare_request(target._hf517_processor, request, "cuda:0")
                    grid = packed["video_grid_thw"]
                    grid_list = grid.detach().cpu().tolist()
                    pixels = packed["pixel_values_videos"].to(dtype=visual.dtype)
                    with NativeWindowScout(target, packed_downproj_fp32_accum=args.packed_downproj_fp32_accum,
                            diagnostic_allow_numerical_mismatch=args.diagnostic_allow_numerical_mismatch) as policy:
                        policy.prepare(grid, device=pixels.device)
                        output = visual(pixels, grid_thw=grid)
                    torch.cuda.synchronize()
                    stats = policy.summary()
                    if (len(stats["calls"]) != 1 or any(stats["calls"][0]["actions"]) or
                            stats["executed_token_blocks"] != stats["original_token_blocks"]):
                        raise AssertionError("Scout altered the full-depth main path")
                    call = stats["calls"][0]
                    if not all(torch.isfinite(tensor).all().item() for tensor in
                               (output.pooler_output, output.last_hidden_state)):
                        raise ValueError("Nonfinite full-depth visual output")
                    costs = {strategy: scout_mac_accounting(grid_list, config,
                        [r["tokens"] for r in call["scout"][strategy]["selected_windows"]], call["anchors"])
                        for strategy in STRATEGIES}
                    native_cost = calculate_vision_cost(grid_list, config, skip_layers=(28, 29))
                    whole_layer = native_cost["full"]["layers"][29]["total_macs"]
                    native_total = native_cost["full"]["vision_major_matrix_macs"]
                    probes_total = sum(cost["diagnostic_extra_three_calls_macs"] for cost in costs.values())
                    costs["actual_diagnostic_total"] = dict(native_vision_major_macs=native_total,
                        two_rules_packed_probe_extra_macs=probes_total,
                        full_window_counterfactual_extra_macs=whole_layer,
                        total_major_macs=native_total+probes_total+whole_layer,
                        packed_scout_block_calls=6, full_window_counterfactual_block_calls=1,
                        packed_downproj_fp32_accum=args.packed_downproj_fp32_accum,
                        precision_note="Same matrix operation counts; reduction-kernel latency is not inferred from MACs",
                        window_strategy_overlap_reused=False,
                        scope="All added calls diagnostic; full counterfactual is excluded from ideal deployed-scout bounds")
                    record = dict(request_id=request["request_id"], video_id=request["video_id"],
                        video_grid_thw=grid_list, timestamps=list(sampled.timestamps), stats=stats,
                        matrix_macs=costs, output_shape=list(output.pooler_output.shape))
                    records.append(record)
                    writer.write(json.dumps(record, allow_nan=False)+"\n")
                    writer.flush()
                    save_json(out / "scout_records.json", records)
                    status.update(completed_visual_forwards=len(records), updated_at=timestamp(),
                                  completed_packed_scout_block_calls=6*len(records),
                                  completed_full_counterfactual_block_calls=len(records),
                                  strict_validation_passed=all(r["stats"]["strict_validation_passed"] for r in records),
                                  numerical_mismatch_videos=[r["video_id"] for r in records
                                                              if not r["stats"]["strict_validation_passed"]])
                    save_json(out / "status.json", status)
                    del output, packed, pixels, grid
                    policy = None
            association = associate_scout_errors(records, pair_rows, prefix_association,
                diagnostic_allow_numerical_mismatch=args.diagnostic_allow_numerical_mismatch)
            save_json(out / "association.json", association)
            status.update(status="completed", current=None, completed_at=timestamp(), updated_at=timestamp(),
                          per_strategy_screens=association["per_strategy_screens"])
            save_json(out / "status.json", status)
            print(json.dumps({"status": "completed", "videos": len(records), "out": str(out)}))
    except BaseException as error:
        if policy is not None and policy.records:
            save_json(out / "failed_call_stats.json", policy.summary())
        status.update(status="failed", failed_at=timestamp(), updated_at=timestamp(), error=repr(error),
                      traceback=traceback.format_exc())
        save_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
