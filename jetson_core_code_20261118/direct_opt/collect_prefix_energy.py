"""Collect prefix27 residual diagnostics on the frozen 12 development videos.

All 32 visual blocks execute. No QA answers, weight training, threshold fitting,
or timing claims are involved. The FP32 collector defaults to cloning the prefix
input; retaining a reference additionally requires the audited native source.
Only the predeclared 28/29 fixed1/fixed2 feature errors may be associated with
these signals. The association screen does not establish task quality.
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
    paths = [Path(__file__), HERE / "prefix_energy.py", HERE / "calibrate_pair12.py",
             ROOT / "training_free/local_depth.py", ROOT / "vision_depth.py",
             ROOT / "run_experiment.py", ROOT / "vendor/hf517_vispec_adapter.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def read_pair_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    # Keep only the registered pair and the feature diagnostics; no QA data is
    # consumed. compare_pair_errors verifies coverage, identities and sampling.
    return [r for r in rows if r["skip_layers"] == [28, 29] and r["fixed_depth"] in (1, 2)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, help="Frozen depth_dev_collect/requests.json")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--pair12-results", help="Existing pair12 candidate_observations.jsonl")
    parser.add_argument("--input-snapshot", choices=("clone", "reference"), default="clone")
    return parser


def main():
    args = build_parser().parse_args()
    requests = load_development_requests(args.requests)
    pair_rows = read_pair_rows(args.pair12_results) if args.pair12_results else None
    from run_experiment import cooperative_lock, save_json, timestamp, TARGET
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    inputs = {"requests": args.requests}
    if args.pair12_results:
        inputs["pair12_results"] = args.pair12_results
    status = dict(status="starting", pid=os.getpid(), started_at=timestamp(), updated_at=timestamp(),
                  args=vars(args), target=TARGET, source_sha256=source_hashes(),
                  input_sha256={key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                for key, path in inputs.items()},
                  original_splits={r["video_id"]: r["original_split"] for r in requests},
                  development_provenance="Exact prior depth_dev_collect videos reused as development before new VDC12 selection; original split labels are provenance only",
                  prefix_layer=27, comparison_layers=[28, 29], anchor_stride=4,
                  total_visual_forwards=12, completed_visual_forwards=0,
                  no_training=True, answer_labels_used=False, question_text_used=False,
                  thresholds_fitted=False, all_visual_blocks_executed=True,
                  scope="Prefix feature diagnostics and descriptive association only; no timing or QA claim")
    save_json(out / "status.json", status)
    save_json(out / "requests_visual_only.json", requests)
    records = []
    try:
        with cooperative_lock(args.physical_gpu, out) as lock:
            status["lock"] = lock
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), TOKENIZERS_PARALLELISM="false",
                              OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            # The lock helper prepends a legacy run. Restore this snapshot's
            # paths before importing model and diagnostic implementations.
            sys.path[:0] = MODULE_PATHS
            import torch
            import cv2
            import transformers
            from torch.nn.attention import sdpa_kernel, SDPBackend
            import hf517_vispec_adapter as bridge
            from prefix_energy import PrefixEnergyCollector, compare_pair_errors, SIGNAL_DEFINITIONS
            bridge._native()
            from jev_spec.runtime import prepare_request
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            target = bridge.load_target(TARGET, "cuda:0", dtype="float16", output_head_dtype="float32",
                                        target_sdpa_backend="math")
            visual = target.model.visual
            if len(visual.blocks) != 32 or {28, 29} & set(visual.fullatt_block_indexes):
                raise ValueError("Expected the registered 32-layer visual architecture")
            native_path = Path(inspect.getsourcefile(type(visual.blocks[27]))).resolve()
            save_json(out / "vision_config.json", visual.config.to_dict())
            status.update(status="running", updated_at=timestamp(), torch=torch.__version__,
                          transformers=transformers.__version__, device_name=torch.cuda.get_device_name(0),
                          sdpa_backend="math_explicit", scalar_accumulation_dtype="float32_reference",
                          native_prefix_source={"path": str(native_path),
                              "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest()},
                          signal_definitions=SIGNAL_DEFINITIONS,
                          memory_scope="Full frozen Target resident; no Draft, no language generation")
            save_json(out / "status.json", status)
            with (out / "results.jsonl").open("w") as writer, torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for request in requests:
                    status.update(current=request["video_id"], updated_at=timestamp())
                    save_json(out / "status.json", status)
                    packed, sampled = prepare_request(target._hf517_processor, request, "cuda:0")
                    grid = packed["video_grid_thw"]
                    pixels = packed["pixel_values_videos"].to(dtype=visual.dtype)
                    with PrefixEnergyCollector(target, input_snapshot=args.input_snapshot) as policy:
                        policy.prepare(grid, device=pixels.device)
                        output = visual(pixels, grid_thw=grid)
                    torch.cuda.synchronize()
                    stats = policy.summary()
                    if (len(stats["calls"]) != 1 or any(stats["calls"][0]["actions"]) or
                            stats["executed_token_blocks"] != stats["original_token_blocks"]):
                        raise AssertionError("Diagnostic collection changed the execution budget")
                    record = dict(request_id=request["request_id"], video_id=request["video_id"],
                                  video_grid_thw=grid.detach().cpu().tolist(),
                                  timestamps=list(sampled.timestamps), stats=stats,
                                  output_shape=list(output.pooler_output.shape))
                    records.append(record)
                    writer.write(json.dumps(record, allow_nan=False) + "\n")
                    writer.flush()
                    save_json(out / "prefix_records.json", records)
                    status.update(completed_visual_forwards=len(records), updated_at=timestamp())
                    save_json(out / "status.json", status)
                    del output, policy, packed, pixels, grid
            if pair_rows is not None:
                association = compare_pair_errors(records, pair_rows)
                save_json(out / "association.json", association)
                status["registered_screen"] = association["registered_screen"]
            status.update(status="completed", current=None, completed_at=timestamp(), updated_at=timestamp())
            save_json(out / "status.json", status)
            print(json.dumps({"status": "completed", "videos": len(records), "out": str(out)}, allow_nan=False))
    except BaseException as error:
        status.update(status="failed", failed_at=timestamp(), updated_at=timestamp(), error=repr(error),
                      traceback=traceback.format_exc())
        save_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
