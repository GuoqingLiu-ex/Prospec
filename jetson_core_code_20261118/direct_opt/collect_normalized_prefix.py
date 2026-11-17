"""One fixed RMS-normalized prefix27 diagnosis on the original 12dev videos.

All visual blocks execute once. The actual merger.ln_q runs twice additionally
on the prefix input/output; no Transformer probes, QA labels, threshold fitting
or timing claim. Original prefix diagnostics remain as controls. A failed
predeclared association screen stops this branch without another aggregator.
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
from collect_prefix_energy import read_pair_rows


def source_hashes():
    paths = [Path(__file__), HERE / "normalized_prefix.py", HERE / "prefix_energy.py",
             HERE / "collect_prefix_energy.py", HERE / "calibrate_pair12.py",
             ROOT / "training_free/local_depth.py", ROOT / "vision_depth.py",
             ROOT / "run_experiment.py", ROOT / "vendor/hf517_vispec_adapter.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, help="Frozen depth_dev_collect/requests.json")
    parser.add_argument("--pair12-results", required=True, help="Existing pair12 candidate_observations.jsonl")
    parser.add_argument("--prefix-association", required=True, help="Original prefix_energy_dev12_r1/association.json")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--physical-gpu", required=True, type=int, choices=range(8))
    return parser


def main():
    args = build_parser().parse_args()
    requests = load_development_requests(args.requests)
    pair_rows = read_pair_rows(args.pair12_results)
    raw_association = json.loads(Path(args.prefix_association).read_text())
    from run_experiment import cooperative_lock, save_json, timestamp, TARGET
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status="starting", pid=os.getpid(), started_at=timestamp(), updated_at=timestamp(),
        args=vars(args), target=TARGET, source_sha256=source_hashes(),
        input_sha256={key: hashlib.sha256(Path(value).read_bytes()).hexdigest() for key,value in
                      (("requests",args.requests),("pair12_results",args.pair12_results),
                       ("prefix_association",args.prefix_association))},
        original_splits={r["video_id"]:r["original_split"] for r in requests},
        development_provenance="Exact prior depth_dev_collect videos reused as development before VDC12 selection; source split labels are provenance only",
        total_visual_forwards=12, completed_visual_forwards=0, completed_extra_rmsnorm_calls=0,
        extra_transformer_calls=0, no_training=True, answer_labels_used=False, question_text_used=False,
        thresholds_fitted=False, aggregators_searched=False, all_visual_blocks_executed=True,
        scope="Feature-L2 proxy diagnosis only; no QA quality or latency claim; stop if the fixed primary screen fails")
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
            from normalized_prefix import (NormalizedPrefixCollector, compare_normalized_pair_errors,
                                            PRIMARY_SIGNAL, SIGNAL_DEFINITION, rmsnorm_metadata)
            bridge._native()
            from jev_spec.runtime import prepare_request
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            target = bridge.load_target(TARGET, "cuda:0", dtype="float16", output_head_dtype="float32",
                                        target_sdpa_backend="math")
            visual = target.model.visual
            if len(visual.blocks) != 32 or {28,29} & set(visual.fullatt_block_indexes):
                raise ValueError("Expected native 32-layer architecture and local blocks28/29")
            norm_info = rmsnorm_metadata(visual.merger.ln_q)
            source = Path(inspect.getsourcefile(type(visual.merger.ln_q))).resolve()
            save_json(out / "vision_config.json", visual.config.to_dict())
            status.update(status="running", updated_at=timestamp(), torch=torch.__version__,
                transformers=transformers.__version__, device_name=torch.cuda.get_device_name(0),
                sdpa_backend="math_explicit", input_snapshot="clone", prefix_layer=27,
                comparison_layers=[28,29], anchor_stride=4, normalization=norm_info,
                native_source={"path":str(source),"sha256":hashlib.sha256(source.read_bytes()).hexdigest()},
                primary_signal=PRIMARY_SIGNAL, signal_definition=SIGNAL_DEFINITION,
                scalar_accumulation_dtype="float32", normalization_dtype="Actual model module and weights; no dtype override",
                memory_scope="Full frozen Target resident; no Draft or language generation",
                overhead_scope="Two extra full-feature RMSNorm calls, temporary outputs and FP32 scans; not free and not a deployment timing benchmark")
            save_json(out / "status.json", status)
            with (out / "results.jsonl").open("w") as writer, torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for request in requests:
                    status.update(current=request["video_id"], updated_at=timestamp())
                    save_json(out / "status.json", status)
                    packed, sampled = prepare_request(target._hf517_processor, request, "cuda:0")
                    grid = packed["video_grid_thw"]
                    pixels = packed["pixel_values_videos"].to(dtype=visual.dtype)
                    with NormalizedPrefixCollector(target) as policy:
                        policy.prepare(grid, device=pixels.device)
                        output = visual(pixels, grid_thw=grid)
                    torch.cuda.synchronize()
                    stats = policy.summary()
                    if (len(stats["calls"]) != 1 or any(stats["calls"][0]["actions"]) or
                            stats["executed_token_blocks"] != stats["original_token_blocks"]):
                        raise AssertionError("Collector changed full-depth execution")
                    if not all(torch.isfinite(x).all().item() for x in (output.pooler_output,output.last_hidden_state)):
                        raise ValueError("Nonfinite full-depth visual output")
                    record = dict(request_id=request["request_id"],video_id=request["video_id"],
                        video_grid_thw=grid.detach().cpu().tolist(),timestamps=list(sampled.timestamps),
                        stats=stats,output_shape=list(output.pooler_output.shape))
                    records.append(record)
                    writer.write(json.dumps(record,allow_nan=False)+"\n")
                    writer.flush()
                    save_json(out / "prefix_records.json", records)
                    status.update(completed_visual_forwards=len(records),completed_extra_rmsnorm_calls=2*len(records),
                                  updated_at=timestamp())
                    save_json(out / "status.json", status)
                    del output,packed,pixels,grid
                    policy = None
            association = compare_normalized_pair_errors(records,pair_rows,raw_association)
            save_json(out / "association.json",association)
            status.update(status="completed",current=None,completed_at=timestamp(),updated_at=timestamp(),
                          registered_screen=association["registered_screen"])
            save_json(out / "status.json",status)
            print(json.dumps({"status":"completed","videos":len(records),"out":str(out)}))
    except BaseException as error:
        if policy is not None and policy.records:
            save_json(out / "failed_call_stats.json",policy.summary())
        status.update(status="failed",failed_at=timestamp(),updated_at=timestamp(),error=repr(error),
                      traceback=traceback.format_exc())
        save_json(out / "status.json",status)
        raise


if __name__ == "__main__":
    main()
