"""Real visual forwards: native/fixed1/fixed2/local/direct, or dev collection.

No language generation or Draft is run. Normal mode reuses vision_microbench's
two warmups/mode, randomized paired repeats, explicit math SDPA, full-policy
wall timing, CUDA intervals and offline FP64 pooler diagnostics. Input IO is
outside timing. Each policy is constructed anew for each cell.

--collect-only runs full-depth visual forwards for every selected request and
exports real prefix statistics. It is not a timing or QA evaluation. Use this
only on development inputs when selecting thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "training_free"), str(ROOT / "vendor")]


def _sources():
    paths = [HERE / "depth.py", Path(__file__), ROOT / "vision_depth.py",
             ROOT / "training_free/local_depth.py", ROOT / "vision_microbench.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _collect(args):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import hf517_vispec_adapter as bridge
    from depth import DirectDepthPolicy
    from run_experiment import cooperative_lock, save_json, timestamp, TARGET
    bridge._native()
    from jev_spec.runtime import prepare_request

    requests = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    if args.request_id:
        wanted = set(args.request_id)
        requests = [r for r in requests if r["request_id"] in wanted]
        if wanted != {r["request_id"] for r in requests}:
            raise ValueError("Unknown requested request ID")
    if not requests:
        raise ValueError("No development requests")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status="starting", args=vars(args), started_at=timestamp(),
                  source_sha256=_sources(), scope="full_depth_prefix_statistics_only",
                  answer_labels_used=False, total_cells=len(requests), completed_cells=0)
    save_json(out / "status.json", status)
    save_json(out / "requests.json", requests)
    records = []
    try:
        with cooperative_lock(args.physical_gpu, out) as lock:
            status["lock"] = lock
            torch.set_num_threads(1)
            target = bridge.load_target(TARGET, "cuda:0", dtype="float16",
                                        output_head_dtype="float32", target_sdpa_backend="math")
            visual = target.model.visual
            save_json(out / "vision_config.json", visual.config.to_dict())
            status.update(status="running", device_name=torch.cuda.get_device_name(0),
                          memory_scope="Full Target resident; no Draft")
            save_json(out / "status.json", status)
            with (out / "results.jsonl").open("w") as writer, torch.inference_mode():
                for request in requests:
                    packed, sampled = prepare_request(target._hf517_processor, request, "cuda:0")
                    grid = packed["video_grid_thw"]
                    pixels = packed["pixel_values_videos"].to(visual.dtype)
                    with DirectDepthPolicy(target, collect_only=True,
                            anchor_stride=args.anchor_stride,
                            summary_backend=args.summary_backend) as policy, sdpa_kernel(SDPBackend.MATH):
                        policy.prepare(grid, device=pixels.device)
                        output = visual(pixels, grid_thw=grid)
                        torch.cuda.synchronize()
                    stats = policy.summary()
                    record = dict(request_id=request["request_id"], video_id=request.get("video_id"),
                                  phase="collect", stats=stats, timestamps=list(sampled.timestamps),
                                  video_grid_thw=grid.cpu().tolist())
                    writer.write(json.dumps(record) + "\n")
                    writer.flush()
                    records.append(dict(video_id=request.get("video_id", request["request_id"]),
                                        request_id=request["request_id"],
                                        features=stats["calls"][0]["features"]))
                    status.update(completed_cells=len(records), current=request["request_id"])
                    save_json(out / "status.json", status)
                    del output, packed, pixels, grid
            save_json(out / "prefix_records.json", records)
            status.update(status="completed", completed_at=timestamp(), current=None)
            save_json(out / "status.json", status)
    except BaseException as error:
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        save_json(out / "status.json", status)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--physical-gpu", required=True, type=int)
    p.add_argument("--thresholds", help="Frozen LOCAL four-key thresholds; required for timing")
    p.add_argument("--request-id", action="append")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--anchor-stride", type=int, default=4)
    p.add_argument("--summary-backend", choices=("segment", "index_add"), default="segment")
    p.add_argument("--collect-only", action="store_true")
    args = p.parse_args()
    if not args.collect_only and not args.thresholds:
        p.error("Timing requires explicit development-frozen --thresholds")
    if Path(args.out).exists():
        p.error("Output directory already exists; use a new directory")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.collect_only:
        return _collect(args)

    import vision_microbench
    import vision_depth
    from local_depth import LocalDepthPolicy
    from depth import DirectDepthPolicy, direct_depth_policy
    original, original_modes, original_argv = vision_depth.depth_policy, vision_microbench.MODES, sys.argv
    thresholds_record = json.loads(Path(args.thresholds).read_text())
    frozen = thresholds_record.get("thresholds", thresholds_record)

    def route(target, mode, **kwargs):
        kwargs["thresholds"] = frozen
        if mode == "local":
            return LocalDepthPolicy(target, **kwargs)
        if mode == "direct":
            return DirectDepthPolicy(target, summary_backend=args.summary_backend, **kwargs)
        return direct_depth_policy(target, mode=mode, **kwargs)

    vision_depth.depth_policy = route
    vision_microbench.MODES = ("native", "fixed1", "fixed2", "local", "direct")
    sys.argv = [original_argv[0], "--out", args.out, "--physical-gpu", str(args.physical_gpu),
                "--requests", args.requests, "--thresholds", args.thresholds,
                "--skip-layers", "28", "29", "--prefix-layer", "27",
                "--anchor-stride", str(args.anchor_stride), "--repeats", str(args.repeats),
                "--seed", str(args.seed)]
    for request in args.request_id or []:
        sys.argv += ["--request-id", request]
    try:
        vision_microbench.main()
    finally:
        vision_depth.depth_policy, vision_microbench.MODES, sys.argv = original, original_modes, original_argv
        out = Path(args.out)
        if out.exists():
            receipt = dict(args=vars(args), thresholds=frozen, source_sha256=_sources(),
                           scope="Full visual module including fresh policy; not E2E or QA",
                           fixed2_note="Quality boundary control; no quality pass is implied")
            result = out / "results.jsonl"
            if result.exists():
                rows = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
                local = {(r["request_id"], r["repeat"]): r for r in rows
                         if r["mode"] == "local" and r["phase"] == "measure"}
                comparison = []
                for r in rows:
                    key = (r["request_id"], r["repeat"])
                    if r["mode"] == "direct" and r["phase"] == "measure" and key in local:
                        a, b = local[key]["stats"]["calls"][0], r["stats"]["calls"][0]
                        comparison.append(dict(request_id=key[0], repeat=key[1],
                                               actions_equal=a["actions"] == b["actions"],
                                               local_actions=a["actions"], direct_actions=b["actions"]))
                receipt["same_input_action_comparison"] = comparison
            (out / "direct_depth_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
