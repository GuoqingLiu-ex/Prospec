"""Finite, label-free development diagnosis of visual layer pairs 20/21 vs 28/29.

Exactly the previously declared 12 development videos are allowed. Original
split labels are preserved as provenance; these videos were reused as the
development set before selecting the new VDC12 holdout. Answer labels and source
questions are unused. Weights remain frozen; no thresholds are fitted and no
timing conclusions are drawn.

Run 12 native visual references and 48 fixed-depth candidate visual forwards.
Continue the 20/21 branch only if all registered conditions pass: its fixed2
maximum-time-block relative L2 macro mean improves by >=10%, >=8/12 videos
improve on that metric, and fixed1 global relative L2 worsens by <=2% relatively.
Otherwise stop this branch; do not broaden the sweep or change the criteria.
Passing is feature-proxy evidence, not a QA-quality or correctness guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "vendor")]
PAIRS = ((20, 21), (28, 29))
DEPTHS = (1, 2)
DECLARED_FRAMES = {
    "v_Ku76zinwWkg": 16, "v_ghxiG0Dnrt0": 16, "v_dRF1C_sBE5M": 16,
    "v_mY7QdIn51S8": 16, "v_Ws12Dt0Ny5Q": 16, "v_vMcXLYBcHsM": 16,
    "v_889_kqLFM0E": 16, "v_4mRkWNLE66E": 16, "7649295416": 12,
    "4170536877": 16, "4494886746": 4, "v_RjztFj0Zl40": 16,
}
CONTINUATION_RULE = {
    "fixed2_max_timeblock_error_macro_relative_improvement_at_least": .10,
    "fixed2_video_error_improvements_at_least": 8,
    "fixed1_global_error_macro_relative_worsening_at_most": .02,
}
EPS = 1e-12


def load_development_requests(path):
    path = Path(path)
    raw = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
           if path.suffix == ".jsonl" else json.loads(path.read_text()))
    if not isinstance(raw, list) or len(raw) != 12:
        raise ValueError("Require exactly the 12 predeclared development requests")
    ids = [str(r["video_id"]) for r in raw]
    if len(set(ids)) != 12 or set(ids) != set(DECLARED_FRAMES):
        raise ValueError("Video IDs differ from the frozen development set")
    if len({r["request_id"] for r in raw}) != 12:
        raise ValueError("Development request IDs must be unique")
    result = []
    for row, video in zip(raw, ids):
        if row["nframes"] != DECLARED_FRAMES[video] or row["max_pixels"] != 200704:
            raise ValueError("Frame count or resolution budget differs from dev capture")
        # Whitelist visual input fields, removing all answers/options and source
        # questions. Neutral text only satisfies prepare_request's API.
        result.append(dict(request_id=row["request_id"], video_id=video,
                           video_path=row["video_path"], nframes=row["nframes"],
                           max_pixels=row["max_pixels"], question="Describe the video.",
                           original_split=row.get("split"),
                           development_role="Previously frozen depth_dev_collect inputs"))
    return result


def validate_registration(registration):
    if (registration.get("pairs") != [list(p) for p in PAIRS] or
            registration.get("fixed_depths") != list(DEPTHS) or
            registration.get("reference_forwards") != 12 or
            registration.get("candidate_forwards") != 48 or
            registration.get("continuation_rule") != CONTINUATION_RULE):
        raise ValueError("Registration differs from the fixed two-pair protocol")


def feature_diagnostics(actual, reference, grid_thw, spatial_merge_size=2):
    """FP64 CPU diagnostics in final native token order, including every token."""
    import torch
    a, b = (x.detach().to(device="cpu", dtype=torch.float64) for x in (actual, reference))
    if a.ndim != 2 or a.shape != b.shape or not a.numel():
        raise ValueError("Expected matching nonempty [merged_tokens,channels] pooler outputs")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite final visual representation")
    grid = grid_thw.tolist() if hasattr(grid_thw, "tolist") else grid_thw
    if type(spatial_merge_size) is not int or spatial_merge_size <= 0:
        raise ValueError("Invalid spatial merge size")
    lengths, identifiers = [], []
    for clip, row in enumerate(grid):
        if (len(row) != 3 or any(type(v) is not int or v <= 0 for v in row) or
                row[1] % spatial_merge_size or row[2] % spatial_merge_size):
            raise ValueError("Invalid or non-merge-aligned grid")
        times, height, width = row
        tokens = height * width // spatial_merge_size**2
        for time in range(times):
            lengths.append(tokens)
            identifiers.append((clip, time))
    if sum(lengths) != a.shape[0]:
        raise ValueError("Final pooler token count disagrees with native grid")
    delta = a - b
    ref_norm = torch.linalg.vector_norm(b, dim=1)
    actual_norm = torch.linalg.vector_norm(a, dim=1)
    token_relative = torch.linalg.vector_norm(delta, dim=1) / ref_norm.clamp_min(EPS)
    cosine = ((a*b).sum(dim=1) /
              (actual_norm.clamp_min(EPS) * ref_norm.clamp_min(EPS))).clamp(-1., 1.)

    def distribution(values):
        return {"p50": float(torch.quantile(values, .5)),
                "p95": float(torch.quantile(values, .95)),
                "min": float(values.min()), "max": float(values.max())}

    time_errors, offset = [], 0
    for (clip, time), length in zip(identifiers, lengths):
        error = torch.linalg.vector_norm(delta[offset:offset+length]) / torch.linalg.vector_norm(
            b[offset:offset+length]).clamp_min(EPS)
        time_errors.append(dict(clip=clip, temporal_block=time, merged_tokens=length,
                                relative_l2=float(error)))
        offset += length
    return dict(shape=list(a.shape), global_relative_l2=float(torch.linalg.vector_norm(delta) /
                torch.linalg.vector_norm(b).clamp_min(EPS)),
                temporal_blocks=time_errors,
                max_temporal_block_relative_l2=max(r["relative_l2"] for r in time_errors),
                token_relative_l2=distribution(token_relative), token_cosine=distribution(cosine),
                token_cosine_error=distribution(1-cosine),
                zero_reference_token_norms=int((ref_norm < EPS).sum()),
                zero_actual_token_norms=int((actual_norm < EPS).sum()),
                scalar_accumulation_dtype="float64_cpu", norm_clamp_min=EPS,
                zero_norm_cosine_definition="Clamped denominator; zero dot product maps to zero cosine",
                interpretation="Feature preservation only; no QA error or correctness guarantee")


def select_continuation(rows):
    """Apply the registered criteria once; no fitted cutoff or extra candidate."""
    by_key = {}
    expected = {(video, pair, depth) for video in DECLARED_FRAMES for pair in PAIRS for depth in DEPTHS}
    for row in rows:
        key = (str(row["video_id"]), tuple(row["skip_layers"]), row["fixed_depth"])
        if key in by_key:
            raise ValueError("Duplicate candidate observation")
        for metric in ("global_relative_l2", "max_temporal_block_relative_l2"):
            value = row["features_vs_native"][metric]
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid feature-proxy observation")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("Require all 48 observations for the exact development videos")
    scores = {}
    for pair in PAIRS:
        scores[str(pair)] = {str(depth): {
            metric: statistics.mean(by_key[(video, pair, depth)]["features_vs_native"][metric]
                                    for video in DECLARED_FRAMES)
            for metric in ("global_relative_l2", "max_temporal_block_relative_l2")}
            for depth in DEPTHS}
    candidate, baseline = scores[str(PAIRS[0])], scores[str(PAIRS[1])]
    new2, old2 = (s["2"]["max_temporal_block_relative_l2"] for s in (candidate, baseline))
    new1, old1 = (s["1"]["global_relative_l2"] for s in (candidate, baseline))
    improved_videos = [video for video in DECLARED_FRAMES if
        by_key[(video, PAIRS[0], 2)]["features_vs_native"]["max_temporal_block_relative_l2"] <
        by_key[(video, PAIRS[1], 2)]["features_vs_native"]["max_temporal_block_relative_l2"]]
    # A zero baseline cannot establish a positive relative improvement.
    improvement = (old2-new2)/old2 if old2 > 0 else None
    worsening = (new1-old1)/old1 if old1 > 0 else (0. if new1 == 0 else None)
    checks = dict(fixed2_macro_local_improvement=old2 > 0 and new2 <= .90 * old2,
                  fixed2_at_least_eight_videos=len(improved_videos) >= 8,
                  fixed1_global_not_worse_than_two_percent=new1 <= 1.02 * old1)
    passed = all(checks.values())
    return dict(continuation_rule=CONTINUATION_RULE, macro_scores=scores, checks=checks,
                fixed2_macro_local_relative_improvement=improvement,
                fixed1_global_relative_worsening=worsening,
                fixed2_improved_video_ids=improved_videos,
                fixed2_improved_videos=len(improved_videos), continue_pair20_21=passed,
                next_action="Candidate only: separately validate gate, QA and latency" if passed else
                            "Stop pair-selection branch; do not broaden layers or change criteria",
                claim_boundary="Feature proxy only; no correctness, QA risk, or performance claim")


def source_hashes():
    paths = [Path(__file__), ROOT / "vision_depth.py", ROOT / "compute_cost.py",
             ROOT / "run_experiment.py", ROOT / "vendor/hf517_vispec_adapter.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=range(8))
    parser.add_argument("--requests", required=True, help="Frozen depth_dev_collect/requests.json or JSONL")
    parser.add_argument("--registration", required=True, help="Pre-existing PAIR12_REGISTRATION.json")
    return parser


def main():
    args = build_parser().parse_args()
    requests = load_development_requests(args.requests)
    registration = json.loads(Path(args.registration).read_text())
    validate_registration(registration)
    from run_experiment import cooperative_lock, save_json, timestamp, TARGET
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status="starting", started_at=timestamp(), updated_at=timestamp(),
                  pid=os.getpid(), args=vars(args), target=TARGET,
                  source_sha256=source_hashes(), input_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for name, path in (("requests", args.requests), ("registration", args.registration))},
                  original_splits={r["video_id"]: r["original_split"] for r in requests},
                  development_provenance="These exact videos were reused as development before new VDC12 selection; original split labels are provenance only",
                  no_training=True, answer_labels_used=False, question_text_used=False,
                  scope="Feature-preservation development diagnosis; no QA or performance evaluation",
                  pairs=[list(p) for p in PAIRS], fixed_depths=list(DEPTHS),
                  continuation_rule=CONTINUATION_RULE, total_visual_forwards=60,
                  completed_reference_forwards=0, completed_candidate_forwards=0)
    save_json(out / "status.json", status)
    save_json(out / "requests_visual_only.json", requests)
    save_json(out / "registration.json", registration)
    rows = []
    try:
        with cooperative_lock(args.physical_gpu, out) as lock:
            status["lock"] = lock
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), TOKENIZERS_PARALLELISM="false",
                              OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            # Lock helpers prepend a legacy directory; keep this run's modules first.
            sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "vendor")]
            import torch
            import cv2
            import transformers
            from torch.nn.attention import sdpa_kernel, SDPBackend
            import hf517_vispec_adapter as bridge
            from vision_depth import depth_policy
            from compute_cost import cost_from_depth_stats
            bridge._native()
            from jev_spec.runtime import prepare_request
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            target = bridge.load_target(TARGET, "cuda:0", dtype="float16", output_head_dtype="float32",
                                        target_sdpa_backend="math")
            visual = target.model.visual
            if any(set(pair) & set(visual.fullatt_block_indexes) for pair in PAIRS):
                raise ValueError("Declared pairs must contain only local-attention layers")
            config = visual.config.to_dict()
            save_json(out / "vision_config.json", config)
            status.update(status="running", updated_at=timestamp(),
                          torch=torch.__version__, transformers=transformers.__version__,
                          device_name=torch.cuda.get_device_name(0), sdpa_backend="math_explicit",
                          memory_scope="Full Target resident; no Draft, no language generation")
            save_json(out / "status.json", status)
            with (out / "candidate_observations.jsonl").open("w") as writer, torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for request in requests:
                    packed, sampled = prepare_request(target._hf517_processor, request, "cuda:0")
                    grid = packed["video_grid_thw"]
                    grid_list = grid.detach().cpu().tolist()
                    pixels = packed["pixel_values_videos"].to(dtype=visual.dtype)
                    output = visual(pixels, grid_thw=grid)
                    reference = output.pooler_output.detach().cpu()
                    del output
                    status["completed_reference_forwards"] += 1
                    same_depth_macs = {}
                    for pair in PAIRS:
                        for depth in DEPTHS:
                            with depth_policy(target, mode=f"fixed{depth}", skip_layers=pair,
                                              prefix_layer=pair[0]-1) as policy:
                                policy.prepare(grid, device=pixels.device)
                                output = visual(pixels, grid_thw=grid)
                            diagnostic = feature_diagnostics(output.pooler_output, reference, grid_list,
                                                             int(visual.spatial_merge_size))
                            stats = policy.summary()
                            cost = cost_from_depth_stats(stats, config)
                            matrix_macs = cost["executed_vision_major_matrix_macs"]
                            if depth in same_depth_macs and same_depth_macs[depth] != matrix_macs:
                                raise AssertionError("Candidate pairs have different matrix MACs at the same fixed depth")
                            same_depth_macs[depth] = matrix_macs
                            row = dict(request_id=request["request_id"], video_id=request["video_id"],
                                       skip_layers=list(pair), prefix_layer=pair[0]-1, fixed_depth=depth,
                                       features_vs_native=diagnostic, video_grid_thw=grid_list,
                                       timestamps=list(sampled.timestamps), stats=stats,
                                       vision_matrix_macs={k:cost[k] for k in (
                                           "full_vision_major_matrix_macs", "executed_vision_major_matrix_macs",
                                           "saved_macs", "vision_major_matrix_mac_reduction")})
                            rows.append(row)
                            writer.write(json.dumps(row, allow_nan=False) + "\n")
                            writer.flush()
                            del output
                            status.update(completed_candidate_forwards=len(rows), updated_at=timestamp(), current=dict(
                                video_id=request["video_id"], pair=pair, fixed_depth=depth))
                            save_json(out / "status.json", status)
                    del reference, packed, pixels, grid
            selection = select_continuation(rows)
            selection["same_matrix_macs_per_video_fixed_depth_verified"] = True
            save_json(out / "selection.json", selection)
            status.update(status="completed", completed_at=timestamp(), updated_at=timestamp(),
                          current=None, selection=selection)
            save_json(out / "status.json", status)
            print(json.dumps(selection, indent=2, allow_nan=False))
    except BaseException as error:
        status.update(status="failed", failed_at=timestamp(), updated_at=timestamp(), error=repr(error),
                      traceback=traceback.format_exc())
        save_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
