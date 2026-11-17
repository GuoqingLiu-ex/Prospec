"""Free continuation after a shared baseline prefix.

This is not a fixed Target trajectory. Draft KV is reset and the request's own
pixels are prefilled with the baseline prefix appended. Later tokens are free
generation and may diverge. The 12:37 attempt died at `import jev_spec` before
`bridge._native()`; this entry calls that initializer first.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "training_free"), str(ROOT / "training_free_opt"), str(ROOT / "vendor")]

from prefix_decode_diag import (
    DIAGNOSTIC_LENGTHS, continuation_metrics, execute_shared_prefix, group_free_continuations,
    plan_diagnostics,
)


def _ready(value):
    if isinstance(value, dict):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--arms", nargs="+", default=["reorder", "reorder_difference50"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    requests = {}
    for line in Path(args.requests).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            requests[row["request_id"]] = row
    measured = []
    for line in Path(args.baseline_results).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") == "measure" and row.get("arm") == "baseline":
            measured.append(row)
    planned = {item["request_id"]: item for item, _selected in plan_diagnostics(measured)}
    by_id = {row["request_id"]: row for row in measured}
    jobs = []
    for request_id, item in planned.items():
        tokens = by_id[request_id]["output_token_ids"]
        for length in DIAGNOSTIC_LENGTHS:
            if item[f"eligible_prefix_{length}"]:
                prefix = [int(token) for token in tokens[:length]]
                if any(token in (151645, 151643) for token in prefix):
                    continue
                for arm in args.arms:
                    jobs.append((request_id, arm, length, prefix))
    (out / "plan.json").write_text(json.dumps({
        "diagnostic_kind": "free_continuation_after_shared_prefix",
        "fixed_target_trajectory": False,
        "jobs": [{"request_id": request_id, "arm": arm, "prefix": length} for request_id, arm, length, _prefix in jobs],
        "baseline_dir_not_modified": str(Path(args.baseline_results).parent),
        "previous_attempt": "long_answer_prefix_replay_pilot12 import failure is kept and not reused",
    }, indent=2) + "\n")
    from run_schemes import NativeDepthPolicy, OLD_RUN, TARGET, cooperative_lock, save_json, timestamp
    from scheme_context import SCHEMES, SchemeRuntime
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    status = {
        "status": "starting", "started_at": timestamp(), "jobs": len(jobs),
        "ok_jobs": 0, "failed_jobs": 0,
        "diagnostic_kind": "free_continuation_after_shared_prefix",
        "fixed_target_trajectory": False,
    }
    save_json(out / "status.json", status)
    results_path = out / "results.jsonl"
    stored = []
    with cooperative_lock(args.physical_gpu, out):
        import torch
        import hf517_vispec_adapter as bridge
        import vispec_joint_adapter as joint
        from hardware_metrics import HardwareSampler
        from vispec_operators import new_tree_budget
        bridge._native()
        from jev_spec.runtime import prepare_request
        torch.set_num_threads(1)
        target = bridge.load_target(TARGET, "cuda:0", dtype="float16",
                                    output_head_dtype="float32", target_sdpa_backend="math")
        processor = target._hf517_processor
        spec = bridge.load_spec(target, str(OLD_RUN / "models/ViSpec-Qwen2.5-VL-7B-Instruct"),
                                total_token=48, depth=3, top_k=8, num_q=2, capacity=4096)
        budget = new_tree_budget(30, 3, 8)
        runtime = SchemeRuntime(spec.spec_layer)
        status.update(status="running", pid=os.getpid())
        save_json(out / "status.json", status)
        with results_path.open("w") as writer:
            for request_id, arm, length, prefix in jobs:
                started = time.perf_counter()
                record = {
                    "status": "failed",
                    "diagnostic_kind": "free_continuation_after_shared_prefix",
                    "fixed_target_trajectory": False,
                    "request_id": request_id,
                    "arm": arm,
                    "prefix_tokens": length,
                    "continuation_budget": length,
                    "prefix_source": {
                        "baseline_results": str(args.baseline_results),
                        "request_id": request_id,
                        "arm": "baseline",
                        "phase": "measure",
                        "prefix_tokens": length,
                    },
                    "prefix_token_ids": prefix,
                    "draft_reset": True,
                    "latency_scope": "includes video resample, prompt-plus-prefix prefill, and free continuation; CUDA synced; not a fixed-trajectory latency",
                }
                sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.1, process_interval_s=0.5)
                sampler.start()
                try:
                    packed, _sampled = prepare_request(processor, requests[request_id], "cuda:0")
                    scheme = SCHEMES[arm]
                    if scheme["depth"] != "fixed0":
                        raise ValueError("this replay keeps target depth fixed0")

                    def generate(extended, max_new_tokens, eos_token_ids, _scheme=scheme):
                        spec.spec_layer.reset_kv()
                        if hasattr(spec.spec_layer, "last_img_hidden"):
                            spec.spec_layer.last_img_hidden = None
                        with NativeDepthPolicy(target) as policy, runtime.activate(joint, _scheme) as state:
                            policy.prepare(extended["video_grid_thw"], device=target.model.visual.device)
                            result = joint.generate_joint(
                                spec, extended, max_new_tokens=max_new_tokens, visual_m=_scheme["ratio"],
                                tree_budget=budget, merge_operator="cosine", context_extractor="new",
                                eos_token_ids=eos_token_ids,
                            )
                        return result, state["compression"], policy.summary()

                    with torch.inference_mode():
                        result, compression, depth_stats = execute_shared_prefix(generate, packed, prefix, length)
                        torch.cuda.synchronize()
                    metrics = continuation_metrics(result.token_ids, result.rounds)
                    record.update(
                        status="ok",
                        stop_reason=result.stop_reason,
                        natural_eos=result.stop_reason == "eos",
                        hit_budget=result.stop_reason == "limit",
                        rounds=_ready(result.rounds),
                        compression=_ready(compression),
                        depth=depth_stats.get("mode"),
                        cuda_synchronized=True,
                        wall_s=time.perf_counter() - started,
                        **metrics,
                    )
                    status["ok_jobs"] += 1
                except Exception as exc:
                    record.update(error=f"{type(exc).__name__}: {exc}", wall_s=time.perf_counter() - started)
                    status["failed_jobs"] += 1
                finally:
                    hardware = sampler.stop()
                    foreign = hardware.get("foreign_processes") or []
                    record["hardware"] = {
                        "foreign_processes": foreign,
                        "duration_s": hardware.get("measurement_duration_s"),
                        "sharing": hardware.get("sharing"),
                        "energy_attribution": hardware.get("energy_attribution"),
                        "process_visibility_complete": hardware.get("process_visibility_complete"),
                    }
                stored.append(record)
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
                save_json(out / "status.json", status)
                print(json.dumps({
                    "request_id": request_id, "arm": arm, "prefix_tokens": length,
                    "status": record["status"], "non_eos": record.get("non_eos_continuation_tokens"),
                    "effective_per_verify": record.get("effective_draft_per_verify"),
                    "error": record.get("error"),
                }), flush=True)
    (out / "continuation_groups.json").write_text(json.dumps(group_free_continuations(stored), indent=2) + "\n")
    status["status"] = "finished_with_failures" if status["failed_jobs"] else "finished"
    status["finished_at"] = timestamp()
    save_json(out / "status.json", status)


if __name__ == "__main__":
    main()
