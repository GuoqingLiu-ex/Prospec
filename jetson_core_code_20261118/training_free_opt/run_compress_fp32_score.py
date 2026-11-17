"""Attention/response cost with the SchemeRuntime FP32 scorer.

The earlier profile_gpu_compress attention/response rows used FactorizedAdaptor.score_tokens
at model precision. Those rows stay the FP16 candidate. Difference does not use this scorer.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_experiment import cooperative_lock
    from hardware_metrics import HardwareSampler
    from factorized_adaptor import FactorizedAdaptor
    from compressors import compress_tokens
    from run_gpu_profile import error_metrics, timed_pairs
    from visual_grid import assert_adaptor_tokens
    from scheme_scores import scheme_runtime_directions, scheme_runtime_scores
    from run_compress_cost import _load_merge_size

    merge_size = _load_merge_size(Path(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    class Adaptor(nn.Module):
        def __init__(self, metadata):
            super().__init__()
            self.hidden_size = metadata["hidden_size"]
            self.num_heads = metadata["num_heads"]
            self.num_q = metadata["num_q"]
            self.head_dim = self.hidden_size // self.num_heads
            d = self.hidden_size
            state = metadata["state_dict"]
            self.q = nn.Parameter(torch.empty(self.num_q, self.num_heads, self.head_dim))
            self.k_proj = nn.Linear(d, d, bias="k_proj.bias" in state)
            self.v_proj = nn.Linear(d, d, bias="v_proj.bias" in state)
            self.o_proj = nn.Linear(d, d, bias="o_proj.bias" in state)
            self.to(device="cuda", dtype=state["q"].dtype)
            self.load_state_dict(state)

        def forward(self, x):
            b, n, d = x.shape
            q = self.q.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)
            k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
            return self.o_proj(y.transpose(1, 2).reshape(b, self.num_q, d))

    def retained(tokens, grid, method, scores):
        saliency = scores.softmax(dim=0).amax(dim=1) if method == "attention" else None
        _compressed, sizes, summary = compress_tokens(
            tokens, grid, 0.5, method=method, saliency=saliency, scores=scores, return_donor_map=True,
        )
        index = summary.get("retained_token_indices")
        if index is None:
            index = summary.get("retained_cell_ids")
        return sizes.detach().cpu(), None if index is None else index.detach().cpu()

    with cooperative_lock(args.physical_gpu, out), torch.inference_mode():
        metadata = torch.load(Path(args.features) / "adaptor.pt", map_location="cuda", weights_only=True)
        adaptor = Adaptor(metadata).eval()
        torch.cuda.synchronize()
        prep0 = time.perf_counter()
        directions, score_bias = scheme_runtime_directions(adaptor)
        torch.cuda.synchronize()
        prep_ms = (time.perf_counter() - prep0) * 1000
        factor = FactorizedAdaptor(adaptor, precision="model")
        payload = {
            "process_index": args.process_index,
            "score_candidate": "scheme_runtime_fp32",
            "not_the_fp16_score_tokens_candidate": True,
            "direction_prep_ms": prep_ms,
            "direction_prep_note": "once per process, matching SchemeRuntime construction; not inside the paired loop",
            "gpu_passed": False,
            "cases": {},
            "timing": {},
        }
        names = sorted(p.stem for p in Path(args.features).glob("*.pt") if p.name != "adaptor.pt")
        sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.05, process_interval_s=0.2)
        sampler.start()
        for name in names:
            loaded = torch.load(Path(args.features) / f"{name}.pt", map_location="cuda", weights_only=True)
            x = loaded["x"] if "x" in loaded else loaded["hidden_states"]
            grid = assert_adaptor_tokens(int(x.shape[1]), loaded["grid_thw"], merge_size)
            tokens = x[0]
            reference = factor(x)
            case = {"n": int(x.shape[1]), "merged_grid": list(grid), "methods": {}}
            fp16_scores = factor.score_tokens(tokens)
            fp32_scores = scheme_runtime_scores(directions, score_bias, tokens)
            for method in ("attention", "response"):
                fp16_sizes, fp16_index = retained(tokens, grid, method, fp16_scores)
                fp32_sizes, fp32_index = retained(tokens, grid, method, fp32_scores)
                same_sizes = torch.equal(fp16_sizes, fp32_sizes)
                same_index = fp16_index is not None and fp32_index is not None and torch.equal(fp16_index, fp32_index)

                def path(method=method):
                    scores = scheme_runtime_scores(directions, score_bias, tokens)
                    saliency = scores.softmax(dim=0).amax(dim=1) if method == "attention" else None
                    compressed, sizes, _summary = compress_tokens(
                        tokens, grid, 0.5, method=method, saliency=saliency, scores=scores,
                    )
                    return factor(compressed.unsqueeze(0), group_sizes=sizes)

                y = path()
                case["methods"][method] = {
                    "groups_match_fp16_score_tokens": bool(same_sizes and same_index),
                    "size_match": bool(same_sizes),
                    "index_match": bool(same_index),
                    "relative_l2_vs_full_reorder": error_metrics(y, reference)["relative_l2"],
                }
                payload["timing"][f"{name}_{method}"] = timed_pairs(lambda xx=x: factor(xx), path, args.warmup, args.repeats, x.device)
            payload["cases"][name] = case
        hardware = sampler.stop()
        foreign = []
        for item in hardware.get("foreign_processes") or []:
            try:
                comm = Path(f"/proc/{item['pid']}/comm").read_text().strip()
            except OSError:
                comm = ""
            foreign.append({**item, "comm": comm})
        payload["hardware"] = {"foreign_processes": foreign, "graphics_only": bool(foreign) and all(row.get("comm") == "Xorg" for row in foreign)}
        slim = {k: v for k, v in payload.items() if k != "timing"}
        slim["timing_summary"] = {
            key: {k: val for k, val in value.items() if k not in {"rows", "order"}}
            for key, value in payload["timing"].items()
        }
        (out / "compress_fp32_score.json").write_text(json.dumps(slim, indent=2) + "\n")
        (out / "timing_raw.json").write_text(json.dumps(payload["timing"]) + "\n")
        print(json.dumps({
            "process": args.process_index,
            "groups": {name: {m: row["groups_match_fp16_score_tokens"] for m, row in case["methods"].items()} for name, case in payload["cases"].items()},
        }))


if __name__ == "__main__":
    main()
