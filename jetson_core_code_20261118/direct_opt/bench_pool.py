"""Paired native vs existing reorder, three process directories, profiler off.

Does not overwrite profile_gpu_round1 or profile_gpu_round2.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "training_free"), str(ROOT / "training_free_opt"), str(ROOT / "direct_opt"), str(ROOT / "vendor")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
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
    from compression import FrozenQueryAdaptor
    from compression_graph import CudaGraphAdaptor
    from structured_pool import StructuredPoolAdaptor
    from run_gpu_profile import error_metrics, timed_pairs

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

    with cooperative_lock(args.physical_gpu, out), torch.inference_mode():
        metadata = torch.load(Path(args.features) / "adaptor.pt", map_location="cuda", weights_only=True)
        adaptor = Adaptor(metadata).eval()
        torch.cuda.synchronize()
        old = FactorizedAdaptor(adaptor, precision="model")
        eager = FrozenQueryAdaptor(adaptor)
        graph = CudaGraphAdaptor(eager)
        candidates = {"bmm": eager, "graph": graph}
        payload = {
            "process_index": args.process_index,
            "physical_gpu": args.physical_gpu,
            "profiler": "off",
            "gpu_passed": False,
            "comparison": 'full-token graph vs complete spatial-pool-plus-adaptor graph',
            "errors": {},
            "timing": {},
        }
        feature_dir = Path(args.features)
        names = sorted(p.stem for p in feature_dir.glob("*.pt") if p.name != "adaptor.pt")
        sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.05, process_interval_s=0.2)
        sampler.start()
        for name in names:
            loaded = torch.load(feature_dir / f"{name}.pt", map_location="cuda", weights_only=True)
            x = loaded["hidden_states"] if isinstance(loaded, dict) and "hidden_states" in loaded else loaded["x"] if isinstance(loaded, dict) else loaded
            raw = loaded['grid_thw']
            if hasattr(raw, 'tolist'): raw = raw.tolist()
            if isinstance(raw[0], (tuple,list)): raw = raw[0]
            grid = (int(raw[0]), int(raw[1])//2, int(raw[2])//2)
            assert grid[0]*grid[1]*grid[2] == x.shape[1]
            graph.prepare_for(x)
            reference = graph(x)
            payload['errors'][name] = {'n':int(x.shape[1]),'grid':grid}
            payload['timing'][name + '_full_AA'] = timed_pairs(lambda: graph(x), lambda: graph(x), args.warmup,args.repeats,x.device)
            for block in [(1,2),(2,2)]:
                pooled = StructuredPoolAdaptor(eager, grid, block_hw=block)
                metadata = pooled.prepare_for(x)
                wrapped = CudaGraphAdaptor(pooled,max_entries=1)
                setup = wrapped.prepare_for(x)
                y = wrapped(x)
                assert torch.equal(y, pooled(x))
                key = name + '_pool' + str(block[0]) + str(block[1])
                payload['errors'][name][str(block)] = error_metrics(y,reference)
                payload.setdefault('pool_metadata',{})[key] = dict(pool=metadata,graph=setup)
                payload['timing'][key] = timed_pairs(lambda:graph(x),lambda:wrapped(x),args.warmup,args.repeats,x.device)
                print(json.dumps({'feature':name,'block':block,'keep_ratio':metadata['actual_keep_ratio'],'save_pct':payload['timing'][key]['paired_save_median_pct'],'rel_l2':payload['errors'][name][str(block)]['relative_l2']}),flush=True)
                wrapped.clear()
        hardware = sampler.stop()
        foreign = hardware.get("foreign_processes") or []
        noted = []
        for item in foreign:
            comm = ""
            try:
                comm = Path(f"/proc/{item['pid']}/comm").read_text().strip()
            except OSError:
                comm = ""
            noted.append({**item, "comm": comm})
        payload["hardware"] = {
            "foreign_processes": noted,
            "process_visibility_complete": hardware.get("process_visibility_complete"),
            "duration_s": hardware.get("measurement_duration_s"),
            "graphics_only": bool(noted) and all(row.get("comm") == "Xorg" for row in noted),
        }
        slim = {k: v for k, v in payload.items() if k != "timing"}
        slim["timing_summary"] = {
            key: {k: val[k] for k in val if k not in {"rows", "order"}}
            for key, val in payload["timing"].items()
        }
        (out / "pool.json").write_text(json.dumps(slim, indent=2) + "\n")
        (out / "timing_raw.json").write_text(json.dumps(payload["timing"]) + "\n")
        print(json.dumps({"process": args.process_index, "graphics_only_foreign": payload["hardware"]["graphics_only"], "keys": list(slim["timing_summary"])}))


if __name__ == "__main__":
    main()
