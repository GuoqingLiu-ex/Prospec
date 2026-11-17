"""Paired wall/CUDA benchmarks on captured real adaptor inputs and weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--physical-gpu', type=int, default=2)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--difference-threshold', type=float)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
    import torch
    from torch import nn
    from torch.nn import functional as F
    from run_experiment import cooperative_lock
    from scheme_context import SchemeRuntime, native_with_sizes
    from compressors import compress_tokens
    from hardware_metrics import HardwareSampler

    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    class Adaptor(nn.Module):
        def __init__(self, metadata):
            super().__init__()
            self.hidden_size = metadata['hidden_size']
            self.num_heads = metadata['num_heads']
            self.num_q = metadata['num_q']
            self.head_dim = self.hidden_size // self.num_heads
            d = self.hidden_size
            state = metadata['state_dict']
            self.q = nn.Parameter(torch.empty(self.num_q, self.num_heads, self.head_dim))
            self.k_proj = nn.Linear(d, d, bias='k_proj.bias' in state)
            self.v_proj = nn.Linear(d, d, bias='v_proj.bias' in state)
            self.o_proj = nn.Linear(d, d, bias='o_proj.bias' in state)
            self.to(device='cuda', dtype=state['q'].dtype)
            self.load_state_dict(state)

        def forward(self, x):
            b, n, d = x.shape
            q = self.q.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)
            k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
            return self.o_proj(y.transpose(1, 2).reshape(b, self.num_q, d))

    def error_metrics(y, reference):
        candidate, expected = y.double(), reference.double()
        diff = candidate - expected
        return dict(max_abs=diff.abs().max().item(),
                    rmse=diff.square().mean().sqrt().item(),
                    relative_l2=(diff.norm()/expected.norm().clamp_min(1e-12)).item(),
                    cosine=F.cosine_similarity(candidate.flatten(), expected.flatten(), dim=0).item())

    with cooperative_lock(args.physical_gpu, out) as lock, torch.inference_mode():
        metadata = torch.load(Path(args.features) / 'adaptor.pt', map_location='cuda', weights_only=True)
        adaptor = Adaptor(metadata).eval()
        draft = type('Draft', (), {'imadpt': adaptor})()
        torch.cuda.synchronize()
        setup_start = time.perf_counter()
        runtime = SchemeRuntime(draft)
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - setup_start) * 1000
        factor_preparation = {
            precision: dict(prepare_time_ms=factor.prepare_time_ms,
                            cache_bytes=factor.cache_bytes,
                            prepare_count=factor.prepare_count,
                            compute_dtype=str(factor.compute_dtype))
            for precision, factor in runtime.factorized.items()
        }
        sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.1,
                                  process_interval_s=0.5, known_shared=False)
        sampler.start()
        methods = ['baseline', 'reorder', 'reorder_fp32', 'attention50', 'difference50',
                   'response50', 'reorder_attention50', 'reorder_difference50', 'reorder_response50']
        if args.difference_threshold is not None:
            methods.extend(['difference_gate', 'reorder_difference_gate'])
        records = []
        errors = []
        rng = random.Random(20261001)
        for path in sorted(Path(args.features).glob('*.pt')):
            if path.name == 'adaptor.pt':
                continue
            case = torch.load(path, map_location='cuda', weights_only=True)
            x = case['x']
            t, h, w = case['grid_thw'][0]
            grid = (t, h//2, w//2)
            reference = adaptor(x)

            def compressed_input(name):
                method = name.replace('reorder_', '').replace('50', '')
                scores = runtime.scores(x[0]) if method in ('attention', 'response') else None
                saliency = scores.softmax(dim=0).amax(dim=1) if method == 'attention' else None
                extra = {'difference_threshold': args.difference_threshold} if method == 'difference_gate' else {}
                z, sizes, _ = compress_tokens(x[0], grid, 0.5, method=method,
                                              saliency=saliency, scores=scores, **extra)
                return z.unsqueeze(0), sizes

            def execute(name):
                if name == 'baseline':
                    return adaptor(x), x.shape[1]
                if name == 'reorder':
                    return runtime.factorized['model'](x), x.shape[1]
                if name == 'reorder_fp32':
                    return runtime.factorized['float32'](x), x.shape[1]
                z, sizes = compressed_input(name)
                y = (runtime.factorized['model'](z, group_sizes=sizes)
                     if name.startswith('reorder_') else native_with_sizes(adaptor, z, sizes))
                return y, z.shape[1]

            for name in methods:
                comparisons = {}
                if name.startswith('reorder_') and name != 'reorder_fp32':
                    # Diagnostics only: both paths receive these exact tensors.
                    # Keep this extra native evaluation outside measured execute().
                    z, sizes = compressed_input(name)
                    y = runtime.factorized['model'](z, group_sizes=sizes)
                    compressed_reference = native_with_sizes(adaptor, z, sizes)
                    count = z.shape[1]
                    comparisons['reorder_error'] = dict(
                        reference='native_same_compressed_input_and_group_sizes',
                        **error_metrics(y, compressed_reference))
                    comparisons['compression_error'] = dict(
                        reference='native_full_input',
                        **error_metrics(compressed_reference, reference))
                else:
                    y, count = execute(name)
                errors.append(dict(request_id=case['request_id'], method=name, n=x.shape[1], kept=count,
                    error_reference='native_full_input', **error_metrics(y, reference), **comparisons))
            for _ in range(args.warmup):
                for name in methods:
                    execute(name)
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                order = list(methods)
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start = time.perf_counter()
                    a.record()
                    y, count = execute(name)
                    b.record()
                    torch.cuda.synchronize()
                    wall_ms = (time.perf_counter()-start)*1000
                    records.append(dict(request_id=case['request_id'], method=name, repeat=repeat,
                                        n=x.shape[1], kept=count, wall_ms=wall_ms, cuda_ms=a.elapsed_time(b)))
            print(json.dumps({'completed': case['request_id'], 'shape': list(x.shape)}), flush=True)
        hardware = sampler.stop()
        (out / 'raw.json').write_text(json.dumps(records, indent=2)+'\n')
        (out / 'errors.json').write_text(json.dumps(errors, indent=2)+'\n')
        summary = []
        for error in errors:
            subset = [r for r in records if r['request_id']==error['request_id'] and r['method']==error['method']]
            summary.append(dict(error, median_wall_ms=statistics.median(r['wall_ms'] for r in subset),
                                median_cuda_ms=statistics.median(r['cuda_ms'] for r in subset)))
        (out / 'summary.json').write_text(json.dumps(dict(args=vars(args), setup_ms=setup_ms,
            factor_preparation=factor_preparation,
            setup_scope='Both factor precision caches and compression score directions; outside per-call timing',
            lock=lock, hardware=hardware, rows=summary), indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
