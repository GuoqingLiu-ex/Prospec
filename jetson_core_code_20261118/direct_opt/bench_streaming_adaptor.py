"""Captured-feature numerical/workspace pilot for hardware-oriented streaming.

This Torch reference does not place its tensors in SRAM. Ordinary module wall
time and CUDA allocated peaks are measured separately from logical workspace.
It is not a full Target+Draft quality test or an actual DRAM measurement.
Run only after the current GPU queue is terminal; this entry owns the lock.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / 'vendor')]

CONFIGS = ((7, 12), (14, 24), (28, 48))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--features', required=True)
    p.add_argument('--native-source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--physical-gpu', type=int, default=7)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args(argv)
    if args.warmup < 1 or args.repeats < 2 or args.physical_gpu < 0:
        p.error('warmup >=1, repeats >=2, and nonnegative GPU required')
    return args


def tensor_error(candidate, reference):
    # Validation copies are outside all latency and peak measurement intervals.
    import torch
    raw_a, raw_b = candidate.detach().cpu(), reference.detach().cpu()
    a, b = raw_a.double(), raw_b.double()
    if a.shape != b.shape:
        raise ValueError('output shape mismatch')
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    d = a - b
    equal_bits = (raw_a.dtype == raw_b.dtype and
                  torch.equal(raw_a.contiguous().view(torch.uint8), raw_b.contiguous().view(torch.uint8)))
    return dict(finite=finite, bitwise_equal=bool(equal_bits),
                max_abs=float(d.abs().max()), rmse=float(d.square().mean().sqrt()),
                relative_l2=float(d.norm() / b.norm().clamp_min(1e-12)))


def dense_accumulation_reference(compiled, x):
    """Same folded parameters, FP32 scores/pooling; model-dtype V/O as candidate."""
    import torch
    from torch.nn import functional as F
    b, _, d = x.shape
    work = x.float()
    scores = F.linear(work, compiled.directions.float(),
                      None if compiled.score_bias is None else compiled.score_bias.float())
    pooled = torch.bmm(scores.transpose(1, 2).softmax(-1), work).to(x.dtype)
    per_head = (pooled.reshape(b, compiled.heads, compiled.q_count, d)
                .permute(1, 0, 2, 3).reshape(compiled.heads, b * compiled.q_count, d))
    values = torch.bmm(per_head, compiled.value_heads.transpose(1, 2))
    if compiled.value_bias is not None:
        values = values + compiled.value_bias
    values = values.reshape(compiled.heads, b, compiled.q_count, compiled.head_dim)
    values = values.permute(1, 2, 0, 3).reshape(b, compiled.q_count, -1)
    return F.linear(values, compiled.output_weight, compiled.output_bias)


def measured_calls(module, x, torch, repeats):
    rows = []
    for repeat in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        start = time.perf_counter()
        y = module(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append(dict(repeat=repeat, wall_ms=elapsed * 1000,
                         baseline_allocated_bytes=base,
                         incremental_peak_bytes=torch.cuda.max_memory_allocated() - base,
                         output_bytes=y.numel() * y.element_size()))
        del y
    return dict(rows=rows, median_wall_ms=statistics.median(r['wall_ms'] for r in rows),
                median_incremental_peak_bytes=statistics.median(r['incremental_peak_bytes'] for r in rows))


def main(argv=None):
    args = parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
    import torch
    from compiled_adaptor import CompiledQueryAdaptor
    from compression import FrozenQueryAdaptor
    from hardware_metrics import HardwareSampler
    from profile_compression_dram import load_original_adaptor
    from run_joint import cooperative_lock, save_json
    from streaming_adaptor import StreamingQueryAdaptor

    folder = Path(args.features)
    captures = sorted(p for p in folder.glob('*.pt') if p.name != 'adaptor.pt')
    if not captures:
        raise ValueError('no captured visual feature tensors')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='starting', pid=os.getpid(), args=vars(args),
                  scope='Captured-feature adaptor only; no full-model quality or DRAM claim',
                  actual_sram_residency=False, timings='profiler off, module wall including host dispatch',
                  timing_diagnostics=dict(check_finite=False, record_workspace=False),
                  diagnostic_workspace='Explicit live Torch tensors; excludes internal operator temporaries',
                  results=[], completed_features=0, expected_features=len(captures))
    save_json(out / 'status.json', status)
    sampler = None
    try:
        torch.set_num_threads(1)
        # This is a numerical experiment: do not silently substitute TF32 in the
        # FP32 score/accumulation reference. Keep the setting with the results.
        torch.backends.cuda.matmul.allow_tf32 = False
        with cooperative_lock(args.physical_gpu, out) as lock, torch.inference_mode():
            sampler = HardwareSampler(device_index=args.physical_gpu,
                                      sample_interval_s=0.1, process_interval_s=0.2)
            sampler.start()
            metadata = torch.load(folder / 'adaptor.pt', map_location='cpu', weights_only=True)
            native, source = load_original_adaptor(args.native_source, metadata, torch)
            native = native.cuda()
            compiled = CompiledQueryAdaptor.from_frozen(FrozenQueryAdaptor(native, backend='bmm'))
            candidates = {
                f'stream_g{g}_b{b}': StreamingQueryAdaptor.from_compiled(
                    compiled, heads_per_group=g, token_tile=b, accumulate_dtype=torch.float32,
                    check_finite=False, record_workspace=False)
                for g, b in CONFIGS
            }
            status.update(status='running', native_source=source, lock=lock,
                          torch_version=torch.__version__, cuda_version=torch.version.cuda,
                          gpu_name=torch.cuda.get_device_name(0), allow_tf32=False,
                          compiled_metadata=compiled.metadata(),
                          candidate_metadata={k:v.metadata() for k,v in candidates.items()})
            save_json(out / 'status.json', status)
            for path in captures:
                payload = torch.load(path, map_location='cpu', weights_only=True)
                x = payload.get('hidden_states', payload.get('x')) if isinstance(payload, dict) else payload
                if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 1:
                    raise ValueError('capture must be [1,N,D]')
                if not bool(torch.isfinite(x).all()):
                    raise ValueError('nonfinite captured input')
                x = x.cuda()
                yn, yc, yd = native(x), compiled(x), dense_accumulation_reference(compiled, x)
                entry = dict(feature=str(path), shape=list(x.shape), dtype=str(x.dtype),
                             compiled_vs_native=tensor_error(yc, yn),
                             dense_fp32_vs_compiled=tensor_error(yd, yc), candidates={})
                for name, module in candidates.items():
                    module.check_finite = True
                    module.record_workspace = True
                    y = module(x)
                    item = dict(vs_native=tensor_error(y, yn), vs_compiled=tensor_error(y, yc),
                                vs_dense_fp32=tensor_error(y, yd),
                                workspace_receipt=module.last_workspace_receipt)
                    module.check_finite = False
                    module.record_workspace = False
                    if not all(item[k]['finite'] for k in ('vs_native','vs_compiled','vs_dense_fp32')):
                        raise RuntimeError('nonfinite streaming output')
                    entry['candidates'][name] = item
                    del y
                del yn, yc, yd
                arms = {'native': native, 'compiled': compiled, **candidates}
                for _ in range(args.warmup):
                    for module in arms.values():
                        module(x)
                torch.cuda.synchronize()
                # Rotate arm order by feature. This is exploratory module timing,
                # not a paired full-request superiority claim.
                order = list(arms)
                shift = status['completed_features'] % len(order)
                order = order[shift:] + order[:shift]
                entry['measurement_order'] = order
                entry['measurements'] = {name:measured_calls(arms[name], x, torch, args.repeats)
                                         for name in order}
                status['results'].append(entry)
                status['completed_features'] += 1
                save_json(out / 'status.json', status)
                print(json.dumps(dict(feature=path.name, completed=status['completed_features'],
                                      module_ms={k:v['median_wall_ms'] for k,v in entry['measurements'].items()})), flush=True)
                del x, payload
            status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if sampler is not None:
            status['hardware'] = sampler.stop()
        save_json(out / 'status.json', status)
    return status


if __name__ == '__main__':
    main()
