"""Same-device native/stride0/grouped math replay on real Target inputs.

Ordinary timing and allocator measurements include each implementation's host
overhead, with diagnostics/profiling outside those measurements. This is a
module experiment, not full-generation throughput, SRAM or DRAM evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from probe_gqa_attention import restore_tensor, native_math, metrics
from probe_stride0_gqa import capture_entries, validate_capture, precision_scope, native_gate, profile_trace

ARMS = ('native', 'stride0', 'grouped')


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, default=0)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=12)
    a = p.parse_args(argv)
    if a.physical_gpu < 0 or a.warmup < 1 or a.repeats < 6 or a.repeats % 6:
        p.error('positive warmup and repeats divisible by six required for balanced orders')
    return a


def orders(repeats, sample_index):
    choices = list(itertools.permutations(ARMS))
    return [choices[(i + sample_index) % len(choices)] for i in range(repeats)]


def measure(calls, reference, torch, warmup, repeats, sample_index):
    for _ in range(warmup):
        for name in ARMS:
            value = calls[name]()
            del value
    torch.cuda.synchronize()
    records = []
    for repeat, order in enumerate(orders(repeats, sample_index)):
        row = dict(repeat=repeat, order=list(order))
        for name in order:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            value = calls[name]()
            torch.cuda.synchronize()
            item = dict(wall_ms=(time.perf_counter() - start) * 1000,
                baseline_allocated_bytes=before,
                incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-before,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            item['error'] = metrics(value, reference, torch)
            del value
            if not item['error']['finite']:
                raise RuntimeError(name + ' returned nonfinite output')
            if name == 'native' and not item['error']['bitwise_equal']:
                raise RuntimeError('native byte-exact replay changed during measurement')
            row[name] = item
        records.append(row)
    return dict(records=records,
        median_wall_ms={name:statistics.median(r[name]['wall_ms'] for r in records) for name in ARMS},
        median_incremental_peak_bytes={name:statistics.median(r[name]['incremental_peak_allocated_bytes']
            for r in records) for name in ARMS},
        scope='Synchronized module wall and incremental PyTorch allocated peak; common inputs/reference resident; '
              'no debug/profiler/metrics inside measured boundaries; all implementation host overhead included')


def replay(entry, args, torch, sample_index, sampler, row):
    from stride0_gqa_math import stride0_gqa_math
    from grouped_gqa_math import grouped_gqa_math
    path = args.capture_dir / entry['file']
    saved = torch.load(path, map_location='cpu', weights_only=True)
    meta, tensors = validate_capture(saved, entry)
    data = {key:restore_tensor(tensors[key], meta['tensors'][key], 'cuda:0', torch)
            for key in ('query', 'key', 'value', 'mask')}
    q, k, v, mask = (data[key] for key in ('query', 'key', 'value', 'mask'))
    reference = tensors['native_output'].to('cuda:0')
    scale = meta['scale']
    calls = dict(native=lambda:native_math(q,k,v,mask,scale,torch),
        stride0=lambda:stride0_gqa_math(q,k,v,mask,scale,debug_intermediates=False)[0],
        grouped=lambda:grouped_gqa_math(q,k,v,mask,scale,record_receipt=False,debug_intermediates=False)[0])
    row.update(metadata=meta, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               native_replay=native_gate(calls['native'],reference,torch), initial={})
    for name in ('stride0','grouped'):
        value = calls[name]()
        row['initial'][name] = metrics(value,reference,torch)
        del value
        if not row['initial'][name]['finite']:
            raise RuntimeError(name+' is nonfinite before ordinary measurement')
    sampler.mark(entry['file']+':ordinary_begin')
    row['ordinary'] = measure(calls,reference,torch,args.warmup,args.repeats,sample_index)
    sampler.mark(entry['file']+':ordinary_end')
    value,receipt,debug = grouped_gqa_math(q,k,v,mask,scale,record_receipt=True,debug_intermediates=False)
    if debug is not None or receipt is None:
        raise RuntimeError('grouped diagnostic receipt contract differs')
    row['grouped_diagnostic'] = dict(receipt=receipt,error=metrics(value,reference,torch))
    del value,debug,receipt
    row['profiles'] = {name:profile_trace(call,torch,args.out/(path.stem+'_'+name+'_trace.json'))
                       for name,call in calls.items()}
    # This check is only a measured local filter. Full model quality and real
    # DRAM remain separate even if every capture passes it.
    rows = row['ordinary']['records']
    row['local_filter'] = dict(
        common_allocated_baseline=all(len({x[name]['baseline_allocated_bytes'] for name in ARMS})==1 for x in rows),
        grouped_all_output_bytes_equal=all(x['grouped']['error']['bitwise_equal'] for x in rows),
        grouped_peak_never_above_stride0=all(x['grouped']['incremental_peak_allocated_bytes']
            <= x['stride0']['incremental_peak_allocated_bytes'] for x in rows),
        grouped_median_faster_than_stride0=row['ordinary']['median_wall_ms']['grouped']
            < row['ordinary']['median_wall_ms']['stride0'])


def main(argv=None):
    args = parse_args(argv)
    source = capture_entries(args.capture_dir)
    args.out.mkdir(parents=True,exist_ok=False)
    report = dict(status='starting',pid=os.getpid(),args={k:str(v) if isinstance(v,Path) else v
        for k,v in vars(args).items()},samples=[],failures=[],arms=list(ARMS),actual_dram_measured=False,
        end_to_end_claim=False,actual_sram_residency=False,
        source_hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ('bench_grouped_gqa.py','grouped_gqa_math.py','stride0_gqa_math.py','probe_stride0_gqa.py',
             'probe_gqa_attention.py','run_joint.py','../hardware_metrics.py')})
    save(args.out/'summary.json',report)
    old_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        from run_joint import cooperative_lock
        with cooperative_lock(args.physical_gpu,args.out) as lock:
            import torch
            if torch.cuda.is_initialized():
                raise RuntimeError('fresh process required before CUDA initialization')
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
            from hardware_metrics import HardwareSampler
            with precision_scope(torch),torch.inference_mode():
                torch.cuda.set_device(0)
                report.update(lock=lock,device=torch.cuda.get_device_name(0),torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    reduced_precision_math_sdp=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
                sampler = HardwareSampler(device_index=args.physical_gpu,sample_interval_s=.1,process_interval_s=.2)
                sampler.start()
                try:
                    report['status'] = 'running'
                    for index,entry in enumerate(source['captures']):
                        row=dict(file=entry['file'],status='running');report['samples'].append(row)
                        save(args.out/'summary.json',report)
                        replay(entry,args,torch,index,sampler,row)
                        row['status']='completed';save(args.out/'summary.json',report)
                    report['status']='completed'
                finally:
                    report['hardware']=sampler.stop()
    except BaseException as exc:
        report['status']='failed'
        report['failures'].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if old_visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=old_visible
        save(args.out/'summary.json',report)
    return report


if __name__=='__main__':main()
