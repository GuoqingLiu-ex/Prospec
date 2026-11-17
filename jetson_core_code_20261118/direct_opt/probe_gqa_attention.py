"""Real Target tree-attention capture and strict backend replay probe.

Capture (diagnostic full request; no performance claims):
  python probe_gqa_attention.py capture --entry active --capture-dir CAP \
      --capture-rounds 0 80 163 -- [normal run_active_ring_layout arguments]
Replay (same GPU cooperative lock as request runners):
  python probe_gqa_attention.py probe --capture-dir CAP --out OUT \
      --physical-gpu 7 --backend cudnn --repeats 20
Run the flex backend explicitly in another OUT only if requested; no fallback.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'vendor'),str(ROOT/'training_free')]


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


@contextmanager
def patch(obj,name,value):
    old=getattr(obj,name);setattr(obj,name,value)
    try:yield
    finally:setattr(obj,name,old)


def capture_main(argv):
    parser=argparse.ArgumentParser(description='Diagnostic real-input capture',allow_abbrev=False)
    parser.add_argument('--entry',choices=('active','fixed'),default='active')
    parser.add_argument('--capture-dir',type=Path,required=True)
    parser.add_argument('--capture-layer',type=int,default=0)
    parser.add_argument('--capture-rounds',type=int,nargs='+',default=[0])
    args,rest=parser.parse_known_args(argv)
    if rest[:1]==['--']:rest=rest[1:]
    if args.capture_layer<0 or any(x<0 for x in args.capture_rounds):parser.error('nonnegative layer/round required')
    import run_joint
    import hf517_vispec_adapter as bridge
    import run_active_ring_layout,run_ring_layout
    from gqa_attention_runtime import AttentionCapture,gqa_attention_policy
    capture=AttentionCapture(args.capture_dir,layer=args.capture_layer,rounds=args.capture_rounds,
                             limit=len(set(args.capture_rounds)))
    original_load,original_save=bridge.load_spec,run_joint.save_json
    policies=[]
    written_out=[]
    with ExitStack() as stack:
        def load(target,*a,**kw):
            spec=original_load(target,*a,**kw)
            policies.append(stack.enter_context(gqa_attention_policy(target,'capture',capture=capture)))
            return spec
        def write(path,value):
            if Path(path).name=='status.json':
                written_out[:]=[Path(path).parent]
                value=dict(value,attention_capture_diagnostic=True,attention_capture_dir=str(args.capture_dir),
                           profiled_wall_is_performance_evidence=False,
                           attention_capture_invalidates_timing_and_memory=True)
            return original_save(path,value)
        stack.enter_context(patch(bridge,'load_spec',load))
        stack.enter_context(patch(run_joint,'save_json',write))
        try:
            (run_active_ring_layout if args.entry=='active' else run_ring_layout).main(rest)
        finally:
            if written_out:mark_capture_rows(written_out[0]/'results.jsonl')
            save(args.capture_dir/'policy.json',dict(policies=policies,captures=len(capture.records),
                 diagnostic_only=True,quality_or_speed_claim=False))
    if not capture.records:raise RuntimeError('no requested real Target tree attention call was captured')


def mark_capture_rows(path):
    """Only this diagnostic run's closed results log is atomically annotated."""
    path=Path(path)
    if not path.exists():return
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row.update(attention_capture_diagnostic=True,capture_enabled=True,valid_timing=False,
                   memory_measurement_valid=False,energy_measurement_valid=False,
                   memory_scope='Diagnostic attention CPU capture; peaks are not deployment evidence',
                   timing_note='Diagnostic attention CPU transfers/saves invalidate timing and energy; do not aggregate as performance')
    temporary=path.with_suffix(path.suffix+'.diagnostic.tmp')
    temporary.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    temporary.replace(path)


def restore_tensor(tensor,description,device,torch):
    """Restore original strides, including unused gaps in an active KV view."""
    if list(tensor.shape)!=description['shape'] or str(tensor.dtype)!=description['dtype']:
        raise ValueError('captured tensor metadata mismatch')
    shape,stride=tuple(description['shape']),tuple(description['stride'])
    if len(shape)!=len(stride) or any(s<0 for s in stride):raise ValueError('invalid captured stride')
    out=torch.empty_strided(shape,stride,dtype=tensor.dtype,device=device)
    out.copy_(tensor)
    return out


def native_math(q,k,v,mask,scale,torch):
    from torch.nn.attention import sdpa_kernel,SDPBackend
    groups=q.shape[1]//k.shape[1]
    def repeat(x):
        b,h,n,d=x.shape
        return x[:,:,None,:,:].expand(b,h,groups,n,d).reshape(b,h*groups,n,d)
    with sdpa_kernel(SDPBackend.MATH):
        result=torch.nn.functional.scaled_dot_product_attention(q,repeat(k),repeat(v),
            attn_mask=mask,dropout_p=0.0,is_causal=False,scale=scale)
    return result.transpose(1,2).contiguous()


def metrics(value,reference,torch):
    if value.shape!=reference.shape:raise ValueError('attention output shape mismatch')
    x,y=value.double(),reference.double()
    finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    equal=(value.dtype==reference.dtype and torch.equal(value.contiguous().view(torch.uint8),reference.contiguous().view(torch.uint8)))
    return dict(finite=finite,bitwise_equal=equal,
                relative_l2=float((x-y).norm()/y.norm().clamp_min(1e-30)) if finite else None,
                max_abs=float((x-y).abs().max()) if finite else None)


def probe_main(argv):
    parser=argparse.ArgumentParser(description='Strict real-input GQA replay probe')
    parser.add_argument('--capture-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    parser.add_argument('--backend',choices=('cudnn','flex'),required=True)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--warmup',type=int,default=3)
    args=parser.parse_args(argv)
    if args.repeats<1 or args.warmup<0:parser.error('invalid repeat/warmup count')
    args.out.mkdir(parents=True,exist_ok=False)
    import run_joint
    report=dict(status='starting',backend=args.backend,diagnostic_only=True,quality_or_e2e_speed_claim=False,
                repeats=args.repeats,warmup=args.warmup,samples=[],failures=[],fallback_allowed=False)
    save(args.out/'summary.json',report)
    try:
        with run_joint.cooperative_lock(args.physical_gpu,args.out) as lock:
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu)
            import torch
            from gqa_attention_runtime import GQAAttentionBackend,validate_attention
            torch.set_num_threads(1)
            summary=json.loads((args.capture_dir/'summary.json').read_text())
            if summary.get('schema')!='real_tree_attention_v1':raise ValueError('unrecognized capture schema')
            report.update(lock=lock,torch=torch.__version__,cudnn=torch.backends.cudnn.version(),
                          device=torch.cuda.get_device_name(0),shared_gpu=True,
                          timing_scope='Paired synchronized attention invocation including checks/packing; excludes capture/model/IO; exploratory only')
            def observation():
                try:
                    import gpu_lock
                    snapshot,processes=gpu_lock._gpu_status(args.physical_gpu)
                    return dict(snapshot=snapshot,processes=processes)
                except Exception as exc:return dict(error=str(exc))
            report['gpu_before']=observation()
            executor=GQAAttentionBackend(args.backend)
            with torch.inference_mode():
                for entry in summary['captures']:
                    path=args.capture_dir/entry['file']
                    saved=torch.load(path,map_location='cpu',weights_only=True)
                    meta,tensors=saved['metadata'],saved['tensors']
                    data={name:restore_tensor(tensors[name],meta['tensors'][name],'cuda:0',torch)
                          for name in ('query','key','value','mask')}
                    q,k,v,m=(data[n] for n in ('query','key','value','mask'))
                    scale=meta['scale'];validate_attention(q,k,v,m)
                    reference=tensors['native_output'].to('cuda:0')
                    row=dict(file=entry['file'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             input=meta['tensors'],source_stride_restored=True,results=[])
                    report['samples'].append(row)
                    actual=native_math(q,k,v,m,scale,torch)
                    row['native_replay']=metrics(actual,reference,torch)
                    if not row['native_replay']['bitwise_equal']:
                        raise RuntimeError('native math replay differs from saved output; do not benchmark an unverified reference')
                    def candidate():return executor(q,k,v,m,scale).transpose(1,2).contiguous()
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    before=torch.cuda.memory_allocated();start=time.perf_counter()
                    proposed=candidate();torch.cuda.synchronize()
                    row['cold_candidate']=dict(synchronized_wall_ms=(time.perf_counter()-start)*1000,
                        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-before,
                        error=metrics(proposed,reference,torch))
                    if not row['cold_candidate']['error']['finite']:
                        raise RuntimeError('candidate returned nonfinite output; performance probe stopped')
                    for _ in range(args.warmup):native_math(q,k,v,m,scale,torch);candidate()
                    torch.cuda.synchronize()
                    rng=random.Random(20261002)
                    for repeat in range(args.repeats):
                        order=['native','candidate'];rng.shuffle(order);pair=dict(repeat=repeat,order=order)
                        for name in order:
                            torch.cuda.synchronize();begin=time.perf_counter()
                            output=native_math(q,k,v,m,scale,torch) if name=='native' else candidate()
                            torch.cuda.synchronize();pair[name+'_wall_ms']=(time.perf_counter()-begin)*1000
                            del output
                        row['results'].append(pair)
                    # Separate diagnostic trace, never part of paired timing.
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as trace:
                        candidate();torch.cuda.synchronize()
                    trace.export_chrome_trace(str(args.out/(path.stem+'_trace.json')))
                    names=sorted({event.key for event in trace.key_averages()})
                    row['observed_operator_names']=names
                    expected='cudnn_attention' if args.backend=='cudnn' else 'flex_attention'
                    row['expected_backend_operator_observed']=any(expected in name for name in names)
                    row['trace_note']='Check CUDA kernel names in trace; no fallback is enabled. Unrecognized trace names do not establish fused execution.'
                    row['paired_median_reduction_percent']=statistics.median(
                        100*(1-r['candidate_wall_ms']/r['native_wall_ms']) for r in row['results'])
                    save(args.out/'summary.json',report)
            report['executor']=executor.metadata();report['gpu_after']=observation();report['status']='completed'
    except Exception as exc:
        report['status']='failed';report['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        if 'executor' in locals():report['executor']=executor.metadata()
        raise
    finally:save(args.out/'summary.json',report)


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ('capture','probe'):
        raise SystemExit('choose capture or probe; see module usage')
    (capture_main if args[0]=='capture' else probe_main)(args[1:])


if __name__=='__main__':main()
