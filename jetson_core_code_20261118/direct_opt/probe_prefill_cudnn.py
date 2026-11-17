"""Capture one native Target language prefill, then gate causal cuDNN replay.

capture --capture-dir CAP --capture-layer 0 -- [memory-runner options, exactly
one --request-id, warmup0/repeats1/max-new-tokens1]. Saves native math reference;
the diagnostic request's timing and memory are marked invalid.
probe --capture-dir CAP --out NEW_OUT --physical-gpu 7 [--repeats 5].
No GPU/model work on import; no fallback; no long-answer or full-memory claim.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parent/'vendor'),str(HERE.parent/'training_free')]
from probe_gqa_attention import patch,restore_tensor,metrics,save,mark_capture_rows


def native_prefill_math(q,k,v,scale,torch):
    from torch.nn.attention import sdpa_kernel,SDPBackend
    # Match HF's mask=None path: keep 4 KV heads; MATH itself handles GQA.
    # Explicitly repeating FP16 K/V before this call is a different memory path.
    with sdpa_kernel(SDPBackend.MATH):
        out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None,
            is_causal=True,dropout_p=0.0,scale=scale,enable_gqa=True)
    return out.transpose(1,2).contiguous()


def parse_capture(argv):
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--capture-dir',type=Path,required=True)
    p.add_argument('--capture-layer',type=int,default=0)
    args,rest=p.parse_known_args(argv)
    if rest[:1]==['--']:rest=rest[1:]
    check=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    check.add_argument('--request-id',action='append',required=True)
    check.add_argument('--warmup',type=int,choices=(0,),default=0)
    check.add_argument('--repeats',type=int,choices=(1,),default=1)
    check.add_argument('--max-new-tokens',type=int,choices=(1,),default=1)
    check.add_argument('--memory-layout',choices=('shared_embedding',))
    check.add_argument('--adaptor',choices=('compiled',))
    opts,_=check.parse_known_args(rest)
    if len(opts.request_id)!=1 or not 0<=args.capture_layer<28:p.error('one request and one layer index 0..27 required')
    if any(x.split('=',1)[0]=='--resume' for x in rest):p.error('capture requires a new output directory')
    return args,[*rest,'--warmup','0','--repeats','1','--max-new-tokens','1']


def capture_main(argv):
    args,rest=parse_capture(argv)
    import run_joint
    import run_memory_layout
    import hf517_vispec_adapter as bridge
    from prefill_cudnn_runtime import prefill_cudnn_policy,PrefillAttentionCapture
    capture=PrefillAttentionCapture(args.capture_dir,layers=(args.capture_layer,))
    policies=[];output=[];old_load,old_save=bridge.load_spec,run_joint.save_json
    with ExitStack() as stack:
        def load(target,*a,**kw):
            spec=old_load(target,*a,**kw)
            policies.append(stack.enter_context(prefill_cudnn_policy(target,mode='capture',capture=capture)))
            return spec
        def write(path,value):
            if Path(path).name=='status.json':
                output[:]=[Path(path).parent]
                value=dict(value,prefill_capture_diagnostic=True,attention_capture_invalidates_timing_and_memory=True)
            old_save(path,value)
        stack.enter_context(patch(bridge,'load_spec',load));stack.enter_context(patch(run_joint,'save_json',write))
        try:run_memory_layout.main(['--memory-layout','shared_embedding','--adaptor','compiled',*rest])
        finally:
            if output:mark_capture_rows(output[0]/'results.jsonl')
    save(args.capture_dir/'policy.json',dict(policies=policies,captures=len(capture.records),diagnostic_only=True))
    if len(capture.records)!=1:raise RuntimeError('exactly one native language prefill capture required')


def parse_probe(argv):
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--capture-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--warmup',type=int,default=1)
    args=p.parse_args(argv)
    if args.repeats<1 or args.warmup<0:p.error('positive repeats and nonnegative warmup required')
    return args


def probe_main(argv):
    args=parse_probe(argv);args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',diagnostic_only=True,quality_or_e2e_claim=False,fallback_allowed=False,
        requested_backend='CUDNN_ATTENTION only',scope='one actual Target language prefill attention invocation',
        source_layout_restored=True,shared_gpu=True,failures=[],pairs=[])
    save(args.out/'summary.json',report)
    try:
        import run_joint
        with run_joint.cooperative_lock(args.physical_gpu,args.out) as lock:
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu)
            import torch
            from prefill_cudnn_runtime import CausalCudnnExecutor,validate_prefill_attention
            torch.set_num_threads(1)
            source=json.loads((args.capture_dir/'summary.json').read_text())
            if source.get('schema')!='real_language_prefill_attention_v1' or len(source['captures'])!=1:
                raise ValueError('one verified language-prefill capture required')
            path=args.capture_dir/source['captures'][0]['file']
            saved=torch.load(path,map_location='cpu',weights_only=True);meta,tensors=saved['metadata'],saved['tensors']
            if meta.get('mask','missing') is not None or meta.get('is_causal') is not True or meta.get('enable_gqa') is not True:
                raise ValueError('capture did not contain implicit causal GQA')
            report.update(lock=lock,torch=torch.__version__,cudnn=torch.backends.cudnn.version(),
                device=torch.cuda.get_device_name(0),capture_file=path.name,
                capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input=meta,
                repeats=args.repeats,warmup=args.warmup,
                precision_flags=dict(math_allow_fp16_bf16_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic))
            def observation():
                try:
                    import gpu_lock
                    snapshot,processes=gpu_lock._gpu_status(args.physical_gpu)
                    return dict(snapshot=snapshot,processes=processes)
                except Exception as exc:return dict(error=str(exc))
            report['gpu_before']=observation();executor=CausalCudnnExecutor()
            with torch.inference_mode():
                data={name:restore_tensor(tensors[name],meta['tensors'][name],'cuda:0',torch) for name in ('query','key','value')}
                q,k,v=(data[name] for name in ('query','key','value'));scale=meta['scale']
                validate_prefill_attention(q,k,v,None,meta['context']['prompt_tokens'])
                reference=tensors['native_output'].to('cuda:0')
                native=native_prefill_math(q,k,v,scale,torch)
                report['native_replay']=metrics(native,reference,torch)
                if not report['native_replay']['bitwise_equal'] or not report['native_replay']['finite']:
                    raise RuntimeError('native causal MATH+GQA replay gate failed')
                del native
                def candidate():return executor(q,k,v,scale).transpose(1,2).contiguous()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=torch.cuda.memory_allocated();start=time.perf_counter()
                proposed=candidate();torch.cuda.synchronize()
                report['cold_candidate']=dict(synchronized_wall_ms=(time.perf_counter()-start)*1000,
                    incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-before,
                    error=metrics(proposed,reference,torch))
                del proposed
                if not report['cold_candidate']['error']['finite']:raise RuntimeError('causal cuDNN returned nonfinite output')
                for _ in range(args.warmup):native_prefill_math(q,k,v,scale,torch);candidate()
                torch.cuda.synchronize();rng=random.Random(20261002)
                for repeat in range(args.repeats):
                    order=['native','cudnn'];rng.shuffle(order);pair=dict(repeat=repeat,order=order)
                    for mode in order:
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        before=torch.cuda.memory_allocated();start=time.perf_counter()
                        out=native_prefill_math(q,k,v,scale,torch) if mode=='native' else candidate()
                        torch.cuda.synchronize();pair[mode+'_wall_ms']=(time.perf_counter()-start)*1000
                        pair[mode+'_incremental_peak_allocated_bytes']=torch.cuda.max_memory_allocated()-before
                        if mode=='cudnn':
                            pair['error']=metrics(out,reference,torch)
                            if not pair['error']['finite']:raise RuntimeError('warm cuDNN output became nonfinite')
                        del out
                    report['pairs'].append(pair)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
                    candidate();torch.cuda.synchronize()
                trace.export_chrome_trace(str(args.out/'causal_cudnn_trace.json'))
                report['operator_names']=sorted({item.key for item in trace.key_averages()})
                report['cudnn_operator_observed']=any('cudnn_attention' in name for name in report['operator_names'])
                report['median']={mode:dict(wall_ms=statistics.median(p[mode+'_wall_ms'] for p in report['pairs']),
                    incremental_peak_allocated_bytes=statistics.median(p[mode+'_incremental_peak_allocated_bytes'] for p in report['pairs'])) for mode in ('native','cudnn')}
            report['executor']=executor.metadata();report['gpu_after']=observation();report['status']='completed'
            report['limitation']='Standalone attention only; long-answer quality, actual whole-request peak and DRAM remain unverified.'
    except Exception as exc:
        report['status']='failed';report['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        if 'executor' in locals():report['executor']=executor.metadata()
        raise
    finally:save(args.out/'summary.json',report)


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ('capture','probe'):raise SystemExit('choose capture or probe')
    (capture_main if args[0]=='capture' else probe_main)(args[1:])


if __name__=='__main__':main()
