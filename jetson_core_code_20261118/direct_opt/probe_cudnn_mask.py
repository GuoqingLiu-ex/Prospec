"""One-capture cuDNN hard-mask dtype diagnostic; no runtime/model patching.

Example (GPU execution is an explicit separate action):
  python probe_cudnn_mask.py --capture-dir CAP --out NEW_OUT --physical-gpu 7

Compare the original FP32 additive mask with a validated FP16 hard mask while
holding captured Q/K/V, scale, source strides and cuDNN-only selection fixed.
The two candidates are independent diagnostics, not backend fallbacks. Native
FP32-mask math must first reproduce the saved output byte for byte. FP16 mask
preparation/checking is timed separately; this probe proves no full-model claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parent/'vendor'),str(HERE.parent/'training_free')]
from probe_gqa_attention import restore_tensor,native_math,metrics,save


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--capture-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    parser.add_argument('--capture-index',type=int,default=0)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=1)
    args=parser.parse_args(argv)
    if args.capture_index<0 or args.repeats<1 or args.warmup<0:
        parser.error('nonnegative capture index/warmup and positive repeats required')
    return args


def fp16_hard_mask(mask):
    """Convert only literal 0 / FP32 min / -inf; reject arbitrary additive bias.

    Each row must contain a zero. Otherwise replacing finite FP32 min with -inf
    changes fully-masked-row math semantics. Validation includes host checks;
    callers must include it in mask-preparation cost. Original tensor is unmodified.
    """
    import torch
    if (not isinstance(mask,torch.Tensor) or mask.dtype!=torch.float32 or mask.ndim!=4
            or tuple(mask.shape[:2])!=(1,1) or min(mask.shape)<1 or mask.requires_grad
            or mask.stride(-1)!=1):
        raise ValueError('FP32 inference hard mask [1,1,Q,K] with contiguous last axis required')
    zeros=mask==0
    allowed=zeros | (mask==torch.finfo(torch.float32).min) | torch.isneginf(mask)
    if not bool(allowed.all()):raise ValueError('arbitrary additive bias or nonfinite positive/NaN mask is forbidden')
    if not bool(zeros.any(dim=-1).all()):raise ValueError('every query must have an unmasked zero entry')
    # FP32 min overflows to -inf in FP16; zero and existing -inf are preserved.
    return mask.to(dtype=torch.float16)


def cudnn_only(query,key,value,mask,scale):
    """Explicit single-backend call; exceptions propagate without a fallback."""
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from gqa_attention_runtime import validate_attention
    validate_attention(query,key,value,mask)
    if query.device.type!='cuda' or query.dtype!=torch.float16:
        raise ValueError('diagnostic requires captured CUDA FP16 Q/K/V')
    if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale) or scale<=0:
        raise ValueError('positive finite captured scale required')
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output=torch.nn.functional.scaled_dot_product_attention(query,key,value,
            attn_mask=mask,dropout_p=0.0,is_causal=False,scale=float(scale),enable_gqa=True)
    if output.shape!=query.shape or output.dtype!=query.dtype or output.device!=query.device:
        raise RuntimeError('cuDNN output contract mismatch')
    return output.transpose(1,2).contiguous()


def main(argv=None):
    args=parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',diagnostic_only=True,quality_or_e2e_speed_claim=False,
                requested_backend='CUDNN_ATTENTION only',fallback_allowed=False,
                mask_semantics='literal zero retained; FP32 min and -inf become FP16 -inf; every query has visible key',
                capture_index=args.capture_index,repeats=args.repeats,warmup=args.warmup,
                candidates={},failures=[],shared_gpu=True,
                timing_scope='synchronized attention replay only; mask validation+conversion separate; capture/model loading excluded')
    save(args.out/'summary.json',report)
    try:
        import run_joint
        with run_joint.cooperative_lock(args.physical_gpu,args.out) as lock:
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu)
            import torch
            torch.set_num_threads(1)
            source=json.loads((args.capture_dir/'summary.json').read_text())
            if source.get('schema')!='real_tree_attention_v1':raise ValueError('unrecognized capture schema')
            if args.capture_index>=len(source['captures']):raise ValueError('capture index outside saved captures')
            entry=source['captures'][args.capture_index];path=args.capture_dir/entry['file']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            meta,tensors=saved['metadata'],saved['tensors']
            report.update(lock=lock,torch=torch.__version__,cudnn=torch.backends.cudnn.version(),
                device=torch.cuda.get_device_name(0),capture_file=entry['file'],
                capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input=meta['tensors'],
                source_stride_restored=True)
            def observation():
                try:
                    import gpu_lock
                    snapshot,processes=gpu_lock._gpu_status(args.physical_gpu)
                    return dict(snapshot=snapshot,processes=processes)
                except Exception as exc:return dict(error=str(exc))
            report['gpu_before']=observation()
            with torch.inference_mode():
                data={name:restore_tensor(tensors[name],meta['tensors'][name],'cuda:0',torch)
                      for name in ('query','key','value','mask')}
                q,k,v,mask=(data[name] for name in ('query','key','value','mask'))
                scale=meta['scale'];reference=tensors['native_output'].to('cuda:0')
                if q.dtype!=torch.float16:raise ValueError('first diagnostic requires original FP16 Q/K/V')
                actual=native_math(q,k,v,mask,scale,torch)
                report['native_fp32_replay']=metrics(actual,reference,torch)
                if not report['native_fp32_replay']['bitwise_equal'] or not report['native_fp32_replay']['finite']:
                    raise RuntimeError('native FP32-mask replay gate failed')
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=torch.cuda.memory_allocated();start=time.perf_counter()
                converted=fp16_hard_mask(mask)
                torch.cuda.synchronize()
                report['mask_prepare']=dict(synchronized_wall_ms=(time.perf_counter()-start)*1000,
                    incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-before,
                    output_bytes=converted.numel()*converted.element_size(),dtype=str(converted.dtype),
                    stride=list(converted.stride()),includes_validation_and_host_checks=True,
                    note='prepare cost is separate; must be added if preparing a mask per attention invocation')
                masks={'original_fp32':mask,'hard_fp16':converted}
                for name,candidate_mask in masks.items():
                    row=dict(mask_dtype=str(candidate_mask.dtype),status='starting',results=[],
                             fallback_calls=0,mask_preparation_in_attention_timing=False)
                    report['candidates'][name]=row
                    try:
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        before=torch.cuda.memory_allocated();start=time.perf_counter()
                        proposed=cudnn_only(q,k,v,candidate_mask,scale);torch.cuda.synchronize()
                        row['cold']=dict(synchronized_wall_ms=(time.perf_counter()-start)*1000,
                            incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-before,
                            error=metrics(proposed,reference,torch))
                        if not row['cold']['error']['finite']:
                            row['status']='nonfinite';continue
                        for _ in range(args.warmup):
                            native_math(q,k,v,mask,scale,torch)
                            cudnn_only(q,k,v,candidate_mask,scale)
                        torch.cuda.synchronize();rng=random.Random(20261002)
                        for repeat in range(args.repeats):
                            order=['native','cudnn'];rng.shuffle(order);pair=dict(repeat=repeat,order=order)
                            for label in order:
                                torch.cuda.synchronize();start=time.perf_counter()
                                output=(native_math(q,k,v,mask,scale,torch) if label=='native' else
                                        cudnn_only(q,k,v,candidate_mask,scale))
                                torch.cuda.synchronize();pair[label+'_wall_ms']=(time.perf_counter()-start)*1000
                                if label=='cudnn':
                                    pair['error']=metrics(output,reference,torch)
                                    if not pair['error']['finite']:raise RuntimeError('warm candidate became nonfinite')
                                del output
                            row['results'].append(pair)
                        row['median_wall_ms']={label:statistics.median(x[label+'_wall_ms'] for x in row['results']) for label in ('native','cudnn')}
                        # Standalone kernel/operator trace; never timed as replay performance.
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
                            cudnn_only(q,k,v,candidate_mask,scale);torch.cuda.synchronize()
                        trace.export_chrome_trace(str(args.out/(name+'_trace.json')))
                        row['operator_names']=sorted({x.key for x in trace.key_averages()})
                        row['cudnn_operator_observed']=any('cudnn_attention' in x for x in row['operator_names'])
                        row['status']='finite_replay_complete'
                    except Exception as exc:
                        row.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
                        # A poisoned CUDA context aborts instead of attempting another candidate.
                        torch.cuda.synchronize()
                    finally:save(args.out/'summary.json',report)
            report['gpu_after']=observation()
            report['status']='diagnostic_completed'
            report['finite_candidates']=[name for name,row in report['candidates'].items() if row['status']=='finite_replay_complete']
            report['interpretation']='Each dtype is an explicit cuDNN experiment; finite replay alone does not establish long-answer quality or backend generality.'
    except Exception as exc:
        report['status']='failed';report['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:save(args.out/'summary.json',report)


if __name__=='__main__':main()
