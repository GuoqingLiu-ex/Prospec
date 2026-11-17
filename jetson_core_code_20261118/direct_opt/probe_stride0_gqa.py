"""Captured Target tree-attention probe; no model/runtime integration.

Example (GPU execution is a separate authorized action):
  python probe_stride0_gqa.py --capture-dir ROOT/gqa_capture_r2_tensors \
      --out NEW_OUT --physical-gpu 7

Default: all manifest captures, two warmups and five alternating-order pairs.
Native replay must match saved bytes before candidate measurement. Debug math
and CPU/CUDA profiler traces run after ordinary timing/allocation measurements.
Python stride-zero views do not prove absence of internal cuBLAS copies. GPU
throughput decline alone is not a rejection of a memory/IO-oriented candidate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parent/'vendor'),str(HERE.parent/'training_free')]
from probe_gqa_attention import restore_tensor,native_math,metrics,save


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--capture-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),default=7)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args(argv)
    if a.warmup<1 or a.repeats<2:p.error('warmup>=1 and repeats>=2 required')
    return a


def capture_entries(directory):
    directory=Path(directory).resolve()
    source=json.loads((directory/'summary.json').read_text())
    if source.get('schema')!='real_tree_attention_v1' or not source.get('captures'):
        raise ValueError('nonempty real_tree_attention_v1 capture manifest required')
    files=set()
    for entry in source['captures']:
        name=entry.get('file')
        if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.pt'):
            raise ValueError('capture filename must be a local .pt basename')
        path=(directory/name).resolve()
        if path.parent!=directory or not path.is_file() or name in files:
            raise ValueError('missing, duplicated or outside-directory capture')
        files.add(name)
    return source


def validate_capture(saved,entry):
    if not isinstance(saved,dict) or set(saved)!={'metadata','tensors'}:
        raise ValueError('invalid capture payload')
    meta,tensors=saved['metadata'],saved['tensors']
    # Native metadata is also the manifest entry in this capture schema.
    if meta!=entry:raise ValueError('manifest and saved capture metadata differ')
    if set(tensors)!={'query','key','value','mask','native_output'}:
        raise ValueError('missing or unexpected captured tensors')
    if meta.get('context',{}).get('backend')!='capture':
        raise ValueError('capture must originate from native attention')
    scale=meta.get('scale')
    if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:
        raise ValueError('finite positive captured scale required')
    for name,tensor in tensors.items():
        description=meta['tensors'][name]
        if list(tensor.shape)!=description['shape'] or str(tensor.dtype)!=description['dtype']:
            raise ValueError('captured tensor metadata mismatch')
    return meta,tensors


@contextmanager
def precision_scope(torch):
    previous=dict(tf32=torch.backends.cuda.matmul.allow_tf32,
        precision=torch.get_float32_matmul_precision(),
        reduced=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    try:yield dict(allow_tf32=False,float32_matmul_precision=torch.get_float32_matmul_precision(),
                   reduced_precision_math_sdp=False)
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous['tf32']
        # Setting allow_tf32=True changes medium -> high; restore the precision
        # string last so the complete original coupled state is preserved.
        torch.set_float32_matmul_precision(previous['precision'])
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous['reduced'])


def native_gate(call,reference,torch,record=None):
    output=call()
    result=metrics(output,reference,torch)
    if record is not None:record.update(result)
    del output
    if not result['finite'] or not result['bitwise_equal']:
        raise RuntimeError('native replay must be finite and byte-exact before candidate execution')
    return result


def paired_measure(calls,reference,torch,warmup,repeats,sample_index=0,record=None):
    """No debug/profile. Invocation includes candidate Python receipt/check cost.

    The fixture and saved output remain resident equally for both arms. Each
    measured output is deleted before the other arm; errors are checked only
    after synchronized wall/allocator measurements have been recorded.
    """
    for _ in range(warmup):
        for label in ('native','candidate'):
            output=calls[label]();del output
    torch.cuda.synchronize()
    record={} if record is None else record
    pairs=[];record['pairs']=pairs
    for repeat in range(repeats):
        order=['native','candidate'] if (repeat+sample_index)%2==0 else ['candidate','native']
        pair=dict(repeat=repeat,order=order)
        pairs.append(pair)
        for label in order:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            before=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
            begin=time.perf_counter();output=calls[label]();torch.cuda.synchronize()
            item=dict(wall_ms=1000*(time.perf_counter()-begin),baseline_allocated_bytes=before,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-before,
                baseline_reserved_bytes=reserved,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                output_bytes=output.numel()*output.element_size())
            item['error']=metrics(output,reference,torch)
            del output
            pair[label]=item
            if not item['error']['finite']:raise RuntimeError(label+' became nonfinite')
            if label=='native' and not item['error']['bitwise_equal']:
                raise RuntimeError('native repeated replay changed bytes')
    record.update(
        median_wall_ms={name:statistics.median(p[name]['wall_ms'] for p in pairs) for name in calls},
        median_incremental_peak_allocated_bytes={name:statistics.median(p[name]['incremental_peak_allocated_bytes'] for p in pairs) for name in calls},
        paired_median_wall_reduction_percent=statistics.median(100*(1-p['candidate']['wall_ms']/p['native']['wall_ms']) for p in pairs),
        allocator_scope='Incremental PyTorch allocated peak above common resident inputs/reference; not model peak or internal library allocation audit',
        timing_scope='Synchronized ordinary replay including Python checks/receipt; no debug/profile/metrics; not end-to-end generation or isolated-device throughput')
    return record


def intermediate_diagnostic(q,k,v,mask,scale,reference,torch,path):
    """Independent dense HF-repeat ordering, outside timing and peak sampling."""
    from stride0_gqa_math import stride0_gqa_math
    output,receipt,debug=stride0_gqa_math(q,k,v,mask,scale,debug_intermediates=True)
    # Retain only CPU copies, then release every candidate debug GPU tensor.
    names=('q_scaled','qk_scores','masked_scores','probabilities','pv_fp32')
    candidate={name:debug[name].detach().cpu() for name in names}
    candidate_output=output.detach().cpu()
    result=dict(storage_receipt=receipt,output=metrics(output,reference,torch),
                note='Separate retained-intermediate diagnostic; never allocator/timing evidence',
                saved_file=Path(path).name)
    del output,debug
    kr=k.repeat_interleave(7,dim=1).float()
    vr=v.repeat_interleave(7,dim=1).float()
    qs=q.float()*math.sqrt(scale)
    ks=kr.transpose(-2,-1)*math.sqrt(scale)
    score=torch.matmul(qs,ks)
    dense={'q_scaled':qs.detach().cpu(),'qk_scores':score.detach().cpu().clone()}
    score.add_(mask);dense['masked_scores']=score.detach().cpu()
    probabilities=torch.ops.aten._safe_softmax.default(score,-1,None)
    pv=torch.matmul(probabilities,vr)
    dense.update(probabilities=probabilities.detach().cpu(),pv_fp32=pv.detach().cpu())
    dense_output=pv.to(q.dtype).transpose(1,2).contiguous()
    result['dense_order_vs_saved_native']=metrics(dense_output,reference,torch)
    result['stages']={name:metrics(candidate[name],dense[name],torch) for name in names}
    torch.save(dict(candidate=candidate,dense_order=dense,candidate_output=candidate_output,
                    dense_order_output=dense_output.detach().cpu(),
                    note=result['note']),path)
    return result


def profile_trace(call,torch,path):
    """Separate profiler call. Its allocations/time are diagnostic only."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as trace:
        output=call();torch.cuda.synchronize();del output
    trace.export_chrome_trace(str(path))
    operators=[]
    for event in trace.key_averages(group_by_input_shape=True):
        operators.append(dict(name=event.key,count=event.count,input_shapes=event.input_shapes,
            self_cpu_memory_bytes=event.self_cpu_memory_usage,
            self_device_memory_bytes=getattr(event,'self_device_memory_usage',None),
            cpu_total_us=event.cpu_time_total,device_total_us=getattr(event,'device_time_total',None)))
    # Chrome contains the actual CUDA activity names, separately from ATen ops.
    raw=json.loads(Path(path).read_text())
    kernels={}
    for event in raw.get('traceEvents',[]):
        if 'kernel' in str(event.get('cat','')).lower():
            name=event.get('name','<unnamed>');row=kernels.setdefault(name,dict(name=name,count=0,total_duration_us=0.0))
            row['count']+=1;row['total_duration_us']+=float(event.get('dur',0))
    materialization=[row for row in operators if any(word in row['name'] for word in
        ('clone','copy','_to_copy','contiguous'))]
    return dict(trace_file=Path(path).name,record_shapes=True,profile_memory=True,
        operators=operators,materialization_operators=materialization,cuda_kernels=list(kernels.values()),
        candidate_repeat_absence_is_not_internal_copy_proof=True,
        note='Profiler results are separate diagnostics, not ordinary latency/peak or actual DRAM bytes; CUDA-library internal packing may not be visible as ATen clone')


def replay_capture(entry,args,torch,sample_index,sampler,row=None,*,device='cuda:0'):
    # Main always uses cuda:0; the explicit device seam permits CPU lifecycle
    # tests without CUDA initialization. There is no CLI/backend fallback.
    path=args.capture_dir/entry['file']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    meta,tensors=validate_capture(saved,entry)
    data={name:restore_tensor(tensors[name],meta['tensors'][name],device,torch)
          for name in ('query','key','value','mask')}
    q,k,v,mask=(data[name] for name in ('query','key','value','mask'))
    reference=tensors['native_output'].to(device);scale=meta['scale']
    from stride0_gqa_math import stride0_gqa_math
    def native():return native_math(q,k,v,mask,scale,torch)
    def candidate():return stride0_gqa_math(q,k,v,mask,scale,debug_intermediates=False)[0]
    row={} if row is None else row
    row.update(file=entry['file'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata=meta,source_stride_restored=True,original_mask_unchanged=True,
        common_fixture='Original-stride Q/K/V/mask plus saved reference resident for both arms; no model loaded')
    row['native_replay']={}
    native_gate(native,reference,torch,row['native_replay'])
    # Distinct initial numerical call, never included in ordinary timing/peaks.
    initial,receipt,_=stride0_gqa_math(q,k,v,mask,scale,debug_intermediates=False)
    row.update(candidate_replay=metrics(initial,reference,torch),storage_receipt=receipt)
    del initial
    if not row['candidate_replay']['finite']:raise RuntimeError('candidate nonfinite before measurement')
    sampler.mark(entry['file']+':ordinary_begin')
    row['ordinary']={}
    paired_measure(dict(native=native,candidate=candidate),reference,torch,
                   args.warmup,args.repeats,sample_index,row['ordinary'])
    sampler.mark(entry['file']+':ordinary_end')
    row['intermediates']=intermediate_diagnostic(q,k,v,mask,scale,reference,torch,
                                              args.out/(path.stem+'_intermediates.pt'))
    torch.cuda.synchronize()
    row['profiles']={label:profile_trace(call,torch,args.out/(path.stem+'_'+label+'_trace.json'))
                     for label,call in (('native',native),('candidate',candidate))}
    return row


def main(argv=None):
    args=parse_args(argv)
    source=capture_entries(args.capture_dir)
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',pid=os.getpid(),samples=[],failures=[],args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        diagnostic_only=True,quality_or_e2e_speed_claim=False,actual_dram_measured=False,
        fallback_allowed=False,throughput_decline_alone_rejects_hardware_candidate=False,
        source_manifest_sha256=hashlib.sha256((args.capture_dir/'summary.json').read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ('probe_stride0_gqa.py','stride0_gqa_math.py','probe_gqa_attention.py')})
    save(args.out/'summary.json',report)
    sampler=None;old_visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        from run_joint import cooperative_lock
        with cooperative_lock(args.physical_gpu,args.out) as lock:
            # Import is harmless; reject reuse of an already-initialized device.
            import torch
            if torch.cuda.is_initialized():raise RuntimeError('probe requires a fresh process before CUDA initialization')
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu)
            from hardware_metrics import HardwareSampler
            with precision_scope(torch),torch.inference_mode():
                torch.cuda.set_device(0)
                report.update(lock=lock,torch_version=torch.__version__,cuda_version=torch.version.cuda,
                    device=torch.cuda.get_device_name(0),allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    reduced_precision_math_sdp=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
                sampler=HardwareSampler(device_index=args.physical_gpu,sample_interval_s=.1,process_interval_s=.2)
                sampler.start()
                try:
                    report['status']='running'
                    for index,entry in enumerate(source['captures']):
                        report['active_capture']=entry['file'];save(args.out/'summary.json',report)
                        row=dict(file=entry['file'],status='running');report['samples'].append(row)
                        replay_capture(entry,args,torch,index,sampler,row)
                        row['status']='completed'
                        save(args.out/'summary.json',report)
                    report['status']='completed';report.pop('active_capture',None)
                finally:
                    report['hardware']=sampler.stop()
                    report['shared_gpu']=bool(report['hardware'].get('foreign_processes'))
                    report['process_visibility_complete']=report['hardware'].get('process_visibility_complete',False)
                    report['timing_interpretation']='Exploratory module replay. Foreign processes or incomplete sampling prevent isolated-device attribution; sampling cannot exclude between-sample activity.'
                    report['energy_interpretation']='Whole probe sampler includes profiling and CPU saves; no module/process energy claim'
    except BaseException as exc:
        report['status']='failed';report['failures'].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if old_visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=old_visible
        save(args.out/'summary.json',report)
    return report


if __name__=='__main__':main()
