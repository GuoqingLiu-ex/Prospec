"""Synthetic-only delta-commit GPU probe. CPU import/help create no CUDA state.

Root launches explicitly, e.g.:
 python probe_delta_kv_commit.py --out NEW --physical-gpu 7 \
   --header-environment RUN/kv_int8_primitive_r3/header_environment.json

No model, real request, runtime patch, GPU setting change or automatic fallback.
Each timing resets source buffers outside the interval. Whole-bank byte equality
(including NaN payloads and signed zero) precedes and follows timings. GPU timing
is primitive-only and excludes source resets/plan preparation/validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
HEADER_SHA='7f7941cb968b86454b1d064574e35a9e216f2cfd2f95c5e04cf81a6f5f78f1b2'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--header-environment',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--warmup',type=int,default=2)
    args=p.parse_args(argv)
    if args.repeats<1 or args.warmup<0:p.error('positive repeats and nonnegative warmup required')
    return args


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def header_environment(path):
    value=json.loads(path.read_text())
    if (value.get('package')!='libpython3.12-dev_3.12.3-1ubuntu0.13_amd64.deb'
            or value.get('sha256')!=HEADER_SHA
            or value.get('header_package_version')!='3.12.3-1ubuntu0.13'
            or value.get('runtime_package_version')!='3.12.3-1ubuntu0.13'
            or value.get('system_installation_modified') is not False):
        raise ValueError('verified exact Ubuntu .13 header receipt required')
    if tuple(sys.version_info[:3])!=(3,12,3):
        raise ValueError('probe requires existing Python 3.12.3 runtime')
    cp=value.get('CPATH','');parts=[Path(x) for x in cp.split(':')]
    if len(parts)!=3 or not all(p.is_absolute() and p.is_dir() for p in parts):
        raise ValueError('three existing absolute .13 include directories required')
    root=parts[0].parents[2]
    expected=[root/'usr/include/python3.12',root/'usr/include',root/'usr/include/x86_64-linux-gnu/python3.12']
    if parts!=expected or 'python_headers_3.12.3_ubuntu0.13' not in str(root):
        raise ValueError('CPATH must refer only to the project-local exact .13 headers')
    hashes={}
    for p in (parts[0]/'Python.h',parts[0]/'patchlevel.h',parts[0]/'pyconfig.h',parts[2]/'pyconfig.h'):
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(receipt=value,receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                actual_header_sha256=hashes,effective_cpath=cp,
                provenance_note='package SHA is prior root verification; current header bytes hashed again')


def cases():
    return [
        dict(name='inplace_single',path=(0,),nodes=1,prefix=127,recent=0,capacity=512,fixed=False),
        dict(name='inplace_path',path=(0,1,2,3),nodes=30,prefix=127,recent=15,capacity=512,fixed=False),
        dict(name='overlap_forward',path=(0,2,3,5),nodes=30,prefix=127,recent=16,capacity=512,fixed=False),
        dict(name='overlap_permutation',path=(0,3,2,1),nodes=30,prefix=127,recent=17,capacity=512,fixed=False),
        dict(name='all30_reverse',path=(0,)+tuple(range(29,0,-1)),nodes=30,prefix=127,recent=18,capacity=512,fixed=False),
        dict(name='exact_capacity_boundary',path=(0,2,5,9),nodes=30,prefix=127,recent=508,capacity=512,fixed=False),
        dict(name='fixed_scratch_nonoverlap',path=(0,2,5,9),nodes=30,prefix=127,recent=20,capacity=512,fixed=True),
    ]


def make_plan(case,device,torch):
    from ring_kv_cache import prepare_tree,prepare_commit
    from delta_kv_commit import prepare_delta_commit
    n=case['nodes'];path=case['path']
    mask=torch.eye(n,dtype=torch.bool);mask[:,0]=True
    for depth,node in enumerate(path):
        mask[node]=False;mask[node,list(path[:depth+1])]=True
    tree=prepare_tree(mask.to(device))
    accepted=prepare_commit(tree,torch.tensor(path,dtype=torch.long,device=device))
    return prepare_delta_commit(accepted,prefix_length=case['prefix'],recent_count=case['recent'],
        recent_capacity=case['capacity'],tree_capacity=n,
        scratch_start=case['prefix']+(case['capacity'] if case['fixed'] else case['recent']))


def synthetic_banks(capacity,torch):
    # Many distinct rows/channels/heads prevent an overlap bug from hiding in a
    # repeated constant. Includes every 16-bit pattern, hence all NaN payloads.
    count=4*capacity*128
    kbits=(torch.arange(count,dtype=torch.int64)*149+17).remainder(65536).sub(32768).to(torch.int16)
    vbits=(torch.arange(count,dtype=torch.int64)*283+91).remainder(65536).sub(32768).to(torch.int16)
    return (kbits.reshape(1,4,capacity,128).view(torch.float16),
            vbits.reshape(1,4,capacity,128).view(torch.float16))


def bitwise(a,b,torch):
    return bool(torch.equal(a.view(torch.uint8),b.view(torch.uint8)))


def ptx_barrier_receipt(ptx):
    """Require visible all-load-before-barrier/store ordering in emitted PTX.

    This conservative check fails if the compiler changes syntax; it does not
    treat an unrecognized lowering as verified. It is additional evidence to
    byte tests, not a replacement for overlap tests or the CUDA memory model.
    """
    lines=ptx.splitlines()
    loads=[i for i,x in enumerate(lines) if re.search(r'\bld\.volatile\.global\.',x)]
    barriers=[i for i,x in enumerate(lines) if re.search(r'\bbar(?:\.sync|rier\.sync)\b',x)]
    stores=[i for i,x in enumerate(lines) if re.search(r'\bst\.global\.',x)]
    ordered=bool(loads and stores and any(max(loads)<b<min(stores) for b in barriers))
    if not ordered:
        raise RuntimeError('PTX does not demonstrate all volatile payload loads before CTA barrier and all stores after')
    return dict(verified=True,volatile_payload_load_instructions=len(loads),
        global_store_instructions=len(stores),barrier_lines=[i+1 for i in barriers],
        final_payload_load_line=max(loads)+1,first_payload_store_line=min(stores)+1,
        ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest())


def compiled_ptx(out,device_index):
    import delta_kv_commit as delta
    if delta._KERNEL is None:raise RuntimeError('no payload kernel compiled')
    cache=delta._KERNEL.device_caches[device_index][0]
    if not cache:raise RuntimeError('Triton compiled-kernel cache empty after moving case')
    entries=[]
    for i,(key,kernel) in enumerate(cache.items()):
        ptx=kernel.asm['ptx'];filename=f'delta_kernel_{i}.ptx'
        (out/filename).write_text(ptx)
        entries.append(dict(cache_key=str(key),file=filename,**ptx_barrier_receipt(ptx)))
    return entries


def main(argv=None):
    args=parse_args(argv)
    header=header_environment(args.header_environment)
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',diagnostic_only=True,quality_claim=False,
        model_loaded=False,whole_request_speed_claim=False,physical_gpu=args.physical_gpu,
        header_environment=header,cases=[],repeats=args.repeats,warmup=args.warmup,
        timing_scope='one layer payload commit; resets, validation, plan/metadata outside interval',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__),HERE/'delta_kv_commit.py',HERE/'ring_kv_cache.py')})
    save(args.out/'summary.json',report)
    old_gpu=os.environ.get('CUDA_VISIBLE_DEVICES');old_cpath=os.environ.get('CPATH')
    try:
        sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parent/'vendor'),str(HERE.parent/'training_free')]
        import run_joint
        with run_joint.cooperative_lock(args.physical_gpu,args.out) as lock:
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu)
            os.environ['CPATH']=header['effective_cpath']
            import torch
            import delta_kv_commit as delta
            if torch.cuda.is_initialized():raise RuntimeError('CUDA was initialized before the probe selected its device')
            torch.set_num_threads(1)
            with torch.cuda.device(0),torch.inference_mode():
                device=torch.device('cuda:0')
                report.update(lock=lock,torch=torch.__version__,device=torch.cuda.get_device_name(0),
                              capability=list(torch.cuda.get_device_capability(0)))
                for c in cases():
                    row=dict(case=c,status='starting',pairs=[]);report['cases'].append(row)
                    start=time.perf_counter();cpu_plan=make_plan(c,'cpu',torch)
                    plan=make_plan(c,device,torch);torch.cuda.synchronize(device)
                    row['plan_wall_ms']=(time.perf_counter()-start)*1000
                    cpu_k,cpu_v=synthetic_banks(c['prefix']+c['capacity']+30,torch)
                    ref_k,ref_v=cpu_k.clone(),cpu_v.clone()
                    delta.reference_commit_(ref_k,ref_v,cpu_plan)
                    original_k,original_v=cpu_k.to(device),cpu_v.to(device)
                    k,v=original_k.clone(),original_v.clone()
                    expected_k,expected_v=ref_k.to(device),ref_v.to(device)
                    def reset():
                        k.copy_(original_k);v.copy_(original_v)
                        torch.cuda.synchronize(device)
                    def check():
                        ok=bitwise(k,expected_k,torch) and bitwise(v,expected_v,torch)
                        if not ok:raise RuntimeError('whole-bank bit equality failed for '+c['name'])
                    reset();start=time.perf_counter()
                    cold=delta.delta_commit_(k,v,plan);torch.cuda.synchronize(device)
                    row['cold_wall_ms']=(time.perf_counter()-start)*1000
                    row['receipt']=cold;check();row['cold_bitwise_equal']=True
                    if plan.moved_tokens:
                        report['compiled_ptx']=compiled_ptx(args.out,0)
                    for _ in range(args.warmup):
                        reset();delta.delta_commit_(k,v,plan);torch.cuda.synchronize(device);check()
                    rng=random.Random(20261002)
                    for rep in range(args.repeats):
                        order=['reference','delta'];rng.shuffle(order);pair=dict(repeat=rep,order=order)
                        for name in order:
                            reset();start_event=torch.cuda.Event(enable_timing=True)
                            end_event=torch.cuda.Event(enable_timing=True)
                            start=time.perf_counter();start_event.record(torch.cuda.current_stream(device))
                            receipt=(delta.reference_commit_(k,v,plan) if name=='reference'
                                     else delta.delta_commit_(k,v,plan))
                            end_event.record(torch.cuda.current_stream(device));torch.cuda.synchronize(device)
                            pair[name]=dict(wall_ms=(time.perf_counter()-start)*1000,
                                            event_ms=start_event.elapsed_time(end_event),receipt=receipt)
                            check()
                        pair['both_bitwise_equal']=True;row['pairs'].append(pair)
                    # Untimed trace verifies actual payload launch count, including zero moves.
                    reset()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as trace:
                        delta.delta_commit_(k,v,plan);torch.cuda.synchronize(device)
                    path=args.out/(c['name']+'_trace.json');trace.export_chrome_trace(str(path))
                    events=json.loads(path.read_text()).get('traceEvents',[])
                    kernels=[x.get('name','') for x in events if x.get('cat')=='kernel']
                    expected_launches=1 if plan.moved_tokens else 0
                    if len(kernels)!=expected_launches or any('_delta_commit_kernel' not in x for x in kernels):
                        raise RuntimeError('trace payload launch count/name mismatch: '+repr(kernels))
                    row.update(status='passed',actual_payload_launches=len(kernels),kernel_names=kernels,
                        median_wall_ms={n:statistics.median(p[n]['wall_ms'] for p in row['pairs']) for n in ('reference','delta')},
                        median_event_ms={n:statistics.median(p[n]['event_ms'] for p in row['pairs']) for n in ('reference','delta')})
                    save(args.out/'summary.json',report)
                report.update(status='completed',all_cases_bitwise_equal=True,
                              compiled_ptx=compiled_ptx(args.out,0))
    except BaseException as exc:
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        for name,value in [('CUDA_VISIBLE_DEVICES',old_gpu),('CPATH',old_cpath)]:
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
        save(args.out/'summary.json',report)
    return report


if __name__=='__main__':
    main()
