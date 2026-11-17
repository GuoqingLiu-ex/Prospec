"""Correctness-only GPU gate for zero-aware head-scale V pack accumulation.

New independent entry. Reuses frozen resource guards/fixtures/fill/cache receipt
helpers without changing their globals. Explicit --allow-shared records other
workloads and grants no timing, DRAM, capacity or model-quality evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid
import weakref

import probe_v_bank_cuda as base

HERE=Path(__file__).resolve().parent
ENV_KEYS=base.ENV_KEYS
OVERRIDE_KEYS=base.OVERRIDE_KEYS
TRITON_VERSION=base.TRITON_VERSION
SHARED_MIN_FREE_BYTES=base.SHARED_MIN_FREE_BYTES
save,require=base.save,base.require
environment,compiler_contract,flags=base.environment,base.compiler_contract,base.flags
probe_observation,validate_observation=base.probe_observation,base.validate_observation
ProcessMonitor=base.ProcessMonitor
cpu_pack,cpu_operand=base.cpu_pack,base.cpu_operand
byte_check,jit_cache_receipt=base.byte_check,base.jit_cache_receipt
cases=base.cases
DIAGNOSTICS=('invalid_prefix_route','nonfinite_visual_source','nonfinite_text_source',
             'negative_prior','nan_prior','positive_inf_prior','negative_inf_prior')
PROTOCOL=dict(base.PROTOCOL,schema='partitioned_v_head_scale_probe_v1',
    numerical='three shapes, new/frozen V8 codes/scales/text byte equality and independent FP32 head-score oracle',
    statistic='per-head max nonzero-group FP32 scale/2, atomic_max accumulation',
    selector_implemented=False,overlay_implemented=False,task_quality_claim=False,
    score_initialization='one [28,4] zero slab per fixture, plus explicitly labelled prior-value injection',
    error_injection='seven frozen baseline guards and seven disposable new head-scale guard diagnostics')


def source_receipt():
    receipt=base.source_receipt()
    for name in ('probe_v_head_scale_cuda.py','v_head_scale_cuda_ops.py'):
        path=HERE/name
        receipt[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return receipt


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--gpu',type=int,choices=range(8),required=True)
    parser.add_argument('--expected-gpu-uuid',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-graphics-pid',type=int,choices=(6360,),action='append',default=[])
    parser.add_argument('--allow-shared',dest='allow_shared_validation',action='store_true',
        help='explicit correctness-only shared GPU; preserve background, no performance/DRAM claim')
    args=parser.parse_args(argv)
    try:
        value=args.expected_gpu_uuid
        if not value.startswith('GPU-') or str(uuid.UUID(value[4:]))!=value[4:]:
            raise ValueError('canonical NVML GPU-UUID required')
        if args.out.exists():raise ValueError('fresh directory required; no retry/resume')
    except ValueError as exc:parser.error(str(exc))
    args.allow_graphics_pid=sorted(set(args.allow_graphics_pid))
    return args


def fixture_source(torch,case):
    """Preserve the three frozen shapes/strides, add explicit score edge heads."""
    source=base.cpu_values(torch,case['P']+5)[:,:,:case['P']]
    positions=list(case['visual']);first=positions[0]
    source[0,0,positions,:]=0.;source[0,0,first,0]=-0.
    source[0,1,positions,:]=0.
    source[0,1,first,:6]=torch.tensor([127.,.5,1.5,2.5,-.5,-1.5],dtype=torch.float16)
    # Head2 keeps the original finite FP16 extremes in every visual row.
    source[0,3,positions,:]=0.
    source[0,3,first,:32]=torch.arange(-16,16,dtype=torch.float32)*2.**-24
    return source


def cpu_head_score_reference(torch,source,positions,prior=None):
    """Independent code/scale definition; does not call the new ops reference."""
    require(source.device.type=='cpu' and source.dtype==torch.float16 and source.ndim==4
        and source.shape[:2]==(1,4) and source.shape[-1]==128,'CPU FP16 source required')
    require(bool(torch.isfinite(source).all()),'finite oracle source required')
    require(bool(positions) and len(set(positions))==len(positions)
        and all(type(i) is int and 0<=i<source.shape[2] for i in positions),'unique explicit visual positions required')
    codes,scales,_=cpu_pack(torch,source,positions,'int8_visual')
    nonzero=(codes.reshape(1,4,len(positions),4,32)!=0).any(-1)
    score=torch.where(nonzero,scales,torch.zeros_like(scales)).amax(dim=(-2,-1))[0]/2.
    if prior is not None:
        require(prior.device.type=='cpu' and prior.dtype==torch.float32 and tuple(prior.shape)==(4,)
            and bool(torch.isfinite(prior).all()) and bool((prior>=0).all()),'finite nonnegative CPU prior required')
        score=torch.maximum(score,prior)
    return score.contiguous()


def validate_workload_report(report):
    expected={c['name']:c for c in cases()}
    rows=report['cases']
    require(len(rows)==3 and {r.get('case') for r in rows}==set(expected),'incomplete/duplicate three-case gate')
    for row in rows:
        require(row.get('status')=='passed' and row.get('mode')=='int8_visual','failed or wrong pack mode')
        require(all(row.get(k) is True for k in ('frozen_pack_bytes_equal','score_FP32_byte_equal',
            'score_surroundings_unchanged','valid_prior_accumulation_byte_equal','score_replay_cache_reused',
            'source_tensor_released','score_tensor_released')),'missing new pack/score/lifecycle evidence')
        case=expected[row['case']]
        require([f.get('S') for f in row.get('fills',[])]==[case['P']+x for x in case['extra']],
            'missing/reordered actual dynamic fill lengths')
        require(all(f.get('same_pointer_replay_cache_reused') is True and f.get('caller_operand_released') is True
            for f in row['fills']),'missing fill replay/lifetime evidence')
        require(bool(row.get('head_score_executor_cache')),'missing actual compiled head-score executor')
    require(len(report['error_diagnostics'])==7 and all(r.get('status')=='passed' for r in report['error_diagnostics']),
        'incomplete frozen device guards')
    guards=report['head_scale_diagnostics']
    require(len(guards)==len(DIAGNOSTICS) and {r.get('kind') for r in guards}==set(DIAGNOSTICS),
        'missing/duplicate new head-scale device guards')
    require(all(r.get('status')=='passed' and r.get('explicit_boundary_rejected') is True
        and r.get('persistent_flag_preserved') is True and r.get('production_reuse_rejected') is True for r in guards),
        'missing persistent rejection evidence')


def run_numerics(torch,device,report,persist,process_check):
    import triton
    require(triton.__version__==TRITON_VERSION,'imported Triton differs from pinned 3.4.0')
    import v_bank_cuda_ops as ops
    import v_head_scale_cuda_ops as head_ops
    from partitioned_v_bank import VisualPartition
    pack_kernel,fill_kernel=ops._get_kernels();(head_kernel,)=head_ops._get_kernels()
    report['triton_version']=triton.__version__
    for kernel in (pack_kernel,fill_kernel,head_kernel):
        require(not jit_cache_receipt(kernel),'fresh probe requires empty in-process executor caches')
    for case in cases():
        row=dict(case=case['name'],mode='int8_visual',status='starting',fills=[])
        report['cases'].append(row);persist();process_check()
        P=case['P'];positions=case['visual'];M=len(positions);R=P-M
        source_cpu=fixture_source(torch,case)
        visual_cpu,scales_cpu,text_cpu=cpu_pack(torch,source_cpu,positions,'int8_visual')
        expected_score=cpu_head_score_reference(torch,source_cpu,positions)
        route=ops.prepare_partition_route(VisualPartition.prepare(P,positions,
            provenance={'synthetic_case':case['name']}),device)
        source=base._cuda_strided(torch,source_cpu,device)
        source_ref=weakref.ref(source);source_storage=source.untyped_storage().data_ptr()
        control,control_text=ops.pack_v_cuda(source,route,mode='int8_visual')
        with torch.inference_mode(False):
            score_slab=torch.zeros((28,4),dtype=torch.float32,device=device)
            scores=score_slab[7]
        scores_ref=weakref.ref(scores);slab_ref=weakref.ref(score_slab)
        snapshot=head_ops.head_score_metadata(scores)
        before_pack=jit_cache_receipt(head_kernel)
        bank,text=head_ops.pack_v_with_head_scale(source,route,scores)
        head_ops.validate_head_score_metadata(scores,snapshot)
        ops.validate_pack_status([control,bank])
        row['frozen_pack_comparisons']=[byte_check(torch,a,b.detach().cpu(),label) for a,b,label in (
            (bank.visual_values,control.visual_values,'new/frozen V8 codes'),
            (bank.scales,control.scales,'new/frozen scales'),(text,control_text,'new/frozen text'))]
        row['independent_CPU_pack_comparisons']=[byte_check(torch,a,b,label) for a,b,label in (
            (bank.visual_values,visual_cpu,'CPU V8 codes'),(bank.scales,scales_cpu,'CPU FP32 scales'),
            (text,text_cpu,'CPU text'))]
        row['score_comparison']=byte_check(torch,scores,expected_score,'independent FP32 head scores')
        slab_cpu=torch.zeros(28,4);slab_cpu[7]=expected_score
        row['surrounding_scores_comparison']=byte_check(torch,score_slab,slab_cpu,'whole slab offset and surroundings')
        after_pack=jit_cache_receipt(head_kernel)
        require(bool(after_pack),'no actual compiled head-score executor observed')
        def raw_replay():
            head_kernel[(4*P,)](source,route.prefix_route,bank.visual_values,bank.scales,text,
                bank.error_flag,scores,P,M,R,*source.stride()[1:],
                num_warps=4,num_stages=1,enable_fp_fusion=False)
        raw_replay();ops.check_status([bank],stage='head_scale_same_pointer_replay')
        row['score_replay_comparison']=byte_check(torch,scores,expected_score,'same-pointer accumulated maximum')
        require(after_pack==jit_cache_receipt(head_kernel),'identical-pointer score replay created an executor key')
        # Explicit diagnostic prior injection, not initialization performed by pack.
        prior=torch.tensor([0.,1.,0.,.125],dtype=torch.float32)
        scores.copy_(prior.to(device));prior_snapshot=head_ops.head_score_metadata(scores)
        raw_replay();ops.check_status([bank],stage='head_scale_valid_prior')
        head_ops.validate_head_score_metadata(scores,prior_snapshot)
        expected_prior=cpu_head_score_reference(torch,source_cpu,positions,prior)
        row['prior_accumulation_comparison']=byte_check(torch,scores,expected_prior,'max with valid finite prior')
        slab_cpu[7]=expected_prior
        byte_check(torch,score_slab,slab_cpu,'surroundings after valid-prior accumulation')
        require(after_pack==jit_cache_receipt(head_kernel),'same-pointer prior replay created an executor key')
        row.update(frozen_pack_bytes_equal=True,score_FP32_byte_equal=True,score_surroundings_unchanged=True,
            valid_prior_accumulation_byte_equal=True,score_replay_cache_reused=True,
            score_values=expected_score.tolist(),prior_values=prior.tolist(),accumulated_values=expected_prior.tolist(),
            caller_slab_zero_initializations=1,controlled_prior_uploads=1,
            head_score_metadata=snapshot,head_score_plan=head_ops.pack_head_scale_launch_plan(P,M),
            head_score_executor_keys_before=before_pack,head_score_executor_cache=after_pack,
            new_head_score_executor_count=len(after_pack)-len(before_pack),
            frozen_pack_executor_cache=jit_cache_receipt(pack_kernel))
        roots=[bank.visual_values,bank.scales,bank.error_flag,text,scores,score_slab,
            control.visual_values,control.scales,control.error_flag,control_text]
        require(all(x.untyped_storage().data_ptr()!=source_storage for x in roots if x.numel()),'pack output aliases source')
        del roots,raw_replay,scores,score_slab,control,control_text,source;gc.collect()
        require(source_ref() is None and scores_ref() is None and slab_ref() is None,'source or scores retained after pack')
        row.update(source_tensor_released=True,score_tensor_released=True,storage=bank.storage_receipt())
        max_extra=max(case['extra']);text_backing_cpu=base.cpu_values(torch,R+max_extra+3,offset=19)
        text_backing_cpu[:,:,:R].copy_(text_cpu)
        text_backing=base._cuda_strided(torch,text_backing_cpu,device);del text
        for extra in case['extra']:
            S=P+extra;current=text_backing[:,:,:R+extra]
            expected=cpu_operand(torch,P,positions,visual_cpu,scales_cpu,text_backing_cpu[:,:,:R+extra])
            out=torch.empty((1,4,S,128),dtype=torch.float32,device=device)
            before=jit_cache_receipt(fill_kernel)
            returned=ops.fill_value_fp32_(out,bank,current);require(returned is out,'fill replaced caller operand')
            ops.check_status([bank],stage='head_scale_probe_fill')
            first=byte_check(torch,out,expected,'FP32 operand')
            after=jit_cache_receipt(fill_kernel)
            ops.fill_value_fp32_(out,bank,current);ops.check_status([bank],stage='head_scale_fill_replay')
            replay=byte_check(torch,out,expected,'FP32 replay')
            require(after==jit_cache_receipt(fill_kernel),'same-pointer fill replay created an executor key')
            out_ref=weakref.ref(out);del out,returned;gc.collect()
            require(out_ref() is None,'bank retained FP32 operand')
            row['fills'].append(dict(S=S,text_stride=list(current.stride()),comparisons=[first,replay],
                executor_keys_before=before,executor_keys_after=after,same_pointer_replay_cache_reused=True,
                caller_operand_released=True))
            persist();process_check()
        row.update(status='passed',dynamic_S_new_executor_counts=[len(x['executor_keys_after'])-len(x['executor_keys_before']) for x in row['fills']],
            cache_scope='Observed actual keys. Different-S counts are recorded, not asserted zero; identical-pointer replay must reuse.')
        del current,text_backing,bank,route;persist()
    base.error_diagnostics(torch,device,report,persist,process_check)
    for row in report['error_diagnostics']:row['guard_scope']='frozen_baseline_pack_fill'
    head_scale_diagnostics(torch,device,report,persist,process_check)


def head_scale_diagnostics(torch,device,report,persist,process_check):
    """Disposable device-error injection; raw repair never clears a prior flag."""
    import v_bank_cuda_ops as ops
    import v_head_scale_cuda_ops as head_ops
    from partitioned_v_bank import VisualPartition
    (kernel,)=head_ops._get_kernels()
    for kind in DIAGNOSTICS:
        row=dict(kind=kind,status='starting',guard_scope='new_head_scale_pack',direct_kernel_diagnostic=True)
        report['head_scale_diagnostics'].append(row);persist();process_check()
        route=ops.prepare_partition_route(VisualPartition.prepare(9,(1,3,6)),device)
        clean_cpu=base.cpu_values(torch,9);source=clean_cpu.to(device)
        with torch.inference_mode(False):scores=torch.zeros(4,dtype=torch.float32,device=device)
        expected_bit=1 if kind=='invalid_prefix_route' else (2 if 'source' in kind else 8)
        if kind=='nonfinite_visual_source':source[0,0,1,0]=float('nan')
        elif kind=='nonfinite_text_source':source[0,0,0,0]=float('inf')
        elif kind in ('negative_prior','nan_prior','positive_inf_prior','negative_inf_prior'):
            scores[0]={'negative_prior':-1.,'nan_prior':float('nan'),
                'positive_inf_prior':float('inf'),'negative_inf_prior':-float('inf')}[kind]
        bank,text=head_ops.pack_v_with_head_scale(source,route,scores)
        def raw(route_tensor):
            kernel[(36,)](source,route_tensor,bank.visual_values,bank.scales,text,bank.error_flag,scores,
                9,3,6,*source.stride()[1:],num_warps=4,num_stages=1,enable_fp_fusion=False)
        if kind=='invalid_prefix_route':
            ops.validate_pack_status([bank]);bad=route.prefix_route.clone();bad[0]=-route.R-1;raw(bad)
        first_flag=int(bank.error_flag.item())
        require(first_flag&expected_bit,'missing expected new-pack device error bit')
        # Diagnostic repair changes only disposable inputs/score, never ERROR.
        source.copy_(clean_cpu.to(device));scores.zero_();raw(route.prefix_route)
        later_flag=int(bank.error_flag.item())
        require(later_flag&expected_bit and later_flag&first_flag==first_flag,'valid replay lost persistent error')
        try:ops.validate_pack_status([bank])
        except RuntimeError:pass
        else:raise RuntimeError('new head-scale pack accepted invalid diagnostic')
        require(bank.failed and not bank.validated,'failed bank not poisoned')
        operand=torch.empty((1,4,9,128),dtype=torch.float32,device=device)
        try:ops.fill_value_fp32_(operand,bank,text)
        except RuntimeError:pass
        else:raise RuntimeError('production fill reused failed bank')
        row.update(status='passed',expected_bit=expected_bit,first_error_flag=first_flag,later_error_flag=later_flag,
            explicit_boundary_rejected=True,persistent_flag_preserved=True,production_reuse_rejected=True,
            repaired_scores_finite=bool(torch.isfinite(scores).all()),invalid_output_accepted=False,
            repair_is_raw_disposable_diagnostic_not_production=True)
        persist()


def run_probe(args,*,torch_module=None,lock_factory=None,backend_factory=None,
              workload=run_numerics,monitor_factory=ProcessMonitor,sleep=time.sleep,
              version_reader=importlib.metadata.version):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(PROTOCOL,status='starting',pid=os.getpid(),probe_nonce=uuid.uuid4().hex,started=time.time(),
        gpu=args.gpu,expected_gpu_uuid=args.expected_gpu_uuid,allowed_graphics=args.allow_graphics_pid,
        guards=[],cases=[],error_diagnostics=[],head_scale_diagnostics=[],environment_before=environment())
    report['allow_shared_validation']=args.allow_shared_validation
    report['resource_mode']='shared_device_correctness_only' if args.allow_shared_validation else 'idle_device_numerical_gate'
    if args.allow_shared_validation:
        report.update(startup='flock -> two shared NVML/UUID/memory guards separated by 2s -> CUDA UUID check',
            minimum_free_memory_bytes=SHARED_MIN_FREE_BYTES,
            memory_guard_scope='engineering resource guard, not an allocator peak or capacity guarantee')
    def persist():save(out/'summary.json',report)
    persist();torch=torch_module;backend=None;monitor=None
    old_visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        report['source_sha256']=source_receipt();persist()
        report['installed_triton_version']=version_reader('triton')
        report['compiler_contract']=compiler_contract(report['environment_before'],report['installed_triton_version'])
        persist()
        if torch is None:import torch
        require(not torch.cuda.is_initialized(),'CUDA initialized before lock/idle device checks')
        report['startup_math_flags']=flags(torch)
        if lock_factory is None:
            from run_joint import cooperative_lock
            lock_factory=cooperative_lock
        if backend_factory is None:
            sys.path.insert(0,str(HERE.parent))
            from hardware_metrics import _make_backend
            backend_factory=_make_backend
        with lock_factory(args.gpu,out) as lock:
            report['lock']=dict(lock);backend=backend_factory(args.gpu)
            for i in range(2):
                row=probe_observation(backend,args);row.update(index=i)
                report['guards'].append(row);persist()
                require(row['gpu_uuid']==args.expected_gpu_uuid,'NVML physical GPU UUID mismatch')
                validate_observation(row,args,startup=True)
                if i==0:sleep(2)
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
            report['effective_environment']=environment()
            props=torch.cuda.get_device_properties(0)
            actual_uuid='GPU-'+str(uuid.UUID(str(props.uuid).removeprefix('GPU-')))
            require(actual_uuid==args.expected_gpu_uuid,'CUDA logical device and NVML UUID mismatch')
            require((props.major,props.minor)==(12,0),'this candidate gate requires SM120')
            report.update(torch=torch.__version__,cuda_device=dict(uuid=actual_uuid,name=props.name,
                capability=[props.major,props.minor]),actual_math_flags_before=flags(torch))
            monitor=monitor_factory(backend,args,out/'process_observations.jsonl');monitor.start()
            with torch.cuda.device(0),torch.inference_mode():
                try:
                    workload(torch,'cuda:0',report,persist,monitor.check)
                finally:
                    # Do not release the cooperative lock with our own queued
                    # kernels still running after a diagnostic failure.
                    torch.cuda.synchronize()
            require(len(report['cases'])==3 and all(r.get('status')=='passed' for r in report['cases']),
                    'incomplete three-case numerical gate')
            require(len(report['error_diagnostics'])==7 and all(r.get('status')=='passed' for r in report['error_diagnostics']),
                    'incomplete seven-case device-guard diagnostics')
            validate_workload_report(report)
            report['process_monitor']=monitor.stop();monitor.check();monitor=None
            report['actual_math_flags_after']=flags(torch)
            require(report['actual_math_flags_after']==report['startup_math_flags'],'math flags changed')
            require(source_receipt()==report['source_sha256'],'executed source changed during probe')
            report.update(status='completed',numerical_gate_passed=True)
    except BaseException as exc:
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if monitor is not None:
            try:report['process_monitor']=monitor.stop()
            except BaseException as exc:report['monitor_stop_error']=repr(exc)
        if backend is not None:
            try:backend.close()
            except BaseException as exc:report.update(status='failed',backend_close_error=repr(exc))
        if old_visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=old_visible
        if torch is not None and 'startup_math_flags' in report:report['exit_math_flags']=flags(torch)
        report.update(environment_after=environment(),finished=time.time())
        if report['environment_after']!=report['environment_before']:
            report.update(status='failed',environment_error='unexpected inherited environment change')
        persist()
    require(report['status']=='completed','probe failed finalization')
    return report


if __name__=='__main__':run_probe(parse_args())
