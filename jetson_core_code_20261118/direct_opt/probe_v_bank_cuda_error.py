"""Correctness-only CUDA gate for the independent V-pack reconstruction-error statistic.

Uses the frozen V-bank probe's device/lock/compiler/monitor contracts. New pack
and frozen pack receive the same synthetic FP16 source; no model is loaded.
Shared validation requires the explicit flag and provides no timing/DRAM claim.
This probe does not install a mixed-precision selector or retain full source V.
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
cpu_values,cpu_pack,cpu_operand=base.cpu_values,base.cpu_pack,base.cpu_operand
byte_check,jit_cache_receipt=base.byte_check,base.jit_cache_receipt
cases=base.cases
PROTOCOL=dict(base.PROTOCOL,schema='partitioned_v_pack_error_probe_v1',
    numerical='six frozen fixtures: FP16 control; V8 new/frozen pack bytes; error FP32 CPU bytes plus FP64 rounding report',
    error_statistic='one FP32 max absolute reconstructed-value error per KV head/visual row',
    selector_implemented=False,mixed_bank_implemented=False,task_quality_claim=False,
    error_injection='direct kernel diagnostic only; original seven baseline guards plus new error-pack guards')


def source_receipt():
    receipt=base.source_receipt()
    for name in ('probe_v_bank_cuda_error.py','v_bank_cuda_error_ops.py','visual_v_quant_risk_reference.py'):
        path=HERE/name
        receipt[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-graphics-pid', type=int, choices=(6360,), action='append', default=[])
    parser.add_argument('--allow-shared-validation', action='store_true',
                        help='explicit correctness-only sharing; no performance or DRAM claim')
    args = parser.parse_args(argv)
    try:
        raw = args.expected_gpu_uuid
        if not raw.startswith('GPU-') or str(uuid.UUID(raw[4:])) != raw[4:]:
            raise ValueError('canonical NVML GPU-UUID required')
        if args.out.exists(): raise ValueError('fresh directory required; no retry/resume')
    except ValueError as exc: parser.error(str(exc))
    args.allow_graphics_pid = sorted(set(args.allow_graphics_pid))
    return args



def cpu_error_reference(torch,source,positions):
    """Independent FP32 code/scale and error definition; FP64 statistics are separate.

    No CUDA/reference pack functions are called. The returned FP32 array is the
    exact target for the GPU statistic; its FP64 comparison is not a QA bound.
    """
    require(source.device.type=='cpu' and source.dtype==torch.float16 and source.ndim==4
        and source.shape[:2]==(1,4) and source.shape[-1]==128,
            'CPU FP16 source required for independent error oracle')
    require(bool(torch.isfinite(source).all()),'oracle source must be finite')
    require(bool(positions) and all(type(i) is int and 0<=i<source.shape[2] for i in positions)
        and len(set(positions))==len(positions),'oracle requires explicit unique visual positions')
    codes,scales,_=cpu_pack(torch,source,positions,'int8_visual')
    chosen=source.index_select(2,torch.tensor(positions,dtype=torch.long)).float()
    represented=(codes.float().reshape(1,4,len(positions),4,32)*scales[...,None]).reshape_as(chosen)
    error32=(represented-chosen).abs().amax(-1)[0]
    error64=(represented.double()-chosen.double()).abs().amax(-1)[0]
    residual=(error32.double()-error64).abs()
    stats=dict(source_dtype='float16',represented_dtype='float32',GPU_statistic_dtype='float32',
        oracle_statistic_dtype='float32',comparison_statistic_dtype='float64',
        reference_error64_max=float(error64.max()),reference_error64_min=float(error64.min()),
        error32_vs_error64_max_absolute_difference=float(residual.max()),
        error32_underestimates_error64_rows=int((error32.double()<error64).sum()),
        row_count=error32.numel(),strict_error_bound_claim=False)
    return error32,stats


def validate_workload_report(report):
    require(len(report['cases'])==6 and all(r.get('status')=='passed' for r in report['cases']),
            'incomplete six-case numerical gate')
    candidate=[r for r in report['cases'] if r.get('mode')=='int8_visual']
    require(len(candidate)==3 and all(r.get('frozen_pack_bytes_equal') is True and
        r.get('error_FP32_byte_equal') is True and r.get('error_output_released') is True and
        r.get('source_tensor_released') is True for r in candidate),
        'missing candidate pack/error/lifetime evidence')
    require(len(report['error_pack_diagnostics'])==3 and
        all(r.get('status')=='passed' and r.get('explicit_boundary_rejected') is True
            for r in report['error_pack_diagnostics']), 'incomplete new error-pack guard diagnostics')


def run_numerics(torch,device,report,persist,process_check):
    import triton
    require(triton.__version__==TRITON_VERSION,'imported Triton differs from pinned 3.4.0')
    import v_bank_cuda_ops as ops
    import v_bank_cuda_error_ops as error_ops
    from partitioned_v_bank import VisualPartition
    pack_kernel,fill_kernel=ops._get_kernels()
    (error_kernel,)=error_ops._get_kernels()
    report['triton_version']=triton.__version__
    for kernel in (pack_kernel,fill_kernel,error_kernel):
        require(not jit_cache_receipt(kernel),'fresh probe requires empty in-process executor caches')
    for case in cases():
        for mode in ops.MODES:
            row=dict(case=case['name'],mode=mode,status='starting',fills=[],error_diagnostic_enabled=mode=='int8_visual')
            report['cases'].append(row);persist();process_check()
            P=case['P'];positions=case['visual'];M=len(positions);R=P-M
            source_cpu=cpu_values(torch,P+5)[:,:,:P]
            visual_cpu,scales_cpu,text_cpu=cpu_pack(torch,source_cpu,positions,mode)
            route=ops.prepare_partition_route(VisualPartition.prepare(P,positions,
                provenance={'synthetic_case':case['name']}),device)
            source=base._cuda_strided(torch,source_cpu,device)
            source_ref=weakref.ref(source);source_storage=source.untyped_storage().data_ptr()
            if mode=='int8_visual':
                control,control_text=ops.pack_v_cuda(source,route,mode=mode)
                # A nonzero storage offset tests a layer slice of one caller-owned slab.
                with torch.inference_mode(False):
                    error_slab=torch.empty((3,4,M),dtype=torch.float32,device=device)
                    error_out=error_slab[1]
                error_ref=weakref.ref(error_out);slab_ref=weakref.ref(error_slab)
                error_snapshot=error_ops.error_output_metadata(error_out)
                bank,text=error_ops.pack_v_with_error(source,route,error_out,mode=mode)
                error_ops.validate_error_output_metadata(error_out,error_snapshot)
                ops.validate_pack_status([control,bank])
                row['frozen_pack_comparisons']=[byte_check(torch,a,b.detach().cpu(),label) for a,b,label in (
                    (bank.visual_values,control.visual_values,'new/frozen V8 codes'),
                    (bank.scales,control.scales,'new/frozen scales'),(text,control_text,'new/frozen text'))]
                expected_error,stats=cpu_error_reference(torch,source_cpu,positions)
                row['error_comparison']=byte_check(torch,error_out,expected_error,'FP32 reconstruction-error statistic')
                row['error_FP64_comparison']=stats
                row.update(frozen_pack_bytes_equal=True,error_FP32_byte_equal=True,
                    error_output_shape=list(error_out.shape),error_output_storage_offset=error_out.storage_offset(),
                    error_output_metadata=error_snapshot,
                    error_pack_plan=error_ops.pack_error_launch_plan(P,M),
                    caller_error_backing_bytes=error_slab.untyped_storage().nbytes(),
                    error_output_logical_bytes=error_out.numel()*error_out.element_size(),
                    error_executor_cache=jit_cache_receipt(error_kernel))
                require(bool(row['error_executor_cache']),'no observed compiled error-pack executor')
                roots=[bank.visual_values,bank.scales,bank.error_flag,text,error_out,error_slab,
                    control.visual_values,control.scales,control.error_flag,control_text]
                require(all(x.untyped_storage().data_ptr()!=source_storage for x in roots if x.numel()),
                        'error pack output aliases source')
                del roots,error_out,error_slab,control,control_text;gc.collect()
                require(error_ref() is None and slab_ref() is None,'bank retained caller error tensor/slab')
                row['error_output_released']=True
            else:
                bank,text=ops.pack_v_cuda(source,route,mode=mode)
                ops.validate_pack_status([bank])
            row['pack_bytes']=[byte_check(torch,bank.visual_values,visual_cpu,'visual'),
                              byte_check(torch,text,text_cpu,'compact prompt text')]
            if scales_cpu is not None:row['pack_bytes'].append(byte_check(torch,bank.scales,scales_cpu,'scales'))
            roots=[bank.visual_values,bank.error_flag,text,route.prefix_route]+([] if bank.scales is None else [bank.scales])
            require(all(x.untyped_storage().data_ptr()!=source_storage for x in roots if x.numel()),'bank aliases source V')
            del roots,source;gc.collect()
            require(source_ref() is None,'full source V retained after pack')
            row.update(source_tensor_released=True,storage=bank.storage_receipt(),
                pack_plan=ops.pack_launch_plan(P,M,mode),pack_executor_cache=jit_cache_receipt(pack_kernel))
            max_extra=max(case['extra'])
            text_backing_cpu=cpu_values(torch,R+max_extra+3,offset=19)
            text_backing_cpu[:,:,:R].copy_(text_cpu)
            text_backing=base._cuda_strided(torch,text_backing_cpu,device);del text
            for extra in case['extra']:
                S=P+extra;current=text_backing[:,:,:R+extra]
                expected=cpu_operand(torch,P,positions,visual_cpu,scales_cpu,text_backing_cpu[:,:,:R+extra])
                out=torch.empty((1,4,S,128),dtype=torch.float32,device=device)
                before=jit_cache_receipt(fill_kernel)
                returned=ops.fill_value_fp32_(out,bank,current)
                require(returned is out,'fill replaced caller operand')
                ops.check_status([bank],stage='error_probe_after_fill')
                first=byte_check(torch,out,expected,'FP32 operand')
                after=jit_cache_receipt(fill_kernel)
                ops.fill_value_fp32_(out,bank,current);ops.check_status([bank],stage='error_probe_replay')
                replay=byte_check(torch,out,expected,'FP32 replay')
                require(after==jit_cache_receipt(fill_kernel),'identical-pointer replay created executor key')
                out_ref=weakref.ref(out);del out,returned;gc.collect()
                require(out_ref() is None,'bank retained caller FP32 operand')
                row['fills'].append(dict(S=S,text_stride=list(current.stride()),comparisons=[first,replay],
                    caller_operand_released=True,executor_keys_before=before,executor_keys_after=after,
                    same_pointer_replay_cache_reused=True))
                persist();process_check()
            row.update(status='passed',cache_scope='Observed executor keys, not assumed shape-only specialization',
                dynamic_S_new_executor_counts=[len(x['executor_keys_after'])-len(x['executor_keys_before']) for x in row['fills']])
            del current,text_backing,bank,route;persist()
    # These seven are explicitly tests of frozen baseline pack/fill device guards.
    base.error_diagnostics(torch,device,report,persist,process_check)
    for row in report['error_diagnostics']:row['guard_scope']='frozen_baseline_pack_fill'
    error_pack_diagnostics(torch,device,report,persist,process_check)


def error_pack_diagnostics(torch,device,report,persist,process_check):
    """New error-pack guards; corrupt routes only through a labelled raw-kernel test."""
    import v_bank_cuda_ops as ops
    import v_bank_cuda_error_ops as error_ops
    from partitioned_v_bank import VisualPartition
    (kernel,)=error_ops._get_kernels()
    for kind in ('nonfinite_visual_source','nonfinite_text_source','invalid_prefix_route'):
        row=dict(kind=kind,status='starting',guard_scope='new_error_pack',
                 direct_kernel_diagnostic=kind=='invalid_prefix_route')
        report['error_pack_diagnostics'].append(row);persist();process_check()
        route=ops.prepare_partition_route(VisualPartition.prepare(9,(1,3,6)),device)
        source=cpu_values(torch,9).to(device)
        if kind=='nonfinite_visual_source':source[0,0,1,0]=float('nan')
        elif kind=='nonfinite_text_source':source[0,0,0,0]=float('inf')
        with torch.inference_mode(False):
            error_out=torch.empty((4,3),dtype=torch.float32,device=device)
        bank,text=error_ops.pack_v_with_error(source,route,error_out)
        if kind=='invalid_prefix_route':
            ops.validate_pack_status([bank])
            bad_route=route.prefix_route.clone();bad_route[0]=-route.R-1
            def raw(route_tensor):
                kernel[(4*9,)](source,route_tensor,bank.visual_values,bank.scales,text,
                    bank.error_flag,error_out,9,3,6,*source.stride()[1:],
                    num_warps=4,num_stages=1,enable_fp_fusion=False)
            raw(bad_route)
            require(int(bank.error_flag.item())&1,'invalid pack route missing error bit')
            # This diagnostic never clears flags. A later valid raw launch must
            # poison all visual statistics, including rows not initially affected.
            raw(route.prefix_route)
            require(bool(torch.isnan(error_out).all()),'persistent pack flag did not poison e')
            row['persistent_error_statistics_nan']=True
        else:
            require(int(bank.error_flag.item())&2,'nonfinite pack missing error bit')
            if kind=='nonfinite_visual_source':
                require(bool(torch.isnan(error_out[0,0])),'nonfinite visual statistic not NaN')
        try:ops.validate_pack_status([bank])
        except RuntimeError:pass
        else:raise RuntimeError('error pack accepted invalid diagnostic')
        require(bank.failed and not bank.validated,'failed error bank was not poisoned')
        operand=torch.empty((1,4,9,128),dtype=torch.float32,device=device)
        try:ops.fill_value_fp32_(operand,bank,text)
        except RuntimeError:pass
        else:raise RuntimeError('production fill reused failed error bank')
        row.update(status='passed',explicit_boundary_rejected=True,production_reuse_rejected=True,
            error_flag=int(bank.error_flag.item()),
            invalid_output_accepted=False,all_first_launch_rows_nan_required=False)
        persist()



def run_probe(args,*,torch_module=None,lock_factory=None,backend_factory=None,
              workload=run_numerics,monitor_factory=ProcessMonitor,sleep=time.sleep,
              version_reader=importlib.metadata.version):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(PROTOCOL,status='starting',pid=os.getpid(),started=time.time(),
        gpu=args.gpu,expected_gpu_uuid=args.expected_gpu_uuid,allowed_graphics=args.allow_graphics_pid,
        guards=[],cases=[],error_diagnostics=[],error_pack_diagnostics=[],environment_before=environment())
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
            require(len(report['cases'])==6 and all(r.get('status')=='passed' for r in report['cases']),
                    'incomplete six-case numerical gate')
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
