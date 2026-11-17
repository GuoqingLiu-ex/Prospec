"""Independent correctness-only GPU probe for physical FP16 head overlays.

No model, automatic queue, retry, timing/DRAM claim or production mutation API.
The protected-source poison cases explicitly bypass ordinary version guards via
Tensor.data in disposable banks; they are not supported production operations.
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
import probe_v_head_scale_cuda as head_probe

HERE=Path(__file__).resolve().parent
BUDGETS=(0,14,112)
DIAGNOSTICS=('protected_poison_unused_base','unprotected_invalid_scale',
    'persistent_prior_flag','invalid_head_route','invalid_prefix_route',
    'nonfinite_original_copy','partial_coverage')
PROTOCOL=dict(base.PROTOCOL,schema='physical_visual_v_head_overlay_probe_v1',
    numerical='three synthetic partitions times k0/14/112; original FP16 overlay and FP32 operand bytes',
    overlay_implemented=True,model_loaded=False,task_quality_claim=False,
    error_injection='disposable Tensor.data poison and flag writes only; never a production mutation API',
    source_retention='28 distinct actual inference tensors, retained as references without stack, then released',
    compilation='actual existing executor keys and SM120 metadata/PTX; no inferred no-recompile claim')


def source_receipt():
    receipt=base.source_receipt()
    for name in ('probe_v_head_overlay_cuda.py','v_head_overlay_cuda_ops.py',
                 'probe_v_head_scale_cuda.py','v_head_scale_cuda_ops.py'):
        path=HERE/name;receipt[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return receipt


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--gpu',type=int,choices=range(8),required=True)
    p.add_argument('--expected-gpu-uuid',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-graphics-pid',type=int,choices=(6360,),action='append',default=[])
    p.add_argument('--allow-shared',dest='allow_shared_validation',action='store_true')
    args=p.parse_args(argv)
    try:
        if not args.expected_gpu_uuid.startswith('GPU-') or str(uuid.UUID(args.expected_gpu_uuid[4:]))!=args.expected_gpu_uuid[4:]:
            raise ValueError('canonical GPU-UUID required')
        if args.out.exists():raise ValueError('fresh directory required, no retry/resume')
    except ValueError as exc:p.error(str(exc))
    args.allow_graphics_pid=sorted(set(args.allow_graphics_pid))
    return args


def cpu_source(torch,case,layer):
    src=head_probe.fixture_source(torch,case)
    # Preserve score edge heads; unique protected values catch cross-layer copy.
    src[0,2,list(case['visual']),70]=layer+.125
    return src


def cpu_route(torch,scores,k):
    order=torch.argsort(scores.flatten(),descending=True,stable=True)[:k]
    route=torch.full((112,),-1,dtype=torch.int32);route[order]=torch.arange(k,dtype=torch.int32)
    return route.view(28,4),order


def cpu_expected(torch,source,positions,visual,scales,current_text,route,layer):
    out=base.cpu_operand(torch,source.shape[2],positions,visual,scales,current_text)
    for h in range(4):
        if int(route[layer,h])>=0:out[0,h,list(positions)]=source[0,h,list(positions)].float()
    return out


def validate_workload_report(report):
    expected={(c['name'],k) for c in base.cases() for k in BUDGETS}
    rows=report.get('cases',[])
    base.require(len(rows)==9 and {(r.get('case'),r.get('budget_heads')) for r in rows}==expected,'nine unique numerical cases required')
    for row in rows:
        base.require(row.get('status')=='passed' and all(row.get(k) is True for k in (
            'GPU_selection_byte_equal','original_overlay_byte_equal','operand_bytes_equal',
            'all_28_copies_completed','unique_backing_matches','source_inference_tensors',
            'source_tensor_released','score_tensor_released','operand_not_retained')),'missing numerical/ownership evidence')
        c=next(c for c in base.cases() if c['name']==row['case'])
        base.require([x['S'] for x in row.get('fills',[])]==[c['P']+e for e in c['extra']],'missing dynamic fills')
        base.require(bool(row.get('copy_executor_cache')) and bool(row.get('fill_executor_cache')),'actual compiled executors required')
    guards=report.get('diagnostics',[])
    base.require(len(guards)==len(DIAGNOSTICS) and {r.get('kind') for r in guards}==set(DIAGNOSTICS)
        and all(r.get('status')=='passed' for r in guards),'complete distinct diagnostics required')
    for row in guards:
        if row['kind']=='protected_poison_unused_base':
            base.require(row.get('comparison',{}).get('byte_equal') is True
                and row.get('ordinary_version_unchanged') is True
                and row.get('protected_base_not_used_for_value') is True,'protected poison evidence missing')
        elif row['kind'] in ('nonfinite_original_copy','partial_coverage'):
            base.require(row.get('ready_rejected') is True and row.get('validated') is False,'invalid setup accepted')
        else:
            base.require(row.get('explicit_boundary_rejected') is True and row.get('production_reuse_rejected') is True
                and type(row.get('error_flag')) is int and row['error_flag']!=0,'invalid fill rejection evidence missing')


def make_fixture(torch,device,case):
    import v_bank_cuda_ops as bankops
    import v_head_scale_cuda_ops as headops
    from partitioned_v_bank import VisualPartition
    route=bankops.prepare_partition_route(VisualPartition.prepare(case['P'],case['visual']),device)
    with torch.inference_mode(False):scores=torch.zeros((28,4),dtype=torch.float32,device=device)
    sources=[];hosts=[];banks=[];texts=[];packed_cpu=[]
    for layer in range(28):
        host=cpu_source(torch,case,layer)
        source=base._cuda_strided(torch,host,device)
        base.require(torch.is_inference(source),'probe must exercise actual inference-mode source')
        bank,text=headops.pack_v_with_head_scale(source,route,scores[layer])
        sources.append(source);hosts.append(host);banks.append(bank);texts.append(text)
        packed_cpu.append(base.cpu_pack(torch,host,case['visual'],'int8_visual'))
    bankops.validate_pack_status(banks)
    expected_score=torch.stack([head_probe.cpu_head_score_reference(torch,x,case['visual']) for x in hosts])
    score_receipt=base.byte_check(torch,scores,expected_score,'all28 actual pack risk')
    return route,sources,hosts,banks,texts,packed_cpu,scores,expected_score,score_receipt


def run_numerics(torch,device,report,persist,process_check):
    import triton
    base.require(triton.__version__==base.TRITON_VERSION,'imported Triton must be 3.4.0')
    import v_head_overlay_cuda_ops as ops
    copy_kernel,fill_kernel=ops._get_kernels()
    base.require(not base.jit_cache_receipt(copy_kernel) and not base.jit_cache_receipt(fill_kernel),'fresh overlay executor caches required')
    report['triton_version']=triton.__version__
    for case in base.cases():
        process_check()
        route,sources,hosts,banks,texts,cpu_packs,scores,expected_score,score_receipt=make_fixture(torch,device,case)
        source_refs=[weakref.ref(x) for x in sources];score_ref=weakref.ref(scores)
        case_rows=[]
        for k in BUDGETS:
            row=dict(case=case['name'],budget_heads=k,status='starting',fills=[],pack_score_comparison=score_receipt)
            report['cases'].append(row);case_rows.append(row);persist();process_check()
            expected_route,selected=cpu_route(torch,expected_score,k)
            overlay=ops.prepare_head_overlay(scores,route,budget_heads=k)
            row['selection_comparison']=base.byte_check(torch,overlay.head_route,expected_route,'stable GPU route')
            expected_overlay=torch.empty((k,route.M,128),dtype=torch.float16)
            for slot,index in enumerate(selected.tolist()):expected_overlay[slot]=hosts[index//4][0,index%4,list(case['visual'])]
            before=base.jit_cache_receipt(copy_kernel)
            for layer in range(28):ops.copy_head_overlay_layer_(overlay,layer,sources[layer],banks[layer])
            ops.validate_overlay_ready(overlay,banks)
            row['overlay_comparison']=base.byte_check(torch,overlay.values,expected_overlay,'original FP16 overlay')
            receipt=overlay.storage_receipt()
            base.require(receipt['completed_layer_copies']==28 and receipt['preparation']['inference_source_layer_count']==28,'copy coverage/inference source mismatch')
            actual_roots=[overlay.head_route,overlay.values]+[x for b in banks for x in (b.visual_values,b.scales)]
            actual={str(x.device)+':'+str(x.untyped_storage().data_ptr()):x.untyped_storage().nbytes() for x in actual_roots if x.numel()}
            base.require(sum(actual.values())==receipt['actual_live_visual_base_overlay_route_unique_backing_bytes']==receipt['persistent_visual_payload_bytes'],'actual unique backing vs payload mismatch')
            del actual_roots
            maxextra=max(case['extra'])
            text_backing_cpu=[base.cpu_values(torch,route.R+maxextra+3,offset=layer+59) for layer in range(28)]
            text_backings=[]
            for layer in range(28):
                text_backing_cpu[layer][:,:,:route.R].copy_(cpu_packs[layer][2])
                text_backings.append(base._cuda_strided(torch,text_backing_cpu[layer],device))
            # Every layer is checked at each length; only one caller GPU operand.
            for extra in case['extra']:
                S=route.P+extra;fillrow=dict(S=S,comparisons=[],executor_keys_before=base.jit_cache_receipt(fill_kernel))
                out=torch.empty((1,4,S,128),dtype=torch.float32,device=device);out_ref=weakref.ref(out)
                for layer in range(28):
                    current=text_backings[layer][:,:,:route.R+extra]
                    returned=ops.fill_overlay_value_fp32_(out,banks[layer],current,overlay,layer)
                    base.require(returned is out,'caller operand replaced')
                    expected=cpu_expected(torch,hosts[layer],case['visual'],cpu_packs[layer][0],cpu_packs[layer][1],
                        text_backing_cpu[layer][:,:,:route.R+extra],expected_route,layer)
                    fillrow['comparisons'].append(base.byte_check(torch,out,expected,'FP32 layer '+str(layer)))
                ops.check_overlay_status(overlay,banks,stage='probe_fill_all_layers')
                fillrow['executor_keys_after']=base.jit_cache_receipt(fill_kernel)
                del out,returned;gc.collect();base.require(out_ref() is None,'overlay retains caller operand')
                row['fills'].append(fillrow);persist();process_check()
            row.update(status='passed',GPU_selection_byte_equal=True,original_overlay_byte_equal=True,
                operand_bytes_equal=True,all_28_copies_completed=True,unique_backing_matches=True,
                source_inference_tensors=True,operand_not_retained=True,storage=receipt,
                copy_executor_before=before,copy_executor_cache=base.jit_cache_receipt(copy_kernel),
                fill_executor_cache=base.jit_cache_receipt(fill_kernel),
                source_tensor_released=False,score_tensor_released=False)
            del current,text_backings,overlay;persist()
        # Keep final packed banks live, proving source/score lifetime independence.
        del sources,scores;gc.collect();torch.cuda.synchronize()
        base.require(all(r() is None for r in source_refs) and score_ref() is None,'source/score retained by banks/overlay')
        for row in case_rows:row.update(source_tensor_released=True,score_tensor_released=True)
        del banks,texts,route;persist()
    run_diagnostics(torch,device,report,persist,process_check)


def run_diagnostics(torch,device,report,persist,process_check):
    import v_head_overlay_cuda_ops as ops
    case=base.cases()[0]
    for kind in DIAGNOSTICS:
        process_check();row=dict(kind=kind,status='starting',disposable_diagnostic_only=True)
        report['diagnostics'].append(row);persist()
        route,sources,hosts,banks,texts,_,scores,expected_score,_=make_fixture(torch,device,case)
        k=112 if kind in ('protected_poison_unused_base','nonfinite_original_copy') else 0
        overlay=ops.prepare_head_overlay(scores,route,budget_heads=k)
        if kind=='nonfinite_original_copy':sources[0][0,0,case['visual'][0],0]=float('inf')
        for layer in range(27 if kind=='partial_coverage' else 28):
            ops.copy_head_overlay_layer_(overlay,layer,sources[layer],banks[layer])
        if kind in ('nonfinite_original_copy','partial_coverage'):
            try:ops.validate_overlay_ready(overlay,banks)
            except (RuntimeError,ValueError):pass
            else:raise RuntimeError('invalid copy/partial coverage accepted')
            row.update(status='passed',ready_rejected=True,validated=overlay.validated)
            base.require(not overlay.validated,'rejected setup marked valid');persist();continue
        ops.validate_overlay_ready(overlay,banks)
        out=torch.empty((1,4,route.P,128),dtype=torch.float32,device=device)
        if kind=='protected_poison_unused_base':
            before=(banks[0].visual_values._version,banks[0].scales._version)
            banks[0].visual_values.data.fill_(-128);banks[0].scales.data.fill_(float('nan'))
            base.require(before==(banks[0].visual_values._version,banks[0].scales._version),'poison hit ordinary version guard')
            ops.fill_overlay_value_fp32_(out,banks[0],texts[0],overlay,0)
            ops.check_overlay_status(overlay,banks,stage='protected_unused_base_poison')
            row['comparison']=base.byte_check(torch,out,hosts[0].float(),'protected output despite bad unused V8/scales')
            row.update(status='passed',protected_base_not_used_for_value=True,ordinary_version_unchanged=True)
        else:
            if kind=='unprotected_invalid_scale':banks[0].scales.data[0,0,0,0]=float('nan')
            elif kind=='persistent_prior_flag':banks[0].error_flag.fill_(16)
            elif kind=='invalid_head_route':overlay.head_route.data[0,0]=112
            elif kind=='invalid_prefix_route':route.prefix_route.data[0]=-route.R-1
            ops.fill_overlay_value_fp32_(out,banks[0],texts[0],overlay,0)
            base.require(bool(torch.isnan(out).any().item()),'invalid fill did not poison output')
            error_flag=int(banks[0].error_flag.item())
            base.require(error_flag!=0,'invalid fill did not set/retain flag')
            try:ops.check_overlay_status(overlay,banks,stage='invalid_overlay_probe')
            except RuntimeError:pass
            else:raise RuntimeError('invalid fill accepted')
            base.require(overlay.failed and not overlay.validated,'failed overlay not poisoned')
            try:ops.fill_overlay_value_fp32_(out,banks[0],texts[0],overlay,0)
            except RuntimeError:pass
            else:raise RuntimeError('failed overlay reused')
            row.update(status='passed',explicit_boundary_rejected=True,production_reuse_rejected=True,error_flag=error_flag)
        persist()


def run_probe(args,*,torch_module=None,lock_factory=None,backend_factory=None,
              workload=run_numerics,monitor_factory=base.ProcessMonitor,sleep=time.sleep,
              version_reader=importlib.metadata.version):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(PROTOCOL,status='starting',pid=os.getpid(),probe_nonce=uuid.uuid4().hex,
        started=time.time(),gpu=args.gpu,expected_gpu_uuid=args.expected_gpu_uuid,
        allowed_graphics=args.allow_graphics_pid,guards=[],cases=[],diagnostics=[],
        environment_before=base.environment(),allow_shared_validation=args.allow_shared_validation,
        resource_mode='shared_device_correctness_only' if args.allow_shared_validation else 'idle_device_numerical_gate')
    if args.allow_shared_validation:
        report.update(startup='flock -> two shared NVML/UUID/memory guards separated by 2s -> CUDA UUID check',
            minimum_free_memory_bytes=base.SHARED_MIN_FREE_BYTES,
            memory_guard_scope='engineering resource guard, not measured capacity or an allocation guarantee')
    def persist():base.save(out/'summary.json',report)
    persist();torch=torch_module;backend=None;monitor=None;old_visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        report['source_sha256']=source_receipt();persist()
        report['installed_triton_version']=version_reader('triton')
        report['compiler_contract']=base.compiler_contract(report['environment_before'],report['installed_triton_version'])
        if torch is None:import torch
        base.require(not torch.cuda.is_initialized(),'CUDA initialized before resource guard')
        report['startup_math_flags']=base.flags(torch)
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
                obs=base.probe_observation(backend,args);obs['index']=i;report['guards'].append(obs);persist()
                base.validate_observation(obs,args,startup=True)
                if i==0:sleep(2)
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);report['effective_environment']=base.environment()
            props=torch.cuda.get_device_properties(0)
            actual_uuid='GPU-'+str(uuid.UUID(str(props.uuid).removeprefix('GPU-')))
            base.require(actual_uuid==args.expected_gpu_uuid and (props.major,props.minor)==(12,0),'CUDA UUID/SM120 mismatch')
            report.update(torch=torch.__version__,cuda_device=dict(uuid=actual_uuid,name=props.name,capability=[12,0]),
                actual_math_flags_before=base.flags(torch))
            monitor=monitor_factory(backend,args,out/'process_observations.jsonl');monitor.start()
            with torch.cuda.device(0),torch.inference_mode():
                try:workload(torch,'cuda:0',report,persist,monitor.check)
                finally:torch.cuda.synchronize()
            validate_workload_report(report)
            report['process_monitor']=monitor.stop();monitor.check();monitor=None
            report['actual_math_flags_after']=base.flags(torch)
            base.require(report['actual_math_flags_after']==report['startup_math_flags'],'math flags changed')
            base.require(source_receipt()==report['source_sha256'],'source changed during probe')
            report.update(status='completed',numerical_gate_passed=True)
    except BaseException as exc:
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)));raise
    finally:
        if monitor is not None:
            try:report['process_monitor']=monitor.stop()
            except BaseException as exc:report['monitor_stop_error']=repr(exc)
        if backend is not None:
            try:backend.close()
            except BaseException as exc:report.update(status='failed',backend_close_error=repr(exc))
        if old_visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=old_visible
        if torch is not None and 'startup_math_flags' in report:report['exit_math_flags']=base.flags(torch)
        report.update(environment_after=base.environment(),finished=time.time())
        if report['environment_after']!=report['environment_before']:
            report.update(status='failed',environment_error='inherited environment changed')
        persist()
    base.require(report['status']=='completed','probe failed finalization')
    return report


if __name__=='__main__':run_probe(parse_args())
