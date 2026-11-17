"""Correctness-only compact mixed V GPU probe, independent from model runners.

The nine cases exercise full 28-layer setup, release old banks before fill, and
actual executor caches. Disposable .data corruption is diagnostic only. No
capacity benchmark, timing, DRAM, model or task-quality claim is made.
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
import probe_v_head_overlay_cuda as overlay_probe

HERE=Path(__file__).resolve().parent
BUDGETS=(0,14,112)
DIAGNOSTICS=('protected_old_payload_poison','invalid_compact_scale','persistent_prior_flag',
    'invalid_head_route','int32_min_head_route','prefix_points_to_generated',
    'nonfinite_original_copy','partial_coverage','duplicate_copy')
PROTOCOL=dict(base.PROTOCOL,schema='compact_mixed_visual_v_heads_probe_v1',
    numerical='three synthetic partitions times k0/14/112; exact compact bytes and old-overlay FP32 result',
    model_loaded=False,task_quality_claim=False,compact_implemented=True,
    source_retention='28 distinct actual inference tensors; old banks and original source released before fills',
    compilation='actual executor metadata and per-length keys; same-pointer replay reuse gate, no inferred zero-specialization claim',
    error_injection='disposable Tensor.data corruption only; never supported production mutation')


def source_receipt():
    receipt=overlay_probe.source_receipt()
    for name in ('v_compact_head_bank_cuda_ops.py','probe_v_compact_head_bank_cuda.py'):
        path=HERE/name;receipt[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return receipt


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--gpu',type=int,choices=range(8),required=True)
    p.add_argument('--expected-gpu-uuid',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-graphics-pid',type=int,choices=(6360,),action='append',default=[])
    p.add_argument('--allow-shared',dest='allow_shared_validation',action='store_true')
    a=p.parse_args(argv)
    try:
        if not a.expected_gpu_uuid.startswith('GPU-') or str(uuid.UUID(a.expected_gpu_uuid[4:]))!=a.expected_gpu_uuid[4:]:
            raise ValueError('canonical GPU-UUID required')
        if a.out.exists():raise ValueError('fresh output directory required')
    except ValueError as exc:p.error(str(exc))
    a.allow_graphics_pid=sorted(set(a.allow_graphics_pid));return a


def cpu_selection(torch,scores,k):
    values=scores.flatten().tolist()
    order=sorted(range(112),key=lambda i:(-values[i],i))
    route=torch.empty(112,dtype=torch.int32)
    for slot,i in enumerate(order[:k]):route[i]=slot
    for slot,i in enumerate(order[k:]):route[i]=-(slot+1)
    return route.view(28,4),order


def cpu_payload(torch,hosts,packs,positions,order,k):
    M=len(positions)
    original=torch.empty((k,M,128),dtype=torch.float16)
    codes=torch.empty((112-k,M,128),dtype=torch.int8)
    scales=torch.empty((112-k,M,4),dtype=torch.float32)
    for s,i in enumerate(order[:k]):original[s]=hosts[i//4][0,i%4,list(positions)]
    for s,i in enumerate(order[k:]):
        codes[s]=packs[i//4][0][0,i%4];scales[s]=packs[i//4][1][0,i%4]
    return codes,scales,original


EVIDENCE=('GPU_selection_byte_equal','codes_scales_byte_equal','original_FP16_byte_equal',
    'old_overlay_operand_byte_equal','all_28_copies_completed','unique_backing_matches',
    'source_inference_tensors','source_tensor_released','score_tensor_released',
    'old_banks_and_payload_released','original_flag_identities_preserved',
    'operand_not_retained','same_pointer_executor_reused')


def validate_workload_report(report):
    rows=report.get('cases',[]);expected={(c['name'],k) for c in base.cases() for k in BUDGETS}
    base.require(len(rows)==9 and {(r.get('case'),r.get('budget_heads')) for r in rows}==expected,'nine unique cases required')
    for r in rows:
        base.require(r.get('status')=='passed' and all(r.get(key) is True for key in EVIDENCE),'incomplete compact evidence')
        case=next(c for c in base.cases() if c['name']==r['case'])
        base.require([x['S'] for x in r.get('fills',[])]==[case['P']+e for e in case['extra']],'dynamic lengths incomplete')
        base.require(bool(r.get('copy_executor_cache')) and bool(r.get('fill_executor_cache')),'actual executors missing')
    ds=report.get('diagnostics',[])
    base.require(len(ds)==len(DIAGNOSTICS) and {r.get('kind') for r in ds}==set(DIAGNOSTICS)
        and all(r.get('status')=='passed' for r in ds),'diagnostics incomplete')
    for r in ds:
        if r['kind']=='protected_old_payload_poison':base.require(r.get('operand_byte_equal') is True,'poison result missing')
        elif r['kind'] in ('nonfinite_original_copy','partial_coverage','duplicate_copy'):
            base.require(r.get('setup_rejected') is True,'bad setup accepted')
        else:base.require(r.get('boundary_rejected') is True and r.get('reuse_rejected') is True
                         and type(r.get('error_flag')) is int and r['error_flag']!=0,'invalid fill accepted')


def run_numerics(torch,device,report,persist,process_check):
    import triton
    import v_compact_head_bank_cuda_ops as ops
    base.require(triton.__version__==base.TRITON_VERSION,'imported Triton must be 3.4.0')
    copy,fill=ops._get_kernels()
    base.require(not base.jit_cache_receipt(copy) and not base.jit_cache_receipt(fill),'fresh compact executor caches required')
    report['triton_version']=triton.__version__
    for case in base.cases():
        for k in BUDGETS:
            process_check()
            route,sources,hosts,banks,texts,packs,scores,cpu_scores,score_check=overlay_probe.make_fixture(torch,device,case)
            source_refs=[weakref.ref(x) for x in sources];score_ref=weakref.ref(scores)
            old_refs=[weakref.ref(x) for b in banks for x in (b,b.visual_values,b.scales)]
            flag_ids=tuple(b.error_flag for b in banks)
            signed,order=cpu_selection(torch,cpu_scores,k)
            cpu_codes,cpu_scales,cpu_fp16=cpu_payload(torch,hosts,packs,case['visual'],order,k)
            row=dict(case=case['name'],budget_heads=k,status='starting',fills=[],pack_score_comparison=score_check)
            report['cases'].append(row);persist()
            owner=ops.prepare_compact_head_bank(scores,route,budget_heads=k)
            row['route_comparison']=base.byte_check(torch,owner.head_route,signed,'signed GPU head route')
            row['copy_executor_before']=base.jit_cache_receipt(copy)
            for layer in range(28):ops.copy_compact_head_layer_(owner,layer,sources[layer],banks[layer])
            ops.validate_compact_ready(owner,banks)
            row['codes_comparison']=base.byte_check(torch,owner.codes,cpu_codes,'compact unchanged codes')
            row['scales_comparison']=base.byte_check(torch,owner.scales,cpu_scales,'compact unchanged scales')
            row['FP16_comparison']=base.byte_check(torch,owner.original_fp16,cpu_fp16,'original FP16 bits')
            storage=owner.storage_receipt()
            roots=[owner.codes,owner.scales,owner.original_fp16,owner.head_route]
            unique={str(x.device)+':'+str(x.untyped_storage().data_ptr()):x.untyped_storage().nbytes() for x in roots if x.numel()}
            base.require(sum(unique.values())==storage['actual_visual_payload_route_unique_backing_bytes']==storage['persistent_visual_payload_bytes'],'unique backing mismatch')
            base.require(storage['actual_existing_error_flags_unique_backing_bytes']==112 and storage['completed_layer_copies']==28,'flag/copy coverage')
            base.require(storage['preparation']['inference_source_layer_count']==28,'actual inference sources required')
            views=[owner.layer_view(l) for l in range(28)]
            base.require(all(v.error_flag is flag_ids[l] for l,v in enumerate(views)),'original flags changed')
            del sources,banks,scores,roots,texts;gc.collect();torch.cuda.synchronize()
            base.require(all(r() is None for r in old_refs+source_refs) and score_ref() is None,'old payload/source/score retained')
            base.require(storage['old_bank_weak_refs_present']==0,'old bank weak refs not cleared')
            # All fills below occur after old banks/codes/scales/source disappear.
            maxextra=max(case['extra']);host_text=[];gpu_text=[]
            for layer in range(28):
                value=base.cpu_values(torch,route.R+maxextra+3,offset=layer+59)
                value[:,:,:route.R].copy_(packs[layer][2]);host_text.append(value)
                gpu_text.append(base._cuda_strided(torch,value,device))
            for extra in case['extra']:
                S=route.P+extra;fr=dict(S=S,comparisons=[],executor_keys_before=base.jit_cache_receipt(fill))
                backing=torch.empty((1,4,S*2+2,128),device=device,dtype=torch.float32)
                out=backing[:,:,1:1+2*S:2];out_ref=weakref.ref(out)
                for layer in range(28):
                    current=gpu_text[layer][:,:,:route.R+extra]
                    returned=ops.fill_compact_value_fp32_(out,views[layer],current)
                    base.require(returned is out,'output replaced')
                    expected=overlay_probe.cpu_expected(torch,hosts[layer],case['visual'],packs[layer][0],packs[layer][1],
                        host_text[layer][:,:,:route.R+extra],signed,layer)
                    fr['comparisons'].append(base.byte_check(torch,out,expected,'old-overlay FP32 layer '+str(layer)))
                ops.check_compact_status(owner,views,stage='probe_after_28_fills')
                before=base.jit_cache_receipt(fill)
                ops.fill_compact_value_fp32_(out,views[-1],current)
                ops.check_compact_status(owner,views,stage='same_pointer_replay')
                after=base.jit_cache_receipt(fill)
                base.require(before==after,'same-pointer replay created an executor')
                fr.update(executor_keys_after=after,same_pointer_replay_keys_before=before,same_pointer_reused=True)
                del out,returned,backing,current;gc.collect();base.require(out_ref() is None,'operand retained')
                row['fills'].append(fr);persist();process_check()
            row.update({key:True for key in EVIDENCE})
            row.update(status='passed',storage=storage,copy_executor_cache=base.jit_cache_receipt(copy),
                fill_executor_cache=base.jit_cache_receipt(fill))
            del owner,views,gpu_text,flag_ids,route;persist()
    run_diagnostics(torch,device,report,persist,process_check)


def run_diagnostics(torch,device,report,persist,process_check):
    import v_compact_head_bank_cuda_ops as ops
    case=base.cases()[0]
    for kind in DIAGNOSTICS:
        process_check();row=dict(kind=kind,status='starting',disposable_diagnostic_only=True)
        report['diagnostics'].append(row);persist()
        route,sources,hosts,banks,texts,packs,scores,cpu_scores,_=overlay_probe.make_fixture(torch,device,case)
        k=112 if kind in ('protected_old_payload_poison','nonfinite_original_copy') else 0
        owner=ops.prepare_compact_head_bank(scores,route,budget_heads=k)
        if kind=='protected_old_payload_poison':
            banks[0].visual_values.data.fill_(-128);banks[0].scales.data.fill_(float('nan'))
        if kind=='nonfinite_original_copy':sources[0].data[0,0,case['visual'][0],0]=float('inf')
        for layer in range(27 if kind=='partial_coverage' else 28):
            ops.copy_compact_head_layer_(owner,layer,sources[layer],banks[layer])
        if kind=='duplicate_copy':
            try:ops.copy_compact_head_layer_(owner,0,sources[0],banks[0])
            except (RuntimeError,ValueError):pass
            else:raise RuntimeError('duplicate copy accepted')
            row.update(status='passed',setup_rejected=True);persist();continue
        if kind in ('nonfinite_original_copy','partial_coverage'):
            try:ops.validate_compact_ready(owner,banks)
            except (RuntimeError,ValueError):pass
            else:raise RuntimeError('invalid/partial setup accepted')
            base.require(not owner.validated,'bad setup marked ready')
            row.update(status='passed',setup_rejected=True);persist();continue
        ops.validate_compact_ready(owner,banks);view=owner.layer_view(0)
        extra=1 if kind=='prefix_points_to_generated' else 0
        current=torch.cat((texts[0],torch.zeros(1,4,extra,128,dtype=torch.float16,device=device)),dim=2)
        out=torch.empty(1,4,route.P+extra,128,dtype=torch.float32,device=device)
        if kind=='invalid_compact_scale':
            s=int(owner.head_route[0,0].item());owner.scales.data[-s-1,0,0]=float('nan')
        elif kind=='persistent_prior_flag':owner.error_flags[0].fill_(4)
        elif kind=='invalid_head_route':owner.head_route.data[0,0]=112
        elif kind=='int32_min_head_route':owner.head_route.data[0,0]=-2147483648
        elif kind=='prefix_points_to_generated':
            route.prefix_route.data[case['nonvisual'][0] if 'nonvisual' in case else route.partition.nonvisual[0]]=-(route.R+1)
        ops.fill_compact_value_fp32_(out,view,current)
        if kind=='protected_old_payload_poison':
            ops.check_compact_status(owner,stage='protected_discarded_old_payload')
            row.update(status='passed',operand_byte_equal=base.byte_check(torch,out,hosts[0].float(),'protected original') ['byte_equal'],
                interpretation='numerical independence from discarded old payload, not a DRAM counter')
        else:
            flag=int(owner.error_flags[0].item())
            try:ops.check_compact_status(owner,stage='disposable_corruption')
            except (RuntimeError,ValueError):pass
            else:raise RuntimeError('device error accepted')
            base.require(flag!=0 and owner.failed,'persistent device error missing')
            # No repair/reset is permitted in production; owner poison survives.
            owner.error_flags[0].zero_()
            try:ops.fill_compact_value_fp32_(out,view,current)
            except (RuntimeError,ValueError):pass
            else:raise RuntimeError('failed owner reused')
            row.update(status='passed',boundary_rejected=True,reuse_rejected=True,error_flag=flag)
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
