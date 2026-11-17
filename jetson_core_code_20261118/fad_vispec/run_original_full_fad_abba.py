"""Ordinary E24 native ViSpec adaptation versus complete FAD, four processes A/B/B/A.

A loads native Target/author Draft directly. B calls the frozen full generation.
Each child has one first-request warmup and two complete E24 repeats. Request
hashing and artifact writes are separate diagnostics. Output/precision differences
remain raw; only completed work enters the package TPS. Root owns GPU admission.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import dataclasses
import gc
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:sys.path.insert(0,str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import run_full_fad as full
pair,guard,g0=full.pair,full.guard,full.g0
require,save,digest,read_json,jsonl=full.require,full.save,full.digest,full.read_json,full.jsonl
STAGES=('00_original','01_full','02_full','03_original')
ARMS=('original','full','full','original')
SCHEMA='fad_original_full_ordinary_ABBA_v1'
CHILD_SCHEMA='fad_original_full_ordinary_child_v1'
REAL_KIND='actual_original_full_FAD_ordinary'
ENV_KEYS=('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
    'CUBLAS_WORKSPACE_CONFIG','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF','PYTHONPATH','OMP_NUM_THREADS',
    'MKL_NUM_THREADS','TOKENIZERS_PARALLELISM','PYTHONUNBUFFERED')


def schedule(requests,seed=20261001):
    require(requests and len({r['request_id'] for r in requests})==len(requests),'nonempty unique requests required')
    cells=[dict(request_id=requests[0]['request_id'],phase='warmup',repeat=0)]
    rng=random.Random(seed)
    for repeat in range(2):
        order=list(requests);rng.shuffle(order)
        cells.extend(dict(request_id=r['request_id'],phase='measure',repeat=repeat) for r in order)
    return cells


def load_original_models(bridge,budget_factory,reference):
    """Native construction only: no optimized parent constructor or undo path."""
    target=bridge.load_target(reference['status']['target'],'cuda:0',dtype='float16',
        output_head_dtype='float32',target_sdpa_backend='math')
    spec=bridge.load_spec(target,reference['status']['draft'],total_token=48,depth=3,top_k=8,num_q=2,capacity=4096)
    for model in (target,spec.spec_layer):
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
    left=target.get_input_embeddings().weight;right=spec.spec_layer.embed_tokens.weight
    require(left is not right and left.untyped_storage().data_ptr()!=right.untyped_storage().data_ptr(),
        'native Draft input embedding must retain independent backing')
    budget=budget_factory(30,3,8)
    metadata=dict(target=reference['status']['target'],draft=reference['status']['draft'],load_receipt=spec._vispec_load_receipt,
        input_embedding='native_independent',adaptor='native_author_imadpt',target_attention='native_math',
        input_embedding_storage_bytes=[left.untyped_storage().nbytes(),right.untyped_storage().nbytes()],
        original_adaptor_class=type(spec.spec_layer.imadpt).__module__+'.'+type(spec.spec_layer.imadpt).__qualname__,
        output_head_dtype=str(target.get_output_embeddings().weight.dtype),
        shared_embedding_installed=False,compiled_adaptor_installed=False,grouped_attention_installed=False,
        no_optimization_undo_path=True,extra_model_weights=False,tree_budget=dict(B=30,d=3,K=8,total_tokens=29))
    return target,spec,budget,metadata


class NativeOriginalOps:
    execution_kind=REAL_KIND
    cleanup_request=pair.PairOps.cleanup_request
    upload=pair.PairOps.upload
    result_row=g0.RealOps.result_row
    def __init__(self,args,reference):
        import torch
        import cv2
        import transformers
        import hf517_vispec_adapter as bridge
        import vispec_joint_adapter as joint
        from vispec_operators import new_tree_budget
        from run_original_grouped_timing import read_flags
        self.torch,self.bridge,self.joint,self.args=torch,bridge,joint,args
        bridge._native()
        from jev_spec.runtime import prepare_request
        self.prepare_request=prepare_request;self.flags=lambda:read_flags(torch)
        torch.set_num_threads(1);cv2.setNumThreads(1)
        self.target,self.spec,self.budget,self.metadata=load_original_models(bridge,new_tree_budget,reference)
        self.processor=self.target._hf517_processor
        self.metadata.update(torch=torch.__version__,transformers=transformers.__version__,device_name=torch.cuda.get_device_name(0),
            flags=self.flags(),public_environment={k:os.environ.get(k) for k in ENV_KEYS},execution_kind=REAL_KIND,
            arm='original',controller_installed=False,StageTimes_hooks_installed=False,
            native_joint_timing_events='Unchanged native generate_joint may retain its two events; no elapsed_time read or added stage synchronization.')
    def prepare_cpu(self,request):return self.prepare_request(self.processor,request,'cpu')
    def signature(self,packed,timestamps):return self.bridge.tensor_signature(packed,timestamps)
    def generate(self,packed):
        return self.joint.generate_joint(self.spec,packed,max_new_tokens=512,visual_m=1.0,tree_budget=self.budget,
            merge_operator='cosine',context_extractor='new',diagnostic_actual_imadpt=False,eos_token_ids=[151645,151643])


def run_original_generation(ops,packed,request,observations):
    obs=observations;obs.update(flags=dict(before=ops.flags()),prefill=dict(calls=0,wrapper_restored=False),
        controller_installed=False,native_joint=True,StageTimes_hooks_installed=False)
    result=None;ops.torch.cuda.synchronize();start=time.perf_counter()
    try:
        with full.observe_prefill(ops.bridge,obs['prefill']):result=ops.generate(packed)
    finally:
        obs['flags']['after']=ops.flags();ops.torch.cuda.synchronize();obs['generation_wall_s']=time.perf_counter()-start
    require(obs['prefill']==dict(calls=1,wrapper_restored=True),'original prefill wrapper did not restore')
    t=time.perf_counter();row=ops.result_row(result);result=None
    obs['output_decode_wall_s']=time.perf_counter()-t
    return row


def validate_native(row,obs):
    r=row['result'];counts=r['forward_counts'];trees=[e for e in r['rounds'] if e['kind']=='tree'];n=len(trees)
    require(set(counts)==set(guard.CORE_COUNTERS),'native public counter keys differ')
    require(counts['target_prefill_calls']==1 and counts['target_verify_calls']==counts['draft_tree_calls']
        ==counts['target_compaction_calls']==counts['budget_switch_calls']==n
        and counts['draft_model_forward_calls']==4*n and counts['target_verify_query_tokens']==30*n,'native D4 workload differs')
    require(all(e['tree_nodes']==30 and e['budget']==dict(B=30,d=3,K=8,total_tokens=29)
        and e['visual_m']==1.0 for e in trees),'native fixed/full visual recipe differs')
    require(obs['flags']['before']==obs['flags']['after'] and obs['native_joint'] and not obs['controller_installed'],
        'original flags/routing differs within generation')
    return dict(passed=True,tree_rounds=n,actual_depths=[4]*n)


def validate_output(row):
    r=row['result'];ids=r['token_ids'];eos={151643,151645}
    require(ids and len(ids)<=512 and all(type(v) is int and v>=0 for v in ids),'invalid complete output IDs')
    require(ids==row['output_token_ids'] and len(ids)==row['num_tokens']==r['output_length']
        and r['stop_reason']==row['stop_reason'] and r['cache_checks'] is True,'public output/cache fields differ')
    require(not any(v in eos for v in ids[:-1]) and ((r['stop_reason']=='eos' and ids[-1] in eos and not row['truncated'])
        or (r['stop_reason']=='limit' and len(ids)==512 and ids[-1] not in eos and row['truncated'])),'natural EOS/512 limit required')
    require([v for e in r['rounds'] for v in e['submitted']]==ids,'all rounds must reconstruct output')
    require(all(type(v) is int and v>=0 for v in r['forward_counts'].values()),'invalid public counts')


def run_case(cell,request,saved,out,ops,backend,arm):
    """Same prepare/upload/generation/decode/release boundaries for both arms."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);obs={};cpu=packed=sampled=monitor=None;start=None
    case=dict(cell,arm=arm,dataset=request['dataset'],completed=False,execution_kind=ops.execution_kind,
        paired_total_memory_accepted=None,quality_pass=None,ordinary_TPS_benefit_claim=False)
    clock=time.perf_counter();stamp=time.time()
    try:
        monitor=guard.ProcessMonitor(backend).start();ops.cleanup_request()
        ops.torch.cuda.reset_peak_memory_stats();start=full.allocator(ops.torch.cuda);stamp=time.time()
        case['resident_before']=pair.resident(ops)
        t=time.perf_counter();cpu,sampled=ops.prepare_cpu(request);case['CPU_prepare_wall_s']=time.perf_counter()-t
        t=time.perf_counter();case['input']=pair.input_record(ops,cpu,sampled);case['CPU_input_signature_diagnostic_wall_s']=time.perf_counter()-t
        case['input_exact']=all(case['input'][k]==saved['input'][k] for k in guard.INPUT_KEYS)
        require(case['input_exact'],'CPU input differs from frozen E24 C0 input')
        t=time.perf_counter();packed=ops.upload(cpu);case['GPU_input_materialization_wall_s']=time.perf_counter()-t
        if arm=='original':row=run_original_generation(ops,packed,request,obs)
        else:row=full.run_generation(ops,packed,request,mode='full',controller=ops.controller,observations=obs)
        case['output']=row;validate_output(row)
        case['execution_check']=validate_native(row,obs) if arm=='original' else full.validate_execution(row,obs,request['request_id'],'full')
        case['C0_auxiliary_comparison']=g0.compare_c0(row,saved['row'])
        case['completed']=True
    except BaseException as exc:case.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        t=time.perf_counter()
        try:ops.cleanup_request()
        except BaseException as exc:case.update(completed=False,cleanup_error=repr(exc))
        finally:packed=cpu=sampled=None
        try:
            gc.collect();ops.torch.cuda.synchronize();case['resident_after']=pair.resident(ops)
            pair.validate_resident(case['resident_after'],case.get('resident_before'))
        except BaseException as exc:case.update(completed=False,cleanup_error=repr(exc))
        case['request_cleanup_and_release_wall_s']=time.perf_counter()-t
        try:
            end=full.allocator(ops.torch.cuda)
            case['memory']=dict(start=start,end=end,whole_observed={f'peak_{k}_bytes':max(start.get(f'peak_{k}_bytes',0) if start else 0,end[f'peak_{k}_bytes'],end[f'{k}_bytes']) for k in ('allocated','reserved')},
                started_unix=stamp,ended_unix=time.time(),resets_within_request=0,empty_cache_used=False,model_loading_excluded=True)
        except BaseException as exc:case.update(completed=False,allocator_error=repr(exc))
        try:
            case['background']=monitor.stop() if monitor is not None else dict(unavailable=True)
            if case.get('memory'):
                m=case['memory'];case['background_review']=guard.background_review(case['background'],m['started_unix'],m['ended_unix'],ops.args.expected_gpu_uuid,ops.args.allow_graphics_pid,os.getpid())
        except BaseException as exc:case.update(completed=False,background_error=repr(exc))
        case['observations']=obs;case['raw_instrumented_request_wall_s']=time.perf_counter()-clock
        case['ordinary_complete_request_wall_s']=sum(case.get(k,0) for k in ('CPU_prepare_wall_s','GPU_input_materialization_wall_s','request_cleanup_and_release_wall_s'))+obs.get('generation_wall_s',0)+obs.get('output_decode_wall_s',0)
        case['timing_scope']='Complete generation includes method selector/feedback/setup/finalization; CPU prepare/upload/decode/release separately recorded. CPU input signature and disk writes excluded from ordinary time.'
        save(out/'input.json',case.get('input',{}));save(out/'observations.json',obs);save(out/'production.json',case.get('output',{}))
        case['artifact_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file()};save(out/'case.json',case)
    return case


def aggregate(cases,cells,arm):
    complete=bool(cells) and len(cases)==len(cells) and all({k:c[k] for k in ('request_id','phase','repeat')}==s and c['arm']==arm
        and c.get('completed') is True and c.get('execution_kind')==REAL_KIND and c.get('execution_check',{}).get('passed')
        and c.get('input_exact') for c,s in zip(cases,cells))
    formal=[c for c in cases if c['phase']=='measure'];tokens=sum(c.get('output',{}).get('num_tokens',0) for c in formal)
    gen=sum(c.get('observations',{}).get('generation_wall_s',0) for c in formal);e2e=sum(c.get('ordinary_complete_request_wall_s',0) for c in formal)
    return dict(mechanical_completed=complete,formal_requests=len(formal),formal_tokens=tokens,
        generation_wall_s=gen,generation_TPS=tokens/gen if complete and gen>0 else None,
        complete_request_wall_s=e2e,complete_request_TPS=tokens/e2e if complete and e2e>0 else None,
        quality_pass=None,precision_matched=None,background_eligible=complete and all(c.get('background_review',{}).get('eligible') is True for c in formal),
        paired_total_memory_accepted=None,ordinary_TPS_benefit_claim=False)


def functional_gate(path,expected):
    require(digest(path)==expected,'functional GPU report SHA differs');r=read_json(path)
    require(r['schema']==full.SCHEMA and r['status']=='completed' and r['args']['mode']=='full'
        and r['source_recheck']['passed'] is True and r['gate']['complete_E24_passed'] is True
        and len(r['cases'])==24 and all(c['completed'] and c['execution_kind']==full.REAL_KIND for c in r['cases']),
        'actual complete full-FAD E24 functional GPU gate required')
    require(r['entry_sha256']==digest(full.__file__),'functional GPU gate used a different full runner')
    old_project=Path(r['execution_source_freeze']['project'])
    for p,h in r['execution_source_freeze']['source_sha256'].items():
        relative=Path(p).relative_to(old_project)
        require(digest(PROJECT/relative)==h,'functional GPU gate execution dependency differs: '+str(relative))
    return dict(path=str(Path(path).resolve()),sha256=expected,pid=r['pid'],mode='full',complete_E24=True)


def compare_runs(runs):
    """All raw work, two adjacent ABBA pairings; no answer or timing deletion."""
    require(len(runs)==4 and [r['arm'] for r in runs]==list(ARMS),'four ABBA runs required')
    pairs=[]
    for ai,bi in ((0,1),(3,2)):
        a=[c for c in runs[ai]['cases'] if c['phase']=='measure'];b=[c for c in runs[bi]['cases'] if c['phase']=='measure']
        require(len(a)==len(b) and a,'complete matched formal workload required')
        for x,y in zip(a,b):
            require(all(x[k]==y[k] for k in ('request_id','phase','repeat')),'paired formal cell differs')
            ar,br=x['output'],y['output'];ac,bc=ar['result'],br['result']
            ids_equal=ac['token_ids']==bc['token_ids']
            mismatch=next((i for i in range(max(len(ac['token_ids']),len(bc['token_ids'])))
                if (ac['token_ids'][i] if i<len(ac['token_ids']) else None)!=(bc['token_ids'][i] if i<len(bc['token_ids']) else None)),None)
            pairs.append(dict(request_id=x['request_id'],repeat=x['repeat'],A_stage=STAGES[ai],B_stage=STAGES[bi],
                token_ids_equal=ids_equal,first_differing_token_index=mismatch,stop_equal=ac['stop_reason']==bc['stop_reason'],
                text_equal=ar['text']==br['text'],length_ratio=br['num_tokens']/ar['num_tokens'],
                output_exact=ids_equal and ac['stop_reason']==bc['stop_reason'] and ar['text']==br['text'] and ar['truncated']==br['truncated'],
                full_rounds_common_counts_equal=ac['rounds']==bc['rounds'] and all(ac['forward_counts'][k]==bc['forward_counts'][k] for k in guard.CORE_COUNTERS),
                precision_match=x['observations']['flags']==y['observations']['flags'],
                input_exact=all(x['input'][k]==y['input'][k] for k in guard.INPUT_KEYS),
                background_eligible=x.get('background_review',{}).get('eligible') is True and y.get('background_review',{}).get('eligible') is True,
                A_tokens=ar['num_tokens'],B_tokens=br['num_tokens'],A_generation_s=x['observations']['generation_wall_s'],B_generation_s=y['observations']['generation_wall_s'],
                A_complete_request_s=x['ordinary_complete_request_wall_s'],B_complete_request_s=y['ordinary_complete_request_wall_s'],
                A_peak=x['memory']['whole_observed'],B_peak=y['memory']['whole_observed'],
                memory_nonincrease=all(y['memory']['whole_observed'][k]<=x['memory']['whole_observed'][k] for k in ('peak_allocated_bytes','peak_reserved_bytes'))))
    def rates(rows):
        result={}
        for arm in ('A','B'):
            tokens=sum(r[arm+'_tokens'] for r in rows);gen=sum(r[arm+'_generation_s'] for r in rows);e2e=sum(r[arm+'_complete_request_s'] for r in rows)
            result[arm]=dict(tokens=tokens,generation_seconds=gen,generation_TPS=tokens/gen if gen>0 else None,
                complete_request_seconds=e2e,complete_request_TPS=tokens/e2e if e2e>0 else None)
        result['generation_TPS_ratio_B_over_A']=result['B']['generation_TPS']/result['A']['generation_TPS'] if rows and result['A']['generation_TPS'] and result['B']['generation_TPS'] else None
        return result
    batch={arm:{k:max(p[arm+'_peak'][k] for p in pairs) for k in ('peak_allocated_bytes','peak_reserved_bytes')} for arm in ('A','B')}
    return dict(pairs=pairs,all_package=rates(pairs),same_output_subset=rates([p for p in pairs if p['output_exact']]),
        same_complete_rounds_common_counts_subset=rates([p for p in pairs if p['full_rounds_common_counts_equal']]),
        all_output_exact=all(p['output_exact'] for p in pairs),all_precision_matched=all(p['precision_match'] for p in pairs),
        all_inputs_exact=all(p['input_exact'] for p in pairs),all_background_eligible=all(p['background_eligible'] for p in pairs),
        batch_maxima=batch,per_pair_and_batch_memory_nonincrease=all(p['memory_nonincrease'] for p in pairs)
            and all(batch['B'][k]<=batch['A'][k] for k in batch['A']),
        quality_fact_review_pending=True,ordinary_TPS_benefit_claim=False,
        scope='All-package rates are raw ordinary timing; output, precision, work and background differences remain explicit. No exact fidelity or pure backend benefit follows from package speed alone.')


def bindings(args):
    execution=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
    require(digest(args.requests)==g0.E24_SHA,'frozen E24 bytes differ');requests=jsonl(args.requests);full.neutral.aggregate([],requests)
    manifest_path=Path(args.requests).parent/'MANIFEST.json';require(digest(manifest_path)==g0.MANIFEST_SHA,'manifest differs')
    manifest=read_json(manifest_path);reference=g0.load_reference(args.c0_dir,requests)
    c0_project=g0.resolve_c0_project(args.c0_dir,args.c0_project);sources=g0.check_sources(reference,manifest,c0_project=c0_project)
    sources.update(execution['source_sha256'])
    policy={n:digest(getattr(args,n)) for n in ('calibration','controller_config')}
    for n,h in policy.items():require(h==getattr(args,n+'_sha256'),'policy input bytes differ')
    proof=functional_gate(args.functional_report,args.functional_report_sha256)
    if args.request_id:
        require(len(set(args.request_id))==len(args.request_id) and set(args.request_id)<={r['request_id'] for r in requests},'pilot IDs must be unique E24 members')
        requests=[r for r in requests if r['request_id'] in args.request_id]
    return dict(execution=execution,reference=reference,source_sha256=sources,policy_input_sha256=policy,
        functional_GPU_gate=proof,requests=requests,schedule=schedule(requests),public_environment={k:os.environ.get(k) for k in ENV_KEYS})


def child_command(args,out,arm):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--child-arm',arm,'--out',str(out),'--physical-gpu',str(args.physical_gpu)]
    for n in ('requests','requests_sha256','c0_dir','expected_gpu_uuid','execution_source_freeze','execution_source_freeze_sha256',
        'calibration','calibration_sha256','controller_config','controller_config_sha256','functional_report','functional_report_sha256','policy_phase'):
        cmd.extend(['--'+n.replace('_','-'),str(getattr(args,n))])
    for n in ('c0_project','frozen_policy_sha256'):
        if getattr(args,n):cmd.extend(['--'+n.replace('_','-'),str(getattr(args,n))])
    for pid in args.allow_graphics_pid:cmd.extend(['--allow-graphics-pid',str(pid)])
    for rid in args.request_id or []:cmd.extend(['--request-id',rid])
    return cmd


def run_child(args,initial):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);cells=initial['schedule'];requests={r['request_id']:r for r in initial['requests']}
    report=dict(schema=CHILD_SCHEMA,status='starting',pid=os.getpid(),arm=args.child_arm,schedule=cells,cases=[],bindings=initial,
        prelaunch=[],source_recheck=dict(passed=False),startup_flags=None,exit_flags=None,loading_memory=None)
    def persist():
        report['gate']=aggregate(report['cases'],cells,args.child_arm)
        if report['status']!='completed' or not report['source_recheck']['passed']:report['gate']['mechanical_completed']=False
        save(out/'ORDINARY_REPORT.json',report)
    persist();failure=None;backend=None;previous={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','MKL_NUM_THREADS')}
    lock,receipt=g0.load_lock_without_torch(PROJECT/'direct_opt/run_joint.py');report['lock_source']=receipt
    try:
        for p in (PROJECT,PROJECT/'vendor',PROJECT/'direct_opt',PROJECT/'training_free'):sys.path.insert(0,str(p))
        from hardware_metrics import _make_backend
        with lock(args.physical_gpu,out) as metadata:
            report['lock']=metadata;backend=_make_backend(args.physical_gpu)
            guard.prelaunch_checks(backend,args,report['prelaunch'],persist=persist)
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            import torch
            report['cuda_device']=guard.bind_cuda_device(torch,args)
            from run_original_grouped_timing import read_flags
            report['startup_flags']=read_flags(torch)
            if args.child_arm=='full':
                from grouped_gqa_runtime import math_precision_scope
                scope=math_precision_scope()
            else:scope=nullcontext()
            with scope,torch.inference_mode():
                torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                ops=NativeOriginalOps(args,initial['reference']) if args.child_arm=='original' else full.ModelOps(args,initial['reference'])
                ops.execution_kind=REAL_KIND
                torch.cuda.synchronize();report['loading_memory']=full.allocator(torch.cuda)
                report.update(status='running',setup=ops.metadata,load_wall_s=time.perf_counter()-t);persist()
                for index,cell in enumerate(cells):
                    observation=guard.observe_processes(backend);report.setdefault('request_guards',[]).append(observation)
                    guard.verify_idle(observation,args.expected_gpu_uuid,args.allow_graphics_pid,os.getpid())
                    report['current']=dict(index=index,**cell);persist();request=requests[cell['request_id']]
                    case=run_case(cell,request,initial['reference']['selected'][cell['request_id']],out/'cases'/f'{index:03d}',ops,backend,args.child_arm)
                    report['cases'].append(case);guard.append(out/'cases.jsonl',case);persist()
                    require(case['completed'],'ordinary generation failed; raw retained, no retry')
                report.update(status='completed',current=None)
            report['exit_flags']=read_flags(torch);require(report['exit_flags']==report['startup_flags'],'startup precision flags not restored')
    except BaseException as exc:failure=exc;report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as exc:report.update(status='failed',backend_close_error=repr(exc));failure=failure or exc
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        try:require(bindings(args)==initial,'source/policy/request/functional proof drift');report['source_recheck']=dict(passed=True)
        except BaseException as exc:report.update(status='failed',source_recheck=dict(passed=False,error=repr(exc)));failure=failure or exc
        report['ended_unix']=time.time();persist()
    if failure is not None:raise failure
    require(report['gate']['mechanical_completed'],'incomplete ordinary child');return report


def run_campaign(args,initial,*,make_backend,popen=subprocess.Popen,sleep=time.sleep):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status='starting',pid=os.getpid(),stages=list(STAGES),bindings=initial,launches=[],completed=[],guards=[],
        quality_pass=None,ordinary_TPS_benefit_claim=False,paired_total_memory_accepted=None)
    persist=lambda:save(out/'ABBA_REPORT.json',report);persist();backend=None;failure=None;runs=[]
    try:
        backend=make_backend(args.physical_gpu)
        for stage,arm in zip(STAGES,ARMS):
            require(bindings(args)==initial,'source/env/request drift before launch')
            for i in range(2):
                observation=guard.observe_processes(backend);report['guards'].append(dict(stage=stage,index=i,**observation));persist()
                guard.verify_idle(observation,args.expected_gpu_uuid,args.allow_graphics_pid,minimum_free_bytes=guard.MINIMUM_FREE_BYTES)
                if i==0:sleep(2)
            require(bindings(args)==initial,'source/env/request drift during idle polls')
            command=child_command(args,out/stage,arm);launch=dict(stage=stage,arm=arm,command=command,pid=None,returncode=None,started_unix=time.time())
            report['launches'].append(launch);report.update(status='running',current=stage);persist()
            with (out/(stage+'.log')).open('w') as stream:
                child=popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=str(PROJECT),env=dict(os.environ),start_new_session=True)
                launch['pid']=child.pid;persist();launch['returncode']=child.wait();launch['ended_unix']=time.time();persist()
            require(launch['returncode']==0,'child nonzero; no retry/resume')
            require(bindings(args)==initial,'source/env/request drift after child')
            r=read_json(out/stage/'ORDINARY_REPORT.json')
            require(r['schema']==CHILD_SCHEMA and r['status']=='completed' and r['pid']==child.pid and r['arm']==arm
                and r['schedule']==initial['schedule'] and r['bindings']==initial and r['source_recheck']['passed']
                and r['startup_flags']==r['exit_flags'] and r['cuda_device']['physical']==args.physical_gpu
                and r['cuda_device']['uuid']==args.expected_gpu_uuid
                and r['gate']==aggregate(r['cases'],initial['schedule'],arm) and r['gate']['mechanical_completed'],'child completion/PID/raw schedule differs')
            for index,case in enumerate(r['cases']):
                case_dir=out/stage/'cases'/f'{index:03d}'
                require(read_json(case_dir/'case.json')==case and set(case['artifact_sha256'])=={'input.json','observations.json','production.json'},'child raw case differs')
                require(all(digest(case_dir/n)==h for n,h in case['artifact_sha256'].items()),'child raw artifact hash differs')
            require(child.pid not in [p['pid'] for p in report['completed']],'four distinct child PIDs required')
            runs.append(r)
            report['completed'].append(dict(stage=stage,arm=arm,pid=child.pid,report_sha256=digest(out/stage/'ORDINARY_REPORT.json')));persist()
        report.update(status='completed',current=None,comparison=compare_runs(runs))
    except BaseException as exc:failure=exc;report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as exc:report.update(status='failed',backend_close_error=repr(exc));failure=failure or exc
        report['ended_unix']=time.time();persist()
    if failure is not None:raise failure
    return report


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('out','requests','requests-sha256','c0-dir','expected-gpu-uuid','execution-source-freeze','execution-source-freeze-sha256',
        'calibration','calibration-sha256','functional-report','functional-report-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--controller-config','--config',dest='controller_config',required=True)
    p.add_argument('--controller-config-sha256','--config-sha256',dest='controller_config_sha256',required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--allow-graphics-pid',type=int,action='append',default=[]);p.add_argument('--c0-project');p.add_argument('--request-id',action='append')
    p.add_argument('--child-arm',choices=('original','full'));p.add_argument('--policy-phase',choices=('calibration_A','holdout_B','test_T200'),default='calibration_A')
    p.add_argument('--frozen-policy-sha256');a=p.parse_args(argv)
    # Reuse the actual full CLI validation with a full mode, without creating files.
    a.mode='full';check=vars(a).copy();check.pop('child_arm');check.pop('functional_report');check.pop('functional_report_sha256')
    forwarded=[]
    for n,v in check.items():
        if v is None:continue
        if isinstance(v,list):
            for item in v:forwarded+=['--'+n.replace('_','-'),str(item)]
        else:forwarded+=['--'+n.replace('_','-'),str(v)]
    full.parse_args(forwarded)
    return a


def main(argv=None):
    args=parse_args(argv);initial=bindings(args)
    if args.child_arm:return run_child(args,initial)
    for p in (PROJECT,PROJECT/'vendor',PROJECT/'direct_opt'):sys.path.insert(0,str(p))
    from hardware_metrics import _make_backend
    return run_campaign(args,initial,make_backend=_make_backend)


if __name__=='__main__':main()
