"""One E24 full-generation FAD mode; cold import is stdlib-only.

Fixed4 P0/feedback are fidelity gates. Dynamic full/actual_only/failure_only use
A-only provisional core costs, not measured complete-policy costs. Whole request
allocator peaks and complete outputs remain raw; no paired TPS/memory benefit is
claimed. Ordinary generation has only its whole boundary synchronization, with
CPU input signatures separately labelled and no per-round timing sync.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import datetime as dt
import gc
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:sys.path.insert(0,str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import production_memory_pair as pair
neutral,guard,g0=pair.neutral,pair.guard,pair.g0
require,save,digest,read_json,jsonl=neutral.require,neutral.save,neutral.digest,neutral.read_json,neutral.jsonl
MODES=('full','actual_only','failure_only','fixed4_p0','fixed4_feedback')
ADAPTIVE=MODES[:3]
SCHEMA='fad_full_generation_E24_v1'
REAL_KIND='actual_FAD_full_generation'
NOTE='One production mode; provisional C_core policy, full outputs and whole allocator observations. Actual paired throughput/memory acceptance pending; C0 engineering fidelity is not original ViSpec TPS.'


class ModelOps(pair.PairOps):
    execution_kind=REAL_KIND
    def __init__(self,args,reference):
        super().__init__(args,reference)
        self.mode=args.mode
        self.controller=None
        self.calibration=None
        self.metadata.update(execution_kind=REAL_KIND,mode=args.mode,controller_installed=args.mode in ADAPTIVE,
            fixed_actual_depth=4 if args.mode not in ADAPTIVE else None,
            comparison_arms=[args.mode],production_feedback='P0 actual receipt' if args.mode=='fixed4_p0' else 'full typed compact feedback',
            ordinary_TPS_eligible=None,ordinary_timing_scope='complete generation boundary synchronization',
            ordinary_TPS_benefit_claim=False,paired_total_memory_accepted=None)
        if args.mode in ADAPTIVE:
            from combined_visual_experiment.fad_vispec.full_controller import FullController,FullControllerConfig
            self.calibration=read_json(args.calibration)
            require(digest(args.calibration)==args.calibration_sha256,'calibration bytes differ')
            require(digest(PROJECT/'fad_vispec/run_depth_cost.py')==self.calibration['source_sha256'],
                'provisional measured-core source differs')
            require(self.flags()==self.calibration['precision_flags'],'actual precision differs from A calibration')
            cfg=dict(context_edges=tuple(self.calibration['context_edges']),acceptance_edges=tuple(self.calibration['acceptance_edges']),
                exploration_probability=.05,hysteresis_relative=.02,conditional_exact_weight=.25,upper_weight=.25,
                prior_mass=4,use_ipw=False,ipw_clip=20,seed=20261004)
            if args.controller_config:
                require(digest(args.controller_config)==args.controller_config_sha256,'controller config differs')
                configured=read_json(args.controller_config)
                if configured.get('schema')=='fad_full_initial_policy_v1':
                    require(configured['calibration_sha256']==args.calibration_sha256,'initial policy calibration differs')
                    cfg.update(configured['config'])
                else:cfg.update(configured)
                cfg['context_edges']=tuple(cfg['context_edges']);cfg['acceptance_edges']=tuple(cfg['acceptance_edges'])
                if 'allowed_depths' in cfg:cfg['allowed_depths']=tuple(cfg['allowed_depths'])
            config=FullControllerConfig(**cfg)
            self.controller=FullController(self.calibration,config=config,mode=args.mode,phase=args.policy_phase,
                device_uuid=args.expected_gpu_uuid,source_sha256=self.calibration['source_sha256'],
                precision_sha256=self.calibration['precision_sha256'],frozen_policy_sha256=args.frozen_policy_sha256)
            self.metadata['policy']=self.controller.policy_receipt()
            if args.controller_config and configured.get('schema')=='fad_full_initial_policy_v1':
                require(configured['policies'][args.mode]['policy_sha256']==self.controller.policy_sha256,'initial mode policy differs')
            self.metadata['calibration_sha256']=args.calibration_sha256
            self.metadata['actual_cost_table']=args.calibration_sha256
    def executor(self,packed,request):
        if self.mode=='fixed4_p0':return super().executor(packed,request)
        from combined_visual_experiment.fad_vispec.production_fad_executor import ProductionFADExecutor
        return ProductionFADExecutor(self.spec,packed,request_key=request['request_id'],max_new_tokens=512,
            tree_budget=self.budget,eos_token_ids=[151645,151643])


@contextmanager
def observe_prefill(bridge,receipt):
    original=bridge._prefill
    def wrapped(*args,**kwargs):
        receipt['calls']+=1;require(receipt['calls']==1,'one prefill required')
        return original(*args,**kwargs)
    bridge._prefill=wrapped
    try:yield
    finally:bridge._prefill=original;receipt['wrapper_restored']=bridge._prefill is original


def run_generation(ops,packed,request,*,mode,controller=None,observations=None):
    """Complete production lifecycle; selector and all feedback belong to timing."""
    require(mode in MODES and ((controller is not None)==(mode in ADAPTIVE)),'mode/controller routing differs')
    obs={} if observations is None else observations
    obs.update(flags=dict(before=ops.flags()),prefill=dict(calls=0,wrapper_restored=False),decisions=[],updates=[],
        controller_begin=None,controller_end=None,cleanup_errors=[],mode=mode)
    executor=policy=cm=None;entered=begun=False;pending=None;error=None;result=None
    ops.torch.cuda.synchronize();started=time.perf_counter()
    try:
        if controller is not None:obs['controller_begin']=controller.begin_request(request['request_id']);begun=True
        cm=ops.grouped();policy=cm.__enter__();entered=True
        executor=ops.executor(packed,request)
        with observe_prefill(ops.bridge,obs['prefill']):
            controls=executor.begin_inplace()
            while controls['stop_reason'] is None:
                ordinal=len(executor.ordinary_history);key=f"{request['request_id']}:round:{ordinal}"
                if controller is not None:
                    visual=int(controls['visual_context'].get('original_count',0))
                    query=int(controls['selected_hidden_waiting'].shape[1])
                    remaining=executor.max_new_tokens-len(controls['generated'])
                    require(remaining>=2,'nonterminal state cannot leave one token before a Draft choice')
                    pending=controller.select(request['request_id'],state_key=key,shape_id=f'v{visual}_q{query}',
                        first_tree=bool(controls['first_tree']),context_length=int(controls['ids'].shape[1]),remaining_tokens=remaining)
                    obs['decisions'].append(copy.deepcopy(pending));depth=pending['actual_depth']
                else:depth=4
                r=executor.run_round_inplace(depth) if mode=='fixed4_p0' else executor.run_round_inplace(depth,state_key=key)
                if controller is not None:
                    obs['updates'].append(controller.observe(pending,r['full_feedback']));pending=None
            result=executor.finish_inplace();executor.validate_coverage(policy);ops.complete_result(result,policy)
    except BaseException as exc:
        error=exc;obs['generation_error']=repr(exc)
    finally:
        def cleanup(stage,action):
            nonlocal error
            try:return action()
            except BaseException as exc:
                obs['cleanup_errors'].append(dict(stage=stage,error=repr(exc)))
                if error is None:error=exc
        if pending is not None:cleanup('controller_cancel',lambda:obs.update(controller_cancel=controller.cancel(pending,reason='generation failed')))
        if begun:cleanup('controller_end',lambda:obs.update(controller_end=controller.end_request(request['request_id'])))
        if executor is not None:
            cleanup('executor_close',executor.close)
            cleanup('execution_receipt',lambda:obs.update(execution=executor.execution_receipt(),executor_closed=executor.closed,
                owned_controls_released=executor._owned_controls is None and executor.cache is None))
        if entered:cleanup('policy_exit',lambda:cm.__exit__(None if error is None else type(error),error,None if error is None else error.__traceback__))
        executor=None
        if policy is not None:cleanup('policy_summary',lambda:obs.update(policy=policy.summary()))
        cleanup('math_flags',lambda:obs['flags'].update(after=ops.flags()))
        cleanup('generation_boundary_sync',ops.torch.cuda.synchronize)
        obs['generation_wall_s']=time.perf_counter()-started
    if error is not None:raise error
    require(obs['prefill']==dict(calls=1,wrapper_restored=True),'prefill hook restoration differs')
    decode_started=time.perf_counter();row=ops.result_row(result)
    obs['output_decode_wall_s']=time.perf_counter()-decode_started
    return row


def validate_execution(row,obs,request_key,mode):
    receipt=obs['execution'];counts=row['result']['forward_counts']
    require(set(counts)==set(guard.CORE_COUNTERS)|neutral.INSTRUMENTATION,'public counter keys differ')
    common={k:counts[k] for k in guard.CORE_COUNTERS}
    rounds=[r for r in row['result']['rounds'] if r['kind']!='prefill_terminal'];trees=[r for r in rounds if r['kind']=='tree']
    history=receipt['ordinary_round_history'];n=len(trees);depths=[h['actual_depth'] for h in history]
    require(receipt['schema']==('fad_production_depth_executor_P0_v1' if mode=='fixed4_p0' else 'fad_production_full_feedback_executor_v1')
        and receipt['request_key']==request_key and not receipt['poisoned'] and not receipt['failures']
        and receipt['all_trial_coverage_validated'] is True and receipt['snapshot_graph_copies']==0
        and not receipt['ledger']['operations'],'execution health/source lifecycle differs')
    require(receipt['all_trial_forward_counts']==common and receipt['all_trial_rounds']==rounds,'public rounds/counters differ')
    require(len(history)==n==common['target_verify_calls']==common['draft_tree_calls']==common['target_compaction_calls']
        and common['target_prefill_calls']==1 and common['draft_model_forward_calls']==sum(depths)
        and common['target_verify_query_tokens']==receipt['actual_verify_nodes']==sum(h['tree_nodes'] for h in history)
        and common['budget_switch_calls']==n,'actual depth/count correspondence differs')
    require(receipt['actual_operation_attempts']==dict(prefill=1,draft_tree=n,target_verify=n,compaction=n),'attempt coverage differs')
    require(all(counts[k]==28*n for k in neutral.INSTRUMENTATION),'grouped layer coverage differs')
    require(obs['executor_closed'] and obs['owned_controls_released'] and obs['policy']['finalized']
        and obs['policy']['mode']=='grouped' and obs['policy']['layer_calls']==[n]*28,'final cleanup/grouped coverage differs')
    require(obs['flags']['before']==obs['flags']['after'],'precision changed within generation')
    for event,h in zip(trees,history):
        d=h['actual_depth'];nodes=9 if d==1 else 30
        require(1<=d<=8 and h['tree_nodes']==event['tree_nodes']==nodes
            and event['budget']==dict(B=nodes,d=d-1,K=8,total_tokens=nodes-1) and event['visual_m']==1.0,'tree depth/full visual recipe differs')
    if mode=='fixed4_p0':require(all(d==4 for d in depths) and not receipt.get('full_feedback_history'),'P0 route differs')
    else:
        feedback=receipt['full_feedback_history'];require(len(feedback)==n and not receipt['feedback_failures']
            and receipt['GPU_feedback_references_released'],'feedback lifetime/count differs')
        for i,(event,d,f) in enumerate(zip(trees,depths,feedback)):
            stop=row['result']['stop_reason'] if i==n-1 else None
            bonus=rounds[-1]['submitted'] if i==n-1 and rounds[-1]['kind']=='terminal_target_token' else []
            require(f['schema']=='fad_full_feedback_v1' and f['request_key']==request_key
                and f['state_key']==f'{request_key}:round:{i}' and f['actual_D']==d and f['A_raw']==event['accepted_draft']
                and f['G_tree']==len(event['submitted']) and f['G_output']==len(event['submitted'])+len(bonus)
                and f['terminal_bonus_ids']==bonus and f['stop_reason']==stop and f['censored'] is (stop is not None)
                and f['no_next_state_guarantee'],'feedback/public outcome differs')
        if mode=='fixed4_feedback':require(all(d==4 for d in depths),'fixed feedback route differs')
    if mode in ADAPTIVE:
        require(len(obs['decisions'])==len(obs['updates'])==n
            and [d['actual_depth'] for d in obs['decisions']]==depths
            and obs['controller_end']['decisions']==obs['controller_end']['observations']==n
            and obs['controller_end']['pending'] is False,'controller complete lifecycle differs')
    else:require(not obs['decisions'] and obs['controller_begin'] is None,'fixed mode installed controller')
    return dict(passed=True,tree_rounds=n,actual_depths=depths,actual_Draft_forward_calls=sum(depths),controller_installed=mode in ADAPTIVE)


def allocator(cuda):
    row={k:int(fn()) for k,fn in dict(allocated_bytes=cuda.memory_allocated,reserved_bytes=cuda.memory_reserved,
        peak_allocated_bytes=cuda.max_memory_allocated,peak_reserved_bytes=cuda.max_memory_reserved).items()}
    require(all(v>=0 for v in row.values()),'negative allocator observation');return row


def run_case(request,saved,out,ops,backend,*,mode):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    case=dict(request_id=request['request_id'],dataset=request['dataset'],split=request['split'],mode=mode,
        execution_kind=ops.execution_kind,completed=False,paired_total_memory_accepted=None,TPS_benefit_claim=False,scope=NOTE)
    monitor=None;cpu=packed=sampled=None;obs={};start=None;raw_started=time.time();clock=time.perf_counter()
    try:
        monitor=guard.ProcessMonitor(backend).start();ops.cleanup_request()
        ops.torch.cuda.reset_peak_memory_stats();start=allocator(ops.torch.cuda)
        raw_started=time.time()
        case['resident_before']=pair.resident(ops)
        t=time.perf_counter();cpu,sampled=ops.prepare_cpu(request);case['CPU_prepare_wall_s']=time.perf_counter()-t
        t=time.perf_counter();inp=pair.input_record(ops,cpu,sampled);case['CPU_input_signature_diagnostic_wall_s']=time.perf_counter()-t
        case['input']=inp;case['input_exact']=all(inp[k]==saved['input'][k] for k in guard.INPUT_KEYS)
        require(case['input_exact'],'CPU input differs from C0')
        t=time.perf_counter();packed=ops.upload(cpu);case['GPU_input_materialization_wall_s']=time.perf_counter()-t
        row=run_generation(ops,packed,request,mode=mode,controller=ops.controller,observations=obs)
        case['output']=row;g0.validate_result(row)
        case['execution_check']=validate_execution(row,obs,request['request_id'],mode)
        case['c0_comparison']=g0.compare_c0(row,saved['row'])
        case['token_ids_equal']=case['c0_comparison']['checks']['token_ids']
        case['stop_equal']=case['c0_comparison']['checks']['stop_reason']
        case['length_ratio']=row['num_tokens']/saved['row']['num_tokens']
        case['within_110_percent']=case['length_ratio']<=1.1
        case['output_fidelity_passed']=all(case['c0_comparison']['checks'][k] for k in
            ('token_ids','stop_reason','output_length','text','truncated'))
        case['flags_exact']=obs['flags']=={k:saved['precision'][k] for k in ('before','after')}
        require(case['flags_exact'],'precision differs from C0')
        if mode.startswith('fixed4_'):require(case['c0_comparison']['exact'],'fixed D4 fidelity differs from C0')
        case['completed']=True
    except BaseException as exc:case.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        t=time.perf_counter()
        try:
            ops.cleanup_request()
        except BaseException as exc:case.update(completed=False,cleanup_error=repr(exc))
        finally:packed=cpu=sampled=None
        try:
            gc.collect();ops.torch.cuda.synchronize()
            case['resident_after']=pair.resident(ops);pair.validate_resident(case['resident_after'],case.get('resident_before'))
        except BaseException as exc:case.update(completed=False,cleanup_error=repr(exc))
        case['request_cleanup_and_release_wall_s']=time.perf_counter()-t
        try:
            end=allocator(ops.torch.cuda)
            case['memory']=dict(start=start,end=end,whole_observed={f'peak_{k}_bytes':max(start.get(f'peak_{k}_bytes',0) if start else 0,end[f'peak_{k}_bytes'],end[f'{k}_bytes']) for k in ('allocated','reserved')},
                started_unix=raw_started,ended_unix=time.time(),resets_within_request=0,empty_cache_used=False,model_loading_excluded=True)
        except BaseException as exc:case.update(completed=False,allocator_error=repr(exc))
        try:case['background']=monitor.stop() if monitor is not None else dict(unavailable=True)
        except BaseException as exc:case.update(completed=False,background_error=repr(exc))
        if case.get('memory'):
            try:
                m=case['memory'];case['background_review']=guard.background_review(case.get('background',{}),m['started_unix'],m['ended_unix'],ops.args.expected_gpu_uuid,ops.args.allow_graphics_pid,os.getpid())
            except BaseException as exc:case.update(completed=False,background_review_error=repr(exc))
        case['observations']=obs;case['raw_instrumented_request_wall_s']=time.perf_counter()-clock
        case['ordinary_complete_request_wall_s']=sum(case.get(k,0) for k in ('CPU_prepare_wall_s','GPU_input_materialization_wall_s','request_cleanup_and_release_wall_s'))+obs.get('generation_wall_s',0)+obs.get('output_decode_wall_s',0)
        case['timing_scope']='Nonoverlapping CPU prepare, upload, complete generation, output decode and request release; CPU input signature diagnostic and artifact writes excluded. No per-round timing synchronization.'
        save(out/'input.json',case.get('input',{}));save(out/'observations.json',obs);save(out/'production.json',case.get('output',{}))
        case['artifact_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
        save(out/'case.json',case)
    return case


def aggregate(cases,requests,mode,*,pilot=False):
    ids=[r['request_id'] for r in requests]
    require(ids and len(ids)==len(set(ids)),'nonempty unique request IDs required')
    if not pilot:require(len(ids)==24 and Counter(r['dataset'] for r in requests)=={'nextqa':12,'videodetailcaption':12},'full E24 required')
    actual=len(cases)==len(ids) and [c['request_id'] for c in cases]==ids and all(c.get('completed') is True
        and c.get('execution_kind')==REAL_KIND and c.get('input_exact') and c.get('flags_exact')
        and c.get('execution_check',{}).get('passed') for c in cases)
    peaks={f'peak_{k}_bytes':max((c['memory']['whole_observed'][f'peak_{k}_bytes'] for c in cases if c.get('memory')),default=None) for k in ('allocated','reserved')}
    output_fidelity=actual and all(c.get('output_fidelity_passed') is True for c in cases)
    return dict(mode=mode,expected_requests=len(ids),recorded_requests=len(cases),pilot=pilot,functional_generation_passed=actual,
        output_fidelity_passed=output_fidelity,complete_E24_passed=actual and not pilot,
        fixed_D4_neutral_passed=actual and mode.startswith('fixed4_') and all(c['c0_comparison']['exact'] for c in cases),
        raw_batch_allocator_maxima=peaks,paired_total_memory_accepted=None,ordinary_TPS_benefit_claim=False,
        full_policy_cost_validated=False,original_ViSpec_comparison_completed=False,scope=NOTE)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('out','requests','requests-sha256','c0-dir','expected-gpu-uuid','execution-source-freeze','execution-source-freeze-sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--allow-graphics-pid',type=int,action='append',default=[]);p.add_argument('--c0-project');p.add_argument('--request-id',action='append')
    for n in ('calibration','calibration-sha256','frozen-policy-sha256'):p.add_argument('--'+n)
    p.add_argument('--controller-config','--config',dest='controller_config')
    p.add_argument('--controller-config-sha256','--config-sha256',dest='controller_config_sha256')
    p.add_argument('--policy-phase',choices=('calibration_A','holdout_B','test_T200'),default='calibration_A')
    p.add_argument('--validate-feedback',action='store_true')
    a=p.parse_args(argv)
    require(a.requests_sha256==g0.E24_SHA,'only frozen full E24 source supported');require(not Path(a.out).exists(),'fresh output required')
    require(a.expected_gpu_uuid.startswith('GPU-') and str(uuid.UUID(a.expected_gpu_uuid[4:]))==a.expected_gpu_uuid[4:],'canonical GPU UUID required')
    require(set(a.allow_graphics_pid).issubset({6360}),'only verified Xorg6360 allowlist');a.allow_graphics_pid=sorted(set(a.allow_graphics_pid))
    require(bool(a.controller_config)==bool(a.controller_config_sha256),'controller config path/SHA pair required')
    if a.mode in ADAPTIVE:require(a.calibration and a.calibration_sha256,'adaptive modes require A calibration path/SHA')
    require(bool(a.calibration)==bool(a.calibration_sha256),'calibration path/SHA pair required')
    return a


def main(argv=None):
    args=parse_args(argv);execution=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
    require(digest(args.requests)==g0.E24_SHA,'E24 bytes differ');all_requests=jsonl(args.requests);neutral.aggregate([],all_requests)
    requests=all_requests
    if args.request_id:
        require(len(set(args.request_id))==len(args.request_id) and set(args.request_id)<={r['request_id'] for r in all_requests},'pilot IDs must be unique E24 members')
        requests=[r for r in all_requests if r['request_id'] in args.request_id]
    manifest_path=Path(args.requests).parent/'MANIFEST.json';require(digest(manifest_path)==g0.MANIFEST_SHA,'manifest differs');manifest=read_json(manifest_path)
    reference=g0.load_reference(args.c0_dir,all_requests);c0_project=g0.resolve_c0_project(args.c0_dir,args.c0_project)
    sources=g0.check_sources(reference,manifest,c0_project=c0_project);sources.update(execution['source_sha256'])
    bound={n:digest(getattr(args,n)) for n in ('calibration','controller_config') if getattr(args,n)}
    for name,h in bound.items():require(h==getattr(args,name+'_sha256'),'policy input SHA differs')
    lock,lock_receipt=g0.load_lock_without_torch(PROJECT/'direct_opt/run_joint.py');out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status='starting',pid=os.getpid(),run_id=uuid.uuid4().hex,args=vars(args),entry_sha256=digest(__file__),
        execution_source_freeze=execution,source_sha256=sources,c0_artifacts_sha256=reference['hashes'],policy_inputs_sha256=bound,
        source_recheck=dict(passed=False),lock_source=lock_receipt,cases=[],prelaunch=[],scope=NOTE)
    def persist():
        report['updated_at']=dt.datetime.now(dt.timezone.utc).isoformat();report['gate']=aggregate(report['cases'],requests,args.mode,pilot=bool(args.request_id))
        if report['status']!='completed' or not report['source_recheck']['passed']:report['gate']['functional_generation_passed']=report['gate']['output_fidelity_passed']=report['gate']['complete_E24_passed']=report['gate']['fixed_D4_neutral_passed']=False
        save(out/'status.json',report);save(out/'FULL_FAD_REPORT.json',report)
    persist();backend=None;failure=None;previous={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','MKL_NUM_THREADS')}
    try:
        for p in (PROJECT,PROJECT/'vendor',PROJECT/'direct_opt',PROJECT/'training_free'):sys.path.insert(0,str(p))
        from hardware_metrics import _make_backend
        with lock(args.physical_gpu,out) as metadata:
            report['lock']=metadata;backend=_make_backend(args.physical_gpu);guard.prelaunch_checks(backend,args,report['prelaunch'],persist=persist)
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            import torch
            report['cuda_device']=guard.bind_cuda_device(torch,args)
            if args.validate_feedback:
                from combined_visual_experiment.fad_vispec.validate_full_feedback_gpu import run_validation
                try:validation=run_validation(device='cuda:0')
                except BaseException as exc:
                    validation=dict(schema='fad_full_feedback_device_validation_v1',passed=False,device_requested='cuda:0',
                        error=repr(exc),traceback=traceback.format_exc(),scope='device primitive diagnostic; model not loaded')
                    save(out/'gpu_feedback_validation.json',validation)
                    report['GPU_feedback_validation_sha256']=digest(out/'gpu_feedback_validation.json');raise
                save(out/'gpu_feedback_validation.json',validation)
                report['GPU_feedback_validation_sha256']=digest(out/'gpu_feedback_validation.json')
                require(validation['passed'] is True and validation['device_requested']=='cuda:0'
                    and validation['device_actual']=='cuda:0','device feedback primitive validation failed')
            from grouped_gqa_runtime import math_precision_scope
            with math_precision_scope(),torch.inference_mode():
                ops=ModelOps(args,reference);report.update(status='running',setup=ops.metadata);persist()
                for i,request in enumerate(requests):
                    observed=guard.observe_processes(backend);report.setdefault('request_guards',[]).append(observed)
                    guard.verify_idle(observed,args.expected_gpu_uuid,args.allow_graphics_pid,os.getpid())
                    report['current']=dict(index=i,request_id=request['request_id']);persist()
                    case=run_case(request,reference['selected'][request['request_id']],out/'cases'/f'{i:02d}',ops,backend,mode=args.mode)
                    report['cases'].append(case);guard.append(out/'cases.jsonl',case);persist();require(case['completed'],'generation failed; raw retained, later requests paused')
                report.update(status='completed',current=None)
    except BaseException as exc:failure=exc;report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as exc:
                report.update(status='failed',backend_close_error=repr(exc));failure=failure or exc
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        try:
            after=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
            require(after==execution,'execution source drift');final=g0.check_sources(reference,manifest,c0_project=c0_project);final.update(after['source_sha256'])
            require(final==sources and digest(manifest_path)==g0.MANIFEST_SHA and digest(args.requests)==g0.E24_SHA,'source/input drift')
            for n,h in reference['hashes'].items():require(digest(Path(args.c0_dir)/n)==h,'C0 artifact drift')
            for n,h in bound.items():require(digest(getattr(args,n))==h,'calibration/config drift')
            report.update(execution_source_freeze_after=after,source_recheck=dict(passed=True,execution_inventory_exact=True))
        except BaseException as exc:report.update(status='failed',source_recheck=dict(passed=False,error=repr(exc)));failure=failure or exc
        report['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat();persist()
    if failure is not None:raise failure
    require(report['gate']['functional_generation_passed'],'incomplete functional generation gate');return report


if __name__=='__main__':main()
