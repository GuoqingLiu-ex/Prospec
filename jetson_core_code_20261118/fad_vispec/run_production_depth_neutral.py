"""Fixed-D4 E24 production-lifecycle fidelity gate; stdlib-only import.

No controller, cost table, branch snapshots, traces or top-2 capture. Actual
phase allocator statistics include input-hash diagnostics in the raw whole
request. A same-process paired generate_joint baseline is still required for
total Target+Draft allocated AND reserved memory acceptance; this gate cannot
infer it from historical shared-card C0. No TPS/DRAM claims.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:sys.path.insert(0,str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import run_depth_cost as guard
g0=guard.g0
require,save,digest,read_json,jsonl=guard.require,guard.save,guard.digest,guard.read_json,guard.jsonl
SCHEMA='fad_production_fixed_D4_E24_v1'
REAL_KIND='actual_fad_production_fixed_D4_P0'
EXECUTOR_SHA='40aa233fb839e97c78f9ec6cf4d7b49aca3d7864920a7746740c1b7c56ec5233'
INSTRUMENTATION={'target_tree_attention_policy_calls','target_grouped_attention_calls'}
NOTE=('Fixed D4 complete E24 fidelity only. Raw memory includes input signature and phase instrumentation. '
      'Prepared-generation is a separately labelled observation, not paired deployment memory acceptance. '
      'No cost/controller, TPS, DRAM or allocator-policy change.')


class PhaseMemory:
    """Disjoint measured segments; read every peak BEFORE any reset.

    Boundaries synchronize for phase attribution. No reset discards an earlier
    peak: whole is the host maximum over every actual start/end/peak reading.
    These explicit syncs are instrumentation, not ordinary throughput evidence.
    """
    def __init__(self,cuda,*,clock=time.perf_counter,unix=time.time):
        self.cuda,self.clock,self.unix=cuda,clock,unix
        self.rows=[];self.current=None;self.started=None;self.finished=None
    def _sample(self):
        x=dict(allocated_bytes=int(self.cuda.memory_allocated()),reserved_bytes=int(self.cuda.memory_reserved()),
            peak_allocated_bytes=int(self.cuda.max_memory_allocated()),peak_reserved_bytes=int(self.cuda.max_memory_reserved()))
        require(all(v>=0 for v in x.values()),'invalid allocator counters');return x
    def switch(self,name):
        self.cuda.synchronize();now=self.clock();stamp=self.unix()
        if self.current is not None:
            self.current.update(end=self._sample(),wall_s=now-self.current['clock'],ended_unix=stamp)
            self.current.pop('clock');self.rows.append(self.current)
        self.cuda.reset_peak_memory_stats()
        self.current=dict(name=name,start=self._sample(),clock=self.clock(),started_unix=self.unix())
        if self.started is None:self.started=self.current['started_unix']
    def finish(self):
        require(self.current is not None and self.finished is None,'memory tracker already finished or not started')
        self.cuda.synchronize();end=self._sample();stamp=self.unix()
        self.current.update(end=end,wall_s=self.clock()-self.current['clock'],ended_unix=stamp)
        self.current.pop('clock');self.rows.append(self.current);self.current=None;self.finished=stamp
        def maxima(rows):
            return {f'peak_{kind}_bytes':max(r[point][f'{prefix}{kind}_bytes'] for r in rows
                for point in ('start','end') for prefix in ('','peak_')) for kind in ('allocated','reserved')}
        generation=[r for r in self.rows if r['name'] in ('generation_setup','begin_before_prefill','prefill',
            'begin_after_prefill','decode_P0','finish_and_coverage','cleanup','summary_and_output_decode')]
        return dict(schema='fad_segmented_actual_allocator_v1',segments=self.rows,
            started_unix=self.started,ended_unix=self.finished,observed_wall_s=self.finished-self.started,
            start=self.rows[0]['start'],end=end,whole_observed=maxima(self.rows),
            prepared_generation=None if not generation else maxima(generation),
            whole_includes_input_signature_diagnostic=True,phase_sync_instrumentation=True,
            reset_discards_previous_peak=False,empty_cache_used=False,
            paired_total_memory_accepted=None,scope=NOTE)


@contextmanager
def prefill_phase(bridge,tracker,receipt):
    """One scoped timing wrapper, preserving exact arguments/result/exceptions."""
    original=bridge._prefill
    def wrapped(*args,**kwargs):
        receipt['calls']+=1;require(receipt['calls']==1,'one actual prefill per request required')
        tracker.switch('prefill')
        try:value=original(*args,**kwargs)
        except BaseException:
            try:tracker.switch('begin_after_prefill')
            except BaseException as exc:receipt['failed_boundary_error']=repr(exc)
            raise
        tracker.switch('begin_after_prefill');return value
    bridge._prefill=wrapped
    try:yield
    finally:bridge._prefill=original;receipt['wrapper_restored']=bridge._prefill is original


class ProductionOps(g0.RealOps):
    execution_kind=REAL_KIND
    def __init__(self,args,reference):
        require(digest(PROJECT/'fad_vispec/production_depth_executor.py')==EXECUTOR_SHA,'production executor freeze differs')
        super().__init__(args,reference)
        from combined_visual_experiment.fad_vispec.production_depth_executor import ProductionDepthExecutor
        self.executor_type=ProductionDepthExecutor
        self.metadata.update(execution_kind=REAL_KIND,teacher_forcing_used=False,diagnostic_snapshots=False,
            focus_pool_installed=False,focus_vocab_installed=False,fixed_actual_depth=4,
            production_feedback='P0 actual host receipt',controller_installed=False,actual_cost_table=None)
    def executor(self,packed,request):
        return self.executor_type(self.spec,packed,request_key=request['request_id'],max_new_tokens=512,
            tree_budget=self.budget,eos_token_ids=[151645,151643])


def run_generation(ops,packed,request,tracker,observations):
    """Actual in-place lifecycle. All method setup/cleanup/summary is measured.

    Dependencies may be injected by CPU tests; their execution_kind is distinct
    and cannot satisfy the actual-model E24 aggregate. No reference drives IDs.
    """
    executor=policy=cm=None;entered=False;error=None;result=None
    observations.update(flags=dict(before=ops.flags()),prefill=dict(calls=0,wrapper_restored=False),cleanup_errors=[])
    try:
        tracker.switch('generation_setup')
        cm=ops.grouped();policy=cm.__enter__();entered=True
        executor=ops.executor(packed,request)
        with prefill_phase(ops.bridge,tracker,observations['prefill']):
            tracker.switch('begin_before_prefill');controls=executor.begin_inplace()
            tracker.switch('decode_P0')
            while controls['stop_reason'] is None:executor.run_round_inplace(actual_depth=4)
            tracker.switch('finish_and_coverage')
            result=executor.finish_inplace()
            executor.validate_coverage(policy);ops.complete_result(result,policy)
    except BaseException as exc:
        error=exc;observations['generation_error']=repr(exc)
    finally:
        try:tracker.switch('cleanup')
        except BaseException as exc:
            observations['cleanup_errors'].append(dict(stage='memory_boundary',error=repr(exc)))
            if error is None:error=exc
        if executor is not None:
            try:executor.close()
            except BaseException as exc:
                observations['cleanup_errors'].append(dict(stage='executor_close',error=repr(exc)))
                if error is None:error=exc
            try:
                observations['execution']=executor.execution_receipt()
                observations['executor_closed']=executor.closed
                observations['owned_controls_released']=executor._owned_controls is None and executor.cache is None
            except BaseException as exc:
                observations['cleanup_errors'].append(dict(stage='executor_receipt',error=repr(exc)))
                if error is None:error=exc
        if entered:
            try:cm.__exit__(None if error is None else type(error),error,None if error is None else error.__traceback__)
            except BaseException as exc:
                observations['cleanup_errors'].append(dict(stage='policy_exit',error=repr(exc)))
                if error is None:error=exc
        # Do not retain executor.inputs/values or borrowed live controls through
        # final allocator sampling. Receipts/results contain only host values.
        executor=None
        try:
            tracker.switch('summary_and_output_decode')
            if policy is not None:observations['policy']=policy.summary()
            observations['flags']['after']=ops.flags()
        except BaseException as exc:
            observations['cleanup_errors'].append(dict(stage='summary',error=repr(exc)))
            if error is None:error=exc
    if error is not None:raise error
    require(observations['prefill']==dict(calls=1,wrapper_restored=True),'prefill wrapper did not restore exactly')
    return ops.result_row(result)


def validate_production(row,observations,request_key):
    """Reconcile existing public counts/events with all actual P0 calls."""
    receipt=observations['execution'];result=row['result'];counts=result['forward_counts']
    require(set(counts)==set(guard.CORE_COUNTERS)|INSTRUMENTATION,'unknown/missing public counters')
    common={k:counts[k] for k in guard.CORE_COUNTERS}
    require(receipt['schema']=='fad_production_depth_executor_P0_v1' and receipt['request_key']==request_key
            and receipt['all_trial_coverage_validated'] is True and not receipt['poisoned'] and not receipt['failures'],
            'production execution health/identity differs')
    require(receipt['snapshot_graph_copies']==0 and receipt['production_feedback_included'] is True
            and receipt['controller_installed'] is False and receipt['actual_cost_table'] is None
            and not receipt['ledger']['operations'],'unexpected snapshot/transfer/controller evidence')
    rounds=[r for r in result['rounds'] if r['kind']!='prefill_terminal']
    require(receipt['all_trial_forward_counts']==common and receipt['all_trial_rounds']==rounds,'P0 actual ledger differs from public result')
    trees=[r for r in rounds if r['kind']=='tree'];n=len(trees)
    history=receipt['ordinary_round_history'];feedback=receipt['production_feedback_history']
    require(len(history)==len(feedback)==n==common['target_verify_calls']
            and common['target_prefill_calls']==1 and common['target_verify_query_tokens']==30*n
            and common['draft_tree_calls']==n and common['draft_model_forward_calls']==4*n
            and common['target_compaction_calls']==n and common['budget_switch_calls']==n
            and receipt['actual_verify_nodes']==30*n,'fixed D4 counts/history differ')
    require(receipt['actual_operation_attempts']==dict(prefill=1,draft_tree=n,target_verify=n,compaction=n),'actual attempts differ')
    require(all(counts[k]==28*n for k in INSTRUMENTATION),'28-layer public instrumentation differs')
    prefix=0
    for i,(event,h,f) in enumerate(zip(trees,history,feedback)):
        require(h==dict(actual_depth=4,tree_nodes=30),'ordinary history changed')
        require(event['budget']==dict(B=30,d=3,K=8,total_tokens=29) and event['budget_action']=='new_B30_d3_K8'
                and event['tree_nodes']==30 and event['visual_m']==1.0,'D4/full-visual event differs')
        a=f['actual'];bonus=int(i==n-1 and bool(rounds) and rounds[-1]['kind']=='terminal_target_token')
        require(f['request_key']==request_key and f['round_index']==i and f['actual_depth']==4
                and f['context']['output_prefix_tokens']==prefix and f['context']['prompt_tokens']==result['prompt_tokens']
                and f['context']['target_cache_before']==event['target_cache_before']
                and f['context']['target_cache_after']==event['target_cache_after']
                and f['context']['semantic_rope_base']==event['semantic_rope_base'], 'P0 context binding differs')
        require(a['A_raw']==event['accepted_draft'] and a['G_tree']==len(event['submitted'])
                and a['terminal_bonus_tokens']==bonus and a['G_output']==len(event['submitted'])+bonus
                and a['submitted_ids']==event['submitted'] and a['selected_tree_indices']==event['selected_tree_indices']
                and a['budget_or_eos_clipped']==event['budget_or_eos_clipped'], 'P0 values differ from actual tree')
        stopped=i==n-1 and result['stop_reason'] is not None
        require(a['stop_reason']==(result['stop_reason'] if stopped else None) and a['censored'] is stopped
                and f['observed_depth_update_eligible'] is (not stopped)
                and set(f['other_depths'])=={str(d) for d in range(1,9) if d!=4}
                and all(v['status']=='unknown' for v in f['other_depths'].values())
                and f['cross_depth_certificate_eligible'] is False and f['future_or_next_state_guarantee'] is False,
                'P0 unobserved/censored semantics differ')
        prefix+=a['G_output']
    policy=observations['policy']
    require(policy['finalized'] is True and policy['mode']=='grouped'
            and policy['candidate_calls']==policy['completed_calls']==28*n
            and policy['layer_calls']==[n]*28 and policy['prefill_calls']==policy['prefill_forward_calls']==1
            and policy['prefill_layer_calls']==[1]*28 and policy['verify_forward_calls']==n
            and policy['verify_query_tokens']==30*n and len(policy['rounds'])==n
            and policy['failures']==[] and policy['fallback_calls']==policy['native_tree_calls']==0
            and policy['original_mask_reused'] is True and policy['position_ids_passed_unchanged'] is True
            and policy['rotary_reapplied'] is False,'finalized policy coverage differs')
    require(observations['executor_closed'] and observations['owned_controls_released']
            and not observations['cleanup_errors'] and observations['prefill']==dict(calls=1,wrapper_restored=True),
            'production cleanup or prefill coverage failed')
    for p,e in zip(policy['rounds'],trees):
        require(p==dict(cache_before=e['target_cache_before'],tree_nodes=30,
            cache_after=e['target_cache_before']+30,layer_calls=[1]*28,
            mask_shape=[1,1,30,e['target_cache_before']+30],mask_dtype='torch.float32',
            position_shape=[3,1,30],position_dtype='torch.int64'), 'actual policy per-round coverage differs')
    return dict(passed=True,tree_rounds=n,actual_verify_nodes=30*n,snapshot_graph_copies=0,
        feedback='P0 observed depth only',unexecuted_depths_unknown=True,controller_installed=False)


def run_case(request,saved,out,ops,backend):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    case=dict(request_id=request['request_id'],dataset=request['dataset'],split=request['split'],completed=False,
        execution_kind=ops.execution_kind,stage='prepare',fixed_actual_depth=4,
        paired_total_memory_accepted=None,TPS_eligible=False,actual_DRAM_measured=False,scope=NOTE)
    tracker=PhaseMemory(ops.torch.cuda);monitor=None;packed=sampled=None;observations={}
    try:
        monitor=guard.ProcessMonitor(backend).start()
        tracker.switch('request_setup');ops.reset()
        tracker.switch('input_prepare');packed,sampled=ops.prepare(request)
        tracker.switch('input_validation')
        inp=dict(input_signature=ops.signature(packed,list(sampled.timestamps)),timestamps=list(sampled.timestamps),
            frame_indices=list(sampled.metadata['frames_indices']),sampled_metadata=dict(sampled.metadata),
            sampled_provenance=dict(sampled.provenance),signature_logical_payload=ops.signature_payload(packed))
        save(out/'input.json',inp)
        case['input_exact']=all(inp[k]==saved['input'][k] for k in guard.INPUT_KEYS)
        require(case['input_exact'],'prepared input differs from C0')
        case['stage']='production_fixed_D4'
        row=run_generation(ops,packed,request,tracker,observations)
        save(out/'production.json',row);save(out/'observations.json',observations)
        save(out/'production_execution.json',observations['execution'])
        save(out/'P0_feedback.json',observations['execution']['production_feedback_history'])
        g0.validate_result(row);case['execution_check']=validate_production(row,observations,request['request_id'])
        case['c0_comparison']=g0.compare_c0(row,saved['row'])
        case['flags_exact']=observations['flags']=={k:saved['precision'][k] for k in ('before','after')}
        require(case['flags_exact'],'actual math flags differ from C0')
        require(case['c0_comparison']['exact'],'P0 D4 differs from C0: complete negative retained')
        case.update(completed=True,stage='completed',output_tokens=row['num_tokens'],stop_reason=row['stop_reason'],
            historical_C0_memory_provenance={k:v for k,v in saved['row'].items() if 'memory' in k or 'peak' in k},
            historical_C0_memory_used_for_acceptance=False)
    except BaseException as exc:
        case.update(error=repr(exc),traceback=traceback.format_exc(),completed=False)
    finally:
        try:
            if tracker.current is not None:tracker.switch('release_inputs_and_artifacts')
            packed=sampled=None
            save(out/'observations.json',observations)
            if 'execution' in observations:save(out/'production_execution.json',observations['execution'])
            if tracker.current is not None:case['memory']=tracker.finish()
        except BaseException as exc:case.update(completed=False,finalization_error=repr(exc))
        try:case['background']=monitor.stop() if monitor is not None else dict(unavailable=True)
        except BaseException as exc:case.update(completed=False,background_error=repr(exc))
        if case.get('memory') and case.get('background'):
            case['background_review']=guard.background_review(case['background'],case['memory']['started_unix'],
                case['memory']['ended_unix'],ops.args.expected_gpu_uuid,ops.args.allow_graphics_pid,os.getpid())
        case['artifact_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='case.json'}
        # This last CPU JSON write is outside allocator/timing observations.
        case['final_case_json_write_outside_observation']=True;save(out/'case.json',case)
    return case


def aggregate(cases,requests):
    expected=[r['request_id'] for r in requests]
    require(len(expected)==24 and len(set(expected))==24
            and Counter(r['dataset'] for r in requests)=={'nextqa':12,'videodetailcaption':12},'complete fixed E24 required')
    exact=(len(cases)==24 and [c['request_id'] for c in cases]==expected and all(
        c.get('completed') is True and c.get('execution_kind')==REAL_KIND and c.get('input_exact') is True
        and c.get('flags_exact') is True and c.get('c0_comparison',{}).get('exact') is True
        and c.get('execution_check',{}).get('passed') is True for c in cases))
    peaks={}
    for kind in ('allocated','reserved'):
        values=[c['memory']['whole_observed']['peak_'+kind+'_bytes'] for c in cases if c.get('memory')]
        peaks['peak_'+kind+'_bytes']=max(values) if values else None
    return dict(expected_requests=24,recorded_requests=len(cases),complete_exact_cases=exact,
        production_D4_neutral_passed=exact,failed_request_ids=[c['request_id'] for c in cases if not c.get('completed')],
        raw_observed_batch_max=peaks,paired_total_memory_accepted=None,
        paired_baseline_pending=True,dynamic_policy_implemented=False,actual_cost_table=None,
        ordinary_TPS_eligible=False,DRAM_gain_measured=False,scope=NOTE)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('out','requests','requests-sha256','c0-dir','expected-gpu-uuid','execution-source-freeze','execution-source-freeze-sha256'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--allow-graphics-pid',type=int,action='append',default=[]);p.add_argument('--c0-project')
    a=p.parse_args(argv)
    require(a.requests_sha256==g0.E24_SHA,'only frozen full E24 supported')
    require(not Path(a.out).exists(),'fresh output required')
    require(a.expected_gpu_uuid.startswith('GPU-') and str(uuid.UUID(a.expected_gpu_uuid[4:]))==a.expected_gpu_uuid[4:],'canonical GPU UUID required')
    require(set(a.allow_graphics_pid).issubset({6360}),'only verified Xorg 6360 may be allowlisted')
    a.allow_graphics_pid=sorted(set(a.allow_graphics_pid));return a


def main(argv=None):
    args=parse_args(argv)
    execution=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
    require(digest(args.requests)==g0.E24_SHA,'E24 bytes differ');requests=jsonl(args.requests)
    require(all(r.get('split')=='E24' for r in requests),'wrong request split')
    aggregate([],requests)
    manifest_path=Path(args.requests).parent/'MANIFEST.json'
    require(digest(manifest_path)==g0.MANIFEST_SHA,'manifest differs');manifest=read_json(manifest_path)
    reference=g0.load_reference(args.c0_dir,requests)
    c0_project=g0.resolve_c0_project(args.c0_dir,args.c0_project)
    sources=g0.check_sources(reference,manifest,c0_project=c0_project);sources.update(execution['source_sha256'])
    lock,lock_receipt=g0.load_lock_without_torch(PROJECT/'direct_opt/run_joint.py')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status='starting',pid=os.getpid(),run_id=uuid.uuid4().hex,args=vars(args),
        entry_sha256=digest(__file__),source_sha256=sources,execution_source_freeze=execution,
        source_recheck=dict(passed=False),c0_artifacts_sha256=reference['hashes'],lock_source=lock_receipt,
        request_sha256=g0.E24_SHA,manifest_sha256=g0.MANIFEST_SHA,cases=[],prelaunch=[],scope=NOTE)
    def persist():
        report['updated_at']=dt.datetime.now(dt.timezone.utc).isoformat();report['gate']=aggregate(report['cases'],requests)
        if report['status']!='completed' or not report['source_recheck']['passed']:report['gate']['production_D4_neutral_passed']=False
        save(out/'status.json',report);save(out/'PRODUCTION_DEPTH_NEUTRAL_REPORT.json',report)
    persist();backend=None;failure=None
    previous={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','MKL_NUM_THREADS')}
    try:
        for p in (PROJECT,PROJECT/'vendor',PROJECT/'direct_opt',PROJECT/'training_free'):sys.path.insert(0,str(p))
        from hardware_metrics import _make_backend
        with lock(args.physical_gpu,out) as metadata:
            report['lock']=metadata;backend=_make_backend(args.physical_gpu)
            guard.prelaunch_checks(backend,args,report['prelaunch'],persist=persist)
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            import torch
            report['cuda_device']=guard.bind_cuda_device(torch,args)
            from grouped_gqa_runtime import math_precision_scope
            with math_precision_scope(),torch.inference_mode():
                ops=ProductionOps(args,reference);report.update(status='running',setup=ops.metadata);persist()
                for i,request in enumerate(requests):
                    observed=guard.observe_processes(backend);report.setdefault('request_guards',[]).append(observed)
                    guard.verify_idle(observed,args.expected_gpu_uuid,args.allow_graphics_pid,os.getpid())
                    report['current']=dict(index=i,request_id=request['request_id']);persist()
                    case=run_case(request,reference['selected'][request['request_id']],out/'cases'/f'{i:02d}',ops,backend)
                    report['cases'].append(case);guard.append(out/'cases.jsonl',case);persist()
                    require(case['completed'],'production E24 request failed; raw retained and later requests paused')
                report.update(status='completed',current=None)
    except BaseException as exc:
        failure=exc;report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as exc:
                report.update(status='failed',backend_close_error=repr(exc))
                if failure is None:failure=exc
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        try:
            after=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
            report['execution_source_freeze_after']=after;require(after==execution,'execution source drift')
            final=g0.check_sources(reference,manifest,c0_project=c0_project);final.update(after['source_sha256'])
            require(final==sources and digest(manifest_path)==g0.MANIFEST_SHA and digest(args.requests)==g0.E24_SHA,'source/manifest/request drift')
            # Preserve original complete C0 artifact bytes as well as its source.
            for name,expected in reference['hashes'].items():require(digest(Path(args.c0_dir)/name)==expected,'C0 artifact drift: '+name)
            report['source_recheck']=dict(passed=True,execution_inventory_exact=True)
        except BaseException as exc:
            report.update(status='failed',source_recheck=dict(passed=False,error=repr(exc)))
            if failure is None:failure=exc
        report['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat();persist()
    if failure is not None:raise failure
    require(report['gate']['production_D4_neutral_passed'],'incomplete actual E24 gate')
    return report


if __name__=='__main__':main()
