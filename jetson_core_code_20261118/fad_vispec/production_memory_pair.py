"""Same-process C0 generate_joint versus fixed-D4 P0 allocator comparison.

Every arm redoes CPU video/processor preparation + CPU-only hash, then the
original tensor .to(cuda) input materialization. Phase synchronization and CPU
hashing make wall/TPS diagnostic only. No snapshots, teacher forcing, allocator
changes or empty_cache. This is not original-author ViSpec or dynamic FAD.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import gc
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
import weakref

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:sys.path.insert(0,str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import run_production_depth_neutral as neutral
require,save,digest=neutral.require,neutral.save,neutral.digest
g0,guard=neutral.g0,neutral.guard
SCHEMA='fad_C0_P0_same_process_memory_pair_v1'
REAL_KIND='actual_C0_P0_memory_pair'
NOTE=('Loaded C0 engineering baseline vs fixed D4 P0; complete CPU prepare and GPU input materialization, '
      'prefill/decode/common cleanup/release. CPU hash and phase sync instrumentation exclude ordinary TPS. '
      'Whole allocated AND reserved required; no empty_cache or start-delta subtraction.')


def schedule():
    return [dict(phase='warmup',repeat=0,arm='original'),dict(phase='warmup',repeat=0,arm='P0'),
            dict(phase='measure',repeat=0,arm='original'),dict(phase='measure',repeat=0,arm='P0'),
            dict(phase='measure',repeat=1,arm='P0'),dict(phase='measure',repeat=1,arm='original')]


class PairOps(neutral.ProductionOps):
    execution_kind=REAL_KIND
    def __init__(self,args,reference):
        super().__init__(args,reference)
        self.metadata.update(execution_kind=REAL_KIND,comparison_arms=['C0_generate_joint','fixed_D4_P0'],
            input_prepare='native CPU prepare plus original value.to(cuda) per arm',ordinary_TPS_eligible=False)
    def prepare_cpu(self,request):return self.prepare_request(self.processor,request,'cpu')
    def upload(self,packed):
        require(all(not self.torch.is_tensor(v) or v.device.type=='cpu' for v in packed.values()),'CPU prepare returned device tensor')
        # Exact last step of the native prepare_inputs, deferred after CPU hash.
        return {k:v.to('cuda:0') if hasattr(v,'to') else v for k,v in packed.items()}
    def cleanup_request(self):
        self.spec.spec_layer.reset_kv()
        self.spec.spec_layer.last_img_hidden=None;self.spec.spec_layer.tree_mask=None
        self.spec.base_model.model.rope_deltas=None
        self.joint.apply_tree_budget(self.spec.spec_layer,self.budget)
        gc.collect();self.torch.cuda.synchronize()


def resident(ops):
    """Identity/layout only: no tensor payload read or saved Tensor references."""
    draft,target=ops.spec.spec_layer,ops.spec.base_model
    tensors=[];hooks=[];modules=[]
    for prefix,model in (('target',target),('draft',draft)):
        for kind,items in (('parameter',model.named_parameters(remove_duplicate=False)),
                           ('buffer',model.named_buffers(remove_duplicate=False))):
            for name,t in items:
                try:version=t._version
                except RuntimeError:version=None  # Inference tensors have no version counter.
                tensors.append(dict(name=prefix+'.'+name,kind=kind,object_id=id(t),dtype=str(t.dtype),
                    device=str(t.device),shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),
                    storage_ptr=t.untyped_storage().data_ptr(),storage_bytes=t.untyped_storage().nbytes(),
                    requires_grad=t.requires_grad,version=version))
        for name,module in model.named_modules():
            modules.append([prefix+'.'+name,id(module)])
            for field in ('_forward_hooks','_forward_pre_hooks','_forward_hooks_with_kwargs','_forward_pre_hooks_with_kwargs','_forward_hooks_always_called'):
                if hasattr(module,field):
                    registry=getattr(module,field)
                    hooks.append([prefix+'.'+name,field,id(registry),
                                  [[key,id(value)] for key,value in sorted(registry.items())]])
    roots={name:getattr(draft,name,None) is None for name in ('stable_kv','last_img_hidden','tree_mask')}
    roots['rope_deltas']=getattr(target.model,'rope_deltas',None) is None
    return dict(model_ids=[id(target),id(draft)],request_roots_none=roots,
        constants=[draft.total_tokens,draft.depth,draft.top_k],tensors=tensors,hooks=hooks,modules=modules)


def validate_resident(value,baseline=None):
    require(all(value['request_roots_none'].values()) and value['constants']==[29,3,8], 'request roots/canonical state not clean')
    if baseline is not None:require(value==baseline,'persistent model/storage/hook identity changed')


@contextmanager
def cache_liveness(bridge,receipt):
    original=bridge._prefill;references=[]
    def wrapped(*a,**k):
        out=original(*a,**k);references.append(weakref.ref(out[0].past_key_values));return out
    bridge._prefill=wrapped
    try:yield references
    finally:
        bridge._prefill=original
        receipt['prefill_wrapper_restored']=bridge._prefill is original


def run_original(ops,packed,tracker,observations):
    """Delegate unchanged generate_joint under identical grouped/math setup."""
    result=policy=None;observations.update(flags=dict(before=ops.flags()),prefill=dict(calls=0,wrapper_restored=False))
    try:
        tracker.switch('generation_setup')
        with ops.grouped() as policy:
            with neutral.prefill_phase(ops.bridge,tracker,observations['prefill']):
                tracker.switch('begin_before_prefill');result=ops.generate(packed)
            tracker.switch('finish_and_coverage');ops.complete_result(result,policy)
        tracker.switch('summary_and_output_decode')
        observations['policy']=policy.summary();row=ops.result_row(result)
        return row
    finally:
        # The original method owns its local cache; no snapshot/restore is added.
        result=None
        observations['flags']['after']=ops.flags()
        if policy is not None:observations['policy']=policy.summary()


def input_record(ops,packed,sampled):
    require(all(not ops.torch.is_tensor(v) or v.device.type=='cpu' for v in packed.values()),'hash must only touch CPU inputs')
    start=time.perf_counter();signature=ops.signature(packed,list(sampled.timestamps));elapsed=time.perf_counter()-start
    return dict(input_signature=signature,timestamps=list(sampled.timestamps),
        frame_indices=list(sampled.metadata['frames_indices']),sampled_metadata=dict(sampled.metadata),
        sampled_provenance=dict(sampled.provenance),CPU_signature_wall_s=elapsed,
        hash_GPU_tensor_materializations=0,input_preparation='native CPU prepare then unchanged .to(cuda) per value')


def validate_policy(row,obs):
    counts=row['result']['forward_counts'];n=counts['target_verify_calls'];p=obs['policy']
    require(set(counts)==set(guard.CORE_COUNTERS)|neutral.INSTRUMENTATION,'unknown/missing counters')
    require(all(counts[k]==28*n for k in neutral.INSTRUMENTATION),'28-layer count mismatch')
    require(p['mode']=='grouped' and p['finalized'] and p['failures']==[] and p['prefill_calls']==1
        and p['prefill_forward_calls']==1 and p['prefill_layer_calls']==[1]*28 and p['layer_calls']==[n]*28
        and p['completed_calls']==p['candidate_calls']==28*n and p['verify_forward_calls']==n
        and p['verify_query_tokens']==30*n and p['fallback_calls']==p['native_tree_calls']==0
        and p['original_mask_reused'] and p['position_ids_passed_unchanged'] and not p['rotary_reapplied'], 'grouped coverage mismatch')
    trees=[r for r in row['result']['rounds'] if r['kind']=='tree']
    require(len(p['rounds'])==len(trees)==n,'round coverage missing')
    for event,actual in zip(trees,p['rounds']):
        require(actual==dict(cache_before=event['target_cache_before'],tree_nodes=30,
            cache_after=event['target_cache_before']+30,layer_calls=[1]*28,
            mask_shape=[1,1,30,event['target_cache_before']+30],mask_dtype='torch.float32',
            position_shape=[3,1,30],position_dtype='torch.int64'),'per-round mask/KV/position coverage mismatch')
    require(obs['prefill']==dict(calls=1,wrapper_restored=True),'prefill boundary not restored')


def run_arm(ops,request,saved,cell,baseline_resident,backend):
    """Real measured arm; no input GPU hash or cached GPU packed input."""
    row=dict(cell,request_id=request['request_id'],execution_kind=ops.execution_kind,completed=False,
             pid=os.getpid(),gpu_uuid=ops.args.expected_gpu_uuid,physical_gpu=ops.args.physical_gpu,
             ordinary_TPS_eligible=False,scope=NOTE)
    tracker=neutral.PhaseMemory(ops.torch.cuda);packed=cpu=sampled=None;obs={};refs=[];monitor=None
    try:
        monitor=guard.ProcessMonitor(backend).start()
        row['resident_before']=resident(ops);validate_resident(row['resident_before'],baseline_resident)
        tracker.switch('complete_prepare_CPU')
        cpu,sampled=ops.prepare_cpu(request)
        # CPU-only signature never creates a device clone. Its wall is explicitly diagnostic.
        tracker.switch('CPU_signature_diagnostic');row['input']=input_record(ops,cpu,sampled)
        row['input_exact']=all(row['input'][k]==saved['input'][k] for k in guard.INPUT_KEYS)
        require(row['input_exact'],'CPU staged input differs from C0')
        tracker.switch('GPU_input_materialization');packed=ops.upload(cpu)
        tracker.switch('generation_setup')
        with cache_liveness(ops.bridge,obs) as refs:
            if cell['arm']=='original':result=run_original(ops,packed,tracker,obs)
            elif cell['arm']=='P0':result=neutral.run_generation(ops,packed,request,tracker,obs)
            else:raise ValueError('unknown arm')
        row['output']=result
    except BaseException as exc:
        row.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        # A failed cleanup/identity check must not discard earlier measured peaks.
        def final_step(stage,action):
            try:return action()
            except BaseException as exc:
                row.setdefault('cleanup_errors',[]).append(dict(stage=stage,error=repr(exc)))
        if tracker.current is not None:final_step('cleanup_boundary',lambda:tracker.switch('common_request_cleanup'))
        final_step('common_request_cleanup',ops.cleanup_request)
        if tracker.current is not None:final_step('release_boundary',lambda:tracker.switch('release_input_and_final_sync'))
        packed=cpu=sampled=None
        final_step('collect',gc.collect);final_step('final_sync',ops.torch.cuda.synchronize)
        row['cache_liveness']=dict(observed_prefills=len(refs),live_cache_count=sum(r() is not None for r in refs),
            scope='weak cache shell references, no saved tensor payload')
        after=final_step('resident_read',lambda:resident(ops))
        if after is not None:
            row['resident_after']=after
            final_step('resident_validation',lambda:validate_resident(after,baseline_resident))
        if tracker.current is not None:
            memory=final_step('final_memory',tracker.finish)
            if memory is not None:row['production_memory']=memory
        if 'production_memory' not in row and tracker.started is not None:
            row['partial_allocator_observations']=dict(segments=tracker.rows,
                current=None if tracker.current is None else {k:v for k,v in tracker.current.items() if k!='clock'},
                complete=False,scope='raw readings retained after allocator observation failure; no whole peak qualification')
        if row.get('cleanup_errors'):row['cleanup_error']='common cleanup or final observation failed'
        try:row['background']=monitor.stop() if monitor is not None else dict(unavailable=True)
        except BaseException as exc:row['background_error']=repr(exc)
        row['observations']=obs
    # All row validation and disk work occurs after the measured GPU lifetime.
    try:
        require(not any(k in row for k in ('error','cleanup_error','background_error')),'arm had error; negative retained')
        require(obs.get('prefill_wrapper_restored') is True and row['cache_liveness']==dict(
            observed_prefills=1,live_cache_count=0,scope='weak cache shell references, no saved tensor payload'),'request cache not released')
        g0.validate_result(row['output']);validate_policy(row['output'],obs)
        if cell['arm']=='P0':row['P0_check']=neutral.validate_production(row['output'],obs,request['request_id'])
        row['c0_comparison']=g0.compare_c0(row['output'],saved['row'])
        row['flags_exact']=obs['flags']=={k:saved['precision'][k] for k in ('before','after')}
        require(row['flags_exact'] and row['c0_comparison']['exact'],'input/precision/output/trajectory differs from C0')
        m=row['production_memory']
        row['background_review']=guard.background_review(row['background'],m['started_unix'],m['ended_unix'],
            ops.args.expected_gpu_uuid,ops.args.allow_graphics_pid,os.getpid())
        row['production_memory_scope']=dict(input_GPU_hash_used=False,CPU_hash_wall_in_instrumented_envelope=True,
            snapshots_used=False,CPU_prepare_redone_per_arm=True,common_cleanup_public=True,
            paired_total_memory_accepted=None)
        row['completed']=True
    except BaseException as exc:row['validation_error']=repr(exc)
    return row


def compare_request(rows):
    """Do not subtract starts or let a later baseline compensate for growth."""
    out=dict(complete=False,paired_total_memory_accepted=None,pairs=[],scope=NOTE,evidence_errors=[])
    cells=[{k:r.get(k) for k in ('phase','repeat','arm')} for r in rows]
    fixed_schedule=cells==schedule()
    actual=bool(rows) and all(r.get('execution_kind')==REAL_KIND for r in rows)
    def peak(row,kind):
        value=row.get('production_memory',{}).get('whole_observed',{}).get('peak_'+kind+'_bytes')
        return value if type(value) is int and value>=0 else None
    maxima={arm:{kind:max((peak(r,kind) for r in rows if r.get('arm')==arm and peak(r,kind) is not None),default=None)
             for kind in ('allocated','reserved')} for arm in ('original','P0')}
    coverage={arm:{kind:sum(r.get('arm')==arm and peak(r,kind) is not None for r in rows)
             for kind in ('allocated','reserved')} for arm in ('original','P0')}
    maximum_complete=fixed_schedule and all(n==3 for by_kind in coverage.values() for n in by_kind.values())
    maxima_ok=(all(maxima['P0'][k]<=maxima['original'][k] for k in ('allocated','reserved'))
               if maximum_complete else None)
    growth=maxima_ok is False;actual_growth=growth and actual
    eligible=fixed_schedule and all(r.get('completed') is True for r in rows)
    all_exact=False;global_binding=False
    if eligible:
        try:
            first=rows[0]
            all_exact=all(g0.compare_c0(r['output'],first['output'])['exact'] for r in rows)
            global_binding=all(r['request_id']==first['request_id'] and r['pid']==first['pid']
                and r['gpu_uuid']==first['gpu_uuid'] and r['physical_gpu']==first['physical_gpu']
                and r['resident_before']==r['resident_after']==first['resident_before']
                and r['observations']['flags']==first['observations']['flags']
                and all(r['input'][k]==first['input'][k] for k in guard.INPUT_KEYS)
                and r['input_exact'] is True and r['flags_exact'] is True
                and r['c0_comparison']['exact'] is True
                and (r['arm']=='original' or r['P0_check']['passed'] is True) for r in rows)
            for r in rows:validate_resident(r['resident_before']);validate_resident(r['resident_after'])
        except (KeyError,TypeError,ValueError) as exc:
            global_binding=False;out['evidence_errors'].append(repr(exc))
    eligible&=global_binding and all_exact and maximum_complete
    # Warm starts are disclosed; formal starts must match exactly. Failed arms
    # still contribute raw peaks and can directly reveal a candidate increase.
    for phase,repeats in (('warmup',(0,)),('measure',(0,1))):
        for repeat in repeats:
            aa=[r for r in rows if r.get('phase')==phase and r.get('repeat')==repeat and r.get('arm')=='original']
            bb=[r for r in rows if r.get('phase')==phase and r.get('repeat')==repeat and r.get('arm')=='P0']
            pair=dict(phase=phase,repeat=repeat,peaks_nonincreasing=None,allocator_starts_equal=False,
                binding_exact=False,background_eligible=False)
            # Even a partial/malformed schedule cannot hide an observed increase.
            for a in aa:
                for b in bb:
                    for kind in ('allocated','reserved'):
                        av,bv=peak(a,kind),peak(b,kind)
                        if av is not None and bv is not None and bv>av:
                            growth=True
                            actual_growth|=a.get('execution_kind')==b.get('execution_kind')==REAL_KIND
            if len(aa)==len(bb)==1:
                a,b=aa[0],bb[0]
                values=[(peak(a,k),peak(b,k)) for k in ('allocated','reserved')]
                if all(av is not None and bv is not None for av,bv in values):
                    pair['peaks_nonincreasing']=all(bv<=av for av,bv in values)
                try:
                    pair['allocator_starts_equal']=all(type(a['production_memory']['start'][k+'_bytes']) is int
                        and type(b['production_memory']['start'][k+'_bytes']) is int
                        and a['production_memory']['start'][k+'_bytes']==b['production_memory']['start'][k+'_bytes']
                        for k in ('allocated','reserved'))
                    pair['binding_exact']=(a['resident_before']==b['resident_before']==a['resident_after']==b['resident_after']
                        and a['request_id']==b['request_id'] and a['pid']==b['pid'] and a['gpu_uuid']==b['gpu_uuid']
                        and a['physical_gpu']==b['physical_gpu'] and a['observations']['flags']==b['observations']['flags']
                        and all(a['input'][k]==b['input'][k] for k in guard.INPUT_KEYS))
                    pair['background_eligible']=a['background_review']['eligible'] is True and b['background_review']['eligible'] is True
                except (KeyError,TypeError,ValueError) as exc:out['evidence_errors'].append(repr(exc))
            out['pairs'].append(pair)
            eligible&=(pair['binding_exact'] and pair['background_eligible']
                       and (pair['allocator_starts_equal'] or phase=='warmup'))
    out.update(complete=fixed_schedule,all_arms_completed=all(r.get('completed') is True for r in rows),
        all_full_results_exact=all_exact,global_binding_exact=global_binding,actual_execution_eligible=actual,
        all_arm_maxima_including_warm=maxima,peak_observation_counts=coverage,
        per_request_maxima_nonincreasing=maxima_ok,raw_growth_observed=growth,
        protocol_qualified=eligible,qualification=eligible and actual,
        paired_total_memory_accepted=False if actual_growth else (True if eligible and actual and not growth else None))
    if not fixed_schedule:out['reason']='missing/faulty fixed warm+ABBA schedule'
    elif not all(r.get('completed') is True for r in rows):out['reason']='failed arm; raw peaks retained'
    return out


def summarize(cases,requests):
    neutral.aggregate([],requests)
    complete=len(cases)==24 and [x['request_id'] for x in cases]==[x['request_id'] for x in requests]
    checked=[c['comparison'] for c in cases]
    def observed(arm,kind):
        return [c.get('all_arm_maxima_including_warm',{}).get(arm,{}).get(kind) for c in checked
                if type(c.get('all_arm_maxima_including_warm',{}).get(arm,{}).get(kind)) is int]
    maxima={arm:{kind:max(observed(arm,kind),default=None) for kind in ('allocated','reserved')}
            for arm in ('original','P0')}
    maxima_complete=complete and all(c.get('complete') is True and all(
        c.get('peak_observation_counts',{}).get(arm,{}).get(kind)==3
        for arm in maxima for kind in maxima[arm]) for c in checked)
    maxima_complete&=all(len(observed(arm,kind))==24 for arm in maxima for kind in maxima[arm])
    maxima_ok=(all(maxima['P0'][k]<=maxima['original'][k] for k in ('allocated','reserved'))
               if maxima_complete else None)
    failed=any(r.get('paired_total_memory_accepted') is False for r in checked)
    if maxima_ok is False and all(c.get('actual_execution_eligible') for c in checked):failed=True
    return dict(expected_requests=24,recorded_requests=len(cases),complete=complete,
        per_request=[dict(request_id=c['request_id'],**c['comparison']) for c in cases],batch_maxima=maxima,
        batch_maxima_complete=maxima_complete,batch_maxima_nonincreasing=maxima_ok,
        paired_total_memory_accepted=False if failed else (True if complete and maxima_ok is True and all(
            c.get('paired_total_memory_accepted') is True for c in checked) else None),
        ordinary_TPS_eligible=False,original_author_ViSpec_comparison=False,dynamic_FAD_implemented=False)


def parse_args(argv=None):
    """The same fixed E24 options, with this comparison's own scope text."""
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('out','requests','requests-sha256','c0-dir','expected-gpu-uuid',
                 'execution-source-freeze','execution-source-freeze-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--allow-graphics-pid',type=int,action='append',default=[]);p.add_argument('--c0-project')
    args=p.parse_args(argv)
    require(args.requests_sha256==g0.E24_SHA,'only frozen full E24 supported')
    require(not Path(args.out).exists(),'fresh output required')
    require(args.expected_gpu_uuid.startswith('GPU-')
        and str(uuid.UUID(args.expected_gpu_uuid[4:]))==args.expected_gpu_uuid[4:],'canonical GPU UUID required')
    require(set(args.allow_graphics_pid).issubset({6360}),'only verified Xorg 6360 may be allowlisted')
    args.allow_graphics_pid=sorted(set(args.allow_graphics_pid));return args


def main(argv=None):
    args=parse_args(argv)
    execution=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
    require(digest(args.requests)==g0.E24_SHA,'E24 bytes differ');requests=guard.jsonl(args.requests)
    require(all(r.get('split')=='E24' for r in requests),'wrong split');neutral.aggregate([],requests)
    manifest_path=Path(args.requests).parent/'MANIFEST.json';require(digest(manifest_path)==g0.MANIFEST_SHA,'manifest differs')
    manifest=guard.read_json(manifest_path);reference=g0.load_reference(args.c0_dir,requests)
    c0_project=g0.resolve_c0_project(args.c0_dir,args.c0_project)
    sources=g0.check_sources(reference,manifest,c0_project=c0_project);sources.update(execution['source_sha256'])
    lock,lock_receipt=g0.load_lock_without_torch(PROJECT/'direct_opt/run_joint.py')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan=schedule();save(out/'schedule.json',plan)
    report=dict(schema=SCHEMA,status='starting',pid=os.getpid(),run_id=uuid.uuid4().hex,args=vars(args),
        entry_sha256=digest(__file__),schedule=plan,schedule_sha256=digest(out/'schedule.json'),
        execution_source_freeze=execution,source_sha256=sources,c0_artifacts_sha256=reference['hashes'],
        source_recheck=dict(passed=False),lock_source=lock_receipt,cases=[],prelaunch=[],scope=NOTE)
    def persist():
        report['updated_at']=dt.datetime.now(dt.timezone.utc).isoformat();report['gate']=summarize(report['cases'],requests)
        if report['status']!='completed' or not report['source_recheck']['passed']:
            report['gate']['paired_total_memory_accepted']=None if report['gate']['paired_total_memory_accepted'] is not False else False
        save(out/'status.json',report);save(out/'PRODUCTION_MEMORY_PAIR_REPORT.json',report)
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
                ops=PairOps(args,reference);ops.cleanup_request();baseline=resident(ops);validate_resident(baseline)
                report.update(status='running',setup=ops.metadata,resident_reference=baseline);persist()
                for i,request in enumerate(requests):
                    rows=[];case_dir=out/'cases'/f'{i:02d}';case_dir.mkdir(parents=True,exist_ok=False)
                    for ordinal,cell in enumerate(plan):
                        report['current']=dict(request_id=request['request_id'],ordinal=ordinal,**cell);persist()
                        try:
                            observed=guard.observe_processes(backend)
                            report.setdefault('arm_guards',[]).append(dict(report['current'],observation=observed))
                            guard.verify_idle(observed,args.expected_gpu_uuid,args.allow_graphics_pid,os.getpid())
                            row=run_arm(ops,request,reference['selected'][request['request_id']],cell,baseline,backend)
                        except BaseException as exc:
                            row=dict(cell,request_id=request['request_id'],execution_kind=REAL_KIND,completed=False,
                                error=repr(exc),traceback=traceback.format_exc(),stage='arm_guard_or_entry',
                                ordinary_TPS_eligible=False,scope=NOTE)
                        path=case_dir/f'arm_{ordinal}.json';save(path,row);rows.append(row)
                        guard.append(out/'arms.jsonl',dict(request_id=request['request_id'],ordinal=ordinal,artifact=str(path.relative_to(out)),sha256=digest(path),completed=row['completed']))
                        case=dict(request_id=request['request_id'],comparison=compare_request(rows),
                            artifact_sha256={p.name:digest(p) for p in case_dir.glob('arm_*.json')})
                        if ordinal==0:report['cases'].append(case)
                        else:report['cases'][-1]=case
                        save(case_dir/'case.json',case);persist()
                        if not row['completed']:break
                    require(len(rows)==6 and all(r['completed'] for r in rows),'failed arm, later requests paused; raw retained')
                report.update(status='completed',current=None)
    except BaseException as exc:failure=exc;report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
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
            require(after==execution,'execution source drift')
            final=g0.check_sources(reference,manifest,c0_project=c0_project);final.update(after['source_sha256'])
            require(final==sources and digest(manifest_path)==g0.MANIFEST_SHA and digest(args.requests)==g0.E24_SHA,'source/manifest/input drift')
            for name,h in reference['hashes'].items():require(digest(Path(args.c0_dir)/name)==h,'C0 artifact drift')
            report['source_recheck']=dict(passed=True,execution_inventory_exact=True)
        except BaseException as exc:
            report.update(status='failed',source_recheck=dict(passed=False,error=repr(exc)))
            if failure is None:failure=exc
        persist()
    if failure is not None:raise failure
    return report


if __name__=='__main__':main()
