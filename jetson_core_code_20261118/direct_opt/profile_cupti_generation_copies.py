"""Two loss-aware CUPTI memcpy/memset ranges around actual generate_joint.

One fixed-V8 or host-plan process, 478-token pilot, warm1/repeat2/max512.
No Kineto import, CUDA initialization on import, fallback or performance claim.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from cupti_copy_activity import CopyActivity, audit_file, digest
from assess_visual_kv import generation
from audit_v_bank_generation import count_contract, expected_precision, COMMON_SOURCES, NEW_SOURCES, BASE

REQUEST='direct_quality_vdc_v_sPK-sbHIb3w'
MODES=('fixed_v8','host_plan')
BASE_ARM='memory_shared_embedding_compiled_fullk_v_int8_visual_grouped_math'
PRODUCERS={'fixed_v8':'vispec_joint_adapter.generate_joint','host_plan':'host_tree_joint.generate_joint'}
FLAGS=dict(allow_tf32=False,float32_matmul_precision='highest',reduced_precision_math_sdp=False)
INSTRUMENTATION={'target_tree_attention_policy_calls','target_v_bank_attention_calls'}
NOTE='CUPTI activity diagnostic: all request timing, energy, allocated and reserved peaks are invalid for ordinary acceptance.'
_ACTIVE=False
SOURCE_NAMES=tuple(dict.fromkeys(COMMON_SOURCES+NEW_SOURCES+tuple('direct_opt/'+n for n in (
    'profile_cupti_generation_copies.py','cupti_copy_activity.py','cupti_copy_activity.cpp',
    'run_v_bank_host_plan.py','v_bank_host_plan_runtime.py','host_tree_joint.py',
    'audit_v_bank_generation.py','assess_visual_kv.py','summarize_dram_campaign.py'))))


def require(ok,message):
    if not ok:raise ValueError(message)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


@contextmanager
def patched(module,name,new):
    old=getattr(module,name);setattr(module,name,new)
    try:yield
    finally:setattr(module,name,old)


def arm(mode):
    require(mode in MODES,'unsupported collector mode')
    return BASE_ARM+('_host_plan' if mode=='host_plan' else '')


def flags(torch):
    return dict(allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        reduced_precision_math_sdp=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))


def signature(result):
    # Executed after stopping the range; these original fields are already host data.
    return json.loads(json.dumps(dict(token_ids=result.token_ids,stop_reason=result.stop_reason,
        rounds=result.rounds,forward_counts=dict(result.forward_counts))))


def validate_row(row,status,mode):
    require(row.get('arm')==arm(mode) and row.get('request_id')==REQUEST,'wrong saved arm/request')
    g=generation(row,status)
    require(g['receipt_valid'] and g['natural_eos'],'invalid trajectory or non-natural EOS: '+str(g.get('trace_validation_error')))
    require(count_contract(row,'int8_visual')['passed'],'public forward counter contract failed')
    expected=expected_precision('int8_visual')
    if mode=='host_plan':expected=dict(expected,target_v_bank_metadata_path='author_cpu_tree_reuse_v1')
    require(row['result']['precision']==expected,'fixed V8 precision identity differs')
    info=row['adaptor_preparation']['v_bank'];counts=row['result']['forward_counts']
    trees=[r for r in row['result']['rounds'] if r['kind']=='tree'];n=len(trees)
    require(info['variant']==info['target_cache_policy']=='full_k_partitioned_visual_v_grouped_math'
        and info['mode']=='int8_visual' and info['reference_only'] is False,'wrong V-bank runtime')
    require(info['finalized'] is True and info['result_coverage_validated'] is True and not info['failures']
        and info['fallback_calls']==0,'runtime not cleanly finalized')
    require(info['prefill_calls']==info['prefill_forward_calls']==1 and info['prefill_layer_calls']==[1]*28,'prefill coverage')
    require(info['verify_forward_calls']==n==counts['target_verify_calls'] and info['layer_calls']==[n]*28,
        'verification coverage')
    require(info['completed_calls']==info['selected_calls']==info['candidate_calls']==info['fill_calls']==info['update_calls']==28*n
        and info['commit_calls']==n and counts['target_tree_attention_policy_calls']==counts['target_v_bank_attention_calls']==28*n,
        'all-layer/counter coverage')
    require(info['original_mask_reused'] is True and info['position_ids_passed_unchanged'] is True
        and info['rotary_reapplied'] is False and info['full_k_native_cat_index_select'] is True,'cache/mask/position recipe')
    require(len(info['rounds'])==n,'missing runtime rounds')
    for rr,tree in zip(info['rounds'],trees):
        require(rr['cache_before']==tree['target_cache_before'] and rr['tree_nodes']==tree['tree_nodes']
            and rr['cache_after']==tree['target_cache_before']+tree['tree_nodes']
            and rr['cache_after_commit']==tree['target_cache_after']
            and rr['selected_tree_indices']==tree['selected_tree_indices']
            and rr['layer_calls']==rr['update_counts']==[1]*28,'runtime tree receipt mismatch')
    if mode=='host_plan':
        reuse=info.get('host_tree_reuse',{})
        require(reuse.get('method')=='author_cpu_tree_reuse_v1' and reuse.get('result_coverage_validated') is True
            and all(reuse.get(k)==n for k in ('captured_trees','bound_forwards','consumed_forwards','accepted_paths','committed_paths'))
            and reuse.get('no_GPU_metadata_readback') is True,'host-plan runtime receipt missing')
        require(len(reuse.get('rounds',[]))==n and all(
            h['base']==t['target_cache_before'] and h['nodes']==t['tree_nodes'] and h['accepted']==t['selected_tree_indices']
            for h,t in zip(reuse['rounds'],trees)),'host accepted-path receipt mismatch')
    else:require('host_tree_reuse' not in info,'unexpected host-plan runtime')
    return dict(natural_eos=True,full_saved_trajectory_valid=True,all_layer_coverage_valid=True,
        full_forward_counts=dict(counts),output_token_ids=list(row['output_token_ids']),
        all_rounds=row['result']['rounds'],text=row['text'])


class CuptiGenerationCollector:
    def __init__(self,torch,out,*,mode,physical_gpu,expected_uuid,library,activity_factory=CopyActivity):
        require(mode in MODES,'unsupported collector mode')
        self.torch=torch;self.out=Path(out);self.mode=mode;self.gpu=physical_gpu;self.uuid=expected_uuid
        self.library=Path(library).resolve();self.factory=activity_factory;self.activity=None
        self.nonce=uuid.uuid4().hex;self.pid=os.getpid();self.current=None;self.schedule=None;self.status={}
        self.calls=0;self.entered=False;self.closed=False;self.lock_verified=False;self.ranges=[];self.errors=[]
        self.signatures=[];self.producer_calls={v:0 for v in PRODUCERS.values()};self.device=None
        self.sources={};self.source_payloads={};self.source_root=BASE

    def snapshot_sources(self,root=BASE):
        self.source_root=Path(root)
        for name in SOURCE_NAMES:
            data=(self.source_root/name).read_bytes();self.source_payloads[name]=data
            self.sources[name]=hashlib.sha256(data).hexdigest()
        self.sources['compiled_collector_library']=digest(self.library)

    def expected_schedule(self):
        return [dict(request_id=REQUEST,arm=arm(self.mode),phase=p,repeat=r)
            for p,r in (('warmup',0),('measure',0),('measure',1))]

    def observe(self,path,value):
        if Path(path).name=='schedule.json':
            require(value==self.expected_schedule(),'exact pilot warm1/repeat2 schedule required');self.schedule=value
        elif Path(path).name=='status.json':
            self.status=value
            if value.get('current') is not None:self.current=dict(value['current'])
            lock=value.get('lock')
            if lock:
                require(lock.get('pid')==self.pid and lock.get('gpu')==self.gpu
                    and lock.get('initial_snapshot',{}).get('gpu_uuid')==self.uuid,'parent lock PID/device/UUID mismatch')
                self.lock_verified=True

    def metadata(self):
        return dict(schema='cupti_full_generation_copies_v1',nonce=self.nonce,pid=self.pid,mode=self.mode,
            generation_calls=self.calls,producer_calls=dict(self.producer_calls),range_count=len(self.ranges),
            diagnostic_process=True,valid_timing=False,valid_memory=False,valid_energy=False,
            scope='actual producer generate_joint body',GPU_copy_kernel_count_measured=False,
            DRAM_measured=False,all_transfer_count_claim=False,throughput_claim=False)

    def wrap(self,original,producer):
        require(producer in PRODUCERS.values(),'unregistered producer')
        @wraps(original)
        def call(*a,**kw):
            require(not self.entered and not self.closed,'nested or finalized collector')
            require(producer==PRODUCERS[self.mode],'wrong actual generation producer')
            ordinal=self.calls
            require(self.schedule==self.expected_schedule() and ordinal<3 and self.current==self.schedule[ordinal],
                'generation is not bound to expected current cell')
            require(self.lock_verified,'parent cooperative lock not observed')
            actual=self.torch.cuda.get_device_properties(0)
            observed=str(actual.uuid);observed=observed if observed.startswith('GPU-') else 'GPU-'+observed
            require(observed==self.uuid and (actual.major,actual.minor)==(12,0),'actual CUDA UUID/SM mismatch')
            before=flags(self.torch);require(before==FLAGS,'actual math flags differ')
            self.device=dict(uuid=observed,logical_device=0,physical_device=self.gpu,capability=[actual.major,actual.minor])
            self.calls+=1;self.producer_calls[producer]+=1;self.entered=True
            capture=None
            try:
                if ordinal:
                    if self.activity is None:
                        self.activity=self.factory(self.library,self.out/'activity.jsonl')
                        self.activity.initialize()
                    capture=dict(id=ordinal,call_index=ordinal,cell=dict(self.current),producer=producer,
                        started=False,stopped=False,generation_returned=False,precision_before=before)
                    self.ranges.append(capture)
                    self.activity.begin(self.torch,ordinal);capture['started']=True
                try:
                    result=original(*a,**kw)
                    if capture is not None:capture['generation_returned']=True
                finally:
                    if capture is not None and capture['started']:
                        pending=sys.exc_info()[1]
                        try:self.activity.end(self.torch);capture['stopped']=True
                        except BaseException as exc:
                            self.errors.append(dict(stage='range_end',error=repr(exc)))
                            if pending is None:raise
                after=flags(self.torch);require(after==before,'precision changed inside generation')
                if capture is not None:capture['precision_after']=after
                self.signatures.append(signature(result))
                return result
            except BaseException as exc:
                self.errors.append(dict(stage='generation',call_index=ordinal,error=repr(exc)));raise
            finally:self.entered=False
        return call

    def finalize(self,*,runner_succeeded):
        require(not self.closed,'collector already finalized');self.closed=True
        self.out.mkdir(parents=True,exist_ok=True)
        if self.activity is not None:
            try:self.activity.close()
            except BaseException as exc:self.errors.append(dict(stage='collector_close',error=repr(exc)))
        checks={}
        for name,payload in self.source_payloads.items():
            try:checks[name]=digest(self.source_root/name)==self.sources[name]
            except OSError:checks[name]=False
            dest=self.out/'collector_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
        try:checks['compiled_collector_library']=digest(self.library)==self.sources.get('compiled_collector_library')
        except OSError:checks['compiled_collector_library']=False
        raw=None
        if (self.out/'activity.jsonl').exists():
            try:raw=audit_file(self.out/'activity.jsonl',expected_pid=self.pid,expected_device=0)
            except BaseException as exc:self.errors.append(dict(stage='activity_audit',error=repr(exc)))
        rows_path=self.out/'results.jsonl';rows=[];row_checks=[]
        if rows_path.exists():
            rows=[json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
            for index,row in enumerate(rows):
                row.update(under_cupti_activity=row.get('phase')=='measure',cupti_copy_nonce=self.nonce,
                    diagnostic_process=True,valid_timing=False,valid_memory=False,valid_energy=False)
                row.setdefault('timing_reasons',[]).append(NOTE)
                row['ordinary_memory_scope']=row.get('memory_scope');row['memory_scope']=NOTE
                entry=dict(row_index=index,passed=False)
                try:
                    entry.update(validate_row(row,self.status,self.mode))
                    sig=self.signatures[index]
                    require(sig['token_ids']==row['output_token_ids'] and sig['stop_reason']==row['stop_reason']
                        and sig['rounds']==row['result']['rounds'],'original-body return does not match saved row')
                    final_counts=row['result']['forward_counts'];body_counts=sig['forward_counts']
                    require(set(final_counts)==set(body_counts)|INSTRUMENTATION and not (set(body_counts)&INSTRUMENTATION)
                        and all(final_counts[k]==v for k,v in body_counts.items()),'body/common counters altered outside producer')
                    entry['passed']=True
                except (ValueError,KeyError,TypeError,IndexError) as exc:entry['error']=repr(exc)
                row_checks.append(entry)
            tmp=rows_path.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(rows_path)
        expected=self.expected_schedule()
        cells=[{k:r.get(k) for k in ('request_id','arm','phase','repeat')} for r in rows]
        status=self.status;a=status.get('args',{})
        status_checks=dict(completed=status.get('status')=='completed',pid=status.get('pid')==self.pid,
            cells=status.get('completed_cells')==status.get('total_cells')==3,
            recipe=a.get('request_id')==[REQUEST] and a.get('arms')==[arm(self.mode)]
                and a.get('warmup')==1 and a.get('repeats')==2 and a.get('max_new_tokens')==512
                and a.get('physical_gpu')==self.gpu and a.get('include_ar') is False and a.get('resume') is False,
            deployment=status.get('memory_layout')=='shared_embedding' and status.get('deployment_adaptor')=='compiled'
                and status.get('v_storage')=='int8_visual' and status.get('attention_backend_fallback') is False,
            schedule=cells==expected and self.schedule==expected,
            generation_calls=self.calls==3 and len(self.signatures)==3,
            producer=self.producer_calls=={v:(3 if v==PRODUCERS[self.mode] else 0) for v in PRODUCERS.values()},
            parent_lock=self.lock_verified)
        if self.mode=='host_plan':status_checks['host_identity']=status.get('host_tree_metadata_path')=='author_cpu_tree_reuse_v1'
        else:status_checks['host_identity']='host_tree_metadata_path' not in status
        repeat_checks={k:len(rows)==3 and all(r.get(k)==rows[0].get(k) for r in rows[1:])
            for k in ('input_shapes','video_grid_thw','timestamps','output_token_ids','text','stop_reason')}
        repeat_checks['full_rounds']=len(rows)==3 and all(r['result']['rounds']==rows[0]['result']['rounds'] for r in rows[1:])
        repeat_checks['all_forward_counts']=len(rows)==3 and all(r['result']['forward_counts']==rows[0]['result']['forward_counts'] for r in rows[1:])
        status_checks['same_prepared_inputs']=all(repeat_checks[k] for k in ('input_shapes','video_grid_thw','timestamps'))
        binding=[]
        if raw is not None:
            try:
                require(len(raw['recorded_ranges'])==len(self.ranges)==2,'exactly two recorded formal ranges required')
                for i,(rr,cap,row) in enumerate(zip(raw['recorded_ranges'],self.ranges,rows[1:])):
                    require(rr['id']==cap['id']==i+1 and cap['call_index']==i+1
                        and cap['cell']==expected[i+1] and cap['producer']==PRODUCERS[self.mode]
                        and cap['started'] and cap['stopped'] and cap['generation_returned']
                        and cap['precision_before']==cap['precision_after']==FLAGS,'range/cell/producer mismatch')
                    binding.append(dict(range=rr,producer_capture=cap,row_index=i+1,
                        output_token_ids=row['output_token_ids'],rounds=row['result']['rounds'],
                        forward_counts=row['result']['forward_counts'],text=row['text'],stop_reason=row['stop_reason'],
                        input_shapes=row['input_shapes'],video_grid_thw=row['video_grid_thw'],timestamps=row['timestamps']))
            except (ValueError,KeyError,TypeError,IndexError) as exc:self.errors.append(dict(stage='range_binding',error=repr(exc)))
        complete=bool(runner_succeeded and checks and all(checks.values()) and all(status_checks.values())
            and len(row_checks)==3 and all(x['passed'] for x in row_checks) and len(binding)==2 and not self.errors)
        receipt=dict(**self.metadata(),status='completed' if complete else 'failed_or_incomplete',
            collection_body_completed=complete,enabled_activity_counters_bound=complete,
            source_sha256=dict(self.sources),source_checks=checks,collector_library=str(self.library),
            CXX_build_provenance_scope='C++ source and SO each hashed; build linkage attested by external parent build receipt',
            device=self.device,status_checks=status_checks,row_checks=row_checks,repeat_checks=repeat_checks,activity_audit=raw,
            range_bindings=binding,errors=list(self.errors),
            measured_body_scope='original producer body: visual/prefill/conversion/Draft/Target/commit; excludes input prep, outer result validation, output decode and policy finalization',
            excluded_profile_artifact_work='close, audit, source archive and JSON finalization occur after parent runner',
            total_Target_Draft_memory_acceptance=False,
            memory_rule='ordinary total allocated and reserved must not increase; profiler rows cannot satisfy this rule',
            environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CPATH')},
            libraries=dict(torch=getattr(self.torch,'__version__',None),cuda=getattr(getattr(self.torch,'version',None),'cuda',None)),
            artifact_sha256={name:digest(self.out/name) for name in ('results.jsonl','requests.json','schedule.json','activity.jsonl') if (self.out/name).exists()})
        save(self.out/'cupti_generation_copies.json',receipt)
        sp=self.out/'status.json'
        if sp.exists():
            recorded=json.loads(sp.read_text());recorded['cupti_copy_collection']=dict(self.metadata(),
                collection_body_completed=complete,enabled_activity_counters_bound=complete)
            save(sp,recorded)
        return receipt


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--request-id',choices=(REQUEST,),default=REQUEST)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--expected-gpu-uuid',required=True)
    p.add_argument('--collector-lib',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        require(a.expected_gpu_uuid=='GPU-'+str(uuid.UUID(a.expected_gpu_uuid.removeprefix('GPU-'))),'canonical UUID required')
        require(not a.out.exists(),'fresh output required');require(a.collector_lib.is_file(),'compiled collector SO required')
        rows=[json.loads(x) for x in a.requests.read_text().splitlines() if x.strip()];ids=[r['request_id'] for r in rows]
        require(len(set(ids))==len(ids) and ids.count(REQUEST)==1,'unique actual pilot request required')
    except (OSError,ValueError,KeyError) as exc:p.error(str(exc))
    return a


def main(argv=None):
    global _ACTIVE
    require(not _ACTIVE,'nested CUPTI generation wrapper')
    args=parse_args(argv)
    import run_joint
    import vispec_joint_adapter as joint
    import host_tree_joint as reused
    import run_v_bank_layout as fixed_runner
    import run_v_bank_host_plan as host_runner
    import torch
    collector=CuptiGenerationCollector(torch,args.out,mode=args.mode,physical_gpu=args.physical_gpu,
        expected_uuid=args.expected_gpu_uuid,library=args.collector_lib)
    collector.snapshot_sources();old_save=run_joint.save_json;success=False;failure=None;_ACTIVE=True
    def observe(path,value):
        collector.observe(path,value)
        if Path(path).name=='status.json':value=dict(value,cupti_copy_collection=collector.metadata())
        return old_save(path,value)
    try:
        with ExitStack() as stack:
            stack.enter_context(patched(run_joint,'save_json',observe))
            stack.enter_context(patched(joint,'generate_joint',collector.wrap(joint.generate_joint,PRODUCERS['fixed_v8'])))
            stack.enter_context(patched(reused,'generate_joint',collector.wrap(reused.generate_joint,PRODUCERS['host_plan'])))
            forwarded=['--v-storage','int8_visual','--requests',str(args.requests),'--request-id',REQUEST,
                '--out',str(args.out),'--physical-gpu',str(args.physical_gpu),'--seed','20261001',
                '--warmup','1','--repeats','2','--max-new-tokens','512']
            if args.mode=='fixed_v8':fixed_runner.main(forwarded)
            else:host_runner.main(['--host-tree-plan','reuse',*forwarded])
            success=True
    except BaseException as exc:
        failure=exc;collector.errors.append(dict(stage='runner',error=repr(exc)));raise
    finally:
        try:
            receipt=collector.finalize(runner_succeeded=success)
            if success:require(receipt['collection_body_completed'],'CUPTI generation collection incomplete; inspect retained receipts')
        except BaseException:
            if failure is None:raise
        finally:_ACTIVE=False
    return receipt


if __name__=='__main__':main()
