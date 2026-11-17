"""One complete generate_joint Kineto trace via the frozen V-bank runner.

Diagnostic only. One request, warmup1 + repeat2 retained; only formal repeat0
is profiled. No model/Torch import, CUDA work or process launch on import/help.
The frozen auditor remains conservative: Torch2.8 exposes no drop-count API.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid
import warnings

from audit_full_generation_copies import stream_trace, file_hash
from assess_visual_kv import generation
from audit_v_bank_generation import (MODES, arm_name, count_contract, checked,
    candidate_coverage, grouped_coverage, FLAGS, BASE, COMMON_SOURCES, NEW_SOURCES)

_ACTIVE=False
CONFIG=dict(record_shapes=False,profile_memory=False,with_stack=False,with_flops=False,with_modules=False)
SOURCE_NAMES=tuple(dict.fromkeys(COMMON_SOURCES+NEW_SOURCES+(
    'direct_opt/profile_full_generation_copies.py','direct_opt/integration.py',
    'direct_opt/compression.py','hardware_metrics.py')+tuple('direct_opt/'+n+'.py' for n in (
        'audit_full_generation_copies','audit_v_bank_generation','assess_visual_kv',
        'audit_grouped_generation','audit_no_weights_generation','audit_stride0_pilot',
        'audit_native_delta_abba','audit_delta_commit_pairs','summarize_dram_campaign'))))
NOTE='Torch/Kineto diagnostic process; trace instrumentation/allocation/start-stop/export invalidate ordinary latency, energy and memory-peak conclusions.'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


@contextmanager
def patched(owner,name,value):
    old=vars(owner).get(name);had=name in vars(owner)
    setattr(owner,name,value)
    try:yield
    finally:
        if had:setattr(owner,name,old)
        else:delattr(owner,name)


@contextmanager
def stderr_to(path):
    """Capture native Kineto/CUPTI stderr, restoring descriptor even on failure."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    sys.stderr.flush();saved=os.dup(2)
    try:
        with path.open('ab',buffering=0) as output:
            os.dup2(output.fileno(),2)
            try:yield
            finally:sys.stderr.flush()
    finally:
        os.dup2(saved,2);os.close(saved)


def flags(torch):
    return dict(allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        reduced_precision_math_sdp=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))


def signature(result):
    # CPU scalars/lists only, after profiler stop. Never retain model tensors.
    return json.loads(json.dumps(dict(token_ids=result.token_ids,stop_reason=result.stop_reason,
        rounds=result.rounds,forward_counts=dict(result.forward_counts))))


class FullGenerationCollector:
    def __init__(self,torch,out,*,mode,request_id,physical_gpu,expected_uuid,stderr_scope=stderr_to):
        self.torch=torch;self.out=Path(out);self.mode=mode;self.request_id=request_id
        self.physical_gpu=physical_gpu;self.expected_uuid=expected_uuid;self.stderr_scope=stderr_scope
        self.run_id=str(uuid.uuid4());self.nonce=uuid.uuid4().hex
        self.calls=0;self.active=False;self.current=None;self.schedule=None;self.status=None
        self.profiler=None;self.return_signature=None;self.sources={};self.source_payloads={}
        self.source_root=BASE;self.lock_verified=False;self.finalized=False
        self.trace_path=self.out/'full_generation.trace.json';self.log=self.out/'copy_profiler.stderr.log'
        self.capture=dict(started=False,stopped=False,generation_returned=False,synchronized_inside_marker=False,
            warmup_calls=0,profiled_calls=0,formal_call_index=1,dropped_events=None,warnings=[],
            dropped_events_evidence='unavailable: Torch2.8 public profiler and _ProfilerResult expose no dropped-event count; absence of warnings is not zero',
            requested_activities=['CPU','CUDA'],configuration=dict(CONFIG),exported=False)
        self.errors=[];self.python_warnings=[];self.device=None;self.before_flags=None;self.after_flags=None

    def snapshot_sources(self,root=BASE):
        self.source_root=Path(root)
        for name in SOURCE_NAMES:
            data=(Path(root)/name).read_bytes()
            self.source_payloads[name]=data;self.sources[name]=hashlib.sha256(data).hexdigest()

    def observe(self,path,value):
        if Path(path).name=='schedule.json':
            expected=[dict(request_id=self.request_id,arm=arm_name(self.mode),phase=p,repeat=r)
                for p,r in [('warmup',0),('measure',0),('measure',1)]]
            if value!=expected:raise RuntimeError('collector requires exact one-request warm1/repeat2 schedule')
            self.schedule=value
        elif Path(path).name=='status.json':
            self.status=value
            if value.get('current') is not None:self.current=dict(value['current'])
            lock=value.get('lock')
            if lock and (lock.get('pid')!=os.getpid() or lock.get('gpu')!=self.physical_gpu or lock.get('initial_snapshot',{}).get('gpu_uuid')!=self.expected_uuid):
                raise RuntimeError('parent cooperative-lock PID/device/UUID mismatch')
            if lock:self.lock_verified=True

    def metadata(self):
        return dict(run_id=self.run_id,nonce=self.nonce,mode=self.mode,generation_calls=self.calls,
            captured_repeat=0,capture=dict(self.capture),diagnostic_process=True,
            valid_timing=False,valid_memory=False,valid_energy=False,scope='original generate_joint body',
            loss_count_available=False,ordinary_performance_claim=False)

    @contextmanager
    def warnings_scope(self):
        with self.stderr_scope(self.log), warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            try:yield
            finally:self.python_warnings.extend(str(w.message) for w in observed)

    def wrap(self,original):
        @wraps(original)
        def generate(*args,**kwargs):
            if self.active:raise RuntimeError('nested generation collection')
            ordinal=self.calls
            if self.schedule is None or ordinal>=3 or self.current!=self.schedule[ordinal]:
                raise RuntimeError('generation call is not bound to the frozen current cell')
            if not self.lock_verified:raise RuntimeError('parent cooperative lock not observed before generation')
            self.calls+=1
            if ordinal!=1:
                result=original(*args,**kwargs)
                if ordinal==0:self.capture['warmup_calls']+=1
                return result
            if self.capture['profiled_calls'] or self.profiler is not None:raise RuntimeError('duplicate formal capture')
            torch=self.torch
            if torch.autograd._profiler_enabled():raise RuntimeError('an existing profiler is active')
            wanted={torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA}
            if not wanted.issubset(torch.profiler.supported_activities()):raise RuntimeError('CPU+CUDA Kineto activities unavailable; no fallback')
            props=torch.cuda.get_device_properties(0)
            actual=str(props.uuid);actual=actual if actual.startswith('GPU-') else 'GPU-'+actual
            if actual!=self.expected_uuid:raise RuntimeError('actual CUDA UUID differs from selected physical device')
            self.device=dict(uuid=actual,name=props.name,capability=[props.major,props.minor],logical_device=0,physical_device=self.physical_gpu)
            self.before_flags=flags(torch)
            if self.before_flags!=FLAGS:raise RuntimeError('actual math flags differ from fixed grouped/V recipe')
            torch.cuda.synchronize()
            self.profiler=torch.profiler.profile(activities=list(wanted),**CONFIG)
            for key,val in dict(record_shapes=0,profile_memory=0,collector_nonce=self.nonce,collector_run_id=self.run_id).items():
                self.profiler.preset_metadata_json(key,json.dumps(val))
            self.capture['profiled_calls']=1;self.active=True;start_attempted=False
            try:
                with self.warnings_scope():
                    try:
                        start_attempted=True;self.profiler.start();self.capture['started']=True
                        with torch.profiler.record_function('FULL_GENERATION/'+self.nonce):
                            try:
                                result=original(*args,**kwargs);self.capture['generation_returned']=True
                            finally:
                                pending=sys.exc_info()[1]
                                try:
                                    torch.cuda.synchronize();self.capture['synchronized_inside_marker']=True
                                except BaseException as exc:
                                    self.errors.append(dict(stage='capture_sync',error=repr(exc)))
                                    if pending is None:raise
                    finally:
                        if start_attempted:
                            pending=sys.exc_info()[1]
                            try:
                                self.profiler.stop();self.capture['stopped']=True
                            except BaseException as exc:
                                self.errors.append(dict(stage='capture_stop',error=repr(exc)))
                                if pending is None:raise
                self.after_flags=flags(torch)
                if self.after_flags!=self.before_flags:raise RuntimeError('precision flags changed during capture')
                self.return_signature=signature(result)
                return result
            except BaseException as exc:
                self.errors.append(dict(stage='capture',error=repr(exc)));raise
            finally:self.active=False
        return generate

    def finalize(self,*,runner_succeeded):
        """Called after parent loop, sampling and policy cleanup; no request timing."""
        if self.finalized:raise RuntimeError('collector already finalized')
        self.finalized=True
        self.out.mkdir(parents=True,exist_ok=True)
        source_checks={}
        for name,payload in self.source_payloads.items():
            try:source_checks[name]=(file_hash(self.source_root/name)==self.sources[name])
            except OSError:source_checks[name]=False
            dest=self.out/'trace_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
        if not all(source_checks.values()):self.errors.append(dict(stage='source',error='source changed since collector snapshot'))
        if self.profiler is not None and self.capture['stopped']:
            try:
                if self.trace_path.exists():raise FileExistsError('trace output already exists')
                with self.warnings_scope():self.profiler.export_chrome_trace(str(self.trace_path))
                self.capture['exported']=True
            except BaseException as exc:self.errors.append(dict(stage='export',error=repr(exc)))
            finally:self.profiler=None
        # Mark only this fresh diagnostic directory, never ordinary raw datasets.
        rows_path=self.out/'results.jsonl';rows=[];row_checks=[];bound_index=None
        if rows_path.exists():
            rows=[json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
            for index,row in enumerate(rows):
                chosen=(row.get('phase'),row.get('repeat'))==('measure',0)
                row.update(copy_trace_run_id=self.run_id,under_torch_profiler=chosen,diagnostic_process=True,
                    valid_timing=False,valid_memory=False,valid_energy=False)
                row.setdefault('timing_reasons',[]).append(NOTE)
                row['ordinary_memory_scope']=row.get('memory_scope');row['memory_scope']=NOTE
                gen=checked(generation,row,self.status or {})
                cov=checked(grouped_coverage,row,'grouped') if self.mode=='grouped_control' else checked(candidate_coverage,row,self.mode)
                count=checked(count_contract,row,self.mode)
                entry=dict(row_index=index,request_id=row.get('request_id'),phase=row.get('phase'),repeat=row.get('repeat'),
                    generation_valid=gen.get('receipt_valid') is True and gen.get('natural_eos') is True,
                    coverage_valid=cov.get('passed') is True,counters_valid=count.get('passed') is True)
                row_checks.append(entry)
                if chosen:
                    if bound_index is not None:raise RuntimeError('duplicate captured result cell')
                    bound_index=index
                    saved=row['result'];sig=self.return_signature
                    entry['matches_returned_body']=bool(sig and sig['token_ids']==row['output_token_ids'] and sig['stop_reason']==row['stop_reason'] and sig['rounds']==saved['rounds'] and all(saved['forward_counts'].get(k)==v for k,v in sig['forward_counts'].items()))
            temporary=rows_path.with_suffix('.jsonl.tmp')
            temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));temporary.replace(rows_path)
        valid_rows=(len(rows)==3 and [(r.get('request_id'),r.get('arm'),r.get('phase'),r.get('repeat')) for r in rows]==
            [(self.request_id,arm_name(self.mode),p,r) for p,r in [('warmup',0),('measure',0),('measure',1)]] and
            all(all(v for k,v in r.items() if k.endswith('_valid') or k=='matches_returned_body') for r in row_checks))
        contexts=set();metadata={};trace_receipt=None;inventory_error=None
        if self.capture['exported']:
            try:
                for kind,obj in stream_trace(self.trace_path):
                    if kind=='metadata':metadata[obj[0]]=obj[1]
                    elif obj.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and obj.get('ph')=='X':
                        context=obj.get('args',{}).get('context')
                        if isinstance(context,(int,str)) and not isinstance(context,bool):contexts.add(context)
                trace_receipt=dict(sha256=file_hash(self.trace_path),bytes=self.trace_path.stat().st_size)
            except (OSError,ValueError,TypeError) as exc:
                inventory_error=repr(exc);self.errors.append(dict(stage='trace_inventory',error=inventory_error))
        stderr_lines=self.log.read_text(errors='replace').splitlines() if self.log.exists() else []
        signals=[line for line in stderr_lines if re.search(r'warning|error|dropped|overflow|cupti',line,re.I)]
        self.capture['warnings']=self.python_warnings+signals
        status=self.status or {};args=status.get('args',{})
        status_checks=dict(completed=status.get('status')=='completed',
            cells=status.get('completed_cells')==status.get('total_cells')==3,
            process=status.get('pid')==os.getpid(),
            selected_request=args.get('request_id')==[self.request_id],
            selected_arm=args.get('arms')==[arm_name(self.mode)],
            protocol=args.get('warmup')==1 and args.get('repeats')==2 and args.get('max_new_tokens')==512 and args.get('include_ar') is False,
            physical_device=args.get('physical_gpu')==self.physical_gpu)
        capture_ok=bool(runner_succeeded and all(status_checks.values()) and self.calls==3 and valid_rows and source_checks and self.lock_verified and not self.errors and
            all(self.capture[k] is True for k in ('started','stopped','generation_returned','synchronized_inside_marker','exported')))
        sidecar=dict(schema_version=1,scope='generate_joint',nonce=self.nonce,run_id=self.run_id,
            arm=arm_name(self.mode),request_id=self.request_id,repeat=0,phase='measure',
            process=dict(os_pid=os.getpid(),parent_pid=os.getppid()),
            gpu=dict(self.device or {},uuid=self.expected_uuid,physical_device=self.physical_gpu,logical_device=0,
                contexts=sorted(contexts,key=str),contexts_source='observed raw GPU activities; not an independent context attestation'),
            capture=dict(self.capture),trace=trace_receipt,
            sources={'trace_sources/'+k:v for k,v in self.sources.items()},
            result=dict(path='results.jsonl',sha256=file_hash(rows_path) if rows_path.exists() else None,row_index=bound_index),
            generation_body_scope='prefill/convert/Draft/tree/commit; excludes outer result-validation, decode, policy-finalization and input preparation',
            precision_before=self.before_flags,precision_after=self.after_flags,
            input_content_hash_available=False,trace_export_outside_request=True,
            environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CPATH','KINETO_LOG_LEVEL','TORCH_PROFILER_VERBOSE')},
            libraries=dict(torch=getattr(self.torch,'__version__',None),cuda=getattr(getattr(self.torch,'version',None),'cuda',None)),
            errors=list(self.errors))
        save(self.out/'trace_sidecar.json',sidecar)
        final=dict(schema='full_generation_copy_collection_v1',status='collected_loss_count_unavailable' if capture_ok else 'failed_or_incomplete',
            collection_body_completed=capture_ok,full_generation_count_valid=False,
            limitation='No explicit dropped-event counter; frozen auditor must retain partial status. No absent field is synthesized as zero.',
            row_checks=row_checks,source_checks=source_checks,status_checks=status_checks,trace_metadata=metadata,
            trace=trace_receipt,stderr=dict(path=self.log.name,sha256=file_hash(self.log) if self.log.exists() else None),
            capture=dict(self.capture),generation_calls=self.calls,run_id=self.run_id,errors=list(self.errors),
            ordinary_timing_memory_energy_valid=False,physical_DRAM_transfer_count=None)
        save(self.out/'copy_collection.json',final)
        status_path=self.out/'status.json'
        if status_path.exists():
            status=json.loads(status_path.read_text());status['copy_trace']=self.metadata()
            status['copy_trace'].update(collection_body_completed=capture_ok,full_generation_count_valid=False)
            save(status_path,status)
        return final


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--v-storage',choices=MODES,required=True)
    p.add_argument('--requests',required=True);p.add_argument('--request-id',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--expected-gpu-uuid',required=True);p.add_argument('--seed',type=int,choices=(20261001,),default=20261001)
    a=p.parse_args(argv)
    try:
        if not a.expected_gpu_uuid.startswith('GPU-') or str(uuid.UUID(a.expected_gpu_uuid[4:]))!=a.expected_gpu_uuid[4:]:raise ValueError('canonical GPU UUID required')
        if a.out.exists():raise ValueError('fresh output required')
        reqs=[json.loads(x) for x in Path(a.requests).read_text().splitlines() if x.strip()]
        ids=[x['request_id'] for x in reqs]
        if len(ids)!=len(set(ids)) or ids.count(a.request_id)!=1:raise ValueError('one actual unique manifest request required')
    except (OSError,ValueError,KeyError) as exc:p.error(str(exc))
    return a


def main(argv=None):
    global _ACTIVE
    if _ACTIVE:raise RuntimeError('nested full-generation collector')
    args=parse_args(argv)
    import run_joint
    import vispec_joint_adapter as joint
    import run_v_bank_layout as runner
    import torch
    collector=FullGenerationCollector(torch,args.out,mode=args.v_storage,request_id=args.request_id,
        physical_gpu=args.physical_gpu,expected_uuid=args.expected_gpu_uuid)
    collector.snapshot_sources()
    old_save=run_joint.save_json
    def save_wrapper(path,value):
        collector.observe(path,value)
        if Path(path).name=='status.json':value=dict(value,copy_trace=collector.metadata())
        return old_save(path,value)
    success=False;_ACTIVE=True;failure=None
    try:
        with ExitStack() as stack:
            stack.enter_context(patched(run_joint,'save_json',save_wrapper))
            stack.enter_context(patched(joint,'generate_joint',collector.wrap(joint.generate_joint)))
            result=runner.main(['--v-storage',args.v_storage,'--requests',args.requests,'--request-id',args.request_id,
                '--out',str(args.out),'--physical-gpu',str(args.physical_gpu),'--seed',str(args.seed),
                '--warmup','1','--repeats','2','--max-new-tokens','512'])
            success=True
    except BaseException as exc:
        failure=exc;collector.errors.append(dict(stage='runner',error=repr(exc)));raise
    finally:
        try:
            receipt=collector.finalize(runner_succeeded=success)
            if success and not receipt['collection_body_completed']:raise RuntimeError('incomplete diagnostic collection; inspect copy_collection.json')
        except BaseException:
            if failure is None:raise
        finally:_ACTIVE=False
    return receipt


if __name__=='__main__':main()
