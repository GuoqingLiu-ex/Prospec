"""Ordinary timing entry with symmetric, read-only precision observations.

Original calls the frozen native/native runner; grouped calls the frozen
shared/compiled/grouped runner. No precision setter, profiler or model operation
is added. Three flag reads and receipt construction before/after each actual
generate_joint are timed in both arms. Sidecar disk writes and final hashes are
outside the measured request; existing status/results files remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from profile_ring_dram import patched

HERE=Path(__file__).resolve().parent
PUBLIC_ENV=('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')
SCHEMA='original_grouped_precision_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_flags(torch):
    return dict(allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        reduced_precision_math_sdp=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--comparison-arm',choices=('original','grouped'),required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--requests',required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--request-id',action='append')
    p.add_argument('--warmup',type=int,choices=(1,),default=1)
    p.add_argument('--repeats',type=int,choices=(2,),default=2)
    p.add_argument('--max-new-tokens',type=int,choices=(512,),default=512)
    p.add_argument('--seed',type=int,choices=(20261001,),default=20261001)
    a=p.parse_args(argv)
    requests=[json.loads(line) for line in Path(a.requests).read_text().splitlines() if line.strip()]
    ids=[r['request_id'] for r in requests]
    if len(ids)!=12 or len(set(ids))!=12:p.error('the fixed manifest must contain twelve unique requests')
    if a.request_id and (len(set(a.request_id))!=len(a.request_id) or not set(a.request_id)<=set(ids)):
        p.error('smoke request IDs must be unique members of the twelve-request manifest')
    if Path(a.out).exists():p.error('fresh output directory required')
    a.selected_ids=[i for i in ids if a.request_id is None or i in a.request_id]
    forwarded=['--out',a.out,'--requests',a.requests,'--physical-gpu',str(a.physical_gpu),
        '--warmup','1','--repeats','2','--max-new-tokens','512','--seed',str(a.seed)]
    for request in a.request_id or []:forwarded+=['--request-id',request]
    return a,forwarded


class PrecisionObserver:
    def __init__(self,torch,args):
        self.torch,self.args=torch,args;self.active=False;self.current=None;self.schedule=None
        self.report=dict(schema=SCHEMA,run_id=uuid.uuid4().hex,pid=os.getpid(),comparison_arm=args.comparison_arm,
            source_sha256=digest(__file__),manifest_sha256=digest(args.requests),selected_request_ids=args.selected_ids,
            smoke_only=args.request_id is not None,abbreviated_recipe=dict(warmup=1,repeats=2,max_new_tokens=512,seed=args.seed),
            default_startup_flags=read_flags(torch),public_environment={k:os.environ.get(k) for k in PUBLIC_ENV},
            calls=[],status='starting',flags_are_read_only=True,precision_setters_added=False,
            scope='Actual generate_joint before/after flag reads and in-memory receipts are included in both arms; '
                  'sidecar writes/hashes are outside request timing. Existing grouped math scope is unchanged.',
            precision_match_claim=None)

    def save(self,path,value):
        if Path(path).name=='schedule.json':self.schedule=[dict(cell) for cell in value]
        elif Path(path).name=='status.json':
            if value.get('pid') not in (None,os.getpid()):raise ValueError('runner PID differs from observer')
            current=value.get('current')
            if current is not None:
                self.current=dict(cell=dict(current),cell_index=value['current_index'])
            self.report['runner_status']=value.get('status')

    def wrap(self,original):
        def generate(*a,**kw):
            index=len(self.report['calls'])
            if self.active or self.current is None or self.schedule is None:
                raise RuntimeError('generation requires one observed scheduled cell and no nesting')
            if index!=self.current['cell_index'] or index>=len(self.schedule) or self.schedule[index]!=self.current['cell']:
                raise RuntimeError('precision observation call does not match actual scheduled cell')
            record=dict(call_index=index,cell_index=index,cell=dict(self.current['cell']),
                before=read_flags(self.torch),generation_completed=False)
            self.report['calls'].append(record);self.active=True
            try:
                result=original(*a,**kw);record['generation_completed']=True;return result
            except BaseException as exc:record['error']=repr(exc);raise
            finally:
                try:record['after']=read_flags(self.torch)
                finally:self.active=False
        return generate

    def flush(self,out,*,final=False):
        out=Path(out)
        if not out.is_dir():return
        if final:
            self.report['exit_flags']=read_flags(self.torch)
            self.report['public_environment_after']={k:os.environ.get(k) for k in PUBLIC_ENV}
            self.report['final_file_sha256']={name:digest(out/name) for name in
                ('status.json','results.jsonl','requests.json','schedule.json') if (out/name).is_file()}
        path=out/'precision_observation.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report,indent=2)+'\n');temporary.replace(path)


def main(argv=None):
    args,forwarded=parse_args(argv)
    import torch
    observer=PrecisionObserver(torch,args)  # default values, before either runner main
    import run_joint
    import vispec_joint_adapter as joint
    original_save=run_joint.save_json
    def save(path,value):
        observer.save(path,value)
        result=original_save(path,value);observer.flush(args.out);return result
    try:
        with ExitStack() as stack:
            stack.enter_context(patched(run_joint,'save_json',save))
            stack.enter_context(patched(joint,'generate_joint',observer.wrap(joint.generate_joint)))
            if args.comparison_arm=='original':
                from run_memory_layout import main as run
                result=run(['--memory-layout','native','--adaptor','native',*forwarded])
            else:
                from run_grouped_gqa_layout import main as run
                result=run(['--tree-attention','grouped',*forwarded])
            expected=1+2*len(args.selected_ids)
            if (observer.active or observer.schedule is None or len(observer.schedule)!=expected
                    or len(observer.report['calls'])!=expected
                    or not all(c['generation_completed'] for c in observer.report['calls'])
                    or observer.report.get('runner_status')!='completed'):
                raise RuntimeError('ordinary run did not complete all precision-observed cells')
            observer.report['status']='completed'
            return result
    except BaseException as exc:
        observer.report.update(status='failed',error=repr(exc));raise
    finally:observer.flush(args.out,final=True)


if __name__=='__main__':main()
