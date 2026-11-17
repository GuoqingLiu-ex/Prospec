"""Explicit fixed-V8 model diagnostic, after the new primitive GPU gate.

All error allocation/D2H and artifact writes are timed diagnostic work. No
selector, ordinary-throughput, task-quality or actual-DRAM claim is made.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

MODE='fixed_int8_pack_error'
ARM='memory_shared_embedding_compiled_fullk_v_int8_visual_grouped_math_pack_error_diagnostic'
_ACTIVE=False


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--diagnostic-mode',choices=(MODE,),required=True)
    p.add_argument('--pack-validation',type=Path,required=True)
    args,rest=p.parse_known_args(argv)
    if any(v.split('=',1)[0]=='--v-storage' for v in rest):p.error('diagnostic fixes int8_visual; no v-storage override')
    import run_v_bank_layout as parent
    parent_args,_=parent.parse_args(['--v-storage','int8_visual',*rest])
    return args,parent_args,rest


def sources():
    here=Path(__file__).resolve().parent
    names=('run_v_bank_error_diagnostic.py','v_bank_model_error_runtime.py','v_bank_cuda_error_ops.py',
        'run_v_bank_layout.py','v_bank_model_runtime.py','v_bank_cuda_ops.py','partitioned_v_bank.py','grouped_v_bank_math.py',
        'probe_v_bank_cuda_error.py','probe_v_bank_cuda.py')
    return {name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in names}


def validate_primitive_gate(path):
    path=Path(path);raw=path.read_bytes();report=json.loads(raw)
    if (report.get('schema')!='partitioned_v_pack_error_probe_v1' or report.get('status')!='completed'
            or report.get('numerical_gate_passed') is not True or report.get('cuda_device',{}).get('capability')!=[12,0]
            or report.get('triton_version')!='3.4.0'):
        raise ValueError('completed new primitive GPU error-pack gate required before model diagnostics')
    from probe_v_bank_cuda_error import validate_workload_report
    validate_workload_report(report)
    guards=report.get('error_diagnostics',[])
    if len(guards)!=7 or any(g.get('status')!='passed' for g in guards):raise ValueError('incomplete frozen device guards')
    expected=sources()
    for name in ('v_bank_cuda_error_ops.py','v_bank_cuda_ops.py','partitioned_v_bank.py',
                 'probe_v_bank_cuda_error.py','probe_v_bank_cuda.py'):
        actual=[v for k,v in report.get('source_sha256',{}).items() if Path(k).name==name]
        if actual!=[expected[name]]:raise ValueError('primitive source missing/ambiguous/changed: '+name)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),status=report['status'],
        schema=report['schema'],cuda_device=report['cuda_device'],source_sha256=report['source_sha256'],
        resource_mode=report.get('resource_mode'),scope='prior primitive correctness gate, not model/quality/performance proof')


def main(argv=None):
    global _ACTIVE
    args,parent_args,rest=parse_args(argv)
    if _ACTIVE:raise RuntimeError('nested error diagnostic runner')
    import run_v_bank_layout as parent
    parent.validate_request_selection(parent_args)
    gate=validate_primitive_gate(args.pack_validation)
    source=sources()
    import run_joint
    import v_bank_model_runtime as frozen
    from v_bank_model_error_runtime import v_bank_error_policy
    old_factory,old_arm,old_save=frozen.v_bank_model_policy,parent.arm_name,run_joint.save_json
    current={}
    artifact_dir=Path(parent_args.out)/'pack_error_arrays'

    def factory(target,mode='int8_visual',**kwargs):
        if mode!='int8_visual' or kwargs:raise ValueError('unexpected diagnostic factory mode/options')
        cell=current.get('cell');index=current.get('index')
        if (not isinstance(cell,dict) or not isinstance(cell.get('request_id'),str)
                or cell.get('phase') not in ('warmup','measure') or type(cell.get('repeat')) is not int
                or type(index) is not int or index<0 or cell.get('arm')!=ARM):
            raise RuntimeError('diagnostic artifact lacks current actual schedule cell')
        path=artifact_dir/f'cell_{index:04d}_{cell["phase"]}_r{cell["repeat"]}.pt'
        return v_bank_error_policy(target,mode=mode,artifact_path=path,cell=cell,
            source_sha256=source,pack_validation=gate)

    def arm(mode):
        if mode!='int8_visual':raise ValueError('diagnostic arm must be fixed int8_visual')
        return ARM

    def save(path,value):
        if Path(path).name=='status.json':
            current.clear()
            if value.get('current') is not None:
                current.update(cell=dict(value['current']),index=value.get('current_index'))
            value=dict(value,diagnostic_mode=MODE,diagnostic_source_sha256=source,pack_validation=gate,
                methodology='fixed_v8_actual_model_pack_error_collection',
                diagnostic_artifact_directory=str(artifact_dir.resolve()),
                diagnostic_scope='current-layer V numerical reconstruction error only; no selector or QA-risk labels',
                diagnostic_artifact_disk_write_in_request=True,
                ordinary_throughput_comparison_eligible=False,performance_pass=None,actual_dram_measured=False)
        return old_save(path,value)

    _ACTIVE=True
    try:
        with ExitStack() as stack:
            for owner,name,value in ((frozen,'v_bank_model_policy',factory),(parent,'arm_name',arm),
                    (run_joint,'save_json',save)):
                stack.callback(setattr,owner,name,getattr(owner,name));setattr(owner,name,value)
            return parent.main(['--v-storage','int8_visual',*rest])
    finally:
        current.clear();_ACTIVE=False


if __name__=='__main__':main()
