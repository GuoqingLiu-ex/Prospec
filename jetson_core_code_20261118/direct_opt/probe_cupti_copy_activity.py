"""Small shared-device diagnostic of loss-aware GPU copy activity recording.

Exactly one explicit H2D, D2D, D2H and memset are issued through CUDA12.8
runtime APIs. No model, replay, DRAM counter or timing comparison. Imports/help
do not initialize CUDA or load the collector library.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import uuid

import probe_v_bank_cuda as guards
from cupti_copy_activity import CopyActivity, audit_file, digest

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--gpu', type=int, choices=range(8), required=True)
    p.add_argument('--expected-gpu-uuid', required=True)
    p.add_argument('--collector-lib', type=Path, required=True)
    p.add_argument('--cudart-lib', type=Path, default=Path('/usr/local/cuda-12.8/lib64/libcudart.so.12'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-graphics-pid', type=int, choices=(6360,), action='append', default=[])
    p.add_argument('--allow-shared', dest='allow_shared_validation', action='store_true')
    a = p.parse_args(argv)
    try:
        if 'GPU-' + str(uuid.UUID(a.expected_gpu_uuid.removeprefix('GPU-'))) != a.expected_gpu_uuid:
            raise ValueError('canonical GPU UUID required')
        if a.out.exists():
            raise ValueError('fresh output directory required')
        if not a.collector_lib.is_file() or not a.cudart_lib.is_file():
            raise ValueError('explicit compiled collector and CUDA12.8 runtime libraries required')
    except ValueError as exc:
        p.error(str(exc))
    return a


def run(a):
    import torch
    from run_joint import cooperative_lock
    sys.path.insert(0, str(HERE.parent))
    from hardware_metrics import _make_backend

    guards.require(not torch.cuda.is_initialized(), 'CUDA initialized before cooperative lock')
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='cupti_copy_activity_smoke_v1', status='starting', pid=os.getpid(),
                  started=time.time(), gpu=a.gpu, expected_gpu_uuid=a.expected_gpu_uuid,
                  shared=a.allow_shared_validation, model_loaded=False, DRAM_measured=False,
                  timing_comparison=False, guards=[], environment_before=guards.environment(),
                  source_sha256={str(p.resolve()):digest(p) for p in
                    (Path(__file__), HERE/'cupti_copy_activity.py', HERE/'cupti_copy_activity.cpp', a.collector_lib)},
                  source_scope='probe/binding/C++/compiled binary; parent source manifest preserves guard dependencies')
    def persist(): guards.save(out/'summary.json', report)
    persist(); backend = monitor = collector = None
    old_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    before = guards.flags(torch); report['math_flags_before'] = before
    try:
        with cooperative_lock(a.gpu, out) as lock:
            report['lock'] = dict(lock); backend = _make_backend(a.gpu)
            for i in range(2):
                row = guards.probe_observation(backend, a)
                row['index'] = i; report['guards'].append(row); persist()
                guards.require(row['gpu_uuid'] == a.expected_gpu_uuid, 'physical UUID mismatch')
                guards.validate_observation(row, a, startup=True)
                if i == 0: time.sleep(2)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
            props = torch.cuda.get_device_properties(0)
            actual = 'GPU-' + str(uuid.UUID(str(props.uuid).removeprefix('GPU-')))
            guards.require(actual == a.expected_gpu_uuid, 'CUDA UUID mismatch')
            guards.require((props.major,props.minor) == (12,0), 'SM120 required')
            report['cuda_device'] = dict(uuid=actual, name=props.name, capability=[props.major,props.minor])
            monitor = guards.ProcessMonitor(backend, a, out/'process_observations.jsonl').start()
            cuda = ctypes.CDLL(str(a.cudart_lib))
            cuda.cudaMemcpy.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
            cuda.cudaMemset.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
            cuda.cudaDeviceSynchronize.argtypes = []
            for name in ('cudaMemcpy','cudaMemset','cudaDeviceSynchronize'):
                getattr(cuda,name).restype = ctypes.c_int
            def check(code): guards.require(code == 0, 'CUDA runtime smoke operation failed: '+str(code))
            with torch.cuda.device(0),torch.inference_mode():
                source = torch.arange(16,dtype=torch.uint8).pin_memory()
                result = torch.empty(16,dtype=torch.uint8,pin_memory=True)
                left = torch.empty(16,dtype=torch.uint8,device='cuda:0')
                right = torch.empty_like(left)
                torch.cuda.synchronize()
                collector = CopyActivity(a.collector_lib, out/'activity.jsonl')
                collector.initialize(); collector.begin(torch,1)
                try:
                    check(cuda.cudaMemcpy(left.data_ptr(),source.data_ptr(),16,1))
                    check(cuda.cudaMemcpy(right.data_ptr(),left.data_ptr(),16,3))
                    check(cuda.cudaMemcpy(result.data_ptr(),right.data_ptr(),16,2))
                    check(cuda.cudaMemset(right.data_ptr(),7,16))
                    check(cuda.cudaDeviceSynchronize())
                finally:
                    collector.end(torch)
                guards.require(torch.equal(result,source), 'H2D/D2D/D2H output mismatch')
                collector.close(); collector = None
            report['activity_audit'] = audit_file(out/'activity.jsonl',expected_pid=os.getpid(),expected_device=0)
            r = report['activity_audit']['recorded_ranges'][0]
            guards.require(r['memcpy_records'] == r['memcpy_operations'] == 3 and r['memcpy_bytes'] == 48
                           and r['memset_records'] == 1 and r['memset_bytes'] == 16,
                           'explicit smoke operations did not produce exact activity counts')
            guards.require(r['directions'] == {str(k):dict(records=1,operations=1,bytes=16) for k in (1,2,8)},
                           'CUPTI H2D/D2H/D2D direction values differ from CUDA12.8 definitions')
            report['process_monitor'] = monitor.stop(); monitor.check(); monitor = None
            report.update(status='completed', explicit_operations_verified=True,
                          GPU_activity_count_gate_passed=True, scope='four explicit operations only')
    except BaseException as exc:
        report.update(status='failed',error=repr(exc)); raise
    finally:
        if collector is not None:
            try: collector.close()
            except BaseException as exc: report['collector_close_error'] = repr(exc)
        if monitor is not None:
            report['process_monitor'] = monitor.stop()
        if backend is not None: backend.close()
        if old_visible is None: os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else: os.environ['CUDA_VISIBLE_DEVICES'] = old_visible
        report.update(finished=time.time(),math_flags_after=guards.flags(torch),environment_after=guards.environment())
        if report['math_flags_after'] != before or report['environment_after'] != report['environment_before']:
            report.update(status='failed', restoration_error='inherited flags/environment changed')
        persist()
    guards.require(report['status'] == 'completed', 'copy activity smoke failed')
    return report


if __name__ == '__main__':
    run(parse_args())
