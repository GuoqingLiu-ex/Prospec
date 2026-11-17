"""Single-process visual-KV NCU supervisor; importing never touches a GPU.

Explicit int8 or fp16_visual, one warmup and two generation ranges. The child
owns the existing cooperative GPU lock. Existing compute processes are refused;
resident graphics require an explicit allowlist and remain in the attribution
limits. No package install, GPU settings change, fallback, or automatic retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from run_active_dram_campaign import observe, retain_workload_pid, save, verify_guard
from profile_visual_kv_dram import (MODES, REQUEST, VARIANT, bind_ncu,
                                    compare_reference, validate_workload)

HERE = Path(__file__).resolve().parent
HEADER_VERSION = '3.12.3-1ubuntu0.13'
HEADER_PACKAGE = 'libpython3.12-dev_'+HEADER_VERSION+'_amd64.deb'
HEADER_SHA256 = '7f7941cb968b86454b1d064574e35a9e216f2cfd2f95c5e04cf81a6f5f78f1b2'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', required=True)
    parser.add_argument('--visual-kv-mode', choices=MODES, required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--scope', choices=('generation',), default='generation')
    parser.add_argument('--warmup', type=int, choices=(1,), default=1)
    parser.add_argument('--repeats', type=int, choices=(2,), default=2)
    parser.add_argument('--requests', default=str(HERE.parents[1]/'coord/quality_requests.jsonl'))
    parser.add_argument('--request-id', default=REQUEST)
    parser.add_argument('--ncu', default='/usr/local/cuda/bin/ncu')
    parser.add_argument('--allow-graphics-pid', type=int, action='append', default=[])
    parser.add_argument('--header-environment', default=str(HERE.parents[1]/'kv_int8_primitive_r3/header_environment.json'))
    parser.add_argument('--reference-summary', help='Optional existing native_full/optimized_full generation DRAM JSON')
    args = parser.parse_args(argv)
    if args.gpu < 0 or any(pid < 1 for pid in args.allow_graphics_pid):
        parser.error('GPU index must be nonnegative and graphics PIDs positive')
    args.allow_graphics_pid = sorted(set(args.allow_graphics_pid))
    return args


def arm_name(mode):
    return 'visualkv_'+mode


def command(args, arm):
    return [args.ncu, '--config-file', 'off', '--target-processes', 'application-only',
        '--replay-mode', 'app-range', '--cache-control', 'none', '--clock-control', 'none',
        '--pipeline-boost-state', 'dynamic', '--disable-extra-suffixes',
        '--metrics', 'dram__bytes_read.sum,dram__bytes_write.sum',
        '--export', str(arm/'dram'), '--csv', '--page', 'raw', '--log-file', str(arm/'metrics.csv'),
        sys.executable, str(HERE/'profile_visual_kv_dram.py'),
        '--profile-scope', 'generation', '--profile-calls', '2',
        '--visual-kv-mode', args.visual_kv_mode,
        '--memory-layout', 'shared_embedding', '--adaptor', 'compiled',
        '--out', str(arm/'workload'), '--physical-gpu', str(args.gpu),
        '--requests', str(Path(args.requests).resolve()), '--request-id', args.request_id,
        '--warmup', '1', '--repeats', '2', '--max-new-tokens', '512']


def header_environment(path, *, runtime_version=None):
    """Reuse the recorded, CPU-tested .13 headers; never download/install here."""
    source = Path(path)
    data = json.loads(source.read_text())
    expected = dict(package=HEADER_PACKAGE, sha256=HEADER_SHA256,
                    header_package_version=HEADER_VERSION, runtime_package_version=HEADER_VERSION,
                    system_installation_modified=False)
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError('Header environment receipt mismatch: '+key)
    if tuple(runtime_version if runtime_version is not None else sys.version_info[:3]) != (3, 12, 3):
        raise ValueError('Profiling must use the CPU-validated Python 3.12.3 environment')
    extension = data.get('cpu_extension', {})
    if extension.get('header_version_hex') != '0x30c03f0' or extension.get('runtime_version_hex') != '0x30c03f0':
        raise ValueError('Missing matching CPU extension import receipt')
    parts = [Path(p) for p in data.get('CPATH', '').split(':') if p]
    if len(parts) != 3 or any(not p.is_absolute() or not p.is_dir() for p in parts):
        raise ValueError('Exact three existing absolute CPATH directories required')
    root = parts[0].parents[2]
    if parts != [root/'usr/include/python3.12', root/'usr/include', root/'usr/include/x86_64-linux-gnu/python3.12']:
        raise ValueError('CPATH does not describe one amd64 Python 3.12 header root')
    files = [parts[0]/name for name in ('Python.h', 'patchlevel.h', 'pyconfig.h')]+[parts[2]/'pyconfig.h']
    if any(not p.is_file() for p in files):
        raise ValueError('Required Python or architecture header missing')
    # Record actual bytes used, in addition to the source-package validation
    # receipt. The module does not claim that a copied JSON proves file identity.
    return dict(data, receipt_path=str(source.resolve()),
        receipt_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        observed_header_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='Existing root-verified package/CPU-import receipt plus current header-file hashes; no package installation')


def source_hashes():
    names = ('run_visual_kv_dram_campaign.py', 'profile_visual_kv_dram.py', 'profile_ring_dram.py',
             'run_active_dram_campaign.py', 'assess_visual_kv.py', 'summarize_dram_campaign.py',
             'run_visual_kv_layout.py', 'visual_kv_runtime.py', 'visual_kv_int8.py', 'triton_mixed_attention.py')
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}


def run_campaign(args, *, make_backend, run_process=subprocess.run, sleep=time.sleep):
    if not Path(args.requests).is_file():
        raise FileNotFoundError('Request manifest missing: '+args.requests)
    headers = header_environment(args.header_environment)
    if args.reference_summary and not Path(args.reference_summary).is_file():
        raise FileNotFoundError('Reference summary missing: '+args.reference_summary)
    out = Path(args.out)
    out.mkdir(exist_ok=False, parents=True)
    name = arm_name(args.visual_kv_mode)
    status = dict(status='starting', scope='generation', variant=VARIANT,
        visual_kv_mode=args.visual_kv_mode, gpu=args.gpu, pid=os.getpid(), started_at=time.time(),
        completed=[], allowed_resident_graphics_pids=args.allow_graphics_pid,
        protocol=dict(request_id=args.request_id, warmup=1, measure=2, max_new_tokens=512),
        lock_owner='child runner only', process_poll_interval_s=0.5,
        profile_scope='generate_joint: visual/prefill, bank conversion and generation; model load, input preparation, output decode and runtime finally excluded',
        background_subtracted=False, selected_paths_required_to_match_native=False, quality_pass=None,
        header_environment=headers, source_sha256=source_hashes(), guards=[])
    save(out/'campaign.json', status)
    backend, thread = None, None
    stop, ready = threading.Event(), threading.Event()
    observations = []

    def watch():
        try:
            with (out/'gpu_observations.jsonl').open('w') as stream:
                while not stop.is_set():
                    row = observe(backend)
                    observations.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush(); ready.set()
                    stop.wait(0.5)
        except Exception as exc:
            status['monitor_error'] = repr(exc); ready.set()

    def guards(stage):
        for i in range(2):
            row = observe(backend)
            row.update(stage=stage, observation_index=i)
            status['guards'].append(row); save(out/'campaign.json', status)
            verify_guard(row, args.allow_graphics_pid, require_idle_util=stage == 'before')
            if i == 0:
                sleep(2)

    try:
        backend = make_backend(args.gpu)
        status.update(gpu_uuid=backend.uuid(), gpu_name=backend.name(), driver=backend.driver_version())
        thread = threading.Thread(target=watch, name='visual-kv-dram-nvml', daemon=True)
        thread.start()
        if not ready.wait(5) or status.get('monitor_error'):
            raise RuntimeError('GPU observation monitor failed to start')
        guards('before')
        arm = out/name; arm.mkdir()
        cmd = command(args, arm)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1',
                   TOKENIZERS_PARALLELISM='false', CPATH=headers['CPATH'])
        save(arm/'command.json', cmd)
        save(arm/'header_environment.json', headers)
        started = time.time()
        save(arm/'launch.json', dict(started_unix_s=started, working_directory=os.getcwd(),
            environment_overrides={key:env[key] for key in ('CUDA_VISIBLE_DEVICES','PYTHONUNBUFFERED','TOKENIZERS_PARALLELISM','CPATH')}))
        status.update(status='running', current=name, updated_at=started); save(out/'campaign.json', status)
        print('START '+name, flush=True)
        receipt = dict(started_unix_s=started, returncode=None)
        try:
            with (arm/'application.log').open('w') as stream:
                result = run_process(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            receipt['returncode'] = result.returncode
        except BaseException as exc:
            receipt['error'] = repr(exc); raise
        finally:
            receipt['finished_unix_s'] = time.time(); save(arm/'exit.json', receipt)
            # Retain PID before any protocol failure, so the monitor does not
            # later classify this very workload as foreign work.
            retain_workload_pid(arm, status)
        guards('after')
        if receipt['returncode']:
            raise RuntimeError('NCU/workload failed for '+name)
        workload, rows, validation = validate_workload(arm/'workload', args.visual_kv_mode, args.request_id)
        if workload['args']['physical_gpu'] != args.gpu or workload.get('profiler_effective_cpath') != headers['CPATH']:
            raise ValueError('Child GPU/CPATH differs from recorded command environment')
        if any(row.get('hardware', {}).get('device', {}).get('uuid') != status['gpu_uuid'] for row in rows):
            raise ValueError('Child result GPU UUID differs from supervisor device')
        status['workload_pid'] = workload['pid']
        records = bind_ncu(workload, rows, arm/'metrics.csv')
        save(arm/'workload_validation.json', validation)
        binding = dict(verified=True, range_count=len(records), records=records,
                       all_calls_natural_eos=validation['all_calls_natural_eos'],
                       all_calls_same_output=validation['all_calls_same_output'],
                       selected_paths_required_to_match_native=False,
                       scope='Observed generation DRAM byte counters; no background subtraction')
        if args.reference_summary:
            binding['reference_comparison'] = compare_reference(workload, rows, records, args.reference_summary)
        save(arm/'range_binding.json', binding)
        status['completed'].append(name)
        status.update(status='completed', current=None, updated_at=time.time(), range_binding_verified=True)
        print('DONE '+name, flush=True)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), updated_at=time.time()); raise
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive(): status['monitor_error'] = 'monitor did not terminate within five seconds'
        if backend is not None:
            try: backend.close()
            except Exception as exc: status['monitor_error'] = 'backend close failed: '+repr(exc)
        own = {status['workload_pid']} if 'workload_pid' in status else set()
        unknown, errors, residents = [], [], []
        for row in observations:
            compute = {p['pid'] for p in row.get('compute', [])}
            graphics = {p['pid'] for p in row.get('graphics', [])}
            foreign = (compute | (graphics-set(args.allow_graphics_pid)))-own
            if foreign: unknown.append(dict(time_unix=row['time_unix'], pids=sorted(foreign)))
            resident = (graphics & set(args.allow_graphics_pid))-own
            if resident: residents.append(dict(time_unix=row['time_unix'], pids=sorted(resident)))
            if any(k.endswith('_error') for k in row): errors.append(row)
        status.update(process_observation_count=len(observations), external_process_observations=unknown,
            resident_graphics_observations=residents, observation_errors=errors,
            attribution=('external_work_or_observation_error' if unknown or errors or status.get('monitor_error') else
                         'resident_graphics_background_not_excluded' if residents else
                         'no_foreign_context_observed_not_proven_exclusive'))
        save(out/'campaign.json', status)
    return status


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(HERE.parent))
    from hardware_metrics import _make_backend
    return run_campaign(args, make_backend=_make_backend)


if __name__ == '__main__':
    main()
