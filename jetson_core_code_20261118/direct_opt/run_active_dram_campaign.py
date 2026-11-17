"""Single-arm NCU supervisor for the active-width ring candidate.

One warmup and two generation ranges for the default 478-token long answer.
No GPU work on import. The child runner owns the cooperative GPU lock; this
supervisor never acquires a second lock, kills another process, or changes GPU
settings. Resident graphics require explicit allowlisting and remain a counter
attribution limitation. Use a fresh output directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ARM = 'active512'
REQUEST = 'direct_quality_vdc_v_sPK-sbHIb3w'
HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--scope', choices=('generation',), default='generation')
    parser.add_argument('--warmup', type=int, choices=(1,), default=1)
    parser.add_argument('--repeats', type=int, choices=(2,), default=2)
    parser.add_argument('--requests', default=str(HERE.parents[1]/'coord/quality_requests.jsonl'))
    parser.add_argument('--request-id', default=REQUEST)
    parser.add_argument('--ncu', default='/usr/local/cuda/bin/ncu')
    parser.add_argument('--allow-graphics-pid', type=int, action='append', default=[])
    args = parser.parse_args(argv)
    if args.gpu < 0 or any(pid < 1 for pid in args.allow_graphics_pid):
        parser.error('GPU index must be nonnegative and graphics PIDs positive')
    args.allow_graphics_pid = sorted(set(args.allow_graphics_pid))
    return args


def command(args, arm):
    """Pure command construction, independently testable without CUDA/NVML."""
    return [args.ncu, '--config-file', 'off', '--target-processes', 'application-only',
            '--replay-mode', 'app-range', '--cache-control', 'none', '--clock-control', 'none',
            '--pipeline-boost-state', 'dynamic', '--disable-extra-suffixes',
            '--metrics', 'dram__bytes_read.sum,dram__bytes_write.sum',
            '--export', str(arm/'dram'), '--csv', '--page', 'raw', '--log-file', str(arm/'metrics.csv'),
            sys.executable, str(HERE/'profile_active_ring_dram.py'),
            '--profile-scope', 'generation', '--profile-calls', '2',
            '--ring-policy', 'protected_prompt', '--recent-capacity', '512',
            '--memory-layout', 'shared_embedding', '--adaptor', 'compiled',
            '--out', str(arm/'workload'), '--physical-gpu', str(args.gpu),
            '--requests', str(Path(args.requests).resolve()), '--request-id', args.request_id,
            '--warmup', '1', '--repeats', '2', '--max-new-tokens', '512']


def observe(backend):
    row = {'time_unix': time.time()}
    for key, call in (('compute', lambda: backend.processes('compute')),
                      ('graphics', lambda: backend.processes('graphics')),
                      ('utilization', backend.utilization)):
        try:
            row[key] = call()
        except Exception as exc:
            row[key+'_error'] = repr(exc)
    return row


def verify_guard(row, allowed_graphics, *, require_idle_util):
    if any(key.endswith('_error') for key in row):
        raise RuntimeError('Cannot verify GPU state: '+str(row))
    unknown_graphics = [p for p in row['graphics'] if p['pid'] not in allowed_graphics]
    # An allowlisted graphics PID is NEVER exempted if listed as compute.
    if row['compute'] or unknown_graphics:
        raise RuntimeError('GPU occupied; preserved existing work: '+str(row))
    if require_idle_util and row.get('utilization', {}).get('gpu_util_pct', 100) != 0:
        raise RuntimeError('GPU utilization is not idle before profiling: '+str(row))


def workload_receipt(arm, request_id):
    status = json.loads((arm/'workload/status.json').read_text())
    expected = dict(status='completed', ring_variant='active_width', profiler_ring_variant='active_width',
                    profiler_scope='generation', profiler_warmup_calls=1, profiler_requested_calls=2,
                    profiler_completed_ranges=2, profiler_capture_complete=True, under_ncu=True,
                    memory_layout='shared_embedding', deployment_adaptor='compiled',
                    ring_policy='protected_prompt', recent_capacity=512, total_cells=3, completed_cells=3)
    for key, value in expected.items():
        if status.get(key) != value:
            raise RuntimeError('Invalid active profile receipt '+key+': '+repr(status.get(key)))
    # run_joint uses argparse action='append', so the saved field is a list.
    # Accept the historical scalar form, but never a multi-request workload.
    selected = status['args']['request_id']
    selected = [selected] if isinstance(selected, str) else selected
    if selected != [request_id]:
        raise RuntimeError('Profile receipt request differs from command')
    return status


def retain_workload_pid(arm, status):
    """Keep child identity even if later protocol/counter validation fails."""
    path = arm/'workload/status.json'
    if not path.exists():
        return
    try:
        pid = json.loads(path.read_text())['pid']
        if type(pid) is not int or pid < 1:
            raise ValueError('workload PID must be a positive integer')
        status['workload_pid'] = pid
        status['workload_pid_source'] = str(path)
    except Exception as exc:
        status['workload_pid_read_error'] = repr(exc)


def run_campaign(args, *, make_backend, run_process=subprocess.run, sleep=time.sleep):
    """Injected boundaries permit CPU tests; production supplies the NVML factory."""
    if not Path(args.requests).is_file():
        raise FileNotFoundError('Request manifest missing: '+args.requests)
    out = Path(args.out)
    out.mkdir(exist_ok=False, parents=True)
    status = {'status': 'starting', 'scope': 'generation', 'variant': 'active_width', 'gpu': args.gpu,
              'pid': os.getpid(), 'started_at': time.time(), 'completed': [],
              'allowed_resident_graphics_pids': args.allow_graphics_pid,
              'protocol': {'request_id': args.request_id, 'warmup': 1, 'measure': 2, 'max_new_tokens': 512},
              'lock_owner': 'child runner only', 'process_poll_interval_s': 0.5,
              'profile_scope': 'generate_joint only; model load, input preparation, output decode and runtime cleanup excluded',
              'background_subtracted': False, 'guards': []}
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
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    ready.set()
                    stop.wait(0.5)
        except Exception as exc:
            status['monitor_error'] = repr(exc)
            ready.set()

    def guards(stage):
        # Post-run utilization can still reflect the completed workload's
        # sensor averaging window; preserve it but use process presence then.
        for i in range(2):
            row = observe(backend)
            row.update(stage=stage, observation_index=i)
            status['guards'].append(row)
            save(out/'campaign.json', status)
            verify_guard(row, args.allow_graphics_pid, require_idle_util=stage == 'before')
            if i == 0:
                sleep(2)

    try:
        backend = make_backend(args.gpu)
        status.update(gpu_uuid=backend.uuid(), gpu_name=backend.name(), driver=backend.driver_version())
        thread = threading.Thread(target=watch, name='active-dram-nvml', daemon=True)
        thread.start()
        if not ready.wait(5) or status.get('monitor_error'):
            raise RuntimeError('GPU observation monitor failed to start')
        guards('before')
        arm = out/ARM
        arm.mkdir()
        cmd = command(args, arm)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1',
                   TOKENIZERS_PARALLELISM='false')
        save(arm/'command.json', cmd)
        started = time.time()
        save(arm/'launch.json', {'started_unix_s': started, 'working_directory': os.getcwd(),
             'environment_overrides': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONUNBUFFERED', 'TOKENIZERS_PARALLELISM')}})
        status.update(status='running', current=ARM, updated_at=started)
        save(out/'campaign.json', status)
        print('START '+ARM, flush=True)
        receipt = {'started_unix_s': started, 'returncode': None}
        try:
            with (arm/'application.log').open('w') as stream:
                result = run_process(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            receipt['returncode'] = result.returncode
        except BaseException as exc:
            receipt['error'] = repr(exc)
            raise
        finally:
            receipt['finished_unix_s'] = time.time()
            save(arm/'exit.json', receipt)
            retain_workload_pid(arm, status)
        # Keep after-run guard evidence even if NCU returned an error.
        guards('after')
        if receipt['returncode']:
            raise RuntimeError('NCU/workload failed for '+ARM)
        workload = workload_receipt(arm, args.request_id)
        status['workload_pid'] = workload['pid']
        from validate_active_ring import bind_ncu
        rows = [json.loads(line) for line in (arm/'workload/results.jsonl').read_text().splitlines() if line.strip()]
        if len(rows) != 3 or sorted(r['phase'] for r in rows) != ['measure', 'measure', 'warmup']:
            raise RuntimeError('Expected all three result cells: one warmup and two measurements')
        # Completed API ranges alone do not prove counters were collected.
        metrics = bind_ncu(workload, rows, arm/'metrics.csv')
        save(arm/'range_binding.json', {'verified': True, 'range_count': len(metrics),
             'raw_counters': [{k: r[k] for k in ('range_id', 'pid', 'request_id', 'repeat', 'read_bytes', 'write_bytes', 'total_bytes', 'passes')}
                              for r in metrics]})
        status['completed'].append(ARM)
        status.update(status='completed', current=None, updated_at=time.time(), range_binding_verified=True)
        print('DONE '+ARM, flush=True)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), updated_at=time.time())
        raise
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                status['monitor_error'] = 'monitor did not terminate within five seconds'
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                # Retain the campaign/exit evidence even on telemetry cleanup
                # failure, including when an earlier child error is propagating.
                status['monitor_error'] = 'backend close failed: '+repr(exc)
        own = {status['workload_pid']} if 'workload_pid' in status else set()
        unknown, errors, residents = [], [], []
        for row in observations:
            compute = {p['pid'] for p in row.get('compute', [])}
            graphics = {p['pid'] for p in row.get('graphics', [])}
            foreign = (compute | (graphics-set(args.allow_graphics_pid)))-own
            if foreign:
                unknown.append({'time_unix': row['time_unix'], 'pids': sorted(foreign)})
            resident = (graphics & set(args.allow_graphics_pid))-own
            if resident:
                residents.append({'time_unix': row['time_unix'], 'pids': sorted(resident)})
            if any(k.endswith('_error') for k in row):
                errors.append(row)
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
