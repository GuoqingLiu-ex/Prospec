"""Sequential native / implicit generation DRAM campaign; never runs on import.

Each child owns the cooperative GPU lock and gets its own fresh directory.
No retry, fallback, process termination, GPU setting change or package install.
Native has no ring/full instrumentation. Counters cover complete generation,
not merely KV maintenance. All observations and failed child outputs are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from run_active_dram_campaign import observe, retain_workload_pid, save, verify_guard
from run_visual_kv_dram_campaign import header_environment
from profile_implicit_kv_dram import MODES, REQUEST, VARIANT, bind_ncu, validate_workload

HERE = Path(__file__).resolve().parent


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
    parser.add_argument('--header-environment', default=str(HERE.parents[1]/'kv_int8_primitive_r3/header_environment.json'))
    args = parser.parse_args(argv)
    if args.gpu not in range(8) or any(pid < 1 for pid in args.allow_graphics_pid):
        parser.error('GPU must be 0..7 and graphics PIDs positive')
    args.allow_graphics_pid = sorted(set(args.allow_graphics_pid))
    return args


def command(args, arm, mode):
    if mode not in MODES:
        raise ValueError('native or implicit mode required')
    return [args.ncu, '--config-file', 'off', '--target-processes', 'application-only',
        '--replay-mode', 'app-range', '--cache-control', 'none', '--clock-control', 'none',
        '--pipeline-boost-state', 'dynamic', '--disable-extra-suffixes',
        '--metrics', 'dram__bytes_read.sum,dram__bytes_write.sum',
        '--export', str(arm/'dram'), '--csv', '--page', 'raw', '--log-file', str(arm/'metrics.csv'),
        sys.executable, str(HERE/'profile_implicit_kv_dram.py'), '--profile-mode', mode,
        '--profile-scope', 'generation', '--profile-calls', '2',
        '--out', str(arm/'workload'), '--physical-gpu', str(args.gpu),
        '--requests', str(Path(args.requests).resolve()), '--request-id', args.request_id,
        '--warmup', '1', '--repeats', '2', '--max-new-tokens', '512']


def source_hashes():
    names = ('run_implicit_dram_campaign.py', 'profile_implicit_kv_dram.py', 'profile_ring_dram.py',
        'run_active_dram_campaign.py', 'run_visual_kv_dram_campaign.py', 'summarize_dram_campaign.py',
        'run_implicit_kv_layout.py', 'implicit_kv_runtime.py', 'active_ring_runtime.py',
        'ring_cache_runtime.py', 'delta_kv_commit.py', 'run_memory_layout.py',
        'run_deployed_adaptor.py', 'run_joint.py', 'compiled_adaptor.py', 'shared_embedding.py')
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}


def compare_bound(arms):
    """Observed ratios are separated from matching checks and GPU attribution."""
    native, candidate = arms['native'], arms['implicit']
    fields = ('output_token_ids', 'rounds', 'forward_counts', 'precision',
              'input_shapes', 'timestamps', 'video_grid_thw')
    pairs = []
    for left, right in zip(native['records'], candidate['records']):
        if (left['request_id'], left['repeat']) != (right['request_id'], right['repeat']):
            raise ValueError('cross-arm request/repeat mismatch')
        checks = {key:left[key] == right[key] for key in fields}
        checks['both_natural_eos'] = left['stop_reason'] == right['stop_reason'] == 'eos'
        pairs.append(dict(request_id=right['request_id'], repeat=right['repeat'], checks=checks,
            matched_workload=all(checks.values()),
            observed_reduction_pct={key:100*(1-right[key]/left[key]) if left[key] else None
                                    for key in ('read_bytes', 'write_bytes', 'total_bytes')}))
    metrics = ('read_bytes', 'write_bytes', 'total_bytes')
    medians = {name:{key:statistics.median(r[key] for r in info['records']) for key in metrics}
               for name, info in arms.items()}
    protocol_same = native['protocol'] == candidate['protocol']
    stable = all(x['validation']['all_calls_same_output'] and x['validation']['all_calls_same_rounds']
                 and x['validation']['all_calls_natural_eos'] for x in arms.values())
    return dict(pairs=pairs, same_protocol=protocol_same, all_calls_stable=stable,
        matched_workload=protocol_same and stable and all(x['matched_workload'] for x in pairs),
        median_bytes=medians,
        observed_reduction_pct_of_medians={key:100*(1-medians['implicit'][key]/medians['native'][key])
            if medians['native'][key] else None for key in metrics},
        counter_scope='Whole generate_joint; includes weights, attention and cache operations',
        attribution='See campaign process observations; matched workload alone is not exclusive process attribution',
        no_background_subtraction=True, quality_pass=None, performance_pass=None,
        historical_full_and_active_results='Kept as separate evidence; old ring/full checks differ from this native entry')


def run_campaign(args, *, make_backend, run_process=subprocess.run, sleep=time.sleep):
    if not Path(args.requests).is_file():
        raise FileNotFoundError('Request manifest missing: '+args.requests)
    # Resolve the existing CPU-verified compilation environment before any GPU
    # launch. This does not install headers or inherit a different CPATH.
    headers = header_environment(args.header_environment)
    out = Path(args.out); out.mkdir(exist_ok=False, parents=True)
    status = dict(status='starting', scope='generation', variant=VARIANT, modes=list(MODES),
        gpu=args.gpu, pid=os.getpid(), started_at=time.time(), completed=[], workload_pids={},
        allowed_resident_graphics_pids=args.allow_graphics_pid,
        protocol=dict(request_id=args.request_id, warmup=1, measure=2, max_new_tokens=512),
        lock_owner='child runner only', process_poll_interval_s=0.5, background_subtracted=False,
        native_ring_full_instrumentation=False, quality_pass=None, performance_pass=None,
        header_environment=headers, source_sha256=source_hashes(), guards=[])
    save(out/'campaign.json', status)
    backend, thread = None, None
    stop, ready = threading.Event(), threading.Event()
    observations, bound = [], {}

    def watch():
        try:
            with (out/'gpu_observations.jsonl').open('w') as stream:
                while not stop.is_set():
                    row = observe(backend); observations.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush(); ready.set()
                    stop.wait(0.5)
        except Exception as exc:
            status['monitor_error'] = repr(exc); ready.set()

    def guards(mode, stage):
        for i in range(2):
            row = observe(backend); row.update(mode=mode, stage=stage, observation_index=i)
            status['guards'].append(row); save(out/'campaign.json', status)
            verify_guard(row, args.allow_graphics_pid, require_idle_util=stage == 'before')
            if i == 0: sleep(2)

    try:
        backend = make_backend(args.gpu)
        status.update(gpu_uuid=backend.uuid(), gpu_name=backend.name(), driver=backend.driver_version())
        thread = threading.Thread(target=watch, name='implicit-dram-nvml', daemon=True); thread.start()
        if not ready.wait(5) or status.get('monitor_error'):
            raise RuntimeError('GPU observation monitor failed to start')
        for mode in MODES:
            guards(mode, 'before')
            arm = out/mode; arm.mkdir()
            cmd = command(args, arm, mode)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1',
                       TOKENIZERS_PARALLELISM='false', CPATH=headers['CPATH'])
            save(arm/'command.json', cmd); save(arm/'header_environment.json', headers)
            started = time.time()
            save(arm/'launch.json', dict(started_unix_s=started, working_directory=os.getcwd(),
                environment_overrides={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONUNBUFFERED','TOKENIZERS_PARALLELISM','CPATH')}))
            status.update(status='running', current=mode, updated_at=started); save(out/'campaign.json', status)
            print('START '+mode, flush=True)
            receipt = dict(started_unix_s=started, returncode=None)
            try:
                with (arm/'application.log').open('w') as stream:
                    result = run_process(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
                receipt['returncode'] = result.returncode
            except BaseException as exc:
                receipt['error'] = repr(exc); raise
            finally:
                receipt['finished_unix_s'] = time.time(); save(arm/'exit.json', receipt)
                child = {}; retain_workload_pid(arm, child)
                save(arm/'pid_receipt.json', child)
                if 'workload_pid' in child: status['workload_pids'][mode] = child['workload_pid']
            guards(mode, 'after')
            if receipt['returncode']:
                raise RuntimeError('NCU/workload failed for '+mode)
            workload, rows, validation = validate_workload(arm/'workload', mode, args.request_id)
            if workload['args']['physical_gpu'] != args.gpu or workload.get('profiler_effective_cpath') != headers['CPATH']:
                raise ValueError('Child GPU/CPATH differs from recorded environment')
            if any(r.get('hardware', {}).get('device', {}).get('uuid') != status['gpu_uuid'] for r in rows):
                raise ValueError('Child result GPU UUID differs from supervisor device')
            records = bind_ncu(workload, rows, arm/'metrics.csv')
            protocol = {key:workload[key] for key in ('target','draft','request_sha256')}
            protocol.update({key:workload['args'][key] for key in ('seed','max_new_tokens')})
            bound[mode] = dict(records=records, validation=validation, protocol=protocol)
            save(arm/'workload_validation.json', validation)
            save(arm/'range_binding.json', dict(verified=True, range_count=len(records), **bound[mode]))
            status['completed'].append(mode); save(out/'campaign.json', status)
            print('DONE '+mode, flush=True)
        save(out/'comparison.json', compare_bound(bound))
        status.update(status='completed', current=None, updated_at=time.time(), range_binding_verified=True)
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
        own = set(status['workload_pids'].values())
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
