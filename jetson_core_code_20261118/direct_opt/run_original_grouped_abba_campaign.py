"""Fresh ordinary A/B/B/A campaign; execution occurs only through main().

Four independent child processes retain the frozen original/grouped recipes.
This supervisor never changes precision flags, GPU settings or parent environment,
never retries/resumes, and never terminates a workload. The child owns its normal
cooperative lock. Two pre-launch NVML observations cannot prove exclusivity;
ordinary request sampling and the independent offline audit remain necessary.

Freeze contract: all project Python sources (exactly the child's source receipt),
all Python under the frozen native and author roots, and the external lock helper.
Every launch and return rechecks sources, manifest and the allowlisted environment.
Mechanical validation is mandatory before the next stage; answer/acceptance or
cross-arm precision differences remain observations, not reasons to erase runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import audit_original_grouped_abba as auditor
from run_active_dram_campaign import observe, verify_guard

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
STAGES = ('00_original', '01_grouped', '02_grouped', '03_original')
ROLES = ('original', 'grouped', 'grouped', 'original')
EXTERNAL_ROOTS = {
    'native': Path('/data1/lgq2026/jev_spec/runs/decoder_A_followup_20261001_0345/code/src'),
    'author': Path('/data1/lgq2026/jev_spec/runs/decoder_vispec_mechanism_cursor1_20261001/code/ViSpec/vispec'),
}
EXTERNAL_FILES = {
    'gpu_lock': Path('/data1/lgq2026/jev_spec/runs/decoder_vispec_mechanism_cursor1_20261001/code/gpu_lock.py'),
}
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'NVIDIA_TF32_OVERRIDE',
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG',
    'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTHONPATH', 'CPATH',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM', 'PYTHONUNBUFFERED')


class GuardStopped(RuntimeError):
    """No child is launched when a pre-launch observation is not acceptable."""


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_recipe():
    return {key: os.environ.get(key) for key in ENV_KEYS}


def source_snapshot():
    project = {str(p.relative_to(PROJECT)): digest(p)
               for p in sorted(PROJECT.rglob('*.py')) if '__pycache__' not in p.parts}
    if any(name not in project for name in auditor.SOURCES):
        raise ValueError('required execution source missing from project snapshot')
    external = {}
    for name, root in EXTERNAL_ROOTS.items():
        paths = sorted(p for p in root.rglob('*.py') if '__pycache__' not in p.parts)
        if not root.is_dir() or not paths:
            raise FileNotFoundError('frozen external source root missing: ' + str(root))
        external[name] = dict(root=str(root), files={str(p.relative_to(root)): digest(p) for p in paths})
    external_files = {name: dict(path=str(p), sha256=digest(p)) for name, p in EXTERNAL_FILES.items()}
    return dict(project=project, external_roots=external, external_files=external_files)


def manifest_receipt(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row['request_id'] for row in rows]
    if len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError('full manifest must contain twelve unique requests')
    return dict(path=str(path), sha256=digest(path), requests=rows, request_ids=ids)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', required=True)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    parser.add_argument('--expected-gpu-uuid', required=True,
                        help='Explicit NVML GPU-UUID recorded when choosing the idle physical GPU')
    parser.add_argument('--allow-graphics-pid', type=int, choices=(6360,), action='append', default=[],
                        help='Only externally confirmed graphics PID 6360 may be explicitly exempted')
    args = parser.parse_args(argv)
    args.allow_graphics_pid = sorted(set(args.allow_graphics_pid))
    try:
        if not args.expected_gpu_uuid.startswith('GPU-'):
            raise ValueError('GPU- prefix missing')
        parsed = uuid.UUID(args.expected_gpu_uuid[4:])
        if str(parsed) != args.expected_gpu_uuid[4:]:
            raise ValueError('canonical UUID required')
    except ValueError:
        parser.error('--expected-gpu-uuid must be an explicit canonical NVML GPU-UUID')
    if Path(args.out).exists():
        parser.error('fresh campaign directory required; no retry or resume')
    return args


def command(args, folder, role):
    if role not in ('original', 'grouped'):
        raise ValueError('unknown comparison arm')
    return [sys.executable, str(HERE/'run_original_grouped_timing.py'),
        '--comparison-arm', role, '--requests', str(Path(args.requests).resolve()),
        '--out', str(folder), '--physical-gpu', str(args.gpu),
        '--warmup', '1', '--repeats', '2', '--max-new-tokens', '512', '--seed', '20261001']


def verify_contract(contract, args):
    if manifest_receipt(args.requests) != contract['manifest']:
        raise ValueError('manifest changed during campaign')
    if source_snapshot() != contract['sources']:
        raise ValueError('execution source changed during campaign')
    if environment_recipe() != contract['environment']:
        raise ValueError('parent environment changed during campaign')


def validate_completed(folder, role, args, contract, launch):
    """Use the frozen offline validator; never rewrite child status/results."""
    run = auditor.read_run(folder, role, args.allow_graphics_pid)
    status, precision = run['status'], run['precision']
    if not run['mechanical_pass']:
        raise ValueError('child completed but mechanical execution validation failed')
    if status['pid'] != launch['pid']:
        raise ValueError('child PID differs from direct process launch')
    if status['request_sha256'] != contract['manifest']['sha256'] or run['requests'] != contract['manifest']['requests']:
        raise ValueError('child manifest differs from frozen launch')
    if status['source_sha256'] != contract['sources']['project']:
        raise ValueError('child full source receipt differs from launch freeze')
    if status['args']['physical_gpu'] != args.gpu or any(
            row['hardware']['device']['uuid'] != args.expected_gpu_uuid for row in run['rows']):
        raise ValueError('child physical GPU/UUID differs from launch')
    expected_env = {key: contract['environment'][key] for key in auditor.PUBLIC_ENV}
    if precision['public_environment'] != expected_env or precision['public_environment_after'] != expected_env:
        raise ValueError('child public precision environment differs from launch')
    if precision['exit_flags'] != precision['default_startup_flags']:
        raise ValueError('child did not restore its startup precision flags')
    return dict(verified=True, pid=status['pid'], role=role, cells=len(run['rows']),
        manifest_sha256=status['request_sha256'], file_sha256=run['hashes'],
        precision_observation_run_id=precision['run_id'], startup_flags=precision['default_startup_flags'],
        exit_flags=precision['exit_flags'], actual_generation_flags=[dict(
            cell_index=i, before=call['before'], after=call['after']) for i, call in enumerate(precision['calls'])],
        protocol={key: status[key] for key in ('target', 'draft', 'torch', 'transformers', 'device_name')},
        sampled_background_clear_cells=sum(detail['hardware']['sampled_background_clear'] for detail in run['details']),
        raw_valid_timing_values=[row.get('valid_timing') for row in run['rows']],
        quality_pass=None, throughput_pass=None,
        scope='Mechanical completion only; cross-arm precision/output/trajectory and background filtering require the offline ABBA audit')


def run_campaign(args, *, make_backend, popen=subprocess.Popen, sleep=time.sleep):
    """CPU-testable boundaries; no NVML/Torch/model imports until main supplies them."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(schema='ordinary_original_grouped_abba_campaign_v1', status='starting', terminal=False,
        pid=os.getpid(), started=time.time(), current=None, completed=[], stages=list(STAGES),
        guards=[], launches=[], lock_owner='child frozen runner only', retry=False, resume=False,
        supervisor_changes_precision=False, supervisor_changes_environment=False,
        prelaunch_observations_do_not_prove_exclusivity=True,
        child_sampler_and_offline_paired_background_review_required=True)
    save(out/'campaign.json', status)
    backend = None
    try:
        contract = dict(gpu=args.gpu, gpu_uuid=args.expected_gpu_uuid,
            allowed_graphics_pids=args.allow_graphics_pid, manifest=manifest_receipt(args.requests),
            sources=source_snapshot(), environment=environment_recipe(),
            python=dict(executable=sys.executable, version=sys.version),
            distributions={name: importlib.metadata.version(name) for name in ('torch', 'transformers')},
            fixed_recipe=dict(warmup=1, repeats=2, max_new_tokens=512, seed=20261001),
            expected_child_environment_changes=dict(CUDA_VISIBLE_DEVICES=str(args.gpu),
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
            source_scope='All project Python, frozen native/author Python and lock helper; third-party binary/weights are not rehashed')
        status['contract'] = contract
        save(out/'source_freeze.json', contract['sources'])
        save(out/'manifest_frozen.json', contract['manifest'])
        save(out/'campaign.json', status)
        backend = make_backend(args.gpu)
        status['device'] = dict(uuid=backend.uuid(), name=backend.name(), driver=backend.driver_version())
        if status['device']['uuid'] != args.expected_gpu_uuid:
            raise GuardStopped('NVML physical GPU UUID differs from frozen physical GPU/UUID contract')
        for stage, role in zip(STAGES, ROLES):
            status.update(status='checking', current=stage)
            save(out/'campaign.json', status)
            verify_contract(contract, args)
            for index in range(2):
                row = observe(backend)
                row.update(stage=stage, observation_index=index, gpu_uuid=backend.uuid())
                status['guards'].append(row)
                save(out/'campaign.json', status)
                try:
                    if row['gpu_uuid'] != args.expected_gpu_uuid:
                        raise RuntimeError('GPU UUID changed')
                    verify_guard(row, args.allow_graphics_pid, require_idle_util=False)
                except Exception as exc:
                    raise GuardStopped(str(exc)) from exc
                if index == 0:
                    sleep(2)
            # The two-second guard interval must not permit a source/env change.
            verify_contract(contract, args)
            folder = out/stage
            cmd = command(args, folder, role)
            launch = dict(stage=stage, role=role, command=cmd, cwd=str(PROJECT),
                started=time.time(), pid=None, environment=environment_recipe(),
                child_environment_overrides={}, source_contract_sha256=digest(out/'source_freeze.json'))
            launch_path = Path(str(folder)+'.launch.json')
            terminal = dict(stage=stage, role=role, started=launch['started'], ended=None,
                            returncode=None, pid=None, child_still_running=False)
            save(launch_path, launch)
            status.update(status='running', current=stage)
            status['launches'].append(str(launch_path)); save(out/'campaign.json', status)
            child = None
            try:
                with Path(str(folder)+'.log').open('w') as stream:
                    child = popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                  cwd=str(PROJECT), env=dict(os.environ), start_new_session=True)
                    launch['pid'] = terminal['pid'] = child.pid
                    save(launch_path, launch)
                    terminal['returncode'] = child.wait()
            except BaseException as exc:
                terminal['error'] = repr(exc)
                if child is not None:
                    terminal['returncode'] = child.poll()
                    terminal['child_still_running'] = terminal['returncode'] is None
                raise
            finally:
                terminal['ended'] = time.time()
                save(Path(str(folder)+'.exit.json'), terminal)
            if terminal['returncode'] != 0:
                raise RuntimeError('child exit is nonzero; preserved without retry: ' + stage)
            verify_contract(contract, args)
            receipt = validate_completed(folder, role, args, contract, launch)
            if status['completed']:
                previous = json.loads((out/(status['completed'][0]+'.validation.json')).read_text())
                if previous['protocol'] != receipt['protocol']:
                    raise ValueError('model/package/device protocol differs across processes')
                if any(receipt['pid'] == json.loads((out/(done+'.validation.json')).read_text())['pid']
                       for done in status['completed']):
                    raise ValueError('four distinct child process identities required')
            save(Path(str(folder)+'.validation.json'), receipt)
            status['completed'].append(stage)
            save(out/'campaign.json', status)
        status.update(status='completed', current=None)
    except BaseException as exc:
        state = 'stopped' if isinstance(exc, GuardStopped) else 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        status.update(status=state, error=repr(exc))
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                status['backend_close_error'] = repr(exc)
                if status['status'] == 'completed':
                    status['status'] = 'failed'
        status.update(terminal=True, ended=time.time(), parent_environment_after=environment_recipe())
        if 'contract' in status and status['parent_environment_after'] != status['contract']['environment']:
            status['environment_contract_error'] = 'parent environment changed; supervisor does not restore by silently setting values'
            if status['status'] == 'completed':
                status['status'] = 'failed'
        save(out/'campaign.json', status)
    if status['status'] != 'completed':
        raise RuntimeError('campaign terminal validation failed: ' + status['status'])
    return status


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(PROJECT))
    from hardware_metrics import _make_backend
    return run_campaign(args, make_backend=_make_backend)


if __name__ == '__main__':
    main()
