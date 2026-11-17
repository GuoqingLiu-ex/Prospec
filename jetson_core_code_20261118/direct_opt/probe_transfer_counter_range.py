"""One small app-range to test ten transfer counters, not model performance.

Nothing runs on import. ``command`` only prints an NCU command; ``run`` is a
separately authorized GPU action; ``bind`` validates retained metadata/raw CSV.
The workload inherits precision/environment settings. Warmup and output reads
are outside the range. More than one replay pass never qualifies for comparison
with the project's older one-pass counter results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
SCHEMA = 'transfer_counter_range_probe_v1'
SIZE, ITERATIONS, WARMUP = 512, 8, 2
METRIC_UNITS = {
    'dram__bytes_read.sum': 'byte',
    'dram__bytes_write.sum': 'byte',
    'dram__sectors_read.sum': 'sector',
    'dram__sectors_write.sum': 'sector',
    'lts__t_requests_srcunit_tex_op_read.sum': 'request',
    'lts__t_requests_srcunit_tex_op_read_lookup_miss.sum': 'request',
    'lts__t_requests_srcunit_tex_op_write.sum': 'request',
    'lts__t_requests_srcunit_tex_op_write_lookup_miss.sum': 'request',
    'smsp__sass_inst_executed_op_global_ld.sum': 'inst',
    'smsp__sass_inst_executed_op_global_st.sum': 'inst',
}
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'NVIDIA_TF32_OVERRIDE',
            'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'CPATH')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def valid_uuid(value):
    return isinstance(value, str) and value.startswith('GPU-') and str(uuid.UUID(value[4:])) == value[4:]


def cuda_uuid_to_nvml(value):
    # Torch2.8 _CUuuid.__str__ emits the bare canonical UUID (Module.cpp:951-994).
    raw = str(value)
    return 'GPU-' + str(uuid.UUID(raw.removeprefix('GPU-')))


def read_flags(torch):
    return dict(allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                reduced_precision_math_sdp=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = p.add_subparsers(dest='action', required=True)
    for name in ('run', 'command'):
        s = sub.add_parser(name, allow_abbrev=False)
        s.add_argument('--out', type=Path, required=True)
        s.add_argument('--physical-gpu', type=int, choices=range(8), required=True)
        s.add_argument('--cuda-device', type=int, choices=range(8), required=True)
        s.add_argument('--expected-gpu-uuid', required=True)
        s.add_argument('--nonce', required=True, help='Fresh 32 lowercase hexadecimal characters')
        if name == 'command':
            s.add_argument('--ncu', default='/usr/local/cuda/bin/ncu')
            s.add_argument('--python', default=sys.executable)
            s.add_argument('--raw-csv', type=Path, required=True)
            s.add_argument('--report', type=Path, required=True)
    b = sub.add_parser('bind', allow_abbrev=False)
    b.add_argument('--metadata', type=Path, required=True)
    b.add_argument('--raw-csv', type=Path, required=True)
    b.add_argument('--expected-nonce', required=True)
    b.add_argument('--expected-gpu-uuid', required=True)
    b.add_argument('--expected-physical-gpu', type=int, choices=range(8), required=True)
    b.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    try:
        require(valid_uuid(a.expected_gpu_uuid), 'canonical NVML GPU UUID required')
        require(re.fullmatch('[0-9a-f]{32}', a.expected_nonce if a.action == 'bind' else a.nonce),
                'nonce must have 32 lowercase hexadecimal characters')
        require(not a.out.exists(), 'fresh output required; no resume or automatic retry')
    except (ValueError, AttributeError) as exc:
        p.error(str(exc))
    return a


def build_ncu_command(*, ncu, python, out, raw_csv, report, physical_gpu,
                      cuda_device, expected_gpu_uuid, nonce):
    """Pure command builder. Caller owns deployment/launch and background review."""
    require(type(physical_gpu) is int and physical_gpu in range(8), 'physical GPU out of range')
    require(type(cuda_device) is int and cuda_device in range(8), 'CUDA ordinal out of range')
    require(valid_uuid(expected_gpu_uuid), 'canonical GPU UUID required')
    require(re.fullmatch('[0-9a-f]{32}', nonce), 'invalid nonce')
    return [str(ncu), '--config-file', 'off', '--target-processes', 'application-only',
        '--replay-mode', 'app-range', '--cache-control', 'none', '--clock-control', 'none',
        '--pipeline-boost-state', 'dynamic', '--metrics', ','.join(METRIC_UNITS),
        '--csv', '--page', 'raw', '--log-file', str(raw_csv), '--export', str(report),
        '--disable-extra-suffixes', str(python), str(Path(__file__).resolve()), 'run',
        '--out', str(out), '--physical-gpu', str(physical_gpu), '--cuda-device', str(cuda_device),
        '--expected-gpu-uuid', expected_gpu_uuid, '--nonce', nonce]


def make_inputs(torch, size=SIZE):
    """Exact binary fractions: every partial dot numerator is < size*8*6."""
    ids = torch.arange(size * size, dtype=torch.int64).reshape(size, size)
    left = ((ids % 17) - 8).to(torch.float32) / 16
    right = ((ids % 13) - 6).to(torch.float32) / 16
    return left, right, torch.mm(left, right)


class MatrixCopyWorkload:
    def __init__(self, torch, device):
        self.torch = torch
        left, right, self.reference = make_inputs(torch)
        self.left, self.right = left.to(device), right.to(device)
        self.product = torch.empty((SIZE, SIZE), dtype=torch.float32, device=device)
        self.copy1 = torch.empty_like(self.product)
        self.copy2 = torch.empty_like(self.product)

    def __call__(self):
        for _ in range(ITERATIONS):
            self.torch.mm(self.left, self.right, out=self.product)
        self.copy1.copy_(self.product)
        self.copy2.copy_(self.copy1)

    def diagnostics(self):
        # Called after profiler Stop. No output readback belongs to the range.
        output = self.copy2.cpu()
        return dict(finite=bool(self.torch.isfinite(output).all()),
            exact_reference=bool(self.torch.equal(output, self.reference)),
            max_abs_error=float((output - self.reference).abs().max()),
            sha256=hashlib.sha256(output.numpy().tobytes()).hexdigest(),
            reference_sha256=hashlib.sha256(self.reference.numpy().tobytes()).hexdigest(),
            shape=list(output.shape), dtype=str(output.dtype), readback_after_profiler_stop=True)


def profile_once(cuda, workload, record):
    """Drain warmup before Start; drain measured work before Stop, even on failure."""
    cuda.synchronize()
    cuda.profiler.start()
    record['start_calls'] += 1
    body_completed = False
    try:
        workload()
        cuda.synchronize()
        body_completed = True
    finally:
        try:
            if not body_completed:
                cuda.synchronize()
        finally:
            cuda.profiler.stop()
            record['stop_calls'] += 1
    record['completed'] = True


def run_probe(args, *, torch_module=None, lock_factory=None, backend_factory=None,
              workload_factory=MatrixCopyWorkload):
    """Injected CPU doubles exercise lifecycle; production always uses CUDA."""
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: os.environ.get(k) for k in ENV_KEYS}
    metadata = dict(schema=SCHEMA, status='starting', pid=os.getpid(), nonce=args.nonce,
        source_sha256=digest(__file__), selected_physical_gpu=args.physical_gpu,
        selected_cuda_device=args.cuda_device, expected_gpu_uuid=args.expected_gpu_uuid,
        metrics=dict(METRIC_UNITS), profiler_api='cudaProfilerStart/Stop',
        requested_range_count=1, actual_bound=False, ncu_attachment_verified=False,
        range=dict(occurrence=0, start_calls=0, stop_calls=0, completed=False),
        environment_before=env, started_unix_s=time.time(), precision_setters_added=False,
        recipe=dict(size=SIZE, iterations=ITERATIONS, warmup=WARMUP, dtype='float32',
            matmul_calls=ITERATIONS, copy_calls=2,
            fixed_tensor_payload_bytes=5*SIZE*SIZE*4,
            matmul_macs=ITERATIONS*SIZE**3,
            explicit_copy_read_bytes=2*SIZE*SIZE*4,
            explicit_copy_write_bytes=2*SIZE*SIZE*4,
            integer_dot_numerator_absolute_bound=SIZE*8*6,
            input_scale_denominator=16, product_scale_denominator=256),
        scope='Counter availability/pass diagnostic only; no model, quality, throughput or DRAM-reduction claim. '
              'Allocation/input transfer/two warmup bodies precede Start; final synchronize precedes Stop; '
              'output CPU transfer and validation follow Stop. Logical copy payloads are not measured DRAM.',
        no_auto_fallback=True, no_resume=True, no_workload_termination=True)
    save(out/'metadata.json', metadata)
    torch = torch_module
    try:
        if torch is None:
            import torch
        require(not torch.cuda.is_initialized(), 'fresh process with no initialized CUDA required')
        if lock_factory is None:
            from run_joint import cooperative_lock
            lock_factory = cooperative_lock
        if backend_factory is None:
            sys.path.insert(0, str(HERE.parent))
            from hardware_metrics import _make_backend
            backend_factory = _make_backend
        with lock_factory(args.physical_gpu, out) as lock:
            metadata['lock'] = dict(lock)
            backend = backend_factory(args.physical_gpu)
            try:
                metadata['nvml_device'] = dict(uuid=backend.uuid(), name=backend.name(), driver=backend.driver_version())
                require(backend.uuid() == args.expected_gpu_uuid, 'NVML physical GPU UUID mismatch')
            finally:
                backend.close()
            props = torch.cuda.get_device_properties(args.cuda_device)
            metadata['cuda_device'] = dict(uuid=cuda_uuid_to_nvml(props.uuid), raw_uuid=str(props.uuid), name=props.name,
                capability=[props.major, props.minor])
            require(metadata['cuda_device']['uuid'] == args.expected_gpu_uuid, 'logical CUDA and physical NVML UUID mismatch')
            require((props.major, props.minor) == (12, 0), 'this diagnostic contract requires SM12.0')
            metadata['precision_before'] = read_flags(torch)
            with torch.cuda.device(args.cuda_device), torch.inference_mode():
                work = workload_factory(torch, f'cuda:{args.cuda_device}')
                for _ in range(WARMUP):
                    work()
                torch.cuda.synchronize()
                metadata['allocator_after_warmup'] = dict(allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    scope='Allocator snapshot includes tensor payload and any allocator-backed workspace; '
                          'does not bound driver/library workspace or whole process memory.')
                profile_once(torch.cuda, work, metadata['range'])
                metadata['output'] = work.diagnostics()
                require(metadata['output']['finite'] and metadata['output']['exact_reference'],
                        'nonfinite or incorrect probe output')
            metadata['precision_after'] = read_flags(torch)
            require(metadata['precision_after'] == metadata['precision_before'], 'precision flags changed')
            metadata['status'] = 'completed'
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    finally:
        if torch is not None and 'precision_before' in metadata:
            metadata['precision_after'] = read_flags(torch)
            if metadata['precision_after'] != metadata['precision_before']:
                metadata.update(status='failed', precision_error='Observed flags changed; no setter or silent restoration used')
        metadata['environment_after'] = {k: os.environ.get(k) for k in ENV_KEYS}
        metadata['finished_unix_s'] = time.time()
        if metadata['environment_after'] != env:
            metadata.update(status='failed', environment_error='Inherited environment changed')
        save(out/'metadata.json', metadata)
    require(metadata['status'] == 'completed', 'probe did not complete')
    return metadata


def parse_raw_csv(path):
    text = Path(path).read_text()
    require('==ERROR==' not in text, 'NCU reported an error')
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('"ID",')]
    require(len(starts) == 1, 'one raw CSV header required')
    reader = csv.DictReader(io.StringIO('\n'.join(lines[starts[0]:])))
    require(len(reader.fieldnames) == len(set(reader.fieldnames)), 'duplicate CSV columns')
    rows = list(reader)
    units = [r for r in rows if r.get('ID') == '']
    require(len(units) == 1, 'one metric unit row required')
    for name, unit in dict(METRIC_UNITS, profiler__replayer_passes='pass').items():
        require(units[0].get(name) == unit, 'unexpected metric unit: ' + name)
    values = [r for r in rows if r.get('ID', '').isdigit()]
    require(len(values) == 1, 'exactly one app-range occurrence required')
    require(len(rows) == 2, 'unexpected non-range CSV records')
    row = values[0]
    require(row['Kernel Name'] == 'range' and row['CC'] == '12.0', 'SM12.0 aggregate range required')
    def number(name):
        value = row.get(name, '').replace(',', '')
        require(bool(re.fullmatch('[0-9]+', value)), 'unavailable/noninteger counter: ' + name)
        return int(value)
    passes = number('profiler__replayer_passes')
    require(passes >= 1, 'positive replay pass count required')
    return dict(range_id=number('ID'), pid=number('Process ID'), cuda_device=number('Device'),
        passes=passes, occurrence=0, metrics={m:dict(value=number(m), unit=u) for m,u in METRIC_UNITS.items()},
        one_pass=passes == 1, raw_csv_sha256=digest(path))


def bind_raw_csv(metadata, csv_path, *, expected_nonce, expected_gpu_uuid, expected_physical_gpu):
    require(metadata.get('schema') == SCHEMA and metadata.get('status') == 'completed', 'completed probe metadata required')
    require(metadata['nonce'] == expected_nonce, 'invocation nonce mismatch')
    require(metadata['source_sha256'] == digest(__file__), 'executed source differs from reviewed probe')
    require(metadata['selected_physical_gpu'] == expected_physical_gpu, 'physical GPU mismatch')
    require(metadata['expected_gpu_uuid'] == metadata['nvml_device']['uuid'] ==
            metadata['cuda_device']['uuid'] == expected_gpu_uuid, 'GPU UUID receipt mismatch')
    require(metadata['cuda_device']['capability'] == [12, 0], 'unexpected recorded SM')
    require(metadata['metrics'] == METRIC_UNITS, 'metric set changed')
    require(metadata['requested_range_count'] == 1 and metadata['range'] ==
            dict(occurrence=0, start_calls=1, stop_calls=1, completed=True), 'incomplete profiler range')
    require(metadata['output']['finite'] is True and metadata['output']['exact_reference'] is True and
            metadata['output']['readback_after_profiler_stop'] is True, 'missing finite/correct output receipt')
    require(metadata['precision_before'] == metadata['precision_after'] and metadata['precision_setters_added'] is False,
            'precision observation changed')
    require(metadata['environment_before'] == metadata['environment_after'], 'environment changed')
    record = parse_raw_csv(csv_path)
    require(record['pid'] == metadata['pid'] and record['cuda_device'] == metadata['selected_cuda_device'],
            'NCU process/device binding mismatch')
    return dict(schema='transfer_counter_range_binding_v1', nonce=expected_nonce,
        physical_gpu=expected_physical_gpu, gpu_uuid=expected_gpu_uuid, pid=metadata['pid'],
        source_sha256=metadata['source_sha256'], raw_range_identity_bound=True,
        actual_bound=record['one_pass'], one_pass_comparison_eligible=record['one_pass'], record=record,
        rejection_reason=None if record['one_pass'] else 'Multiple replay passes; rejected as a comparison with prior one-pass captures',
        counter_availability_only=True, model_hardware_benefit=None, background_attribution_proven=False,
        limitations=['One small synthetic app-range; availability does not establish model benefit.',
            'Successful one-pass collection on this workload does not guarantee one pass on model ranges.',
            'Cache/clock controls are disabled; no background subtraction or process-isolation claim.',
            'Each metric retains its native byte/sector/request/inst unit; counts are not interchangeable.'])


def main(argv=None):
    args = parse_args(argv)
    if args.action == 'command':
        values = vars(args).copy()
        values.pop('action')
        print(json.dumps(dict(command=build_ncu_command(**values), nonce=args.nonce,
            executed=False, environment_mutations={}), indent=2))
        return 0
    if args.action == 'run':
        run_probe(args)
        return 0
    metadata = json.loads(args.metadata.read_text())
    report = bind_raw_csv(metadata, args.raw_csv, expected_nonce=args.expected_nonce,
        expected_gpu_uuid=args.expected_gpu_uuid, expected_physical_gpu=args.expected_physical_gpu)
    report['metadata_sha256'] = digest(args.metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, report)
    print(json.dumps(dict(actual_bound=report['actual_bound'], passes=report['record']['passes'])))
    return 0 if report['actual_bound'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
