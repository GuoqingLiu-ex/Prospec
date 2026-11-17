"""Whole-generation NCU ranges for native ViSpec or implicit KV deployment.

Each fresh process has one unprofiled warmup and two measured calls for one
request. Native routes directly through run_memory_layout(native/native): no
ring/full policy, extra cache guards or phase synchronization are installed.
Implicit routes through its frozen single-arm deployment runner. Only the real
generate_joint call is inside cudaProfilerStart/Stop, including vision, native
prefill, candidate conversion, Draft and Target verification, and KV commits.
Loading, input preparation, text decoding and policy cleanup are excluded.

This entry collects no counters by itself. Run under NCU app-range replay and
bind its actual CSV to saved PID/cells with bind_ncu. Counter attribution still
requires the supervisor's process observations. NCU wall/energy/allocator data
are diagnostic; logical copy counts are not DRAM measurements. No CUDA work
or torch import occurs on import/argument parsing.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import uuid

from profile_ring_dram import GenerationProfiler, patched
from summarize_dram_campaign import equal, load_metrics, require, validate_output


REQUEST = 'direct_quality_vdc_v_sPK-sbHIb3w'
VARIANT = 'active_width_delta_commit_implicit_layout'
MODES = ('native', 'implicit')
ARMS = {
    'native': 'memory_native_native',
    'implicit': 'memory_shared_embedding_compiled_ring_active_width_protected_prompt_512_delta_commit_implicit_layout',
}
DIAGNOSTIC_NOTE = 'NCU diagnostic run; wall, energy and allocator peaks are not ordinary performance evidence'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--profile-mode', choices=MODES, default='native')
    parser.add_argument('--profile-scope', choices=('generation',), default='generation')
    parser.add_argument('--profile-calls', type=int, choices=(2,), default=2)
    parser.add_argument('--warmup', type=int, choices=(1,), default=1)
    parser.add_argument('--repeats', type=int, choices=(2,), default=2)
    parser.add_argument('--max-new-tokens', type=int, choices=(512,), default=512)
    parser.add_argument('--out', required=True)
    parser.add_argument('--request-id', action='append')
    args, rest = parser.parse_known_args(argv)
    args.request_id = args.request_id or [REQUEST]
    if len(args.request_id) != 1:
        parser.error('exactly one request per profiling process is required')
    forbidden = ('--resume', '--paired-baseline', '--memory-layout', '--adaptor',
                 '--ring-policy', '--recent-capacity', '--commit-backend', '--arms',
                 '--include-ar', '--capture-visual-dir', '--collect-local',
                 '--deployment-adaptor', '--fusion-mode', '--visual-kv-mode', '--gqa-backend')
    if any(token.startswith('--') and any(flag.startswith(token.split('=', 1)[0])
                                          for flag in forbidden) for token in rest):
        parser.error('fresh single-arm native or implicit profile only; the recipe is fixed')
    return args, [*rest, '--out', args.out, '--request-id', args.request_id[0],
                  '--warmup', '1', '--repeats', '2', '--max-new-tokens', '512']


def _identity(value, mode):
    equal((value.get('memory_layout'), value.get('deployment_adaptor')),
          ('native', 'native') if mode == 'native' else ('shared_embedding', 'compiled'),
          'profile model identity')
    if mode == 'native':
        require('ring_variant' not in value and 'ring_policy' not in value,
                'native baseline must not use ring/full instrumentation')
    else:
        equal(value.get('ring_variant'), VARIANT, 'implicit runtime identity')
        equal(value.get('paired_delta_baseline'), False, 'implicit profile must be single-arm')


def mark_diagnostic_rows(folder, run_id):
    """Mark only this invocation's newly written rows, after the writer closes.

    Existing/failing-to-open output directories cannot be touched: a fresh
    UUID in the status written by our save hook must match before any rewrite.
    Numeric observations are preserved, including incomplete runs.
    """
    folder = Path(folder)
    status_path, rows_path = folder/'status.json', folder/'results.jsonl'
    if not status_path.is_file() or not rows_path.is_file():
        return
    status = json.loads(status_path.read_text())
    if status.get('profiler_run_id') != run_id:
        return
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row.update(under_ncu=True, ncu_attachment_verified=False, valid_timing=False,
                   valid_memory=False, valid_energy=False, profiled_wall_is_performance_evidence=False,
                   profiler_mode=status['profiler_mode'], profiler_run_id=run_id)
        row.setdefault('unprofiled_memory_scope_label', row.get('memory_scope'))
        row['memory_scope'] = DIAGNOSTIC_NOTE
        row['timing_reasons'] = list(dict.fromkeys([*row.get('timing_reasons', []), 'under_ncu_diagnostic']))
    temporary = rows_path.with_suffix('.jsonl.profiler-tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    temporary.replace(rows_path)


def validate_workload(folder, mode, request_id):
    """Validate the actual three calls, without granting a cross-mode quality pass."""
    folder = Path(folder)
    status = json.loads((folder/'status.json').read_text())
    _identity(status, mode)
    expected = dict(status='completed', profiler_mode=mode, profiler_scope='generation',
        profiler_warmup_calls=1, profiler_requested_calls=2, profiler_completed_ranges=2,
        profiler_capture_complete=True, generation_calls=3, under_ncu=True,
        total_cells=3, completed_cells=3)
    for key, value in expected.items():
        equal(status.get(key), value, 'profile status '+key)
    require(type(status.get('pid')) is int and status['pid'] > 0, 'positive workload PID required')
    equal(status['args'].get('arms'), [ARMS[mode]], 'single profile arm')
    equal(status['args'].get('request_id'), [request_id], 'single profile request')
    for key, value in (('warmup', 1), ('repeats', 2), ('max_new_tokens', 512)):
        equal(status['args'].get(key), value, 'profile argument '+key)
    rows = [json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines() if line.strip()]
    equal([(r['phase'], r['repeat']) for r in rows],
          [('warmup', 0), ('measure', 0), ('measure', 1)], 'complete ordered workload cells')
    for row in rows:
        equal((row['arm'], row['request_id']), (ARMS[mode], request_id), 'result identity')
        validate_output(row, status)
        equal(row['under_ncu'], True, 'diagnostic row marker')
        equal(row['valid_timing'], False, 'NCU timing must stay diagnostic')
        equal(row['valid_memory'], False, 'NCU allocator observations must stay diagnostic')
        equal(row['profiler_run_id'], status['profiler_run_id'], 'row belongs to this invocation')
        tree = [r for r in row['result']['rounds'] if r['kind'] == 'tree']
        info = (row.get('adaptor_preparation') or {}).get('ring_kv')
        if mode == 'native':
            require(info is None, 'native baseline unexpectedly installed a ring policy')
        else:
            require(info is not None and info.get('finalized') and not info.get('failed'),
                    'implicit cache lifecycle incomplete')
            equal(info['variant'], VARIANT, 'result runtime variant')
            equal(info['prefill_calls'], row['result']['forward_counts']['target_prefill_calls'],
                  'runtime prefill coverage')
            equal(info['completed_verifications'], len(tree), 'runtime verification coverage')
            equal(len(info['rounds']), len(tree), 'runtime tree records')
            for rt, original in zip(info['rounds'], tree):
                equal((rt['logical_base'], rt['nodes'], rt['committed_tokens'], rt['semantic_length_after']),
                      (original['target_cache_before'], original['tree_nodes'], len(original['submitted']),
                       original['target_cache_after']), 'runtime/decoder tree trajectory')
                equal(rt['active_attention_key_tokens'], rt['logical_base']+rt['nodes'], 'full active attention width')
                require(rt['caller_attention_mask_reused'], 'implicit original mask not reused')
    return status, rows, dict(verified=True, profile_mode=mode, cells=len(rows),
        all_calls_same_output=all(r['output_token_ids'] == rows[0]['output_token_ids'] for r in rows),
        all_calls_same_rounds=all(r['result']['rounds'] == rows[0]['result']['rounds'] for r in rows),
        all_calls_natural_eos=all(r['stop_reason'] == 'eos' for r in rows),
        quality_pass=None, ncu_attachment_verified=False,
        note='Workload receipt only; actual DRAM requires bound NCU counters and reviewed process observations')


def bind_ncu(status, rows, csv_path):
    """Bind exactly two one-pass app-range counters to this process and cells."""
    mode = status['profiler_mode']
    require(mode in MODES, 'unsupported profile mode')
    _identity(status, mode)
    require(status['under_ncu'] and status['profiler_capture_complete'], 'incomplete profiling')
    equal(status['profiler_scope'], 'generation', 'generation counter scope')
    equal(status['profiler_warmup_calls'], 1, 'one unprofiled warmup')
    metrics, ranges = load_metrics(Path(csv_path)), status['profiler_ranges']
    measured = {(r['request_id'], r['repeat']): r for r in rows if r['phase'] == 'measure'}
    for count in (len(metrics), len(ranges), len(measured), status['profiler_requested_calls'],
                  status['profiler_completed_ranges']):
        equal(count, 2, 'two complete actual ranges required')
    equal({m['pid'] for m in metrics}, {status['pid']}, 'NCU/workload PID binding')
    records, used = [], set()
    for index, (capture, metric) in enumerate(zip(ranges, metrics)):
        equal((capture['range_index'], capture['call_index'], capture['completed']),
              (index, index+1, True), 'profile range order/completion')
        cell = capture['cell']
        equal(cell['phase'], 'measure', 'warmup must not be profiled')
        key = (cell['request_id'], cell['repeat'])
        require(key in measured and key not in used, 'missing/duplicate range cell')
        used.add(key)
        row = measured[key]
        equal(cell['arm'], row['arm'], 'range/result arm')
        equal(row['arm'], ARMS[mode], 'range recipe')
        validate_output(row, status)
        records.append(dict(metric, request_id=key[0], repeat=key[1], arm=row['arm'],
            profiler_range_index=index, generation_call_index=capture['call_index'],
            tokens=row['num_tokens'], stop_reason=row['stop_reason'], verify_rounds=row['verify_rounds'],
            output_token_ids=row['output_token_ids'], rounds=row['result']['rounds'],
            forward_counts=row['result']['forward_counts'], precision=row['result']['precision'],
            input_shapes=row['input_shapes'], timestamps=row['timestamps'], video_grid_thw=row['video_grid_thw']))
    equal(used, set(measured), 'all measured cells bound')
    return records


def main(argv=None):
    args, rest = parse_args(argv)
    import run_joint
    import vispec_joint_adapter as joint
    import torch

    if args.profile_mode == 'native':
        import run_memory_layout as runner
        forwarded = ['--memory-layout', 'native', '--adaptor', 'native', *rest]
    else:
        import run_implicit_kv_layout as runner
        forwarded = rest
    profiler = GenerationProfiler(torch.cuda, scope='generation', warmup_calls=1, profile_calls=2)
    run_id = uuid.uuid4().hex
    original_save = run_joint.save_json
    def save(path, value):
        profiler.observe_save(path, value)
        if Path(path).name == 'status.json':
            _identity(value, args.profile_mode)
            value = dict(value, **profiler.metadata(), profiler_mode=args.profile_mode,
                profiler_run_id=run_id, profiler_ring_variant=VARIANT if args.profile_mode == 'implicit' else None,
                native_ring_full_instrumentation=False,
                profiler_effective_cpath=os.environ.get('CPATH'),
                profiler_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return original_save(path, value)
    try:
        with ExitStack() as stack:
            stack.enter_context(patched(run_joint, 'save_json', save))
            stack.enter_context(patched(joint, 'generate_joint', profiler.wrap(joint.generate_joint)))
            runner.main(forwarded)
            info = profiler.metadata()
            require(info['profiler_capture_complete'] and info['generation_calls'] == 3,
                    'expected one warmup and two completed generation ranges')
    finally:
        mark_diagnostic_rows(args.out, run_id)
    status, rows, receipt = validate_workload(args.out, args.profile_mode, args.request_id[0])
    Path(args.out, 'implicit_profile_validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return dict(info, profile_mode=args.profile_mode, workload_validation=receipt)


if __name__ == '__main__':
    main()
