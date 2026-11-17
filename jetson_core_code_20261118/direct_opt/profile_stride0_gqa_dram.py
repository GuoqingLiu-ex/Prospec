"""Whole-generation DRAM ranges for an isolated native/stride0 comparison.

Both arms use shared embedding, compiled adaptor and native full DynamicCache.
Only Target tree attention changes. NCU timing, energy and allocator peaks are
diagnostic. Actual counters must be bound to the saved workload PID and ranges.
No GPU work occurs on import. Runtime quality must pass before this is used.
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
from profile_implicit_kv_dram import mark_diagnostic_rows
from summarize_dram_campaign import equal, load_metrics, require, validate_output

REQUEST = 'direct_quality_vdc_v_sPK-sbHIb3w'
ARMS = {mode: 'memory_shared_embedding_compiled_fullkv_tree_' + mode + '_math'
        for mode in ('native', 'stride0')}
MATH_PRECISION = dict(allow_tf32=False, float32_matmul_precision='highest',
                      reduced_precision_math_sdp=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--tree-attention', choices=tuple(ARMS), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--request-id', default=REQUEST)
    args, rest = p.parse_known_args(argv)
    forbidden = ('--warmup', '--repeats', '--max-new-tokens', '--resume', '--arms',
                 '--memory-layout', '--adaptor', '--include-ar', '--paired-baseline',
                 '--profile-scope', '--profile-calls')
    for token in rest:
        if token.startswith('--') and any(flag.startswith(token.split('=', 1)[0])
                                         for flag in forbidden):
            p.error('fresh fixed single-request profile: one warmup, two ranges, max512')
    return args, [*rest, '--tree-attention', args.tree_attention, '--out', args.out,
                  '--request-id', args.request_id, '--warmup', '1', '--repeats', '2',
                  '--max-new-tokens', '512']


def identity(status, mode):
    equal(status.get('tree_attention_mode'), mode, 'tree attention mode')
    equal(status.get('memory_layout'), 'shared_embedding', 'fixed embedding layout')
    equal(status.get('deployment_adaptor'), 'compiled', 'fixed visual adaptor')
    equal(status.get('target_cache_policy'), 'native_full_dynamic_cache', 'fixed cache')
    equal(status.get('new_kv_policy_installed'), False, 'no cache policy change')
    require('ring_variant' not in status, 'ring runtime must not be installed')


def validate_row(row, status, mode):
    """Check the same execution receipt for workload and raw-counter binding."""
    require(mode in ARMS, 'supported mode required')
    identity(status, mode)
    equal(status.get('profiler_mode'), mode, 'profile mode')
    run_id = status.get('profiler_run_id')
    require(isinstance(run_id, str) and bool(run_id), 'nonempty invocation identity')
    equal(status['args']['arms'], [ARMS[mode]], 'single fixed arm')
    request_ids = status['args']['request_id']
    require(isinstance(request_ids, list) and len(request_ids) == 1, 'single request')
    equal((row['arm'], row['request_id']), (ARMS[mode], request_ids[0]), 'row identity')
    validate_output(row, status)
    equal(row.get('under_ncu'), True, 'diagnostic marker')
    for key in ('valid_timing', 'valid_memory', 'valid_energy'):
        equal(row.get(key), False, key + ' is diagnostic')
    equal(row.get('profiler_mode'), mode, 'row profile mode')
    equal(row.get('profiler_run_id'), run_id, 'invocation binding')
    runtime = row['adaptor_preparation']['tree_attention']
    counts = row['result']['forward_counts']
    precision = row['result']['precision']
    rounds = [r for r in row['result']['rounds'] if r['kind'] == 'tree']
    for label, flags in (('status', status.get('tree_attention_precision')),
                         ('runtime', runtime.get('precision')),
                         ('result', precision.get('target_tree_math_precision'))):
        equal(flags, MATH_PRECISION, label + ' math precision')
    expected_precision = dict(
        target_sdpa='math' if mode == 'native' else 'math_prefill+stride0_fp32_math_tree',
        target_prefill_attention_backend='math',
        target_verify_attention_backend='math' if mode == 'native' else 'stride0_four_group_fp32_math',
        target_tree_attention_mode=mode, target_cache_policy='native_full_dynamic_cache',
        draft_attention_backend='unchanged')
    for key, value in expected_precision.items():
        equal(precision.get(key), value, 'result precision ' + key)
    equal(runtime['mode'], mode, 'runtime mode')
    require(runtime['finalized'] and runtime['result_coverage_validated'], 'complete runtime')
    equal(runtime['failures'], [], 'runtime failures')
    equal(runtime['fallback_calls'], 0, 'no backend fallback')
    equal(runtime['prefill_calls'], counts['target_prefill_calls'], 'prefill coverage')
    equal(runtime['prefill_layer_calls'], [1] * 28, 'native prefill coverage')
    verify = counts['target_verify_calls']
    equal(len(rounds), verify, 'verification rounds')
    equal(runtime['verify_forward_calls'], verify, 'runtime verify coverage')
    equal(runtime['verify_query_tokens'], counts['target_verify_query_tokens'], 'runtime query coverage')
    equal(runtime['selected_calls'], 28 * verify, 'selected layer coverage')
    equal(runtime['completed_calls'], 28 * verify, 'completed layer coverage')
    equal(runtime['layer_calls'], [verify] * 28, 'individual layer coverage')
    equal(runtime['candidate_calls'], 28 * verify if mode == 'stride0' else 0,
          'candidate coverage')
    equal(runtime['native_tree_calls'], 28 * verify if mode == 'native' else 0,
          'native tree coverage')
    equal(counts['target_tree_attention_policy_calls'], runtime['completed_calls'], 'saved policy calls')
    equal(counts['target_stride0_attention_calls'], runtime['candidate_calls'], 'saved candidate calls')
    equal(len(runtime['rounds']), verify, 'runtime round count')
    for observed, actual in zip(runtime['rounds'], rounds):
        equal((observed['cache_before'], observed['tree_nodes'], observed['cache_after']),
              (actual['target_cache_before'], actual['tree_nodes'],
               actual['target_cache_before'] + actual['tree_nodes']), 'full tree append')


def validate_workload(folder, mode, request_id):
    folder = Path(folder)
    status = json.loads((folder / 'status.json').read_text())
    identity(status, mode)
    for key, expected in dict(status='completed', profiler_mode=mode,
            profiler_scope='generation', profiler_warmup_calls=1,
            profiler_requested_calls=2, profiler_completed_ranges=2,
            profiler_capture_complete=True, generation_calls=3, under_ncu=True,
            total_cells=3, completed_cells=3).items():
        equal(status.get(key), expected, 'status ' + key)
    require(type(status.get('pid')) is int and status['pid'] > 0, 'workload PID')
    equal(status['args']['arms'], [ARMS[mode]], 'single fixed arm')
    equal(status['args']['request_id'], [request_id], 'single request')
    for key, expected in (('warmup', 1), ('repeats', 2), ('max_new_tokens', 512)):
        equal(status['args'][key], expected, 'argument ' + key)
    rows = [json.loads(line) for line in (folder / 'results.jsonl').read_text().splitlines()
            if line.strip()]
    equal([(r['phase'], r['repeat']) for r in rows],
          [('warmup', 0), ('measure', 0), ('measure', 1)], 'complete ordered cells')
    for row in rows:
        validate_row(row, status, mode)
    return status, rows


def bind_ncu(status, rows, csv_path):
    mode = status['profiler_mode']
    require(mode in ARMS, 'supported mode required')
    identity(status, mode)
    require(status['under_ncu'] and status['profiler_capture_complete'], 'complete ranges')
    equal(status['profiler_scope'], 'generation', 'whole generation scope')
    metrics = load_metrics(Path(csv_path))
    ranges = status['profiler_ranges']
    measured = [r for r in rows if r['phase'] == 'measure']
    for value in (len(metrics), len(ranges), len(measured), status['profiler_requested_calls'],
                  status['profiler_completed_ranges']):
        equal(value, 2, 'two actual ranges')
    equal({m['pid'] for m in metrics}, {status['pid']}, 'NCU/workload PID binding')
    records = []
    for index, (metric, capture, row) in enumerate(zip(metrics, ranges, measured)):
        equal((capture['range_index'], capture['call_index'], capture['completed']),
              (index, index + 1, True), 'completed range order')
        equal(row['repeat'], index, 'measured row order')
        equal(tuple(capture['cell'].get(key) for key in ('phase', 'repeat', 'arm', 'request_id')),
              tuple(row[key] for key in ('phase', 'repeat', 'arm', 'request_id')), 'range/cell binding')
        equal(capture['cell'].get('phase'), 'measure', 'warmup excluded')
        equal(capture['cell'].get('arm'), ARMS[mode], 'range recipe')
        equal(row['arm'], ARMS[mode], 'row recipe')
        validate_row(row, status, mode)
        records.append(dict(metric, request_id=row['request_id'], repeat=index,
            arm=row['arm'], profiler_range_index=index, generation_call_index=index + 1,
            tokens=row['num_tokens'], stop_reason=row['stop_reason'],
            output_token_ids=row['output_token_ids'], rounds=row['result']['rounds'],
            forward_counts=row['result']['forward_counts'], precision=row['result']['precision']))
    return records


def main(argv=None):
    args, forwarded = parse_args(argv)
    import run_joint
    import run_stride0_gqa_layout as runner
    import vispec_joint_adapter as joint
    import torch

    profiler = GenerationProfiler(torch.cuda, scope='generation', warmup_calls=1, profile_calls=2)
    run_id = uuid.uuid4().hex
    original_save = run_joint.save_json

    def save(path, value):
        profiler.observe_save(path, value)
        if Path(path).name == 'status.json':
            identity(value, args.tree_attention)
            value = dict(value, **profiler.metadata(), profiler_mode=args.tree_attention,
                profiler_run_id=run_id, profiler_effective_cpath=os.environ.get('CPATH'),
                profiler_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return original_save(path, value)

    try:
        with ExitStack() as stack:
            stack.enter_context(patched(run_joint, 'save_json', save))
            stack.enter_context(patched(joint, 'generate_joint', profiler.wrap(joint.generate_joint)))
            runner.main(forwarded)
            info = profiler.metadata()
            require(info['profiler_capture_complete'] and info['generation_calls'] == 3,
                    'one warmup plus two generation ranges required')
    finally:
        mark_diagnostic_rows(args.out, run_id)
    status, rows = validate_workload(args.out, args.tree_attention, args.request_id)
    receipt = dict(mode=args.tree_attention, workload_validated=True, cells=len(rows),
        all_calls_same_output=all(r['output_token_ids'] == rows[0]['output_token_ids'] for r in rows),
        all_calls_same_rounds=all(r['result']['rounds'] == rows[0]['result']['rounds'] for r in rows),
        all_calls_natural_eos=all(r['stop_reason'] == 'eos' for r in rows),
        actual_counters_bound=False, cross_mode_quality_pass=None)
    (Path(args.out) / 'stride0_profile_validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    main()
