"""NCU generation ranges for one explicit visual-KV representation.

One unprofiled warmup and two measured calls. generate_joint contains native
visual/prefill, visual-bank conversion and generation. Model loading, input
preparation, text decoding and runtime finally/cleanup are outside the range.
No GPU work occurs on import, and all patches are scoped. NCU wall/energy/
allocator readings are diagnostic. Use a new output directory.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import statistics

from profile_ring_dram import GenerationProfiler, patched
from assess_visual_kv import candidate_receipt, generation
from summarize_dram_campaign import equal, load_metrics, require

VARIANT = 'segmented_visual_no_eviction'
REQUEST = 'direct_quality_vdc_v_sPK-sbHIb3w'
MODES = ('int8', 'fp16_visual')
BYTES = ('read_bytes', 'write_bytes', 'total_bytes')


def result_arm(mode):
    require(mode in MODES, 'explicit visual KV mode required')
    return 'memory_shared_embedding_compiled_visualkv_'+mode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--visual-kv-mode', choices=MODES, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--request-id', action='append')
    parser.add_argument('--profile-scope', choices=('generation',), default='generation')
    parser.add_argument('--profile-calls', type=int, choices=(2,), default=2)
    parser.add_argument('--warmup', type=int, choices=(1,), default=1)
    parser.add_argument('--repeats', type=int, choices=(2,), default=2)
    parser.add_argument('--max-new-tokens', type=int, choices=(512,), default=512)
    args, rest = parser.parse_known_args(argv)
    args.request_id = args.request_id or [REQUEST]
    if len(args.request_id) != 1:
        parser.error('one request per profiling process is required')
    forbidden = ('--resume', '--include-ar', '--capture-dir', '--capture-visual-dir', '--collect-local')
    if any(item.split('=', 1)[0] in forbidden for item in rest):
        parser.error('profiling requires a fresh single-arm run without extra capture instrumentation')
    return args, [*rest, '--visual-kv-mode', args.visual_kv_mode, '--out', args.out,
                  '--request-id', args.request_id[0], '--warmup', '1', '--repeats', '2',
                  '--max-new-tokens', '512']


def validate_workload(folder, mode, request_id):
    """Validate every warmup/measurement independently; no native path equality gate."""
    folder = Path(folder)
    status = json.loads((folder/'status.json').read_text())
    expected = dict(status='completed', kv_variant=VARIANT, visual_kv_mode=mode,
        profiler_kv_variant=VARIANT, profiler_visual_kv_mode=mode, profiler_scope='generation',
        profiler_warmup_calls=1, profiler_requested_calls=2, profiler_completed_ranges=2,
        profiler_capture_complete=True, under_ncu=True, generation_calls=3,
        memory_layout='shared_embedding', deployment_adaptor='compiled',
        visual_kv_max_generated=512, visual_kv_tree_capacity=30,
        target_history_eviction=False, draft_history_eviction=False,
        visual_quantized=mode == 'int8', total_cells=3, completed_cells=3)
    for key, value in expected.items():
        equal(status.get(key), value, 'visual profile receipt '+key)
    require(type(status.get('pid')) is int and status['pid'] > 0, 'positive workload PID required')
    selected = status['args']['request_id']
    selected = [selected] if isinstance(selected, str) else selected
    equal(selected, [request_id], 'profile receipt request differs from command')
    for key, value in (('warmup', 1), ('repeats', 2), ('max_new_tokens', 512)):
        equal(status['args'].get(key), value, 'profile argument '+key)
    equal(status['args'].get('arms'), [result_arm(mode)], 'single explicit result arm')
    rows = [json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines() if line.strip()]
    equal(len(rows), 3, 'complete three-cell result coverage')
    equal([(r.get('phase'), r.get('repeat')) for r in rows],
          [('warmup', 0), ('measure', 0), ('measure', 1)], 'warmup then two measured cells')
    cells = []
    for row in rows:
        equal(row.get('request_id'), request_id, 'single request result identity')
        equal(row.get('arm'), result_arm(mode), 'visual result arm identity')
        gen = generation(row, status)
        require(gen['receipt_valid'], 'invalid output/trace: '+str(gen['checks'])+' '+str(gen['trace_validation_error']))
        receipt = candidate_receipt(row, status, mode)
        require(receipt['passed'], 'invalid visual KV receipt: '+str(receipt['checks']))
        info = row['adaptor_preparation']['visual_kv']
        tree = [r for r in row['result']['rounds'] if r['kind'] == 'tree']
        equal(len(info['rounds']), len(tree), 'runtime/decoder verification coverage')
        trace = []
        for original, runtime in zip(tree, info['rounds']):
            equal(runtime['logical_base'], original['target_cache_before'], 'runtime semantic base')
            equal(runtime['nodes'], original['tree_nodes'], 'runtime query node count')
            equal(runtime['committed_tokens'], len(original['submitted']), 'runtime committed count')
            equal(runtime['semantic_length_after'], original['target_cache_after'], 'runtime committed semantic length')
            equal(runtime['visual_tokens']+runtime['fp16_attention_tokens'],
                  runtime['logical_base']+runtime['nodes'], 'both banks must cover full attention history')
            equal(runtime['logical_attention_tokens'], runtime['logical_base']+runtime['nodes'],
                  'logical attention width includes visual bank')
            trace.append(dict(selected_tree_indices=original['selected_tree_indices'],
                submitted=original['submitted'], logical_base=runtime['logical_base'],
                semantic_rope_base=runtime['semantic_rope_base'], nodes=runtime['nodes'],
                committed_tokens=runtime['committed_tokens'],
                visual_tokens=runtime['visual_tokens'], fp16_attention_tokens=runtime['fp16_attention_tokens'],
                logical_attention_tokens=runtime['logical_attention_tokens']))
        cells.append(dict(phase=row['phase'], repeat=row['repeat'], request_id=request_id,
            arm=row['arm'], tokens=row['num_tokens'], stop_reason=row['stop_reason'],
            natural_eos=gen['natural_eos'], verify_rounds=row['verify_rounds'],
            output_token_ids=row['output_token_ids'], forward_counts=row['result']['forward_counts'],
            precision=row['result']['precision'], mixed_attention_calls=receipt['mixed_attention_calls'],
            round_trace=trace, terminal_records=[r for r in row['result']['rounds'] if r['kind'] != 'tree']))
    return status, rows, dict(verified=True, cells=cells,
        all_calls_same_output=all(r['output_token_ids'] == rows[0]['output_token_ids'] for r in rows),
        all_calls_natural_eos=all(c['natural_eos'] for c in cells), quality_pass=None,
        selected_paths_required_to_match_native=False,
        interpretation='Each call is validated against its own saved trace; no native-output or quality pass inferred')


def bind_ncu(status, rows, csv_path):
    """Bind ordered one-pass app ranges to measured cells and the actual child PID."""
    equal(status['profiler_scope'], 'generation', 'NCU scope')
    equal(status['profiler_kv_variant'], VARIANT, 'NCU visual-KV identity')
    equal(status['profiler_visual_kv_mode'], status['visual_kv_mode'], 'NCU mode identity')
    require(status['under_ncu'] and status['profiler_capture_complete'], 'incomplete profiler instrumentation')
    metrics, ranges = load_metrics(Path(csv_path)), status['profiler_ranges']
    measured = {(r['request_id'], r['repeat']): r for r in rows if r['phase'] == 'measure'}
    equal(len(measured), 2, 'two unique measured cells')
    equal(len(metrics), 2, 'two actual NCU range counters')
    equal(len(ranges), 2, 'two profiler receipts')
    equal(status['profiler_completed_ranges'], 2, 'completed ranges')
    equal(status['profiler_requested_calls'], 2, 'requested ranges')
    equal({m['pid'] for m in metrics}, {status['pid']}, 'NCU/workload PID binding')
    used, records = set(), []
    for ordinal, (capture, metric) in enumerate(zip(ranges, metrics)):
        equal(capture['range_index'], ordinal, 'profile range order')
        equal(capture['call_index'], 1+ordinal, 'generation call order after warmup')
        require(capture['completed'], 'incomplete generation range')
        cell = capture['cell']
        equal(cell['phase'], 'measure', 'warmup must not be profiled')
        key = (cell['request_id'], cell['repeat'])
        require(key in measured and key not in used, 'range mapped to missing/duplicate cell')
        used.add(key)
        row = measured[key]
        equal(cell['arm'], row['arm'], 'range/result arm')
        records.append(dict(metric, request_id=key[0], repeat=key[1],
            profiler_range_index=ordinal, generation_call_index=capture['call_index'],
            tokens=row['num_tokens'], stop_reason=row['stop_reason'], verify_rounds=row['verify_rounds'],
            output_token_ids=row['output_token_ids'], forward_counts=row['result']['forward_counts'],
            precision=row['result']['precision'], input_shapes=row['input_shapes'], timestamps=row['timestamps'],
            video_grid_thw=row['video_grid_thw'],
            output_sha256=hashlib.sha256(json.dumps(row['output_token_ids']).encode()).hexdigest()))
    equal(used, set(measured), 'all measured cells bound')
    return records


def compare_reference(status, rows, records, reference_path):
    """Optional observed ratios; output/compute differences remain explicit."""
    old = json.loads(Path(reference_path).read_text())
    comparisons = {}
    device_ok = status['args']['physical_gpu'] == old['gpu'] and all(
        row.get('hardware', {}).get('device', {}).get('uuid') == old['gpu_uuid'] for row in rows)
    for name in ('native_full', 'optimized_full'):
        arm = old['arms'][name]
        require(arm['range_binding_verified'], 'reference range binding missing')
        protocol = {key: status.get(key) == value for key, value in arm['protocol'].items()}
        protocol.update(same_device=device_ok, same_seed=status['args']['seed'] == arm['seed'],
                        same_generation_limit=status['args']['max_new_tokens'] == arm['max_new_tokens'])
        baseline = {(r['request_id'], r['repeat']): r for r in arm['records']}
        equal(set(baseline), {(r['request_id'], r['repeat']) for r in records}, 'reference request/repeat cells')
        pairs = []
        for candidate in records:
            base = baseline[(candidate['request_id'], candidate['repeat'])]
            checks = dict(protocol)
            checks.update({key: candidate[key] == base[key] for key in ('input_shapes', 'timestamps', 'video_grid_thw')})
            checks['parameter_dtypes'] = all(candidate['precision'].get(k) is not None and candidate['precision'].get(k) == base['precision'].get(k)
                                             for k in ('target', 'draft', 'output_head'))
            checks['outputs_equal'] = candidate['output_token_ids'] == base['output_token_ids']
            checks['both_natural_eos'] = candidate['stop_reason'] == base['stop_reason'] == 'eos'
            checks['forward_counts_equal'] = candidate['forward_counts'] == base['forward_counts']
            pairs.append(dict(request_id=candidate['request_id'], repeat=candidate['repeat'], checks=checks,
                matched_comparison_eligible=all(checks.values()), baseline_range_id=base['range_id'],
                candidate_range_id=candidate['range_id'],
                observed_ratio_candidate_over_reference={m:candidate[m]/base[m] if base[m] else None for m in BYTES}))
        first = arm['records'][0]
        all_calls_equal = all(r['output_token_ids'] == first['output_token_ids'] and r['stop_reason'] == 'eos' for r in rows)
        medians = {m:statistics.median(r[m] for r in records) for m in BYTES}
        comparisons[name] = dict(pairs=pairs, all_three_calls_match_reference_output=all_calls_equal,
            matched_comparison_eligible=all_calls_equal and all(p['matched_comparison_eligible'] for p in pairs),
            observed_ratio_of_medians={m:medians[m]/arm['summary'][m]['median'] if arm['summary'][m]['median'] else None for m in BYTES},
            selected_path_identity_required=False,
            interpretation='Observed generation DRAM counter ratios; no background subtraction, quality or speed pass; intended attention/visual-storage precision differs')
    return dict(reference=str(reference_path), reference_attribution=old['attribution'], comparisons=comparisons)


def main(argv=None):
    args, rest = parse_args(argv)
    import run_joint
    import run_visual_kv_layout
    import vispec_joint_adapter as joint
    import torch

    profiler = GenerationProfiler(torch.cuda, scope='generation', warmup_calls=1, profile_calls=2)
    original_save = run_joint.save_json
    def save(path, value):
        profiler.observe_save(path, value)
        if Path(path).name == 'status.json':
            if value.get('kv_variant') != VARIANT or value.get('visual_kv_mode') != args.visual_kv_mode:
                raise RuntimeError('visual-KV status identity missing or changed')
            value = dict(value, **profiler.metadata(), profiler_kv_variant=VARIANT,
                profiler_visual_kv_mode=args.visual_kv_mode, profiler_effective_cpath=os.environ.get('CPATH'),
                profiler_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return original_save(path, value)
    with ExitStack() as stack:
        stack.enter_context(patched(run_joint, 'save_json', save))
        stack.enter_context(patched(joint, 'generate_joint', profiler.wrap(joint.generate_joint)))
        run_visual_kv_layout.main(rest)
        info = profiler.metadata()
        require(info['profiler_capture_complete'] and info['generation_calls'] == 3,
                'profiling did not complete one warmup and two generation ranges')
    status, rows, receipt = validate_workload(args.out, args.visual_kv_mode, args.request_id[0])
    # Validation is after both measured ranges and policy cleanup, with patches
    # restored. The supervisor independently binds counters; this is no NCU proof.
    Path(args.out, 'visual_profile_validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return dict(info, profiler_kv_variant=VARIANT, profiler_visual_kv_mode=args.visual_kv_mode,
                workload_validation=receipt)


if __name__ == '__main__':
    main()
