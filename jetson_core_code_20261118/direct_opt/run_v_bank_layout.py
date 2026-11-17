"""Fixed V-only cache experiment via frozen full-request parent runners.

Control delegates to grouped unchanged. Candidates retain full K, original
Draft/acceptance and the unused probability cast. CPU/import/help perform no
GPU work. See docs/plans/2026-10-03-v-bank-runner.md for timing/peak boundaries.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys

MODES = ('grouped_control', 'fp16_partition', 'int8_visual')
METHOD = 'full_k_partitioned_visual_v_grouped_math'
_ACTIVE = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--v-storage', choices=MODES, required=True)
    p.add_argument('--requests', required=True)
    p.add_argument('--request-id', action='append')
    p.add_argument('--out', required=True)
    p.add_argument('--physical-gpu', type=int, choices=range(8), required=True)
    p.add_argument('--warmup', type=int, choices=(1,), default=1)
    p.add_argument('--repeats', type=int, choices=(2,), default=2)
    p.add_argument('--max-new-tokens', type=int, choices=(512,), default=512)
    p.add_argument('--seed', type=int, default=20261001)
    args = p.parse_args(argv)
    rest = ['--requests', args.requests, '--out', args.out, '--physical-gpu', str(args.physical_gpu),
            '--warmup', '1', '--repeats', '2', '--max-new-tokens', '512', '--seed', str(args.seed)]
    for value in args.request_id or []:
        rest.extend(('--request-id', value))
    return args, rest


def validate_request_selection(args):
    """CPU-only input gate; leave manifest contents and request order intact."""
    if Path(args.out).exists():
        raise FileExistsError('V-bank comparison requires a fresh output directory')
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    ids = [r['request_id'] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('manifest requires unique nonempty request IDs')
    selected = args.request_id
    if selected is not None:
        if len(set(selected)) != len(selected) or set(selected) - set(ids):
            raise ValueError('request selection is duplicated or missing from manifest')
        ids = [i for i in ids if i in selected]
    if len(ids) not in (1, 12):
        raise ValueError('fixed protocol requires one pilot or twelve requests')
    return ids


def arm_name(mode):
    if mode not in MODES:
        raise ValueError('unsupported V storage')
    if mode == 'grouped_control':
        return 'memory_shared_embedding_compiled_fullkv_tree_grouped_math'
    return 'memory_shared_embedding_compiled_fullk_v_' + mode + '_grouped_math'


def dependency_hashes():
    root = Path(__file__).resolve().parent
    names = ('run_v_bank_layout.py', 'v_bank_model_runtime.py', 'grouped_v_bank_math.py',
             'partitioned_v_bank.py', 'v_bank_cuda_ops.py', 'stride0_gqa_runtime.py',
             'run_grouped_gqa_layout.py', 'run_memory_layout.py', 'run_deployed_adaptor.py', 'run_joint.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _candidate_main(args, rest):
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from v_bank_model_runtime import v_bank_model_policy, math_precision_scope, precision_flags, VARIANT

    if VARIANT != METHOD:
        raise RuntimeError('V-bank runtime variant differs from runner contract')
    old_runtime = deployed.DeployedRuntime
    old_main, old_save = run_joint.main, run_joint.save_json
    old_load, old_generate = bridge.load_spec, joint.generate_joint
    loaded, current, last_request = {}, {}, {}
    sources = dependency_hashes()
    mode, arm = args.v_storage, arm_name(args.v_storage)

    def load(target, *a, **kw):
        spec = old_load(target, *a, **kw)
        loaded['target'] = target
        return spec

    class Runtime(old_runtime):
        @contextmanager
        def activate(self, joint_module, scheme):
            if current:
                raise RuntimeError('nested V-bank request activation')
            expected = dict(depth='fixed0', adaptor='compiled', compression=None, ratio=1.0,
                            v_storage=mode, target_cache_policy=METHOD, tree_attention='grouped_v_bank')
            if scheme != expected:
                raise ValueError('request differs from fixed V-bank/shared/compiled/full-visual recipe')
            if 'target' not in loaded:
                raise RuntimeError('target must be loaded before V-bank activation')
            last_request.clear()
            last_request.update(summary_available=False, mode=mode)
            with super().activate(joint_module, scheme) as state:
                policy = None
                try:
                    with v_bank_model_policy(loaded['target'], mode=mode) as policy:
                        current['policy'] = policy
                        yield state
                finally:
                    current.clear()
                    if policy is not None:
                        # Context already performed finish + restoration + release.
                        info = policy.summary()
                        state['preparation']['v_bank'] = info
                        state['preparation']['v_storage'] = mode
                        state['memory_phase_peaks'] = info['memory_phase_peaks']
                        last_request.clear()
                        last_request.update(summary_available=True, mode=mode, finalized=info.get('finalized'),
                            result_coverage_validated=info.get('result_coverage_validated', False),
                            failures=info.get('failures', []),
                            memory_phase_peaks=info['memory_phase_peaks'])

    def generate(*a, **kw):
        if 'policy' not in current:
            raise RuntimeError('generation requires active V-bank policy')
        result = old_generate(*a, **kw)
        policy = current['policy']
        policy.validate_result(result)
        new_counts = dict(target_tree_attention_policy_calls=policy.record['completed_calls'],
                          target_v_bank_attention_calls=policy.record['candidate_calls'])
        if set(new_counts) & set(result.forward_counts):
            raise RuntimeError('policy counters collide with existing generation counters')
        result.precision = dict(result.precision,
            target_sdpa='math_prefill+grouped_v_bank_fp32_math_tree',
            target_prefill_attention_backend='math',
            target_verify_attention_backend='grouped_two_bmm_v_bank_fp32_math',
            target_tree_attention_mode='grouped_v_bank', target_tree_math_precision=precision_flags(),
            target_cache_policy=METHOD, target_key_storage='full_fp16_native_cat_index_select',
            target_visual_value_storage=('int8_group32_fp32_scale' if mode == 'int8_visual' else 'fp16_partition'),
            target_nonvisual_value_storage='compact_fp16_native_cat_index_select',
            target_value_operand='one_direct_fill_fp32_four_kv_heads',
            v_storage=mode, draft_attention_backend='unchanged',
            tree_attention_discarded_probability_cast_removed=False,
            tree_attention_bitwise_equivalence_claim=False)
        result.forward_counts.update(new_counts)
        return result

    def named_main():
        old = 'memory_shared_embedding_compiled'
        if sys.argv[-2:] != ['--arms', old]:
            raise RuntimeError('unexpected V-bank parent arm')
        run_joint.SCHEMES = {arm: dict(run_joint.SCHEMES[old], v_storage=mode,
            target_cache_policy=METHOD, tree_attention='grouped_v_bank')}
        run_joint.ARMS = {arm: ('fixed0', 1.0)}
        sys.argv[-1] = arm
        return old_main()

    def save(path, value):
        if Path(path).name == 'status.json':
            value = dict(value, v_storage=mode, target_cache_policy=METHOD,
                v_bank_source_sha256=sources, v_bank_last_request=dict(last_request),
                new_kv_policy_installed=True, target_history_eviction=False,
                kv_quantized=(mode == 'int8_visual'), quantized_component=('visual_value_only' if mode == 'int8_visual' else None),
                target_key_storage='full_fp16_native_cat_index_select',
                tree_attention_mode='grouped_v_bank', tree_attention_precision=precision_flags(),
                native_unused_probability_cast_preserved=True, attention_backend_fallback=False,
                ordinary_primitive_record_receipt=False, maximum_representative_receipts_per_request=1,
                request_peak_rule='max(policy prefill/conversion/decode peaks, parent post-cleanup peak); allocated/reserved separately',
                request_timing_scope='input preparation through policy finish/release/summary and final synchronization; disk status writes excluded',
                generation_timing_scope='generate_joint including result device-status validation; excludes output decode and policy finalization',
                logical_io_scope='payload/accounting receipts only, not actual DRAM or kernel counts',
                methodology='fixed_visual_v_storage_grouped_control_vs_fp16_partition_vs_int8_visual',
                quality_claim='Actual output IDs, full rounds and long-answer facts require evaluation',
                output_equivalence_pass=None, quality_pass=None, performance_pass=None, actual_dram_measured=False)
        return old_save(path, value)

    with ExitStack() as stack:
        for owner, name, replacement in ((deployed, 'DeployedRuntime', Runtime), (bridge, 'load_spec', load),
                (joint, 'generate_joint', generate), (run_joint, 'main', named_main), (run_joint, 'save_json', save)):
            stack.callback(setattr, owner, name, getattr(owner, name))
            setattr(owner, name, replacement)
        with math_precision_scope():
            return memory.main(['--memory-layout', 'shared_embedding', '--adaptor', 'compiled', *rest])


def main(argv=None):
    args, rest = parse_args(argv)
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError('nested V-bank runner invocation')
    validate_request_selection(args)
    _ACTIVE = True
    try:
        if args.v_storage == 'grouped_control':
            from run_grouped_gqa_layout import main as grouped_main
            return grouped_main(['--tree-attention', 'grouped', *rest])
        return _candidate_main(args, rest)
    finally:
        _ACTIVE = False


if __name__ == '__main__':
    main()
