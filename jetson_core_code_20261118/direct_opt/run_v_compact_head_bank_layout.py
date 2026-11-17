"""Compact mixed-head value bank experiment, requiring two actual primitive GPU gates.

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

MODES = ('grouped_control', 'overlay_control', 'compact14')
METHOD = 'full_k_compact_mixed_head_bank14_grouped_math'
_ACTIVE = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--v-storage', choices=MODES, required=True)
    p.add_argument('--head-pack-validation')
    p.add_argument('--compact-validation')
    p.add_argument('--overlay-validation')
    p.add_argument('--requests', required=True)
    p.add_argument('--request-id', action='append')
    p.add_argument('--out', required=True)
    p.add_argument('--physical-gpu', type=int, choices=range(8), required=True)
    p.add_argument('--warmup', type=int, choices=(1,), default=1)
    p.add_argument('--repeats', type=int, choices=(2,), default=2)
    p.add_argument('--max-new-tokens', type=int, choices=(512,), default=512)
    p.add_argument('--seed', type=int, choices=(20261001,), default=20261001)
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
    if mode == 'overlay_control':
        return 'memory_shared_embedding_compiled_fullk_v_head_overlay14_grouped_math'
    return 'memory_shared_embedding_compiled_fullk_v_compact_head_bank14_grouped_math'


def dependency_hashes():
    root = Path(__file__).resolve().parent
    names = ('run_v_compact_head_bank_layout.py', 'v_compact_head_bank_model_runtime.py', 'grouped_compact_head_bank_math.py',
        'v_head_scale_cuda_ops.py','v_compact_head_bank_cuda_ops.py','v_head_overlay_cuda_ops.py','probe_v_head_scale_cuda.py',
        'probe_v_compact_head_bank_cuda.py','v_bank_model_runtime.py','grouped_v_bank_math.py',
        'partitioned_v_bank.py','v_bank_cuda_ops.py','ring_kv_cache.py','stride0_gqa_runtime.py',
        'run_v_bank_layout.py','run_grouped_gqa_layout.py','run_memory_layout.py','run_deployed_adaptor.py','run_joint.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def validate_primitive_gates(head_path,compact_path):
    if not head_path or not compact_path:
        raise ValueError('both actual row-head and compact-head GPU probe receipts are required')
    schemas={'head':('partitioned_v_head_scale_probe_v1','probe_v_head_scale_cuda'),
             'compact':('compact_mixed_visual_v_heads_probe_v1','probe_v_compact_head_bank_cuda')}
    receipts={}
    for kind,path in (('head',head_path),('compact',compact_path)):
        path=Path(path);raw=path.read_bytes();report=json.loads(raw)
        schema,module=schemas[kind]
        if (report.get('schema')!=schema or report.get('status')!='completed'
                or report.get('numerical_gate_passed') is not True
                or report.get('cuda_device',{}).get('capability')!=[12,0]
                or report.get('triton_version')!='3.4.0'
                or type(report.get('pid')) is not int or report['pid']<1
                or not isinstance(report.get('probe_nonce'),str) or len(report['probe_nonce'])!=32):
            raise ValueError(kind+' requires a completed actual SM120 primitive gate')
        import importlib
        probe=importlib.import_module(module)
        probe.validate_workload_report(report)
        cache_fields=('head_score_executor_cache',) if kind=='head' else ('copy_executor_cache','fill_executor_cache')
        for row in report['cases']:
            if row.get('cpu_lifecycle_double'):
                raise ValueError('CPU lifecycle fixture is not an actual primitive gate')
            for field in cache_fields:
                entries=row.get(field,[])
                if not entries or any(e.get('target')!={'backend':'cuda','arch':120,'warp_size':32}
                        or e.get('ptx_target') not in ('sm_120','sm_120a')
                        or not isinstance(e.get('ptx_sha256'),str) or len(e['ptx_sha256'])!=64
                        or not isinstance(e.get('cache_key'),str) or not e['cache_key'] for e in entries):
                    raise ValueError(kind+' lacks actual SM120 compiled-executor evidence')
        expected_sources=probe.source_receipt()
        names=[Path(k).name for k in expected_sources]
        if len(names)!=len(set(names)):
            raise RuntimeError('probe source receipt has ambiguous basenames')
        for source,expected in expected_sources.items():
            name=Path(source).name
            actual=[v for k,v in report.get('source_sha256',{}).items() if Path(k).name==name]
            if actual!=[expected]:raise ValueError(kind+' source missing/ambiguous/changed: '+name)
        receipts[kind]=dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
            schema=schema,pid=report['pid'],probe_nonce=report['probe_nonce'],cuda_device=report['cuda_device'],
            resource_mode=report.get('resource_mode'),source_sha256=report['source_sha256'],
            scope='prior primitive correctness gate, not model quality/latency/DRAM proof')
    return receipts


def _candidate_main(args, rest, gates):
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from v_compact_head_bank_model_runtime import compact_head_bank_model_policy, math_precision_scope, precision_flags, VARIANT

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
                            v_storage=mode, target_cache_policy=METHOD, tree_attention='grouped_compact_head_bank')
            if scheme != expected:
                raise ValueError('request differs from fixed V-bank/shared/compiled/full-visual recipe')
            if 'target' not in loaded:
                raise RuntimeError('target must be loaded before V-bank activation')
            last_request.clear()
            last_request.update(summary_available=False, mode=mode)
            with super().activate(joint_module, scheme) as state:
                policy = None
                try:
                    with compact_head_bank_model_policy(loaded['target']) as policy:
                        current['policy'] = policy
                        yield state
                finally:
                    current.clear()
                    if policy is not None:
                        # Context already performed finish + restoration + release.
                        info = policy.summary()
                        state['preparation']['compact_head_bank'] = info
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
                          target_compact_head_bank_attention_calls=policy.record['candidate_calls'])
        if set(new_counts) & set(result.forward_counts):
            raise RuntimeError('policy counters collide with existing generation counters')
        result.precision = dict(result.precision,
            target_sdpa='math_prefill+grouped_compact_head_bank_fp32_math_tree',
            target_prefill_attention_backend='math',
            target_verify_attention_backend='grouped_two_bmm_compact_head_bank_fp32_math',
            target_tree_attention_mode='grouped_compact_head_bank', target_tree_math_precision=precision_flags(),
            target_cache_policy=METHOD, target_key_storage='full_fp16_native_cat_index_select',
            target_visual_value_storage='compact_98_v8_group32_plus_14_original_fp16_heads',
            compact_protected_heads=14, compact_quantized_heads=98, compact_head_units=112, compact_selection='stable_scale_proxy_GPU',
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
            target_cache_policy=METHOD, tree_attention='grouped_compact_head_bank')}
        run_joint.ARMS = {arm: ('fixed0', 1.0)}
        sys.argv[-1] = arm
        return old_main()

    def save(path, value):
        if Path(path).name == 'status.json':
            value = dict(value, v_storage=mode, target_cache_policy=METHOD,
                compact_head_bank_source_sha256=sources, compact_head_bank_last_request=dict(last_request),
                primitive_gates=gates, compact_protected_heads=14, compact_quantized_heads=98, compact_head_units=112,
                new_kv_policy_installed=True, target_history_eviction=False,
                kv_quantized=True, quantized_component='unprotected_visual_value_only',
                target_key_storage='full_fp16_native_cat_index_select',
                tree_attention_mode='grouped_compact_head_bank', tree_attention_precision=precision_flags(),
                native_unused_probability_cast_preserved=True, attention_backend_fallback=False,
                ordinary_primitive_record_receipt=False, maximum_representative_receipts_per_request=1,
                memory_acceptance_rule='whole Target+Draft allocated AND reserved must not increase versus matched previous-version control; theoretical backing is insufficient',
                allocator_empty_cache_called=False,
                request_peak_rule='max(policy prefill/conversion/decode peaks, parent post-cleanup peak); allocated/reserved separately',
                request_timing_scope='input preparation through policy finish/release/summary and final synchronization; disk status writes excluded',
                generation_timing_scope='generate_joint including result device-status validation; excludes output decode and policy finalization',
                logical_io_scope='payload/accounting receipts only, not actual DRAM or kernel counts',
                methodology='compact_98_V8_14_original_FP16_global_banks_frozen_row_scale_proxy',
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
            if args.head_pack_validation or args.overlay_validation or args.compact_validation:
                raise ValueError('grouped control delegates unchanged and accepts no primitive gates')
            from run_v_bank_layout import main as frozen_main
            return frozen_main(['--v-storage','grouped_control',*rest])
        if args.v_storage == 'overlay_control':
            if args.compact_validation:
                raise ValueError('overlay control cannot use a compact gate')
            if not args.head_pack_validation or not args.overlay_validation:
                raise ValueError('overlay control requires its original head/overlay gates')
            from run_v_head_overlay_layout import main as frozen_main
            return frozen_main(['--v-storage','head_overlay14','--head-pack-validation',args.head_pack_validation,
                '--overlay-validation',args.overlay_validation,*rest])
        if args.overlay_validation:
            raise ValueError('compact candidate cannot use old overlay gate')
        gates=validate_primitive_gates(args.head_pack_validation,args.compact_validation)
        return _candidate_main(args, rest, gates)
    finally:
        _ACTIVE = False


if __name__ == '__main__':
    main()
