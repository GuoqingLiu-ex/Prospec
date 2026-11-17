"""Isolated V-bank author-CPU-tree reuse entry, frozen native control optional.

No new quantization/attention/depth/acceptance algorithm. Request timing/peak
handling is copied from the frozen V-bank entry. See the host-tree-plan contract.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager,ExitStack
import hashlib
from pathlib import Path
import sys
import run_v_bank_layout as parent

METHOD=parent.METHOD
_ACTIVE=False

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--host-tree-plan',choices=('frozen','reuse'),required=True)
    outer,rest=p.parse_known_args(argv)
    args,parent_rest=parent.parse_args(rest)
    if args.v_storage=='grouped_control':p.error('choose fixed fp16_partition or int8_visual')
    return outer,args,rest,parent_rest


def arm_name(mode):return parent.arm_name(mode)+'_host_plan'


def dependency_hashes():
    paths=[Path(__file__).with_name(n) for n in ('run_v_bank_host_plan.py','v_bank_host_plan_runtime.py','host_tree_joint.py')]
    return dict(parent.dependency_hashes(),**{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _candidate_main(args, rest):
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from v_bank_model_runtime import math_precision_scope, precision_flags, VARIANT
    from v_bank_host_plan_runtime import v_bank_host_plan_policy as v_bank_model_policy
    import host_tree_joint as reused_joint
    generation_contract=reused_joint.validate_generation_copy()

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
        result = reused_joint.generate_joint(*a, host_tree_broker=current['policy'].broker, **kw)
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
                methodology='fixed_visual_v_storage_host_tree_metadata_reuse',
                host_tree_metadata_path='author_cpu_tree_reuse_v1',
                generation_copy_contract=generation_contract,
                host_plan_runtime_source_sha256=sources,
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
    global _ACTIVE
    outer,args,rest,parent_rest=parse_args(argv)
    if _ACTIVE:raise RuntimeError('nested host-plan runner')
    parent.validate_request_selection(args)
    _ACTIVE=True
    try:
        if outer.host_tree_plan=='frozen':return parent.main(rest)
        return _candidate_main(args,parent_rest)
    finally:_ACTIVE=False


if __name__=='__main__':main()
