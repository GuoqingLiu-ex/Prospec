"""One fixed full-history FP16 recipe with exact delta-commit maintenance.

No attention change, eviction or experiment on import. Use a fresh --out;
standard request selection, warmup/repeats and physical-GPU options pass through.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

VARIANT = 'active_width_delta_commit'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--max-new-tokens', type=int, choices=(512,), default=512)
    parser.add_argument('--recent-capacity', type=int, choices=(512,), default=512)
    parser.add_argument('--ring-policy', choices=('protected_prompt',), default='protected_prompt')
    parser.add_argument('--paired-baseline', action='store_true',
                        help='randomized matched requests with original active-width commit')
    args, rest = parser.parse_known_args(argv)
    forbidden = ('--resume', '--adaptor', '--memory-layout', '--visual-kv-mode', '--gqa-backend')
    # Parent parsers accept abbreviations; do not forward a shortened option
    # that can bypass this runner's fixed recipe/fresh-output constraint.
    if any(x.startswith('--') and any(option.startswith(x.split('=', 1)[0])
                                     for option in forbidden) for x in rest):
        parser.error('use fresh output and the fixed shared/compiled FP16 math recipe')
    return args, [*rest, '--max-new-tokens', '512']


def main(argv=None):
    args, rest = parse_args(argv)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from delta_commit_runtime import delta_commit_cache_policy
    from active_ring_runtime import active_ring_cache_policy

    original_runtime = deployed.DeployedRuntime
    original_main, original_save = run_joint.main, run_joint.save_json
    original_load, original_generate = bridge.load_spec, joint.generate_joint
    loaded, current = {}, {}

    def load(target, *a, **kw):
        result = original_load(target, *a, **kw)
        loaded['target'] = target
        return result

    class Runtime(original_runtime):
        @contextmanager
        def activate(self, joint_module, scheme):
            with super().activate(joint_module, scheme) as state:
                policy = None
                is_delta = scheme.get('commit_layout', 'delta') == 'delta'
                factory = delta_commit_cache_policy if is_delta else active_ring_cache_policy
                expected_variant = VARIANT if is_delta else 'active_width'
                try:
                    with factory(loaded['target'], recent_capacity=512,
                                 tree_capacity=30) as policy:
                        current['policy'] = policy
                        yield state
                finally:
                    current.clear()
                    if policy is not None:
                        info = policy.summary()
                        if info.get('variant') != expected_variant:
                            raise RuntimeError('delta commit provenance mismatch')
                        state['preparation']['ring_kv'] = info
                        state['preparation']['ring_variant'] = expected_variant
                        state['memory_phase_peaks'] = info['memory_phase_peaks']

    def generate(*a, **kw):
        result = original_generate(*a, **kw)
        policy = current['policy']
        verifies = result.forward_counts['target_verify_calls']
        if (len(policy.rounds) != verifies or policy.pending is not None or policy.failed
                or policy.prefill_calls != result.forward_counts['target_prefill_calls']):
            raise RuntimeError('delta commit did not cover every Target verification')
        return result

    def named_main():
        old = sys.argv[-1]
        if sys.argv[-2] != '--arms' or old != 'memory_shared_embedding_compiled':
            raise RuntimeError('unexpected delta-commit deployment arm')
        arm = old + '_ring_active_width_protected_prompt_512_delta_commit'
        scheme = dict(run_joint.SCHEMES[old])
        run_joint.SCHEMES = {arm: scheme}
        if args.paired_baseline:
            baseline = old + '_ring_active_width_protected_prompt_512'
            run_joint.SCHEMES[baseline] = dict(scheme, commit_layout='active')
            run_joint.SCHEMES[arm] = dict(scheme, commit_layout='delta')
        run_joint.ARMS = {name: ('fixed0', 1.0) for name in run_joint.SCHEMES}
        sys.argv[-1:] = list(run_joint.SCHEMES)
        original_main()

    def save(path, value):
        if Path(path).name == 'status.json':
            value = dict(value, ring_variant=VARIANT, ring_policy='protected_prompt',
                         recent_capacity=512, target_history_eviction=False,
                         draft_history_eviction=False, kv_quantized=False,
                         attention_backend_changed=False,
                         paired_active_baseline=args.paired_baseline,
                         commit_method='skip_inplace_rows_and_fuse_moved_rows',
                         quality_claim='Candidate: no automatic quality/DRAM/speed pass')
        original_save(path, value)

    deployed.DeployedRuntime = Runtime
    bridge.load_spec, joint.generate_joint = load, generate
    run_joint.main, run_joint.save_json = named_main, save
    try:
        memory.main(['--memory-layout', 'shared_embedding', '--adaptor', 'compiled', *rest])
    finally:
        deployed.DeployedRuntime = original_runtime
        bridge.load_spec, joint.generate_joint = original_load, original_generate
        run_joint.main, run_joint.save_json = original_main, original_save


if __name__ == '__main__':
    main()
