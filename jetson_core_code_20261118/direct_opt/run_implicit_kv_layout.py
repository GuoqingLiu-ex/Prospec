"""Fixed full-history FP16 deployment with implicit contiguous KV metadata.

Use a fresh output directory. The optional paired baseline is the existing
delta-commit cache, with the same shared embedding and compiled adaptor.
No model, CUDA context, or experiment starts on import.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys


VARIANT = 'active_width_delta_commit_implicit_layout'
BASELINE_VARIANT = 'active_width_delta_commit'
BASE_ARM = 'memory_shared_embedding_compiled_ring_active_width_protected_prompt_512_delta_commit'
CANDIDATE_ARM = BASE_ARM + '_implicit_layout'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--max-new-tokens', type=int, choices=(512,), default=512)
    parser.add_argument('--recent-capacity', type=int, choices=(512,), default=512)
    parser.add_argument('--ring-policy', choices=('protected_prompt',), default='protected_prompt')
    parser.add_argument('--paired-baseline', action='store_true',
                        help='randomized matched requests against the original delta-commit runtime')
    args, rest = parser.parse_known_args(argv)
    forbidden = ('--resume', '--adaptor', '--memory-layout', '--visual-kv-mode',
                 '--gqa-backend', '--commit-backend', '--arms', '--include-ar',
                 '--capture-visual-dir', '--collect-local', '--deployment-adaptor',
                 '--fusion-mode', '--difference-threshold')
    if any(token.startswith('--') and any(option.startswith(token.split('=', 1)[0])
                                          for option in forbidden) for token in rest):
        parser.error('use fresh output and the fixed shared/compiled FP16 math recipe')
    return args, [*rest, '--max-new-tokens', '512']


def main(argv=None):
    args, rest = parse_args(argv)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from implicit_kv_runtime import implicit_cache_policy, VARIANT as RUNTIME_VARIANT
    from delta_commit_runtime import delta_commit_cache_policy

    if RUNTIME_VARIANT != VARIANT:
        raise RuntimeError('implicit runtime variant differs from runner')
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
            layout = scheme.get('cache_layout')
            if layout not in ('implicit', 'delta'):
                raise ValueError('runner requires an explicit implicit or delta cache layout')
            if current:
                raise RuntimeError('nested request cache activation is unsupported')
            with super().activate(joint_module, scheme) as state:
                policy = None
                expected = VARIANT if layout == 'implicit' else BASELINE_VARIANT
                factory = implicit_cache_policy if layout == 'implicit' else delta_commit_cache_policy
                options = dict(recent_capacity=512, tree_capacity=30)
                if layout == 'implicit':
                    options.update(mode='protected_prompt', commit_backend='triton')
                try:
                    with factory(loaded['target'], **options) as policy:
                        current['policy'] = policy
                        yield state
                finally:
                    current.clear()
                    if policy is not None:
                        info = policy.summary()
                        if info.get('variant') != expected:
                            raise RuntimeError('cache layout provenance mismatch')
                        state['preparation'].update(ring_kv=info, ring_variant=expected,
                                                    cache_layout=layout)
                        state['memory_phase_peaks'] = info['memory_phase_peaks']

    def generate(*a, **kw):
        if 'policy' not in current:
            raise RuntimeError('generation requires an active request cache policy')
        result = original_generate(*a, **kw)
        policy = current['policy']
        counts = result.forward_counts
        if (len(policy.rounds) != counts['target_verify_calls'] or
                policy.pending is not None or policy.failed or
                policy.prefill_calls != counts['target_prefill_calls']):
            raise RuntimeError('cache policy did not cover every Target prefill and verification')
        return result

    def named_main():
        old = sys.argv[-1]
        if sys.argv[-2] != '--arms' or old != 'memory_shared_embedding_compiled':
            raise RuntimeError('unexpected implicit-cache deployment arm')
        scheme = dict(run_joint.SCHEMES[old])
        run_joint.SCHEMES = {CANDIDATE_ARM: dict(scheme, cache_layout='implicit')}
        if args.paired_baseline:
            run_joint.SCHEMES[BASE_ARM] = dict(scheme, cache_layout='delta')
        run_joint.ARMS = {name: ('fixed0', 1.0) for name in run_joint.SCHEMES}
        sys.argv[-1:] = list(run_joint.SCHEMES)
        original_main()

    def save(path, value):
        if Path(path).name == 'status.json':
            value = dict(value, ring_variant=VARIANT, ring_policy='protected_prompt',
                recent_capacity=512, target_history_eviction=False,
                draft_history_eviction=False, kv_quantized=False,
                attention_backend_changed=False, paired_delta_baseline=args.paired_baseline,
                candidate_ring_variant=VARIANT,
                compared_ring_variants=([VARIANT, BASELINE_VARIANT] if args.paired_baseline else [VARIANT]),
                commit_method='skip_inplace_rows_and_fuse_moved_rows',
                quality_claim='Candidate: no automatic quality, memory, DRAM or speed pass')
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
