"""Independent active-width ring candidate; no experiment starts on import.

Uses shared Target/Draft embedding storage, compiled full-token adaptor, native
visual depth and unchanged ViSpec decoding. The fixed-width runner stays intact.
Use a new --out and otherwise the standard requests/repeats/warmup/GPU options.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

VARIANT = 'active_width'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--ring-policy', default='protected_prompt', choices=('protected_prompt', 'pure_recent'))
    parser.add_argument('--recent-capacity', type=int, default=512)
    parser.add_argument('--memory-layout', choices=('shared_embedding',), default='shared_embedding')
    parser.add_argument('--adaptor', choices=('compiled',), default='compiled')
    args, rest = parser.parse_known_args(argv)
    if args.recent_capacity < 1:
        parser.error('recent capacity must be positive')
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument('--out')
    probe.add_argument('--resume', action='store_true')
    options, _ = probe.parse_known_args(rest)
    if options.resume:
        try:
            old = json.loads((Path(options.out)/'status.json').read_text())
        except (OSError, ValueError, TypeError) as exc:
            parser.error('cannot validate active ring resume: '+str(exc))
        expected = dict(ring_variant=VARIANT, ring_policy=args.ring_policy,
                        recent_capacity=args.recent_capacity,
                        memory_layout='shared_embedding', deployment_adaptor='compiled')
        if any(old.get(k) != v for k, v in expected.items()):
            parser.error('resume requires the same active_width variant, policy, capacity, memory layout and adaptor')
    return args, rest


def main(argv=None):
    args, rest = parse_args(argv)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    from active_ring_runtime import active_ring_cache_policy

    original_runtime = deployed.DeployedRuntime
    original_main, original_save = run_joint.main, run_joint.save_json
    original_load = bridge.load_spec
    loaded = {}

    def load_spec(target, *positional, **kwargs):
        result = original_load(target, *positional, **kwargs)
        loaded['target'] = target
        return result

    class ActiveRuntime(original_runtime):
        @contextmanager
        def activate(self, joint, scheme):
            with super().activate(joint, scheme) as state:
                policy = None
                try:
                    with active_ring_cache_policy(loaded['target'], mode=args.ring_policy,
                            recent_capacity=args.recent_capacity, tree_capacity=30) as policy:
                        yield state
                finally:
                    if policy is not None:
                        info = policy.summary()
                        if info.get('variant') != VARIANT:
                            raise RuntimeError('active runtime returned the wrong variant provenance')
                        state['preparation']['ring_kv'] = info
                        state['preparation']['ring_variant'] = VARIANT
                        state['memory_phase_peaks'] = info['memory_phase_peaks']

    def save_json(path, value):
        if Path(path).name == 'status.json':
            value = dict(value, ring_variant=VARIANT, ring_policy=args.ring_policy,
                         recent_capacity=args.recent_capacity, ring_tree_capacity=30,
                         target_history_eviction=True, draft_history_eviction=False,
                         attention_layout='filled_history_plus_current_tree',
                         quality_claim='Active-width ring is an experimental eviction layout; no quality or speed pass claimed',
                         kv_dram_measured=False)
        original_save(path, value)

    def named_main():
        old = sys.argv[-1]
        if sys.argv[-2] != '--arms' or old != 'memory_shared_embedding_compiled' or old not in run_joint.SCHEMES:
            raise RuntimeError('unexpected active ring memory/adaptor arm')
        arm = old+'_ring_'+VARIANT+'_'+args.ring_policy+'_'+str(args.recent_capacity)
        run_joint.SCHEMES = {arm: dict(run_joint.SCHEMES[old])}
        run_joint.ARMS = {arm: ('fixed0', 1.0)}
        sys.argv[-1] = arm
        original_main()

    deployed.DeployedRuntime = ActiveRuntime
    bridge.load_spec = load_spec
    run_joint.main, run_joint.save_json = named_main, save_json
    try:
        memory.main(['--memory-layout', args.memory_layout, '--adaptor', args.adaptor, *rest])
    finally:
        deployed.DeployedRuntime = original_runtime
        bridge.load_spec = original_load
        run_joint.main, run_joint.save_json = original_main, original_save


if __name__ == '__main__':
    main()
