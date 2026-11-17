"""Independent all-token visual KV experiment; importing never launches a model.

Both modes use shared embeddings, the compiled full-token adaptor, unchanged
visual depth and the same two-bank attention kernel. int8 changes only visual
KV precision; fp16_visual is its numerical traversal control. No token eviction.
Root must validate the standalone CUDA kernel before running this entry point.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

VARIANT = 'segmented_visual_no_eviction'
MAX_GENERATED = 512
TREE_CAPACITY = 30


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--visual-kv-mode', required=True, choices=('int8', 'fp16_visual'))
    parser.add_argument('--memory-layout', choices=('shared_embedding',), default='shared_embedding')
    parser.add_argument('--adaptor', choices=('compiled',), default='compiled')
    args, rest = parser.parse_known_args(argv)
    # The new entry point owns every policy/attention choice. No silent stacking.
    forbidden = ('--ring-policy', '--recent-capacity', '--backend', '--gqa-backend',
                 '--profile-scope', '--capture-dir')
    if any(x.split('=', 1)[0] in forbidden for x in rest):
        parser.error('visual KV mode controls the KV and attention implementation')
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument('--out')
    probe.add_argument('--resume', action='store_true')
    probe.add_argument('--max-new-tokens', type=int, default=MAX_GENERATED)
    options, _ = probe.parse_known_args(rest)
    if not 1 <= options.max_new_tokens <= MAX_GENERATED:
        parser.error('max-new-tokens must be 1..512; the fixed text bank never evicts')
    if options.resume:
        try:
            old = json.loads((Path(options.out)/'status.json').read_text())
        except (OSError, ValueError, TypeError) as exc:
            parser.error('cannot validate visual KV resume: '+str(exc))
        expected = dict(kv_variant=VARIANT, visual_kv_mode=args.visual_kv_mode,
                        visual_kv_max_generated=MAX_GENERATED, visual_kv_tree_capacity=TREE_CAPACITY,
                        memory_layout='shared_embedding', deployment_adaptor='compiled')
        if any(old.get(k) != v for k, v in expected.items()):
            parser.error('resume requires identical visual mode, variant, capacities, memory layout and adaptor')
    return args, rest


def main(argv=None):
    args, rest = parse_args(argv)
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint_module
    from direct_opt.visual_kv_runtime import visual_kv_policy

    original_runtime = deployed.DeployedRuntime
    original_main, original_save = run_joint.main, run_joint.save_json
    original_load = bridge.load_spec
    original_generate = joint_module.generate_joint
    loaded, current = {}, {}

    def load_spec(target, *positional, **kwargs):
        result = original_load(target, *positional, **kwargs)
        loaded['target'] = target
        return result

    class VisualRuntime(original_runtime):
        @contextmanager
        def activate(self, joint, scheme):
            with super().activate(joint, scheme) as state:
                policy = None
                try:
                    with visual_kv_policy(loaded['target'], mode=args.visual_kv_mode,
                            max_generated=MAX_GENERATED, tree_capacity=TREE_CAPACITY) as policy:
                        current['policy'] = policy
                        yield state
                finally:
                    current.clear()
                    if policy is not None:
                        info = policy.summary()
                        if info.get('variant') != VARIANT or info.get('visual_mode') != args.visual_kv_mode:
                            raise RuntimeError('visual KV provenance mismatch')
                        state['preparation']['visual_kv'] = info
                        state['preparation']['kv_variant'] = VARIANT
                        state['preparation']['visual_kv_mode'] = args.visual_kv_mode
                        state['memory_phase_peaks'] = info['memory_phase_peaks']

    def generate(*positional, **kwargs):
        result = original_generate(*positional, **kwargs)
        policy = current['policy']
        verifies = result.forward_counts['target_verify_calls']
        expected = verifies * len(policy.attention_indices)
        if (policy.mixed_attention_calls != expected or policy.failed or
                policy.completed_target_forwards != verifies or len(policy.rounds) != verifies or
                policy.pending is not None or policy.prefill_calls != result.forward_counts['target_prefill_calls']):
            raise RuntimeError('mixed attention did not cover exactly all completed Target verification calls')
        result.precision = dict(result.precision,
            target_sdpa='math_prefill+triton_mixed_tree',
            target_visual_kv_storage='int8_group32_fp32scale' if args.visual_kv_mode == 'int8' else 'fp16',
            target_verify_attention_backend='triton_mixed',
            target_prefill_attention_backend='math', draft_attention_backend='unchanged')
        return result

    def save_json(path, value):
        if Path(path).name == 'status.json':
            value = dict(value, kv_variant=VARIANT, visual_kv_mode=args.visual_kv_mode,
                         visual_kv_max_generated=MAX_GENERATED, visual_kv_tree_capacity=TREE_CAPACITY,
                         target_history_eviction=False, draft_history_eviction=False,
                         visual_quantized=args.visual_kv_mode == 'int8',
                         attention_layout='visual_bank_plus_complete_fp16_text_tree',
                         quality_claim='Experimental precision/layout candidate; no quality or speed pass claimed',
                         kv_dram_measured=False)
        original_save(path, value)

    def named_main():
        old = sys.argv[-1]
        if sys.argv[-2] != '--arms' or old != 'memory_shared_embedding_compiled' or old not in run_joint.SCHEMES:
            raise RuntimeError('unexpected visual KV memory/adaptor arm')
        arm = old+'_visualkv_'+args.visual_kv_mode
        run_joint.SCHEMES = {arm: dict(run_joint.SCHEMES[old])}
        run_joint.ARMS = {arm: ('fixed0', 1.0)}
        sys.argv[-1] = arm
        original_main()

    deployed.DeployedRuntime = VisualRuntime
    bridge.load_spec = load_spec
    joint_module.generate_joint = generate
    run_joint.main, run_joint.save_json = named_main, save_json
    try:
        memory.main(['--memory-layout', args.memory_layout, '--adaptor', args.adaptor, *rest])
    finally:
        deployed.DeployedRuntime = original_runtime
        bridge.load_spec = original_load
        joint_module.generate_joint = original_generate
        run_joint.main, run_joint.save_json = original_main, original_save


if __name__ == '__main__':
    main()
