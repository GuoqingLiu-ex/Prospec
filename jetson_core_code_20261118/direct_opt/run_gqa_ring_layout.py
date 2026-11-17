"""Independent full-KV, active-width ring with explicit fused tree attention.

Native prefill and Draft attention stay unchanged. Backend failures abort; no
fallback. First-call compilation/engine setup is inside the request timing.
This runner launches no model at import and requires a fresh result directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gqa-backend', choices=('cudnn', 'flex'), required=True)
    parser.add_argument('--ring-policy', choices=('protected_prompt',), default='protected_prompt')
    parser.add_argument('--recent-capacity', type=int, choices=(512,), default=512)
    parser.add_argument('--max-new-tokens', type=int, choices=(512,), default=512)
    args, rest = parser.parse_known_args(argv)
    if any(x == '--resume' or x.startswith('--resume=') for x in rest):
        parser.error('GQA experiments require a fresh output directory')
    return args, [*rest, '--ring-policy', args.ring_policy, '--recent-capacity', '512', '--max-new-tokens', '512']


def main(argv=None):
    args, rest = parse_args(argv)
    import run_joint
    import run_active_ring_layout as active
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from gqa_attention_runtime import GQAAttentionBackend, gqa_attention_policy

    old_runtime, old_main, old_save = deployed.DeployedRuntime, run_joint.main, run_joint.save_json
    old_load, old_generate = bridge.load_spec, joint.generate_joint
    loaded, current = {}, {}

    def load(target, *a, **kw):
        result = old_load(target, *a, **kw)
        loaded['target'] = target
        loaded['executor'] = GQAAttentionBackend(args.gqa_backend)
        return result

    class Runtime(old_runtime):
        @contextmanager
        def activate(self, joint_module, scheme):
            with super().activate(joint_module, scheme) as state:
                receipt = None
                try:
                    with gqa_attention_policy(loaded['target'], args.gqa_backend,
                                              executor=loaded['executor']) as receipt:
                        current['receipt'] = receipt
                        yield state
                finally:
                    current.clear()
                    if receipt is not None:
                        state['preparation']['tree_attention'] = receipt

    def generate(*a, **kw):
        result = old_generate(*a, **kw)
        receipt = current['receipt']
        expected = result.forward_counts['target_verify_calls'] * len(loaded['target'].model.language_model.layers)
        if receipt['selected_calls'] != expected or receipt['failures'] or receipt['fallback_calls']:
            raise RuntimeError('GQA did not cover exactly all Target verification attention calls')
        result.precision = dict(result.precision,
            target_sdpa='math_prefill+' + args.gqa_backend + '_tree',
            target_verify_attention_backend=args.gqa_backend,
            target_prefill_attention_backend='math', draft_attention_backend='unchanged')
        return result

    def named_main():
        old = sys.argv[-1]
        prefix = 'memory_shared_embedding_compiled_ring_active_width_'
        if sys.argv[-2] != '--arms' or not old.startswith(prefix) or old not in run_joint.SCHEMES:
            raise RuntimeError('unexpected active ring arm during GQA setup')
        arm = old + '_gqa_' + args.gqa_backend
        run_joint.SCHEMES = {arm: dict(run_joint.SCHEMES[old])}
        run_joint.ARMS = {arm: ('fixed0', 1.0)}
        sys.argv[-1] = arm
        old_main()

    def save(path, value):
        if Path(path).name == 'status.json':
            value = dict(value, tree_attention_backend=args.gqa_backend,
                attention_backend_fallback=False, attention_scope='Target verification only',
                attention_prepare_scope='first-use engine setup/compilation included in request',
                kv_quantized=False, kv_token_coverage='all prompt and generated tokens at capacity512/max512')
        old_save(path, value)

    deployed.DeployedRuntime, run_joint.main, run_joint.save_json = Runtime, named_main, save
    bridge.load_spec, joint.generate_joint = load, generate
    try:
        active.main(rest)
    finally:
        deployed.DeployedRuntime, run_joint.main, run_joint.save_json = old_runtime, old_main, old_save
        bridge.load_spec, joint.generate_joint = old_load, old_generate


if __name__ == '__main__':
    main()
