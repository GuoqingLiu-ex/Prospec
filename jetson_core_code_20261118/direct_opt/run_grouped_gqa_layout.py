"""Full Target+Draft requests with an isolated tree-attention execution choice.

Both modes use shared embedding, compiled adaptor, full visual depth/tokens and
the native full DynamicCache. Only Target verification attention differs.
Default candidate is grouped; native is the same-layout math control. Fresh
output required; no GPU/model work runs on import.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--tree-attention',choices=('native','grouped'),default='grouped')
    p.add_argument('--max-new-tokens',type=int,choices=(512,),default=512)
    a,rest=p.parse_known_args(argv)
    forbidden=('--resume','--arms','--include-ar','--adaptor','--memory-layout','--deployment-adaptor',
        '--capture-visual-dir','--collect-local','--difference-threshold','--fusion-mode',
        '--paired-baseline','--ring-policy','--recent-capacity','--commit-backend','--visual-kv-mode',
        '--gqa-backend','--heads-per-group','--token-tile')
    if any(token.startswith('--') and any(flag.startswith(token.split('=',1)[0]) for flag in forbidden) for token in rest):
        p.error('fresh single native/grouped tree arm; shared/compiled/full DynamicCache recipe is fixed')
    return a,[*rest,'--max-new-tokens','512']


def arm_name(mode):return 'memory_shared_embedding_compiled_fullkv_tree_'+mode+'_math'


def main(argv=None):
    args,rest=parse_args(argv)
    if args.tree_attention=='native':
        # Same frozen entry point used for the existing native full-cache control.
        from run_stride0_gqa_layout import main as native_main
        return native_main(['--tree-attention','native',*rest])
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from grouped_gqa_runtime import grouped_attention_policy,math_precision_scope,precision_flags

    old_runtime=deployed.DeployedRuntime
    old_main,old_save=run_joint.main,run_joint.save_json
    old_load,old_generate=bridge.load_spec,joint.generate_joint
    loaded,current={},{}
    arm=arm_name(args.tree_attention)

    def load(target,*a,**kw):
        spec=old_load(target,*a,**kw);loaded['target']=target;return spec

    class Runtime(old_runtime):
        @contextmanager
        def activate(self,joint_module,scheme):
            if current:raise RuntimeError('nested tree attention request activation')
            if scheme.get('tree_attention')!=args.tree_attention:
                raise ValueError('tree attention arm differs from selected mode')
            with super().activate(joint_module,scheme) as state:
                policy=None
                try:
                    with grouped_attention_policy(loaded['target'],args.tree_attention) as policy:
                        current['policy']=policy
                        yield state
                finally:
                    current.clear()
                    if policy is not None:
                        state['preparation']['tree_attention']=policy.summary()
                        state['preparation']['tree_attention_mode']=args.tree_attention

    def generate(*a,**kw):
        if 'policy' not in current:raise RuntimeError('generation requires active tree attention policy')
        result=old_generate(*a,**kw)
        policy=current['policy'];policy.validate_result(result)
        result.precision=dict(result.precision,
            target_sdpa=('math' if args.tree_attention=='native' else 'math_prefill+grouped_fp32_math_tree'),
            target_prefill_attention_backend='math',
            target_verify_attention_backend=('math' if args.tree_attention=='native' else 'grouped_two_bmm_fp32_math'),
            target_tree_attention_mode=args.tree_attention,target_tree_math_precision=precision_flags(),
            target_cache_policy='native_full_dynamic_cache',draft_attention_backend='unchanged',
            tree_attention_bitwise_equivalence_claim=False)
        result.forward_counts.update(target_tree_attention_policy_calls=policy.record['completed_calls'],
                                     target_grouped_attention_calls=policy.record['candidate_calls'])
        return result

    def named_main():
        old='memory_shared_embedding_compiled'
        if sys.argv[-2:]!=['--arms',old]:raise RuntimeError('unexpected tree attention parent arm')
        run_joint.SCHEMES={arm:dict(run_joint.SCHEMES[old],tree_attention=args.tree_attention,
                                  target_cache_policy='native_full_dynamic_cache')}
        run_joint.ARMS={arm:('fixed0',1.0)};sys.argv[-1]=arm
        old_main()

    def save(path,value):
        if Path(path).name=='status.json':
            value=dict(value,tree_attention_mode=args.tree_attention,target_cache_policy='native_full_dynamic_cache',
                new_kv_policy_installed=False,target_history_eviction=False,kv_quantized=False,
                tree_attention_scope='Target verification only; prefill/vision/Draft/cache ops unchanged',
                tree_attention_precision=precision_flags(),attention_backend_fallback=False,
                primitive_receipt_computation_in_request='first tree layer only',
                ordinary_primitive_record_receipt=False,maximum_representative_receipts_per_request=1,
                query_pack_logical_bytes_counted=True,
                methodology='same_shared_compiled_fullkv_native_vs_grouped_tree_math',
                quality_claim='Numerical execution candidate; actual output and full rounds require checking',
                output_equivalence_pass=None,quality_pass=None,performance_pass=None,actual_dram_measured=False)
        old_save(path,value)

    deployed.DeployedRuntime=Runtime
    bridge.load_spec,joint.generate_joint=load,generate
    run_joint.main,run_joint.save_json=named_main,save
    try:
        with math_precision_scope():
            memory.main(['--memory-layout','shared_embedding','--adaptor','compiled',*rest])
    finally:
        deployed.DeployedRuntime=old_runtime
        bridge.load_spec,joint.generate_joint=old_load,old_generate
        run_joint.main,run_joint.save_json=old_main,old_save


if __name__=='__main__':main()
