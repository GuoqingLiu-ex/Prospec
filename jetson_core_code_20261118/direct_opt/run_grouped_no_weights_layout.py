"""Full-request control/candidate for removing unused FP16 attention weights.

Contract: grouped delegates directly to the frozen run_grouped_gqa_layout entry.
grouped_no_discarded_weights changes only Target tree attention's unused FP16
probability cast. Both retain shared embedding, compiled visual adaptor, full
visual computation/tokens, native full DynamicCache and the original generation
algorithm, prompts, EOS and 512-token limit. No ring, quantization or depth mode.
Fresh independent output directory required by the parent runner; no resume.

All request policy/counter/first-receipt costs remain in the original request
scope. Removed-copy bytes are logical payload, not measured DRAM or kernel
counts. Actual output IDs, full rounds and hardware observations come unchanged
from run_joint and must be checked after execution. Import/help run no GPU work.

Example (root schedules execution):
 python run_grouped_no_weights_layout.py --tree-attention grouped_no_discarded_weights \
   --requests REQUESTS --request-id ID --out FRESH --physical-gpu 6 --warmup 1 --repeats 2
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack,contextmanager
from pathlib import Path
import sys

METHOD='grouped_no_discarded_weights'
_ACTIVE=False


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--tree-attention',choices=('grouped',METHOD),default=METHOD)
    parser.add_argument('--max-new-tokens',type=int,choices=(512,),default=512)
    args,rest=parser.parse_known_args(argv)
    forbidden=('--resume','--arms','--include-ar','--adaptor','--memory-layout','--deployment-adaptor',
        '--capture-visual-dir','--collect-local','--difference-threshold','--fusion-mode',
        '--paired-baseline','--ring-policy','--recent-capacity','--commit-backend','--visual-kv-mode',
        '--gqa-backend','--heads-per-group','--token-tile','--thresholds','--anchor-stride',
        '--skip-layers','--prefix-layer','--local-thresholds')
    if any(token.startswith('--') and any(flag.startswith(token.split('=',1)[0]) for flag in forbidden) for token in rest):
        parser.error('fresh single grouped control/candidate; shared/compiled/full visual/full native KV recipe is fixed')
    return args,[*rest,'--max-new-tokens','512']


def arm_name(mode):
    if mode not in ('grouped',METHOD):raise ValueError('unsupported tree attention mode')
    return 'memory_shared_embedding_compiled_fullkv_tree_'+mode+'_math'


def _candidate_main(args,rest):
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from grouped_gqa_no_weights_runtime import grouped_no_weights_attention_policy,math_precision_scope,precision_flags

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
            if (scheme.get('tree_attention')!=METHOD or scheme.get('depth')!='fixed0'
                    or scheme.get('adaptor')!='compiled' or scheme.get('ratio')!=1.0
                    or scheme.get('compression') is not None
                    or scheme.get('target_cache_policy')!='native_full_dynamic_cache'):
                raise ValueError('tree attention request differs from fixed full-visual/native-KV recipe')
            if 'target' not in loaded:raise RuntimeError('target must be loaded before policy activation')
            with super().activate(joint_module,scheme) as state:
                policy=None
                try:
                    with grouped_no_weights_attention_policy(loaded['target'],METHOD) as policy:
                        current['policy']=policy
                        yield state
                finally:
                    current.clear()
                    if policy is not None:
                        state['preparation']['tree_attention']=policy.summary()
                        state['preparation']['tree_attention_mode']=METHOD

    def generate(*a,**kw):
        if 'policy' not in current:raise RuntimeError('generation requires active tree attention policy')
        result=old_generate(*a,**kw)
        policy=current['policy'];policy.validate_result(result)
        result.precision=dict(result.precision,
            target_sdpa='math_prefill+grouped_no_discarded_weights_fp32_math_tree',
            target_prefill_attention_backend='math',
            target_verify_attention_backend='grouped_two_bmm_no_discarded_weights_fp32_math',
            target_tree_attention_mode=METHOD,target_tree_math_precision=precision_flags(),
            target_cache_policy='native_full_dynamic_cache',draft_attention_backend='unchanged',
            tree_attention_discarded_probability_cast_removed=True,
            tree_attention_bitwise_equivalence_claim=False)
        result.forward_counts.update(target_tree_attention_policy_calls=policy.record['completed_calls'],
            target_grouped_no_discarded_weights_attention_calls=policy.record['candidate_calls'])
        return result

    def named_main():
        old='memory_shared_embedding_compiled'
        if sys.argv[-2:]!=['--arms',old]:raise RuntimeError('unexpected tree attention parent arm')
        run_joint.SCHEMES={arm:dict(run_joint.SCHEMES[old],tree_attention=METHOD,
            target_cache_policy='native_full_dynamic_cache')}
        run_joint.ARMS={arm:('fixed0',1.0)};sys.argv[-1]=arm
        old_main()

    def save(path,value):
        if Path(path).name=='status.json':
            value=dict(value,tree_attention_mode=METHOD,target_cache_policy='native_full_dynamic_cache',
                new_kv_policy_installed=False,target_history_eviction=False,kv_quantized=False,
                tree_attention_scope='Target verification only; prefill/vision/Draft/cache ops unchanged',
                tree_attention_precision=precision_flags(),attention_backend_fallback=False,
                primitive_receipt_computation_in_request='first tree layer only',
                ordinary_primitive_record_receipt=False,maximum_representative_receipts_per_request=1,
                query_pack_logical_bytes_counted=True,discarded_probability_cast_removed=True,
                removed_probability_copy_scope='logical payload only; not measured kernels or DRAM',
                methodology='same_shared_compiled_fullkv_grouped_vs_grouped_no_discarded_weights',
                quality_claim='Output-only execution candidate; actual IDs and full rounds require checking',
                output_equivalence_pass=None,quality_pass=None,performance_pass=None,actual_dram_measured=False)
        old_save(path,value)

    # Install after all closures exist; register each restore before its mutation.
    with ExitStack() as stack:
        for owner,name,value in ((deployed,'DeployedRuntime',Runtime),(bridge,'load_spec',load),
                (joint,'generate_joint',generate),(run_joint,'main',named_main),(run_joint,'save_json',save)):
            stack.callback(setattr,owner,name,getattr(owner,name));setattr(owner,name,value)
        with math_precision_scope():
            return memory.main(['--memory-layout','shared_embedding','--adaptor','compiled',*rest])


def main(argv=None):
    args,rest=parse_args(argv)
    global _ACTIVE
    if _ACTIVE:raise RuntimeError('nested no-weights runner invocation')
    _ACTIVE=True
    try:
        if args.tree_attention=='grouped':
            from run_grouped_gqa_layout import main as grouped_main
            return grouped_main(['--tree-attention','grouped',*rest])
        return _candidate_main(args,rest)
    finally:
        _ACTIVE=False


if __name__=='__main__':main()
