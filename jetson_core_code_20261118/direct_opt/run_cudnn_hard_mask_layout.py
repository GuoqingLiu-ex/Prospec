"""Independent cuDNN tree attention with one shared FP16 hard mask per round.

Uses full prompt protection + active KV capacity512, shared embeddings, compiled
adaptor and full visual depth. Native prefill/vision/Draft remain unchanged.
No fallback, resume, eviction, automatic experiment start, or preset quality pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

VARIANT='cudnn_hard_fp16_shared_round'


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--gqa-backend',choices=('cudnn',),default='cudnn')
    parser.add_argument('--ring-policy',choices=('protected_prompt',),default='protected_prompt')
    parser.add_argument('--recent-capacity',type=int,choices=(512,),default=512)
    parser.add_argument('--max-new-tokens',type=int,choices=(512,),default=512)
    parser.add_argument('--memory-layout',choices=('shared_embedding',),default='shared_embedding')
    parser.add_argument('--adaptor',choices=('compiled',),default='compiled')
    args,rest=parser.parse_known_args(argv)
    if any(x.split('=',1)[0] in ('--resume','--visual-kv-mode','--capture-dir','--backend') for x in rest):
        parser.error('use a new output directory and the fixed unquantized cuDNN recipe')
    return args,[*rest,'--max-new-tokens','512']


def main(argv=None):
    args,rest=parse_args(argv)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from cudnn_hard_mask_runtime import cudnn_hard_mask_policy
    old_runtime=deployed.DeployedRuntime
    old_main,old_save=run_joint.main,run_joint.save_json
    old_load,old_generate=bridge.load_spec,joint.generate_joint
    loaded,current={},{}

    def load(target,*a,**kw):
        result=old_load(target,*a,**kw);loaded['target']=target;return result

    class Runtime(old_runtime):
        @contextmanager
        def activate(self,joint_module,scheme):
            with super().activate(joint_module,scheme) as state:
                policy=None
                try:
                    with cudnn_hard_mask_policy(loaded['target'],recent_capacity=512,tree_capacity=30) as policy:
                        current['policy']=policy
                        yield state
                finally:
                    current.clear()
                    if policy is not None:
                        info=policy.summary()
                        if info.get('attention_variant')!=VARIANT:raise RuntimeError('hard-mask provenance mismatch')
                        state['preparation']['ring_kv']=info
                        state['preparation']['ring_variant']='active_width'
                        state['preparation']['tree_attention_variant']=VARIANT
                        state['memory_phase_peaks']=info['memory_phase_peaks']

    def generate(*a,**kw):
        result=old_generate(*a,**kw)
        policy=current['policy'];verifies=result.forward_counts['target_verify_calls']
        expected=verifies*len(policy.attention_indices)
        if (policy.selected_calls!=expected or policy.completed_forwards!=verifies or
                policy.mask_preparations!=verifies or policy.mask_clears!=verifies or
                len(policy.rounds)!=verifies or policy.pending is not None or policy.prepared_mask is not None or
                policy.failed or policy.attention_failures or
                policy.prefill_calls!=result.forward_counts['target_prefill_calls']):
            raise RuntimeError('cuDNN/shared-mask counts do not cover exactly all Target verification calls')
        result.precision=dict(result.precision,target_sdpa='math_prefill+cudnn_tree_hard_fp16_mask',
            target_verify_attention_backend='cudnn',target_prefill_attention_backend='math',
            target_verify_mask_dtype='torch.float16',target_verify_mask_values='0,-inf',
            target_verify_mask_scope='one shared Tensor per verification forward',
            target_kv_storage='fp16',draft_attention_backend='unchanged')
        return result

    def named_main():
        old=sys.argv[-1]
        if sys.argv[-2]!='--arms' or old!='memory_shared_embedding_compiled' or old not in run_joint.SCHEMES:
            raise RuntimeError('unexpected deployment arm during hard-mask setup')
        arm=old+'_ring_active_width_protected_prompt_512_cudnn_hard_fp16_sharedmask'
        run_joint.SCHEMES={arm:dict(run_joint.SCHEMES[old])};run_joint.ARMS={arm:('fixed0',1.0)}
        sys.argv[-1]=arm;old_main()

    def save(path,value):
        if Path(path).name=='status.json':
            value=dict(value,ring_variant='active_width',ring_policy='protected_prompt',recent_capacity=512,
                tree_attention_variant=VARIANT,tree_attention_backend='cudnn',attention_backend_fallback=False,
                target_history_eviction=False,draft_history_eviction=False,kv_quantized=False,
                attention_scope='Target verification only; original math prefill, vision and Draft unchanged',
                mask_scope='hard FP16 constructed once per tree and shared across Target layers; all checks/preparation in request',
                quality_claim='Experimental fused-arithmetic/hard-mask candidate; no long-answer quality or speed pass claimed')
        old_save(path,value)

    deployed.DeployedRuntime=Runtime;bridge.load_spec=load;joint.generate_joint=generate
    run_joint.main,run_joint.save_json=named_main,save
    try:memory.main(['--memory-layout','shared_embedding','--adaptor','compiled',*rest])
    finally:
        deployed.DeployedRuntime=old_runtime;bridge.load_spec=old_load;joint.generate_joint=old_generate
        run_joint.main,run_joint.save_json=old_main,old_save


if __name__=='__main__':main()
