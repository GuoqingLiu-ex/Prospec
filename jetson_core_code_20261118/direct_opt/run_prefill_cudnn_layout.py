"""Same cuDNN language-prefill backend across three independent KV deployments.

--kv-mode full keeps native Target KV + math tree attention; fp16_visual and
int8 use the same mixed tree kernel, with visual precision as their sole change.
All use shared embeddings, compiled adaptor, full visual depth, max <=512 tokens.
No experiment starts on import. Use one mode per fresh output/process.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--kv-mode',required=True,choices=('full','fp16_visual','int8'))
    p.add_argument('--memory-layout',choices=('shared_embedding',),default='shared_embedding')
    p.add_argument('--adaptor',choices=('compiled',),default='compiled')
    p.add_argument('--max-new-tokens',type=int,default=512)
    args,rest=p.parse_known_args(argv)
    if not 1<=args.max_new_tokens<=512:p.error('max-new-tokens must be 1..512; no eviction')
    if any(x.split('=',1)[0] in ('--resume','--visual-kv-mode','--gqa-backend','--ring-policy','--recent-capacity','--capture-dir') for x in rest):
        p.error('one fixed prefill/KV recipe per fresh result directory')
    return args,[*rest,'--max-new-tokens',str(args.max_new_tokens)]


def main(argv=None):
    args,rest=parse_args(argv)
    root=str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:sys.path.insert(0,root)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    import vispec_joint_adapter as joint
    from prefill_cudnn_runtime import prefill_cudnn_policy
    from ring_cache_runtime import ring_cache_policy
    from direct_opt.visual_kv_runtime import visual_kv_policy
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
                prefill=kv=None
                try:
                    with prefill_cudnn_policy(loaded['target']) as prefill:
                        context=(ring_cache_policy(loaded['target'],'full',recent_capacity=512,tree_capacity=30)
                            if args.kv_mode=='full' else visual_kv_policy(loaded['target'],args.kv_mode,
                                max_generated=512,tree_capacity=30))
                        with context as kv:
                            current.update(prefill=prefill,kv=kv)
                            yield state
                finally:
                    current.clear()
                    if prefill is not None:state['preparation']['language_prefill_attention']=prefill
                    if kv is not None:
                        info=kv.summary()
                        state['preparation']['ring_kv' if args.kv_mode=='full' else 'visual_kv']=info
                        state['preparation']['prefill_kv_mode']=args.kv_mode
                        state['memory_phase_peaks']=info['memory_phase_peaks']

    def generate(*a,**kw):
        result=old_generate(*a,**kw)
        prefill,kv=current['prefill'],current['kv']
        if (prefill['active'] or prefill['prefill_calls']!=1 or prefill['target_forward_calls']!=1 or
                prefill['selected_calls']!=28 or prefill['completed_calls']!=28 or
                prefill['layer_calls']!=[1]*28 or prefill['failures'] or prefill['fallback_calls'] or
                result.forward_counts['target_prefill_calls']!=1):
            raise RuntimeError('cuDNN prefill did not cover exactly one native prefill and all 28 language layers')
        verifies=result.forward_counts['target_verify_calls']
        if kv.failed or kv.pending is not None or len(kv.rounds)!=verifies:
            raise RuntimeError('KV runtime does not match completed verification count')
        if args.kv_mode!='full' and (kv.mixed_attention_calls!=28*verifies or
                kv.completed_target_forwards!=verifies):
            raise RuntimeError('mixed tree kernel must cover every Target verification layer')
        tree='math' if args.kv_mode=='full' else 'triton_mixed'
        result.precision=dict(result.precision,target_sdpa='cudnn_prefill+'+tree+'_tree',
            target_prefill_attention_backend='cudnn',target_prefill_mask='None, causal=True',
            target_prefill_gqa='28Q/4KV, enable_gqa=True',target_verify_attention_backend=tree,
            target_visual_kv_storage='int8_group32_fp32scale' if args.kv_mode=='int8' else 'fp16',
            draft_attention_backend='unchanged')
        return result

    def named_main():
        old=sys.argv[-1]
        if sys.argv[-2:]!=['--arms','memory_shared_embedding_compiled'] or old not in run_joint.SCHEMES:
            raise RuntimeError('unexpected deployment arm for cuDNN prefill')
        arm=old+'_cudnn_prefill_kv_'+args.kv_mode
        run_joint.SCHEMES={arm:dict(run_joint.SCHEMES[old])};run_joint.ARMS={arm:('fixed0',1.0)}
        sys.argv[-1]=arm;old_main()

    def save(path,value):
        if Path(path).name=='status.json':
            value=dict(value,prefill_attention_variant='causal_cudnn_gqa',prefill_kv_mode=args.kv_mode,
                target_prefill_attention_backend='cudnn',target_prefill_scope='empty-cache complete-prompt language layers only',
                target_verify_attention_backend='math' if args.kv_mode=='full' else 'triton_mixed',
                target_history_eviction=False,draft_history_eviction=False,attention_backend_fallback=False,
                visual_quantized=args.kv_mode=='int8',
                quality_claim='New causal prefill backend; actual-output quality and full-request memory must be checked')
        old_save(path,value)

    deployed.DeployedRuntime=Runtime;bridge.load_spec=load;joint.generate_joint=generate
    run_joint.main,run_joint.save_json=named_main,save
    try:memory.main(['--memory-layout','shared_embedding','--adaptor','compiled',*rest])
    finally:
        deployed.DeployedRuntime=old_runtime;bridge.load_spec=old_load;joint.generate_joint=old_generate
        run_joint.main,run_joint.save_json=old_main,old_save


if __name__=='__main__':main()
