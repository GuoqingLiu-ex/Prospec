"""Single-arm full-request streaming adaptor candidate, with native full KV.

Shared embedding and compiled setup are reused, then only Draft.imadpt is
replaced. All visual tokens/layers, Draft/Target decoding and ordinary HF
DynamicCache remain unchanged. FP32 online pooling changes numerical rounding.
Use fresh output; no model, CUDA context or experiment starts on import.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
from pathlib import Path
import sys
import time
import weakref

CONFIGS = ((7,12),(14,24),(28,48))
METHOD = 'streaming_online_softmax_fp32'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--heads-per-group',type=int,choices=(7,14,28),default=28)
    p.add_argument('--token-tile',type=int,choices=(12,24,48),default=48)
    p.add_argument('--max-new-tokens',type=int,choices=(512,),default=512)
    args,rest=p.parse_known_args(argv)
    if (args.heads_per_group,args.token_tile) not in CONFIGS:
        p.error('use one fixed G/B pair: 7/12, 14/24 or 28/48')
    forbidden=('--resume','--arms','--include-ar','--memory-layout','--adaptor','--deployment-adaptor',
        '--paired-baseline','--capture-visual-dir','--collect-local','--difference-threshold',
        '--fusion-mode','--ring-policy','--recent-capacity','--commit-backend','--visual-kv-mode','--gqa-backend')
    if any(token.startswith('--') and any(flag.startswith(token.split('=',1)[0])
                                          for flag in forbidden) for token in rest):
        p.error('fresh single streaming deployment only; full visual input and native DynamicCache are fixed')
    return args,[*rest,'--max-new-tokens','512']


def arm_name(group,tile):
    return f'memory_shared_embedding_streaming_g{group}_t{tile}_fp32_fullkv'


class AdaptorCalls:
    """Scalar/shape-only hook receipts; no Tensor or module references retained."""
    def __init__(self):
        self.active=False
        self.counts=None

    def begin(self,module):
        if self.active:
            raise RuntimeError('nested streaming requests are unsupported')
        if module.check_finite or module.record_workspace:
            raise RuntimeError('request runner requires finite/workspace diagnostics disabled')
        module.last_workspace_receipt=None
        self.counts=dict(started_calls=0,completed_calls=0,input_lengths=[],input_shapes=[],
                         input_tokens=0,receipts=[])
        self.active=True

    def before(self,module,inputs):
        if not self.active:
            raise RuntimeError('streaming adaptor called outside a request')
        if len(inputs)!=1 or inputs[0].ndim!=3 or inputs[0].shape[0]!=1:
            raise ValueError('expected one batch-one visual feature Tensor')
        shape=list(inputs[0].shape)
        self.counts['started_calls']+=1
        self.counts['input_shapes'].append(shape)
        self.counts['input_lengths'].append(shape[1])
        self.counts['input_tokens']+=shape[1]

    def after(self,module,inputs,output):
        info=module.last_workspace_receipt
        if not self.active or info is None or not info['completed']:
            raise RuntimeError('streaming call has no completed functional receipt')
        if list(output.shape)!=[1,module.num_q,module.output_dim]:
            raise RuntimeError('streaming output shape differs from native adaptor interface')
        self.counts['completed_calls']+=1
        self.counts['receipts'].append(copy.deepcopy(info))

    def validate_result(self,result):
        c=self.counts
        if not self.active or c['started_calls']!=c['completed_calls']:
            raise RuntimeError('streaming call coverage incomplete')
        counts=result.forward_counts
        visual=counts.get('merge_visual_tokens_in',0)
        if visual!=counts.get('merge_visual_tokens_out',0):
            raise RuntimeError('streaming runner must preserve all visual tokens')
        expected=int(counts['draft_tree_calls']>0 and visual>0)
        if c['completed_calls']!=expected or c['input_lengths']!=([visual] if expected else []):
            raise RuntimeError('actual streaming calls/lengths differ from first Draft visual context')
        # Existing imadpt_calls counts input overrides and is zero at m=1.
        # Keep it intact; add separately named actual observations.
        counts.update(streaming_imadpt_actual_calls=c['completed_calls'],
                      streaming_imadpt_actual_input_tokens=c['input_tokens'])
        result.imadpt.update(actual_calls=c['completed_calls'],actual_input_lengths=list(c['input_lengths']),
                             actual_count_source='streaming_module_hooks')
        result.precision.update(draft_visual_adaptor=METHOD,draft_visual_pool_dtype='float32',
                                draft_visual_projection_dtype='float16',
                                visual_adaptor_bitwise_equivalence_claim=False)

    def end(self,error=None):
        self.active=False
        info=copy.deepcopy(self.counts)
        info.update(failed=error is not None,request_hooks_and_receipt_bookkeeping_in_e2e=True,
            dense_input_finite_check=False,live_workspace_snapshots=False,
            module_shape_and_basic_storage_receipts_still_computed=True,
            target_cache_policy='native_full_dynamic_cache',new_kv_policy_installed=False)
        return info


def install_streaming(draft,*,heads_per_group,token_tile):
    """Permanent replacement after the parent's verified compiled installation."""
    import torch
    from compiled_adaptor import CompiledQueryAdaptor
    from streaming_adaptor import StreamingQueryAdaptor
    source=draft.imadpt
    if not isinstance(source,CompiledQueryAdaptor) or source.input_dtype!=torch.float16:
        raise ValueError('requires the parent compiled FP16 adaptor')
    gpu=source.input_device.type=='cuda'
    if gpu: torch.cuda.synchronize()
    before=torch.cuda.memory_allocated() if gpu else None
    start=time.perf_counter()
    ref=weakref.ref(source)
    stream=StreamingQueryAdaptor.from_compiled(source,heads_per_group=heads_per_group,
        token_tile=token_tile,accumulate_dtype=torch.float32,check_finite=False,record_workspace=False)
    draft.imadpt=stream
    del source
    gc.collect()
    if gpu: torch.cuda.synchronize()
    if ref() is not None:
        raise RuntimeError('compiled source module still retained after streaming installation')
    calls=AdaptorCalls()
    handles=(stream.register_forward_pre_hook(calls.before),stream.register_forward_hook(calls.after))
    metadata=dict(module=stream.metadata(),compiled_source_module_released=True,
        allow_tf32_at_installation=bool(torch.backends.cuda.matmul.allow_tf32),
        synchronized_install_wall_ms=1000*(time.perf_counter()-start),
        allocated_before_streaming_install_bytes=before,
        allocated_after_streaming_install_bytes=torch.cuda.memory_allocated() if gpu else None,
        load_and_install_peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else None,
        load_and_install_peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu else None,
        setup_scope='after native load and compiled folding; buffer aliases and hooks installed once, no startup saving claim',
        numerical_change='FP32 score/online accumulation then FP16 pooled values and original FP16 V/O',
        full_visual_token_coverage=True,actual_sram_residency=False,quality_pass=None,performance_pass=None)
    return calls,handles,metadata


def main(argv=None):
    args,rest=parse_args(argv)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import vispec_joint_adapter as joint
    import torch

    original_runtime=deployed.DeployedRuntime
    original_main,original_save=run_joint.main,run_joint.save_json
    original_generate=joint.generate_joint
    previous_tf32=torch.backends.cuda.matmul.allow_tf32
    installation,current={},{}
    arm=arm_name(args.heads_per_group,args.token_tile)

    class Runtime(original_runtime):
        def __init__(self,draft,**kwargs):
            super().__init__(draft,**kwargs)
            self.streaming_calls,self.streaming_handles,info=install_streaming(draft,
                heads_per_group=args.heads_per_group,token_tile=args.token_tile)
            self.streaming_module=draft.imadpt
            self.metadata.update(mode=METHOD,base_compiled_installation_preserved=True,streaming_installation=info)
            kwargs['receipt'].update(self.metadata)
            installation.update(info)

        @contextmanager
        def activate(self,joint_module,scheme):
            if current:
                raise RuntimeError('nested streaming runtime activation')
            if self.draft.imadpt is not self.streaming_module:
                raise RuntimeError('installed streaming adaptor was replaced')
            if torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError('streaming FP32 pooling requires allow_tf32=False')
            with super().activate(joint_module,scheme) as state:
                state['preparation']['streaming_allow_tf32']=False
                self.streaming_calls.begin(self.streaming_module)
                current['calls']=self.streaming_calls
                error=None
                try:
                    yield state
                except BaseException as exc:
                    error=exc
                    raise
                finally:
                    current.clear()
                    state['preparation']['streaming_request']=self.streaming_calls.end(error)
                    state['preparation']['streaming_method']=METHOD
                    state['adaptor']=METHOD
                    self.streaming_module.last_workspace_receipt=None

    def generate(*a,**kw):
        if 'calls' not in current:
            raise RuntimeError('generation requires active streaming deployment')
        result=original_generate(*a,**kw)
        current['calls'].validate_result(result)
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError('allow_tf32 changed during streaming generation')
        result.precision['streaming_allow_tf32']=False
        return result

    def named_main():
        old='memory_shared_embedding_compiled'
        if sys.argv[-2:]!=['--arms',old]:
            raise RuntimeError('unexpected streaming parent arm')
        scheme=dict(run_joint.SCHEMES[old],adaptor=METHOD,streaming_heads_per_group=args.heads_per_group,
                    streaming_token_tile=args.token_tile)
        run_joint.SCHEMES={arm:scheme};run_joint.ARMS={arm:('fixed0',1.0)}
        sys.argv[-1]=arm
        original_main()

    def save(path,value):
        if Path(path).name=='status.json':
            value=dict(value,deployment_adaptor=METHOD,base_deployment_adaptor='compiled',
                streaming_method=METHOD,streaming_heads_per_group=args.heads_per_group,
                streaming_token_tile=args.token_tile,streaming_installation=dict(installation),
                streaming_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                target_cache_policy='native_full_dynamic_cache',new_kv_policy_installed=False,
                methodology='single_streaming_adaptor_full_request_candidate',
                quality_claim='Numerical reassociation candidate; output/round equivalence and facts require measurement',
                output_equivalence_pass=None,quality_pass=None,performance_pass=None,
                actual_sram_residency=False)
        return original_save(path,value)

    deployed.DeployedRuntime=Runtime
    run_joint.main,run_joint.save_json=named_main,save
    joint.generate_joint=generate
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        memory.main(['--memory-layout','shared_embedding','--adaptor','compiled',*rest])
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous_tf32
        deployed.DeployedRuntime=original_runtime
        run_joint.main,run_joint.save_json=original_main,original_save
        joint.generate_joint=original_generate


if __name__=='__main__':
    main()
