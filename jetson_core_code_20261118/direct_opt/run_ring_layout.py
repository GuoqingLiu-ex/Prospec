"""Compare full Target KV with protected-prompt and recent-only ring caches.

The Draft retains full history. Target eviction changes the model context and
requires factual quality review. All modes share request/stage instrumentation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--ring-policy",required=True,choices=("full","protected_prompt","pure_recent"))
    p.add_argument("--recent-capacity",type=int,default=128)
    args,rest=p.parse_known_args(argv)
    if args.recent_capacity<1:
        p.error("recent capacity must be positive")
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument("--out")
    probe.add_argument("--resume",action="store_true")
    options,_=probe.parse_known_args(rest)
    if options.resume:
        try:
            old=json.loads((Path(options.out)/"status.json").read_text())
        except (OSError,ValueError,TypeError) as exc:
            p.error("cannot validate ring resume: "+str(exc))
        if old.get("ring_policy")!=args.ring_policy or old.get("recent_capacity")!=args.recent_capacity:
            p.error("resume requires the same ring policy and capacity")
    return args,rest


def main(argv=None):
    args,rest=parse_args(argv)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    from ring_cache_runtime import ring_cache_policy

    original_runtime=deployed.DeployedRuntime
    original_main,original_save=run_joint.main,run_joint.save_json
    original_load=bridge.load_spec
    loaded={}

    def load_spec(target,*positional,**kwargs):
        result=original_load(target,*positional,**kwargs)
        loaded["target"]=target
        return result

    class RingRuntime(original_runtime):
        @contextmanager
        def activate(self,joint,scheme):
            with super().activate(joint,scheme) as state:
                policy=None
                try:
                    with ring_cache_policy(loaded["target"],mode=args.ring_policy,
                                           recent_capacity=args.recent_capacity,tree_capacity=30) as policy:
                        yield state
                finally:
                    if policy is not None:
                        info=policy.summary()
                        state["preparation"]["ring_kv"]=info
                        state["memory_phase_peaks"]=info["memory_phase_peaks"]

    def save_json(path,value):
        if Path(path).name=="status.json":
            value=dict(value,ring_policy=args.ring_policy,recent_capacity=args.recent_capacity,
                       ring_tree_capacity=30,
                       target_history_eviction=args.ring_policy!="full",
                       draft_history_eviction=False,
                       quality_claim=("Native full-history Target cache reference" if args.ring_policy=="full" else
                                      "Target ring eviction is lossy; long-answer factual quality requires review"),
                       kv_dram_measured=False)
        original_save(path,value)

    def named_main():
        old=sys.argv[-1]
        if sys.argv[-2]!="--arms" or old not in run_joint.SCHEMES:
            raise RuntimeError("unexpected memory arm")
        arm=old+"_ring_"+args.ring_policy+"_"+str(args.recent_capacity)
        run_joint.SCHEMES={arm:dict(run_joint.SCHEMES[old])}
        run_joint.ARMS={arm:("fixed0",1.0)}
        sys.argv[-1]=arm
        original_main()

    deployed.DeployedRuntime=RingRuntime
    bridge.load_spec=load_spec
    run_joint.main,run_joint.save_json=named_main,save_json
    try:
        memory.main(rest)
    finally:
        deployed.DeployedRuntime=original_runtime
        bridge.load_spec=original_load
        run_joint.main,run_joint.save_json=original_main,original_save


if __name__=="__main__":
    main()
