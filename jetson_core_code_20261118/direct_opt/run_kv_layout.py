"""Measure accepted-tail KV copying with the same long-answer deployment.

No cache token is evicted. DynamicLayer append remains unchanged, so this
isolates compaction savings before testing larger cache-layout changes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--kv-layout",required=True,choices=("native","tail"))
    args,rest=p.parse_known_args(argv)
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument("--out")
    probe.add_argument("--resume",action="store_true")
    options,_=probe.parse_known_args(rest)
    if options.resume:
        try:
            old=json.loads((Path(options.out)/"status.json").read_text())
        except (OSError,ValueError,TypeError) as exc:
            p.error("cannot validate KV resume: "+str(exc))
        if old.get("kv_layout")!=args.kv_layout:
            p.error("resume requires the same kv_layout")
    return args,rest


def main(argv=None):
    args,rest=parse_args(argv)
    import run_joint
    import run_memory_layout as memory
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge
    from tail_kv_compaction import compact_tail

    original_runtime=deployed.DeployedRuntime
    original_main,original_save=run_joint.main,run_joint.save_json

    class KVRuntime(original_runtime):
        @contextmanager
        def activate(self,joint,scheme):
            previous=bridge._compact
            calls=[]

            def compact(cache,selected,base,tree_nodes):
                info=compact_tail(cache,selected,base,tree_nodes)
                # Keep per-round totals, not hundreds of repeated layer shapes.
                calls.append({k:v for k,v in info.items() if k!="layers"})

            with super().activate(joint,scheme) as state:
                if args.kv_layout=="tail":
                    bridge._compact=compact
                try:
                    yield state
                finally:
                    bridge._compact=previous
                    state["preparation"]["kv_layout"]={
                        "method":args.kv_layout,"calls":calls,"instrumented":args.kv_layout=="tail",
                        "call_count":len(calls) if args.kv_layout=="tail" else None,
                        "logical_compaction_copy_bytes":sum(c["logical_copy_total_bytes"] for c in calls) if args.kv_layout=="tail" else None,
                        "reference_full_compaction_copy_bytes":sum(c["reference_full_prefix_copy_total_bytes"] for c in calls) if args.kv_layout=="tail" else None,
                        "validation_and_accounting_timed":True,
                        "dram_measured":False,"tokens_evicted":0,
                        "append_implementation":"original DynamicLayer torch.cat",
                        "scope":"Target K/V compaction payload only; all attention reads and append copies still occur"}

    def save_json(path,value):
        if Path(path).name=="status.json":
            value=dict(value,kv_layout=args.kv_layout,
                       cache_policy="full history retained; accepted-tail compaction only",
                       kv_dram_measured=False)
        original_save(path,value)

    def named_main():
        old=sys.argv[-1]
        if sys.argv[-2]!="--arms" or old not in run_joint.SCHEMES:
            raise RuntimeError("unexpected memory arm")
        arm=old+"_kv_"+args.kv_layout
        run_joint.SCHEMES={arm:dict(run_joint.SCHEMES[old])}
        run_joint.ARMS={arm:("fixed0",1.0)}
        sys.argv[-1]=arm
        original_main()

    deployed.DeployedRuntime=KVRuntime
    run_joint.main,run_joint.save_json=named_main,save_json
    try:
        memory.main(rest)
    finally:
        deployed.DeployedRuntime=original_runtime
        run_joint.main,run_joint.save_json=original_main,original_save


if __name__=="__main__":
    main()
