"""Single-process deployment of an exact input-embedding storage layout.

The frozen native loader runs first. Sharing is installed once, after exact
weight verification, before requests. Startup peak includes the duplicate;
request peaks measure the deployed layout. No weights are offloaded or cast.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--memory-layout", required=True, choices=("native", "shared_embedding"))
    parser.add_argument("--adaptor", default="native", choices=("native", "compiled"))
    args, rest = parser.parse_known_args(argv)
    forbidden = ("--arms", "--include-ar", "--capture-visual-dir", "--collect-local",
                 "--deployment-adaptor", "--fusion-mode", "--difference-threshold")
    if any(token.startswith("--") and any(flag.startswith(token.split("=", 1)[0])
           for flag in forbidden) for token in rest):
        parser.error("memory layout runner controls arms and requires full visual depth/tokens")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--resume", action="store_true")
    probe.add_argument("--out")
    resume, _ = probe.parse_known_args(rest)
    if resume.resume:
        if resume.out is None:
            parser.error("resume requires --out")
        try:
            previous = json.loads((Path(resume.out) / "status.json").read_text())
        except (OSError, ValueError) as exc:
            parser.error("cannot validate resume status: " + str(exc))
        if (previous.get("memory_layout") != args.memory_layout or
                previous.get("deployment_adaptor") != args.adaptor):
            parser.error("resume requires the same memory layout and adaptor")
    return args, rest


def main(argv=None):
    args, rest = parse_args(argv)
    import run_joint
    import run_deployed_adaptor as deployed
    import hf517_vispec_adapter as bridge

    original_load = bridge.load_spec
    original_runtime = deployed.DeployedRuntime
    original_save, original_argv = run_joint.save_json, sys.argv
    saved_runner = {key: getattr(run_joint, key) for key in
                    ("SCHEMES", "ARMS", "SchemeRuntime", "main")}
    receipt = {}

    def load_spec(target, *positional, **kwargs):
        import torch
        from shared_embedding import install_shared_embedding, unique_model_storage_bytes
        spec = original_load(target, *positional, **kwargs)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        storage_before = unique_model_storage_bytes(target, spec.spec_layer)
        start = time.perf_counter()
        info = {"mode": args.memory_layout, "installed": False}
        if args.memory_layout == "shared_embedding":
            info.update(install_shared_embedding(target, spec.spec_layer))
            info["installed"] = True
        gc.collect()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        info.update(mode=args.memory_layout,
                    synchronized_install_wall_ms=(time.perf_counter()-start)*1000,
                    allocated_before_bytes=before, allocated_after_bytes=after,
                    allocated_reduction_bytes=before-after,
                    unique_model_storage_before_bytes=storage_before,
                    unique_model_storage_after_bytes=unique_model_storage_bytes(target, spec.spec_layer),
                    reserved_after_bytes=torch.cuda.memory_reserved(),
                    load_and_install_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    load_and_install_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    startup_scope="Original duplicate weights loaded before sharing; no startup peak saving claimed",
                    request_extra_weight_copy_bytes=0,
                    request_embedding_lookup_count_changed=False,
                    request_matrix_mac_changed=False,
                    dram_traffic_measured=False)
        receipt.update(info)
        return spec

    class MemoryRuntime(original_runtime):
        def __init__(self, draft, **kwargs):
            super().__init__(draft, **kwargs)
            self.metadata.update(memory_layout=args.memory_layout,
                                 memory_installation=dict(receipt))
            kwargs["receipt"].update(self.metadata)

    def save_json(path, value):
        if Path(path).name == "status.json":
            value = dict(value, memory_layout=args.memory_layout,
                         memory_installation=dict(receipt),
                         methodology="independent_deployment_exact_embedding_layout",
                         adaptive_claim=None,
                         quality_claim="All visual layers and tokens retained; actual long outputs and tree records checked separately",
                         timing_claim=False)
        original_save(path, value)

    def run_named_arm():
        original_arm = "deployment_" + args.adaptor
        if sys.argv[-2:] != ["--arms", original_arm]:
            raise RuntimeError("unexpected deployed arm")
        arm = "memory_" + args.memory_layout + "_" + args.adaptor
        run_joint.SCHEMES = {arm: dict(run_joint.SCHEMES[original_arm])}
        run_joint.ARMS = {arm: ("fixed0", 1.0)}
        sys.argv[-1] = arm
        saved_runner["main"]()

    bridge.load_spec = load_spec
    deployed.DeployedRuntime = MemoryRuntime
    run_joint.save_json, run_joint.main = save_json, run_named_arm
    sys.argv = [str(Path(__file__)), "--deployment-adaptor", args.adaptor, *rest]
    try:
        deployed.main()
    finally:
        bridge.load_spec = original_load
        deployed.DeployedRuntime = original_runtime
        run_joint.save_json, sys.argv = original_save, original_argv
        for key, value in saved_runner.items():
            setattr(run_joint, key, value)


if __name__ == "__main__":
    main()
