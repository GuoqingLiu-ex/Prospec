"""One deployed adaptor per process, reusing the complete long-answer runner.

Unlike the switchable research runtime, this entry point constructs only native,
retained BMM, or compiled BMM. It never constructs other compression schemes.
The original checkpoint is loaded first, so installation can only demonstrate
steady-state savings. Loading/conversion peak is preserved separately.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
from itertools import chain
from pathlib import Path
import sys
import weakref

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODES = ("native", "retained_bmm", "compiled")


def unique_storage_bytes(module):
    storages = {}
    for tensor in chain(module.parameters(), module.buffers()):
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class DeployedRuntime:
    def __init__(self, draft, *, mode, receipt, difference_threshold=None):
        import torch
        from compression import FrozenQueryAdaptor
        from compiled_adaptor import CompiledQueryAdaptor

        if mode not in MODES or difference_threshold is not None:
            raise ValueError("Require one supported deployment with no token merging")
        self.draft, self.mode = draft, mode
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        native = draft.imadpt
        weak = {"original_module": weakref.ref(native), "q": weakref.ref(native.q),
                "k_weight": weakref.ref(native.k_proj.weight)}
        if native.k_proj.bias is not None:
            weak["k_bias"] = weakref.ref(native.k_proj.bias)
        native_bytes = unique_storage_bytes(native)
        preparation = None
        if mode != "native":
            prepared = FrozenQueryAdaptor(native, backend="bmm", precision="model")
            preparation = prepared.preparation_metadata()
            draft.imadpt = (prepared if mode == "retained_bmm" else
                            CompiledQueryAdaptor.from_frozen(prepared))
            del prepared
        del native
        gc.collect()
        torch.cuda.synchronize()
        released = {name: ref() is None for name, ref in weak.items()}
        if mode == "compiled" and not all(released.values()):
            raise RuntimeError(f"Original adaptor/Q/K still referenced: {released}")
        self.metadata = dict(mode=mode, preparation=preparation,
            original_adaptor_unique_storage_bytes=native_bytes,
            deployed_adaptor_unique_storage_bytes=unique_storage_bytes(draft.imadpt),
            allocated_before_install_bytes=before,
            allocated_after_install_bytes=torch.cuda.memory_allocated(),
            reserved_after_install_bytes=torch.cuda.memory_reserved(),
            load_and_install_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            load_and_install_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            original_objects_released=released,
            other_scheme_runtimes_constructed=False,
            checkpoint_modified=False,
            startup_scope="Original checkpoint loaded on GPU before folding; no startup saving claim",
            request_peak_scope="Parent runner resets peak before input preparation and records through natural EOS or explicit limit")
        receipt.update(self.metadata)

    @contextmanager
    def activate(self, joint, scheme):
        import torch
        if (scheme["depth"] != "fixed0" or scheme["ratio"] != 1.0 or
                scheme["compression"] is not None):
            raise ValueError("Deployment comparison requires full visual depth and all tokens")
        preparation = dict(self.metadata,
            request_start_allocated_bytes=torch.cuda.memory_allocated(),
            request_start_reserved_bytes=torch.cuda.memory_reserved())
        yield dict(compression={"method": "identity", "actual_ratio": 1.0},
                   adaptor=self.mode, preparation=preparation)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Remaining arguments are passed to run_joint.py (including requests, output, GPU, repeats and EOS limit).")
    parser.add_argument("--deployment-adaptor", choices=MODES, required=True)
    options, rest = parser.parse_known_args()
    forbidden = ("--arms", "--include-ar", "--capture-visual-dir", "--collect-local")
    if any(arg.startswith("--") and any(flag.startswith(arg.split("=", 1)[0])
           for flag in forbidden) for arg in rest):
        parser.error("This entry point uses one installed adaptor; arms, AR, capture and local collection are unsupported")
    import run_joint
    arm = "deployment_" + options.deployment_adaptor
    scheme = dict(depth="fixed0", adaptor=options.deployment_adaptor, compression=None, ratio=1.0)
    receipt = {}
    run_joint.SCHEMES = {arm: scheme}
    run_joint.ARMS = {arm: ("fixed0", 1.0)}
    run_joint.SchemeRuntime = lambda draft, **kwargs: DeployedRuntime(
        draft, mode=options.deployment_adaptor, receipt=receipt, **kwargs)
    original_save = run_joint.save_json

    def save_json(path, value):
        if Path(path).name == "status.json":
            value = dict(value, deployment_adaptor=options.deployment_adaptor,
                         deployment_installation=dict(receipt),
                         methodology="single_deployed_adaptor_complete_long_requests",
                         adaptive_claim=None,
                         quality_claim="Full visual depth and all input tokens; evaluate output equivalence from actual token and round records")
        original_save(path, value)

    run_joint.save_json = save_json
    sys.argv = [str(Path(__file__)), *rest, "--arms", arm]
    run_joint.main()


if __name__ == "__main__":
    main()
