"""One visual-fusion candidate per process on the complete long-answer runner.

imadpt remains native, all visual layers/tokens are retained, and Target/tree
algorithms are unchanged. Thin img_fc/fc wrappers enforce the existing adjacent
pair. The original caller's concatenations remain and are timed, as are wrapper
checks, global-cache misses and the fusion module's own concatenations.

Example: --fusion-mode cached_image --requests requests.jsonl --request-id ID
         --out NEW_DIR --physical-gpu 0 --repeats 1 --warmup 0 --max-new-tokens 512
No experiment starts merely by importing this file.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import sys
import time
import weakref

MODES = ("native", "cached_image", "folded_pair")


class FusionPairState:
    """Plain coordinator; weak references avoid a module -> Draft parent cycle."""

    def __init__(self, draft, fusion, source_refs, source_storage, torch):
        self.draft_ref, self.fusion_ref = weakref.ref(draft), weakref.ref(fusion)
        self.source_refs, self.source_storage = source_refs, source_storage
        self.torch, self.width = torch, fusion.hidden_size
        self.pending = None
        self.active = False
        self.counts = {}

    @property
    def fusion(self):
        module = self.fusion_ref()
        if module is None:
            raise RuntimeError("fusion module no longer exists")
        return module

    def begin_request(self):
        if self.active:
            self.fail("cannot nest visual fusion requests")
        self.pending = None
        self.fusion.reset_cache()
        self.cache_before = self.fusion.cache_info()
        self.counts = {"img_fc_calls": 0, "fc_calls": 0, "completed_pairs": 0,
                       "total_rows": 0, "empty_pairs": 0, "pairing_errors": []}
        self.active = True

    def fail(self, message):
        self.pending = None
        if "pairing_errors" in self.counts:
            self.counts["pairing_errors"].append(message)
        self.fusion.reset_cache()
        raise RuntimeError("visual fusion pairing: " + message)

    def current_global(self):
        draft = self.draft_ref()
        if draft is None:
            self.fail("Draft no longer exists")
        g = getattr(draft, "last_img_hidden", None)
        if not isinstance(g, self.torch.Tensor):
            self.fail("Draft last_img_hidden must be a Tensor")
        return g

    def input_tensor(self, value, name):
        if not self.active:
            self.fail("call outside an active request")
        if (not isinstance(value, self.torch.Tensor) or value.ndim not in (2, 3) or
                value.shape[-1] != 2*self.width or (value.ndim == 3 and value.shape[0] != 1)):
            self.fail(name + " requires [N,2D] or [1,N,2D]")

    def image(self, value):
        self.input_tensor(value, "img_fc")
        self.counts["img_fc_calls"] += 1
        if self.pending is not None:
            self.fail("img_fc called before the previous fc completed")
        g = self.current_global()
        hidden = value[..., :self.width]
        self.pending = {"hidden": hidden, "global": g, "shape": tuple(value.shape)}
        # Same output shape as img_fc. The original caller still concatenates it
        # with embeddings, but fc below deliberately consumes the original h.
        return hidden

    def finish_pair(self, value, fusion):
        self.input_tensor(value, "fc")
        self.counts["fc_calls"] += 1
        pending = self.pending
        if pending is None:
            self.fail("fc has no preceding img_fc")
        if tuple(value.shape) != pending["shape"]:
            self.fail("fc shape differs from its paired img_fc")
        if self.current_global() is not pending["global"]:
            self.fail("global visual Tensor object changed between img_fc and fc")
        try:
            output = fusion(value[..., :self.width], pending["hidden"], pending["global"])
        except BaseException as exc:
            self.counts["pairing_errors"].append(f"fusion failed: {type(exc).__name__}: {exc}")
            fusion.reset_cache()
            raise
        finally:
            self.pending = None
        rows = math.prod(value.shape[:-1])
        self.counts["completed_pairs"] += 1
        self.counts["total_rows"] += rows
        self.counts["empty_pairs"] += int(rows == 0)
        return output

    def end_request(self, error=None):
        dangling = self.pending is not None
        self.pending = None
        self.active = False
        if error is not None or dangling:
            self.fusion.reset_cache()
        if dangling:
            self.counts["pairing_errors"].append("request ended with an unmatched img_fc")
        after = self.fusion.cache_info()
        result = dict(self.counts, instrumented=True,
                      cache_delta={k: after[k] - self.cache_before[k] for k in ("hits", "misses", "resets")},
                      reset_at_request_start=True, cache_after=after,
                      original_caller_concatenations_retained=True,
                      original_img_input_storage_retained_until_fc=True,
                      fusion_internal_concatenation_also_executed=True,
                      forward_checks_and_cache_misses_in_request_timing=True,
                      in_place_global_mutation_supported=False)
        if dangling and error is None:
            raise RuntimeError("visual fusion pairing: request ended with an unmatched img_fc")
        return result

    def installation_metadata(self):
        current = self.fusion.metadata()
        original_keys = {(s["device"], s["data_ptr"]) for s in self.source_storage["storages"]}
        shared = [s for s in current["deployed_weights"]["storages"]
                  if (s["device"], s["data_ptr"]) in original_keys]
        return {"fusion": current, "original_objects_released": {k: ref() is None for k, ref in self.source_refs.items()},
                "source_unique_storage_bytes": self.source_storage["unique_storage_bytes"],
                "source_storage_still_referenced_by_fusion_bytes": sum(s["storage_bytes"] for s in shared),
                "original_caller_concatenations_retained": True,
                "wrapper_strategy": "img_fc returns h with original output shape; following fc executes the fused pair",
                "draft_reference": "weakref", "global_cache_reset": "each request entry, before generation"}


def install_visual_fusion(draft, mode):
    """Permanent per-process installation; no original-module restoration cache."""
    import torch
    from frozen_visual_fusion import FrozenVisualFusion, _storage_info

    if mode not in ("cached_image", "folded_pair"):
        raise ValueError("installation requires cached_image or folded_pair")
    img, fc = draft.img_fc, draft.fc
    refs = {"img_fc": weakref.ref(img), "fc": weakref.ref(fc)}
    for name, module in (("img_fc", img), ("fc", fc)):
        for key, parameter in module.named_parameters():
            refs[name + "." + key] = weakref.ref(parameter)
    del module, parameter
    storage = _storage_info([(name + "." + key, p) for name, module in (("img_fc", img), ("fc", fc))
                             for key, p in module.named_parameters()])
    fusion = FrozenVisualFusion(img, fc, mode=mode)
    state = FusionPairState(draft, fusion, refs, storage, torch)

    class ImageWrapper(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.state = shared
            self.in_features, self.out_features = 2*shared.width, shared.width

        def forward(self, value):
            return self.state.image(value)

    class FinalWrapper(torch.nn.Module):
        def __init__(self, shared, module):
            super().__init__()
            self.state = shared
            self.fusion = module  # Register weights exactly once in the Draft tree.
            self.in_features, self.out_features = 2*shared.width, shared.width

        def forward(self, value):
            return self.state.finish_pair(value, self.fusion)

    draft.img_fc = ImageWrapper(state)
    draft.fc = FinalWrapper(state, fusion)
    del img, fc, fusion
    gc.collect()
    info = state.installation_metadata()
    if not all(info["original_objects_released"][key] for key in ("img_fc", "fc")):
        raise RuntimeError("original fusion modules still referenced after installation")
    if mode == "folded_pair" and info["source_storage_still_referenced_by_fusion_bytes"]:
        raise RuntimeError("folded fusion unexpectedly retains original weight storage")
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fusion-mode", choices=MODES, required=True)
    options, rest = parser.parse_known_args(argv)
    forbidden = ("--arms", "--include-ar", "--capture-visual-dir", "--collect-local", "--deployment-adaptor")
    if any(token.startswith("--") and any(flag.startswith(token.split("=", 1)[0]) for flag in forbidden) for token in rest):
        parser.error("fusion entry requires native imadpt/full visual depth; AR, arms, capture and other deployment modes are unsupported")
    # Parent arguments otherwise omit fusion-mode. Validate it before any resume
    # can reuse an earlier schedule or skip rows from a different candidate.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--resume", action="store_true")
    probe.add_argument("--out")
    resume, _ = probe.parse_known_args(rest)
    if resume.resume:
        if resume.out is None:
            parser.error("resume requires an explicit --out")
        try:
            previous = json.loads((Path(resume.out) / "status.json").read_text())
        except (OSError, ValueError) as exc:
            parser.error("cannot validate fusion resume status: " + str(exc))
        if previous.get("fusion_mode") != options.fusion_mode:
            parser.error("resume requires the same recorded fusion_mode; use a new output directory")
    return options, rest


def main(argv=None):
    options, rest = parse_args(argv)
    import run_deployed_adaptor as deployed
    import run_joint
    original_runtime, original_save, original_argv = deployed.DeployedRuntime, run_joint.save_json, sys.argv
    saved_runner = {k: getattr(run_joint, k) for k in ("SCHEMES", "ARMS", "SchemeRuntime", "main")}
    receipt = {}

    class FusionRuntime(original_runtime):
        def __init__(self, draft, **kwargs):
            import torch
            if kwargs.get("mode") != "native":
                raise ValueError("fusion comparison requires native imadpt")
            super().__init__(draft, **kwargs)
            self.pair_state = None
            torch.cuda.synchronize()
            previous_peak = torch.cuda.max_memory_allocated()
            previous_reserved_peak = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            start = time.perf_counter()
            info = {"mode": options.fusion_mode, "original_modules_used": True}
            if options.fusion_mode != "native":
                self.pair_state = install_visual_fusion(draft, options.fusion_mode)
                info = dict(self.pair_state.installation_metadata(), mode=options.fusion_mode, original_modules_used=False)
            torch.cuda.synchronize()
            info.update(synchronized_install_wall_ms=(time.perf_counter() - start)*1000,
                        allocated_before_bytes=before, allocated_after_bytes=torch.cuda.memory_allocated(),
                        incremental_install_peak_allocated_bytes=torch.cuda.max_memory_allocated() - before,
                        install_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        install_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        performance_pass=False, output_equivalence_pass=False,
                        scope="installation after native checkpoint load; original caller cats/checks retained; no startup saving claim")
            self.metadata.update(fusion_mode=options.fusion_mode, fusion_installation=info,
                                 allocated_after_install_bytes=torch.cuda.memory_allocated(),
                                 reserved_after_install_bytes=torch.cuda.memory_reserved(),
                                 load_and_install_peak_allocated_bytes=max(previous_peak, torch.cuda.max_memory_allocated()),
                                 load_and_install_peak_reserved_bytes=max(previous_reserved_peak, torch.cuda.max_memory_reserved()))
            kwargs["receipt"].update(self.metadata)
            receipt.update(self.metadata)

        @contextmanager
        def activate(self, joint, scheme):
            with super().activate(joint, scheme) as state:
                if self.pair_state is None:
                    state["preparation"]["fusion_request"] = {"instrumented": False, "mode": "native",
                                                              "img_fc_calls": None, "fc_calls": None, "cache_delta": None}
                    yield state
                    return
                self.pair_state.begin_request()
                error = None
                try:
                    yield state
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    state["preparation"]["fusion_request"] = self.pair_state.end_request(error=error)

    def save_json(path, value):
        if Path(path).name == "status.json":
            value = dict(value, fusion_mode=options.fusion_mode, fusion_installation=receipt.get("fusion_installation", {}),
                         methodology="single_visual_fusion_candidate_complete_long_requests",
                         quality_claim="FP16 fusion reassociation may change Draft/output; validate actual token and round records",
                         performance_pass=False, output_equivalence_pass=False,
                         unchanged_components=["native imadpt", "full visual depth and tokens", "Draft Transformer", "tree construction", "Target verification"])
        original_save(path, value)

    deployed.DeployedRuntime = FusionRuntime
    run_joint.save_json = save_json

    def run_named_fusion_arm():
        # deployed.main sets imadpt=native and appends --arms deployment_native.
        # Give results a unique fusion identity before run_joint parses that CLI.
        if sys.argv[-2:] != ["--arms", "deployment_native"]:
            raise RuntimeError("unexpected deployed entry arm configuration")
        arm = "fusion_" + options.fusion_mode
        run_joint.SCHEMES = {arm: dict(run_joint.SCHEMES["deployment_native"])}
        run_joint.ARMS = {arm: ("fixed0", 1.0)}
        sys.argv[-1] = arm
        saved_runner["main"]()

    run_joint.main = run_named_fusion_arm
    sys.argv = [str(Path(__file__)), "--deployment-adaptor", "native", *rest]
    try:
        deployed.main()
    finally:
        deployed.DeployedRuntime, run_joint.save_json, sys.argv = original_runtime, original_save, original_argv
        for key, value in saved_runner.items():
            setattr(run_joint, key, value)


if __name__ == "__main__":
    main()
