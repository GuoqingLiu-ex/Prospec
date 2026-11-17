"""Capture paired native img_fc/fc inputs during one complete diagnostic request.

Only this entry point is instrumented: full visual depth, native adaptor, one
request, one repeat, no warmup, natural EOS with a 512-token limit. CPU copies
and serialization happen in hooks, so ALL recorded timing/memory/energy values
are diagnostic and must not be used for performance conclusions.

The observer saves at most eight unique (row_count, global Tensor object) pairs.
One slot is reserved for a 3D eight-row call (the current K=8 branch shape).
Empty rows are counted without capture. All hook returns are None.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import weakref


DIAGNOSTIC_NOTE = (
    "Native observation with synchronous CPU copies and file writes in hooks; "
    "timing, memory peaks and energy are diagnostic only, not performance evidence."
)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def mark_diagnostic_results(path):
    """Annotate only this entry point's output, including partially finished runs."""
    path = Path(path)
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    reason = "visual_fusion_capture_cpu_copy_and_file_write_instrumentation"
    for row in rows:
        reasons = list(row.get("timing_reasons", []))
        if reason not in reasons:
            reasons.append(reason)
        row.update(visual_fusion_capture=True, valid_timing=False, timing_valid=False,
                   memory_valid=False, energy_attribution_valid=False,
                   timing_reasons=reasons, timing_note=DIAGNOSTIC_NOTE,
                   memory_scope="Target and Draft resident; fusion-hook CPU copies and file writes are diagnostic instrumentation; recorded peaks are not deployment or request-memory evidence")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


class VisualFusionCapture:
    """Observe the strict img_fc -> fc sequence; never replace module outputs."""

    def __init__(self, draft, directory, *, request_id, max_captures=8,
                 checkpoint_reference=None):
        import torch

        if not isinstance(max_captures, int) or not 2 <= max_captures <= 8:
            raise ValueError("max_captures must be between two and eight")
        self.torch, self.draft = torch, draft
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.request_id, self.max_captures = request_id, max_captures
        self.width = int(draft.img_fc.out_features)
        if (draft.img_fc.in_features != 2 * self.width or
                draft.fc.in_features != 2 * self.width or
                draft.fc.out_features != self.width):
            raise ValueError("expected consecutive Linear(2D,D) modules")
        self.checkpoint_reference = checkpoint_reference
        self._pending = None
        self._global_objects = {}
        self._next_global_index = 1
        self._saved_keys = set()
        self._branch_saved = False
        self._closed = False
        self._errors = []
        self.captures = []
        self.img_calls = self.fc_calls = self.completed_pairs = 0
        self.img_total_rows = self.fc_total_rows = 0
        self.empty_rows = self.cpu_snapshot_calls = 0
        self.img_shapes, self.fc_shapes = Counter(), Counter()
        self.global_calls, self.global_rows = Counter(), Counter()
        self.skip_reasons = Counter()
        # Preserve the actual post-load dtype/values; do not reconstruct from
        # the checkpoint's FP32 tensors in the downstream module benchmark.
        weights = {}
        for module_name in ("img_fc", "fc"):
            module = getattr(draft, module_name)
            for parameter_name in ("weight", "bias"):
                parameter = getattr(module, parameter_name)
                weights[f"{module_name}.{parameter_name}"] = (
                    None if parameter is None else
                    parameter.detach().to(device="cpu", copy=True).contiguous())
        self.weight_shapes = {name: None if value is None else list(value.shape)
                              for name, value in weights.items()}
        torch.save(weights, self.directory / "weights.pt")
        del weights
        self.handles = [
            draft.img_fc.register_forward_pre_hook(self._img_pre),
            draft.img_fc.register_forward_hook(self._img_post),
            draft.fc.register_forward_pre_hook(self._fc_pre),
            draft.fc.register_forward_hook(self._fc_post),
        ]

    def _fail(self, message):
        self._errors.append(message)
        raise RuntimeError("visual fusion capture pairing: " + message)

    def _input(self, inputs, name):
        if len(inputs) != 1 or not isinstance(inputs[0], self.torch.Tensor):
            self._fail(name + " must receive exactly one positional Tensor")
        tensor = inputs[0]
        if tensor.ndim < 2 or tensor.shape[-1] != 2 * self.width:
            self._fail(name + " input must have last dimension 2D")
        return tensor

    def _cpu(self, tensor):
        self.cpu_snapshot_calls += 1
        return tensor.detach().to(device="cpu", copy=True).contiguous()

    def _equal_bits(self, left, right):
        return (left.dtype == right.dtype and left.shape == right.shape and
                self.torch.equal(left.contiguous().view(self.torch.uint8),
                                 right.contiguous().view(self.torch.uint8)))

    def _global_index(self, value):
        # Tensor equality is elementwise: do not use WeakKeyDictionary here.
        # Weak refs also avoid keeping obsolete visual outputs alive on GPU.
        if value is None:
            return 0
        if not isinstance(value, self.torch.Tensor):
            self._fail("last_img_hidden must be a Tensor or None")
        identity = id(value)
        previous = self._global_objects.get(identity)
        if previous is not None and previous[0]() is value:
            return previous[1]
        index = self._next_global_index
        self._next_global_index += 1
        self._global_objects[identity] = (weakref.ref(value), index)
        return index

    def _img_pre(self, _module, inputs):
        self.img_calls += 1
        if self._pending is not None:
            self._fail("a new img_fc call arrived before the preceding fc completed")
        x = self._input(inputs, "img_fc")
        shape = tuple(x.shape)
        self.img_shapes[shape] += 1
        rows = math.prod(shape[:-1])
        self.img_total_rows += rows
        global_value = getattr(self.draft, "last_img_hidden", None)
        global_index = self._global_index(global_value)
        self.global_calls[global_index] += 1
        self.global_rows[global_index] += rows
        key = (rows, global_index)
        branch_shape = x.ndim == 3 and rows == 8
        reason = None
        if rows == 0:
            reason = "empty_rows"
            self.empty_rows += 1
        elif rows == 8 and x.ndim == 2 and not self._branch_saved:
            # A prefill of exactly eight rows could consume the same (8,g)
            # key needed by the first branch. Prefer its real branch input.
            reason = "eight_row_prefill_key_reserved_for_branch"
        elif key in self._saved_keys:
            reason = "duplicate_row_count_and_global_object"
        elif len(self.captures) >= self.max_captures:
            reason = "capture_cap"
        elif (not self._branch_saved and not branch_shape and
              len(self.captures) >= self.max_captures - 1):
            reason = "reserved_for_eight_row_branch"
        if reason is not None:
            self.skip_reasons[reason] += 1
        pending = dict(stage="img_forward", selected=reason is None,
                       input_shape=shape, output_shape=shape[:-1] + (self.width,),
                       key=key, global_index=global_index, rows=rows,
                       branch_shape=branch_shape, img_call_index=self.img_calls)
        self._pending = pending
        if pending["selected"]:
            observed = self._cpu(x)
            hidden = observed[..., :self.width].contiguous()
            visual_rows = observed[..., self.width:].contiguous()
            if global_value is None:
                # Defensive no-image support. The original Model normally uses
                # an explicit zero Tensor; infer only from a nonempty input.
                visual = visual_rows.reshape(rows, self.width)[:1].clone()
                global_source = "inferred_from_input_tail_no_global_tensor"
            else:
                visual = self._cpu(global_value)
                global_source = "draft.last_img_hidden"
            try:
                expanded = visual.expand_as(visual_rows)
            except RuntimeError as exc:
                self._fail("global visual shape cannot expand to img_fc input: " + str(exc))
            if not self._equal_bits(expanded, visual_rows):
                self._fail("img_fc visual half differs from current global visual")
            pending.update(hidden=hidden, global_visual=visual,
                           global_source=global_source)

    def _img_post(self, _module, _inputs, output):
        pending = self._pending
        if pending is None or pending["stage"] != "img_forward":
            self._fail("img_fc output has no matching input")
        if (not isinstance(output, self.torch.Tensor) or
                tuple(output.shape) != pending["output_shape"]):
            self._fail("unexpected img_fc output shape")
        pending["stage"] = "await_fc"
        if pending["selected"]:
            pending["native_img_output"] = self._cpu(output)

    def _fc_pre(self, _module, inputs):
        self.fc_calls += 1
        x = self._input(inputs, "fc")
        self.fc_shapes[tuple(x.shape)] += 1
        self.fc_total_rows += math.prod(x.shape[:-1])
        pending = self._pending
        if pending is None or pending["stage"] != "await_fc":
            self._fail("fc did not immediately follow a completed img_fc")
        if tuple(x.shape) != pending["input_shape"]:
            self._fail("fc leading dimensions differ from paired img_fc")
        pending.update(stage="fc_forward", fc_call_index=self.fc_calls)
        if pending["selected"]:
            observed = self._cpu(x)
            if not self._equal_bits(observed[..., self.width:], pending["native_img_output"]):
                self._fail("fc hidden half differs from the paired img_fc output")
            pending["embed"] = observed[..., :self.width].contiguous()

    def _fc_post(self, _module, _inputs, output):
        pending = self._pending
        if pending is None or pending["stage"] != "fc_forward":
            self._fail("fc output has no matching input")
        if (not isinstance(output, self.torch.Tensor) or
                tuple(output.shape) != pending["output_shape"]):
            self._fail("unexpected fc output shape")
        if pending["selected"]:
            filename = f"sample_{len(self.captures) + 1:03d}.pt"
            tensors = {name: pending[name] for name in (
                "embed", "hidden", "global_visual", "native_img_output")}
            tensors["native_fusion_output"] = self._cpu(output)
            metadata = dict(
                schema_version=1, request_id=self.request_id,
                capture_index=len(self.captures) + 1,
                img_fc_call_index=pending["img_call_index"],
                fc_call_index=pending["fc_call_index"],
                row_count=pending["rows"], global_object_index=pending["global_index"],
                global_source=pending["global_source"],
                shape={name: list(tensor.shape) for name, tensor in tensors.items()},
                img_fc_input_shape=list(pending["input_shape"]),
                phase_hint=("eight_row_branch_shape" if pending["branch_shape"] else
                            "prefill_text_shape" if len(pending["input_shape"]) == 2 else
                            "draft_sync_shape"),
                phase_hint_note="Shape-based hint; original call sequence is recorded separately.",
                dtype=str(output.dtype), timing_valid=False, memory_valid=False,
                diagnostic_note=DIAGNOSTIC_NOTE,
            )
            destination = self.directory / filename
            temporary = destination.with_suffix(".pt.tmp")
            self.torch.save(dict(metadata, **tensors), temporary)
            temporary.replace(destination)
            self.captures.append(dict(metadata, file=filename))
            self._saved_keys.add(pending["key"])
            self._branch_saved = self._branch_saved or pending["branch_shape"]
        self.completed_pairs += 1
        self._pending = None

    def summary(self):
        def counts(counter):
            return [dict(shape=list(shape), calls=count)
                    for shape, count in sorted(counter.items())]
        return dict(
            schema_version=1, request_id=self.request_id, max_captures=self.max_captures,
            captured_pairs=len(self.captures), completed_pairs=self.completed_pairs,
            img_fc_calls=self.img_calls, fc_calls=self.fc_calls,
            img_fc_total_rows=self.img_total_rows, fc_total_rows=self.fc_total_rows,
            img_fc_shapes=counts(self.img_shapes), fc_shapes=counts(self.fc_shapes),
            global_states=[dict(global_object_index=index, calls=self.global_calls[index],
                                total_rows=self.global_rows[index])
                           for index in sorted(self.global_calls)],
            empty_row_pairs=self.empty_rows, skip_reasons=dict(self.skip_reasons),
            cpu_snapshot_calls=self.cpu_snapshot_calls,
            eight_row_branch_shape_captured=self._branch_saved,
            captures=self.captures, pairing_errors=list(self._errors),
            checkpoint_reference=self.checkpoint_reference,
            weights_file="weights.pt", weight_shapes=self.weight_shapes,
            weight_scope="CPU snapshots of the four live native tensors after model loading/casting; outside request hooks",
            weight_keys=["img_fc.weight", "img_fc.bias", "fc.weight", "fc.bias"],
            model_dtype=str(self.draft.img_fc.weight.dtype),
            timing_valid=False, memory_valid=False, energy_attribution_valid=False,
            diagnostic_note=DIAGNOSTIC_NOTE,
        )

    def close(self, error=None):
        if self._closed:
            return
        self._closed = True
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        dangling = self._pending is not None
        if dangling and error is None:
            error = "request ended with an unmatched img_fc/fc pair"
            self._errors.append(error)
        self._pending = None
        value = self.summary()
        value.update(status="failed" if error or self._errors else "completed",
                     error=error, hooks_removed=True)
        _write_json(self.directory / "summary.json", value)
        if dangling and error == "request ended with an unmatched img_fc/fc pair":
            raise RuntimeError(error)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", required=True, help="New independent output directory")
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--requests", required=True, help="Existing request JSONL manifest")
    parser.add_argument("--request-id", help="Required if the manifest has more than one row")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    if args.request_id is not None:
        rows = [row for row in rows if row["request_id"] == args.request_id]
    if len(rows) != 1:
        parser.error("select exactly one existing request (duplicate IDs are not accepted)")
    args.request_id = rows[0]["request_id"]
    args.requests = str(Path(args.requests).resolve())
    args.out = str(Path(args.out).resolve())
    if Path(args.out).exists():
        parser.error("output directory already exists; use a new directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    import run_deployed_adaptor as deployed
    import run_joint

    original_runtime = deployed.DeployedRuntime
    original_save = run_joint.save_json
    original_parent_state = {name: getattr(run_joint, name)
                             for name in ("SCHEMES", "ARMS", "SchemeRuntime")}
    original_argv = sys.argv
    observers = []
    checkpoint = str(run_joint.OLD_RUN / "models/ViSpec-Qwen2.5-VL-7B-Instruct/model.safetensors")

    class CaptureRuntime(original_runtime):
        def __init__(self, draft, **kwargs):
            if kwargs.get("mode") != "native":
                raise ValueError("capture requires native adaptor")
            super().__init__(draft, **kwargs)
            self.capture = VisualFusionCapture(
                draft, Path(args.out) / "fusion_capture", request_id=args.request_id,
                checkpoint_reference=checkpoint)
            observers.append(self.capture)

        @contextmanager
        def activate(self, joint, scheme):
            try:
                with super().activate(joint, scheme) as state:
                    state["preparation"].update(
                        visual_fusion_capture=True, timing_valid=False, memory_valid=False,
                        diagnostic_note=DIAGNOSTIC_NOTE)
                    yield state
            except BaseException as exc:
                self.capture.close(error=f"{type(exc).__name__}: {exc}")
                raise
            else:
                self.capture.close()

    def save_json(path, value):
        if Path(path).name == "status.json":
            value = dict(value, methodology="native_visual_fusion_input_capture",
                         visual_fusion_capture=True, timing_valid=False, memory_valid=False,
                         diagnostic_note=DIAGNOSTIC_NOTE,
                         capture_fixed_configuration=dict(requests=1, repeats=1, warmup=0,
                                                          max_new_tokens=512, adaptor="native"))
        original_save(path, value)

    deployed.DeployedRuntime = CaptureRuntime
    run_joint.save_json = save_json
    sys.argv = [str(Path(__file__)), "--deployment-adaptor", "native",
                "--out", args.out, "--physical-gpu", str(args.physical_gpu),
                "--requests", args.requests, "--request-id", args.request_id,
                "--repeats", "1", "--warmup", "0", "--max-new-tokens", "512",
                "--seed", str(args.seed)]
    failure = None
    try:
        deployed.main()
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            for observer in observers:
                observer.close(error=failure)
            mark_diagnostic_results(Path(args.out) / "results.jsonl")
        finally:
            deployed.DeployedRuntime = original_runtime
            run_joint.save_json = original_save
            for name, value in original_parent_state.items():
                setattr(run_joint, name, value)
            sys.argv = original_argv


if __name__ == "__main__":
    main()
