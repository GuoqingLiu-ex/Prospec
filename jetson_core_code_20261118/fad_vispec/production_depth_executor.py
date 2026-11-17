"""P0 sequential production candidate: one live native KV, no snapshots.

Only fixed caller-selected D1..8 is supported. The arithmetic, original host
reads, masks/RoPE, posterior/clip/commit/bonus and Draft state are inherited from
the audited ordinary core. P0 uses its existing Python receipt; all unexecuted
depths remain unknown. There is no controller or device cost table.

This is a CPU-validated candidate, not a model/GPU, memory, speed or zero-event
claim. begin_inplace returns a BORROWED controls dict: read-only to callers and
cleared on close, including failure. No cached parent/winner graph is retained.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
from contextlib import ExitStack

import torch

from .ordinary_depth_round import OrdinaryDepthExecutor
from ..focusdraft.real_system_executor import (RealExecutor, _ACTIVE, _DRAFT_FIELDS,
    _MISSING, _hook_ids, _remove_new_hooks)
from ..focusdraft.focus_replay import canonical_ast_dump

SCHEMA = "fad_production_depth_executor_P0_v1"
ROUND_SCHEMA = "fad_production_actual_round_P0_v1"
BASE_EXECUTOR_SHA = "9cbb7999c52c82b9a75f375f2d172f49addd63febf9c260445f9593c8c42ae74"
ORDINARY_CORE_SHA = "bbb9515b0a8502298b77ead0c94736518e005acbb14de7081fbd0a6040bb46ab"

_BEGIN_EDITS = [('def begin(self):', 'def begin_inplace(self):'), ('top2=self._top2(out.logits[0, -1:].detach(), [0], "prefill"))', 'top2=[])'), ('        self._bind_top2_outputs(controls)\n', ''), ('        return self._snapshot(controls)', '        self._working = self._owned_controls = controls\n        return controls')]
_FINISH_EDITS = [('def finish(self, state):', 'def finish_inplace(self):'), ('    if state.request_key != self.request_key or state.controls["stop_reason"] is None:', '    self._check_owned()\n    c = self._owned_controls\n    if c["stop_reason"] is None:'), ('    c = self.restore(state)', '    cache_length = self.bridge._length(self.cache)'), ('state.cache_length', 'cache_length'), ('terminal snapshot cache length/pending boundary mismatch', 'terminal live cache length/pending boundary mismatch')]


def verify_production_source():
    """Cold source/AST guard only; no eval/exec and no model/device operation."""
    paths = [(RealExecutor, BASE_EXECUTOR_SHA), (OrdinaryDepthExecutor, ORDINARY_CORE_SHA)]
    sources = {}
    for kind, expected in paths:
        path = Path(inspect.getfile(kind)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("frozen production dependency source changed: " + path.name)
        sources[str(path)] = digest
    ast_receipts = {}
    for base_name, name, edits in (("begin", "begin_inplace", _BEGIN_EDITS),
                                   ("finish", "finish_inplace", _FINISH_EDITS)):
        expected = textwrap.dedent(inspect.getsource(getattr(RealExecutor, base_name)))
        for old, new in edits:
            count = 2 if old == "state.cache_length" else 1
            if expected.count(old) != count:
                raise ValueError("production lifecycle registered edit does not match base")
            expected = expected.replace(old, new)
        actual = textwrap.dedent(inspect.getsource(getattr(ProductionDepthExecutor, name)))
        dump = lambda code: canonical_ast_dump(ast.parse(code).body[0])
        if dump(expected) != dump(actual):
            raise ValueError("production lifecycle unregistered arithmetic/order change: " + name)
        ast_receipts[name] = hashlib.sha256(dump(actual).encode()).hexdigest()
    if ProductionDepthExecutor._run_tree is not OrdinaryDepthExecutor._run_tree:
        raise ValueError("production must use the unchanged ordinary tree core")
    return dict(source_sha256=sources, lifecycle_ast_sha256=ast_receipts,
        registered_changes=["bind one owned live controls dict instead of final begin snapshot capture",
            "remove diagnostic prefill top2 and empty top2 binding",
            "finish checks live ownership/cache length instead of snapshot install"],
        arithmetic_scope="unchanged ordinary core Target/Draft/posterior/clip/cache/nextroot",
        GPU_validated=False)


def build_p0_feedback(receipt, *, request_key, round_index, prompt_tokens,
                      output_prefix_tokens, max_new_tokens):
    """Only existing host ints/lists; no Tensor/GPU/readback API is called."""
    event = receipt["step_rounds"][0]
    depth = receipt["actual_depth"]
    stopped = receipt["stop_reason"] is not None
    return dict(schema="fad_actual_only_feedback_P0_v1", request_key=request_key,
        round_index=round_index, actual_depth=depth,
        context=dict(prompt_tokens=prompt_tokens, output_prefix_tokens=output_prefix_tokens,
            remaining_before=max_new_tokens-output_prefix_tokens,
            target_cache_before=event["target_cache_before"],
            target_cache_after=event["target_cache_after"],
            semantic_rope_base=event["semantic_rope_base"], tree_nodes=event["tree_nodes"]),
        actual=dict(A_raw=receipt["accepted_draft"], G_tree=receipt["tree_submitted_including_root"],
            G_output=receipt["all_new_output_tokens"],
            terminal_bonus_tokens=receipt["terminal_bonus_tokens"],
            submitted_ids=list(receipt["submitted"]),
            selected_tree_indices=list(receipt["selected_tree_indices"]),
            stop_reason=receipt["stop_reason"], censored=stopped,
            budget_or_eos_clipped=event["budget_or_eos_clipped"]),
        other_depths={str(d): dict(status="unknown", reason="not executed at this state")
                      for d in range(1,9) if d != depth},
        observed_depth_update_eligible=not stopped,
        search_failure_certificate=None, cross_depth_certificate_eligible=False,
        future_or_next_state_guarantee=False,
        actual_cost_table=None, controller_installed=False,
        cost_scope="CPU receipt construction is included; original core host reads remain. No extra Tensor/readback API in this helper; hardware-event count is unmeasured.")


class ProductionDepthExecutor(OrdinaryDepthExecutor):
    """One request-owned mutable controls graph and original full DynamicCache.

    Call begin_inplace, then run_round_inplace(D) until controls.stop_reason,
    then finish_inplace. Do not modify the returned controls or underlying model
    externally. A partial round failure permanently poisons this instance; close
    it. There is no snapshot restore/retry/branch API on this path.
    """
    def __init__(self, *args, **kwargs):
        self.production_source = verify_production_source()
        self._owned_controls = None
        self.production_feedback_history = []
        super().__init__(*args, **kwargs)

    def _check_owned(self):
        self._healthy()
        if (not self.begin_called or self.cache is None or type(self._owned_controls) is not dict
                or self._working is not self._owned_controls):
            raise ValueError("no live controls owned by this production executor")
        self._canonical()

    def _check_working(self, working):
        self._check_owned()
        if working is not self._owned_controls:
            raise ValueError("foreign production controls are forbidden")

    @property
    def controls(self):
        self._check_owned()
        return self._owned_controls

    def _snapshot_forbidden(self, *args, **kwargs):
        raise RuntimeError("snapshot/restore diagnostic API is disabled in production; use in-place lifecycle")

    begin = restore = capture_working = finish = step = _snapshot = _snapshot_forbidden

    @torch.inference_mode()
    def begin_inplace(self):
        self._healthy()
        if self.begin_called:
            raise RuntimeError("begin may run only once")
        if self.target in _ACTIVE or self.draft in _ACTIVE:
            raise RuntimeError("nested real executors on a model are forbidden")
        self.begin_called = True
        _ACTIVE.add(self.target); _ACTIVE.add(self.draft)
        self._saved = [(self.draft, name, vars(self.draft).get(name, _MISSING)) for name in _DRAFT_FIELDS]
        self._saved.append((self.target.model, "rope_deltas", vars(self.target.model).get("rope_deltas", _MISSING)))
        self._stack = ExitStack()
        feature_hook_ids = _hook_ids([self.target.model.language_model, self.target.model.language_model.norm])
        try:
            self.bridge._version()
            self.values = self.bridge._values(self.inputs, self.max_new_tokens,
                self.spec.capacity, self.joint.BMAX)
            ids = self.values["input_ids"]
            self.prompt = int(ids.shape[1])
            self.stops = self.bridge._stops(self.target, self.eos_token_ids)
            self.ops = self.bridge._author()
            self.draft.reset_kv()
            if hasattr(self.draft, "last_img_hidden"):
                self.draft.last_img_hidden = None
            visual_mask = self.bridge._visual_mask(self.target, self.values)
            self._enter_counter(self.bridge._visual_counter(self.target, self.execution_counts), [self.target.model.visual])
            self._enter_counter(self.bridge._draft_counter(self.draft, self.execution_counts), [self.draft, self.draft.layers[0]])
            self.execution_attempts["prefill"] += 1
            out, delta, features = self.bridge._prefill(self.target, self.values,
                self.execution_counts, capture=True)
            self.cache = out.past_key_values
            root = int(out.logits[0, -1].argmax().item())
            reason = "eos" if root in self.stops else ("limit" if self.max_new_tokens == 1 else None)
            # Snapshot immutable constants must already be the frozen request
            # budget (loader may have total_tokens=47). No Draft call occurs here.
            self.joint.apply_tree_budget(self.draft, self.budget)
            controls = dict(generated=[root] if reason else [], rounds=[],
                stop_reason=reason, root=root, first_tree=True, counts={}, ids=ids,
                delta=delta, visual_mask=visual_mask,
                selected_hidden_waiting=features["hidden"],
                first_features=features, visual_context={}, budget_trace=[], imadpt={},
                pending_tokens=[root] if reason else [],
                top2=[])
            if reason:
                controls["rounds"] = [{"kind": "prefill_terminal", "submitted": [root], "pending_tokens": [root]}]
                controls["first_features"] = None
                controls["selected_hidden_waiting"] = None
            else:
                context = self.joint.prepare_draft_visual_context(features["inputs_embeds"],
                    visual_mask, self.joint._grid_raw(self.values), self.joint.SPATIAL_MERGE_SIZE,
                    1.0, merge_operator="cosine")
                self.execution_counts["merge_visual_tokens_in"] += context.original_count
                self.execution_counts["merge_visual_tokens_out"] += context.kept_count
                controls["visual_context"] = self.joint._context_dict(context)
                controls["imadpt"] = {"calls": 0, "overridden": 0, "input_tokens": 0,
                    "actual_input_lengths": [], "actual_calls": 0}
            controls["counts"] = dict(self.execution_counts)
            self._working = self._owned_controls = controls
            return controls
        except BaseException as exc:
            self.poisoned = True
            self.failures.append({"stage": "begin", "type": type(exc).__name__, "message": str(exc)})
            _remove_new_hooks(feature_hook_ids)
            try:
                self.close()
            except BaseException as cleanup_exc:
                self.failures[-1]["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            raise


    def finish_inplace(self):
        self._healthy()
        self._check_owned()
        c = self._owned_controls
        if c["stop_reason"] is None:
            raise ValueError("finish requires this request's terminal state")
        cache_length = self.bridge._length(self.cache)
        expected = self.prompt + len(c["generated"]) - len(c["pending_tokens"])
        if cache_length != expected:
            raise RuntimeError("terminal live cache length/pending boundary mismatch")
        return self.joint.JointResult(list(c["generated"]), c["stop_reason"], self.prompt,
            len(c["generated"]), cache_length, list(c["pending_tokens"]), True,
            dict(c["counts"]), copy.deepcopy(c["rounds"]),
            copy.deepcopy(getattr(self.target, "_vispec_precision", {})),
            visual_context=copy.deepcopy(c["visual_context"]), budget_trace=copy.deepcopy(c["budget_trace"]),
            imadpt=copy.deepcopy(c["imadpt"]))


    @torch.inference_mode()
    def run_round_inplace(self, actual_depth=4):
        self._check_owned()
        working = self._owned_controls
        previous = len(working["generated"])
        try:
            receipt = super().run_round_inplace(working, actual_depth)
            feedback = build_p0_feedback(receipt, request_key=self.request_key,
                round_index=len(self.ordinary_history)-1, prompt_tokens=self.prompt,
                output_prefix_tokens=previous, max_new_tokens=self.max_new_tokens)
            receipt["schema"] = ROUND_SCHEMA
            receipt["production_feedback_included"] = True
            receipt["production_feedback"] = feedback
            self.production_feedback_history.append(copy.deepcopy(feedback))
            return receipt
        except BaseException as exc:
            # Precondition errors occur before any model operation and remain
            # ordinary argument errors. Failed math/core/feedback cannot resume.
            if self.poisoned:
                if self.failures:
                    self.failures[-1]["recovery"] = "close and discard this request; production has no parent snapshot"
            elif len(self.ordinary_history) > len(self.production_feedback_history):
                self.poisoned = True
                self._working = None
                self.failures.append(dict(stage="production_feedback", type=type(exc).__name__,
                    message=str(exc), parent_restored=False,
                    recovery="close and discard this request; production has no parent snapshot"))
            raise

    def execution_receipt(self):
        receipt = super().execution_receipt()
        receipt.update(schema=SCHEMA, production_source=copy.deepcopy(self.production_source),
            production_feedback_included=True,
            production_feedback_history=copy.deepcopy(self.production_feedback_history),
            controller_installed=False, actual_cost_table=None,
            snapshot_graph_copies=0,
            top2_scope="disabled in both prefill and ordinary rounds",
            scope="one sequential live native KV and one owned controls graph; prefill, all actual rounds, Python P0 feedback and cleanup belong to full request. Original host guards/reads remain; zero new helper Tensor APIs is not a zero hardware-event or memory/speed claim.")
        return receipt

    def close(self):
        # Even an externally borrowed controls dict cannot retain this request's
        # prefill features/selected-hidden/ids after normal or failed close.
        owned = self._owned_controls
        try:
            super().close()
        finally:
            if owned is not None:
                owned.clear()
            self._owned_controls = self._working = None
