"""Real single-round FAD core; caller owns snapshots OUTSIDE its timed scope.

Default calls the frozen author Draft and native Target/cache bridge. No fake
model, reference-token driver, trace collector, top2, all-node argmax/export or
feedback is installed. This core still includes original author host tree work,
posterior, clip, full-cache commit and next-state updates. It is not full-policy
cost: controller/production feedback and end-to-end memory remain unmeasured.

Usage (outer grouped/precision context unchanged):
    with OrdinaryDepthExecutor(spec, inputs, request_key=...) as executor:
        parent = executor.begin()                 # prefill + snapshot, untimed
        working = executor.restore(parent)        # graph copy, untimed
        synchronize(); start_timer()
        receipt = executor.run_round_inplace(working, actual_depth=4)
        synchronize(); stop_timer()
        child = executor.capture_working(working) # optional graph copy, untimed

Controls must be the live dict returned by THIS executor.restore; parent.controls
is read-only and rejected. The same working dict may continue sequentially.
Failure poisons the executor and restores canonical constants, not mutated KV;
the caller may explicitly restore(parent) AFTER stopping its failed timer. No
failed interval is a valid cost. close restores original outer attributes/hooks.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import SimpleNamespace

import torch

from ..focusdraft.real_system_executor import RealExecutor, _difference
from ..focusdraft.focus_replay import canonical_ast_dump
from .traced_topk import (generate_traced_topk, verify_source_contract,
                          _setup_source_guard, _check_loaded_author)

SCHEMA = "fad_ordinary_single_round_core_v1"
BASE_EXECUTOR_SHA = "9cbb7999c52c82b9a75f375f2d172f49addd63febf9c260445f9593c8c42ae74"
CANONICAL = (29, 3, 8)

# Literal, counted source edits are audited at setup. They do not exec/eval code.
# _run_tree below is a static function; all unspecified arithmetic/order is the
# frozen RealExecutor implementation. Comment-only changes do not alter the AST.
_SOURCE_EDITS = [('nodes = self.budget.nodes_including_root', 'nodes = min(30, 1 + 8 + 64 * (self._requested_depth - 1))'), ('draft.topK_genrate(', 'self._proposal('), ('    self._active_tree_diagnostic = self._tree_diagnostic(tokens, retrieve, depths, allowed_tree)\n', ''), ('    old_output_len = len(c["generated"])\n', ''), ('budget_action=self.budget.action_id,', 'budget_action=self.budget.action_id if self._requested_depth == 4 else f"fad_D{self._requested_depth}",'), ('budget=dict(B=nodes, d=self.budget.expansions, K=self.budget.beam_width, total_tokens=self.budget.total_tokens),', 'budget=dict(B=nodes, d=self._requested_depth - 1, K=8, total_tokens=nodes - 1),'), ('    # The logits for selected y_i predict y_(i+1). The last prediction is\n    # retained as diagnostic even at EOS; it is not an invented output token.\n    c["top2"].extend(self._top2(tree_logits[best, :keep],\n        range(old_output_len + 1, old_output_len + keep + 1), "tree", event["selected_tree_indices"]))\n', '')]


def verify_core_source():
    """Setup-only source/AST receipt; not a numerical or device validation."""
    path = Path(inspect.getfile(RealExecutor))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BASE_EXECUTOR_SHA:
        raise ValueError("frozen RealExecutor source changed")
    expected = textwrap.dedent(inspect.getsource(RealExecutor._run_tree))
    for old, new in _SOURCE_EDITS:
        count = 2 if old == "draft.topK_genrate(" else 1
        if expected.count(old) != count:
            raise ValueError("ordinary core source edit no longer matches base")
        expected = expected.replace(old, new)
    actual = textwrap.dedent(inspect.getsource(OrdinaryDepthExecutor._run_tree))
    dump = lambda src: canonical_ast_dump(ast.parse(src).body[0])
    if dump(expected) != dump(actual):
        raise ValueError("ordinary core contains unregistered arithmetic/order changes")
    author = verify_source_contract()
    return dict(base_executor_path=str(path), base_executor_sha256=digest,
        ordinary_run_tree_ast_sha256=hashlib.sha256(dump(actual).encode()).hexdigest(),
        author=author, allowed_changes=["D-dependent safe tree budget and event",
            "trace=False loaded author call", "remove tree/top2 diagnostic only"],
        scope="source correspondence only; actual model/GPU parity untested")


class OrdinaryDepthExecutor(RealExecutor):
    """Reuse native lifecycle/counters, expose only in-place ordinary tree calls.

    begin/restore/capture_working/finish remain explicit OFFLINE orchestration;
    run_round_inplace never invokes any of them. Instance working ownership is
    required; no user-supplied callable replaces production math. CPU tests use
    inherited explicit joint/snapshot dependency injection.
    """
    def __init__(self, *args, **kwargs):
        self.core_source = verify_core_source()
        self._working = None
        self._requested_depth = None
        self.ordinary_history = []
        super().__init__(*args, **kwargs)
        if type(self.draft.top_k) is not int or self.draft.top_k != 8:
            raise ValueError("ordinary FAD freezes K8")
        function = getattr(self.draft.topK_genrate, "__func__", None)
        if function is None:
            raise ValueError("bound original author topK_genrate required")
        # Pay cold source checks before any round; trace=False retains cached
        # source/argument checks in its actual hot call.
        _setup_source_guard()
        _check_loaded_author(function)

    def _canonical(self):
        values = (self.draft.total_tokens, self.draft.depth, self.draft.top_k)
        if values != CANONICAL or any(type(x) is not int for x in values):
            raise RuntimeError("ordinary live constants are not canonical B30/d3/K8")

    def _top2(self, *args, **kwargs):
        # Base begin expects the list. No topk, tensor read or transfer occurs.
        return []

    def step(self, *args, **kwargs):
        raise RuntimeError("ordinary core requires restore then run_round_inplace; step would clone snapshots")

    @torch.inference_mode()
    def restore(self, parent):
        self._canonical()
        self._working = None
        working = super().restore(parent)
        self._working = working
        return working

    def _check_working(self, working):
        if (not self.begin_called or self.cache is None or type(working) is not dict
                or working is not self._working):
            raise ValueError("working controls must come from this executor.restore")
        self._canonical()

    @torch.inference_mode()
    def capture_working(self, working):
        """Explicit untimed diagnostic copy, never called by the round core."""
        self._healthy()
        self._check_working(working)
        return self._snapshot(working)

    def _proposal(self, *args, **kwargs):
        result = generate_traced_topk(self.draft, *args, **kwargs,
            actual_depth=self._requested_depth, budget_cap=30, trace=False)
        if result.trace is not None:
            raise RuntimeError("ordinary proposal unexpectedly retained a trace")
        return result.tree

    @torch.inference_mode()
    def run_round_inplace(self, working, actual_depth=4):
        """Mutate installed live controls and return a small tensor-free receipt.

        Includes ordinary Python event/counter bookkeeping. No controller,
        certificate production, snapshots, output text decode or file writes.
        On failure, stop timer before calling restore(parent); calls/attempts
        remain cumulative so partial work is not mistaken for a successful row.
        """
        self._healthy()
        self._check_working(working)
        if type(actual_depth) is not int or not 1 <= actual_depth <= 8:
            raise ValueError("actual_depth must be an integer in [1,8]")
        if self._requested_depth is not None:
            raise RuntimeError("nested ordinary round forbidden")
        if working["stop_reason"] is not None:
            raise ValueError("cannot run a terminal state")
        self._coverage_validated = False
        before = dict(self.execution_counts)
        old_rounds, old_outputs = len(working["rounds"]), len(working["generated"])
        self._requested_depth = actual_depth
        try:
            self._run_tree(working)
            delta = _difference(self.execution_counts, before)
            working["counts"] = {key: working["counts"][key] + delta[key] for key in delta}
            step_rounds = copy.deepcopy(working["rounds"][old_rounds:])
            self.execution_rounds.extend(copy.deepcopy(step_rounds))
            event = step_rounds[0]
            receipt = dict(schema=SCHEMA, actual_depth=actual_depth,
                tree_nodes=event["tree_nodes"], tree_submitted_including_root=len(event["submitted"]),
                all_new_output_tokens=len(working["generated"]) - old_outputs,
                terminal_bonus_tokens=sum(len(r["submitted"]) for r in step_rounds[1:]),
                accepted_draft=event["accepted_draft"], committed_draft=event["committed_draft"],
                submitted=list(event["submitted"]), selected_tree_indices=list(event["selected_tree_indices"]),
                stop_reason=working["stop_reason"], step_rounds=step_rounds,
                execution_counts_delta=delta, production_feedback_included=False)
            self.ordinary_history.append(dict(actual_depth=actual_depth, tree_nodes=event["tree_nodes"]))
            return receipt
        except BaseException as exc:
            self.poisoned = True
            self._working = None
            self.failures.append(dict(stage="ordinary_round", actual_depth=actual_depth,
                type=type(exc).__name__, message=str(exc),
                execution_counts_delta=_difference(self.execution_counts, before),
                parent_restored=False, recovery="caller must stop failed timer then explicitly restore(parent)"))
            raise
        finally:
            # Proposal already has its own finally. This also covers errors in
            # other modules after partial KV writes; no tensor/cache copy here.
            self.draft.total_tokens, self.draft.depth, self.draft.top_k = CANONICAL
            self._requested_depth = None

    def _run_tree(self, c):
        j, b, draft, target, counts = self.joint, self.bridge, self.draft, self.target, self.execution_counts
        ids, root = c["ids"], c["root"]
        j.apply_tree_budget(draft, self.budget)
        counts["budget_switch_calls"] += 1
        nodes = min(30, 1 + 8 + 64 * (self._requested_depth - 1))
        self.execution_attempts["draft_tree"] += 1
        if c["first_tree"]:
            context = SimpleNamespace(**c["visual_context"], merged_embeddings=None)
            with j.first_tree_imadpt(draft, context, False) as (actual, state):
                tree = self._proposal(c["selected_hidden_waiting"],
                    torch.cat((ids, ids.new_tensor([[root]])), 1), target.get_output_embeddings(), None,
                    inputs_embeds=c["first_features"]["inputs_embeds"], image_mask=c["visual_mask"])
            counts["imadpt_calls"] += state["calls"]
            counts["imadpt_input_tokens"] += sum(state["input_lengths"])
            counts["imadpt_overridden_calls"] += int(state["overridden"])
            if context.original_count == 0 and c["visual_mask"] is not None and bool(c["visual_mask"].any()):
                raise RuntimeError("fail closed: visual span present but empty context")
            if state["overridden"]:
                raise RuntimeError("full-token original adaptor must not be overridden")
            c["imadpt"] = dict(calls=counts["imadpt_calls"], overridden=counts["imadpt_overridden_calls"],
                input_tokens=counts["imadpt_input_tokens"], actual_input_lengths=actual.get("actual_input_lengths", []),
                actual_calls=actual.get("calls", 0))
            c["first_tree"], c["first_features"] = False, None
        else:
            tree = self._proposal(c["selected_hidden_waiting"],
                torch.cat((ids, ids.new_tensor([[root]])), 1), target.get_output_embeddings(), None)
        counts["draft_tree_calls"] += 1
        tokens, retrieve, tree_mask, depths = tree
        base = int(ids.shape[1])
        if b._length(self.cache) != base or tokens.shape != (1, nodes) or int(tokens[0, 0]) != root:
            raise RuntimeError("tree root/size/native committed cache boundary mismatch")
        retrieve, tree_mask, depths = retrieve.to(ids.device), tree_mask.to(ids.device), depths.to(ids.device)
        if tree_mask.shape != (1, 1, nodes, nodes) or depths.shape != (nodes,):
            raise RuntimeError("author tree mask/depth shape mismatch")
        allowed_tree = tree_mask.bool()
        if not bool(allowed_tree[0, 0].diagonal().all()) or not bool(allowed_tree[0, 0, :, 0].all()):
            raise RuntimeError("tree nodes must attend to self and root")
        if not torch.equal(depths, allowed_tree[0, 0].sum(-1).long() - 1):
            raise RuntimeError("tree RoPE depth disagrees with ancestor mask")
        allowed = torch.cat((torch.ones((1, 1, nodes, base), device=ids.device, dtype=torch.bool), allowed_tree), -1)
        mask = torch.zeros(allowed.shape, device=ids.device, dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)
        position_ids = (base + c["delta"] + depths).view(1, 1, nodes).expand(3, 1, nodes)
        with b._capture_features(target) as feature:
            self.execution_attempts["target_verify"] += 1
            verified = target(input_ids=tokens.to(ids.device), past_key_values=self.cache,
                attention_mask=mask, position_ids=position_ids, use_cache=True,
                logits_to_keep=0, return_dict=True)
        counts["target_verify_calls"] += 1
        counts["target_verify_query_tokens"] += nodes
        if verified.past_key_values is not self.cache:
            raise RuntimeError("actual Target replaced native cache shell")
        if b._length(self.cache) != base + nodes or "hidden" not in feature:
            raise RuntimeError("native verification did not expose complete tree KV/feature")
        padded_tokens = torch.cat((tokens, tokens.new_full((1, 1), -1)), 1)
        if retrieve.ndim != 2 or bool((retrieve < -1).any()) or bool((retrieve >= nodes).any()):
            raise RuntimeError("invalid retrieve indices")
        candidates, tree_logits = padded_tokens[0, retrieve], verified.logits[0, retrieve]
        best, accepted, _ = self.ops.evaluate_posterior(tree_logits, candidates, None)
        best, accepted = int(best), int(accepted)
        proposed = candidates[best, :accepted + 1].tolist()
        if not proposed or proposed[0] != root or any(token < 0 for token in proposed):
            raise RuntimeError("greedy posterior selected invalid root/path")
        committed, reason = b._clip(proposed, self.max_new_tokens - len(c["generated"]), self.stops)
        keep = len(committed)
        selected = retrieve[best, :keep]
        self.execution_attempts["compaction"] += 1
        b._compact(self.cache, selected, base, nodes)
        counts["target_compaction_calls"] += 1
        counts["target_compaction_selected_tokens"] += keep
        c["ids"] = torch.cat((ids, ids.new_tensor([committed])), 1)
        c["generated"].extend(committed)
        event = dict(kind="tree", tree_nodes=nodes, budget_action=self.budget.action_id if self._requested_depth == 4 else f"fad_D{self._requested_depth}",
            budget=dict(B=nodes, d=self._requested_depth - 1, K=8, total_tokens=nodes - 1),
            accepted_draft=accepted, committed_draft=keep - 1, submitted=committed,
            target_cache_before=base, target_cache_after=b._length(self.cache),
            discarded_tree_nodes=nodes - keep, budget_or_eos_clipped=keep != len(proposed),
            pending_tokens=[], selected_tree_indices=selected.tolist(), semantic_rope_base=base + c["delta"],
            selectedH_waiting_sync=False, visual_m=1.0)
        c["rounds"].append(event); c["budget_trace"].append(event["budget"])
        c["stop_reason"], c["pending_tokens"] = reason, []
        if reason:
            c["selected_hidden_waiting"] = None
            return
        root = int(tree_logits[best, keep - 1].argmax().item())
        c["root"] = root
        reason = "eos" if root in self.stops else ("limit" if len(c["generated"]) + 1 == self.max_new_tokens else None)
        c["stop_reason"] = reason
        if reason:
            c["generated"].append(root)
            c["pending_tokens"] = [root]
            c["rounds"].append(dict(kind="terminal_target_token", submitted=[root],
                target_cache_after=b._length(self.cache), pending_tokens=[root],
                selectedH_waiting_sync=True, known_root=root))
            c["selected_hidden_waiting"] = None
        else:
            c["selected_hidden_waiting"] = feature["hidden"][:, selected]

    def execution_receipt(self):
        receipt = super().execution_receipt()
        receipt.update(schema=SCHEMA, core_source=copy.deepcopy(self.core_source),
            ordinary_round_history=copy.deepcopy(self.ordinary_history),
            actual_verify_nodes=sum(r["tree_nodes"] for r in self.ordinary_history),
            top2_scope="disabled; no top2 operation in begin or ordinary rounds",
            production_feedback_included=False,
            scope="ordinary core excludes caller-owned snapshot install/capture, prefill and controller/feedback; inherited ledger records explicit outside-core orchestration; not measured TPS, DRAM or deployment whole-memory acceptance")
        return receipt

    def close(self):
        try:
            super().close()
        finally:
            self._working = None
            self._requested_depth = None


def run_round_inplace(executor, working_controls, actual_depth=4):
    """Functional spelling of OrdinaryDepthExecutor.run_round_inplace."""
    if not isinstance(executor, OrdinaryDepthExecutor):
        raise TypeError("OrdinaryDepthExecutor required")
    return executor.run_round_inplace(working_controls, actual_depth)
