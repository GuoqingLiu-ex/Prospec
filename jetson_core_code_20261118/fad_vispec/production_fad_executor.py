"""Sequential production FAD executor with one compact device feedback readback.

P0 begin/finish/close and ordinary Target/posterior/clip/KV arithmetic are retained.
Only proposal metadata retention and the original nextroot readback change. The
160-byte int64 packet replaces one 8-byte root readback; CUDA events/DRAM, total
allocator peaks, real-model fidelity and throughput still need actual validation.
This module executes a caller-selected D; the caller owns the full controller.
"""
from __future__ import annotations
import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace

import torch
from .production_depth_executor import ProductionDepthExecutor
from .ordinary_depth_round import OrdinaryDepthExecutor
from .compact_search_feedback import generate_compact_search, Kind, Flag

SCHEMA = "fad_production_full_feedback_executor_v1"
ROUND_SCHEMA = "fad_production_full_feedback_round_v1"
FEEDBACK_SCHEMA = "fad_full_feedback_v1"
ROOT_READ_OLD = "root = int(tree_logits[best, keep - 1].argmax().item())"
ROOT_READ_NEW = 'next_root = tree_logits[best, keep - 1].argmax()\nself._packet = build_feedback_packet(self._compact_result, next_root=next_root,\n    best=best, accepted_draft=accepted, submitted_ids=committed,\n    remaining_tokens=self._remaining_before, stop_ids=self.stops,\n    request_key=self.request_key, state_key=self._state_key)\n_require(isinstance(self._packet, torch.Tensor) and self._packet.dtype == torch.int64\n    and tuple(self._packet.shape) == (20,) and self._packet.device == ids.device\n    and self._packet.is_contiguous(), "feedback packet must be contiguous int64[20] on the request device")\nself._feedback_attempt.update(packet_built=True, logical_packet_bytes=160,\n    logical_baseline_root_bytes=8, additional_logical_D2H_bytes=152)\nself._host_packet = self._packet.tolist()\nself._feedback_attempt["packet_readback_completed"] = True\nroot = _decode_base(self._host_packet)[0]'



def _require(ok, message):
    if not ok:
        raise ValueError(message)


def build_feedback_packet(*args, **kwargs):
    # Lazy to allow CPU fixture injection without importing a GPU implementation.
    from .full_feedback_tensor import build_feedback_packet as build
    return build(*args, **kwargs)


def _decode_base(values):
    _require(type(values) is list and len(values) == 20
        and all(type(v) is int for v in values), "host packet must contain twenty plain integers")
    root, kind, witness, flags = values[:4]
    _require(root >= 0 and kind in {int(k) for k in Kind} and witness >= -1
        and 0 <= flags < 256, "invalid packet root/kind/witness/flags")
    for code, value in zip(values[4:12], values[12:20]):
        _require(code in (0, 1, 2) and (value == -1 if code == 0 else value >= 1),
            "invalid auxiliary kind/value")
    return root, kind, witness, flags


def decode_feedback_packet(values, *, actual_depth, request_key, state_key, receipt,
                           remaining_tokens):
    """Decode only already-read host values; no tensor coercion, export or hash.

    Actual counts come from the completed executor receipt. Conditional entries
    refer only to this state and do not certify an actual counterfactual GPU run.
    EOS/limit and clip-return suppress every auxiliary entry.
    """
    _require(type(actual_depth) is int and 1 <= actual_depth <= 8, "invalid actual depth")
    _require(type(request_key) is str and bool(request_key)
        and type(state_key) is str and bool(state_key), "request/state binding required")
    _require(type(remaining_tokens) is int and 1 <= remaining_tokens <= 512, "remaining budget")
    stop = receipt["stop_reason"]
    _require(stop in (None, "eos", "limit"), "invalid final stop reason")
    a, g, total = (receipt[k] for k in ("accepted_draft", "tree_submitted_including_root", "all_new_output_tokens"))
    bonus = [token for row in receipt["step_rounds"][1:] for token in row["submitted"]]
    _require(type(a) is int and 0 <= a <= actual_depth and type(g) is int
        and 1 <= g <= min(a + 1, remaining_tokens) and type(total) is int
        and total == g + len(bonus) <= remaining_tokens and len(bonus) <= 1
        and all(type(v) is int and v >= 0 for v in bonus)
        and receipt["terminal_bonus_tokens"] == len(bonus), "actual A/G/bonus receipt contradiction")
    _require(stop is not None or (g == a + 1 and not bonus), "nonterminal actual counts")
    _require((total == remaining_tokens if stop == "limit" else True)
        and (total < remaining_tokens if stop is None else True), "actual stop/budget contradiction")
    _require(not bonus or (stop is not None and g == a + 1), "terminal bonus contradicts clip")
    base = None if values is None else _decode_base(values)
    _require(not bonus or base is None or bonus[0] == base[0], "terminal bonus differs from packet root")
    required_flags = int(Flag.PATH_COVERED | Flag.FINITE_SCORES)
    qualified = base is not None and (base[3] & required_flags) == required_flags \
        and not (base[3] & int(Flag.CENSORED | Flag.INSUFFICIENT_EVIDENCE)) \
        and base[1] not in (int(Kind.UNKNOWN), int(Kind.TERMINAL)) and stop is None
    local_kind = "terminal" if stop is not None else (Kind(base[1]).name.lower() if qualified else "unknown")
    entries = []
    for d in range(1, 9):
        kind, value, scope = "unknown", None, "unobserved"
        if d == actual_depth:
            kind, value, scope = "observed", g, "observed_execution"
        elif qualified:
            code, v = values[3 + d], values[11 + d]
            if code:
                _require(v <= min(d + 1, remaining_tokens), "conditional G_tree exceeds depth/budget")
            if code == 1:
                _require(d < actual_depth, "conditional exact requires a shallower depth")
                kind, value, scope = "conditional_exact", v, "same_state_structural_conditional"
            elif code == 2:
                _require(d > actual_depth and base[1] in (1, 2, 3) and v <= g,
                    "conditional upper requires deeper local failure proof and actual G_tree bound")
                kind, value, scope = "conditional_upper", v, "same_state_structural_conditional"
        entries.append(dict(depth=d, kind=kind, value=value, evidence_scope=scope))
    return dict(schema=FEEDBACK_SCHEMA, request_key=request_key, state_key=state_key,
        actual_D=actual_depth, A_raw=a, G_tree=g, G_output=total, terminal_bonus_ids=bonus,
        censored=stop is not None, stop_reason=stop, stop_kind=local_kind,
        evidence_scope="observed_execution", no_next_state_guarantee=True, entries=entries,
        conditional_counterfactuals_are_actual_measurements=False,
        local_evidence=None if base is None else dict(kind_code=base[1], witness_node_id=base[2], flags=base[3],
            usable_local_failure_fact=qualified and base[1] in (1, 2, 3),
            auxiliary_applied=qualified, action_hard_disable_eligible=False),
        packet_readback=None if values is None else list(values))


def verify_fad_source():
    """One bounded correspondence check; inherited setup verifies dependencies."""
    old = textwrap.dedent(inspect.getsource(OrdinaryDepthExecutor._run_tree))
    _require(old.count(ROOT_READ_OLD) == 1, "ordinary root readback source changed")
    expected = old.replace("    " + ROOT_READ_OLD, textwrap.indent(ROOT_READ_NEW, "    "))
    actual = textwrap.dedent(inspect.getsource(ProductionFADExecutor._run_tree))
    dump = lambda source: ast.dump(ast.parse(source), include_attributes=False)
    _require(dump(expected) == dump(actual), "unregistered production FAD tree arithmetic change")
    _require(ProductionFADExecutor.begin_inplace is ProductionDepthExecutor.begin_inplace
        and ProductionFADExecutor.finish_inplace is ProductionDepthExecutor.finish_inplace,
        "production FAD must retain P0 begin/finish")
    return dict(round_changes=["compact proposal references", "one root argmax, one int64[20] packet readback"],
        P0_begin_finish_inherited=True, original_posteriors_and_KV_commit_unchanged=True,
        actual_GPU_validated=False)


class ProductionFADExecutor(ProductionDepthExecutor):
    def __init__(self, *args, **kwargs):
        self._compact_result = self._packet = self._host_packet = self._state_key = None
        self._remaining_before = None
        self._feedback_attempt = None
        self.full_feedback_history = []
        self.feedback_failures = []
        super().__init__(*args, **kwargs)
        self.fad_source = verify_fad_source()

    def _proposal(self, *args, **kwargs):
        _require(self._compact_result is None, "nested compact proposal")
        self._compact_result = generate_compact_search(self.draft, *args, **kwargs,
            actual_depth=self._requested_depth)
        return self._compact_result.tree

    def _release_feedback(self):
        result = self._compact_result
        try:
            if result is not None:
                result.search.release()
        finally:
            self._compact_result = self._packet = self._host_packet = self._state_key = None
            self._remaining_before = self._feedback_attempt = None

    def run_round_inplace(self, actual_depth=4, *, state_key=None):
        self._check_owned()
        _require(self._state_key is None, "nested production FAD round")
        key = state_key if state_key is not None else f"{self.request_key}:round:{len(self.ordinary_history)}"
        _require(type(key) is str and bool(key), "state key required")
        self._state_key = key
        self._remaining_before = self.max_new_tokens - len(self.controls["generated"])
        self._feedback_attempt = dict(actual_depth=actual_depth, state_key=key, packet_built=False,
            packet_readback_completed=False, logical_packet_bytes=0, logical_baseline_root_bytes=0,
            additional_logical_D2H_bytes=0, actual_memcpy_events=None, actual_GPU_kernels=None)
        previous = len(self.ordinary_history)
        failure = None
        try:
            receipt = super().run_round_inplace(actual_depth)
            feedback = decode_feedback_packet(self._host_packet, actual_depth=actual_depth,
                request_key=self.request_key, state_key=key, receipt=receipt,
                remaining_tokens=self._remaining_before)
            receipt.update(schema=ROUND_SCHEMA, full_feedback=feedback,
                feedback_transfer=copy.deepcopy(self._feedback_attempt))
            self.full_feedback_history.append(copy.deepcopy(feedback))
            return receipt
        except BaseException as exc:
            failure = exc
            if self.poisoned or len(self.ordinary_history) > previous:
                self.poisoned = True
                self._working = None
                self.feedback_failures.append(dict(self._feedback_attempt, error=repr(exc)))
                if not self.failures or self.failures[-1].get("stage") != "ordinary_round":
                    self.failures.append(dict(stage="full_feedback", type=type(exc).__name__,
                        message=str(exc), parent_restored=False,
                        recovery="close and discard this request; production has no parent snapshot"))
            raise
        finally:
            try:
                self._release_feedback()
            except BaseException as exc:
                self.poisoned = True
                self._working = None
                self.feedback_failures.append(dict(stage="feedback_release", error=repr(exc)))
                if failure is None:
                    raise

    def execution_receipt(self):
        receipt = super().execution_receipt()
        receipt.update(schema=SCHEMA, fad_source=copy.deepcopy(self.fad_source),
            full_feedback_history=copy.deepcopy(self.full_feedback_history),
            feedback_failures=copy.deepcopy(self.feedback_failures),
            GPU_feedback_references_released=self._compact_result is None and self._packet is None,
            controller_installed=False, actual_cost_table=None,
            feedback_cost_scope="160B packet replaces 8B root readback on nonclip rounds; 152B logical extra, CUDA events/DRAM and whole allocator effect unmeasured")
        return receipt

    def close(self):
        try:
            self._release_feedback()
        finally:
            super().close()

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
        next_root = tree_logits[best, keep - 1].argmax()
        self._packet = build_feedback_packet(self._compact_result, next_root=next_root,
            best=best, accepted_draft=accepted, submitted_ids=committed,
            remaining_tokens=self._remaining_before, stop_ids=self.stops,
            request_key=self.request_key, state_key=self._state_key)
        _require(isinstance(self._packet, torch.Tensor) and self._packet.dtype == torch.int64
            and tuple(self._packet.shape) == (20,) and self._packet.device == ids.device
            and self._packet.is_contiguous(), "feedback packet must be contiguous int64[20] on the request device")
        self._feedback_attempt.update(packet_built=True, logical_packet_bytes=160,
            logical_baseline_root_bytes=8, additional_logical_D2H_bytes=152)
        self._host_packet = self._packet.tolist()
        self._feedback_attempt["packet_readback_completed"] = True
        root = _decode_base(self._host_packet)[0]
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
