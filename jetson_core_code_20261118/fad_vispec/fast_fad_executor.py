"""Independent low-overhead FAD executor candidate; CPU validation only.

Keep all frozen mathematical operations, safety checks and isolated histories.
Only reuse one request-owned 160B packet and shallow-copy a flat scalar transfer
receipt. This does not remove P0/full audit histories or claim net speed/memory.
The production packet implementation is CUDA-only; tests explicitly patch it.
"""
from __future__ import annotations
import ast
import copy
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import SimpleNamespace

import torch
from . import full_feedback_fused as fused
from . import production_fad_executor as base
from .production_fad_executor import (_require, _decode_base, decode_feedback_packet,
                                      ROUND_SCHEMA, ProductionFADExecutor)
from .production_depth_executor import ProductionDepthExecutor

SCHEMA = 'fad_fast_executor_candidate_v1'
EXPECTED = {
    'production_fad_executor.py': 'de1e4171085dc2cbc38fcb56c1bc366f93391ac0dbfc431fa4ca28776c657a76',
    'production_depth_executor.py': '40aa233fb839e97c78f9ec6cf4d7b49aca3d7864920a7746740c1b7c56ec5233',
    'ordinary_depth_round.py': 'bbb9515b0a8502298b77ead0c94736518e005acbb14de7081fbd0a6040bb46ab',
    'full_feedback_fused.py': 'e858f721cc5fc6909b0a69300174eecc1eb9067f53fd61c6478a1d83906f4c01',
}
EDITS = {
    '_run_tree': [('self._packet = build_feedback_packet(', 'self._packet = self._build_packet(')],
    'run_round_inplace': [
        ('receipt = super().run_round_inplace(actual_depth)',
         'receipt = ProductionDepthExecutor.run_round_inplace(self, actual_depth)'),
        ('feedback_transfer=copy.deepcopy(self._feedback_attempt)',
         'feedback_transfer=dict(self._feedback_attempt)'),
    ],
}


@lru_cache(maxsize=1)
def verify_fast_source():
    """Cold exact dependency and registered AST check; never a GPU claim."""
    root = Path(__file__).parent
    for name, expected in EXPECTED.items():
        _require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected,
                 'frozen fast-executor dependency changed: '+name)
    dump = lambda source: ast.dump(ast.parse(source), include_attributes=False)
    for name, edits in EDITS.items():
        expected = textwrap.dedent(inspect.getsource(getattr(ProductionFADExecutor,name)))
        for old,new in edits:
            _require(expected.count(old)==1, 'registered fast executor edit differs')
            expected = expected.replace(old,new)
        actual = textwrap.dedent(inspect.getsource(getattr(FastFADExecutor,name)))
        _require(dump(expected)==dump(actual), 'unregistered fast executor change: '+name)
    _require(FastFADExecutor.begin_inplace is ProductionDepthExecutor.begin_inplace
             and FastFADExecutor.finish_inplace is ProductionDepthExecutor.finish_inplace,
             'fast executor must inherit frozen begin/finish')
    return dict(schema=SCHEMA, source_sha256=dict(EXPECTED), registered_edits=copy.deepcopy(EDITS),
                GPU_validated=False, actual_speed_or_memory_benefit=None)


class FastFADExecutor(ProductionFADExecutor):
    """Same public executor API; replace only the caller's executor class.

    Buffer allocated at the first nonclip packet, reused after each existing
    synchronous tolist has consumed it, and dropped on poison/close. No extra
    Tensor readback, current-stream change, safety bypass or CPU fallback.
    """
    def __init__(self, *args, **kwargs):
        self._packet_buffer = None
        self._buffer_allocations = self._buffer_reuses = self._packet_builds = 0
        self._fast_source = verify_fast_source()
        super().__init__(*args, **kwargs)

    def _build_packet(self, result, **kwargs):
        device = kwargs['next_root'].device
        if self._packet_buffer is None:
            self._packet_buffer = torch.empty((20,), dtype=torch.int64, device=device)
            self._buffer_allocations += 1
        else:
            _require(self._packet_buffer.device == device, 'packet buffer request device changed')
            self._buffer_reuses += 1
        # Fused retains its full metadata, layout and alias guards, and has no
        # CPU/Torch fallback. The existing tree body still validates the result.
        out = fused.build_feedback_packet(result, out=self._packet_buffer, **kwargs)
        _require(out is self._packet_buffer, 'fused packet must use the caller buffer')
        self._packet_builds += 1
        return out

    def _release_feedback(self):
        try:
            super()._release_feedback()
        except BaseException:
            self._packet_buffer = None
            raise
        finally:
            if self.poisoned:
                self._packet_buffer = None

    def close(self):
        try:
            super().close()
        finally:
            self._packet_buffer = None

    def execution_receipt(self):
        receipt = super().execution_receipt()
        # The old flag must not claim all GPU references gone while the
        # request buffer is deliberately still alive between successful rounds.
        round_released = receipt['GPU_feedback_references_released']
        receipt['feedback_round_references_released'] = round_released
        receipt['GPU_feedback_references_released'] = round_released and self._packet_buffer is None
        receipt['fast_executor'] = dict(schema=SCHEMA, source=copy.deepcopy(self._fast_source),
            packet_buffer_allocations=self._buffer_allocations,
            packet_buffer_reuses=self._buffer_reuses, successful_packet_builds=self._packet_builds,
            live_packet_buffer_bytes=0 if self._packet_buffer is None else 160,
            packet_buffer_released=self._packet_buffer is None,
            transfer_receipt_copy='independent shallow dict of generated immutable scalars',
            all_public_rounds_and_feedback_histories_retained=True,
            scope='160B request workspace and allocation count only; actual allocated/reserved peaks and throughput unmeasured')
        return receipt

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
                receipt = ProductionDepthExecutor.run_round_inplace(self, actual_depth)
                feedback = decode_feedback_packet(self._host_packet, actual_depth=actual_depth,
                    request_key=self.request_key, state_key=key, receipt=receipt,
                    remaining_tokens=self._remaining_before)
                receipt.update(schema=ROUND_SCHEMA, full_feedback=feedback,
                    feedback_transfer=dict(self._feedback_attempt))
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
            self._packet = self._build_packet(self._compact_result, next_root=next_root,
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
