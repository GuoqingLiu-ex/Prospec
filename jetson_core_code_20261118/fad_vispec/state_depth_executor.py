"""Offline same-state FAD depth trials, using real Target and author Draft.

Only proposal generation changes D. Live/snapshot constants remain canonical
B30/d3/K8. Full candidate/Target metadata exports are paid diagnostics, not a
zero-transfer deployment. Different-shape Target invariance is NOT asserted.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import torch

from ..focusdraft.real_system_executor import RealExecutor
from .traced_topk import generate_traced_topk
from .feedback import validate_round_trace, build_depth_feedback

SCHEMA = "fad_actual_state_depth_executor_v1"
BASE_EXECUTOR_SHA = "9cbb7999c52c82b9a75f375f2d172f49addd63febf9c260445f9593c8c42ae74"
PREMISES = ("same_state", "search_prefix_preserved", "target_prefix_invariant",
            "no_reexpand", "append_only_fixed_scores", "no_resurrection")


def verify_base_source():
    path=Path(inspect.getfile(RealExecutor))
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != BASE_EXECUTOR_SHA:
        raise ValueError("frozen RealExecutor source changed")
    return {"path":str(path),"sha256":actual}


class StateDepthExecutor(RealExecutor):
    """Explicit offline diagnostics. No candidate vector or FocusPool is used.

    Source/precision IDs are external receipt identifiers, not self-attestations.
    Inherited dependency injection permits CPU mocks; it does not count as GPU
    validation. Snapshot controls acquire diagnostic state IDs but no trace/KV
    references beyond the inherited snapshot graph.
    """
    def __init__(self, spec, inputs, *, source_receipt, precision_receipt,
                 input_provenance=None, **kwargs):
        for name,value in (("source_receipt",source_receipt),("precision_receipt",precision_receipt)):
            if type(value) is not str or not value:
                raise ValueError(name+" must identify the externally saved receipt")
        self.base_source=verify_base_source()
        # JSON roundtrip refuses tensors/nonfinite metadata before model work.
        self.input_provenance=json.loads(json.dumps(input_provenance,allow_nan=False))
        self.source_receipt,self.precision_receipt=source_receipt,precision_receipt
        self._state_namespace=uuid.uuid4().hex
        self._snapshot_sequence=0
        self._requested_depth=None
        self._active_fad=None
        self._trace_record=self._pending_feedback=None
        self.prefill_terminal_feedback=None
        self.depth_trial_history=[]
        super().__init__(spec,inputs,**kwargs)
        if type(self.draft.top_k) is not int or self.draft.top_k != 8:
            raise ValueError("FAD freezes K8")

    def _canonical(self):
        actual=(self.draft.total_tokens,self.draft.depth,self.draft.top_k)
        if actual != (29,3,8) or any(type(x) is not int for x in actual):
            raise RuntimeError("live snapshot Draft constants are not canonical B30/d3/K8")

    def _snapshot(self, controls):
        self._canonical()
        self._snapshot_sequence+=1
        key=f"{self._state_namespace}:{self._snapshot_sequence}"
        controls["fad_state_key"]=key
        controls["fad_snapshot_id"]=key
        return super()._snapshot(controls)

    @torch.inference_mode()
    def begin(self):
        parent=super().begin()
        try:
            self._canonical()
            if parent.controls["stop_reason"] is not None:
                c=parent.controls
                trace=dict(schema='fad_actual_author_topk_trace_v1',actual_depth=0,
                    beam_width=8,budget_cap=30,root_token=c['root'],
                    token_ids=[],parent_ids=[],scores=[],generation_depth=[],checkpoints=[],selected_ids=[])
                record=self._envelope(c,trace,dict(state_key=c['fad_state_key'],kind='prefill_terminal',
                    submitted_ids=list(c['generated']),stop_reason=c['stop_reason']),
                    self.prompt,self.max_new_tokens)
                self.prefill_terminal_feedback=dict(feedback_input=record,
                    structural_feedback=build_depth_feedback(validate_round_trace(record),list(range(1,9))))
            return parent
        except BaseException as exc:
            self.poisoned=True
            self.failures.append(dict(stage='begin_FAD_diagnostic',type=type(exc).__name__,message=str(exc)))
            self.close()
            raise

    @torch.inference_mode()
    def restore(self,parent):
        self._canonical()
        return super().restore(parent)

    @torch.inference_mode()
    def step(self,parent,*,actual_depth=4):
        if type(actual_depth) is not int or not 1<=actual_depth<=8:
            raise ValueError("actual_depth must be an integer in [1,8]")
        if self._requested_depth is not None:
            raise RuntimeError("nested FAD step is forbidden")
        self._canonical()
        self._requested_depth=actual_depth
        self._active_fad=self._trace_record=self._pending_feedback=None
        try:
            child,trial=super().step(parent,None)
            # Diagnostic export/validation occurred inside _run_tree, hence in
            # inherited trial ledger and exception/parent-restore transaction.
            if self._active_fad is None:
                raise RuntimeError("missing completed FAD diagnostic")
            trial.update(copy.deepcopy(self._active_fad))
            self.depth_trial_history.append(dict(actual_depth=actual_depth,
                tree_nodes=trial['step_rounds'][0]['tree_nodes'],
                parent_state_key=trial['feedback_input']['binding']['state_key'],
                child_state_key=child.controls['fad_state_key']))
            return child,trial
        except BaseException:
            if self.failures:
                self.failures[-1]['fad_trace']=copy.deepcopy(self._trace_record)
                self.failures[-1]['fad_pending_feedback']=copy.deepcopy(self._pending_feedback)
            raise
        finally:
            self._requested_depth=None
            self._active_fad=self._trace_record=self._pending_feedback=None

    def _proposal(self,*args,**kwargs):
        self._canonical()
        got=generate_traced_topk(self.draft,*args,**kwargs,
            actual_depth=self._requested_depth,budget_cap=30,trace=True)
        self._canonical()
        self._trace_record=got.trace.to_cpu_record()
        e=self._trace_record['export_receipt']
        self.ledger.add('fad_trace_explicit_export',
            calls=e['blocking_device_to_host_calls'],
            read_bytes=e['logical_device_read_bytes'],write_bytes=e['logical_cpu_write_bytes'],
            host_sync_boundaries=e['host_boundary_calls'])
        self.ledger.add('fad_trace_CPU_metadata',calls=1)
        return got.tree

    def _envelope(self,c,trace,verification,base,remaining):
        return dict(schema='fad_round_feedback_input_v1',
            binding=dict(request_key=self.request_key,state_key=c['fad_state_key'],
                snapshot_id=c['fad_snapshot_id'],source_receipt=self.source_receipt,
                precision_receipt=self.precision_receipt,cache_before=base,
                rope_delta=int(c['delta']),remaining_tokens=remaining,stop_ids=list(self.stops),
                input_provenance=copy.deepcopy(self.input_provenance),
                state_identity_scope='actual executor-bound snapshot ID; not a KV-byte hash'),
            trace=copy.deepcopy(trace),verification=verification,
            premises={name:dict(status='missing',evidence_ids=[]) for name in PREMISES})

    def _complete_feedback(self,c,event):
        p=self._pending_feedback;t=self._trace_record
        ids=t['selected_ids'];tree=self._active_tree_diagnostic
        if tree['tokens']['values'][0] != [t['root_token']]+[t['token_ids'][i-1] for i in ids[1:]]:
            raise RuntimeError('trace selection does not map to actual verification tokens')
        mask=tree['ancestor_mask']['values'][0][0]
        contexts={str(node):dict(state_key=p['parent_state_key'],
            ancestor_ids=[ids[j] for j,allowed in enumerate(mask[row]) if allowed],
            position_ids=[p['position_ids'][axis][0][row] for axis in range(3)])
            for row,node in enumerate(ids)}
        raw=[ids[i] for i in p['raw_selected_rows']]
        retrieve=[[ids[i] for i in row if i>=0] for row in tree['retrieve']['values']]
        bonus=[] if c['rounds'][-1]['kind']=='tree' else list(c['rounds'][-1]['submitted'])
        v=dict(state_key=p['parent_state_key'],kind='tree',verified_ids=ids[:],
            node_contexts=contexts,next_ids={str(node):p['next_ids'][row] for row,node in enumerate(ids)},
            retrieve_paths=retrieve,selected_raw_ids=raw,accepted_draft=event['accepted_draft'],
            submitted_ids=list(event['submitted']),
            selected_committed_ids=[ids[i] for i in event['selected_tree_indices']],
            terminal_bonus_ids=bonus,stop_reason=c['stop_reason'])
        record=self._envelope(c,t,v,p['cache_before'],p['remaining_tokens'])
        validated=validate_round_trace(record)
        self.ledger.add('fad_CPU_feedback_validation',calls=1)
        self._active_fad=dict(schema=SCHEMA,actual_depth=self._requested_depth,
            canonical_snapshot_budget=dict(B=30,d=3,K=8),trace=copy.deepcopy(t),
            feedback_input=record,structural_feedback=build_depth_feedback(validated,list(range(1,9))),
            diagnostic_only=True,actual_counterfactual_invariance_measured=False)
        self._canonical()

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
        self._active_tree_diagnostic = self._tree_diagnostic(tokens, retrieve, depths, allowed_tree)
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
        target_next = verified.logits[0].argmax(-1)
        self.ledger.add("fad_all_node_argmax", calls=1)
        next_cpu = self.ledger.transfer(target_next, "cpu", "fad_Target_next_IDs")
        positions_cpu = self.ledger.transfer(position_ids, "cpu", "fad_semantic_positions")
        self.ledger.add("fad_prediction_position_python_metadata", calls=1)
        self._pending_feedback = dict(
            next_ids=next_cpu.tolist(), position_ids=positions_cpu.tolist(),
            parent_state_key=c["fad_state_key"], snapshot_id=c["fad_snapshot_id"],
            cache_before=base, rope_delta=c["delta"],
            remaining_tokens=self.max_new_tokens-len(c["generated"]))
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
        old_output_len = len(c["generated"])
        c["ids"] = torch.cat((ids, ids.new_tensor([committed])), 1)
        c["generated"].extend(committed)
        event = dict(kind="tree", tree_nodes=nodes,
            budget_action=self.budget.action_id if self._requested_depth==4 else f"fad_D{self._requested_depth}",
            budget=dict(B=nodes, d=self._requested_depth-1, K=8, total_tokens=nodes-1),
            accepted_draft=accepted, committed_draft=keep - 1, submitted=committed,
            target_cache_before=base, target_cache_after=b._length(self.cache),
            discarded_tree_nodes=nodes - keep, budget_or_eos_clipped=keep != len(proposed),
            pending_tokens=[], selected_tree_indices=selected.tolist(), semantic_rope_base=base + c["delta"],
            selectedH_waiting_sync=False, visual_m=1.0)
        self._pending_feedback["raw_selected_rows"] = self._active_tree_diagnostic['retrieve']['values'][best][:accepted+1]
        self.ledger.add("fad_raw_selected_CPU_metadata", calls=1)
        c["rounds"].append(event); c["budget_trace"].append(event["budget"])
        # The logits for selected y_i predict y_(i+1). The last prediction is
        # retained as diagnostic even at EOS; it is not an invented output token.
        c["top2"].extend(self._top2(tree_logits[best, :keep],
            range(old_output_len + 1, old_output_len + keep + 1), "tree", event["selected_tree_indices"]))
        c["stop_reason"], c["pending_tokens"] = reason, []
        if reason:
            c["selected_hidden_waiting"] = None
            self._complete_feedback(c, event)
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
        self._complete_feedback(c, event)

    def execution_receipt(self):
        receipt=super().execution_receipt()
        receipt.update(schema=SCHEMA,base_executor_source=self.base_source,
            source_receipt=self.source_receipt,precision_receipt=self.precision_receipt,
            input_provenance=copy.deepcopy(self.input_provenance),
            canonical_snapshot_budget=dict(B=30,d=3,K=8),
            actual_depths=[r['actual_depth'] for r in self.depth_trial_history],
            depth_trial_history=copy.deepcopy(self.depth_trial_history),
            actual_verify_nodes=sum(r['tree_nodes'] for r in self.execution_rounds if r.get('kind')=='tree'),
            prefill_terminal_feedback=copy.deepcopy(self.prefill_terminal_feedback),
            scope='offline actual multi-depth trials including losers, snapshot clones, full trace exports and Target argmax metadata; not deployment cost, GPU counter or throughput')
        return receipt
