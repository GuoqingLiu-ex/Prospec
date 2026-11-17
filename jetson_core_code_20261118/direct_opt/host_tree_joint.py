"""Independent generation copy: only registered host metadata insertion points.

Original Draft/Target calls, GPU checks, posterior, clipping and EOS are retained.
No GPU metadata is read to reconstruct the author CPU tree/retrieve receipt.
"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import torch
import vispec_joint_adapter as original
from vispec_joint_adapter import (bridge, BASELINE_BUDGET, BMAX, SPATIAL_MERGE_SIZE,
    budget_action, validate_budget, JointResult, prepare_draft_visual_context,
    first_tree_imadpt, apply_tree_budget, _context_dict_legacy, _context_dict, _grid_raw)

FROZEN_JOINT_SHA256='6bd51c2990b8460c40315d780f897082509c1a87dab04494f6db99e7d5b38168'
REGISTERED_EDITS = [('    diagnostic_actual_imadpt=False,\n',
  '    diagnostic_actual_imadpt=False,\n    host_tree_broker=None,\n'),
 ('    del choose_m  # request-level m is fixed; reserved for a later trained head.',
  '    if host_tree_broker is None:\n'
  '        raise ValueError("explicit request-local host tree broker required")\n'
  '    del choose_m  # request-level m is fixed; reserved for a later trained head.'),
 ('            tokens, retrieve, tree_mask, depths = tree',
  '            tokens, retrieve, tree_mask, depths = tree\n'
  '            host_tree_broker.capture(tree, cache, int(ids.shape[1]), delta)'),
 ('            position_ids = (base + delta + depths).view(1, 1, nodes).expand(3, 1, nodes)',
  '            position_ids = (base + delta + depths).view(1, 1, nodes).expand(3, 1, nodes)\n'
  '            host_tree_broker.bind_forward(cache, tokens, mask, position_ids, retrieve, base, '
  'delta)'),
 ('            selected = retrieve[best, :keep]',
  '            selected = retrieve[best, :keep]\n'
  '            accepted_tuple = host_tree_broker.bind_acceptance(best, keep, selected)'),
 ('"selected_tree_indices": selected.tolist(),', '"selected_tree_indices": list(accepted_tuple),')]

def validate_generation_copy():
    path=Path(original.__file__)
    source=path.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=FROZEN_JOINT_SHA256:
        raise RuntimeError('frozen joint source changed; re-audit required')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='generate_joint')
    expected=ast.get_source_segment(source,node)
    for before,after in REGISTERED_EDITS:
        if expected.count(before)!=1:raise RuntimeError('registered insertion no longer unique')
        expected=expected.replace(before,after)
    actual=next(n for n in ast.parse(Path(__file__).read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='generate_joint')
    if ast.dump(ast.parse(expected).body[0],include_attributes=False)!=ast.dump(actual,include_attributes=False):
        raise RuntimeError('generation copy changed outside registered metadata edits')
    return dict(original_file=str(path),original_sha256=FROZEN_JOINT_SHA256,
        registered_edits=len(REGISTERED_EDITS),model_math_and_call_order_unchanged=True,
        scope='static source contract; actual GPU output/full-round equivalence remains unverified')


def generate_joint(
    spec,
    inputs,
    max_new_tokens=1024,
    *,
    visual_m=1.0,
    tree_budget=None,
    choose_m=None,
    choose_budget=None,
    on_round=None,
    eos_token_ids=None,
    capacity=None,
    spatial_merge_size=SPATIAL_MERGE_SIZE,
    merge_operator="cosine",
    context_extractor="new",
    diagnostic_actual_imadpt=False,
    host_tree_broker=None,
):
    """Native Target + author Draft with optional static m and (B,d,K) budget."""
    if host_tree_broker is None:
        raise ValueError("explicit request-local host tree broker required")
    del choose_m  # request-level m is fixed; reserved for a later trained head.
    bridge._version()
    target, draft = spec.base_model, spec.spec_layer
    if tree_budget is None:
        tree_budget = BASELINE_BUDGET
    if isinstance(tree_budget, str):
        tree_budget = budget_action(tree_budget)
    tree_budget = validate_budget(tree_budget)
    values = bridge._values(inputs, max_new_tokens, capacity or spec.capacity, BMAX)
    ids = values["input_ids"]
    prompt = int(ids.shape[1])
    counts = bridge._counts()
    counts.update(
        imadpt_calls=0,
        imadpt_input_tokens=0,
        imadpt_overridden_calls=0,
        budget_switch_calls=0,
        merge_visual_tokens_in=0,
        merge_visual_tokens_out=0,
    )
    draft.reset_kv()
    if hasattr(draft, "last_img_hidden"):
        draft.last_img_hidden = None
    if max_new_tokens == 0:
        return JointResult(
            [], "limit", prompt, 0, 0, [], True, dict(counts), [],
            dict(getattr(target, "_vispec_precision", {})),
        )
    stops = bridge._stops(target, eos_token_ids)
    ops = bridge._author()
    generated = []
    rounds = []
    budget_trace = []
    visual_mask = bridge._visual_mask(target, values)
    current_budget = tree_budget
    actual_imadpt = {"calls": 0, "actual_input_lengths": []}
    with bridge._visual_counter(target, counts), bridge._draft_counter(draft, counts):
        out, delta, features = bridge._prefill(target, values, counts, capture=True)
        cache = out.past_key_values
        timing = bridge._start(ids)
        root = int(out.logits[0, -1].argmax().item())
        reason = "eos" if root in stops else ("limit" if max_new_tokens == 1 else None)
        if reason:
            return JointResult(
                [root], reason, prompt, 1, bridge._length(cache), [root], True,
                dict(counts),
                [{"kind": "prefill_terminal", "submitted": [root], "pending_tokens": [root]}],
                dict(getattr(target, "_vispec_precision", {})),
                visual_context={},
                budget_trace=[],
                imadpt={},
            )
        context = prepare_draft_visual_context(
            features["inputs_embeds"], visual_mask, _grid_raw(values),
            spatial_merge_size, visual_m, merge_operator=merge_operator,
        )
        counts["merge_visual_tokens_in"] += context.original_count
        counts["merge_visual_tokens_out"] += context.kept_count
        selected_hidden_waiting = features["hidden"]
        waiting_sync = True
        first_tree = True
        while True:
            if choose_budget is not None:
                current_budget = validate_budget(choose_budget({
                    "generated": list(generated),
                    "root": root,
                    "rounds": list(rounds),
                    "visual_m": visual_m,
                }))
                counts["budget_switch_calls"] += 1
            apply_tree_budget(draft, current_budget)
            counts["budget_switch_calls"] += 1
            nodes = current_budget.nodes_including_root
            if first_tree:
                with first_tree_imadpt(draft, context, diagnostic_actual_imadpt) as (actual_imadpt, imadpt_state):
                    tree = draft.topK_genrate(
                        selected_hidden_waiting,
                        torch.cat((ids, ids.new_tensor([[root]])), 1),
                        target.get_output_embeddings(),
                        None,
                        inputs_embeds=features["inputs_embeds"],
                        image_mask=visual_mask,
                    )
                counts["imadpt_calls"] += imadpt_state["calls"]
                counts["imadpt_input_tokens"] += sum(imadpt_state["input_lengths"])
                counts["imadpt_overridden_calls"] += int(imadpt_state["overridden"])
                first_tree = False
                if context.original_count == 0 and visual_mask is not None and bool(visual_mask.any()):
                    raise RuntimeError("fail closed: visual span present but empty context")
                if context.original_count > 0 and float(visual_m) < 1.0:
                    if (not imadpt_state["overridden"]) or imadpt_state["calls"] != 1:
                        raise RuntimeError("fail closed: m<1 must replace imadpt input exactly once")
                if context.original_count > 0 and float(visual_m) >= 1.0 and imadpt_state["overridden"]:
                    raise RuntimeError("fail closed: m=1/VISPEC_ORIG must not override")
                if diagnostic_actual_imadpt and context.original_count > 0:
                    if actual_imadpt.get("calls", 0) < 1:
                        raise RuntimeError("fail closed: visual imadpt did not run")
                    if float(visual_m) < 1.0:
                        if actual_imadpt["calls"] != 1 or actual_imadpt["actual_input_lengths"] != [context.kept_count]:
                            raise RuntimeError("fail closed: actual imadpt length != kept K")
                    elif actual_imadpt["actual_input_lengths"] != [context.original_count]:
                        raise RuntimeError("fail closed: m=1 actual imadpt length != N")
                del features
            else:
                tree = draft.topK_genrate(
                    selected_hidden_waiting,
                    torch.cat((ids, ids.new_tensor([[root]])), 1),
                    target.get_output_embeddings(),
                    None,
                )
            waiting_sync = False
            counts["draft_tree_calls"] += 1
            tokens, retrieve, tree_mask, depths = tree
            host_tree_broker.capture(tree, cache, int(ids.shape[1]), delta)
            base = int(ids.shape[1])
            if bridge._length(cache) != base or tokens.shape != (1, nodes) or int(tokens[0, 0]) != root:
                raise RuntimeError("tree root/size/native committed cache boundary mismatch")
            retrieve = retrieve.to(ids.device)
            tree_mask = tree_mask.to(ids.device)
            depths = depths.to(ids.device)
            if tree_mask.shape != (1, 1, nodes, nodes) or depths.shape != (nodes,):
                raise RuntimeError("author tree mask/depth shape mismatch")
            allowed_tree = tree_mask.bool()
            if not bool(allowed_tree[0, 0].diagonal().all()) or not bool(allowed_tree[0, 0, :, 0].all()):
                raise RuntimeError("tree nodes must attend to self and root")
            if not torch.equal(depths, allowed_tree[0, 0].sum(-1).long() - 1):
                raise RuntimeError("tree RoPE depth disagrees with ancestor mask")
            allowed = torch.cat(
                (torch.ones((1, 1, nodes, base), device=ids.device, dtype=torch.bool), allowed_tree), -1
            )
            mask = torch.zeros(allowed.shape, device=ids.device, dtype=torch.float32).masked_fill(
                ~allowed, torch.finfo(torch.float32).min
            )
            position_ids = (base + delta + depths).view(1, 1, nodes).expand(3, 1, nodes)
            host_tree_broker.bind_forward(cache, tokens, mask, position_ids, retrieve, base, delta)
            with bridge._capture_features(target) as feature:
                verified = target(
                    input_ids=tokens.to(ids.device),
                    past_key_values=cache,
                    attention_mask=mask,
                    position_ids=position_ids,
                    use_cache=True,
                    logits_to_keep=0,
                    return_dict=True,
                )
            counts["target_verify_calls"] += 1
            counts["target_verify_query_tokens"] += nodes
            cache = verified.past_key_values
            if bridge._length(cache) != base + nodes or "hidden" not in feature:
                raise RuntimeError("native verification did not expose complete tree KV/feature")
            padded_tokens = torch.cat((tokens, tokens.new_full((1, 1), -1)), 1)
            if retrieve.ndim != 2 or bool((retrieve < -1).any()) or bool((retrieve >= nodes).any()):
                raise RuntimeError("invalid retrieve indices")
            candidates = padded_tokens[0, retrieve]
            tree_logits = verified.logits[0, retrieve]
            best, accepted, _ = ops.evaluate_posterior(tree_logits, candidates, None)
            best, accepted = int(best), int(accepted)
            proposed = candidates[best, : accepted + 1].tolist()
            if not proposed or proposed[0] != root or any(token < 0 for token in proposed):
                raise RuntimeError("greedy posterior selected invalid root/path")
            committed, reason = bridge._clip(proposed, max_new_tokens - len(generated), stops)
            keep = len(committed)
            selected = retrieve[best, :keep]
            accepted_tuple = host_tree_broker.bind_acceptance(best, keep, selected)
            bridge._compact(cache, selected, base, nodes)
            counts["target_compaction_calls"] += 1
            counts["target_compaction_selected_tokens"] += keep
            ids = torch.cat((ids, ids.new_tensor([committed])), 1)
            generated.extend(committed)
            event = {
                "kind": "tree",
                "tree_nodes": nodes,
                "budget_action": current_budget.action_id,
                "budget": {
                    "B": current_budget.nodes_including_root,
                    "d": current_budget.expansions,
                    "K": current_budget.beam_width,
                    "total_tokens": current_budget.total_tokens,
                },
                "accepted_draft": accepted,
                "committed_draft": keep - 1,
                "submitted": committed,
                "target_cache_before": base,
                "target_cache_after": bridge._length(cache),
                "discarded_tree_nodes": nodes - keep,
                "budget_or_eos_clipped": keep != len(proposed),
                "pending_tokens": [],
                "selected_tree_indices": list(accepted_tuple),
                "semantic_rope_base": base + delta,
                "selectedH_waiting_sync": False,
                "visual_m": visual_m,
            }
            rounds.append(event)
            budget_trace.append(event["budget"])
            if on_round is not None:
                on_round(event)
            if reason:
                result = JointResult(
                    list(generated), reason, prompt, len(generated),
                    bridge._length(cache), [], True, dict(counts), rounds,
                    dict(getattr(target, "_vispec_precision", {})),
                    visual_context=(_context_dict_legacy if context_extractor=="legacy" else _context_dict)(context),
                    budget_trace=budget_trace,
                    imadpt={
                        "calls": counts["imadpt_calls"],
                        "overridden": counts["imadpt_overridden_calls"],
                        "input_tokens": counts["imadpt_input_tokens"],
                        "actual_input_lengths": actual_imadpt.get("actual_input_lengths", []),
                        "actual_calls": actual_imadpt.get("calls", 0),
                    },
                )
                if timing is not None:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    result._timing_events = (timing, end)
                return result
            root = int(tree_logits[best, keep - 1].argmax().item())
            reason = "eos" if root in stops else (
                "limit" if len(generated) + 1 == max_new_tokens else None
            )
            if reason:
                generated.append(root)
                rounds.append({
                    "kind": "terminal_target_token",
                    "submitted": [root],
                    "target_cache_after": bridge._length(cache),
                    "pending_tokens": [root],
                    "selectedH_waiting_sync": True,
                    "known_root": root,
                })
                result = JointResult(
                    list(generated), reason, prompt, len(generated),
                    bridge._length(cache), [root], True, dict(counts), rounds,
                    dict(getattr(target, "_vispec_precision", {})),
                    visual_context=(_context_dict_legacy if context_extractor=="legacy" else _context_dict)(context),
                    budget_trace=budget_trace,
                    imadpt={
                        "calls": counts["imadpt_calls"],
                        "overridden": counts["imadpt_overridden_calls"],
                        "input_tokens": counts["imadpt_input_tokens"],
                        "actual_input_lengths": actual_imadpt.get("actual_input_lengths", []),
                        "actual_calls": actual_imadpt.get("calls", 0),
                    },
                )
                if timing is not None:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    result._timing_events = (timing, end)
                return result
            selected_hidden_waiting = feature["hidden"][:, selected]
            waiting_sync = True
            del waiting_sync
