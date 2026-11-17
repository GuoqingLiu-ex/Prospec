"""FAD diagnostic trace around the unchanged author proposal/search arithmetic.

All trace tensors remain on the proposal device. Explicit to_cpu_record() is
an offline diagnostic with extra copies/synchronization; it is not deployment.
The original tree builder's CPU transfers still occur. No Target is called.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from ..focusdraft.focus_replay import canonical_ast_dump

SCHEMA = "fad_actual_author_topk_trace_v1"
AUTHOR_SHA256 = "bc2fd29bb7629fe0662f4f4cca7e228a2994c5980f8778141ca5f46c0384481d"
AUTHOR_AST_SHA256 = "4fc5f62bad6df0eadd74b20b507b9f57a370dde3568a8392c11b851ac7cecf85"
AUTHOR_RELATIVE = "coordination/decoder_matched_vispec_20261001/review_source/ViSpec-49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/model/cnets_ours.py"
AUTHOR_PATH = Path(__file__).resolve().parents[2] / AUTHOR_RELATIVE
_MISSING = object()


def _hash_node(node):
    return hashlib.sha256(canonical_ast_dump(node).encode()).hexdigest()


@lru_cache(maxsize=None)
def _check_loaded_author(function):
    """Cached source check holds only a Python function, never a live Draft."""
    function = inspect.unwrap(function)
    node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    if _hash_node(node) != AUTHOR_AST_SHA256:
        raise ValueError("loaded topK_genrate AST differs from frozen author")
    return True


def verify_source_contract(author_path=None):
    """Explicit setup check; no Torch model construction or device operations."""
    path = Path(author_path) if author_path is not None else AUTHOR_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != AUTHOR_SHA256:
        raise ValueError("author source SHA differs")
    original = next(n for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == "topK_genrate")
    if _hash_node(original) != AUTHOR_AST_SHA256:
        raise ValueError("author topK AST differs")
    node = ast.parse(textwrap.dedent(inspect.getsource(_traced_author))).body[0]
    node = copy.deepcopy(node)
    node.name = "topK_genrate"
    node.args.kwonlyargs = []
    node.args.kw_defaults = []
    class RemoveTrace(ast.NodeTransformer):
        def visit_Expr(self, item):
            value = item.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name) and value.func.value.id == "_trace"):
                return None
            return self.generic_visit(item)
    node = RemoveTrace().visit(node)
    if _hash_node(node) != AUTHOR_AST_SHA256:
        raise ValueError("traced function changed author arithmetic/control AST")
    return {"author_sha256": AUTHOR_SHA256, "author_topk_ast_sha256": AUTHOR_AST_SHA256,
            "removed_instrumentation_only": ["initial", "expansion", "selection"],
            "scope": "source mechanics; not GPU numerical or performance validation"}


@lru_cache(maxsize=1)
def _setup_source_guard():
    return verify_source_contract()


@dataclass(frozen=True)
class DepthCheckpoint:
    D: int
    pool_size: int
    frontier_ids: torch.Tensor
    expanded_ids: torch.Tensor
    eliminated_ids: torch.Tensor


@dataclass(frozen=True)
class CandidateTrace:
    actual_depth: int
    budget_cap: int
    beam_width: int
    root_token: torch.Tensor
    token_ids: torch.Tensor
    parent_ids: torch.Tensor
    scores: torch.Tensor
    generation_depth: torch.Tensor
    checkpoints: tuple
    selected_ids: torch.Tensor
    ancestor_closed: torch.Tensor
    finite_scores: torch.Tensor

    def _named_tensors(self):
        for name in ("root_token", "token_ids", "parent_ids", "scores", "generation_depth",
                     "selected_ids", "ancestor_closed", "finite_scores"):
            yield name, getattr(self, name)
        for i, cp in enumerate(self.checkpoints):
            for name in ("frontier_ids", "expanded_ids", "eliminated_ids"):
                yield f"checkpoint_{i + 1}_{name}", getattr(cp, name)

    def storage_receipt(self):
        """Actual named backing metadata only, not allocator peak or DRAM."""
        tensors = list(self._named_tensors())
        unique = {}
        for _, value in tensors:
            storage = value.untyped_storage()
            unique[(str(value.device), storage.data_ptr())] = storage.nbytes()
        return {"schema": SCHEMA, "actual_depth": self.actual_depth,
                "pool_size_nonroot": self.scores.numel(),
                "actual_verify_nodes": self.selected_ids.numel(),
                "score_dtype": str(self.scores.dtype),
                "named_logical_bytes": sum(t.numel() * t.element_size() for _, t in tensors),
                "unique_backing_bytes": sum(unique.values()),
                "unique_storage_count": len(unique),
                "device": str(self.scores.device),
                "new_host_tensor_reads_in_capture": 0,
                "trace_gpu_work": "root clone, parent expansion, node/checkpoint index construction, fixed-size sort for eliminated IDs, ancestor/finite reductions",
                "original_host_transfers_removed": False,
                "tie_order": "original torch.topk; no deterministic total-order certificate",
                "contains_target_logits_or_cache": False,
                "scope": "retained named tensor backing, excludes transient Torch/kernel workspace, Python objects and original tree; not DRAM or measured peak"}

    def to_cpu_record(self):
        """Explicit diagnostic export. No exporter is invoked by generation."""
        host = {}
        device_copies = device_bytes = 0
        for name, value in self._named_tensors():
            if value.device.type != "cpu":
                device_copies += 1
                device_bytes += value.numel() * value.element_size()
            host[name] = value.detach().to("cpu", non_blocking=False)
        result = {"schema": SCHEMA, "actual_depth": self.actual_depth,
                  "budget_cap": self.budget_cap, "beam_width": self.beam_width,
                  "root_token": int(host["root_token"][0]),
                  "score_dtype": str(self.scores.dtype),
                  "scores_bytes_hex": bytes(host["scores"].contiguous().view(torch.uint8).tolist()).hex(),
                  "ancestor_closed": bool(host["ancestor_closed"]),
                  "finite_scores": bool(host["finite_scores"]),
                  "tie_order": "original torch.topk; certificate tie premise unproved"}
        for name in ("token_ids", "parent_ids", "scores", "generation_depth", "selected_ids"):
            result[name] = host[name].tolist()
        result["checkpoints"] = [{"D": cp.D, "pool_size": cp.pool_size,
             **{name: host[f"checkpoint_{cp.D}_{name}"].tolist()
                for name in ("frontier_ids", "expanded_ids", "eliminated_ids")}}
             for cp in self.checkpoints]
        result["export_receipt"] = {
            "blocking_device_to_host_calls": device_copies,
            "logical_device_read_bytes": device_bytes, "logical_cpu_write_bytes": device_bytes,
            "host_boundary_calls": device_copies,
            "scope": "explicit diagnostic .to boundaries and logical payload, not physical transfer/kernel count or latency; CPU list/hex conversion additional"}
        return result


@dataclass(frozen=True)
class TracedTopKResult:
    tree: tuple
    trace: CandidateTrace | None


class _TraceCollector:
    def __init__(self, actual_depth, budget_cap):
        self.actual_depth, self.budget_cap = actual_depth, budget_cap
        self.frontiers = []

    def initial(self, root_token, indices):
        self.root_token = root_token
        self.frontiers.append(indices)

    def expansion(self, i, indices):
        if i + 1 != len(self.frontiers):
            raise RuntimeError("expansion trace order differs")
        self.frontiers.append(indices)

    def selection(self, parents, scores, tokens, selected):
        self.parents, self.scores, self.tokens, self.selected = tuple(parents), scores, tokens, selected

    def finish(self):
        if len(self.frontiers) != self.actual_depth:
            raise RuntimeError("trace expansion count differs")
        k, device = 8, self.scores.device
        parents = torch.cat(self.parents).repeat_interleave(k)
        selected = torch.cat((torch.zeros(1, device=device, dtype=torch.long), self.selected + 1))
        checkpoints = []
        generation = []
        for d in range(1, self.actual_depth + 1):
            pool_size = k + (d - 1) * k * k
            offset = 0 if d == 1 else k + (d - 2) * k * k
            frontier = self.frontiers[d - 1] + offset + 1
            expanded = torch.cat(self.parents[:d])
            active = torch.zeros(pool_size + 1, dtype=torch.bool, device=device)
            active[expanded] = True
            active[frontier] = True
            ids = torch.arange(1, pool_size + 1, device=device)
            # Length is known from K and depth; no CUDA nonzero/dynamic-shape
            # indexing (which would introduce a host synchronization boundary).
            eliminated_count = pool_size - d * k
            eliminated = ids.masked_fill(active[1:], pool_size + 1).sort().values[:eliminated_count]
            checkpoints.append(DepthCheckpoint(d, pool_size, frontier, expanded, eliminated))
            generation.append(torch.full((k if d == 1 else k*k,), d, device=device, dtype=torch.long))
        return CandidateTrace(self.actual_depth, self.budget_cap, k,
            self.root_token.detach().clone(), self.tokens, parents, self.scores,
            torch.cat(generation), tuple(checkpoints), selected,
            torch.isin(parents[self.selected], selected).all(), torch.isfinite(self.scores).all())


def generate_traced_topk(draft, hidden_states, input_ids, head, logits_processor=None,
                        inputs_embeds=None, embed_weights=None, image_mask=None, *,
                        actual_depth=4, budget_cap=30, trace=True):
    """Run one real author proposal, with temporary D/B constants restored.

    The trace-off path directly calls the loaded author method. Both paths
    preserve author changes to stable_kv/last_img_hidden/tree_mask. Failure does
    not restore those computational states: an owning executor must install
    its legal parent snapshot. This module never manipulates Target KV.
    """
    if type(actual_depth) is not int or not 1 <= actual_depth <= 8:
        raise ValueError("actual_depth must be an integer in [1,8]")
    if type(budget_cap) is not int or budget_cap != 30:
        raise ValueError("FAD first stage freezes root-inclusive budget_cap=30")
    if type(trace) is not bool:
        raise ValueError("trace must be bool")
    if logits_processor is not None:
        raise ValueError("FAD supports greedy logits_processor=None only")
    if type(getattr(draft, "top_k", None)) is not int or draft.top_k != 8:
        raise ValueError("FAD first stage freezes K=8")
    if (not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3
            or hidden_states.shape[0] != 1 or hidden_states.shape[1] < 1
            or not hidden_states.is_floating_point()
            or not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long
            or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2):
        raise ValueError("one nonempty hidden batch and long prefix+root IDs required")
    function = getattr(draft.topK_genrate, "__func__", None)
    if function is None:
        raise ValueError("bound original author topK_genrate required")
    _setup_source_guard()
    _check_loaded_author(function)
    old = {name: vars(draft).get(name, _MISSING) for name in ("total_tokens", "depth")}
    try:
        draft.total_tokens = min(budget_cap - 1, 8 + 64 * (actual_depth - 1))
        draft.depth = actual_depth - 1
        args = (hidden_states, input_ids, head, logits_processor, inputs_embeds, embed_weights, image_mask)
        if not trace:
            return TracedTopKResult(draft.topK_genrate(*args), None)
        collector = _TraceCollector(actual_depth, budget_cap)
        tree = _traced_author(draft, *args, _trace=collector)
        return TracedTopKResult(tree, collector.finish())
    finally:
        for name, value in old.items():
            if value is _MISSING:
                vars(draft).pop(name, None)
            else:
                vars(draft)[name] = value


@torch.no_grad()
def _traced_author(
    self,
    hidden_states,
    input_ids,
    head,
    logits_processor,
    inputs_embeds=None,
    embed_weights=None,
    image_mask=None,
    *, _trace,
):

    input_ids = input_ids.to(hidden_states.device)
    total_tokens = self.total_tokens
    depth = self.depth
    top_k = self.top_k

    sample_token = input_ids[:, -1]

    scores_list = []
    parents_list = []
    ss_token = []

    if inputs_embeds is not None:
        inputs_embeds = inputs_embeds.clone()
        if inputs_embeds.shape[-2] >= input_ids.shape[-1]:
            raise ValueError(
                "inputs_embeds length must be less than input_ids length"
            )
        if embed_weights is not None:
            if embed_weights.dim() != 3 or embed_weights.shape[-1] != 1:
                raise ValueError(
                    "embed_weights should be a 3D tensor with shape (vocab_size, hidden_size, 1)"
                )
            inputs_embeds[
                : embed_weights.shape[0], : embed_weights.shape[1]
            ] *= embed_weights
        inputs_embeds.to(input_ids.device)
        new_embeds = self.embed_tokens(input_ids[:, inputs_embeds.shape[-2] :])
        inputs_embeds = torch.cat((inputs_embeds[:, 1:, :], new_embeds), dim=-2)

    input_ids = input_ids[:, 1:]
    input_ids = input_ids.to(hidden_states.device)

    len_posi = input_ids.shape[1]
    self.reset()

    if hasattr(self, "stable_kv") and self.stable_kv is not None:
        out_hidden, past_key_values = self(
            hidden_states,
            input_ids=input_ids[:, -hidden_states.shape[1] :],
            past_key_values=self.stable_kv,
            use_cache=True,
            image_mask=image_mask,
        )
    else:
        if inputs_embeds is not None:
            input_ids = None
        out_hidden, past_key_values = self(
            hidden_states,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            image_mask=image_mask,
        )
    self.stable_kv = past_key_values
    last_hidden = out_hidden[:, -1]

    last_headout = head(last_hidden)

    last_p = self.logsoftmax(last_headout)
    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values
    scores = topk_p[0]
    scores_list.append(scores[None])
    parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
    ss_token.append(topk_index)
    input_ids = topk_index
    input_hidden = last_hidden[None].repeat(1, top_k, 1)
    tree_mask = self.tree_mask_init
    topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

    _trace.initial(sample_token, topk_cs_index)

    # 4
    for i in range(depth):
        self.tree_mask = tree_mask
        position_ids = len_posi + self.position_ids
        out_hidden, past_key_values = self(
            input_hidden,
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
            image_mask=image_mask,
        )
        len_posi += 1

        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + top_k**2 * bias2 + bias1
        parents = topk_cs_index + bias
        parents_list.append(parents)

        last_headout = head(out_hidden[0])
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        cu_scores = topk_p + scores[:, None]

        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        scores = topk_cs_p

        out_ids = topk_cs_index // top_k
        input_hidden = out_hidden[:, out_ids]
        input_ids = topk_index.view(-1)[topk_cs_index][None]

        ss_token.append(topk_index)
        scores_list.append(cu_scores)
        tree_mask = torch.cat(
            (tree_mask[:, :, out_ids], self.tree_mask_init), dim=3
        )

        _trace.expansion(i, topk_cs_index)

    scores_list = torch.cat(scores_list, dim=0).view(-1)
    ss_token_list = torch.cat(ss_token, dim=0).view(-1)
    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values

    draft_tokens = ss_token_list[top_scores_index]
    draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
    _trace.selection(parents_list, scores_list, ss_token_list, top_scores_index)
    mask_index = torch.searchsorted(
        top_scores_index, draft_parents - 1, right=False
    )
    mask_index[draft_parents == 0] = -1
    mask_index = mask_index + 1
    mask_index_list = mask_index.tolist()
    tree_mask = torch.eye(total_tokens + 1).bool()
    tree_mask[:, 0] = True
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

    tree_position_ids = torch.sum(tree_mask, dim=1) - 1

    tree_mask = tree_mask.float()[None, None]
    draft_tokens = draft_tokens[None]

    del parents_list, scores_list, ss_token, ss_token_list, draft_parents

    max_depth = torch.max(tree_position_ids) + 1
    noleaf_index = torch.unique(mask_index).tolist()
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num

    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
    retrieve_indices = retrieve_indices.tolist()

    rid = 0
    position_ids_list = tree_position_ids.tolist()

    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = position_ids_list[i]
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1

    if logits_processor is not None:
        maxitem = total_tokens + 5

        def custom_sort(lst):
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = sorted(retrieve_indices, key=custom_sort)

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    del (
        mask_index,
        mask_index_list,
        noleaf_index,
        noleaf_num,
        leaf_num,
        max_depth,
        rid,
    )
    tree_position_ids = tree_position_ids.to(hidden_states.device)

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
