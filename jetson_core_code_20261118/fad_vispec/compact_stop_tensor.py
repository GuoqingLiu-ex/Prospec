"""P1 full-graph tensor correctness prototype; no deployment or host export.

Only the author's already-CPU retrieve/mask are read on the host. Every search
value and uploaded position stays on its original device. This deliberately
checks the full CPU-reference contract, with many additional tensor operations;
it is not a fused reducer, a zero-event claim, or an integrated FAD controller.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import torch
from . import compact_search_feedback as reference

SCHEMA = 'fad_compact_stop_tensor_P1_v1'
REFERENCE_SHA = 'ecb4e9d2aeb97ae79bdcf7c7df4de957c7ed0371d34faf5a79340207f5584a26'
Kind, Flag = reference.Kind, reference.Flag


def _require(ok, message):
    if not ok: raise ValueError(message)


@lru_cache(maxsize=1)
def source_receipt():
    _require(hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest() == REFERENCE_SHA,
             'frozen compact reference source drift')
    return dict(reference_sha256=REFERENCE_SHA, author=reference.source_receipt())


@dataclass(frozen=True)
class TensorStopResult:
    packet: torch.Tensor | None
    receipt: dict


def _workspace(named):
    """Metadata snapshot only; never read tensor values or retain tensors."""
    rows, unique = [], {}
    for name, value in named.items():
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        unique[key] = storage.nbytes()
        rows.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
            device=str(value.device), logical_bytes=value.numel()*value.element_size(),
            backing_bytes=storage.nbytes(), storage_offset=value.storage_offset()))
    return dict(tensors=rows, unique_backing_bytes=sum(unique.values()), unique_storages=len(unique),
        scope='named simultaneously live tensors only; excludes expression temporaries, backend workspace and allocator overhead')


def _clip_host(proposed, remaining, stops):
    kept = list(proposed[:remaining])
    for i, token in enumerate(kept):
        if token in stops: return kept[:i+1], 'eos'
    return kept, 'limit' if len(kept) == remaining else None


def _host_structure(mask, retrieve, m):
    """Host operands were built by the author on CPU; no device transfer.

    Return a safe total set of parent/depth placeholders even on invalid host
    data. The device result then becomes UNKNOWN, never a definite failure.
    """
    rows = mask[0, 0].tolist()
    raw_rows = retrieve.tolist()
    valid = all(x in (0, 1) for row in rows for x in row)
    paths, parents, depths = [], [], []
    for i, row in enumerate(rows):
        path = [j for j, x in enumerate(row) if x]
        good = bool(path) and path[0] == 0 and path[-1] == i
        parent = path[-2] if len(path) > 1 else 0
        good = good and (path == [0] if i == 0 else parent < i and paths[parent] + [i] == path)
        valid = valid and good
        paths.append(path); parents.append(parent if 0 <= parent < m else 0); depths.append(len(path)-1)
    cleaned = []
    for row in raw_rows:
        good = all(-1 <= v < m for v in row)
        k = row.index(-1) if -1 in row else len(row)
        good = good and k > 0 and all(v == -1 for v in row[k:])
        path = row[:k]
        good = good and bool(path) and 0 <= path[-1] < m and path == paths[path[-1]]
        valid = valid and good
        cleaned.append(path)
    leaves = set(range(m)) - set(parents[1:])
    ends = [p[-1] for p in cleaned if p]
    valid = valid and len(cleaned) == len(leaves) and len(ends) == len(cleaned) and set(ends) == leaves
    return valid, parents, depths


def _gather1(values, index):
    # Dynamic scalar indices must remain device tensors: no Python indexing.
    return values.gather(0, index.reshape(1)).squeeze(0)


def _full_graph(result, raw_indices, proposed_ids):
    search = result.search
    scores, tokens = search.scores, search.tokens
    tree_tokens, retrieve, mask, positions = result.tree
    n, m, D = tokens.numel(), search.selected.numel()+1, search.actual_depth
    host_valid, host_parent, host_depth = _host_structure(mask, retrieve, m)
    # arange is device generation, not a host constant tensor upload.
    ids = torch.arange(1, n+1, dtype=torch.int64, device=tokens.device)
    zero = torch.zeros_like(ids[:1])
    expanded = torch.cat(search.parents)
    parents = expanded.repeat_interleave(8)
    parent_full = torch.cat((zero, parents))
    levels = torch.where(ids <= 8, 1, 2 + (ids-9)//64)
    levels_full = torch.cat((zero, levels))
    safe_selected = search.selected.clamp(0, n-1)
    selected = torch.cat((zero, safe_selected+1))
    safe_parents = parents.clamp(0, n)
    finite = torch.isfinite(scores).all() & (scores <= 0).all()
    valid = finite & (tokens >= 0).all() & (parents >= 0).all() & (parents < ids).all()
    valid = valid & ((levels_full.gather(0, safe_parents)+1) == levels).all()
    valid = valid & ((parents == 0) | (scores <= scores.gather(0, (safe_parents-1).clamp(0, n-1)))).all()
    valid = valid & (search.parents[0] == 0).all()
    valid = valid & (search.selected >= 0).all() & (search.selected < n).all()
    valid = valid & (search.selected[1:] > search.selected[:-1]).all()
    # Each eight-child group belongs to one real parent; repeated parents are
    # rejected by per-depth frontier uniqueness below, with disjoint depths.
    child_tokens = tokens.reshape(-1, 8)
    child_unique = (child_tokens[:, :, None] == child_tokens[:, None, :]).sum(dim=(1, 2)) == 8
    valid = valid & child_unique.all()
    frontier_offset = 0 if D == 1 else 8+64*(D-2)
    frontier = search.frontier + frontier_offset + 1
    frontiers = list(search.parents[1:]) + [frontier]
    frontier_scores = []
    for d, front in enumerate(frontiers, 1):
        start, count = (0, 8) if d == 1 else (8+64*(d-2), 64)
        index = front-start-1
        good_index = (index >= 0).all() & (index < count).all()
        unique = (front[:, None] == front[None, :]).sum() == 8
        safe = index.clamp(0, count-1)
        layer = scores[start:start+count]
        chosen = torch.zeros_like(layer, dtype=torch.bool).scatter_(0, safe, True)
        fs = layer.gather(0, safe)
        cutoff_ok = fs.amin() >= torch.where(chosen, -torch.inf, layer).amax()
        valid = valid & good_index & unique & cutoff_ok
        frontier_scores.append(fs)
    valid = valid & ~(frontier[:, None] == expanded[None, :]).any()
    chosen = torch.zeros_like(tokens, dtype=torch.bool).scatter_(0, safe_selected, True)
    valid = valid & (scores.gather(0, safe_selected).amin() >= torch.where(chosen, -torch.inf, scores).amax())
    valid = valid & (tree_tokens[0, 1:] == tokens.gather(0, safe_selected)).all()
    # Host mask describes its own complete chains. Bind every such edge and
    # depth to the actual pool IDs and original uploaded position tensor.
    for i in range(m):
        actual_parent = _gather1(parent_full, selected[i])
        valid = valid & (actual_parent == selected[host_parent[i]]) & (positions[i] == host_depth[i])
    valid = valid & host_valid
    proposed_valid = torch.ones_like(ids[0], dtype=torch.bool)
    for index, token in zip(raw_indices, proposed_ids):
        proposed_valid = proposed_valid & (tree_tokens[0, index] == token)
    # Keep this separately: terminal classification skips the full graph in
    # the frozen CPU reference, but the actual submitted prefix must match.
    workspace = _workspace(dict(node_ids=ids, expanded=expanded, parents=parents,
        parent_full=parent_full, levels=levels, levels_full=levels_full,
        safe_selected=safe_selected, selected=selected, safe_parents=safe_parents,
        selected_mask=chosen, frontier=frontier))
    return dict(valid=valid, proposed_valid=proposed_valid, ids=ids, selected=selected,
        parents=parents, levels_full=levels_full, expanded=expanded, frontier=frontier,
        frontier_scores=frontier_scores, workspace=workspace)


@torch.no_grad()
def classify_stop_tensor(result, *, request_key, state_key, best, accepted_draft,
                         proposed_ids=None, submitted_ids, remaining_tokens, stop_ids, next_root):
    """Return a device int64[4] packet, with no tensor-value host access.

    proposed_ids/submitted_ids are existing host posterior/clip outputs, not a
    request for new D2H. Host binding/provenance is the caller's responsibility.
    Device-value violations produce UNKNOWN; structural/host violations raise.
    clip-return returns None without reading/allocating device data. The packet
    does not execute a controller, alter model output, commit KV or certify a
    different depth. Only a future caller can replace the original root read.
    """
    _require(type(request_key) is str and request_key and type(state_key) is str and state_key,
             'request/state binding required')
    _require(type(best) is int and best >= 0 and type(accepted_draft) is int and accepted_draft >= 0,
             'posterior indices')
    _require(type(remaining_tokens) is int and 1 <= remaining_tokens <= 512, 'remaining_tokens')
    # On the caller's non-clip branch proposed and submitted are identical.
    # A truncated clip branch must pass its existing full proposed list.
    if proposed_ids is None: proposed_ids = submitted_ids
    for name, values in (('proposed', proposed_ids), ('submitted', submitted_ids), ('stop', stop_ids)):
        _require(type(values) in (list, tuple, set) if name == 'stop' else type(values) in (list, tuple), name+' host sequence')
        _require(bool(values) and all(type(v) is int and 0 <= v < 2**63 for v in values), name+' host IDs')
    source_receipt()
    search, tree = result.search, result.tree
    _require(type(search.actual_depth) is int and 1 <= search.actual_depth <= 8, 'depth 1..8')
    search.validate(tree)
    tokens, retrieve, mask, positions = tree
    m = search.selected.numel()+1
    _require(tokens.dtype == positions.dtype == torch.int64 and tokens.device == positions.device == search.scores.device
             and tokens.shape == (1, m) and positions.shape == (m,), 'device tree token/position metadata')
    _require(retrieve.device.type == mask.device.type == 'cpu' and retrieve.dtype == torch.int64
             and mask.dtype == torch.float32 and retrieve.ndim == 2 and mask.shape == (1, 1, m, m),
             'original CPU retrieve/mask metadata required; no transfer fallback')
    _require(best < retrieve.shape[0] and accepted_draft < retrieve.shape[1], 'posterior path extent')
    raw = retrieve[best, :accepted_draft+1].tolist()  # Already-CPU author data.
    _require(raw and raw[0] == 0 and all(0 <= i < m for i in raw) and len(proposed_ids) == len(raw), 'accepted raw path')
    kept, clip_reason = _clip_host(proposed_ids, remaining_tokens, set(stop_ids))
    _require(list(submitted_ids) == kept, 'submitted does not equal original clip')
    receipt = dict(schema=SCHEMA, request_key=request_key, state_key=state_key, actual_depth=search.actual_depth,
        CPU_reference_only=False, CUDA_validated=False, production_eligible=None, cross_depth_target_bound=None,
        future_or_next_state_guarantee=False, action_hard_disable_eligible=False,
        actual_target_provenance_independently_checked=False, caller_existing_host_proposed_required=True,
        A_raw=accepted_draft, G_tree=len(kept), clip_reason=clip_reason,
        new_device_to_host_calls=0, host_index_tensor_uploads=0, nextroot_argmax_calls=0,
        actual_GPU_kernels=None, actual_host_events=None, actual_DRAM=None, allocator_peak=None,
        retained_search=search.storage_receipt(), named_workspace=[],
        scope='full-graph Torch correctness prototype; tensor operations and reference lifetimes cost real work; not integrated FAD')
    if clip_reason is not None:
        _require(next_root is None, 'clip-return must not add nextroot read')
        receipt.update(packet_bytes=0, baseline_root_bytes=0, additional_logical_D2H_bytes=0,
            tensor_reduction_run=False, kind='terminal', censored=True,
            clip_device_proposed_binding='not read; original caller host proposed/clip must be bound')
        return TensorStopResult(None, receipt)
    _require(isinstance(next_root, torch.Tensor) and next_root.dtype == torch.int64
             and next_root.ndim == 0 and next_root.device == search.scores.device and not next_root.requires_grad,
             'existing nextroot argmax must be same-device int64 scalar')
    graph = _full_graph(result, raw, proposed_ids)
    valid = graph['valid'] & graph['proposed_valid'] & (next_root >= 0)
    ids, selected, parents = graph['ids'], graph['selected'], graph['parents']
    u = selected[raw[-1]]
    children = parents == u
    correct = children & (search.tokens == next_root)
    count, child_count = correct.sum(), children.sum()
    correct_node = torch.where(correct, ids, 0).amax()
    correct_score = _gather1(search.scores, (correct_node-1).clamp(0, search.scores.numel()-1))
    strict_global = (search.scores > correct_score).sum()
    expanded = (graph['expanded'] == u).any()
    frontier = (graph['frontier'] == u).any()
    correct_verified = (selected == correct_node).any()
    level = _gather1(graph['levels_full'], u)
    u_score = _gather1(search.scores, (u-1).clamp(0, search.scores.numel()-1))
    strict_beam = torch.zeros_like(next_root)
    for d, scores in enumerate(graph['frontier_scores'], 1):
        strict_beam = strict_beam + torch.where(level == d, (scores > u_score).sum(), 0)
    complete = expanded & (child_count == 8) & (count <= 1)
    missing = complete & (count == 0)
    pruned = complete & (count == 1) & ~correct_verified & (strict_global >= 29)
    eliminated = ~expanded & ~frontier & (child_count == 0) & (strict_beam == 8)
    frontier_unknown = ~expanded & frontier
    kind = torch.zeros_like(next_root)
    for predicate, code in ((missing, Kind.CANDIDATE_MISSING), (pruned, Kind.VERIFICATION_PRUNED),
                            (eliminated, Kind.SEARCH_ELIMINATED), (frontier_unknown, Kind.FRONTIER_UNKNOWN)):
        kind = torch.where(predicate, int(code), kind)
    witness = torch.where(complete & (count == 1), correct_node, u)
    witness = torch.where(valid, witness, -1)
    kind = torch.where(valid, kind, int(Kind.UNKNOWN))
    flags = torch.where(valid, int(Flag.LOCAL_SEARCH_ONLY | Flag.FINITE_SCORES | Flag.PATH_COVERED), int(Flag.LOCAL_SEARCH_ONLY))
    flags = flags | torch.where(valid & (pruned | eliminated), int(Flag.STRICT_EXCLUSION), 0)
    flags = flags | torch.where(kind == int(Kind.UNKNOWN), int(Flag.INSUFFICIENT_EVIDENCE), 0)
    eos = torch.zeros_like(next_root, dtype=torch.bool)
    for token in stop_ids: eos = eos | (next_root == token)
    terminal = eos | (len(kept)+1 == remaining_tokens)
    # Like the CPU oracle, a terminal bonus is censored before graph reasoning.
    terminal = terminal & graph['proposed_valid'] & (next_root >= 0)
    terminal_flags = torch.where(eos, int(Flag.LOCAL_SEARCH_ONLY | Flag.CENSORED | Flag.EOS),
                                int(Flag.LOCAL_SEARCH_ONLY | Flag.CENSORED | Flag.LIMIT))
    kind = torch.where(terminal, int(Kind.TERMINAL), kind)
    witness = torch.where(terminal, -1, witness)
    flags = torch.where(terminal, terminal_flags, flags)
    packet = torch.stack((next_root, kind, witness, flags))
    receipt['named_workspace'] = [graph['workspace'], _workspace(dict(children=children, correct=correct,
        packet=packet, kind=kind, witness=witness, flags=flags, strict_global=strict_global, strict_beam=strict_beam))]
    receipt.update(packet_bytes=32, baseline_root_bytes=8, additional_logical_D2H_bytes=24,
        tensor_reduction_run=True, device_validation_read_back=False,
        new_tensor_work=['device arange/zeros', 'cat/repeat parent expansion', 'safe gather/scatter',
            'complete child/path/frontier/TopB comparisons', 'finite/min/max/sum/any reductions',
            'nextroot comparison and int64 packet stack'],
        retained_output_unique_bytes=packet.untyped_storage().nbytes(),
        named_workspace_is_peak=False, includes_unnamed_expression_temporaries=False,
        terminal_classification='device packet only; caller must preserve original bonus/stop behavior')
    return TensorStopResult(packet, receipt)
