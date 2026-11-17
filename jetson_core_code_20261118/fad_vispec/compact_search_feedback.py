"""P1 retained author-search operands and CPU-only stop feedback reference.

No Target, cache, controller, CUDA reducer, implicit export or deployment hook.
Existing author arithmetic is reused through its frozen metadata-only copy.
The 32-byte packet is a protocol, not measured GPU/event/DRAM evidence.
"""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from functools import lru_cache
from pathlib import Path
import weakref

import torch
from . import traced_topk as author

SCHEMA = 'fad_compact_search_feedback_P1_cpu_v1'
TRACER_SHA = 'ff7fe7f36e87d3fd96c2ba2ab29e3ece1ba609ec320140178558054893a63a12'
_MISSING = object()


class Kind(IntEnum):
    UNKNOWN = 0
    CANDIDATE_MISSING = 1
    VERIFICATION_PRUNED = 2
    SEARCH_ELIMINATED = 3
    FRONTIER_UNKNOWN = 4
    TERMINAL = 5


class Flag(IntFlag):
    LOCAL_SEARCH_ONLY = 1
    PATH_COVERED = 2
    FINITE_SCORES = 4
    STRICT_EXCLUSION = 8
    CENSORED = 16
    EOS = 32
    LIMIT = 64
    INSUFFICIENT_EVIDENCE = 128


def _require(ok, message):
    if not ok: raise ValueError(message)


def source_receipt():
    path = Path(author.__file__)
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == TRACER_SHA, 'frozen metadata-copy source drift')
    receipt = author.verify_source_contract()
    return dict(tracer_sha256=TRACER_SHA, author=receipt,
        reused_function='traced_topk._traced_author',
        changes='collector semantics only; same three initial/expansion/selection insertions; no arithmetic edits',
        original_host_tree_work_preserved=True, actual_GPU_parity=False)


@lru_cache(maxsize=1)
def _source_guard():
    return source_receipt()


def _signature(t):
    try: version = t._version
    except RuntimeError: version = None
    return (id(t), tuple(t.shape), tuple(t.stride()), t.storage_offset(), str(t.device),
            t.dtype, t.untyped_storage().data_ptr(), t.untyped_storage().nbytes(), version)


class CompactSearch:
    """Own only existing pool operands; caller must not mutate them.

    Group parents map node n>0 to cat(groups)[(n-1)//8], without a retained
    per-node repeat. Frontier is the last local Top8 index vector. Earlier
    frontiers are already present as the following expansion's parent group.
    Tree bindings are weak; no input/model/hidden/cache reference is retained.
    """
    def __init__(self, depth, parents, scores, tokens, selected, frontier, tree):
        self.actual_depth = depth
        self.parents, self.scores, self.tokens = tuple(parents), scores, tokens
        self.selected, self.frontier = selected, frontier
        self.released = False
        self._tree_refs = tuple(weakref.ref(t) for t in tree)
        self._versions = {name: _signature(t) for name, t in self.named_tensors()}
        self._tree_versions = tuple(_signature(t) for t in tree)

    def named_tensors(self):
        _require(not self.released, 'search already released')
        for name in ('scores', 'tokens', 'selected', 'frontier'):
            yield name, getattr(self, name)
        for i, value in enumerate(self.parents): yield f'parents_{i}', value

    def validate(self, tree):
        _require(not self.released, 'search already released')
        _require(len(tree) == 4 and all(ref() is t for ref, t in zip(self._tree_refs, tree)),
                 'tree object does not belong to this proposal')
        _require(tuple(_signature(t) for t in tree) == self._tree_versions, 'tree metadata/version changed')
        _require({n: _signature(t) for n, t in self.named_tensors()} == self._versions,
                 'search metadata/version changed')
        n = 8 + 64 * (self.actual_depth - 1)
        _require(len(self.parents) == self.actual_depth and self.scores.shape == self.tokens.shape == (n,)
                 and self.selected.shape == (min(29, n),) and self.frontier.shape == (8,), 'incomplete search shapes')
        device = self.scores.device
        _require(self.scores.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64), 'score dtype')
        for name, t in self.named_tensors():
            _require(t.device == device, 'search device differs')
            if name != 'scores': _require(t.dtype == torch.long, 'search indices/tokens must be int64')
        _require(self.parents[0].shape == (1,) and all(p.shape == (8,) for p in self.parents[1:]), 'parent groups incomplete')

    def storage_receipt(self):
        rows = list(self.named_tensors()); unique = {}
        for _, t in rows:
            s = t.untyped_storage(); unique[(str(t.device), s.data_ptr())] = s.nbytes()
        return dict(schema=SCHEMA, actual_depth=self.actual_depth,
            pool_nodes_nonroot=self.tokens.numel(), actual_verify_nodes=self.selected.numel()+1,
            named_logical_bytes=sum(t.numel()*t.element_size() for _, t in rows),
            retained_unique_backing_bytes=sum(unique.values()), retained_unique_storages=len(unique),
            new_tensor_operations_in_collector=0, full_checkpoint_copies=False,
            inference_version_guard_available=all(v[-1] is not None for v in self._versions.values()),
            scope='existing search backing retained across Target; excludes returned tree, transient workspace, Python metadata and allocator peak; not DRAM')

    def release(self):
        self.parents = (); self.scores = self.tokens = self.selected = self.frontier = None
        self._tree_refs = (); self._versions = {}; self._tree_versions = (); self.released = True

    def __enter__(self): return self
    def __exit__(self, *exc): self.release()


@dataclass(frozen=True)
class CompactSearchResult:
    tree: tuple
    search: CompactSearch


class _Collector:
    def __init__(self, depth):
        self.depth, self.calls = depth, 0
        self.parents = self.scores = self.tokens = self.selected = self.frontier = None
    def initial(self, root, frontier):
        # Do not keep sample_token: it is a view of the complete input prefix.
        _require(self.calls == 0, 'duplicate initial metadata')
        self.frontier = frontier; self.calls = 1
    def expansion(self, i, frontier):
        _require(self.calls == i + 1, 'expansion metadata order differs')
        self.frontier = frontier; self.calls += 1
    def selection(self, parents, scores, tokens, selected):
        _require(self.calls == self.depth and self.parents is None, 'selection metadata incomplete/duplicate')
        self.parents, self.scores, self.tokens, self.selected = tuple(parents), scores, tokens, selected
    def finish(self, tree):
        _require(self.parents is not None, 'missing final selection')
        return CompactSearch(self.depth, self.parents, self.scores, self.tokens, self.selected, self.frontier, tree)
    def release(self):
        self.parents = self.scores = self.tokens = self.selected = self.frontier = None


@torch.no_grad()
def generate_compact_search(draft, hidden_states, input_ids, head, logits_processor=None,
                            inputs_embeds=None, embed_weights=None, image_mask=None, *, actual_depth=4):
    """Real author computation, reference retention only; no CPU tensor reads added.

    Exception restores D/B attributes, not partially written Draft KV. The
    owning executor must poison/close or restore its own legal state.
    """
    _require(type(actual_depth) is int and 1 <= actual_depth <= 8, 'actual_depth must be integer 1..8')
    _require(type(draft.top_k) is int and draft.top_k == 8, 'fixed K8 required')
    _require(logits_processor is None, 'greedy only')
    _require(isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3
             and hidden_states.shape[0] == 1 and hidden_states.shape[1] > 0 and hidden_states.is_floating_point(), 'hidden shape/dtype')
    _require(isinstance(input_ids, torch.Tensor) and input_ids.dtype == torch.long
             and input_ids.ndim == 2 and input_ids.shape[0] == 1 and input_ids.shape[1] >= 2, 'prefix+root IDs shape/dtype')
    function = getattr(draft.topK_genrate, '__func__', None)
    _require(function is not None, 'bound author method required')
    _source_guard(); author._check_loaded_author(function)
    saved = {n: vars(draft).get(n, _MISSING) for n in ('total_tokens', 'depth')}
    collector = _Collector(actual_depth)
    try:
        draft.total_tokens = min(29, 8 + 64*(actual_depth-1)); draft.depth = actual_depth-1
        tree = author._traced_author(draft, hidden_states, input_ids, head, logits_processor,
            inputs_embeds, embed_weights, image_mask, _trace=collector)
        return CompactSearchResult(tree, collector.finish(tree))
    finally:
        collector.release()
        for name, value in saved.items():
            if value is _MISSING: vars(draft).pop(name, None)
            else: vars(draft)[name] = value


class _Insufficient(Exception): pass


def _evidence(ok, reason):
    if not ok: raise _Insufficient(reason)


def _cpu_graph(result):
    """Offline reference scans; future GPU reduction must count its own work."""
    search, tree = result.search, result.tree
    search.validate(tree)
    _require(all(t.device.type == 'cpu' for _, t in search.named_tensors())
             and all(t.device.type == 'cpu' for t in tree), 'CPU reference only; no implicit CUDA export/fallback')
    scores, tokens, selected = search.scores.tolist(), search.tokens.tolist(), [0]+(search.selected+1).tolist()
    groups = [p.tolist() for p in search.parents]
    _evidence(all(math.isfinite(s) and s <= 0 for s in scores), 'nonfinite_or_invalid_log_scores')
    _evidence(groups[0] == [0], 'missing_root_expansion')
    parents = [None] + [p for group in groups for p in group for _ in range(8)]
    levels = [0] + [1]*8 + [d for d in range(2, search.actual_depth+1) for _ in range(64)]
    n = len(scores)
    _evidence(all(type(t) is int and t >= 0 for t in tokens), 'invalid_token_ids')
    _evidence(len(selected) == len(set(selected)) and selected[1:] == sorted(selected[1:])
              and all(0 <= v <= n for v in selected), 'invalid_topB_indices')
    for node in range(1, n+1):
        p = parents[node]
        _evidence(0 <= p < node and levels[p]+1 == levels[node], 'invalid_parent_depth')
        _evidence(p == 0 or scores[node-1] <= scores[p-1], 'nonmonotone_path_score')
    expanded = [p for group in groups for p in group]
    _evidence(len(expanded) == len(set(expanded)), 'reexpanded_parent')
    frontier_offset = 0 if search.actual_depth == 1 else 8+64*(search.actual_depth-2)
    frontier = [v+frontier_offset+1 for v in search.frontier.tolist()]
    # Actual stage frontiers are next stage parent operands, not reconstructed TopK.
    frontiers = groups[1:]+[frontier]
    for d, f in enumerate(frontiers, 1):
        _evidence(len(set(f)) == 8 and all(1 <= v <= n and levels[v] == d for v in f), 'frontier_coverage')
        layer = [i for i in range(1, n+1) if levels[i] == d]
        _evidence(min(scores[v-1] for v in f) >= max((scores[v-1] for v in layer if v not in f), default=-math.inf), 'frontier_not_actual_Top8')
    _evidence(not set(expanded)&set(frontier), 'frontier_expanded_overlap')
    for p in expanded:
        children = [i for i in range(1, n+1) if parents[i] == p]
        _evidence(len(children) == 8 and len({tokens[i-1] for i in children}) == 8, 'incomplete_or_ambiguous_children')
    _evidence(min(scores[v-1] for v in selected[1:]) >= max((scores[v-1] for v in range(1,n+1) if v not in selected), default=-math.inf), 'selection_not_actual_TopB')
    tree_tokens, retrieve, mask, depth = tree
    m = len(selected)
    _evidence(tuple(tree_tokens.shape) == (1,m) and tuple(mask.shape) == (1,1,m,m)
              and tuple(depth.shape) == (m,) and retrieve.ndim == 2, 'tree_shape_coverage')
    _evidence(tree_tokens[0,1:].tolist() == [tokens[v-1] for v in selected[1:]], 'tree_token_map')
    _evidence(all(x in (0,1) for x in mask.flatten().tolist()), 'mask_not_binary')
    paths = {}
    for index, node in enumerate(selected):
        path = [node]
        while path[-1] != 0: path.append(parents[path[-1]])
        path.reverse(); paths[node] = path
        _evidence(all(v in selected for v in path), 'unverified_parent_path')
        want = {selected.index(v) for v in path}
        got = {j for j,x in enumerate(mask[0,0,index].tolist()) if x}
        _evidence(got == want and int(depth[index]) == len(path)-1, 'mask_or_position_parent_disagreement')
    rows = []
    for row in retrieve.tolist():
        _evidence(all(type(v) is int and -1 <= v < m for v in row), 'retrieve_index_range')
        k = row.index(-1) if -1 in row else len(row)
        _evidence(all(v == -1 for v in row[k:]) and k > 0, 'retrieve_padding')
        ids = [selected[v] for v in row[:k]]
        _evidence(ids == paths[ids[-1]], 'retrieve_parent_path')
        rows.append(row[:k])
    leaves = {node for node in selected if not any(parents[v] == node for v in selected[1:])}
    _evidence(len(rows) == len(leaves) and {selected[r[-1]] for r in rows} == leaves, 'retrieve_leaf_coverage')
    return dict(scores=scores,tokens=tokens,parents=parents,levels=levels,selected=selected,
                expanded=set(expanded),frontier=set(frontier),frontiers=frontiers,rows=rows,paths=paths)


def _clip(values, remaining, stops):
    values = list(values[:remaining])
    for i, value in enumerate(values):
        if value in stops: return values[:i+1], 'eos'
    return values, 'limit' if len(values) == remaining else None


def classify_stop_cpu(result, *, request_key, state_key, best, accepted_draft,
                      next_root, submitted_ids, remaining_tokens, stop_ids,
                      stop_reason=None, terminal_bonus_ids=()):
    """Classify only one actual posterior stop, using host-observed metadata.

    best/accepted originate from unchanged evaluate_posterior. next_root is its
    existing stop-logit argmax (None on clip-return). Caller must bind them to
    this proposal and actual Target execution; this CPU helper cannot prove
    that provenance or cross-shape Target equality. No shallow replay or upper
    bound is returned. Failure facts concern only the current search graph.
    """
    _require(isinstance(request_key,str) and bool(request_key) and isinstance(state_key,str) and bool(state_key), 'request/state binding required')
    _require(type(best) is int and best >= 0 and type(accepted_draft) is int and accepted_draft >= 0, 'posterior indices')
    _require(type(remaining_tokens) is int and 1 <= remaining_tokens <= 512, 'remaining_tokens')
    _require(type(stop_ids) in (list,tuple,set) and stop_ids and all(type(x) is int and x >= 0 for x in stop_ids), 'stop IDs')
    _require(type(submitted_ids) in (list,tuple) and submitted_ids and all(type(x) is int and x >= 0 for x in submitted_ids), 'submitted IDs')
    _require(stop_reason in (None,'eos','limit'), 'stop reason')
    _require(type(terminal_bonus_ids) in (list,tuple), 'terminal bonus IDs')
    result.search.validate(result.tree)
    _require(all(t.device.type == 'cpu' for t in result.tree), 'CPU reference only; no implicit CUDA export/fallback')
    tokens,retrieve,_,_ = result.tree
    _require(best < retrieve.shape[0] and accepted_draft < retrieve.shape[1], 'posterior path extent')
    raw_indices = retrieve[best,:accepted_draft+1].tolist()
    _require(raw_indices and raw_indices[0] == 0 and all(0 <= i < tokens.shape[1] for i in raw_indices), 'accepted raw path')
    proposed = [int(tokens[0,i]) for i in raw_indices]
    clipped, clip_reason = _clip(proposed, remaining_tokens, set(stop_ids))
    _require(list(submitted_ids) == clipped, 'submitted does not equal original clip')
    if clip_reason is not None:
        _require(next_root is None and not terminal_bonus_ids and stop_reason == clip_reason,
                 'clip-return must not add nextroot read or bonus')
    else:
        _require(type(next_root) is int and 0 <= next_root < 2**63, 'existing nextroot argmax required')
        reason = 'eos' if next_root in stop_ids else ('limit' if len(clipped)+1 == remaining_tokens else None)
        _require(reason == stop_reason and list(terminal_bonus_ids) == ([next_root] if reason else []), 'terminal bonus/stop differs')
    flags = Flag.LOCAL_SEARCH_ONLY
    report = dict(schema=SCHEMA,request_key=request_key,state_key=state_key,
        actual_depth=result.search.actual_depth,kind='unknown',kind_code=int(Kind.UNKNOWN),witness_node_id=-1,
        actual=dict(A_raw=accepted_draft,G_tree=len(clipped),G_output=len(clipped)+len(terminal_bonus_ids),
                    terminal_bonus_ids=list(terminal_bonus_ids),censored=stop_reason is not None),
        certificate=None,cross_depth_target_bound=None,future_or_next_state_guarantee=False,
        actual_target_provenance_independently_checked=False,CPU_reference_only=True,
        production_eligible=None,action_hard_disable_eligible=False,
        conditional_counterfactuals='not evaluated here; existing typed feedback module remains unchanged',
        missing_evidence=[],evidence={},packet=None,
        packet_cost=dict(protocol='int64[4]: next_root, kind, witness_pool_node_id, flags',
            logical_D2H_bytes=0 if clip_reason else 32,baseline_root_bytes=0 if clip_reason else 8,
            additional_logical_D2H_bytes=0 if clip_reason else 24,
            actual_host_events=None,actual_GPU_kernels=None,actual_DRAM=None,GPU_reducer_implemented=False,
            CPU_reference_scan_included=True))
    kind, witness = Kind.UNKNOWN, -1
    if stop_reason is not None:
        kind = Kind.TERMINAL;flags |= Flag.CENSORED | (Flag.EOS if stop_reason == 'eos' else Flag.LIMIT)
    else:
        try:
            g = _cpu_graph(result)
            flags |= Flag.FINITE_SCORES | Flag.PATH_COVERED
            _evidence(best < len(g['rows']) and raw_indices == g['rows'][best][:accepted_draft+1], 'actual_retrieve_prefix')
            u = g['selected'][raw_indices[-1]]; witness = u
            report['evidence'].update(stop_node_id=u,stop_path_node_ids=g['paths'][u],
                next_root=next_root,selection_tie_contract='original torch.topk; no stable total order')
            children = [i for i in range(1,len(g['parents'])) if g['parents'][i] == u]
            correct = [i for i in children if g['tokens'][i-1] == next_root]
            if u in g['expanded']:
                _evidence(len(children) == 8 and len(correct) <= 1, 'complete_expansion_required')
                if not correct:
                    kind = Kind.CANDIDATE_MISSING
                    report['evidence']['complete_children_node_ids'] = children
                else:
                    witness = correct[0]
                    _evidence(witness not in g['selected'], 'matching_verified_child_conflicts_with_posterior_stop')
                    strict = sum(s > g['scores'][witness-1] for s in g['scores'])
                    report['evidence']['strict_global_dominators'] = strict
                    _evidence(strict >= 29, 'TopB_boundary_tie_or_insufficient_strict_dominators')
                    kind = Kind.VERIFICATION_PRUNED;flags |= Flag.STRICT_EXCLUSION
            elif u in g['frontier']:
                kind = Kind.FRONTIER_UNKNOWN
                report['missing_evidence'].append('stop_parent_not_expanded_at_current_depth')
            else:
                d = g['levels'][u]
                strict = sum(g['scores'][v-1] > g['scores'][u-1] for v in g['frontiers'][d-1])
                report['evidence']['strict_beam_dominators'] = strict
                _evidence(not children and strict == 8, 'beam_boundary_tie_or_incomplete_elimination')
                kind = Kind.SEARCH_ELIMINATED;flags |= Flag.STRICT_EXCLUSION
        except _Insufficient as exc:
            kind = Kind.UNKNOWN;flags |= Flag.INSUFFICIENT_EVIDENCE
            report['missing_evidence'].append(str(exc))
    report.update(kind=kind.name.lower(),kind_code=int(kind),witness_node_id=witness,flags=int(flags),
                  usable_local_failure_fact=kind in (Kind.CANDIDATE_MISSING,Kind.VERIFICATION_PRUNED,Kind.SEARCH_ELIMINATED))
    if clip_reason is None:
        report['packet'] = torch.tensor([next_root,int(kind),witness,int(flags)],dtype=torch.int64)
    return report


def packet_to_host_reference(packet):
    """CPU decode only. Deployment must replace (not supplement) root.item()."""
    _require(isinstance(packet,torch.Tensor) and packet.device.type == 'cpu' and packet.dtype == torch.int64
             and tuple(packet.shape) == (4,) and packet.is_contiguous(), 'contiguous CPU int64[4] packet required')
    root,kind,witness,flags = packet.tolist()
    _require(root >= 0 and kind in [int(k) for k in Kind] and witness >= -1 and 0 <= flags < 256, 'invalid packet values')
    return dict(next_root=root,kind=Kind(kind).name.lower(),witness_node_id=witness,flags=flags)
