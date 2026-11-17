"""CPU structural feedback only; no model execution or next-state guarantee.

Consumes CandidateTrace.to_cpu_record() plus separately bound Target observations.
A source premise is evidence supplied by the caller, not an attestation performed
by this module. Reused predictions and deeper bounds remain conditional.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import struct

SCHEMA = 'fad_structural_feedback_v1'
K, CAP = 8, 30
COMMON = ('same_state', 'search_prefix_preserved', 'target_prefix_invariant')


def _need(ok, reason):
    if not ok:
        raise ValueError(reason)


def _integer(x, name, lower=0):
    _need(type(x) is int and x >= lower, name)
    return x


def _ids(xs, name, *, unique=True):
    _need(type(xs) is list and all(type(x) is int and x >= 0 for x in xs), name)
    _need(not unique or len(set(xs)) == len(xs), name + ': duplicate')
    return xs


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _score_bytes(scores, dtype):
    formats = {'torch.float16': 'e', 'torch.float32': 'f', 'torch.float64': 'd'}
    if dtype == 'torch.bfloat16':
        out = bytearray()
        for x in scores:
            word = struct.unpack('<I', struct.pack('<f', x))[0]
            _need(word & 65535 == 0, 'score not exactly representable in bfloat16')
            out.extend(struct.pack('<H', word >> 16))
        return bytes(out)
    _need(dtype in formats, 'unsupported original score dtype')
    try:
        packed = struct.pack('<' + formats[dtype] * len(scores), *scores)
    except (OverflowError, struct.error) as exc:
        raise ValueError('invalid original score representation') from exc
    unpacked = struct.unpack('<' + formats[dtype] * len(scores), packed)
    _need(all(a == b for a, b in zip(scores, unpacked)), 'score precision was changed during export')
    return packed


def _nodes(t):
    result = {0: dict(id=0, parent=None, token=t['root_token'], depth=0, path=[0], score=0.)}
    edges = set()
    for i, (token, parent, score, depth) in enumerate(zip(t['token_ids'], t['parent_ids'],
                                                        t['scores'], t['generation_depth']), 1):
        _integer(token, 'token ID'); _integer(parent, 'parent ID')
        _need(parent < i and parent in result, 'parent must precede child')
        _need(type(score) in (float, int) and math.isfinite(score) and score <= 0., 'nonfinite/positive cumulative logscore')
        _need(type(depth) is int and depth == result[parent]['depth'] + 1, 'generation depth/path mismatch')
        _need(score <= result[parent]['score'], 'descendant cumulative logscore increased')
        _need((parent, token) not in edges, 'duplicate token under the same parent is ambiguous')
        edges.add((parent, token))
        result[i] = dict(id=i, parent=parent, token=token, depth=depth,
                         path=result[parent]['path'] + [i], score=float(score))
    return result


def _rank_ok(selected, pool, nodes, count):
    _need(len(selected) == count and set(selected) <= set(pool), 'invalid selection size/membership')
    omitted = set(pool) - set(selected)
    _need(not omitted or min(nodes[i]['score'] for i in selected) >=
          max(nodes[i]['score'] for i in omitted), 'selection is not TopK/TopB')


def _contexts(ids, nodes, binding):
    return {str(i): dict(state_key=binding['state_key'], ancestor_ids=nodes[i]['path'],
                position_ids=[binding['cache_before'] + binding['rope_delta'] + nodes[i]['depth']]*3)
            for i in ids}


def _paths(selected, nodes):
    parents = {nodes[i]['parent'] for i in selected if i}
    return [nodes[i]['path'] for i in selected if i not in parents]


def _posterior(paths, nodes, next_ids):
    matches = []
    for path in paths:
        accepted = 0
        for parent, child in zip(path, path[1:]):
            if str(parent) not in next_ids:
                return None
            if next_ids[str(parent)] != nodes[child]['token']:
                break
            accepted += 1
        matches.append(accepted)
    best = max(matches)
    return paths[matches.index(best)][:best + 1]


def _clip(tokens, remaining, stops):
    clipped = tokens[:remaining]
    for i, token in enumerate(clipped):
        if token in stops:
            return clipped[:i + 1], 'eos'
    return clipped, 'limit' if len(clipped) == remaining else None


def _outcome(path, nodes, predictions, binding):
    tokens = [nodes[i]['token'] for i in path]
    submitted, stop = _clip(tokens, binding['remaining_tokens'], binding['stop_ids'])
    selected = path[:len(submitted)]
    bonus = []
    complete = True
    if stop is None:
        nxt = predictions.get(str(selected[-1]))
        if nxt is None:
            complete = False
        elif nxt in binding['stop_ids']:
            bonus, stop = [nxt], 'eos'
        elif len(submitted) + 1 == binding['remaining_tokens']:
            bonus, stop = [nxt], 'limit'
    return dict(A_raw=len(path)-1, G_tree=len(submitted),
                G_output=len(submitted)+len(bonus) if complete else None,
                submitted_ids=submitted, selected_committed_ids=selected,
                terminal_bonus_ids=bonus, stop_reason=stop, outcome_complete=complete)


@dataclass(frozen=True)
class ValidatedRoundTrace:
    """Owns a deep copy; mutation after validation is detected before use."""
    record: dict
    sha256: str


def validate_round_trace(record):
    r = deepcopy(record)
    _need(type(r) is dict and r['schema'] == 'fad_round_feedback_input_v1', 'input schema')
    b, t, v = r['binding'], r['trace'], r['verification']
    for key in ('request_key', 'state_key', 'snapshot_id', 'source_receipt', 'precision_receipt'):
        _need(type(b[key]) is str and bool(b[key]), 'missing state/source binding: '+key)
    _integer(b['cache_before'], 'cache_before', 1)
    _integer(b['remaining_tokens'], 'remaining_tokens', 1)
    _need(b['remaining_tokens'] <= 512 and type(b['rope_delta']) is int, 'remaining/rope delta')
    _ids(b['stop_ids'], 'stop IDs'); _need(bool(b['stop_ids']), 'missing stops')
    _integer(t['root_token'], 'root token')
    _need(t['schema']=='fad_actual_author_topk_trace_v1', 'trace export schema')
    _need(type(t['beam_width']) is int and type(t['budget_cap']) is int and
          t['beam_width'] == K and t['budget_cap'] == CAP, 'fixed K8/Bcap30 required')
    _need(v['state_key'] == b['state_key'], 'verification belongs to another state')
    _need(type(r['premises']) is dict, 'premises mapping')
    for name, premise in r['premises'].items():
        _need(type(name) is str and type(premise) is dict and premise['status'] in
              ('source_contract', 'measured', 'missing', 'refuted'), 'invalid premise record')
        _need(type(premise['evidence_ids']) is list and all(type(x) is str and x for x in premise['evidence_ids']), 'premise evidence list')
    # No Draft tree is called when prefill already produces a terminal token.
    if v['kind'] == 'prefill_terminal':
        _need(t['actual_depth'] == 0 and all(t[k] == [] for k in
              ('token_ids','parent_ids','scores','generation_depth','checkpoints','selected_ids')), 'prefill terminal has a fabricated tree')
        root = t['root_token']; expected = 'eos' if root in b['stop_ids'] else ('limit' if b['remaining_tokens']==1 else None)
        _need(expected is not None and v['submitted_ids']==[root] and v['stop_reason']==expected,
              'invalid prefill terminal')
        return ValidatedRoundTrace(r, _digest(r))
    _need(v['kind'] == 'tree' and type(t['actual_depth']) is int and 1 <= t['actual_depth'] <= 8, 'actual depth/kind')
    _need(t['root_token'] not in b['stop_ids'] and b['remaining_tokens'] > 1, 'terminal root must bypass Draft')
    m = K + (t['actual_depth']-1)*K*K
    _need(all(type(t[k]) is list and len(t[k])==m for k in
              ('token_ids','parent_ids','scores','generation_depth')), 'candidate pool shape')
    packed = _score_bytes(t['scores'], t['score_dtype'])
    _need(type(t['scores_bytes_hex']) is str and packed.hex()==t['scores_bytes_hex'], 'score byte receipt mismatch')
    nodes = _nodes(t)
    _need(t['finite_scores'] is True and t['ancestor_closed'] is True, 'exported structural flag failed')
    checkpoints = t['checkpoints']
    _need(len(checkpoints)==t['actual_depth'], 'missing checkpoint')
    prior_frontier = [0]
    for d, cp in enumerate(checkpoints, 1):
        size = K+(d-1)*K*K
        _need(type(cp['D']) is int and type(cp['pool_size']) is int and
              cp['D']==d and cp['pool_size']==size, 'checkpoint order/pool size')
        pool = list(range(1,size+1)); available = [0]+pool
        _need(all(nodes[i]['depth']<=d for i in available) and
              all(nodes[i]['depth']==d for i in range((K+(d-2)*K*K if d>1 else 0)+1,size+1)), 'checkpoint generation prefix changed')
        children = {p:[i for i in pool if nodes[i]['parent']==p] for p in available}
        expanded = [p for p in available if children[p]]
        for parent in expanded:
            _need(len(children[parent])==K, 'expanded parent lacks complete K children')
        new_parents = {nodes[i]['parent'] for i in pool if nodes[i]['depth']==d}
        _need(new_parents==set(prior_frontier), 'search reexpanded/eliminated parent or omitted frontier')
        frontier = _ids(cp['frontier_ids'], 'frontier IDs')
        _rank_ok(frontier, [i for i in pool if nodes[i]['depth']==d], nodes, K)
        _need(set(_ids(cp['expanded_ids'],'expanded IDs'))==set(expanded), 'expanded status differs from edges')
        eliminated = set(available)-set(expanded)-set(frontier)
        _need(set(_ids(cp['eliminated_ids'],'eliminated IDs'))==eliminated, 'eliminated status is not permanent search retirement')
        prior_frontier=frontier
    selected = _ids(t['selected_ids'], 'verified selection')
    _need(selected and selected[0]==0 and selected[1:]==sorted(selected[1:]), 'final selection order differs from author pool order')
    _rank_ok(selected[1:], list(range(1,m+1)), nodes, min(CAP-1,m))
    if _tie_proved(r):
        fixed = sorted(range(1,m+1), key=lambda i:(-nodes[i]['score'],i))[:min(CAP-1,m)]
        _need(selected[1:]==sorted(fixed), 'declared fixed selection order contradicts actual TopB')
    _need(all(set(nodes[i]['path'])<=set(selected) for i in selected), 'verified selection is not ancestor closed')
    _need(v['verified_ids']==selected and v['node_contexts']==_contexts(selected,nodes,b), 'Target node/path/mask/position binding differs')
    predictions=v['next_ids']; _need(type(predictions) is dict, 'Target predictions mapping')
    _need(all(k in {str(i) for i in selected} and type(y) is int and y>=0 for k,y in predictions.items()), 'prediction node/ID invalid')
    paths=v['retrieve_paths']; expected_paths=_paths(selected,nodes)
    _need(type(paths) is list and len(paths)==len(expected_paths) and
          sorted(paths)==sorted(expected_paths), 'retrieve must contain exactly the complete leaf paths')
    rawpath=_ids(v['selected_raw_ids'],'raw accepted path')
    _need(rawpath and any(rawpath==path[:len(rawpath)] for path in paths), 'raw selected path not in retrieve')
    _need(v['accepted_draft']==len(rawpath)-1, 'raw accepted/root off by one')
    inferred=_posterior(paths,nodes,predictions)
    _need(inferred is None or inferred==rawpath, 'actual greedy posterior/path mismatch')
    for p,c in zip(rawpath,rawpath[1:]):
        _need(str(p) not in predictions or predictions[str(p)]==nodes[c]['token'], 'accepted edge disagrees with Target')
    outcome=_outcome(rawpath,nodes,predictions,b)
    for name in ('submitted_ids','selected_committed_ids'):
        _need(v[name]==outcome[name], 'clip/path mismatch: '+name)
    if outcome['outcome_complete']:
        _need(v['terminal_bonus_ids']==outcome['terminal_bonus_ids'] and v['stop_reason']==outcome['stop_reason'], 'bonus/EOS/limit mismatch')
    else:
        _need(v['terminal_bonus_ids']==[] and v['stop_reason'] is None, 'missing prediction cannot justify terminal bonus')
    return ValidatedRoundTrace(r, _digest(r))


def _missing(r, names):
    return [name for name in names if r['premises'].get(name,{}).get('status') not in
            ('source_contract','measured') or not r['premises'].get(name,{}).get('evidence_ids')]


def _tie_proved(r):
    tie=r.get('selection_tie_contract',{})
    return bool(tie.get('key')=='score_desc_node_id_asc' and
            type(tie.get('evidence_ids')) is list and tie['evidence_ids'] and
            all(type(x) is str and x for x in tie['evidence_ids']))


def _select(cp, nodes, r):
    ranked=sorted(range(1,cp['pool_size']+1),key=lambda i:(-nodes[i]['score'],i))
    take=min(CAP-1,len(ranked))
    boundary_tie=len(ranked)>take and nodes[ranked[take-1]]['score']==nodes[ranked[take]]['score']
    if boundary_tie and not _tie_proved(r):
        return None,'unproved_selection_boundary_tie'
    return [0]+sorted(ranked[:take]),None


def _interval(lo,hi,exact=None,*,censored=False):
    return dict(lower=lo,upper=hi,exact_value=exact,censored=censored)


def _entry(r,d):
    b=r['binding']; amax=min(d,CAP-1); gmax=min(b['remaining_tokens'],amax+1)
    return dict(requested_D=d,author_expansions=d-1,K=K,verification_node_cap=CAP,
        planned_verification_nodes=min(CAP,1+K+(d-1)*K*K),
        feedback_type='unknown',claim_scope='unobserved',valid_for_state_key=b['state_key'],
        A_raw=_interval(0,amax),G_tree=_interval(1,gmax),G_output=_interval(1,min(b['remaining_tokens'],gmax+1)),
        certificate_kind=None,stop_class=None,witness_node_ids=[],missing_evidence=[],
        eligible_for_exact_update=False,eligible_for_upper_constraint=False,
        actual_counterfactual_tested=False)


def _exact(e,outcome,scope,*,censored):
    e.update(feedback_type='exact',claim_scope=scope,eligible_for_exact_update=not censored,
             stop_class='terminal_or_clipped' if censored else 'observed_acceptance')
    for key in ('A_raw','G_tree','G_output'):
        value=outcome[key]
        if value is not None:
            e[key]=_interval(value,value,value,censored=censored and key=='A_raw')
    e['replayed_outcome']=deepcopy(outcome)


def build_depth_feedback(validated, requested_depths):
    _need(isinstance(validated,ValidatedRoundTrace), 'validate trace before feedback')
    _need(_digest(validated.record)==validated.sha256,'validated input was mutated')
    _ids(requested_depths,'requested depths')
    _need(requested_depths and all(1<=d<=8 for d in requested_depths),'requested D must be 1..8')
    r=validated.record; b,t,v=r['binding'],r['trace'],r['verification']
    result=dict(schema=SCHEMA,trace_sha256=validated.sha256,request_key=b['request_key'],
        valid_for_state_key=b['state_key'],premises=deepcopy(r['premises']),entries=[],
        diagnostic_work={'CPU_reconstructed_selection_pool_entries':0},
        model_or_GPU_executed=False,actual_counterfactual_tested=False,
        next_state_hard_constraint_allowed=False,cost_or_throughput_claim=False)
    if v['kind']=='prefill_terminal':
        result['actual_observation']=dict(A_raw=None,G_tree=0,G_output=1,stop_reason=v['stop_reason'],censored=True)
        for d in requested_depths:
            e=_entry(r,d);e.update(stop_class='terminal_or_clipped',missing_evidence=['no_Draft_tree_executed'])
            e['G_tree']=_interval(0,0,0);e['G_output']=_interval(1,1,1);e['A_raw']['censored']=True
            result['entries'].append(e)
        return result
    nodes=_nodes(t);actual=_outcome(v['selected_raw_ids'],nodes,v['next_ids'],b)
    # The actual submitted count is recorded even if the final next-ID is absent.
    actual['G_output']=len(v['submitted_ids'])+len(v['terminal_bonus_ids'])
    censored=v['stop_reason'] in ('eos','limit')
    result['actual_observation']=dict(actual,stop_reason=v['stop_reason'],censored=censored,
        actual_D=t['actual_depth'], actual_verification_nodes=len(v['verified_ids']),
        posterior_recomputed=_posterior(v['retrieve_paths'],nodes,v['next_ids']) is not None)
    for d in requested_depths:
        e=_entry(r,d)
        if d==t['actual_depth']:
            _exact(e,actual,'observed_execution',censored=censored)
        elif d<t['actual_depth']:
            cp=t['checkpoints'][d-1]
            result['diagnostic_work']['CPU_reconstructed_selection_pool_entries']+=cp['pool_size']
            selected,reason=_select(cp,nodes,r)
            missing=_missing(r,COMMON)
            if reason:missing.append(reason)
            if selected is not None and not set(selected)<=set(v['verified_ids']):
                missing.append('shallower_tree_not_fully_verified')
                e['uncovered_node_ids']=sorted(set(selected)-set(v['verified_ids']))
            if selected is not None and not all(set(nodes[i]['path'])<=set(selected) for i in selected):
                missing.append('shallower_selection_not_ancestor_closed')
            if missing:e['missing_evidence']=missing
            else:
                # Author retrieval order is pool-ID order of leaves for greedy(None).
                path=_posterior(_paths(selected,nodes),nodes,v['next_ids'])
                if path is None:e['missing_evidence']=['missing_Target_next_ID_for_shallow_replay']
                else:
                    outcome=_outcome(path,nodes,v['next_ids'],b)
                    if not outcome['outcome_complete']:e['missing_evidence']=['missing_Target_next_ID_for_bonus']
                    else:
                        _exact(e,outcome,'reused_target_structural',censored=outcome['stop_reason'] is not None)
                        e['covered_node_ids']=selected
                        e['eligible_for_exact_update']=not censored and outcome['stop_reason'] is None
        elif censored:
            e.update(stop_class='terminal_or_clipped',missing_evidence=['EOS_or_limit_censors_depth_feedback'])
            e['A_raw']['censored']=True
        else:
            u=v['selected_raw_ids'][-1];y=v['next_ids'].get(str(u));cp=t['checkpoints'][-1]
            needed=list(COMMON);kind=None;witness=[u];reason=None
            if y is None:reason='missing_Target_next_ID_at_stop'
            else:
                children=[i for i in nodes if i and nodes[i]['parent']==u]
                correct=next((i for i in children if nodes[i]['token']==y),None)
                if u in cp['expanded_ids'] and correct is None:
                    kind='expanded_missing_child';needed+=['no_reexpand'];witness+=children
                elif correct is not None and correct not in v['verified_ids']:
                    kind='correct_child_filtered';needed+=['append_only_fixed_scores'];witness.append(correct)
                    dominators=[i for i in nodes if i and nodes[i]['score']>nodes[correct]['score']]
                    if len(dominators)>=CAP-1:witness+=dominators[:CAP-1]
                    elif _tie_proved(r):
                        ranked=sorted(range(1,len(nodes)),key=lambda i:(-nodes[i]['score'],i))
                        if correct in ranked[:CAP-1]:reason='fixed_order_does_not_certify_exclusion'
                        else:witness+=ranked[:CAP-1]
                    else:reason='unproved_pruning_tie_order'
                elif u in cp['eliminated_ids']:
                    kind='parent_permanently_eliminated';needed+=['no_resurrection']
                elif u in cp['frontier_ids']:
                    reason='expandable_frontier_leaf';e['stop_class']='frontier_unknown'
                else:reason='stop_structure_does_not_support_certificate'
            missing=_missing(r,needed)
            if reason:missing.append(reason)
            if missing:e['missing_evidence']=missing
            elif kind:
                upper=min(actual['A_raw'],d,CAP-1);g=min(b['remaining_tokens'],upper+1)
                e.update(feedback_type='upper_bound',claim_scope='conditional_deeper_bound',
                    certificate_kind=kind,stop_class=kind,witness_node_ids=witness,
                    eligible_for_upper_constraint=True)
                e['A_raw']=_interval(0,upper);e['G_tree']=_interval(1,g)
                e['G_output']=_interval(1,min(b['remaining_tokens'],g+1))
        result['entries'].append(e)
    return result
