"""Host-only FAD estimator with separate actual and conditional channels.

No model, Tensor, GPU, file or network API. A-only calibration is caller-bound
evidence, not attested measurements. Conditional feedback is a soft statistical
signal, never an observed label or a restriction on the next state's actions.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
import re

DEPTHS = tuple(range(1, 9))
MODES = ('actual_only', 'failure_only', 'full')
STOP_KINDS = ('candidate_missing', 'verification_pruned', 'search_eliminated',
              'frontier_unknown', 'unknown', 'terminal')
SCHEMA = 'fad_full_controller_v1'
FEEDBACK_SCHEMA = 'fad_full_feedback_v1'
COST_SCOPE = 'draft_plus_target_verify_and_commit_round'


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _number(x, *, positive=False):
    return type(x) in (int, float) and math.isfinite(x) and (x > 0 if positive else x >= 0)


def _sha(x):
    return type(x) is str and re.fullmatch('[0-9a-f]{64}', x) is not None


def _digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _plain(x, active=None):
    if x is None or type(x) in (str, int, bool):
        return
    if type(x) is float:
        _need(math.isfinite(x), 'nonfinite metadata'); return
    _need(type(x) in (dict, list, tuple), 'plain host metadata required')
    active = set() if active is None else active
    _need(id(x) not in active, 'cyclic metadata')
    active.add(id(x))
    if type(x) is dict:
        _need(all(type(k) is str for k in x), 'metadata keys must be strings')
        values = x.values()
    else:
        values = x
    for v in values:
        _plain(v, active)
    active.remove(id(x))


def _token(x):
    """Immutable small decision receipt; no hash or table/history copy."""
    if type(x) is dict:
        return tuple((k, _token(v)) for k, v in sorted(x.items()))
    if type(x) in (list, tuple):
        return tuple(_token(v) for v in x)
    return (type(x).__name__, x)


def _edges(x):
    return (type(x) is tuple and all(type(v) is int and v > 0 for v in x)
            and tuple(sorted(set(x))) == x)


@dataclass(frozen=True)
class FullControllerConfig:
    # Explicit constructor arguments: no silently selected strategy weights.
    context_edges: tuple
    acceptance_edges: tuple
    exploration_probability: float
    hysteresis_relative: float
    conditional_exact_weight: float
    upper_weight: float
    prior_mass: float
    use_ipw: bool
    ipw_clip: float
    seed: int
    default_depth: int = 4
    allowed_depths: tuple = (2, 3, 4, 5, 6, 7, 8)
    online_updates: bool = True

    def __post_init__(self):
        _need(_edges(self.context_edges) and _edges(self.acceptance_edges), 'ordered positive bucket edges')
        _need(type(self.allowed_depths) in (tuple, list) and self.allowed_depths
              and all(type(d) is int and d in DEPTHS for d in self.allowed_depths)
              and len(set(self.allowed_depths)) == len(self.allowed_depths), 'unique allowed depths D1..8')
        object.__setattr__(self, 'allowed_depths', tuple(sorted(self.allowed_depths)))
        _need(type(self.default_depth) is int and self.default_depth == 4
              and self.default_depth in self.allowed_depths, 'D4 fallback must remain allowed')
        _need(_number(self.exploration_probability) and self.exploration_probability <= 1, 'epsilon in [0,1]')
        for name in ('hysteresis_relative', 'conditional_exact_weight', 'upper_weight'):
            _need(_number(getattr(self, name)), 'finite nonnegative ' + name)
        _need(_number(self.prior_mass, positive=True), 'positive frozen prior mass')
        _need(type(self.use_ipw) is bool and type(self.online_updates) is bool, 'explicit boolean update options')
        _need(_number(self.ipw_clip) and self.ipw_clip >= 1, 'IPW cap >=1')
        _need(type(self.seed) is int and self.seed >= 0, 'nonnegative seed')


def shape_id(visual_tokens, query_tokens):
    _need(type(visual_tokens) is int and visual_tokens >= 0
          and type(query_tokens) is int and query_tokens >= 1, 'host visual/raw selected-hidden lengths')
    return f'v{visual_tokens}_q{query_tokens}'


def cost_bucket_key(shape, first_tree, context_length, context_edges):
    _need(type(shape) is str and re.fullmatch(r'v[0-9]+_q[1-9][0-9]*', shape), 'shape ID vM_qT required')
    _need(type(first_tree) is bool and type(context_length) is int and context_length >= 1, 'host first/context')
    _need(_edges(context_edges), 'context edges')
    return f's{shape}|f{int(first_tree)}|c{bisect_right(context_edges, context_length)}'


def feature_bucket_key(base, previous_stop_kind, previous_A, acceptance_edges):
    _need(type(base) is str and _edges(acceptance_edges), 'base/acceptance edges')
    if previous_stop_kind == 'start':
        _need(previous_A is None, 'start has no previous A'); suffix = 'none'
    else:
        _need(previous_stop_kind in STOP_KINDS and previous_stop_kind != 'terminal'
              and type(previous_A) is int and 0 <= previous_A <= 8, 'previous stop/A')
        suffix = str(bisect_right(acceptance_edges, previous_A))
    return f'{base}|k{previous_stop_kind}|a{suffix}'


@dataclass
class _Moment:
    samples: int = 0
    weight: float = 0.0
    weighted_sum: float = 0.0

    def add(self, value, weight=1.0):
        self.samples += 1
        self.weight += weight
        self.weighted_sum += weight * value


@dataclass
class _Channels:
    actual: _Moment = field(default_factory=_Moment)
    conditional: _Moment = field(default_factory=_Moment)
    upper_histogram: dict = field(default_factory=dict)


@dataclass
class _Request:
    rng: random.Random
    derived_seed: int
    estimates: dict = field(default_factory=dict)
    last_depth: dict = field(default_factory=dict)
    seen: set = field(default_factory=set)
    pending: tuple | None = None
    previous_kind: str = 'start'
    previous_A: int | None = None
    terminal: bool = False
    decisions: int = 0
    observations: int = 0
    feedback_counts: dict = field(default_factory=lambda: dict(observed=0, conditional_exact=0,
        conditional_upper=0, unknown=0, censored=0))


class FullController:
    def __init__(self, calibration, *, config, mode, phase, device_uuid,
                 source_sha256, precision_sha256, frozen_policy_sha256=None,
                 allow_logical_fixtures=False):
        _need(isinstance(config, FullControllerConfig) and mode in MODES, 'explicit config/mode')
        _need(phase in ('calibration_A', 'holdout_B', 'test_T200'), 'A/B/T200 phase')
        _plain(calibration)
        t = deepcopy(calibration)  # once at setup; no per-round calibration copy/hash
        _need(type(t) is dict and t.get('schema') == 'fad_full_calibration_v1'
              and t.get('split') == 'D80-A', 'A-only calibration schema')
        _need(type(allow_logical_fixtures) is bool, 'explicit fixture flag')
        logical = t.get('kind') == 'logical_fixture'
        _need((logical and allow_logical_fixtures and t.get('unit') == 'logical_units') or
              (not logical and t.get('kind') == 'measured_device_round_seconds' and t.get('unit') == 'seconds'),
              'measured cost or explicitly enabled logical fixture required')
        bindings = dict(device_uuid=device_uuid, source_sha256=source_sha256, precision_sha256=precision_sha256)
        _need(type(device_uuid) is str and device_uuid and _sha(source_sha256) and _sha(precision_sha256)
              and all(t.get(k) == v for k, v in bindings.items()), 'cost device/source/precision mismatch')
        _need(_sha(t.get('measurement_receipt_sha256')) and _sha(t.get('prior_receipt_sha256')), 'measurement/prior evidence hash')
        _need(t.get('context_edges') == list(config.context_edges)
              and t.get('acceptance_edges') == list(config.acceptance_edges), 'calibration/config bucket edges differ')
        _need(t.get('cost_scope') == COST_SCOPE and _number(t.get('estimated_additional_cost_s'))
              and t.get('deployment_cost_validated') is False, 'explicit provisional core + estimated added cost required')
        shapes = t.get('shape_metadata')
        _need(type(shapes) is dict and shapes, 'shape metadata required')
        for name, row in shapes.items():
            _need(type(row) is dict and set(row) == {'visual_tokens', 'query_tokens'}
                  and name == shape_id(row['visual_tokens'], row['query_tokens']), 'shape metadata differs')
        costs = t.get('costs'); priors = t.get('actual_priors')
        _need(type(costs) is dict and costs and type(priors) is dict, 'cost/prior mappings required')
        base_pattern = re.compile(r's(v[0-9]+_q[1-9][0-9]*)\|f([01])\|c([0-9]+)')
        for key, by_d in costs.items():
            match = base_pattern.fullmatch(key)
            _need(match is not None and match[1] in shapes and int(match[3]) <= len(config.context_edges), 'invalid cost bucket')
            _need(type(by_d) is dict and {str(d) for d in config.allowed_depths} <= set(by_d)
                  and set(by_d) <= {str(d) for d in DEPTHS}, 'cost bucket must cover allowed depths')
            for row in by_d.values():
                _need(type(row) is dict and set(row) == {'cost', 'samples'} and _number(row['cost'], positive=True)
                      and type(row['samples']) is int and row['samples'] >= 1, 'invalid measured cost row')
        for key, by_d in priors.items():
            base, *typed = key.split('|k')
            _need(base in costs and len(typed) <= 1, 'prior has no measured base bucket')
            if typed:
                _need(any(key == feature_bucket_key(base, k, a, config.acceptance_edges)
                          for k, values in [('start', [None])] + [(k, range(9)) for k in STOP_KINDS if k != 'terminal']
                          for a in values), 'invalid typed prior bucket')
            _need(type(by_d) is dict and set(by_d) <= {str(d) for d in DEPTHS}, 'prior depths')
            for d, row in by_d.items():
                _need(type(row) is dict and set(row) == {'sum_G', 'samples'}
                      and type(row['samples']) is int and row['samples'] >= 1
                      and type(row['sum_G']) is int and row['samples'] <= row['sum_G'] <= row['samples']*(int(d)+1),
                      'actual prior must contain valid uncensored G_tree totals')
        self._config, self._mode, self._phase = config, mode, phase
        self._logical_fixture = logical
        self._costs, self._priors = costs, priors
        self._additional = t['estimated_additional_cost_s']
        self._requests = {}
        self._policy = dict(schema=SCHEMA, config=asdict(config), mode=mode, calibration_sha256=_digest(t),
            bindings=bindings, cost_scope=COST_SCOPE, estimated_additional_cost_s=self._additional,
            deployment_cost_validated=False, prior_scope='A-only actual uncensored G_tree',
            upper_is_truth=False, upper_hard_disables_actions=False, conditional_exact_is_observed_truth=False,
            soft_upper_formula='w/(1+w) * mean(max(0, predicted_G - conditional_upper))',
            offline_fixture=logical, no_next_state_guarantee=True)
        self._policy_sha256 = _digest(self._policy)
        if phase != 'calibration_A':
            _need(_sha(frozen_policy_sha256) and frozen_policy_sha256 == self.policy_sha256, 'B/T200 frozen policy mismatch')
        elif frozen_policy_sha256 is not None:
            _need(frozen_policy_sha256 == self.policy_sha256, 'frozen policy mismatch')

    @property
    def config(self): return self._config

    @property
    def mode(self): return self._mode

    @property
    def phase(self): return self._phase

    @property
    def logical_fixture(self): return self._logical_fixture

    @property
    def policy_sha256(self): return self._policy_sha256

    def policy_receipt(self):
        return dict(deepcopy(self._policy), policy_sha256=self.policy_sha256, phase=self.phase,
            source_and_measurement_independently_verified=False, hardware_benefit_claim=False,
            cost_scope_detail='C_core includes first-tree adaptor; excludes actual new feedback/selector cost. Added estimate is provisional.',
            execution_scope='host-only estimator; no deployment, zero-I/O or throughput claim')

    def begin_request(self, request_key):
        _need(type(request_key) is str and request_key and request_key not in self._requests, 'new request key required')
        seed = int.from_bytes(hashlib.sha256(f'{self.config.seed}:{request_key}'.encode()).digest()[:8], 'big')
        self._requests[request_key] = _Request(random.Random(seed), seed)
        return dict(request_key=request_key, derived_seed=seed, policy_sha256=self.policy_sha256)

    def _request(self, key):
        _need(type(key) is str and key in self._requests, 'unknown request')
        return self._requests[key]

    def _channels(self, r, key, d):
        return r.estimates.setdefault(key, {}).setdefault(d, _Channels())

    def _estimate(self, r, base, typed, d, remaining):
        key = base if self.mode == 'actual_only' else typed
        pkey = key if str(d) in self._priors.get(key, {}) else base
        prior = self._priors.get(pkey, {}).get(str(d), dict(samples=0, sum_G=0))
        channel = r.estimates.get(key, {}).get(d, _Channels())
        prior_weight = self.config.prior_mass if prior['samples'] else 0.
        weight = prior_weight + channel.actual.weight
        total = (prior_weight*prior['sum_G']/prior['samples'] if prior['samples'] else 0.) + channel.actual.weighted_sum
        if weight == 0:
            return None
        actual_mean = total / weight
        predicted = actual_mean
        mix_weight = self.config.conditional_exact_weight if self.mode == 'full' else 0.
        if mix_weight and channel.conditional.weight:
            predicted = (total + mix_weight*channel.conditional.weighted_sum) / (weight + mix_weight*channel.conditional.weight)
        penalty = 0.
        if self.mode == 'full' and channel.upper_histogram:
            count = sum(channel.upper_histogram.values())
            gap = sum(n*max(0., predicted-u) for u, n in channel.upper_histogram.items())/count
            penalty = (self.config.upper_weight/(1+self.config.upper_weight))*gap
            predicted -= penalty
        predicted = max(1., min(d+1., predicted))
        before_cap = predicted
        predicted = min(predicted, remaining)
        cost = self._costs[base][str(d)]['cost'] + self._additional
        return dict(actual_mean_G=actual_mean, predicted_G=predicted, predicted_before_remaining_cap=before_cap,
            cost=cost, utility=predicted/cost,
            prior_bucket=pkey if prior['samples'] else None, prior_raw_samples=prior['samples'],
            prior_effective_weight=prior_weight, actual_online_weight=channel.actual.weight,
            conditional_samples=channel.conditional.samples,
            conditional_upper_samples=sum(channel.upper_histogram.values()), soft_upper_penalty=penalty)

    def select(self, request_key, *, state_key, shape_id, first_tree, context_length, remaining_tokens):
        r = self._request(request_key)
        _need(not r.terminal and r.pending is None, 'terminal or unresolved previous decision')
        _need(type(state_key) is str and state_key and state_key not in r.seen, 'fresh state key required')
        _need(type(remaining_tokens) is int and 2 <= remaining_tokens <= 512, 'no tree choice for terminal/invalid remaining count')
        base = cost_bucket_key(shape_id, first_tree, context_length, self.config.context_edges)
        typed = feature_bucket_key(base, r.previous_kind, r.previous_A, self.config.acceptance_edges)
        estimates = {}
        if base in self._costs:
            estimates = {d: self._estimate(r, base, typed, d, remaining_tokens) for d in self.config.allowed_depths}
            estimates = {d: e for d, e in estimates.items() if e is not None}
        default = self.config.default_depth
        exploit, held = default, False
        reason = 'missing_cost_D4_bypass' if base not in self._costs else 'no_actual_prior_D4'
        if default in estimates:
            previous = r.last_depth.get(base, default)
            anchor = previous if previous in estimates else default
            best = min(estimates, key=lambda d: (-estimates[d]['utility'], d))
            if estimates[best]['utility'] > estimates[anchor]['utility']*(1+self.config.hysteresis_relative):
                exploit, reason = best, 'estimated_G_per_provisional_C'
            else:
                exploit, reason, held = anchor, 'hysteresis', best != anchor
        epsilon = float(self.config.exploration_probability) if base in self._costs else 0.
        draw = r.rng.random() if epsilon else None
        explore = draw is not None and draw < epsilon
        chosen = r.rng.choice(self.config.allowed_depths) if explore else exploit
        probability = epsilon/len(self.config.allowed_depths) + (1-epsilon if chosen == exploit else 0.)
        r.decisions += 1; r.seen.add(state_key); r.last_depth[base] = chosen
        # The small decision token never copies the table or feedback history.
        decision = dict(schema=SCHEMA, request_key=request_key, state_key=state_key,
            decision_id=f'{request_key}:{r.decisions}', actual_depth=chosen, mode=self.mode,
            shape_id=shape_id, first_tree=first_tree, context_length=context_length,
            remaining_tokens=remaining_tokens, cost_bucket=base, feature_bucket=typed,
            previous_stop_kind=r.previous_kind, previous_A=r.previous_A,
            exploitation_depth=exploit, exploration=explore, exploration_probability=epsilon,
            uniform_exploration_depths=list(self.config.allowed_depths),
            selected_action_probability=probability, random_draw=draw, reason=reason, hysteresis_held=held,
            estimates={str(d): e for d, e in estimates.items()}, policy_sha256=self.policy_sha256,
            deployment_cost_validated=False, no_next_state_guarantee=True,
            conditional_actions_hard_disabled=False)
        r.pending = _token(decision)
        return decision

    def _feedback(self, decision, f):
        _plain(f)
        _need(type(f) is dict and f.get('schema') == FEEDBACK_SCHEMA, 'full feedback schema')
        _need(f.get('request_key') == decision['request_key'] and f.get('state_key') == decision['state_key'], 'feedback state/request mismatch')
        _need(type(f.get('actual_D')) is int and f['actual_D'] == decision['actual_depth'], 'actual depth mismatch')
        _need(f.get('evidence_scope') == 'observed_execution' and f.get('no_next_state_guarantee') is True,
              'observed scope/no-next-state declaration required')
        a, g, total, stop, censored = (f.get(k) for k in ('A_raw', 'G_tree', 'G_output', 'stop_reason', 'censored'))
        _need(type(a) is int and 0 <= a <= f['actual_D'] and type(g) is int and 1 <= g <= a+1
              and type(total) is int and g <= total <= decision['remaining_tokens'], 'actual A/G/output extent')
        _need(type(censored) is bool and stop in (None, 'eos', 'limit') and censored == (stop is not None), 'stop/censor mismatch')
        _need(censored or g == a+1, 'uncensored G_tree must equal A_raw+1')
        _need((stop != 'limit' or total == decision['remaining_tokens'])
              and (censored or total < decision['remaining_tokens']), 'limit/remaining mismatch')
        bonus = f.get('terminal_bonus_ids')
        _need(type(bonus) is list and len(bonus) <= 1 and all(type(x) is int and x >= 0 for x in bonus)
              and total == g+len(bonus) and (not bonus or (censored and g == a+1)), 'terminal bonus/count mismatch')
        kind = f.get('stop_kind')
        _need(kind in STOP_KINDS and (kind == 'terminal') == censored, 'local stop kind/censor mismatch')
        entries = f.get('entries')
        _need(type(entries) is list and len(entries) == 8 and all(type(e) is dict for e in entries), 'eight typed entries required')
        _need(all(type(e.get('depth')) is int and e['depth'] in DEPTHS for e in entries)
              and {e['depth'] for e in entries} == set(DEPTHS), 'unique depth entries D1..8')
        parsed = []
        for e in entries:
            d, k, v, scope = (e.get(n) for n in ('depth', 'kind', 'value', 'evidence_scope'))
            _need(k in ('observed', 'conditional_exact', 'conditional_upper', 'unknown'), 'typed evidence kind')
            if d == f['actual_D']:
                _need(k == 'observed' and type(v) is int and v == g and scope == 'observed_execution', 'actual entry mismatch')
            elif k == 'unknown':
                _need(v is None and scope == 'unobserved', 'unknown cannot be a value')
            else:
                _need(not censored and k in ('conditional_exact', 'conditional_upper')
                      and scope == 'same_state_structural_conditional', 'conditional is not observed truth/terminal evidence')
                _need(type(v) is int and 1 <= v <= min(d+1, decision['remaining_tokens']), 'conditional G_tree extent')
                if k == 'conditional_upper':
                    _need(d > f['actual_D'] and v <= g and kind in
                          ('candidate_missing', 'verification_pruned', 'search_eliminated'), 'upper depth/local failure/value mismatch')
                else:
                    _need(d < f['actual_D'], 'conditional exact must be a shallower depth')
            parsed.append((d, k, v))
        return parsed

    def observe(self, decision, feedback):
        _plain(decision)
        _need(type(decision) is dict, 'decision receipt')
        r = self._request(decision.get('request_key'))
        _need(r.pending is not None and _token(decision) == r.pending, 'foreign/mutated/repeated decision')
        parsed = self._feedback(decision, feedback)  # validate all before any mutation
        censored = feedback['censored']
        probability = decision['selected_action_probability']
        raw_weight = 1/probability if self.config.use_ipw else 1.
        weight = min(raw_weight, self.config.ipw_clip) if self.config.use_ipw else 1.
        keys = (decision['cost_bucket'], decision['feature_bucket'])
        actual_updated, conditional_updated, upper_updated = [], [], []
        for d, kind, value in parsed:
            r.feedback_counts[kind] += 1
            if censored or not self.config.online_updates or d not in self.config.allowed_depths:
                continue
            if kind == 'observed':
                for key in keys:
                    self._channels(r, key, d).actual.add(value, weight)
                actual_updated.append(d)
            elif self.mode == 'full' and kind == 'conditional_exact':
                self._channels(r, keys[1], d).conditional.add(value)
                conditional_updated.append(d)
            elif self.mode == 'full' and kind == 'conditional_upper':
                hist = self._channels(r, keys[1], d).upper_histogram
                hist[value] = hist.get(value, 0)+1
                upper_updated.append(d)
        r.observations += 1
        if censored:
            r.feedback_counts['censored'] += 1
            r.terminal = True
        r.previous_kind, r.previous_A = feedback['stop_kind'], feedback['A_raw']
        r.pending = None
        return dict(decision_id=decision['decision_id'], actual_updated=actual_updated,
            conditional_exact_updated=conditional_updated, conditional_upper_updated=upper_updated,
            actual_update_weight=weight if actual_updated else None,
            untruncated_actual_ipw=raw_weight if actual_updated else None,
            actual_ipw_clipped=bool(actual_updated and weight < raw_weight),
            conditional_propensity_correction=False, censored=censored,
            conditional_updates_are_truth=False, no_next_state_guarantee=True,
            hard_disabled_depths=[], deployment_cost_validated=False)

    def cancel(self, decision, *, reason):
        _plain(decision)
        _need(type(decision) is dict, 'decision receipt')
        r = self._request(decision.get('request_key'))
        _need(r.pending == _token(decision) and type(reason) is str and reason, 'pending cancellation/reason')
        r.pending = None
        return dict(decision_id=decision['decision_id'], cancelled=True, reason=reason, statistics_updated=False)

    def request_summary(self, request_key):
        r = self._request(request_key)
        return dict(request_key=request_key, decisions=r.decisions, observations=r.observations,
            terminal=r.terminal, pending=r.pending is not None, policy_sha256=self.policy_sha256,
            feedback_counts=dict(r.feedback_counts), online_estimates={key: {str(d): dict(
                actual=asdict(s.actual), conditional_exact=asdict(s.conditional),
                conditional_upper_histogram=dict(s.upper_histogram)) for d, s in by_d.items()}
                for key, by_d in r.estimates.items()},
            no_cross_request_online_updates=True, deployment_cost_validated=False,
            full_method_claim='typed CPU estimator only; production evidence, integration and net benefit unvalidated')

    def end_request(self, request_key):
        _need(self._request(request_key).pending is None, 'pending decision at request end')
        result = self.request_summary(request_key)
        del self._requests[request_key]
        return result
