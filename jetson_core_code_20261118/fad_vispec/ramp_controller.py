"""Small deterministic host policy; conditional feedback remains a proxy.

Default policy: cap D6 (bounds D3..6), initial hold two completed nonterminal
rounds. Three consecutive hard failures with equal conditional G at D-1 lower
the cap by one. Two consecutive frontier stops with A_raw == D request one
D+1 probe. Both actions require the current hold to have elapsed. Evidence may
accumulate during the hold. A probe retains the higher cap only when actual G
exceeds conditional G at its base depth by at least one. Every cap change or
completed probe resets evidence and starts a four-round residence period.

A terminal observation never learns or advances residence. Cancellation does
not consume evidence; an uncompleted queued probe remains available. Each
request owns its cap, counters, pending receipt and seen state keys. There is
no cost table, random exploration, Tensor, per-round hash, or feedback history.
The inherited full-feedback validator is used without its estimator state.
These rules are a policy based on conditional signals, not counterfactual
truth or restrictions on the validity of another depth in a future state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import re

from .full_controller import FullController, _digest, _need, _plain, _token

SCHEMA = 'fad_ramp_controller_v1'
HARD_FAILURES = ('candidate_missing', 'verification_pruned', 'search_eliminated')


@dataclass(frozen=True)
class RampConfig:
    initial_depth: int = 6
    min_depth: int = 3
    max_depth: int = 6
    initial_hold: int = 2
    min_residence: int = 4
    hard_failure_streak: int = 3
    frontier_streak: int = 2

    def __post_init__(self):
        _need(all(type(v) is int for v in asdict(self).values()), 'integer ramp settings required')
        _need(3 <= self.min_depth <= self.initial_depth <= self.max_depth <= 6,
              'ramp depths must satisfy 3 <= min <= initial <= max <= 6')
        _need(all(getattr(self, k) > 0 for k in
                  ('initial_hold', 'min_residence', 'hard_failure_streak', 'frontier_streak')),
              'positive hold and evidence counts required')


@dataclass
class _Request:
    cap: int
    hold: int
    residence: int = 0
    hard_streak: int = 0
    frontier_streak: int = 0
    probe_queued: bool = False
    pending: tuple | None = None
    terminal: bool = False
    decisions: int = 0
    observations: int = 0
    cancellations: int = 0
    descents: int = 0
    probes_completed: int = 0
    probes_retained: int = 0
    seen: set = field(default_factory=set)


class RampController:
    def __init__(self, *, config=None):
        self._config = RampConfig() if config is None else config
        _need(type(self._config) is RampConfig, 'RampConfig required')
        self._requests = {}
        self._policy = dict(schema=SCHEMA, config=asdict(self.config), mode='ramp',
            initial_hold_unit='completed_nonterminal_rounds_at_cap',
            cap_change_and_probe_hold_unit='completed_nonterminal_rounds_at_cap',
            cost_table_used=False, random_exploration=False,
            conditional_is_counterfactual_truth=False, conditional_actions_hard_disabled=False,
            no_next_state_guarantee=True, no_cross_request_online_updates=True,
            execution_scope='host-only state machine; full feedback validation retained',
            deployment_cost_validated=False, hardware_benefit_claim=False)
        self._policy_sha256 = _digest(self._policy)  # setup only

    @property
    def config(self): return self._config

    @property
    def mode(self): return 'ramp'

    @property
    def policy_sha256(self): return self._policy_sha256

    def policy_receipt(self):
        return dict(deepcopy(self._policy), policy_sha256=self.policy_sha256)

    def begin_request(self, request_key):
        _need(type(request_key) is str and request_key and request_key not in self._requests,
              'new request key required')
        self._requests[request_key] = _Request(self.config.initial_depth, self.config.initial_hold)
        return dict(request_key=request_key, policy_sha256=self.policy_sha256,
                    initial_depth=self.config.initial_depth, initial_hold=self.config.initial_hold)

    def _request(self, key):
        _need(type(key) is str and key in self._requests, 'unknown request')
        return self._requests[key]

    def select(self, request_key, *, state_key, shape_id, first_tree, context_length, remaining_tokens):
        r = self._request(request_key)
        _need(not r.terminal and r.pending is None, 'terminal or unresolved previous decision')
        _need(type(state_key) is str and state_key and state_key not in r.seen, 'fresh state key required')
        _need(type(shape_id) is str and re.fullmatch(r'v[0-9]+_q[1-9][0-9]*', shape_id)
              and type(first_tree) is bool and type(context_length) is int and context_length >= 1,
              'host shape/first/context required')
        _need(type(remaining_tokens) is int and 2 <= remaining_tokens <= 512,
              'no tree choice for terminal/invalid remaining count')
        r.decisions += 1
        r.seen.add(state_key)
        decision = dict(schema=SCHEMA, request_key=request_key, state_key=state_key,
            decision_id=f'{request_key}:{r.decisions}', actual_depth=r.cap + int(r.probe_queued),
            remaining_tokens=remaining_tokens, mode='ramp', base_depth=r.cap,
            probe=r.probe_queued, residence_completed=r.residence, hold_required=r.hold,
            hard_streak=r.hard_streak, frontier_streak=r.frontier_streak,
            selected_action_probability=1., reason='one_step_probe' if r.probe_queued else 'current_cap',
            policy_sha256=self.policy_sha256)
        r.pending = _token(decision)
        return decision

    def _pending(self, decision):
        _plain(decision)
        _need(type(decision) is dict, 'decision receipt')
        r = self._request(decision.get('request_key'))
        _need(r.pending is not None and _token(decision) == r.pending,
              'foreign/mutated/repeated decision')
        return r

    def _reset(self, r):
        r.residence = r.hard_streak = r.frontier_streak = 0
        r.probe_queued = False
        r.hold = self.config.min_residence

    def observe(self, decision, feedback):
        r = self._pending(decision)
        # Validate the complete producer contract before changing any state.
        parsed = FullController._feedback(self, decision, feedback)
        base, before = decision['base_depth'], r.cap
        exact = {d: v for d, kind, v in parsed if kind == 'conditional_exact'}
        censored = feedback['censored']
        action, learned = 'none', False
        if censored:
            r.terminal = True
            r.probe_queued = False
            action = 'terminal_no_learning'
        elif decision['probe']:
            learned = True
            success = (feedback['stop_kind'] != 'unknown' and base in exact
                       and feedback['G_tree'] >= exact[base] + 1)
            r.cap = decision['actual_depth'] if success else base
            r.probes_completed += 1
            r.probes_retained += int(success)
            self._reset(r)
            action = 'probe_retained' if success else 'probe_returned_to_base'
        else:
            learned = True
            r.residence += 1
            hard = feedback['stop_kind'] in HARD_FAILURES and exact.get(base-1) == feedback['G_tree']
            frontier = feedback['stop_kind'] == 'frontier_unknown' and feedback['A_raw'] == base
            r.hard_streak = r.hard_streak + 1 if hard else 0
            r.frontier_streak = r.frontier_streak + 1 if frontier else 0
            if r.residence >= r.hold:
                if r.hard_streak >= self.config.hard_failure_streak and base > self.config.min_depth:
                    r.cap -= 1
                    r.descents += 1
                    self._reset(r)
                    action = 'cap_decreased'
                elif r.frontier_streak >= self.config.frontier_streak and base < self.config.max_depth:
                    r.probe_queued = True
                    action = 'probe_queued'
        r.observations += 1
        r.pending = None
        return dict(decision_id=decision['decision_id'], action=action,
            cap_before=before, cap_after=r.cap, probe_queued=r.probe_queued,
            residence_completed=r.residence, hold_required=r.hold,
            hard_streak=r.hard_streak, frontier_streak=r.frontier_streak,
            censored=censored, policy_state_updated=learned,
            conditional_updates_are_truth=False, no_next_state_guarantee=True)

    def cancel(self, decision, *, reason):
        r = self._pending(decision)
        _need(type(reason) is str and reason, 'nonempty cancellation reason')
        r.pending = None
        r.cancellations += 1
        return dict(decision_id=decision['decision_id'], cancelled=True,
                    reason=reason, statistics_updated=False)

    def request_summary(self, request_key):
        r = self._request(request_key)
        return dict(request_key=request_key, decisions=r.decisions, observations=r.observations,
            cancellations=r.cancellations, terminal=r.terminal, pending=r.pending is not None,
            cap=r.cap, probe_queued=r.probe_queued, residence_completed=r.residence,
            hold_required=r.hold, hard_streak=r.hard_streak, frontier_streak=r.frontier_streak,
            descents=r.descents, probes_completed=r.probes_completed, probes_retained=r.probes_retained,
            policy_sha256=self.policy_sha256, no_cross_request_online_updates=True,
            conditional_is_counterfactual_truth=False, no_next_state_guarantee=True)

    def end_request(self, request_key):
        _need(self._request(request_key).pending is None, 'pending decision at request end')
        result = self.request_summary(request_key)
        del self._requests[request_key]
        return result
