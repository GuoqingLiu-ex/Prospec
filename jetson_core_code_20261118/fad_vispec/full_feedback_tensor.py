"""Conditional multi-depth feedback from the current accepted path.

This does not run Target or claim cross-shape floating-point equality. Shallow
values and deeper bounds are auxiliary, conditional observations. All dynamic
values remain on the search device until the caller's existing root readback.
The returned int64[20] replaces int64 root.item(): 160 rather than 8 logical
D2H bytes. Additional GPU operations, allocations and CPU work are not free.
"""
from __future__ import annotations

import torch

from .compact_search_feedback import Flag, Kind
from .compact_stop_tensor import classify_stop_tensor

UNKNOWN, CONDITIONAL_EXACT, CONDITIONAL_UPPER = 0, 1, 2
PACKET_ELEMENTS = 20


def _empty_aux(base):
    return base.new_zeros((8,)), base.new_full((8,), -1)


@torch.no_grad()
def build_feedback_packet(result, *, next_root, best, accepted_draft,
                          submitted_ids, remaining_tokens, request_key,
                          state_key, stop_ids):
    """Return root/local-kind/witness/flags, eight kinds, eight G_tree values.

    Called only on the original non-clipped nextroot path. The author already
    builds retrieve on CPU; reading that existing CPU object adds no device
    download. It is not permissible to move a CUDA retrieve tensor to CPU here.
    Existing posterior best/accepted/submitted host values bind the path.
    """
    result.search.validate(result.tree)
    retrieve = result.tree[1]
    if retrieve.device.type != 'cpu':
        raise ValueError('original host retrieve required; no implicit device download')
    classified = classify_stop_tensor(result, request_key=request_key,
        state_key=state_key, best=best, accepted_draft=accepted_draft,
        proposed_ids=submitted_ids, submitted_ids=submitted_ids,
        remaining_tokens=remaining_tokens, stop_ids=stop_ids, next_root=next_root)
    base = classified.packet
    if base is None:
        raise ValueError('clip-return must bypass feedback packet construction')
    kinds, values = _empty_aux(base)
    search = result.search
    raw = retrieve[best, :accepted_draft + 1].tolist()
    # No indexing a malformed path even when device validity will be false.
    if (not raw or raw[0] != 0 or any(type(i) is not int or i < 0 or
            i > search.selected.numel() for i in raw)):
        return torch.cat((base, kinds, values)).contiguous()
    zero = base.new_zeros(())
    path = torch.stack([zero if i == 0 else search.selected[i-1]+1 for i in raw])
    required = int(Flag.PATH_COVERED | Flag.FINITE_SCORES)
    valid = ((base[3] & required) == required) & (base[1] >= 1) & (base[1] <= 4)
    valid = valid & ((base[3] & int(Flag.CENSORED | Flag.INSUFFICIENT_EVIDENCE)) == 0)
    parents = torch.cat(search.parents).repeat_interleave(8)
    u = path[-1]
    actual_g = accepted_draft + 1
    failure = ((base[1] == int(Kind.CANDIDATE_MISSING)) |
               (base[1] == int(Kind.VERIFICATION_PRUNED)) |
               (base[1] == int(Kind.SEARCH_ELIMINATED)))

    for depth in range(1, 9):
        if depth == search.actual_depth:
            continue  # The host owns the actual observed value and censoring.
        if depth > search.actual_depth:
            eligible = valid & failure
            kinds[depth-1] = torch.where(eligible, CONDITIONAL_UPPER, UNKNOWN)
            values[depth-1] = torch.where(eligible, actual_g, -1)
            continue
        n = 8 + 64 * (depth - 1)
        budget = min(29, n)
        ranked = torch.topk(search.scores[:n], min(n, budget+1), sorted=True)
        selected = ranked.indices[:budget] + 1
        # An unstable boundary tie gives no identity guarantee for shallow TopB.
        strict = (ranked.values[budget-1] > ranked.values[budget]
                  if n > budget else base.new_ones((), dtype=torch.bool))
        selected_parents = parents[selected-1]
        closed = ((selected_parents == 0) |
                  (selected_parents[:, None] == selected[None, :]).any(-1)).all()
        present = (path[1:, None] == selected[None, :]).any(-1)
        prefix = present.long().cumprod(0).sum()
        # All steps before the first missing accepted edge were actually
        # verified. The missing edge fixes the stop under the stated hypothesis.
        missing = prefix < accepted_draft
        correct_child = ((parents[:n] == u) & (search.tokens[:n] == next_root))
        correct_is_selected = correct_child[selected-1].any()
        determined = missing | ~correct_is_selected
        eligible = valid & strict & closed & determined
        gain = torch.where(missing, prefix + 1, actual_g)
        eligible = eligible & (gain >= 1) & (gain <= depth + 1)
        kinds[depth-1] = torch.where(eligible, CONDITIONAL_EXACT, UNKNOWN)
        values[depth-1] = torch.where(eligible, gain, -1)
    return torch.cat((base, kinds, values)).contiguous()


def cost_scope():
    return dict(packet_elements=20, dtype='int64', logical_D2H_bytes=160,
        baseline_root_bytes=8, additional_logical_D2H_bytes=152,
        readback_API='caller replaces one root.item with one packet.tolist',
        actual_CUPTI_events=None, actual_DRAM_bytes=None,
        new_work=['P1 graph/stop validation', 'parent expansion and accepted path map',
            'up to D-1 shallow topk and ancestry/path checks', 'conditional vector fill and pack'],
        additional_Target_forward=0, cross_shape_exactness_proven=False,
        next_state_hard_guarantee=False)
