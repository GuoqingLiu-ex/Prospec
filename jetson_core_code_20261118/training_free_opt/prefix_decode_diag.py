"""Shared-prefix continuation. Free generation stays on its own clock.

Do not disable EOS to invent tokens past a natural baseline stop. Each call
rebuilds Draft state by running that candidate from this request's own pixels
with the baseline token prefix appended. Adaptor microbenchmarks are not a
substitute.
"""
from __future__ import annotations

import torch

DIAGNOSTIC_LENGTHS = (128, 256)
MAX_NEW = 512
EOS_TOKEN_IDS = (151645, 151643)


def extend_prompt_with_prefix(packed: dict, prefix_token_ids) -> dict:
    """Append baseline text tokens. Video tensors stay on this request."""
    prefix = [int(token) for token in prefix_token_ids]
    if not prefix or any(token < 0 for token in prefix):
        raise ValueError("prefix token ids must be nonnegative and nonempty")
    if any(token in EOS_TOKEN_IDS for token in prefix):
        raise ValueError("prefix already contains EOS; do not continue past the baseline stop")
    ids = packed.get("input_ids")
    if not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("input_ids must be a batch-one prompt")
    extra = torch.tensor([prefix], dtype=ids.dtype, device=ids.device)
    extended = dict(packed)
    extended["input_ids"] = torch.cat((ids, extra), 1)
    mask = packed.get("attention_mask")
    if mask is not None:
        if not torch.is_tensor(mask) or mask.shape != ids.shape:
            raise ValueError("attention_mask must match the prompt")
        ones = torch.ones((1, len(prefix)), dtype=mask.dtype, device=mask.device)
        extended["attention_mask"] = torch.cat((mask, ones), 1)
    types = packed.get("mm_token_type_ids")
    if types is not None:
        if not torch.is_tensor(types) or types.shape != ids.shape:
            raise ValueError("mm_token_type_ids must match the prompt")
        tail = types[:, -1:]
        extended["mm_token_type_ids"] = torch.cat((types, tail.expand(1, len(prefix))), 1)
    return extended


def execute_shared_prefix(generate_fn, packed, prefix_token_ids, continuation_tokens,
                          eos_token_ids=EOS_TOKEN_IDS):
    """Fresh Draft execution after this request's visual prefill plus the prefix.

    `generate_fn(extended_inputs, max_new_tokens, eos_token_ids)` must reset
    Draft KV itself, as generate_joint does. Continuation stops on EOS and is
    not padded to the budget.
    """
    if continuation_tokens not in DIAGNOSTIC_LENGTHS:
        raise ValueError("continuation workload must be 128 or 256")
    if len(prefix_token_ids) not in DIAGNOSTIC_LENGTHS:
        raise ValueError("shared prefix length must be 128 or 256")
    extended = extend_prompt_with_prefix(packed, prefix_token_ids)
    return generate_fn(extended, int(continuation_tokens), list(eos_token_ids))


def continuation_metrics(token_ids, rounds, eos_ids=EOS_TOKEN_IDS):
    """Separate clipped draft accepts from tokens that were actually committed.

    mean_accepted_draft_before_clip counts the pre-clip accepted_draft field.
    effective_draft_per_verify is effective_draft_tokens / verify_rounds.
    """
    eos = set(int(token) for token in eos_ids)
    ids = [int(token) for token in token_ids]
    tree = [row for row in rounds if row.get("kind") == "tree"]
    effective = 0
    for row in tree:
        submitted = row.get("submitted") or []
        effective += sum(int(token) not in eos for token in submitted[1:])
    verify = len(tree)
    return {
        "continuation_token_ids": ids,
        "continuation_tokens": len(ids),
        "non_eos_continuation_tokens": sum(token not in eos for token in ids),
        "verify_rounds": verify,
        "effective_draft_tokens": effective,
        "effective_draft_per_verify": (effective / verify if verify else None),
        "mean_accepted_draft_before_clip": (
            sum(int(row["accepted_draft"]) for row in tree) / verify if verify else None
        ),
        "accepted_draft_before_clip": [int(row["accepted_draft"]) for row in tree],
        "first_shared_round": next((row for row in rounds if row.get("kind") == "tree"), None),
    }


def group_free_continuations(records):
    """Split free continuations that stayed identical from those that diverged."""
    grouped = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        key = (record["request_id"], int(record["prefix_tokens"]))
        grouped.setdefault(key, []).append(record)
    rows = []
    for (request_id, prefix_tokens), items in grouped.items():
        signatures = {tuple(item["continuation_token_ids"]) for item in items}
        rows.append({
            "request_id": request_id,
            "prefix_tokens": prefix_tokens,
            "arms": [item["arm"] for item in items],
            "group": "same_continuation" if len(signatures) == 1 else "diverged",
        })
    return rows


def execute_fixed_target_trajectory(*_args, **_kwargs):
    """Not the free-continuation replay.

    A fixed trajectory has to rebuild Draft state and then force the baseline's
    later tokens. Free generation after the shared prefix is a different diagnostic.
    """
    raise NotImplementedError(
        "Fixed Target trajectory replay is separate from free continuation after a shared prefix."
    )


def plan_diagnostics(baseline_rows):
    """Select requests whose baseline naturally reached the continuation length."""
    selected = {128: [], 256: []}
    for row in baseline_rows:
        n = int(row.get("output_tokens", row.get("num_tokens")))
        natural = bool(row.get("natural_eos", row.get("stop_reason") == "eos"))
        if n >= 128:
            selected[128].append(row["request_id"])
        if n >= 256:
            selected[256].append(row["request_id"])
        row_ok = {
            "request_id": row["request_id"],
            "baseline_output_tokens": n,
            "natural_eos": natural,
            "hit_max_new": n >= MAX_NEW and not natural,
            "eligible_prefix_128": n > 128 or (n == 128 and not natural),
            "eligible_prefix_256": n > 256 or (n == 256 and not natural),
        }
        yield row_ok, selected
    return selected
