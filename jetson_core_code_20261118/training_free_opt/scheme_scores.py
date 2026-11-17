"""FP32 Q/K scores used by SchemeRuntime. Distinct from FactorizedAdaptor.score_tokens."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def scheme_runtime_directions(adaptor):
    """One-time FP32 composition, matching SchemeRuntime.__init__."""
    query = adaptor.q.detach().float().permute(1, 0, 2)
    weight = adaptor.k_proj.weight.detach().float().reshape(
        adaptor.num_heads, adaptor.head_dim, adaptor.hidden_size)
    directions = torch.einsum("hqd,hdi->hqi", query, weight).reshape(-1, adaptor.hidden_size)
    directions.div_(math.sqrt(adaptor.head_dim))
    bias = adaptor.k_proj.bias
    score_bias = None if bias is None else torch.einsum(
        "hqd,hd->hq", query, bias.detach().float().reshape(adaptor.num_heads, adaptor.head_dim)
    ).reshape(-1) / math.sqrt(adaptor.head_dim)
    return directions, score_bias


def scheme_runtime_scores(directions, score_bias, hidden):
    """Per-call score. hidden is cast to FP32 here, as SchemeRuntime.scores does."""
    return F.linear(hidden.float(), directions, score_bias)
