"""Same merge math, cached 2x2 topology. Does not skip score/select/aggregate."""
from __future__ import annotations

from compressors import _topology, compress_tokens as _compress_tokens

_TOPOLOGY_CACHE: dict[tuple, tuple] = {}


def topology_cached(x, grid):
    key = (tuple(int(v) for v in grid), str(x.device), int(x.shape[0]))
    hit = _TOPOLOGY_CACHE.get(key)
    if hit is None:
        hit = _topology(x, grid)
        _TOPOLOGY_CACHE[key] = hit
    return hit


def compress_tokens_cached(x, grid_thw, *args, **kwargs):
    import compressors
    original = compressors._topology
    compressors._topology = topology_cached
    try:
        return _compress_tokens(x, grid_thw, *args, **kwargs)
    finally:
        compressors._topology = original
