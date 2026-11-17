"""Convert Qwen video_grid_thw into the post-merger grid used by imadpt."""
from __future__ import annotations

import math

import torch


def single_video_grid(grid_thw) -> tuple[int, int, int]:
    """Return one (T, H, W). A batch of clips must be split by the caller."""
    if torch.is_tensor(grid_thw):
        grid_thw = grid_thw.detach().cpu().tolist()
    if not isinstance(grid_thw, (list, tuple)) or not grid_thw:
        raise ValueError("grid_thw must be (T, H, W) or a length-1 list of that triple")
    if isinstance(grid_thw[0], (list, tuple)):
        if len(grid_thw) != 1:
            raise ValueError("split multi-clip grids per sequence before conversion")
        grid_thw = grid_thw[0]
    if len(grid_thw) != 3:
        raise ValueError("grid_thw must contain T, H, W")
    t, h, w = (int(v) for v in grid_thw)
    if min(t, h, w) <= 0:
        raise ValueError("grid_thw entries must be positive")
    return t, h, w


def merged_visual_grid(grid_thw, spatial_merge_size: int) -> tuple[int, int, int]:
    """Map pre-merger (T, H, W) to (T, H//s, W//s).

    T counts temporal patches and is not divided. spatial_merge_size comes from
    the model config; H and W must be divisible by it.
    """
    if isinstance(spatial_merge_size, bool) or not isinstance(spatial_merge_size, int) or spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be a positive integer")
    t, h, w = single_video_grid(grid_thw)
    if h % spatial_merge_size or w % spatial_merge_size:
        raise ValueError(f"H,W {(h, w)} are not divisible by spatial_merge_size {spatial_merge_size}")
    return t, h // spatial_merge_size, w // spatial_merge_size


def assert_adaptor_tokens(num_tokens: int, grid_thw, spatial_merge_size: int) -> tuple[int, int, int]:
    """Return the merged grid after checking it produces exactly num_tokens."""
    grid = merged_visual_grid(grid_thw, spatial_merge_size)
    product = math.prod(grid)
    if int(num_tokens) != product:
        raise ValueError(f"adaptor tokens {num_tokens} != merged grid {grid} product {product}")
    return grid
