"""Full-token adaptor with cached V/O views and size-based native fallback.

Hypothesis H1: the previous FactorizedAdaptor recasts V/O weights every forward.
Hypothesis H2: N<512 falls back to the original adaptor on this GPU family.
Dispatch uses only (N, device type, dtype), never a video id or label.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from factorized_adaptor import FactorizedAdaptor


# Three GPU processes rejected the old N<512 native fallback: reorder was faster at 468.
# N below the smallest measured winning size stays on native. Not an acceptance cutoff.
# Dispatch never uses a video id.
SMALL_TOKEN_FALLBACK = 468


def dispatch_full_token(num_tokens: int, device_type: str = "cuda",
                        small_limit: int = SMALL_TOKEN_FALLBACK) -> str:
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if device_type == "cuda" and num_tokens < small_limit:
        return "native"
    return "reorder"


class CachedFactorizedAdaptor(FactorizedAdaptor):
    """Same real-arithmetic contract as FactorizedAdaptor, cached V/O/bias."""

    def __init__(self, adaptor: nn.Module, precision: str = "model",
                 check_projections: bool = True) -> None:
        super().__init__(adaptor, precision=precision)
        # Default True: CachedFactorizedAdaptor.forward walks V/O identities on
        # every call. That walk made round 1 slower than FactorizedAdaptor.
        # check_projections=False is only an ablation of this class's own forward.
        # It is not how run_full_token avoids the walk.
        self.check_projections = bool(check_projections)

    def _store_cache(self, name: str, tensor: Tensor | None) -> None:
        if name in self._buffers:
            self._buffers[name] = tensor
        else:
            self.register_buffer(name, tensor, persistent=False)

    def _projection_signature(self) -> tuple[Any, ...]:
        value, output = self.adaptor.v_proj, self.adaptor.o_proj
        return (
            *self._key_signature(),
            id(value.weight), tuple(value.weight.shape), value.weight.dtype, value.weight.device,
            None if value.bias is None else (id(value.bias), tuple(value.bias.shape)),
            id(output.weight), tuple(output.weight.shape), output.weight.dtype,
            None if output.bias is None else (id(output.bias), tuple(output.bias.shape)),
        )

    @property
    def cache_bytes(self) -> int:
        tensors = [
            self._score_directions, self._score_bias,
            getattr(self, "_value_weight", None), getattr(self, "_value_bias", None),
            getattr(self, "_output_weight", None), getattr(self, "_output_bias", None),
        ]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    @torch.no_grad()
    def prepare(self) -> dict[str, Any]:
        meta = super().prepare()
        dtype = self.compute_dtype
        value, output = self.adaptor.v_proj, self.adaptor.o_proj
        needs_cast = value.weight.dtype != dtype or output.weight.dtype != dtype
        if needs_cast:
            self._store_cache(
                "_value_weight",
                value.weight.detach().to(dtype=dtype).reshape(
                    self.num_heads, self.head_dim, self.in_features
                ).contiguous(),
            )
            self._store_cache(
                "_value_bias",
                None if value.bias is None else value.bias.detach().to(dtype=dtype).reshape(
                    self.num_heads, 1, self.head_dim
                ).contiguous(),
            )
            self._store_cache("_output_weight", output.weight.detach().to(dtype=dtype).contiguous())
            self._store_cache(
                "_output_bias",
                None if output.bias is None else output.bias.detach().to(dtype=dtype).contiguous(),
            )
        else:
            self._store_cache("_value_weight", None)
            self._store_cache("_value_bias", None)
            self._store_cache("_output_weight", None)
            self._store_cache("_output_bias", None)
        self._prepared_projection_signature = self._projection_signature()
        meta["cache_bytes"] = self.cache_bytes
        meta["cached_value_output"] = needs_cast
        meta["persistent_full_weight_copies"] = int(needs_cast)
        meta["prepare_includes_value_output_cast"] = False
        meta["note"] = "prepare_time_ms is the parent Q/K cache only; V/O cast is separate and zero on the model-precision path"
        return meta

    def _check_input(self, hidden_states: Tensor, *, allow_unbatched: bool = False) -> None:
        super()._check_input(hidden_states, allow_unbatched=allow_unbatched)
        if self.check_projections and getattr(self, "_prepared_projection_signature", None) != self._projection_signature():
            raise RuntimeError("V/O or Q/K parameters changed; call prepare() before forward")

    @torch.no_grad()
    def forward(self, hidden_states: Tensor, group_sizes: Tensor | None = None) -> Tensor:
        self._check_input(hidden_states)
        batch, length, _ = hidden_states.shape
        h, q, d = self.num_heads, self.num_queries, self.head_dim
        dtype = self.compute_dtype
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            work_input = hidden_states if hidden_states.dtype == dtype else hidden_states.to(dtype=dtype)
            scores = self._project_scores(work_input).reshape(batch, length, h, q).permute(0, 2, 3, 1)
            if group_sizes is not None:
                scores = scores + self._group_bias(group_sizes, batch, length, scores)
            probabilities = scores.softmax(dim=-1)
            pooled = torch.bmm(probabilities.reshape(batch, h * q, length), work_input)
            pooled = pooled.reshape(batch, h, q, self.in_features).permute(1, 0, 2, 3)
            pooled = pooled.reshape(h, batch * q, self.in_features)
            if self._value_weight is None:
                value_weight = self.adaptor.v_proj.weight.reshape(h, d, self.in_features)
                value_bias = None if self.adaptor.v_proj.bias is None else self.adaptor.v_proj.bias.reshape(h, 1, d)
                output_weight = self.adaptor.o_proj.weight
                output_bias = self.adaptor.o_proj.bias
            else:
                value_weight = self._value_weight
                value_bias = self._value_bias
                output_weight = self._output_weight
                output_bias = self._output_bias
            values = torch.bmm(pooled, value_weight.transpose(1, 2))
            if value_bias is not None:
                values = values + value_bias
            values = values.reshape(h, batch, q, d).permute(1, 2, 0, 3).reshape(batch, q, h * d)
            output = F.linear(values, output_weight, output_bias)
        return output.to(dtype=hidden_states.dtype)


def run_full_token(native: Callable, cached: CachedFactorizedAdaptor, hidden_states: Tensor,
                   group_sizes: Tensor | None = None, small_limit: int = SMALL_TOKEN_FALLBACK):
    """Selected full-token path. Cached.forward is not selected.

    The reorder branch calls FactorizedAdaptor.forward. That method checks Q/K
    through FactorizedAdaptor._check_input, including when `cached` is a
    CachedFactorizedAdaptor whose own forward would also walk V/O. Constructing
    a quiet object is not required and does not define this default.
    """
    choice = dispatch_full_token(hidden_states.shape[1], hidden_states.device.type, small_limit)
    if choice == "native":
        if group_sizes is None:
            return native(hidden_states), choice
        from scheme_context import native_with_sizes
        return native_with_sizes(cached.adaptor, hidden_states, group_sizes), choice
    return FactorizedAdaptor.forward(cached, hidden_states, group_sizes), "reorder"
