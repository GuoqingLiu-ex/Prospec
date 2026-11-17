"""Inference-only reassociation of ViSpec's fixed-query visual adaptor.

Each head keeps its own attention distribution.  For fixed queries, compute
``Q K(X).T`` using cached ``Q W_K`` directions, then evaluate ``(P X) W_V``.
This is equal in real arithmetic, not necessarily in floating-point arithmetic.
The original adaptor parameters are shared, never cloned or modified.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def adaptor_macs(
    *,
    num_tokens: int,
    in_features: int,
    num_heads: int,
    head_dim: int,
    num_queries: int,
    out_features: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Count multiply-accumulates; one MAC is two FLOPs in the usual convention.

    Includes K/V, attention, and output projections.  Excludes softmax,
    elementwise bias/scale/group-size operations, casts and data movement.
    Preparation is a one-time query/key contraction, independent of N and B.
    Counts do not imply a measured speedup.
    """
    dims = (num_tokens, in_features, num_heads, head_dim, num_queries, batch_size)
    if any(not isinstance(value, int) or value < 1 for value in dims):
        raise ValueError("All dimensions must be positive integers")
    output = in_features if out_features is None else out_features
    if not isinstance(output, int) or output < 1:
        raise ValueError("out_features must be a positive integer")
    n, d, h, k, q, b = dims
    width = h * k
    original = {
        "key_projection": b * n * d * width,
        "value_projection": b * n * d * width,
        "query_key": b * n * q * width,
        "attention_value": b * n * q * width,
        "output_projection": b * q * width * output,
    }
    factorized = {
        "score_projection": b * n * d * h * q,
        "feature_aggregation": b * n * d * h * q,
        "value_projection": b * q * d * width,
        "output_projection": b * q * width * output,
    }
    return {
        "original": original,
        "factorized": factorized,
        "original_total": sum(original.values()),
        "factorized_total": sum(factorized.values()),
        "prepare_query_key": q * d * width,
        "flops_per_mac": 2,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _tensor_signature(tensor: Tensor | None) -> tuple[Any, ...] | None:
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError:  # Tensors created under inference_mode have no counter.
        version = None
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.device, tensor.dtype, version)


class FactorizedAdaptor(nn.Module):
    """Wrap an already-loaded, frozen ViSpec ImgAdaptor.

    ``precision='model'`` uses the original parameter dtype. ``'float32'``
    performs arithmetic in FP32 and casts the result to the input dtype. FP32
    copies of V/O, if needed, are temporary *forward* costs, not retained caches.
    Caller autocast is disabled inside these explicit precision modes.

    Construction calls :meth:`prepare` once. ``prepare_time_ms`` includes device
    completion; ``cache_bytes`` counts only the cached directions and key bias.
    Re-run prepare after changing Q/K parameters or moving/casting the module.
    Do not use this inference-only wrapper to train the original adaptor.

    Optional ``group_sizes`` is a positive finite tensor [N] or [B,N] containing
    the number of original tokens represented by each input token. Values are
    checked on CPU; CUDA values are trusted to avoid a host synchronization.
    """

    def __init__(self, adaptor: nn.Module, precision: str = "model") -> None:
        super().__init__()
        if precision not in {"model", "float32"}:
            raise ValueError("precision must be 'model' or 'float32'")
        self.adaptor = adaptor  # Registers a reference; no parameter copies.
        self.precision = precision
        self._validate_adaptor()
        self.num_queries, self.num_heads, self.head_dim = adaptor.q.shape
        self.in_features = adaptor.k_proj.in_features
        self.out_features = adaptor.o_proj.out_features
        self.register_buffer("_score_directions", None, persistent=False)
        self.register_buffer("_score_bias", None, persistent=False)
        self._prepared_signature: tuple[Any, ...] | None = None
        self.prepare_time_ms = 0.0
        self.prepare_count = 0
        self.prepare()

    def _validate_adaptor(self) -> None:
        q = getattr(self.adaptor, "q", None)
        if not isinstance(q, Tensor) or q.ndim != 3 or min(q.shape) < 1:
            raise ValueError("adaptor.q must have shape [queries, heads, head_dim]")
        if not q.is_floating_point():
            raise ValueError("The adaptor must use floating-point parameters")
        for name in ("k_proj", "v_proj", "o_proj"):
            if not isinstance(getattr(self.adaptor, name, None), nn.Linear):
                raise TypeError(f"adaptor.{name} must be nn.Linear")
        key, value, output = self.adaptor.k_proj, self.adaptor.v_proj, self.adaptor.o_proj
        width = q.shape[1] * q.shape[2]
        if (key.out_features != width or value.out_features != width
                or key.in_features != value.in_features or output.in_features != width):
            raise ValueError("Q, K, V and O projection dimensions are incompatible")
        tensors = [q, key.weight, value.weight, output.weight]
        tensors += [layer.bias for layer in (key, value, output) if layer.bias is not None]
        if any(tensor.device != q.device or tensor.dtype != q.dtype for tensor in tensors):
            raise ValueError("All adaptor parameters must have the same dtype and device")

    def _key_signature(self) -> tuple[Any, ...]:
        return tuple(_tensor_signature(tensor) for tensor in (
            self.adaptor.q, self.adaptor.k_proj.weight, self.adaptor.k_proj.bias
        ))

    @property
    def compute_dtype(self) -> torch.dtype:
        return torch.float32 if self.precision == "float32" else self.adaptor.q.dtype

    @property
    def cache_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (
            self._score_directions, self._score_bias
        ) if tensor is not None)

    @torch.no_grad()
    def prepare(self) -> dict[str, Any]:
        """Refresh the small Q/K cache and return separately timed setup metadata."""
        self._validate_adaptor()
        device, dtype = self.adaptor.q.device, self.compute_dtype
        _synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=False):
            query = self.adaptor.q.detach().to(dtype=dtype).permute(1, 0, 2)
            key_weight = self.adaptor.k_proj.weight.detach().to(dtype=dtype).reshape(
                self.num_heads, self.head_dim, self.in_features
            )
            scale = 1.0 / math.sqrt(self.head_dim)
            self._score_directions = torch.bmm(query, key_weight).mul_(scale).contiguous()
            key_bias = self.adaptor.k_proj.bias
            self._score_bias = None if key_bias is None else torch.bmm(
                query, key_bias.detach().to(dtype=dtype).reshape(self.num_heads, self.head_dim, 1)
            ).squeeze(-1).mul_(scale).contiguous()
        self._prepared_signature = self._key_signature()
        _synchronize(device)
        self.prepare_time_ms = (time.perf_counter() - start) * 1000.0
        self.prepare_count += 1
        return {
            "prepare_time_ms": self.prepare_time_ms,
            "prepare_count": self.prepare_count,
            "cache_bytes": self.cache_bytes,
            "precision": self.precision,
            "compute_dtype": str(dtype),
            "device": str(device),
            "shares_original_parameters": True,
            "persistent_full_weight_copies": 0,
        }

    def _check_input(self, hidden_states: Tensor, *, allow_unbatched: bool = False) -> None:
        ranks = (2, 3) if allow_unbatched else (3,)
        if hidden_states.ndim not in ranks or hidden_states.shape[-1] != self.in_features:
            raise ValueError(f"Expected {'[N,D] or ' if allow_unbatched else ''}[B,N,D], D={self.in_features}")
        if min(hidden_states.shape) < 1:
            raise ValueError("Empty batches and empty visual sequences are unsupported")
        if hidden_states.device != self.adaptor.q.device or hidden_states.dtype != self.adaptor.q.dtype:
            raise ValueError("Input dtype and device must match the original adaptor")
        if self._prepared_signature != self._key_signature():
            raise RuntimeError("Q/K parameters changed or moved; call prepare() before forward")

    def _project_scores(self, work_input: Tensor) -> Tensor:
        bias = None if self._score_bias is None else self._score_bias.reshape(-1)
        return F.linear(work_input, self._score_directions.reshape(-1, self.in_features), bias)

    @torch.no_grad()
    def score_tokens(self, hidden_states: Tensor) -> Tensor:
        """Return scaled K/Q scores [N,H*Q] or [B,N,H*Q], head-major order.

        Includes K bias. Does not apply softmax or group-size correction. Useful
        for local compressors; calling it separately is extra runtime work.
        """
        self._check_input(hidden_states, allow_unbatched=True)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return self._project_scores(hidden_states.to(dtype=self.compute_dtype))

    @torch.no_grad()
    def attention_scores(self, hidden_states: Tensor) -> Tensor:
        """Return scaled K/Q scores [B,H,Q,N], before softmax/group-size bias."""
        self._check_input(hidden_states)
        scores = self.score_tokens(hidden_states)
        return scores.reshape(*hidden_states.shape[:2], self.num_heads, self.num_queries).permute(0, 2, 3, 1)

    def _group_bias(self, group_sizes: Tensor, batch: int, length: int, reference: Tensor) -> Tensor:
        sizes = torch.as_tensor(group_sizes, device=reference.device, dtype=reference.dtype)
        if sizes.shape == (length,):
            sizes = sizes.view(1, 1, 1, length)
        elif sizes.shape == (batch, length):
            sizes = sizes.view(batch, 1, 1, length)
        else:
            raise ValueError("group_sizes must have shape [N] or [B,N]")
        if sizes.device.type == "cpu" and not bool(torch.all(torch.isfinite(sizes) & (sizes > 0))):
            raise ValueError("group_sizes must be positive and finite")
        return sizes.log()

    @torch.no_grad()
    def forward(self, hidden_states: Tensor, group_sizes: Tensor | None = None) -> Tensor:
        # Bind this class's checker. A subclass override of _check_input must
        # not run just because the instance is passed to FactorizedAdaptor.forward.
        FactorizedAdaptor._check_input(self, hidden_states)
        batch, length, _ = hidden_states.shape
        h, q, d = self.num_heads, self.num_queries, self.head_dim
        dtype = self.compute_dtype
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            work_input = hidden_states.to(dtype=dtype)
            scores = self._project_scores(work_input).reshape(batch, length, h, q).permute(0, 2, 3, 1)
            if group_sizes is not None:
                scores = scores + self._group_bias(group_sizes, batch, length, scores)
            probabilities = scores.softmax(dim=-1)

            # Flatten only H/Q, so X is never broadcast/materialized H times.
            pooled = torch.bmm(probabilities.reshape(batch, h * q, length), work_input)
            pooled = pooled.reshape(batch, h, q, self.in_features).permute(1, 0, 2, 3)
            pooled = pooled.reshape(h, batch * q, self.in_features)
            value_weight = self.adaptor.v_proj.weight.to(dtype=dtype).reshape(h, d, self.in_features)
            values = torch.bmm(pooled, value_weight.transpose(1, 2))
            if self.adaptor.v_proj.bias is not None:
                values = values + self.adaptor.v_proj.bias.to(dtype=dtype).reshape(h, 1, d)
            values = values.reshape(h, batch, q, d).permute(1, 2, 0, 3).reshape(batch, q, h * d)
            output = F.linear(
                values,
                self.adaptor.o_proj.weight.to(dtype=dtype),
                None if self.adaptor.o_proj.bias is None else self.adaptor.o_proj.bias.to(dtype=dtype),
            )
        return output.to(dtype=hidden_states.dtype)

    def macs(self, num_tokens: int, batch_size: int = 1) -> dict[str, Any]:
        return adaptor_macs(
            num_tokens=num_tokens, in_features=self.in_features, num_heads=self.num_heads,
            head_dim=self.head_dim, num_queries=self.num_queries,
            out_features=self.out_features, batch_size=batch_size,
        )

    def extra_repr(self) -> str:
        return (f"precision={self.precision!r}, queries={self.num_queries}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}, cache_bytes={self.cache_bytes}")
