"""Inference-only implementations of ViSpec's frozen, fixed-query adaptor.

Construct after loading/casting the original adaptor. Weights are assumed frozen;
call ``prepare()`` after changing them or moving/casting this wrapper. The hot path
does not walk parameter signatures. ``assert_frozen()`` is an explicit diagnostic.

``backend='bmm'`` preserves the existing reassociation and its model-dtype softmax.
``backend='sdpa'`` uses the H*Q directions as independent query rows in ONE attention
head: SDPA(QWk/sqrt(d), X, X, scale=1). Thus X is not replicated H times. This is
equal in real arithmetic, but changes rounding and backend selection. D=3584 may
force a math SDPA backend: this is a candidate to benchmark, not a speedup claim.

Neither implementation removes visual evidence or changes Target execution.
Optional group_sizes only supports an upstream, separately timed compressor.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _signature(tensor: Tensor | None):
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError:
        version = None  # inference tensors do not expose a mutation counter
    return tensor.data_ptr(), tuple(tensor.shape), tensor.dtype, tensor.device, version


class FrozenQueryAdaptor(nn.Module):
    """Drop-in ``imadpt(x[B,N,D]) -> [B,Q,Dout]`` with explicit frozen setup.

    ``backend``: bmm or sdpa. ``value_projection``: bmm or einsum.
    ``precision``: model or float32. Model mode aliases original V/O weights;
    float32 may retain V/O copies and reports their full size. Preparation time
    includes all copies, projection composition and device completion.

    All runtime input casts, group-size correction, attention, V and O projection
    occur inside forward and must be timed. No calls to adaptor.forward are made,
    so the wrapper can replace that method without recursion.
    """

    def __init__(self, adaptor: nn.Module, *, backend: str = "bmm",
                 value_projection: str = "bmm", precision: str = "model"):
        super().__init__()
        if backend not in ("bmm", "sdpa"):
            raise ValueError("backend must be bmm or sdpa")
        if value_projection not in ("bmm", "einsum"):
            raise ValueError("value_projection must be bmm or einsum")
        if precision not in ("model", "float32"):
            raise ValueError("precision must be model or float32")
        self.adaptor = adaptor
        self.pooling = backend
        self.value_projection = value_projection
        self.precision = precision
        self.prepare_count = 0
        for name in ("directions", "score_bias", "value_heads", "value_bias",
                     "output_weight", "output_bias"):
            self.register_buffer(name, None, persistent=False)
        self.prepare()

    def _parameters_used(self):
        a = self.adaptor
        return (a.q, a.k_proj.weight, a.k_proj.bias, a.v_proj.weight,
                a.v_proj.bias, a.o_proj.weight, a.o_proj.bias)

    @torch.no_grad()
    def prepare(self) -> dict[str, Any]:
        a = self.adaptor
        if not isinstance(getattr(a, "q", None), Tensor) or a.q.ndim != 3:
            raise ValueError("adaptor.q must be [Q,H,d]")
        if any(not isinstance(getattr(a, name, None), nn.Linear)
               for name in ("k_proj", "v_proj", "o_proj")):
            raise TypeError("K/V/O must be nn.Linear modules")
        self.q_count, self.heads, self.head_dim = a.q.shape
        self.input_dim = a.k_proj.in_features
        self.output_dim = a.o_proj.out_features
        self.hq = self.heads * self.q_count
        width = self.heads * self.head_dim
        if min(self.q_count, self.heads, self.head_dim, self.input_dim) < 1:
            raise ValueError("adaptor dimensions must be positive")
        if (a.k_proj.out_features != width or a.v_proj.out_features != width
                or a.v_proj.in_features != self.input_dim or a.o_proj.in_features != width):
            raise ValueError("Q/K/V/O dimensions are incompatible")
        params = self._parameters_used()
        if any(p is not None and (p.dtype != a.q.dtype or p.device != a.q.device) for p in params):
            raise ValueError("all adaptor parameters must share dtype/device")
        self.input_dtype, self.input_device = a.q.dtype, a.q.device
        self.compute_dtype = torch.float32 if self.precision == "float32" else a.q.dtype
        _sync(self.input_device)
        started = time.perf_counter()
        with torch.autocast(device_type=self.input_device.type, enabled=False):
            q = a.q.detach().to(self.compute_dtype).permute(1, 0, 2)
            k = a.k_proj.weight.detach().to(self.compute_dtype).reshape(
                self.heads, self.head_dim, self.input_dim)
            scale = 1.0 / math.sqrt(self.head_dim)
            self.directions = torch.bmm(q, k).mul_(scale).reshape(self.hq, self.input_dim)
            self.score_bias = None if a.k_proj.bias is None else torch.bmm(
                q, a.k_proj.bias.detach().to(self.compute_dtype).reshape(self.heads, self.head_dim, 1)
            ).reshape(self.hq).mul_(scale)
            # Views in model precision. FP32 copies are explicit persistent costs.
            self.value_heads = a.v_proj.weight.detach().to(self.compute_dtype).reshape(
                self.heads, self.head_dim, self.input_dim)
            self.value_bias = None if a.v_proj.bias is None else a.v_proj.bias.detach().to(
                self.compute_dtype).reshape(self.heads, 1, self.head_dim)
            self.output_weight = a.o_proj.weight.detach().to(self.compute_dtype)
            self.output_bias = None if a.o_proj.bias is None else a.o_proj.bias.detach().to(self.compute_dtype)
        self._prepared_signature = tuple(_signature(p) for p in params)
        self.prepare_count += 1
        _sync(self.input_device)
        self.prepare_time_ms = (time.perf_counter() - started) * 1000.0
        self.projection_copy_bytes = sum(
            p.numel() * torch.empty((), dtype=self.compute_dtype).element_size()
            for p in (a.v_proj.weight, a.v_proj.bias, a.o_proj.weight, a.o_proj.bias)
            if p is not None and p.dtype != self.compute_dtype)
        self.cache_bytes = self.projection_copy_bytes + sum(
            p.numel() * p.element_size() for p in (self.directions, self.score_bias) if p is not None)
        return self.preparation_metadata()

    def preparation_metadata(self) -> dict[str, Any]:
        return dict(prepare_time_ms=self.prepare_time_ms, prepare_count=self.prepare_count,
                    cache_bytes=self.cache_bytes, projection_copy_bytes=self.projection_copy_bytes,
                    precision=self.precision, pooling=self.pooling,
                    value_projection=self.value_projection, hot_parameter_signature_checks=False,
                    setup_scope="Q/K composition, V/O aliases or copies, device completion; CUDA may already be initialized")

    def assert_frozen(self) -> None:
        """Explicit debug guard; inference tensors may lack mutation counters."""
        if tuple(_signature(p) for p in self._parameters_used()) != self._prepared_signature:
            raise RuntimeError("adaptor parameters changed; call prepare()")

    def _check_input(self, x: Tensor, *, unbatched: bool = False) -> None:
        if x.ndim not in ((2, 3) if unbatched else (3,)) or x.shape[-1] != self.input_dim:
            raise ValueError("expected [B,N,D] (or [N,D] for score_tokens)")
        if min(x.shape) < 1 or x.dtype != self.input_dtype or x.device != self.input_device:
            raise ValueError("input must be nonempty and match prepared adaptor dtype/device")

    def _group_log_bias(self, sizes: Tensor | None, batch: int, length: int):
        if sizes is None:
            return None
        sizes = torch.as_tensor(sizes, device=self.input_device, dtype=self.compute_dtype)
        if sizes.shape == (length,):
            sizes = sizes.reshape(1, 1, length)
        elif sizes.shape == (batch, length):
            sizes = sizes.reshape(batch, 1, length)
        else:
            raise ValueError("group_sizes must be [N] or [B,N]")
        if sizes.device.type == "cpu" and not bool(torch.all(torch.isfinite(sizes) & (sizes > 0))):
            raise ValueError("group sizes must be positive and finite")
        return sizes.log()

    @torch.no_grad()
    def score_tokens(self, x: Tensor) -> Tensor:
        """[N,D]/[B,N,D] -> [N,HQ]/[B,N,HQ], including key bias.

        Scoring precision is exactly this instance's precision. Use this same
        method in microbenchmarks and end-to-end compression integration.
        """
        self._check_input(x, unbatched=True)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.to(self.compute_dtype), self.directions, self.score_bias)

    @torch.no_grad()
    def forward(self, x: Tensor, group_sizes: Tensor | None = None) -> Tensor:
        self._check_input(x)
        batch, length, _ = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.to(self.compute_dtype)
            log_sizes = self._group_log_bias(group_sizes, batch, length)
            if self.pooling == "bmm":
                scores = F.linear(work, self.directions, self.score_bias).transpose(1, 2)
                if log_sizes is not None:
                    scores = scores + log_sizes
                pooled = torch.bmm(scores.softmax(-1), work)  # [B,HQ,D]
            else:
                # Key-projection bias is constant across tokens for each query,
                # and therefore cancels in softmax. Directions already include
                # 1/sqrt(original head_dim); SDPA MUST use scale=1, not 1/sqrt(D).
                query = self.directions.reshape(1, 1, self.hq, self.input_dim).expand(batch, -1, -1, -1)
                memory = work.unsqueeze(1)
                mask = None if log_sizes is None else log_sizes.unsqueeze(1)
                pooled = F.scaled_dot_product_attention(
                    query, memory, memory, attn_mask=mask, dropout_p=0.0,
                    is_causal=False, scale=1.0).squeeze(1)
            if self.value_projection == "bmm":
                per_head = (pooled.reshape(self.heads, self.q_count, self.input_dim)
                            if batch == 1 else pooled.reshape(batch, self.heads, self.q_count, self.input_dim)
                            .permute(1, 0, 2, 3).reshape(self.heads, batch * self.q_count, self.input_dim))
                values = torch.bmm(per_head, self.value_heads.transpose(1, 2))
                if self.value_bias is not None:
                    values = values + self.value_bias
                values = values.reshape(self.heads, batch, self.q_count, self.head_dim).permute(1, 2, 0, 3)
            else:
                per_head = pooled.reshape(batch, self.heads, self.q_count, self.input_dim)
                values = torch.einsum("bhqi,hdi->bqhd", per_head, self.value_heads)
                if self.value_bias is not None:
                    values = values + self.value_bias.reshape(1, 1, self.heads, self.head_dim)
            values = values.reshape(batch, self.q_count, self.heads * self.head_dim)
            result = F.linear(values, self.output_weight, self.output_bias)
        return result.to(x.dtype)


def make_adaptor(adaptor: nn.Module, variant: str = "lean_bmm", *, precision: str = "model"):
    """Factory names are suitable for benchmark result rows; no auto selection."""
    variants = {"lean_bmm": ("bmm", "bmm"), "lean_einsum": ("bmm", "einsum"),
                "sdpa_bmm": ("sdpa", "bmm"), "sdpa_einsum": ("sdpa", "einsum")}
    if variant not in variants:
        raise ValueError(f"unknown variant {variant!r}; choose {tuple(variants)}")
    pooling, value_projection = variants[variant]
    return FrozenQueryAdaptor(adaptor, backend=pooling, value_projection=value_projection, precision=precision)
