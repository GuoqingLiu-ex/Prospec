"""Standalone deployment of an already prepared, frozen BMM visual adaptor.

This performs no new algebra or weight composition. It retains only the prepared
directions and V/O tensors, with no reference to the source module or Q/K. The
caller must replace draft.imadpt and release all other source/runtime references.
This conversion alone establishes no startup-peak saving: Q/K existed at setup.

Buffers share storage with the prepared tensors. Treat both sides as immutable;
this is a frozen deployment module, not an independent mutable weight snapshot.
Convert after the final model dtype is selected. Moving devices is permitted,
but casting an already composed module is rejected at forward time.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compression import FrozenQueryAdaptor


BUFFER_NAMES = ("directions", "score_bias", "value_heads", "value_bias",
                "output_weight", "output_bias")


def _storage_key(tensor: Tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr()


def _storage_metadata(named_tensors):
    """Return only Python data, never Tensor or storage objects that retain memory."""
    unique = {}
    logical = 0
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        logical += tensor.numel() * tensor.element_size()
        key = _storage_key(tensor)
        if key not in unique:
            unique[key] = {"device": key[0], "data_ptr": key[1],
                           "storage_bytes": tensor.untyped_storage().nbytes(), "buffers": []}
        unique[key]["buffers"].append({"name": name, "shape": list(tensor.shape),
                                       "stride": list(tensor.stride()), "dtype": str(tensor.dtype)})
    return {"logical_tensor_bytes": logical,
            "unique_storage_bytes": sum(row["storage_bytes"] for row in unique.values()),
            "storages": list(unique.values())}


class CompiledQueryAdaptor(nn.Module):
    """Use from_frozen(prepared); forward keeps its exact BMM operation order."""

    def __init__(self):
        super().__init__()
        self._ready = False

    @classmethod
    @torch.no_grad()
    def from_frozen(cls, prepared: FrozenQueryAdaptor) -> "CompiledQueryAdaptor":
        if not isinstance(prepared, FrozenQueryAdaptor):
            raise TypeError("expected a prepared FrozenQueryAdaptor")
        if (prepared.pooling, prepared.value_projection, prepared.precision) != ("bmm", "bmm", "model"):
            raise ValueError("standalone deployment currently requires bmm pooling, bmm V projection, model precision")
        prepared.assert_frozen()
        original = prepared.adaptor
        removed = [("q", original.q), ("k_proj.weight", original.k_proj.weight),
                   ("k_proj.bias", original.k_proj.bias)]
        removed_keys = {_storage_key(tensor) for _, tensor in removed if tensor is not None}
        for name in BUFFER_NAMES:
            tensor = getattr(prepared, name)
            if tensor is not None and _storage_key(tensor) in removed_keys:
                raise ValueError(f"{name} shares Q/K storage; standalone release would not be valid")
        result = cls()
        for name in BUFFER_NAMES:
            tensor = getattr(prepared, name)
            result.register_buffer(name, None if tensor is None else tensor.detach(), persistent=True)
        for name in ("q_count", "heads", "head_dim", "input_dim", "output_dim", "hq"):
            setattr(result, name, int(getattr(prepared, name)))
        result.hidden_size = result.input_dim
        result.num_q, result.num_heads = result.q_count, result.heads
        result.pooling, result.value_projection, result.precision = "bmm", "bmm", "model"
        result._compiled_dtype = prepared.compute_dtype
        result.prepare_time_ms = float(prepared.prepare_time_ms)
        result.cache_bytes = int(prepared.cache_bytes)
        result._removed_metadata = _storage_metadata(removed)
        result._native_metadata = _storage_metadata([
            ("q", original.q), ("k_proj.weight", original.k_proj.weight),
            ("k_proj.bias", original.k_proj.bias), ("v_proj.weight", original.v_proj.weight),
            ("v_proj.bias", original.v_proj.bias), ("o_proj.weight", original.o_proj.weight),
            ("o_proj.bias", original.o_proj.bias)])
        result._ready = True
        return result.eval()

    @property
    def input_dtype(self):
        return self.directions.dtype

    @property
    def compute_dtype(self):
        return self.directions.dtype

    @property
    def input_device(self):
        return self.directions.device

    def metadata(self) -> dict[str, Any]:
        if not self._ready:
            raise RuntimeError("construct via CompiledQueryAdaptor.from_frozen")
        current = _storage_metadata([(name, getattr(self, name)) for name in BUFFER_NAMES])
        return {
            "implementation": "standalone_frozen_bmm", "source_module_retained": False,
            "composition_reused": True, "precision": self.precision,
            "compiled_dtype": str(self._compiled_dtype),
            "prepare_time_ms": self.prepare_time_ms, "cache_bytes": self.cache_bytes,
            "buffers": current,
            "source_qk": self._removed_metadata,
            "source_native_adaptor": self._native_metadata,
            "theoretical_release_vs_retained_bmm_bytes": self._removed_metadata["unique_storage_bytes"],
            "theoretical_net_reduction_vs_native_bytes": self._native_metadata["unique_storage_bytes"] - current["unique_storage_bytes"],
            "release_requires": "caller releases original adaptor, prepared wrapper, bound methods, state dictionaries and runtime/graph references",
            "mutation_contract": "all buffers alias frozen source storage; V/O or cached-direction mutation is shared; later Q/K mutation cannot recompose this module",
            "scope": "steady-state resident tensors; actual allocator/request peaks require independent full-model measurement",
            "startup_peak_saving_proven": False,
        }

    def preparation_metadata(self):
        return self.metadata()

    def _check_input(self, x: Tensor, *, unbatched=False):
        if not self._ready:
            raise RuntimeError("construct via CompiledQueryAdaptor.from_frozen")
        if self.compute_dtype != self._compiled_dtype:
            raise RuntimeError("do not cast a composed module; prepare from the final model dtype")
        if x.ndim not in ((2, 3) if unbatched else (3,)) or x.shape[-1] != self.input_dim:
            raise ValueError("expected [B,N,D] (or [N,D] for score_tokens)")
        if min(x.shape) < 1 or x.dtype != self.input_dtype or x.device != self.input_device:
            raise ValueError("input must be nonempty and match prepared adaptor dtype/device")

    def _group_log_bias(self, sizes, batch, length):
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
            scores = F.linear(work, self.directions, self.score_bias).transpose(1, 2)
            if log_sizes is not None:
                scores = scores + log_sizes
            pooled = torch.bmm(scores.softmax(-1), work)
            per_head = (pooled.reshape(self.heads, self.q_count, self.input_dim)
                        if batch == 1 else pooled.reshape(batch, self.heads, self.q_count, self.input_dim)
                        .permute(1, 0, 2, 3).reshape(self.heads, batch * self.q_count, self.input_dim))
            values = torch.bmm(per_head, self.value_heads.transpose(1, 2))
            if self.value_bias is not None:
                values = values + self.value_bias
            values = values.reshape(self.heads, batch, self.q_count, self.head_dim).permute(1, 2, 0, 3)
            values = values.reshape(batch, self.q_count, self.heads * self.head_dim)
            result = F.linear(values, self.output_weight, self.output_bias)
        return result.to(x.dtype)
