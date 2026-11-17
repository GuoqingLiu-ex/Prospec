"""Explicit CUDA graph cache for a frozen visual adaptor.

This caches an execution graph and its storage, NEVER the result for an input.
Every forward copies current inputs, replays all adaptor operations, and by
default clones the result. Benchmark that entire forward, including both copies.

Use after constructing the frozen adaptor on CUDA:

    graphed = CudaGraphAdaptor(adaptor, clone_output=True, max_entries=4)
    setup = graphed.prepare_for(x)  # warmup/capture/first replay, measured once
    y = graphed(x)                 # copy_(x), replay, output.clone()

Preparation and replay must be called from the same CUDA stream for a given
shape. Warmup/capture use a separate, synchronized stream. No hidden capture,
cache eviction, compilation, or synchronization is performed in forward.
The module's parameters and buffers must remain unchanged after capture. Call
clear(), refresh the underlying module, then prepare_for() after any change.

clone_output=False returns graph-owned storage, overwritten by the next replay
of that entry. Consumers must finish using it on the same stream before then.
clone_output=True returns independent storage safe to retain across replays.
Neither option makes the wrapper safe for arbitrary cross-stream consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class _Entry:
    graph: Any
    static_input: Tensor
    static_sizes: Tensor | None
    static_output: Tensor
    replay_stream_id: int
    metadata: dict[str, Any]
    lock: Any = field(default_factory=threading.Lock)


class CudaGraphAdaptor(nn.Module):
    """Shape-keyed graph wrapper for module(x, group_sizes=None) -> Tensor.

    max_entries bounds retained graph pools. Exceeding it raises rather than
    silently evicting/recompiling in a measured request. Input tensors may be
    noncontiguous: copying into the contiguous static buffer is still timed.
    Group sizes may be CPU or CUDA tensors; the actual transfer is timed too.
    """

    def __init__(self, module: nn.Module, *, clone_output: bool = True,
                 max_entries: int = 4, warmup: int = 3):
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("module must be an nn.Module")
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("max_entries must be positive")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 1:
            raise ValueError("warmup must be positive")
        self.module = module
        self.clone_output = bool(clone_output)
        self.max_entries = max_entries
        self.warmup = warmup
        self._entries: dict[tuple, _Entry] = {}
        self._prepare_lock = threading.Lock()
        self.prepare_count = 0
        self.total_prepare_ms = 0.0

    @staticmethod
    def _key(x: Tensor, group_sizes: Tensor | None) -> tuple:
        if not isinstance(x, Tensor) or x.ndim != 3 or min(x.shape) < 1:
            raise ValueError("input must be a nonempty [B,N,D] Tensor")
        if x.device.type != "cuda":
            raise ValueError("CudaGraphAdaptor requires CUDA input; use the eager adaptor on CPU")
        if not x.is_floating_point():
            raise ValueError("input must have floating point dtype")
        if group_sizes is not None:
            if not isinstance(group_sizes, Tensor):
                raise TypeError("group_sizes must be a Tensor or None")
            if tuple(group_sizes.shape) not in ((x.shape[1],), tuple(x.shape[:2])):
                raise ValueError("group_sizes must be [N] or [B,N]")
            if group_sizes.device.type not in ("cpu", "cuda"):
                raise ValueError("group_sizes must be on CPU or CUDA")
            if group_sizes.device.type == "cuda" and group_sizes.device != x.device:
                raise ValueError("CUDA group_sizes must be on the input device")
            size_key = (tuple(group_sizes.shape), group_sizes.dtype)
        else:
            size_key = None
        return (x.device.index, tuple(x.shape), x.dtype, size_key)

    @staticmethod
    def _copy_inputs(entry: _Entry, x: Tensor, group_sizes: Tensor | None) -> None:
        entry.static_input.copy_(x)
        if entry.static_sizes is not None:
            entry.static_sizes.copy_(group_sizes)

    @torch.inference_mode()
    def prepare_for(self, x: Tensor, group_sizes: Tensor | None = None) -> dict[str, Any]:
        """Capture one shape explicitly and return its complete setup metadata.

        Includes allocating/copying static I/O, warmup, capture, one initial
        replay, and device completion. CUDA/module setup may already be warm.
        Calling again for a prepared shape returns its metadata without work.
        """
        key = self._key(x, group_sizes)
        with self._prepare_lock, torch.cuda.device(x.device):
            caller = torch.cuda.current_stream(x.device)
            if key in self._entries:
                entry = self._entries[key]
                if entry.replay_stream_id != caller.cuda_stream:
                    raise RuntimeError("entry was prepared for another replay stream; clear and prepare on this stream")
                return {**entry.metadata, "cache_hit": True}
            if len(self._entries) >= self.max_entries:
                raise RuntimeError("graph cache is full; explicitly clear it or increase max_entries before measurement")
            torch.cuda.synchronize(x.device)
            allocated_before = torch.cuda.memory_allocated(x.device)
            reserved_before = torch.cuda.memory_reserved(x.device)
            started = time.perf_counter()
            static_input = torch.empty_like(x, memory_format=torch.contiguous_format)
            static_input.copy_(x)
            static_sizes = None
            if group_sizes is not None:
                static_sizes = torch.empty(group_sizes.shape, dtype=group_sizes.dtype, device=x.device)
                static_sizes.copy_(group_sizes)
            capture_stream = torch.cuda.Stream(device=x.device)
            capture_stream.wait_stream(caller)
            with torch.cuda.stream(capture_stream):
                for _ in range(self.warmup):
                    warm_output = self.module(static_input, group_sizes=static_sizes)
                if not isinstance(warm_output, Tensor):
                    raise TypeError("captured adaptor must return one Tensor")
                del warm_output
            caller.wait_stream(capture_stream)
            torch.cuda.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                static_output = self.module(static_input, group_sizes=static_sizes)
            if not isinstance(static_output, Tensor):
                raise TypeError("captured adaptor must return one Tensor")
            caller.wait_stream(capture_stream)
            graph.replay()
            torch.cuda.synchronize(x.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            allocated_after = torch.cuda.memory_allocated(x.device)
            reserved_after = torch.cuda.memory_reserved(x.device)
            self.prepare_count += 1
            self.total_prepare_ms += elapsed_ms
            meta = {
                "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device),
                "group_shape": None if group_sizes is None else list(group_sizes.shape),
                "group_dtype": None if group_sizes is None else str(group_sizes.dtype),
                "prepare_time_ms": elapsed_ms, "warmup_calls": self.warmup,
                "prepare_index": self.prepare_count, "first_replay_included": True,
                "clone_output": self.clone_output,
                "static_input_bytes": static_input.numel() * static_input.element_size(),
                "static_group_bytes": 0 if static_sizes is None else static_sizes.numel() * static_sizes.element_size(),
                "static_output_bytes": static_output.numel() * static_output.element_size(),
                "allocated_before_bytes": allocated_before,
                "allocated_after_bytes": allocated_after,
                "allocated_delta_bytes": allocated_after - allocated_before,
                "reserved_before_bytes": reserved_before,
                "reserved_after_bytes": reserved_after,
                "reserved_delta_bytes": reserved_after - reserved_before,
                "memory_note": "allocator deltas include retained graph allocations and warmup caching; not an isolated peak-memory reduction",
                "setup_scope": "static allocation/copy, warmup, capture, initial replay and device sync; excludes underlying adaptor construction",
                "forward_scope": "current input/group copy, replay, and output clone if enabled; no output result cache",
                "replay_stream_id": int(caller.cuda_stream),
            }
            self._entries[key] = _Entry(graph, static_input, static_sizes, static_output,
                                       caller.cuda_stream, meta)
            return {**meta, "cache_hit": False}

    @torch.inference_mode()
    def forward(self, x: Tensor, group_sizes: Tensor | None = None) -> Tensor:
        key = self._key(x, group_sizes)
        entry = self._entries.get(key)
        if entry is None:
            raise RuntimeError("shape has no prepared graph; call prepare_for outside request timing")
        if torch.cuda.current_stream(x.device).cuda_stream != entry.replay_stream_id:
            raise RuntimeError("forward must use the same CUDA stream as prepare_for")
        # Serialize host enqueueing so two threads cannot interleave input-copy
        # and replay on a shared static buffer. Device execution stays asynchronous.
        with entry.lock:
            self._copy_inputs(entry, x, group_sizes)
            entry.graph.replay()
            return entry.static_output.clone() if self.clone_output else entry.static_output

    def preparation_metadata(self) -> dict[str, Any]:
        return {"entries": len(self._entries), "max_entries": self.max_entries,
                "prepare_count": self.prepare_count, "total_prepare_ms": self.total_prepare_ms,
                "clone_output": self.clone_output,
                "cases": [dict(entry.metadata) for entry in self._entries.values()]}

    def clear(self) -> None:
        """Explicitly release graphs after queued work finishes; outside timing."""
        with self._prepare_lock:
            for device in {entry.static_input.device for entry in self._entries.values()}:
                torch.cuda.synchronize(device)
            self._entries.clear()

    def _apply(self, fn, recurse=True):
        if self._entries:
            raise RuntimeError("clear CUDA graphs before moving/casting the wrapper")
        return super()._apply(fn, recurse=recurse)


# Alternative spelling for callers that use CUDA as an initialism.
CUDAGraphAdaptor = CudaGraphAdaptor
