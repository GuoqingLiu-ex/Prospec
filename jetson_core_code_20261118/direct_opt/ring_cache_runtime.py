"""Scoped Target KV layout for the existing HF 5.17 ViSpec joint executor.

The native prefill and Draft cache are unchanged.  Logical token offsets and
the caller's explicit mRoPE positions remain separate from physical KV slots.
Evicting modes change Target context and do not promise lossless generation.
This module does not launch a model or modify any HF class globally.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from pathlib import Path
import sys
from types import MethodType

import torch

_ACTIVE = None
_MISSING = object()
MODES = ("full", "protected_prompt", "pure_recent")


def _restore_attribute(obj, name, previous):
    if previous is _MISSING:
        obj.__dict__.pop(name, None)
    else:
        setattr(obj, name, previous)


def _bridge():
    vendor = str(Path(__file__).resolve().parents[1] / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    import hf517_vispec_adapter
    return hf517_vispec_adapter


class RingCacheState:
    """One request's coordinator; finalized summaries contain no tensors."""

    def __init__(self, target, bridge, mode, recent_capacity, tree_capacity):
        self.target = target
        self.bridge = bridge
        self.mode = mode
        self.recent_capacity = recent_capacity
        self.tree_capacity = tree_capacity
        self.cache = None
        self.stores = []
        self.layer_restore = []
        self.plan = None
        self.pending = None
        self.in_forward = False
        self.update_counts = []
        self.prefill_calls = 0
        self.prompt = None
        self.delta = None
        self.bytes_per_token = None
        self.device = None
        self.current_phase = "prefill"
        self.finalized = False
        self.failed = False
        self.final_layer_metadata = []
        self.rounds = []
        self.memory = {"prefill": None, "conversion": None, "decode": None,
                       "full_request_peak_allocated_bytes": None,
                       "full_request_peak_reserved_bytes": None}
        self.prefill_metadata = None
        self.original_prefill = bridge._prefill
        self.original_length = bridge._length
        self.original_compact = bridge._compact
        self.original_forward = target.forward

    def _peak(self):
        if self.device is None or self.device.type != "cuda":
            return {"peak_allocated_bytes": None, "peak_reserved_bytes": None,
                    "allocated_at_boundary_bytes": None, "reserved_at_boundary_bytes": None,
                    "note": "CPU: CUDA allocator measurements unavailable"}
        torch.cuda.synchronize(self.device)
        return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                "allocated_at_boundary_bytes": int(torch.cuda.memory_allocated(self.device)),
                "reserved_at_boundary_bytes": int(torch.cuda.memory_reserved(self.device))}

    def _close_phase(self, name):
        self.memory[name] = self._peak()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def prefill(self, target, values, counts, **kwargs):
        if target is not self.target:
            return self.original_prefill(target, values, counts, **kwargs)
        if self.prefill_calls:
            raise RuntimeError("ring policy requires one prefill per context")
        self.prefill_calls += 1
        self.device = values["input_ids"].device
        out, delta, features = self.original_prefill(target, values, counts, **kwargs)
        self._close_phase("prefill")
        self.current_phase = "conversion"
        self.cache = out.past_key_values
        self.prompt = int(values["input_ids"].shape[1])
        self.delta = int(delta)
        if self.original_length(self.cache) != self.prompt:
            raise RuntimeError("prefill cache must contain the entire prompt")
        self.bytes_per_token = 0
        for layer in self.cache.layers:
            k, v = layer.keys, layer.values
            if k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1 or k.shape[-2] != self.prompt:
                raise RuntimeError("ring runtime requires batch-one uniform full-prefill KV")
            if k.dtype != v.dtype or k.device != v.device or k.device != self.device:
                raise RuntimeError("prefill KV dtype/device mismatch")
            self.bytes_per_token += 2 * k.shape[1] * k.shape[-1] * k.element_size()
        del k, v
        if self.mode != "full":
            from ring_kv_cache import RingKVLayer
            for index, layer in enumerate(self.cache.layers):
                store = RingKVLayer.from_prefill(
                    layer.keys, layer.values, mode=self.mode,
                    recent_capacity=self.recent_capacity, tree_capacity=self.tree_capacity)
                self.stores.append(store)
                saved = {name: layer.__dict__.get(name, _MISSING)
                         for name in ("update", "get_seq_length", "get_mask_sizes")}
                self.layer_restore.append((layer, saved))
                # Replace owning references immediately: no old-prefix copy is held.
                layer.keys, layer.values = store.keys, store.values

                def update(instance, keys, values, *args, _index=index, **kwargs):
                    return self._update(_index, keys, values, args, kwargs)

                def length(instance, _index=index):
                    item = self.stores[_index]
                    return item.next_position + item.active_tree_nodes

                def mask_sizes(instance, query_length, _index=index):
                    return int(self.stores[_index].keys.shape[-2]), 0

                layer.update = MethodType(update, layer)
                layer.get_seq_length = MethodType(length, layer)
                layer.get_mask_sizes = MethodType(mask_sizes, layer)
        self.prefill_metadata = self._cache_metadata()
        self._close_phase("conversion")
        self.current_phase = "decode"
        return out, delta, features

    def length(self, cache):
        if cache is not self.cache or self.mode == "full":
            return self.original_length(cache)
        lengths = [x.next_position + x.active_tree_nodes for x in self.stores]
        if not lengths or len(set(lengths)) != 1:
            raise RuntimeError("ring layers disagree on semantic cache length")
        return lengths[0]

    def _update(self, index, keys, values, args, kwargs):
        if not self.in_forward or self.plan is None:
            raise RuntimeError("ring update outside the planned verification forward")
        if args or kwargs:
            raise RuntimeError("unexpected HF cache update arguments")
        if self.update_counts[index]:
            raise RuntimeError("a Target layer updated its ring twice in one forward")
        result = self.stores[index].append_tree(keys, values, self.plan)
        self.update_counts[index] += 1
        return result

    def forward(self, *args, **kwargs):
        cache = kwargs.get("past_key_values")
        if cache is not self.cache or cache is None:
            return self.original_forward(*args, **kwargs)
        if self.failed:
            raise RuntimeError("ring request previously failed")
        if self.pending is not None or self.in_forward:
            raise RuntimeError("previous verification tree must be committed before the next forward")
        if args:
            raise RuntimeError("ring verification requires explicit keyword arguments")
        mask = kwargs.get("attention_mask")
        pos = kwargs.get("position_ids")
        ids = kwargs.get("input_ids")
        base = self.length(cache)
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or
                not isinstance(mask, torch.Tensor) or mask.ndim != 4 or
                not torch.is_floating_point(mask) or not isinstance(pos, torch.Tensor)):
            raise RuntimeError("ring requires batch-one tokens, additive 4D mask and explicit positions")
        nodes = int(ids.shape[1])
        if nodes < 1 or nodes > self.tree_capacity or mask.shape != (1, 1, nodes, base + nodes):
            raise RuntimeError("logical verification mask/tree shape mismatch")
        if (pos.shape != (3, 1, nodes) or pos.dtype != torch.long or
                not bool((mask[..., :base] == 0).all()) or bool((mask > 0).any()) or
                bool(torch.isnan(mask).any())):
            raise RuntimeError("unsupported verification positions or additive history mask")
        from ring_kv_cache import prepare_tree
        plan = prepare_tree(mask[..., -nodes:] == 0)
        expected = (base + self.delta + plan.depth_tensor).view(1, 1, nodes).expand(3, 1, nodes)
        if not torch.equal(pos, expected):
            raise RuntimeError("explicit mRoPE positions disagree with logical base and tree depth")
        self.pending = {"logical_base": base, "nodes": nodes,
                        "semantic_rope_base": base + self.delta}
        self.plan = plan
        self.update_counts = [0] * len(self.stores)
        self.in_forward = True
        try:
            if self.mode != "full":
                allowed = self.stores[0].build_attention_mask(plan)
                physical_mask = torch.zeros(allowed.shape, device=mask.device, dtype=mask.dtype)
                physical_mask.masked_fill_(~allowed, torch.finfo(mask.dtype).min)
                kwargs = dict(kwargs, attention_mask=physical_mask)
            result = self.original_forward(**kwargs)
            if self.mode != "full" and (any(x != 1 for x in self.update_counts) or
                    any(x.active_tree_nodes != nodes for x in self.stores)):
                raise RuntimeError("every Target layer must write exactly one complete tree")
            if self.length(cache) != base + nodes:
                raise RuntimeError("post-forward semantic tree length mismatch")
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.in_forward = False

    def compact(self, cache, selected, base, tree_nodes):
        if cache is not self.cache:
            return self.original_compact(cache, selected, base, tree_nodes)
        if self.pending is None or self.in_forward or self.failed:
            raise RuntimeError("ring commit has no completed verification forward")
        if self.pending["logical_base"] != base or self.pending["nodes"] != tree_nodes:
            raise RuntimeError("commit boundary differs from the verified tree")
        keep = int(selected.numel())
        before = self._cache_metadata()
        try:
            if self.mode == "full":
                self.original_compact(cache, selected, base, tree_nodes)
                reports = []
            else:
                from ring_kv_cache import prepare_commit
                prepared = prepare_commit(self.plan, selected)
                reports = [store.commit_prepared(prepared) for store in self.stores]
            if self.length(cache) != base + keep:
                raise RuntimeError("commit semantic length mismatch")
            after = self._cache_metadata()
            b = self.bytes_per_token
            self.rounds.append({**self.pending, "committed_tokens": keep,
                "semantic_length_after": base + keep,
                "cache_before_commit": before, "cache_after_commit": after,
                "original_append_and_compact_logical_rw_bytes": 2*b*(2*base + tree_nodes + keep),
                "original_history_copy_logical_rw_bytes": 4*b*base,
                "removed_history_copy_component_logical_rw_bytes": 0 if self.mode == "full" else 4*b*base,
                "layer_commit_reports": reports[:1],
                "all_layer_commit_reports_equal": all(x == reports[0] for x in reports) if reports else True})
            self.pending = None
            self.plan = None
        except BaseException:
            self.failed = True
            raise

    def _cache_metadata(self):
        if self.cache is None:
            return None
        if self.mode == "full":
            length = self.original_length(self.cache)
            return {"semantic_length": length, "occupied_tokens_per_layer": length,
                    "allocated_kv_storage_bytes": sum(
                        t.untyped_storage().nbytes() for x in self.cache.layers for t in (x.keys, x.values))}
        metadata = [x.metadata() for x in self.stores]
        return {"semantic_length": self.length(self.cache),
                "layer_count": len(metadata), "first_layer": metadata[0],
                "allocated_kv_storage_bytes": sum(x["allocated_kv_storage_bytes"] for x in metadata),
                "owned_storage_bytes": sum(x["owned_storage_bytes"] for x in metadata)}

    def finish(self):
        if self.finalized:
            return
        self.memory[self.current_phase] = self._peak()
        for name in ("allocated", "reserved"):
            peaks = [v.get("peak_" + name + "_bytes") for v in self.memory.values() if isinstance(v, dict)]
            peaks = [v for v in peaks if v is not None]
            self.memory["full_request_peak_" + name + "_bytes"] = max(peaks) if peaks else None
        try:
            self.final_cache_metadata = self._cache_metadata()
            self.final_layer_metadata = [x.metadata() for x in self.stores]
        except Exception as exc:
            self.final_cache_metadata = {"incomplete_cache_error": str(exc)}
            self.failed = True
        self.finalized = True

    def release(self):
        # The request-local cache is discarded, not reconstructed as a full
        # chronological DynamicCache (which would require retaining evicted KV).
        for layer, saved in reversed(self.layer_restore):
            for name, previous in saved.items():
                _restore_attribute(layer, name, previous)
            layer.keys = layer.keys.new_empty((1, layer.keys.shape[1], 0, layer.keys.shape[-1]))
            layer.values = layer.values.new_empty((1, layer.values.shape[1], 0, layer.values.shape[-1]))
        self.layer_restore.clear()
        self.stores.clear()
        self.plan = None
        self.pending = None
        self.cache = None

    def summary(self):
        result = {"mode": self.mode, "recent_capacity": self.recent_capacity,
                  "tree_capacity": self.tree_capacity, "prompt_tokens": self.prompt,
                  "bytes_per_token_all_target_layers": self.bytes_per_token,
                  "prefill_calls": self.prefill_calls, "completed_verifications": len(self.rounds),
                  "finalized": self.finalized, "failed": self.failed,
                  "draft_cache_policy": "unchanged_full_history",
                  "prefill_cache": self.prefill_metadata,
                  "final_cache": getattr(self, "final_cache_metadata", None),
                  "final_layer_metadata": self.final_layer_metadata,
                  "rounds": self.rounds, "memory_phase_peaks": self.memory,
                  "full_request_peak_allocated_bytes": self.memory["full_request_peak_allocated_bytes"],
                  "full_request_peak_reserved_bytes": self.memory["full_request_peak_reserved_bytes"],
                  "scope": "Target KV only; native full prefill; logical copy bytes are not DRAM counters",
                  "memory_scope": "caller resets before request; prefill includes preceding input preparation; max of all phases",
                  "timing_scope": "phase synchronization, conversion, mask mapping and validation are inside the request",
                  "cache_after_context": "request-local ring cache invalidated; no continuation outside context",
                  "lossless_claim": False}
        return copy.deepcopy(result)


@contextmanager
def ring_cache_policy(target, mode="protected_prompt", recent_capacity=128, tree_capacity=30):
    """Wrap exactly one native joint request; always restore Python methods.

    Caller must read ``state.summary()`` after context exit to obtain decode
    peaks, then use its maximum of phases as the complete request peak.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("nested or concurrent ring policies are unsupported")
    if mode not in MODES:
        raise ValueError("unsupported Target KV mode")
    if any(type(x) is not int or x < 1 for x in (recent_capacity, tree_capacity)):
        raise ValueError("ring and tree capacities must be positive integers")
    bridge = _bridge()
    state = RingCacheState(target, bridge, mode, recent_capacity, tree_capacity)
    original_forward_attribute = target.__dict__.get("forward", _MISSING)
    _ACTIVE = state
    try:
        bridge._prefill = state.prefill
        bridge._length = state.length
        bridge._compact = state.compact
        target.forward = state.forward
        try:
            yield state
        except BaseException:
            state.failed = True
            raise
    finally:
        try:
            state.finish()
        finally:
            bridge._prefill = state.original_prefill
            bridge._length = state.original_length
            bridge._compact = state.original_compact
            _restore_attribute(target, "forward", original_forward_attribute)
            state.release()
            _ACTIVE = None
