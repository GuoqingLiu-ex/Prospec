"""Value-preserving accepted-tail compaction for the HF 5.17 ViSpec bridge.

``compact_tail`` has the same signature as ``hf517_vispec_adapter._compact``.
It mutates an exclusively owned, inference-only DynamicCache. Unlike the
reference implementation, it does not copy the committed prefix. External
aliases to cache tensors must not be used as snapshots across this call.

The final narrow() is a view: rejected tail capacity stays allocated until a
later cache replacement (normally DynamicLayer.update's torch.cat). Reports
contain storage capacity and logical copy traffic, never measured DRAM bytes
or allocator peaks. No bridge or model is patched by importing this module.
"""
from __future__ import annotations

import operator

import torch


def _integer(value, name, *, minimum):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _storage_info(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr()), storage.nbytes()


def _nonoverlapping_layout(tensor):
    # A sufficient stride-only condition for safe writes. Allows native cache
    # slices with gaps between heads/batches; rejects expand/as_strided overlap.
    span = 1
    for stride, size in sorted((stride, size) for size, stride
                               in zip(tensor.shape, tensor.stride()) if size > 1):
        if stride < span:
            return False
        span += (size - 1) * stride
    return True


def _validate(cache, selected, base, tree_nodes):
    # Deferred dependency: CPU unit tests can exercise the exact contract even
    # on development machines without the server's transformers installation.
    from transformers.cache_utils import DynamicCache, DynamicLayer

    if not isinstance(cache, DynamicCache) or cache.offloading or not cache.layers:
        raise TypeError("bridge requires initialized non-offloaded DynamicCache")
    if any(type(layer) is not DynamicLayer for layer in cache.layers):
        raise TypeError("bridge supports full-attention DynamicLayer only")
    base = _integer(base, "base", minimum=0)
    tree_nodes = _integer(tree_nodes, "tree_nodes", minimum=1)
    if (not isinstance(selected, torch.Tensor) or selected.ndim != 1
            or selected.dtype != torch.long or selected.numel() < 1):
        raise ValueError("accepted path must contain a root in a 1D torch.long tensor")
    if selected.device.type not in {"cpu", "cuda"}:
        raise ValueError("accepted path must be on CPU or CUDA")
    # Matches the bridge's existing validation/synchronization. No full K/V
    # tensor is transferred to CPU. This check belongs inside request timing.
    positions = selected.tolist()
    if (positions[0] != 0 or len(set(positions)) != len(positions)
            or any(x < 0 or x >= tree_nodes for x in positions)):
        raise RuntimeError("accepted path contains invalid tree indices")

    expected = base + tree_nodes
    lengths = [int(layer.get_seq_length()) for layer in cache.layers]
    if len(set(lengths)) != 1:
        raise RuntimeError("native Target layers have inconsistent KV lengths")
    if lengths[0] != expected:
        raise RuntimeError("native tree cache length differs from actual write count")

    seen_storages = set()
    layer_info = []
    common_batch = None
    for layer_idx, layer in enumerate(cache.layers):
        if not layer.is_initialized:
            raise TypeError("bridge requires initialized DynamicLayer tensors")
        keys, values = layer.keys, layer.values
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("DynamicLayer keys and values must be tensors")
        if (keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape
                or keys.shape[-2] != expected or any(size < 1 for size in keys.shape)):
            raise ValueError("K/V must have matching [batch, heads, sequence, head_dim] shapes")
        if keys.device != values.device or keys.dtype != values.dtype:
            raise ValueError("each layer's K/V device and dtype must match")
        if (keys.device.type not in {"cpu", "cuda"} or keys.layout != torch.strided
                or values.layout != torch.strided or not keys.is_floating_point()):
            raise ValueError("K/V must be floating-point strided CPU or CUDA tensors")
        if keys.requires_grad or values.requires_grad:
            raise ValueError("tail compaction requires inference-only K/V")
        if not _nonoverlapping_layout(keys) or not _nonoverlapping_layout(values):
            raise ValueError("K/V must have nonoverlapping writable layouts")
        if common_batch is None:
            common_batch = keys.shape[0]
        elif keys.shape[0] != common_batch:
            raise ValueError("all cache layers must have the same batch size")
        info = {"layer": layer_idx, "shape_before": list(keys.shape),
                "dtype": str(keys.dtype), "device": str(keys.device)}
        for label, tensor in (("keys", keys), ("values", values)):
            storage_id, capacity = _storage_info(tensor)
            # The native DynamicLayer owns separate K/V allocations. Reject
            # unusual aliases instead of risking an in-place cross-layer write.
            if storage_id in seen_storages:
                raise ValueError("K/V cache tensors must not share backing storage")
            seen_storages.add(storage_id)
            info[f"{label}_logical_bytes_before"] = _tensor_bytes(tensor)
            info[f"{label}_storage_bytes_before"] = capacity
            info[f"{label}_selected_bytes"] = (
                _tensor_bytes(tensor) // expected * len(positions))
        layer_info.append(info)
    return base, tree_nodes, len(positions), layer_info


@torch.inference_mode()
def compact_tail(cache, selected, base, tree_nodes):
    """Gather the accepted tree tail, copy it back, and shorten logical views.

    All guards run before the first cache mutation. Validation follows the
    bridge's root/unique/bounds rules; accepted indices need not be sorted.
    Each K/V gather finishes before its possibly overlapping destination is
    overwritten. Only one gathered tensor remains live at a time.

    Returns JSON-serializable per-call bookkeeping; callers may ignore it just
    as they ignore the original bridge's None return. All bookkeeping and the
    selected.tolist() synchronization must remain inside request timing.
    """
    base, tree_nodes, keep, layers = _validate(cache, selected, base, tree_nodes)
    new_length = base + keep
    device_indices = {}
    for layer, info in zip(cache.layers, layers):
        device = layer.keys.device
        if device not in device_indices:
            device_indices[device] = selected.to(device=device)
        index = device_indices[device]
        for label in ("keys", "values"):
            tensor = getattr(layer, label)
            # Gather all selected rows before copy_: selected can reorder rows
            # or read locations that the destination is about to overwrite.
            scratch = tensor.narrow(-2, base, tree_nodes).index_select(-2, index)
            tensor.narrow(-2, base, keep).copy_(scratch)
            del scratch
            logical = tensor.narrow(-2, 0, new_length)
            setattr(layer, label, logical)
            info[f"{label}_logical_bytes_after"] = _tensor_bytes(logical)
            info[f"{label}_storage_bytes_after"] = _storage_info(logical)[1]
        info["shape_after"] = list(layer.keys.shape)
    if any(int(layer.get_seq_length()) != new_length for layer in cache.layers):
        raise RuntimeError("accepted path compaction failed")

    def total(suffix):
        return sum(layer[f"{label}_{suffix}"]
                   for layer in layers for label in ("keys", "values"))

    selected_bytes = total("selected_bytes")
    logical_before, logical_after = total("logical_bytes_before"), total("logical_bytes_after")
    storage_before, storage_after = total("storage_bytes_before"), total("storage_bytes_after")
    return {
        "method": "accepted_tail_gather_copy_view",
        "base": base, "tree_nodes": tree_nodes, "accepted_tokens": keep,
        "length_before": base + tree_nodes, "length_after": new_length,
        "layers": layers,
        "logical_cache_bytes_before": logical_before,
        "logical_cache_bytes_after": logical_after,
        "unique_storage_bytes_before": storage_before,
        "unique_storage_bytes_after": storage_after,
        "backing_minus_logical_bytes_after": storage_after - logical_after,
        "immediate_backing_storage_release_bytes": storage_before - storage_after,
        "selected_payload_bytes": selected_bytes,
        "scratch_payload_peak_bytes": max(
            layer[f"{label}_selected_bytes"]
            for layer in layers for label in ("keys", "values")),
        "logical_copy_read_bytes": 2 * selected_bytes,
        "logical_copy_write_bytes": 2 * selected_bytes,
        "logical_copy_total_bytes": 4 * selected_bytes,
        "reference_full_prefix_copy_total_bytes": 2 * logical_after,
        "traffic_scope": "K/V payload of gather then copy; excludes indices, allocator, and CPU validation",
        "traffic_is_measured_dram": False,
        "scratch_is_allocator_peak": False,
        "selected_validation_transfers_to_cpu": selected.device.type == "cuda",
        "capacity_note": "narrow views retain backing storage until a later replacement such as torch.cat",
    }
