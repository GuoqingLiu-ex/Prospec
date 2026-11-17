"""Bounded, explicitly lossy KV primitives; no model or HF monkey-patching.

Physical layout: [protected prompt | recent ring | tree scratch]. K/V already
contain the model's RoPE rotation and are copied without modification. Sequence
offsets never wrap: only physical recent slots wrap. Attention must use the
returned physical mask, never a causal triangle based on physical slot order.
All reported traffic is logical payload traffic, not measured DRAM traffic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import operator

import torch


def _positive(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _check_kv(keys, values):
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise TypeError("keys and values must be tensors")
    if (keys.ndim != 4 or keys.shape != values.shape or keys.shape[0] != 1
            or keys.shape[1] < 1 or keys.shape[-1] < 1):
        raise ValueError("matching K/V [1, heads, sequence, head_dim] required")
    if keys.device != values.device or keys.dtype != values.dtype:
        raise ValueError("K/V dtype and device must match")
    if (not keys.is_floating_point() or keys.device.type not in {"cpu", "cuda"}
            or keys.layout != torch.strided or values.layout != torch.strided):
        raise ValueError("floating-point strided CPU or CUDA K/V required")
    if keys.requires_grad or values.requires_grad:
        raise ValueError("ring KV is inference-only")


@dataclass(frozen=True)
class RingTreePlan:
    """A tree mask validated once per verification round, shared across layers.

    Treat mask/depth_tensor as immutable. Metadata positions are sequence
    offsets, not mRoPE position IDs (the caller keeps its original mRoPE delta).
    """
    mask: torch.Tensor
    depth_tensor: torch.Tensor
    ancestors: tuple
    depths: tuple
    nodes: int


@dataclass(frozen=True, init=False, eq=False)
class RingCommitPlan:
    """One validated acceptance decision, shared by this round's layers.

    Construct through prepare_commit only. The private index tensor is an owned
    copy, and must not be mutated. The public positions tuple and tree identity
    are immutable. A new RingTreePlan identifies each verification round.
    """
    tree_plan: RingTreePlan
    positions: tuple
    _indices: torch.Tensor = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError("use prepare_commit(plan, selected) to construct RingCommitPlan")

    @property
    def accepted_tokens(self):
        return len(self.positions)


@torch.inference_mode()
def prepare_commit(plan, selected):
    """Validate an EOS-clipped accepted path once, outside the layer loop.

    selected.tolist() is the single acceptance-validation host transfer per
    round. The resulting device index copy is also prepared once and shared;
    no K/V data is read. Keep this work inside full-request timing.
    """
    if not isinstance(plan, RingTreePlan):
        raise TypeError("prepare_commit requires a RingTreePlan")
    if (not isinstance(selected, torch.Tensor) or selected.ndim != 1
            or selected.dtype != torch.long or selected.numel() < 1
            or selected.device.type not in {"cpu", "cuda"}):
        raise ValueError("selected must be a nonempty 1D torch.long accepted path")
    positions = tuple(selected.tolist())
    if (positions[0] != 0 or len(set(positions)) != len(positions)
            or any(i < 0 or i >= plan.nodes for i in positions)):
        raise ValueError("selected contains invalid tree indices")
    ancestors = set()
    for node in positions:
        ancestors.add(node)
        if set(plan.ancestors[node]) != ancestors:
            raise ValueError("selected must be a complete ancestor path, possibly EOS-clipped")
    prepared = object.__new__(RingCommitPlan)
    object.__setattr__(prepared, "tree_plan", plan)
    object.__setattr__(prepared, "positions", positions)
    # Own the index data: later changes to the caller's selected cannot silently
    # change a decision already validated on CPU. One copy for all layers.
    object.__setattr__(prepared, "_indices", selected.detach().to(device=plan.mask.device, copy=True))
    return prepared


@torch.inference_mode()
def prepare_tree(ancestor_mask):
    """Validate a boolean [N,N] or [1,1,N,N] rooted tree ancestor mask.

    This copies only the small tree mask to CPU for validation. It must remain
    inside request timing. No K/V or feature tensor is transferred to CPU.
    """
    if not isinstance(ancestor_mask, torch.Tensor) or ancestor_mask.dtype != torch.bool:
        raise ValueError("ancestor_mask must be a boolean tensor")
    if ancestor_mask.ndim == 4 and ancestor_mask.shape[:2] == (1, 1):
        ancestor_mask = ancestor_mask[0, 0]
    if (ancestor_mask.ndim != 2 or ancestor_mask.shape[0] != ancestor_mask.shape[1]
            or ancestor_mask.shape[0] < 1 or ancestor_mask.device.type not in {"cpu", "cuda"}):
        raise ValueError("ancestor_mask must be a nonempty square [N,N] or [1,1,N,N]")
    rows = ancestor_mask.tolist()
    ancestors = tuple(tuple(j for j, yes in enumerate(row) if yes) for row in rows)
    depths = tuple(len(row) - 1 for row in ancestors)
    if ancestors[0] != (0,):
        raise ValueError("tree root must attend only to itself")
    for node, row in enumerate(ancestors):
        row_set = set(row)
        if node not in row_set or 0 not in row_set:
            raise ValueError("each tree node must include itself and root")
        if any(not set(ancestors[parent]) < row_set for parent in row if parent != node):
            raise ValueError("tree ancestor mask must be transitively closed and acyclic")
        if sorted(depths[parent] for parent in row) != list(range(len(row))):
            raise ValueError("tree ancestors must form one path")
    return RingTreePlan(ancestor_mask.detach().clone(),
                        torch.tensor(depths, device=ancestor_mask.device, dtype=torch.long),
                        ancestors, depths, len(rows))


class RingKVLayer:
    """One independent attention layer, batch one, with a fixed slot budget."""

    @classmethod
    @torch.inference_mode()
    def from_prefill(cls, keys, values, *, mode="protected_prompt", recent_capacity=128,
                     tree_capacity=30):
        _check_kv(keys, values)
        if mode not in {"protected_prompt", "pure_recent"}:
            raise ValueError("mode must be protected_prompt or pure_recent")
        recent_capacity = _positive(recent_capacity, "recent_capacity")
        tree_capacity = _positive(tree_capacity, "tree_capacity")
        self = cls()
        self.mode = mode
        self.prompt_length = int(keys.shape[-2])
        self.prefix_length = self.prompt_length if mode == "protected_prompt" else 0
        self.recent_capacity, self.tree_capacity = recent_capacity, tree_capacity
        self.history_capacity = self.prefix_length + recent_capacity
        self.scratch_start = self.history_capacity
        self.physical_capacity = self.history_capacity + tree_capacity
        self.next_position = self.prompt_length
        self.active_tree_nodes = 0
        self._plan = None
        shape = (*keys.shape[:2], self.physical_capacity, keys.shape[-1])
        # Invalid slots must contain initialized finite data: NaN in an unused
        # key could survive addition of an -inf mask in a math attention path.
        self.keys = torch.zeros(shape, dtype=keys.dtype, device=keys.device)
        self.values = torch.zeros_like(self.keys)
        self.valid_mask = torch.zeros(self.physical_capacity, dtype=torch.bool, device=keys.device)
        self.slot_positions = torch.full((self.physical_capacity,), -1, dtype=torch.long, device=keys.device)
        self.recent_count = 0 if self.prefix_length else min(self.prompt_length, recent_capacity)
        self.write_pointer = self.recent_count % recent_capacity
        kept = self.prompt_length if self.prefix_length else self.recent_count
        if kept:
            source_start = 0 if self.prefix_length else self.prompt_length - kept
            self.keys[..., :kept, :].copy_(keys.narrow(-2, source_start, kept))
            self.values[..., :kept, :].copy_(values.narrow(-2, source_start, kept))
            self.valid_mask[:kept].fill_(True)
            self.slot_positions[:kept].copy_(torch.arange(source_start, source_start + kept,
                                                         device=keys.device))
        self.payload_copy_totals = {
            "read_bytes": 0, "write_bytes": 0, "total_bytes": 0,
            "kv_zero_fill_write_bytes": 0, "metadata_update_write_bytes": 0,
        }
        self.operation_counts = {"prefill": 0, "append_tree": 0, "commit": 0}
        self.last_operation = None
        payload = 2 * keys.shape[0] * keys.shape[1] * kept * keys.shape[-1] * keys.element_size()
        self._record("prefill", payload, payload,
                     zero_bytes=_bytes(self.keys) + _bytes(self.values),
                     metadata_bytes=9 * (self.physical_capacity + kept),
                     discarded_prompt_tokens=self.prompt_length - kept)
        return self

    @property
    def valid_history_mask(self):
        return self.valid_mask[:self.history_capacity]

    def _check_plan(self, plan):
        if not isinstance(plan, RingTreePlan):
            raise TypeError("use prepare_tree(ancestor_mask) to construct RingTreePlan")
        if plan.nodes > self.tree_capacity:
            raise ValueError("tree exceeds scratch capacity")
        if plan.mask.device != self.keys.device:
            raise ValueError("tree plan and KV store must be on the same device")

    @torch.inference_mode()
    def build_attention_mask(self, plan):
        """Build once, before any layer append: True means allowed attention."""
        self._check_plan(plan)
        if self.active_tree_nodes and self._plan is not plan:
            raise RuntimeError("a different tree is already pending")
        mask = torch.zeros((1, 1, plan.nodes, self.physical_capacity),
                           dtype=torch.bool, device=self.keys.device)
        mask[..., :self.history_capacity].copy_(self.valid_history_mask)
        mask[..., self.scratch_start:self.scratch_start + plan.nodes].copy_(plan.mask)
        return mask

    @torch.inference_mode()
    def append_tree(self, keys, values, plan):
        """Copy a tree into scratch and return physical K/V, without packing."""
        _check_kv(keys, values)
        self._check_plan(plan)
        if self.active_tree_nodes:
            raise RuntimeError("commit the pending tree before appending another")
        if (keys.shape[:2] != self.keys.shape[:2] or keys.shape[-1] != self.keys.shape[-1]
                or keys.shape[-2] != plan.nodes or keys.dtype != self.keys.dtype
                or keys.device != self.keys.device):
            raise ValueError("tree K/V shape, dtype, and device must match store and plan")
        owned = {self.keys.untyped_storage().data_ptr(), self.values.untyped_storage().data_ptr()}
        if keys.untyped_storage().data_ptr() in owned or values.untyped_storage().data_ptr() in owned:
            raise ValueError("append input must not alias ring backing storage")
        start, count = self.scratch_start, plan.nodes
        self.keys.narrow(-2, start, count).copy_(keys)
        self.values.narrow(-2, start, count).copy_(values)
        self.valid_mask[start:start + count].fill_(True)
        self.slot_positions[start:start + count].copy_(plan.depth_tensor + self.next_position)
        self.active_tree_nodes, self._plan = count, plan
        payload = _bytes(keys) + _bytes(values)
        self._record("append_tree", payload, payload, metadata_bytes=9 * count,
                     tree_nodes=count, tree_base=self.next_position)
        return self.keys, self.values

    @torch.inference_mode()
    def commit(self, selected):
        """Compatibility wrapper; multi-layer callers should prepare once."""
        if not self.active_tree_nodes:
            raise RuntimeError("no pending tree to commit")
        return self.commit_prepared(prepare_commit(self._plan, selected))

    @torch.inference_mode()
    def commit_prepared(self, commit_plan):
        """Apply a shared decision without host reads or path revalidation.

        If more than C tokens are accepted, only the final C are gathered and
        stored. The semantic position still advances by the entire path length.
        """
        if not self.active_tree_nodes:
            raise RuntimeError("no pending tree to commit")
        if not isinstance(commit_plan, RingCommitPlan):
            raise TypeError("commit_prepared requires a RingCommitPlan")
        if commit_plan.tree_plan is not self._plan:
            raise ValueError("commit plan does not belong to this pending tree")
        if commit_plan._indices.device != self.keys.device:
            raise ValueError("commit plan and KV store must be on the same device")
        accepted = commit_plan.accepted_tokens
        kept = min(accepted, self.recent_capacity)
        index = commit_plan._indices[-kept:]
        # Gather before writing: this remains safe even if a future layout
        # places scratch adjacent to or overlapping the ring destination.
        gathered_keys = self.keys.narrow(-2, self.scratch_start, self.active_tree_nodes).index_select(-2, index)
        gathered_values = self.values.narrow(-2, self.scratch_start, self.active_tree_nodes).index_select(-2, index)
        absolute = torch.arange(self.next_position + accepted - kept, self.next_position + accepted,
                                device=self.keys.device, dtype=torch.long)
        begin = (self.write_pointer + accepted - kept) % self.recent_capacity
        first = min(kept, self.recent_capacity - begin)
        for source, destination, count in ((0, begin, first), (first, 0, kept - first)):
            if not count:
                continue
            destination += self.prefix_length
            self.keys.narrow(-2, destination, count).copy_(gathered_keys.narrow(-2, source, count))
            self.values.narrow(-2, destination, count).copy_(gathered_values.narrow(-2, source, count))
            self.slot_positions[destination:destination + count].copy_(absolute[source:source + count])
            self.valid_mask[destination:destination + count].fill_(True)
        evicted = max(0, self.recent_count + accepted - self.recent_capacity)
        self.next_position += accepted
        self.write_pointer = (self.write_pointer + accepted) % self.recent_capacity
        self.recent_count = min(self.recent_capacity, self.recent_count + accepted)
        self.valid_mask[self.scratch_start:].fill_(False)
        self.slot_positions[self.scratch_start:].fill_(-1)
        self.active_tree_nodes, self._plan = 0, None
        payload = _bytes(gathered_keys) + _bytes(gathered_values)
        return self._record("commit", 2 * payload, 2 * payload,
                            metadata_bytes=9 * (kept + self.tree_capacity),
                            accepted_tokens=accepted, stored_tokens=kept,
                            evicted_recent_tokens=evicted,
                            scratch_payload_peak_bytes=payload)

    def _record(self, operation, reads, writes, *, zero_bytes=0, metadata_bytes=0, **details):
        totals = self.payload_copy_totals
        totals["read_bytes"] += int(reads)
        totals["write_bytes"] += int(writes)
        totals["total_bytes"] += int(reads + writes)
        totals["kv_zero_fill_write_bytes"] += int(zero_bytes)
        totals["metadata_update_write_bytes"] += int(metadata_bytes)
        self.operation_counts[operation] += 1
        report = {"operation": operation, "copy_read_bytes": int(reads),
                  "copy_write_bytes": int(writes), "copy_total_bytes": int(reads + writes),
                  "kv_zero_fill_write_bytes": int(zero_bytes),
                  "metadata_update_write_bytes": int(metadata_bytes),
                  "next_position": self.next_position, "write_pointer": self.write_pointer,
                  "recent_count": self.recent_count, "traffic_is_measured_dram": False,
                  **details}
        self.last_operation = report
        return dict(report)

    def metadata(self):
        """No tensor data copied to CPU; positions are exposed separately."""
        kv_bytes = self.keys.untyped_storage().nbytes() + self.values.untyped_storage().nbytes()
        owned_bytes = kv_bytes + self.slot_positions.untyped_storage().nbytes() + self.valid_mask.untyped_storage().nbytes()
        return {
            "mode": self.mode, "lossy": True, "prompt_length": self.prompt_length,
            "prefix_length": self.prefix_length, "recent_capacity": self.recent_capacity,
            "tree_capacity": self.tree_capacity, "physical_capacity": self.physical_capacity,
            "next_position": self.next_position, "active_tree_nodes": self.active_tree_nodes,
            "write_pointer": self.write_pointer, "recent_count": self.recent_count,
            "valid_history_tokens": self.prefix_length + self.recent_count,
            "allocated_kv_storage_bytes": int(kv_bytes), "owned_storage_bytes": int(owned_bytes),
            "payload_copy_totals": dict(self.payload_copy_totals),
            "operation_counts": dict(self.operation_counts),
            "traffic_scope": "K/V payload copies; zero-fill and metadata writes reported separately; attention reads excluded",
            "positions_are_sequence_offsets": True, "rotated_keys_modified": False,
        }
