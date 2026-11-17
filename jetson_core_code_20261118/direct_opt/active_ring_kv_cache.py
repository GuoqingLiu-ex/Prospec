"""CPU-developed candidate: fixed KV storage, active-width attention views.

Not deployed or GPU-validated. This preserves RingKVLayer's retention policy,
not full-context semantics after eviction. Layout while the ring is unfilled:
[protected prompt | filled recent | current tree | unused backing]. Once full,
its tree starts at the fixed end of the recent region. No history is packed or
copied to create the attention view; narrow retains the entire backing storage.

Integration contract (no existing runtime is changed by this module):
- Substitute ActiveRingKVLayer when creating stores. keys/values remain full
  backing tensors; append_tree returns the active K/V views used by attention.
- Build one common build_attention_mask(plan) before all layer appends; its
  last dimension is attention_length(plan), not physical_capacity.
- A runtime get_mask_sizes(query_length) must return
  (store.history_length + query_length, 0), with query_length <= tree_capacity.
  The current runtime's keys.shape[-2] instead reports reserved capacity.
- get_seq_length remains next_position + active_tree_nodes: semantic positions
  and mRoPE offsets must never be replaced by physical/active widths.
- Consumers must preserve exclusive cache ownership and not keep view snapshots
  after commit/append. Narrow views may be non-contiguous across KV heads;
  downstream repacking, all checks/masks and copy costs require GPU measurement.

All traffic counters are logical payload bytes, not measured DRAM. The fixed
backing allocation and initial zero fill are unchanged. This candidate removes
padding from verification attention shapes; no speed/quality claim is made.
"""
from __future__ import annotations

import torch

from ring_kv_cache import (RingKVLayer, RingCommitPlan, _bytes, prepare_tree,
                           prepare_commit)


class ActiveRingKVLayer(RingKVLayer):
    """Compatible primitive API with explicit active attention width."""

    @classmethod
    @torch.inference_mode()
    def from_prefill(cls, keys, values, *, mode="protected_prompt", recent_capacity=128,
                     tree_capacity=30):
        self = super().from_prefill(keys, values, mode=mode,
                                   recent_capacity=recent_capacity, tree_capacity=tree_capacity)
        self.scratch_start = self.history_length
        self.attention_shape_totals = {
            'verification_calls': 0, 'query_tokens': 0, 'active_key_tokens': 0,
            'fixed_capacity_key_tokens': 0, 'active_query_key_pairs': 0,
            'fixed_capacity_query_key_pairs': 0,
        }
        return self

    @property
    def history_length(self):
        """Currently filled history slots; equal to capacity only after fill."""
        return self.prefix_length + self.recent_count

    @property
    def valid_history_mask(self):
        # Pending scratch can live in the not-yet-filled recent region. It must
        # never be exposed as history to a caller constructing an ancestor mask.
        return self.valid_mask[:self.history_length]

    def attention_length(self, plan=None):
        """Future width with plan, or current width including a pending tree."""
        if plan is None:
            return self.history_length + self.active_tree_nodes
        self._check_plan(plan)
        if self.active_tree_nodes and self._plan is not plan:
            raise RuntimeError('a different tree is already pending')
        return self.history_length + plan.nodes

    def attention_view(self):
        """Zero-copy narrow views; backing storage still retains full capacity."""
        count = self.attention_length()
        return self.keys.narrow(-2, 0, count), self.values.narrow(-2, 0, count)

    @torch.inference_mode()
    def build_attention_mask(self, plan):
        width = self.attention_length(plan)
        mask = torch.zeros((1, 1, plan.nodes, width), dtype=torch.bool, device=self.keys.device)
        mask[..., :self.history_length].copy_(self.valid_history_mask)
        mask[..., self.history_length:].copy_(plan.mask)
        return mask

    @torch.inference_mode()
    def append_tree(self, keys, values, plan):
        # The inherited guards and copy accounting also apply to a moving
        # scratch start. No K/V is returned to the model until both writes finish.
        super().append_tree(keys, values, plan)
        active = self.attention_length()
        totals = self.attention_shape_totals
        totals['verification_calls'] += 1
        totals['query_tokens'] += plan.nodes
        totals['active_key_tokens'] += active
        totals['fixed_capacity_key_tokens'] += self.physical_capacity
        totals['active_query_key_pairs'] += plan.nodes * active
        totals['fixed_capacity_query_key_pairs'] += plan.nodes * self.physical_capacity
        self.last_operation.update(active_attention_tokens=active,
                                   excluded_padding_tokens=self.physical_capacity-active,
                                   active_view_payload_copy_bytes=0)
        return self.attention_view()

    @torch.inference_mode()
    def commit_prepared(self, commit_plan):
        """Gather first, clear OLD scratch metadata, then write accepted history.

        Clearing after writing would invalidate newly committed recent slots
        when scratch overlaps their destination. No scratch K/V zeroing or
        history packing is needed: unused backing is excluded from attention.
        """
        if not self.active_tree_nodes:
            raise RuntimeError('no pending tree to commit')
        if not isinstance(commit_plan, RingCommitPlan):
            raise TypeError('commit_prepared requires a RingCommitPlan')
        if commit_plan.tree_plan is not self._plan:
            raise ValueError('commit plan does not belong to this pending tree')
        if commit_plan._indices.device != self.keys.device:
            raise ValueError('commit plan and KV store must be on the same device')
        accepted = commit_plan.accepted_tokens
        kept = min(accepted, self.recent_capacity)
        index = commit_plan._indices[-kept:]
        old_start, old_nodes = self.scratch_start, self.active_tree_nodes
        gathered_keys = self.keys.narrow(-2, old_start, old_nodes).index_select(-2, index)
        gathered_values = self.values.narrow(-2, old_start, old_nodes).index_select(-2, index)
        absolute = torch.arange(self.next_position+accepted-kept, self.next_position+accepted,
                                device=self.keys.device, dtype=torch.long)
        # All validation/gathers complete before any mutation. Only metadata
        # for the previous tree is cleared, BEFORE overlap destinations are set.
        self.valid_mask[old_start:old_start+old_nodes].fill_(False)
        self.slot_positions[old_start:old_start+old_nodes].fill_(-1)
        begin = (self.write_pointer+accepted-kept) % self.recent_capacity
        first = min(kept, self.recent_capacity-begin)
        for source, destination, count in ((0, begin, first), (first, 0, kept-first)):
            if not count:
                continue
            destination += self.prefix_length
            self.keys.narrow(-2, destination, count).copy_(gathered_keys.narrow(-2, source, count))
            self.values.narrow(-2, destination, count).copy_(gathered_values.narrow(-2, source, count))
            self.slot_positions[destination:destination+count].copy_(absolute[source:source+count])
            self.valid_mask[destination:destination+count].fill_(True)
        evicted = max(0, self.recent_count+accepted-self.recent_capacity)
        self.next_position += accepted
        self.write_pointer = (self.write_pointer+accepted) % self.recent_capacity
        self.recent_count = min(self.recent_capacity, self.recent_count+accepted)
        self.active_tree_nodes, self._plan = 0, None
        self.scratch_start = self.history_length
        payload = _bytes(gathered_keys)+_bytes(gathered_values)
        return self._record('commit', 2*payload, 2*payload,
                            metadata_bytes=9*(kept+old_nodes), accepted_tokens=accepted,
                            stored_tokens=kept, evicted_recent_tokens=evicted,
                            scratch_payload_peak_bytes=payload, previous_scratch_start=old_start,
                            previous_scratch_nodes=old_nodes, next_scratch_start=self.scratch_start,
                            active_view_payload_copy_bytes=0)

    def metadata(self):
        result = super().metadata()
        totals = dict(self.attention_shape_totals)
        totals['excluded_padding_key_tokens'] = totals['fixed_capacity_key_tokens']-totals['active_key_tokens']
        totals['excluded_padding_query_key_pairs'] = (totals['fixed_capacity_query_key_pairs']-
                                                     totals['active_query_key_pairs'])
        result.update(layout='active_width_ring_candidate', status='CPU_tested_candidate_not_GPU_validated',
                      reserved_physical_capacity=self.physical_capacity,
                      active_attention_tokens=self.attention_length(), scratch_start=self.scratch_start,
                      attention_shape_totals=totals, active_view_payload_copy_bytes=0,
                      active_views_may_be_noncontiguous=True,
                      backing_capacity_released_by_narrow=False)
        return result
