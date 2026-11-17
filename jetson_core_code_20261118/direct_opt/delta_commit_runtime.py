"""Exact accepted-path maintenance candidate; attention stays on active-width.

Only a protected prompt with enough recent capacity is supported. No eviction,
quantization or attention backend changes; the root runner owns GPU scheduling.
"""
from contextlib import contextmanager, ExitStack

import torch

import ring_cache_runtime as fixed
from active_ring_runtime import ActiveRingCacheState
from delta_kv_commit import prepare_delta_commit, delta_commit_
from ring_kv_cache import prepare_commit

VARIANT = 'active_width_delta_commit'


class DeltaCommitCacheState(ActiveRingCacheState):
    @torch.inference_mode()
    def compact(self, cache, selected, base, tree_nodes):
        if cache is not self.cache:
            return self.original_compact(cache, selected, base, tree_nodes)
        if self.pending is None or self.in_forward or self.failed:
            raise RuntimeError('delta commit has no completed verification')
        if self.pending['logical_base'] != base or self.pending['nodes'] != tree_nodes:
            raise RuntimeError('delta commit boundary differs from verified tree')
        before = self._cache_metadata()
        try:
            accepted = prepare_commit(self.plan, selected)
            first = self.stores[0]
            signature = lambda s: (s.mode, s.prefix_length, s.recent_count,
                s.recent_capacity, s.scratch_start, s.tree_capacity, s.next_position,
                s.active_tree_nodes, s.write_pointer)
            if any(signature(s) != signature(first) or s._plan is not self.plan
                   for s in self.stores):
                raise RuntimeError('delta commit layers disagree on pending layout')
            plan = prepare_delta_commit(accepted, mode=self.mode,
                prefix_length=first.prefix_length, recent_count=first.recent_count,
                recent_capacity=first.recent_capacity, scratch_start=first.scratch_start,
                tree_capacity=first.tree_capacity)
            count = accepted.accepted_tokens
            positions = torch.arange(first.next_position, first.next_position+count,
                                     device=self.device, dtype=torch.long)
            reports = []
            for store in self.stores:
                receipt = delta_commit_(store.keys, store.values, plan,
                    backend='triton' if store.keys.device.type == 'cuda' else 'cpu_reference')
                start, nodes = store.scratch_start, store.active_tree_nodes
                # Payload reads/writes have finished in stream order. Clear the
                # old tree's metadata before setting overlapping accepted slots.
                store.valid_mask[start:start+nodes].fill_(False)
                store.slot_positions[start:start+nodes].fill_(-1)
                store.slot_positions[start:start+count].copy_(positions)
                store.valid_mask[start:start+count].fill_(True)
                store.next_position += count
                store.recent_count += count
                store.write_pointer = store.recent_count % store.recent_capacity
                store.active_tree_nodes, store._plan = 0, None
                store.scratch_start = store.history_length
                reports.append(store._record('commit', receipt['copy_read_bytes'],
                    receipt['copy_write_bytes'], metadata_bytes=9*(count+nodes),
                    accepted_tokens=count, stored_tokens=count, evicted_recent_tokens=0,
                    scratch_payload_peak_bytes=receipt['scratch_payload_peak_bytes'],
                    moved_tokens=receipt['moved_tokens'],
                    payload_backend=receipt['backend'],
                    payload_kernel_launches=receipt['launches'],
                    previous_scratch_start=start, previous_scratch_nodes=nodes,
                    next_scratch_start=store.scratch_start,
                    active_view_payload_copy_bytes=0))
            if self.length(cache) != base+count:
                raise RuntimeError('delta commit semantic length mismatch')
            after = self._cache_metadata()
            b = self.bytes_per_token
            self.rounds.append({**self.pending, 'variant': VARIANT,
                'committed_tokens': count, 'semantic_length_after': base+count,
                'cache_before_commit': before, 'cache_after_commit': after,
                'original_append_and_compact_logical_rw_bytes': 2*b*(2*base+tree_nodes+count),
                'original_history_copy_logical_rw_bytes': 4*b*base,
                'removed_history_copy_component_logical_rw_bytes': 4*b*base,
                'layer_commit_reports': reports[:1],
                'all_layer_commit_reports_equal': all(r == reports[0] for r in reports)})
            self.pending = self.plan = None
        except BaseException:
            self.failed = True
            raise

    def summary(self):
        value = super().summary()
        value.update(variant=VARIANT, target_history_eviction=False,
                     commit_method='skip_inplace_rows_and_fuse_moved_rows',
                     attention_backend_changed=False, kv_quantized=False,
                     validation_claim='Candidate; full-model output validation required')
        return value


@contextmanager
def delta_commit_cache_policy(target, *, recent_capacity=512, tree_capacity=30):
    if fixed._ACTIVE is not None:
        raise RuntimeError('nested or concurrent cache policies are unsupported')
    original = fixed.RingCacheState
    state = None
    try:
        with ExitStack() as stack:
            try:
                fixed.RingCacheState = DeltaCommitCacheState
                state = stack.enter_context(fixed.ring_cache_policy(target,
                    mode='protected_prompt', recent_capacity=recent_capacity,
                    tree_capacity=tree_capacity))
            finally:
                fixed.RingCacheState = original
            yield state
    finally:
        fixed.RingCacheState = original
        if state is not None and fixed._ACTIVE is state:
            fixed._ACTIVE = None
