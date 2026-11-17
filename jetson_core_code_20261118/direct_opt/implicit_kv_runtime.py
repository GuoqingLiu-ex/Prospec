"""No-eviction Target cache with implicit positions and the caller's hard mask.

This independent engineering candidate keeps active-width K/V and delta payload
commit. It removes per-layer validity/position tensors and their GPU updates;
no attention arithmetic, backend, RoPE, acceptance or Draft path is changed.
Only [entire prompt | accepted generated history | current tree] is supported.
Overflow raises; there is no wrap, eviction or fallback. CPU tests cannot prove
GPU bitwise output or throughput. All traffic accounting is logical, not DRAM.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import operator
from types import MethodType

import torch

import ring_cache_runtime as fixed
from active_ring_kv_cache import ActiveRingKVLayer
from active_ring_runtime import ActiveRingCacheState
from delta_kv_commit import prepare_delta_commit, delta_commit_
from ring_kv_cache import _check_kv, _positive, _bytes, prepare_tree, prepare_commit

VARIANT = 'active_width_delta_commit_implicit_layout'


class ImplicitKVLayer(ActiveRingKVLayer):
    """Contiguous valid history is represented by two host integers, not maps."""

    @classmethod
    @torch.inference_mode()
    def from_prefill(cls, keys, values, *, mode='protected_prompt',
                     recent_capacity=512, tree_capacity=30):
        _check_kv(keys, values)
        if mode != 'protected_prompt':
            raise ValueError('implicit layout requires the entire protected prompt')
        self = cls()
        self.mode = mode
        self.recent_capacity = _positive(recent_capacity, 'recent_capacity')
        self.tree_capacity = _positive(tree_capacity, 'tree_capacity')
        if self.tree_capacity > 30:
            raise ValueError('delta payload commit supports at most 30 tree nodes')
        self.prompt_length = self.prefix_length = int(keys.shape[-2])
        if not self.prompt_length:
            raise ValueError('implicit layout requires a nonempty prefill')
        self.history_capacity = self.prefix_length + self.recent_capacity
        self.physical_capacity = self.history_capacity + self.tree_capacity
        self.next_position = self.prompt_length
        self.recent_count = self.write_pointer = self.active_tree_nodes = 0
        self.scratch_start = self.prompt_length
        self._plan = None
        shape = (*keys.shape[:2], self.physical_capacity, keys.shape[-1])
        # Keep the established initialization guarantee. No metadata tensors are
        # allocated, and all unused payload remains excluded from attention.
        self.keys = torch.zeros(shape, dtype=keys.dtype, device=keys.device)
        self.values = torch.zeros_like(self.keys)
        self.keys[..., :self.prompt_length, :].copy_(keys)
        self.values[..., :self.prompt_length, :].copy_(values)
        self.payload_copy_totals = dict(read_bytes=0, write_bytes=0, total_bytes=0,
            kv_zero_fill_write_bytes=0, metadata_update_write_bytes=0)
        self.operation_counts = dict(prefill=0, append_tree=0, commit=0)
        self.attention_shape_totals = dict(verification_calls=0, query_tokens=0,
            active_key_tokens=0, fixed_capacity_key_tokens=0,
            active_query_key_pairs=0, fixed_capacity_query_key_pairs=0)
        self.last_operation = None
        payload = _bytes(keys) + _bytes(values)
        self._record('prefill', payload, payload,
                     zero_bytes=_bytes(self.keys)+_bytes(self.values),
                     metadata_bytes=0, discarded_prompt_tokens=0)
        return self

    @property
    def valid_history_mask(self):
        raise RuntimeError('implicit layout must reuse the validated caller mask')

    def build_attention_mask(self, plan):
        raise RuntimeError('implicit layout must not rematerialize a mask')

    @torch.inference_mode()
    def append_tree(self, keys, values, plan):
        _check_kv(keys, values)
        self._check_plan(plan)
        if self.active_tree_nodes:
            raise RuntimeError('commit the pending tree before another append')
        if (self.scratch_start != self.history_length or
                self.next_position != self.history_length or
                not 0 <= self.recent_count <= self.recent_capacity):
            raise RuntimeError('implicit history must be contiguous without eviction')
        if (keys.shape[:2] != self.keys.shape[:2] or keys.shape[-1] != self.keys.shape[-1]
                or keys.shape[-2] != plan.nodes or keys.dtype != self.keys.dtype
                or keys.device != self.keys.device):
            raise ValueError('tree K/V shape, dtype and device must match store')
        owned = {self.keys.untyped_storage().data_ptr(), self.values.untyped_storage().data_ptr()}
        if keys.untyped_storage().data_ptr() in owned or values.untyped_storage().data_ptr() in owned:
            raise ValueError('append input must not alias owned K/V storage')
        start, count = self.scratch_start, plan.nodes
        self.keys.narrow(-2, start, count).copy_(keys)
        self.values.narrow(-2, start, count).copy_(values)
        self.active_tree_nodes, self._plan = count, plan
        active = self.attention_length()
        totals = self.attention_shape_totals
        totals['verification_calls'] += 1
        totals['query_tokens'] += count
        totals['active_key_tokens'] += active
        totals['fixed_capacity_key_tokens'] += self.physical_capacity
        totals['active_query_key_pairs'] += count*active
        totals['fixed_capacity_query_key_pairs'] += count*self.physical_capacity
        payload = _bytes(keys)+_bytes(values)
        self._record('append_tree', payload, payload, metadata_bytes=0,
                     tree_nodes=count, tree_base=self.next_position,
                     active_attention_tokens=active,
                     excluded_padding_tokens=self.physical_capacity-active,
                     active_view_payload_copy_bytes=0)
        return self.attention_view()

    def commit_prepared(self, plan):
        raise RuntimeError('implicit store requires the coordinated delta commit')

    def metadata(self):
        kv = self.keys.untyped_storage().nbytes()+self.values.untyped_storage().nbytes()
        totals = dict(self.attention_shape_totals)
        totals['excluded_padding_key_tokens'] = totals['fixed_capacity_key_tokens']-totals['active_key_tokens']
        totals['excluded_padding_query_key_pairs'] = totals['fixed_capacity_query_key_pairs']-totals['active_query_key_pairs']
        return dict(mode=self.mode, lossy=False, prompt_length=self.prompt_length,
            prefix_length=self.prefix_length, recent_capacity=self.recent_capacity,
            tree_capacity=self.tree_capacity, physical_capacity=self.physical_capacity,
            reserved_physical_capacity=self.physical_capacity, next_position=self.next_position,
            active_tree_nodes=self.active_tree_nodes, write_pointer=self.write_pointer,
            recent_count=self.recent_count, valid_history_tokens=self.history_length,
            allocated_kv_storage_bytes=int(kv), owned_storage_bytes=int(kv),
            metadata_tensor_storage_bytes=0, metadata_update_write_bytes=0,
            payload_copy_totals=dict(self.payload_copy_totals),
            operation_counts=dict(self.operation_counts),
            positions_are_sequence_offsets=True, rotated_keys_modified=False,
            layout=VARIANT, active_attention_tokens=self.attention_length(),
            scratch_start=self.scratch_start, attention_shape_totals=totals,
            active_view_payload_copy_bytes=0, active_views_may_be_noncontiguous=True,
            backing_capacity_released_by_narrow=False,
            validity_representation='implicit_contiguous_history_and_shared_tree_plan',
            traffic_scope='K/V payload copies and initialization; attention reads excluded')


class ImplicitCacheState(ActiveRingCacheState):
    """Same scoped lifecycle; maps and mask rebuilding are absent by design."""
    commit_backend = 'triton'

    def prefill(self, target, values, counts, **kwargs):
        if target is not self.target:
            return self.original_prefill(target, values, counts, **kwargs)
        if self.prefill_calls:
            raise RuntimeError('implicit policy requires one prefill per context')
        self.prefill_calls += 1
        self.device = values['input_ids'].device
        out, delta, features = self.original_prefill(target, values, counts, **kwargs)
        self._close_phase('prefill')
        self.current_phase = 'conversion'
        self.cache = out.past_key_values
        self.prompt = int(values['input_ids'].shape[1])
        self.delta = int(delta)
        if self.original_length(self.cache) != self.prompt:
            raise RuntimeError('prefill cache must contain the entire prompt')
        self.bytes_per_token = 0
        for layer in self.cache.layers:
            k, v = layer.keys, layer.values
            if (k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1 or
                    k.shape[-2] != self.prompt or k.dtype != v.dtype or
                    k.device != v.device or k.device != self.device):
                raise RuntimeError('implicit runtime requires uniform batch-one full-prefill KV')
            self.bytes_per_token += 2*k.shape[1]*k.shape[-1]*k.element_size()
        del k, v
        for index, layer in enumerate(self.cache.layers):
            store = ImplicitKVLayer.from_prefill(layer.keys, layer.values, mode=self.mode,
                recent_capacity=self.recent_capacity, tree_capacity=self.tree_capacity)
            self.stores.append(store)
            saved = {name: layer.__dict__.get(name, fixed._MISSING)
                     for name in ('update', 'get_seq_length', 'get_mask_sizes')}
            self.layer_restore.append((layer, saved))
            layer.keys, layer.values = store.keys, store.values

            def update(instance, keys, values, *args, _index=index, **kwargs):
                return self._update(_index, keys, values, args, kwargs)

            def length(instance, _index=index):
                item = self.stores[_index]
                return item.next_position+item.active_tree_nodes

            def mask_sizes(instance, query_length, _index=index):
                if isinstance(query_length, bool):
                    raise ValueError('query length must be a positive integer')
                try:
                    query = operator.index(query_length)
                except TypeError as exc:
                    raise ValueError('query length must be a positive integer') from exc
                item = self.stores[_index]
                if query < 1 or query > item.tree_capacity:
                    raise ValueError('query length exceeds tree capacity')
                if item.active_tree_nodes and query != item.active_tree_nodes:
                    raise RuntimeError('query length differs from pending tree')
                return item.history_length+query, 0

            layer.update = MethodType(update, layer)
            layer.get_seq_length = MethodType(length, layer)
            layer.get_mask_sizes = MethodType(mask_sizes, layer)
        self.prefill_metadata = self._cache_metadata()
        self._close_phase('conversion')
        self.current_phase = 'decode'
        return out, delta, features

    def _layout_signature(self, store):
        return (store.mode, store.prefix_length, store.recent_count,
                store.recent_capacity, store.scratch_start, store.tree_capacity,
                store.next_position, store.active_tree_nodes, store.write_pointer)

    def _check_layout(self):
        first = self.stores[0]
        if (first.mode != 'protected_prompt' or first.next_position != first.history_length
                or first.scratch_start != first.history_length or
                not 0 <= first.recent_count <= first.recent_capacity):
            raise RuntimeError('implicit cache is not a contiguous no-eviction layout')
        signature = self._layout_signature(first)
        if any(self._layout_signature(s) != signature for s in self.stores):
            raise RuntimeError('implicit layers disagree on physical attention layout')

    @torch.inference_mode()
    def forward(self, *args, **kwargs):
        cache = kwargs.get('past_key_values')
        if cache is not self.cache or cache is None:
            return self.original_forward(*args, **kwargs)
        if self.failed:
            raise RuntimeError('implicit request previously failed')
        if self.pending is not None or self.in_forward:
            raise RuntimeError('previous tree must be committed before another forward')
        try:
            self._check_layout()
            if args:
                raise RuntimeError('verification requires explicit keyword arguments')
            mask, pos, ids = (kwargs.get(k) for k in ('attention_mask', 'position_ids', 'input_ids'))
            base = self.length(cache)
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or
                    not isinstance(mask, torch.Tensor) or mask.ndim != 4 or
                    not torch.is_floating_point(mask) or not isinstance(pos, torch.Tensor)):
                raise RuntimeError('batch-one tokens, additive 4D hard mask and positions required')
            nodes = int(ids.shape[1])
            if (nodes < 1 or nodes > self.tree_capacity or mask.shape != (1, 1, nodes, base+nodes)
                    or pos.shape != (3, 1, nodes) or pos.dtype != torch.long or
                    ids.device != self.device or mask.device != self.device or pos.device != self.device):
                raise RuntimeError('logical mask/tree/position shape or device mismatch')
            # Check only the tree square for bias; history must all be zero.
            # Both are round-level GPU reductions/host checks, kept in timing.
            tree_mask = mask[..., -nodes:]
            if (not bool((mask[..., :base] == 0).all()) or
                    not bool(((tree_mask == 0) | (tree_mask == torch.finfo(mask.dtype).min)).all())):
                raise RuntimeError('implicit layout requires strict hard mask {0, finfo.min}')
            plan = prepare_tree(tree_mask == 0)
            expected = (base+self.delta+plan.depth_tensor).view(1, 1, nodes).expand(3, 1, nodes)
            if not torch.equal(pos, expected):
                raise RuntimeError('explicit mRoPE positions disagree with logical base and tree depth')
            width = self.stores[0].history_length+nodes
            if width != mask.shape[-1]:
                raise RuntimeError('caller mask does not match active K/V width')
            self.pending = dict(logical_base=base, nodes=nodes, semantic_rope_base=base+self.delta,
                variant=VARIANT, active_attention_key_tokens=width,
                reserved_physical_capacity=self.stores[0].physical_capacity,
                excluded_padding_key_tokens=self.stores[0].physical_capacity-width,
                caller_attention_mask_reused=True, caller_position_ids_reused=True,
                mask_rematerialization_bytes=0)
            self.plan = plan
            self.update_counts = [0]*len(self.stores)
            self.in_forward = True
            # Preserve mask/position object, dtype, stride and values exactly.
            result = self.original_forward(**kwargs)
            if (any(x != 1 for x in self.update_counts) or
                    any(x.active_tree_nodes != nodes for x in self.stores)):
                raise RuntimeError('every Target layer must write exactly one complete tree')
            if self.length(cache) != base+nodes:
                raise RuntimeError('post-forward semantic tree length mismatch')
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.in_forward = False

    @torch.inference_mode()
    def compact(self, cache, selected, base, tree_nodes):
        if cache is not self.cache:
            return self.original_compact(cache, selected, base, tree_nodes)
        if self.pending is None or self.in_forward or self.failed:
            raise RuntimeError('implicit commit has no completed verification')
        if self.pending['logical_base'] != base or self.pending['nodes'] != tree_nodes:
            raise RuntimeError('implicit commit boundary differs from verified tree')
        before = self._cache_metadata()
        try:
            accepted = prepare_commit(self.plan, selected)
            self._check_layout()
            first = self.stores[0]
            if any(s._plan is not self.plan for s in self.stores):
                raise RuntimeError('implicit commit layers disagree on pending tree')
            # All capacity/layout/path checks finish before any layer is mutated.
            plan = prepare_delta_commit(accepted, mode=self.mode,
                prefix_length=first.prefix_length, recent_count=first.recent_count,
                recent_capacity=first.recent_capacity, scratch_start=first.scratch_start,
                tree_capacity=first.tree_capacity)
            count = accepted.accepted_tokens
            reports = []
            for store in self.stores:
                receipt = delta_commit_(store.keys, store.values, plan, backend=self.commit_backend)
                start, nodes = store.scratch_start, store.active_tree_nodes
                store.next_position += count
                store.recent_count += count
                store.write_pointer = store.recent_count % store.recent_capacity
                store.active_tree_nodes, store._plan = 0, None
                store.scratch_start = store.history_length
                reports.append(store._record('commit', receipt['copy_read_bytes'],
                    receipt['copy_write_bytes'], metadata_bytes=0,
                    accepted_tokens=count, stored_tokens=count, evicted_recent_tokens=0,
                    scratch_payload_peak_bytes=receipt['scratch_payload_peak_bytes'],
                    moved_tokens=receipt['moved_tokens'], payload_backend=receipt['backend'],
                    payload_kernel_launches=receipt['launches'],
                    previous_scratch_start=start, previous_scratch_nodes=nodes,
                    next_scratch_start=store.scratch_start, active_view_payload_copy_bytes=0))
            if self.length(cache) != base+count:
                raise RuntimeError('implicit commit semantic length mismatch')
            after = self._cache_metadata()
            b = self.bytes_per_token
            self.rounds.append({**self.pending, 'committed_tokens': count,
                'semantic_length_after': base+count, 'cache_before_commit': before,
                'cache_after_commit': after,
                'original_append_and_compact_logical_rw_bytes': 2*b*(2*base+tree_nodes+count),
                'original_history_copy_logical_rw_bytes': 4*b*base,
                'removed_history_copy_component_logical_rw_bytes': 4*b*base,
                'layer_commit_reports': reports[:1],
                'all_layer_commit_reports_equal': all(r == reports[0] for r in reports)})
            self.pending = self.plan = None
        except BaseException:
            self.failed = True
            raise

    def _cache_metadata(self):
        result = super()._cache_metadata()
        if result is not None:
            result['variant'] = VARIANT
        return result

    def summary(self):
        value = super().summary()
        value.update(variant=VARIANT, target_history_eviction=False,
            commit_method='skip_inplace_rows_and_fuse_moved_rows',
            attention_backend_changed=False, kv_quantized=False,
            metadata_tensor_storage_bytes=0, metadata_update_write_bytes=0,
            metadata_zero_scope='per-layer validity/slot-position maps; shared tree-plan tensors retained',
            caller_attention_mask_reused=True, mask_rematerialization_bytes=0,
            mask_reuse_scope='full additive mask; validation boolean temporaries and tree plan retained',
            per_layer_metadata_gpu_operations=0,
            round_tree_path_mrope_validation_retained=True,
            validation_claim='Candidate; GPU bitwise outputs and throughput unverified')
        return value


@contextmanager
def implicit_cache_policy(target, mode='protected_prompt', *, recent_capacity=512,
                          tree_capacity=30, commit_backend='triton'):
    """One request, strict overflow error, explicit CPU oracle for CPU tests."""
    if mode != 'protected_prompt':
        raise ValueError('implicit layout only supports protected_prompt without eviction')
    if commit_backend not in ('triton', 'cpu_reference'):
        raise ValueError('commit_backend must be triton or cpu_reference')
    if type(tree_capacity) is not int or not 1 <= tree_capacity <= 30:
        raise ValueError('tree capacity must be an integer in [1, 30]')
    if fixed._ACTIVE is not None:
        raise RuntimeError('nested or concurrent cache policies are unsupported')
    original = fixed.RingCacheState
    state = None
    try:
        with ExitStack() as stack:
            try:
                fixed.RingCacheState = ImplicitCacheState
                state = stack.enter_context(fixed.ring_cache_policy(target, mode,
                    recent_capacity=recent_capacity, tree_capacity=tree_capacity))
                state.commit_backend = commit_backend
            finally:
                fixed.RingCacheState = original
            yield state
    finally:
        fixed.RingCacheState = original
        if state is not None and fixed._ACTIVE is state:
            fixed._ACTIVE = None
