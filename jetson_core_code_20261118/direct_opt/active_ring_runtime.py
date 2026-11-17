"""Scoped active-width Target KV candidate, reusing the audited ring lifecycle.

Native prefill, absolute mRoPE, verification/commit guards, counters and phase
peaks retain the existing runtime logic. Only conversion and mask-size hooks
change. This module launches no model and leaves fixed-width source untouched.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import operator
from types import MethodType

import ring_cache_runtime as fixed
from active_ring_kv_cache import ActiveRingKVLayer

VARIANT = 'active_width'


class ActiveRingCacheState(fixed.RingCacheState):
    """An independent state class with active views and distinct provenance."""

    def prefill(self, target, values, counts, **kwargs):
        # Same native-prefill/conversion boundaries and ownership contract as
        # fixed.RingCacheState.prefill; no parent primitive is monkey-patched.
        if target is not self.target:
            return self.original_prefill(target, values, counts, **kwargs)
        if self.prefill_calls:
            raise RuntimeError('ring policy requires one prefill per context')
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
            if k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1 or k.shape[-2] != self.prompt:
                raise RuntimeError('ring runtime requires batch-one uniform full-prefill KV')
            if k.dtype != v.dtype or k.device != v.device or k.device != self.device:
                raise RuntimeError('prefill KV dtype/device mismatch')
            self.bytes_per_token += 2*k.shape[1]*k.shape[-1]*k.element_size()
        del k, v
        for index, layer in enumerate(self.cache.layers):
            store = ActiveRingKVLayer.from_prefill(layer.keys, layer.values, mode=self.mode,
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
                return item.next_position + item.active_tree_nodes

            def mask_sizes(instance, query_length, _index=index):
                if isinstance(query_length, bool):
                    raise ValueError('active ring query length must be a positive integer')
                try:
                    query = operator.index(query_length)
                except TypeError as exc:
                    raise ValueError('active ring query length must be a positive integer') from exc
                item = self.stores[_index]
                if query < 1 or query > item.tree_capacity:
                    raise ValueError('active ring query length exceeds tree capacity')
                if item.active_tree_nodes and query != item.active_tree_nodes:
                    raise RuntimeError('mask size query differs from pending tree')
                return item.history_length + query, 0

            layer.update = MethodType(update, layer)
            layer.get_seq_length = MethodType(length, layer)
            layer.get_mask_sizes = MethodType(mask_sizes, layer)
        self.prefill_metadata = self._cache_metadata()
        self._close_phase('conversion')
        self.current_phase = 'decode'
        return out, delta, features

    def _update(self, index, keys, values, args, kwargs):
        output = super()._update(index, keys, values, args, kwargs)
        expected = self.stores[index].attention_length(self.plan)
        if output[0].shape[-2] != expected or output[1].shape[-2] != expected:
            raise RuntimeError('active ring returned an incorrect attention width')
        return output

    def forward(self, *args, **kwargs):
        is_own = self.cache is not None and kwargs.get('past_key_values') is self.cache
        if is_own:
            history = {s.history_length for s in self.stores}
            capacity = {s.physical_capacity for s in self.stores}
            if len(history) != 1 or len(capacity) != 1:
                raise RuntimeError('active ring layers disagree on physical attention layout')
        result = super().forward(*args, **kwargs)
        if is_own:
            # The parent commits this metadata into its per-round trace.
            width = self.stores[0].attention_length()
            self.pending.update(variant=VARIANT, active_attention_key_tokens=width,
                                reserved_physical_capacity=self.stores[0].physical_capacity,
                                excluded_padding_key_tokens=self.stores[0].physical_capacity-width)
        return result

    def _cache_metadata(self):
        result = super()._cache_metadata()
        if result is not None:
            result.update(variant=VARIANT,
                          active_attention_key_tokens=self.stores[0].attention_length() if self.stores else None,
                          reserved_physical_capacity=self.stores[0].physical_capacity if self.stores else None)
        return result

    def summary(self):
        result = super().summary()
        result.update(variant=VARIANT, attention_layout='filled_history_plus_current_tree',
                      backing_allocation='fixed_capacity_unchanged',
                      attention_width_rule='prefix_length + recent_count + current_tree_nodes',
                      zero_history_packing=True,
                      scope='Target KV only; active-width views; native full prefill; logical bytes are not measured DRAM')
        return result


@contextmanager
def active_ring_cache_policy(target, mode='protected_prompt', recent_capacity=128, tree_capacity=30):
    """Use the original policy's guards, cleanup and peak boundaries.

    The class factory is temporarily replaced ONLY during context entry; parent
    state construction captures all existing profiler patches. The class is
    restored before user/model code executes. The shared fixed._ACTIVE guard
    rejects nesting with either fixed or active policies.
    """
    if mode not in ('protected_prompt', 'pure_recent'):
        raise ValueError('active-width candidate requires protected_prompt or pure_recent')
    if fixed._ACTIVE is not None:
        raise RuntimeError('nested or concurrent ring policies are unsupported')
    original = fixed.RingCacheState
    state = None
    try:
        with ExitStack() as stack:
            try:
                fixed.RingCacheState = ActiveRingCacheState
                state = stack.enter_context(fixed.ring_cache_policy(target, mode,
                    recent_capacity=recent_capacity, tree_capacity=tree_capacity))
            finally:
                fixed.RingCacheState = original
            yield state
    finally:
        fixed.RingCacheState = original
        # The parent's release normally clears this. Also avoid retaining a
        # poisoned context if release itself raises after method restoration.
        if state is not None and fixed._ACTIVE is state:
            fixed._ACTIVE = None
