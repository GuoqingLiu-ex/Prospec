"""Scoped Target tree policy using the independent output-only grouped math.

Prefill, cache ownership, verification lifecycle and original mask/positions
reuse the frozen grouped policy. No frozen module globals or cache functions
are patched. Removed-copy totals are logical payload, not GPU/DRAM evidence.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import torch

import grouped_gqa_runtime as grouped
from grouped_gqa_no_weights_math import grouped_gqa_no_weights_math
from stride0_gqa_runtime import math_precision_scope, precision_flags

METHOD = 'grouped_no_discarded_weights'
_PRIMITIVE_METHOD = 'grouped_two_bmm_no_discarded_weights'
_MISSING = object()
_MARKERS = ('_stride0_tree_policy', '_strict_gqa_policy', '_grouped_tree_policy',
            '_grouped_no_weights_tree_policy')
_TREE_KWARGS = frozenset(('position_ids', 'sliding_window', 'output_attentions', 'use_cache'))


def _scalar_tree(value):
    """First-receipt-only guard; no tensor reads or retained tensor values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) and _scalar_tree(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_scalar_tree(v) for v in value)
    return False


class GroupedNoWeightsAttentionPolicy(grouped.GroupedAttentionPolicy):
    def __init__(self, target, bridge, previous_attention):
        super().__init__(target, bridge, previous_attention)
        self.mode = METHOD
        self._discarded_probability_elements = 0
        self.record.update(mode=METHOD, method=METHOD, implementation=METHOD,
            primitive_implementation=_PRIMITIVE_METHOD,
            discarded_probability_cast_removed=True,
            actual_kernel_count_measured=False,
            ordinary_GPU_output_equivalence_verified=False,
            extra_options_contract='Only inert position_ids=None, sliding_window=None, '
                'output_attentions=False, HF decoder use_cache=True; explicit causal mode unsupported')

    def attention(self, module, query, key, value, attention_mask, dropout=0.0,
                  scaling=None, is_causal=None, position_bias=None, **kwargs):
        index = self.indices.get(id(module))
        if index is None or self.in_prefill:
            return super().attention(module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, is_causal=is_causal,
                position_bias=position_bias, **kwargs)
        self.record['selected_calls'] += 1
        try:
            pending = self.pending
            if self.failed or pending is None:
                raise RuntimeError('Target attention outside recognized prefill/tree forward')
            n, t = pending['base'] + pending['nodes'], pending['nodes']
            operands = (query, key, value, attention_mask)
            if not all(isinstance(x, torch.Tensor) for x in operands):
                raise ValueError('tree operands must be tensors')
            unknown_options = set(kwargs) - _TREE_KWARGS
            if unknown_options:
                raise ValueError('unsupported tree attention options: ' + ', '.join(sorted(unknown_options)))
            if (query.shape != (1, 28, t, 128) or key.shape != (1, 4, n, 128)
                    or value.shape != key.shape
                    or any(x.dtype != torch.float16 for x in (query, key, value))
                    or attention_mask.shape != (1, 1, t, n)
                    or attention_mask.dtype != torch.float32
                    or any(x.device != query.device for x in operands)
                    or attention_mask is not pending['mask']
                    or any(x.requires_grad or x.stride(-1) != 1 for x in operands)
                    or pending['layer_calls'][index] or dropout != 0.0
                    or position_bias is not None
                    or (is_causal is not None and is_causal is not False)
                    or kwargs.get('position_ids') is not None
                    or kwargs.get('sliding_window') is not None
                    or kwargs.get('output_attentions', False) is not False
                    or ('use_cache' in kwargs and kwargs['use_cache'] is not True)):
                raise ValueError('tree operands, original mask or supported attention options changed')
            query_contiguous = query.is_contiguous()
            if not query_contiguous and query.stride()[-3:] != (128, 28 * 128, 1):
                raise ValueError('unsupported query layout for exact lightweight Q-pack accounting')
            first = self.record['representative_receipt'] is None
            output, receipt, debug = grouped_gqa_no_weights_math(
                query, key, value, attention_mask, scaling,
                record_receipt=first, debug_intermediates=False)
            if debug is not None:
                raise RuntimeError('debug tensors are forbidden in full requests')
            if first:
                self._retain_first_receipt(receipt, query, t, n, query_contiguous)
            elif receipt is not None:
                raise RuntimeError('ordinary primitive path unexpectedly constructed a receipt')
            if (not isinstance(output, torch.Tensor) or output.shape != (1, t, 28, 128)
                    or output.dtype != torch.float16 or output.device != query.device
                    or not output.is_contiguous()):
                raise RuntimeError('tree attention output layout/dtype/device differs from HF contract')
            self._kv_elements += 4 * n * 128
            if not query_contiguous:
                self._query_pack_bytes += 4 * 28 * t * 128
                self._query_pack_calls += 1
            self._discarded_probability_elements += 28 * t * n
            pending['layer_calls'][index] += 1
            self.record['layer_calls'][index] += 1
            self.record['completed_calls'] += 1
            self.record['candidate_calls'] += 1
            return output, None
        except BaseException as exc:
            self._fail('attention', exc)
            raise

    def _retain_first_receipt(self, receipt, query, t, n, contiguous):
        if not isinstance(receipt, dict) or not _scalar_tree(receipt):
            raise RuntimeError('first primitive receipt must contain only scalar metadata')
        description = receipt.get('input', {}).get('query', {})
        removed = receipt.get('removed_discarded_probability_copy', {})
        elements = 28 * t * n
        if (receipt.get('implementation') != _PRIMITIVE_METHOD
                or receipt.get('debug_intermediates') is not False
                or receipt.get('discarded_probability_materialization') is not False
                or receipt.get('native_unused_probability_cast_preserved') is not False
                or receipt.get('original_mask_object') is not True
                or receipt.get('mask_values_transformed') is not False
                or description.get('shape') != list(query.shape)
                or description.get('stride') != list(query.stride())
                or description.get('dtype') != str(query.dtype)
                or description.get('contiguous') is not contiguous
                or removed.get('elements') != elements
                or removed.get('read_bytes') != 4 * elements
                or removed.get('write_bytes') != 2 * elements
                or removed.get('read_plus_write_bytes') != 6 * elements):
            raise RuntimeError('first no-weights receipt disagrees with actual operands/method')
        # This field belongs to the policy; the frozen primitive has no Q-pack
        # receipt. The two supported layouts make the preserve-format cast and
        # subsequent contiguous() allocation predictable without host reads.
        stored = dict(receipt, query_pack_copy=not contiguous,
            query_layout_checked_by_policy=True, policy_method=METHOD)
        self.record['representative_receipt'] = stored
        self.record['primitive_receipt_calls'] += 1

    def validate_result(self, result):
        rounds = [row for row in result.rounds if row.get('kind') == 'tree']
        expected_elements = 28 * 28 * sum(row['tree_nodes'] *
            (row['target_cache_before'] + row['tree_nodes']) for row in rounds)
        if (self.record['method'] != METHOD or self.record['mode'] != METHOD
                or self._discarded_probability_elements != expected_elements):
            self.record['result_coverage_validated'] = False
            raise RuntimeError('no-weights copy accounting differs from actual verification rounds')
        super().validate_result(result)

    def summary(self):
        result = super().summary()
        elements = self._discarded_probability_elements
        result['logical_discarded_probability_copy_totals'] = dict(
            elements=elements, read_bytes=4 * elements, write_bytes=2 * elements,
            read_plus_write_bytes=6 * elements,
            counterfactual_software_cast_calls=self.record['completed_calls'],
            comparison='removed versus frozen grouped at the same actual attention shapes',
            scope='Logical FP32 read and FP16 write payload only; separate from '
                  'the inherited KV-only minus Q-pack table, not actual GPU kernels or DRAM',
            actual_kernel_count_measured=False, actual_dram_measured=False,
            throughput_measured=False, attention_major_mac_reduction=0,
            kv_resident_capacity_reduction=0)
        return result


def _restore_forward(target, previous):
    if previous is _MISSING:
        if 'forward' in target.__dict__:
            delattr(target, 'forward')
    else:
        target.forward = previous


def _restore_registry(interface, had_local, previous):
    if had_local:
        interface['sdpa'] = previous
    elif 'sdpa' in interface._local_mapping:
        del interface['sdpa']


def _release(policy):
    policy.cache = None
    policy.pending = None
    policy.prefill_ids = None
    policy.in_prefill = False
    policy.record['finalized'] = True


@contextmanager
def grouped_no_weights_attention_policy(target, mode=METHOD, *, interface=None, bridge=None):
    if mode != METHOD:
        raise ValueError('choose grouped_no_discarded_weights; controls use their frozen policies')
    if interface is None:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
        interface = ALL_ATTENTION_FUNCTIONS
    if bridge is None:
        import hf517_vispec_adapter as bridge
    if not hasattr(interface, '_local_mapping'):
        raise TypeError('HF local attention registry required')
    previous = interface['sdpa']
    if any(getattr(previous, marker, False) for marker in _MARKERS):
        raise RuntimeError('nested tree attention policies unsupported')
    policy = GroupedNoWeightsAttentionPolicy(target, bridge, previous)
    had_local = 'sdpa' in interface._local_mapping
    local = interface._local_mapping.get('sdpa')
    old_forward = target.__dict__.get('forward', _MISSING)

    def attention(*args, **kwargs):
        return policy.attention(*args, **kwargs)

    for marker in _MARKERS:
        setattr(attention, marker, True)
    # Register all restorations before installing any hook, so a partial
    # installation failure also unwinds and one restoration error cannot skip
    # the remaining callbacks.
    with ExitStack() as stack:
        stack.callback(_release, policy)
        stack.callback(_restore_registry, interface, had_local, local)
        stack.callback(_restore_forward, target, old_forward)
        stack.callback(setattr, bridge, '_prefill', policy.original_prefill)
        bridge._prefill = policy.prefill
        target.forward = policy.forward
        interface['sdpa'] = attention
        yield policy
