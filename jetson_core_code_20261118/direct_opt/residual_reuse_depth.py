"""Bounded development diagnostic: clamp the old gate to one skipped block.

The reuse arm replaces skipped block 28 with x27 + (x27 - x26), coefficient 1.
It uses only the already executed block 27. No future block, fitted coefficient,
new score, or threshold is introduced. Block 29 always executes normally.

The inspected Transformers 5.17 Qwen2_5_VLVisionBlock.forward uses out-of-place
residual additions (server source SHA256 507cb2c383ead0e0e22b16ec9c8af2eb82703843
cd5849112715751d990ff15e, lines 366 and 372). Reuse therefore retains the prefix
input reference, not a full clone. This assumption does not support other blocks
that mutate their input. No tensor-value check or extra device sync is added.
"""
from __future__ import annotations

from types import MethodType

import torch

try:
    from .depth import DirectDepthPolicy
except ImportError:
    from depth import DirectDepthPolicy


class ResidualReuseDepthPolicy(DirectDepthPolicy):
    def __init__(self, target, thresholds, *, residual_mode="identity", anchor_stride=4):
        if residual_mode not in {"identity", "reuse"}:
            raise ValueError("residual_mode must be identity or reuse")
        if anchor_stride != 4:
            raise ValueError("This diagnostic retains the old stride-4 anchors")
        super().__init__(target, thresholds=thresholds, anchor_stride=4,
                         protect_first=True, summary_backend="segment")
        self.residual_mode = residual_mode
        self._prefix_input = None

    def _start(self, hidden_states, grid_thw):
        self._prefix_input = None
        super()._start(hidden_states, grid_thw)
        self._current.update(
            policy="direct_one_prefix_residual_v1", residual_mode=self.residual_mode,
            residual_coefficient=1 if self.residual_mode == "reuse" else 0,
            original_gate_actions=None, selected_rows=0, selected_time_blocks=[], residual_elements=0,
            retained_prefix_logical_bytes=0, retained_prefix_storage_bytes=0,
            selected_prefix_materialized_bytes=0, delta_temporary_bytes=0,
            residual_elementwise_ops=0, residual_logical_io_bytes=0, residual_applied=False,
            prefix_input_contract="Inspected Qwen block27 does not mutate its input",
            residual_memory_scope="Logical live tensors and explicit temporaries; not measured DRAM traffic or incremental allocator peak",
            residual_cost_scope="Reference lifetime, selected-row gather, subtract/add and route copies are inside visual/request timing")

    def _actions_from_scores(self, scores):
        original = super()._actions_from_scores(scores)
        self._current["original_gate_actions"] = original
        return tuple(int(action > 0) for action in original)

    def _gate(self, before_mean, after):
        super()._gate(before_mean, after)
        selected = sum(length for length, action in zip(self._time_lengths, self._current["actions"])
                       if action == 1)
        self._current["selected_rows"] = selected
        self._current["selected_time_blocks"] = [i for i, a in enumerate(self._current["actions"]) if a]
        if self.residual_mode == "reuse":
            if self._prefix_input is None:
                raise RuntimeError("Missing actual block27 input")
            if (self._prefix_input.shape != after.shape or self._prefix_input.dtype != after.dtype
                    or self._prefix_input.device != after.device):
                raise ValueError("Prefix input/output shape, dtype and device must match")
            self._current["retained_prefix_logical_bytes"] = self._prefix_input.numel() * self._prefix_input.element_size()
            self._current["retained_prefix_storage_bytes"] = self._prefix_input.untyped_storage().nbytes()
            if not selected:
                self._prefix_input = None

    def _selected(self, layer, original, hidden_states, args, kwargs):
        if layer != self.skip_layers[0] or self.residual_mode == "identity":
            return super()._selected(layer, original, hidden_states, args, kwargs)
        route = self._route
        if route is None:
            raise RuntimeError("Prefix decision was not executed")
        selected_rows = self._current["selected_rows"]
        try:
            # Parent creates owned grouped storage for every mixed route. The
            # skip-1 rows are its tail; block29 later restores the native order.
            output = super()._selected(layer, original, hidden_states, args, kwargs)
            if not selected_rows:
                return output
            before = self._prefix_input
            if before is None:
                raise RuntimeError("Prefix input was released before residual reuse")
            if route["uniform"] == 1:
                # Not reachable with protected first tubelets, but preserve the
                # ownership contract if the router is tested independently.
                delta = hidden_states - before
                output = hidden_states + delta
                materialized = 0
            else:
                selected_times = [t for t in route["order"] if self._current["actions"][t] == 1]
                pieces = [before[self._time_offsets[t]:self._time_offsets[t + 1]] for t in selected_times]
                selected_before = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
                skip_rows = output[route["n0"]:]
                if skip_rows.shape != selected_before.shape or skip_rows.shape[0] != selected_rows:
                    raise RuntimeError("Selected prefix rows do not align with skip route")
                delta = skip_rows - selected_before
                skip_rows.add_(delta)
                materialized = 0 if len(pieces) == 1 else selected_before.numel() * selected_before.element_size()
            elements = delta.numel()
            self._current.update(residual_applied=True, residual_elements=elements,
                selected_prefix_materialized_bytes=materialized,
                delta_temporary_bytes=elements * delta.element_size(),
                residual_elementwise_ops=2 * elements,
                # Two pointwise ops each read two inputs and write one output;
                # materialized cat additionally reads/writes its selected rows.
                # Cache effects/kernel fusion make actual DRAM bytes different.
                residual_logical_io_bytes=6 * elements * delta.element_size() + 2 * materialized)
            return output
        finally:
            self._prefix_input = None

    def __enter__(self):
        super().__enter__()
        try:
            if self.residual_mode == "reuse":
                block = self.visual.blocks[self.prefix_layer]
                parent_prefix = block.forward
                policy = self

                def prefix_forward(module, hidden_states, *args, **kwargs):
                    if policy._current is None:
                        return parent_prefix(hidden_states, *args, **kwargs)
                    policy._prefix_input = hidden_states
                    try:
                        return parent_prefix(hidden_states, *args, **kwargs)
                    except BaseException:
                        policy._prefix_input = None
                        raise

                self._replace(block, "forward", MethodType(prefix_forward, block))
                parent_visual = self.visual.forward

                def visual_forward(module, *args, **kwargs):
                    try:
                        return parent_visual(*args, **kwargs)
                    finally:
                        policy._prefix_input = None

                self._replace(self.visual, "forward", MethodType(visual_forward, self.visual))
            return self
        except BaseException:
            super().__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self._prefix_input = None

    def summary(self):
        return dict(super().summary(), policy="direct_one_prefix_residual_v1",
                    residual_mode=self.residual_mode, trained=False,
                    quality_status="Development diagnostic; length and factual quality unvalidated",
                    thresholds_changed=False, gate_change="Clamp old action>0 to exactly one")
