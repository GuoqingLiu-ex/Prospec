"""All-token online-softmax adaptor: a Torch functional hardware reference.

The frozen query/K composition is reused. Heads are processed in groups; each
group scans input tiles and retains only max, denominator and pooled ORIGINAL
feature numerator. No full token K/V projection is built; scores are bounded by
the configured group and tile (which may cover a short full sequence). V/O use
the existing model-dtype weights after each pooled group is
rounded back to model dtype.

This is NOT an SRAM-resident implementation or a fused kernel. All weights and
the caller's complete X stay in ordinary Torch storage. Casts, BLAS workspace,
Python loops, launches, validation and optional storage observations are real
costs. FP32/FP64 scoring and online reduction change rounding versus deployed
FP16 softmax/BMM; neither bitwise output nor long-answer quality is promised.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compiled_adaptor import BUFFER_NAMES, CompiledQueryAdaptor, _storage_metadata, _storage_key


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name+' must be a positive integer')
    return value


def theoretical_pooling_sram_bytes(*, heads_per_group, token_tile, input_dim=3584,
                                   queries=2, projected_width=3584, reserve_bytes=32768):
    """Paper-plan bytes for batch=1, FP16 inputs/directions, FP32 state.

    Assumes in-place score/probability storage, double-buffered FP16 input and
    stage reuse. It is NOT the Torch reference's actual live storage. It excludes
    full V/O weights, whose tiled projection phases must be budgeted separately.
    """
    g = _positive(heads_per_group, 'heads_per_group')
    t = _positive(token_tile, 'token_tile')
    d = _positive(input_dim, 'input_dim')
    q = _positive(queries, 'queries')
    width = _positive(projected_width, 'projected_width')
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError('reserve_bytes must be a nonnegative integer')
    terms = dict(raw_accumulator_fp32=4*g*q*d, folded_directions_fp16=2*g*q*d,
                 double_input_blocks_fp16=4*t*d, inplace_score_probability_fp32=4*g*q*t,
                 softmax_max_and_sum_fp32=8*g*q, score_bias_fp16=2*g*q,
                 all_projected_values_fp16=2*q*width, planning_reserve=reserve_bytes)
    return dict(terms_bytes=terms, total_bytes=sum(terms.values()), actual_sram_measured=False,
                assumptions='batch1, FP16 X/directions/bias, FP32 online state; fused staged schedule, no spills')


class _Workspace:
    """Count explicit live tensor storages; never retain tensors in the receipt."""
    def __init__(self, persistent, x, enabled):
        self.enabled = enabled
        self.excluded = {_storage_key(t) for t in persistent if t is not None} | {_storage_key(x)}
        self.peak = 0
        self.observations = 0
        self.phases = {}

    def observe(self, phase, **tensors):
        if not self.enabled:
            return
        info = _storage_metadata([(name, t) for name, t in tensors.items() if t is not None])
        storages = [s for s in info['storages'] if (s['device'], s['data_ptr']) not in self.excluded]
        count = sum(s['storage_bytes'] for s in storages)
        self.peak = max(self.peak, count)
        self.observations += 1
        if phase not in self.phases or count > self.phases[phase]['unique_storage_bytes']:
            self.phases[phase] = dict(unique_storage_bytes=count, storages=storages)

    def receipt(self):
        return dict(enabled=self.enabled, observations=self.observations,
            peak_explicit_live_workspace_storage_bytes=self.peak if self.enabled else None,
            peak_by_phase=self.phases,
            scope='Named live tensor snapshots; persistent buffers and caller X backing excluded. '
                  'Includes returned output while live; excludes Python objects, transient operator/BLAS '
                  'workspace and validation temporaries. NOT allocator peak or SRAM residency.')


class StreamingQueryAdaptor(nn.Module):
    """Use from_compiled; frozen aliases, no original module or Q/K retained."""
    def __init__(self):
        super().__init__()
        self._ready = False
        self.last_workspace_receipt = None

    @classmethod
    @torch.no_grad()
    def from_compiled(cls, compiled: CompiledQueryAdaptor, *, heads_per_group=7,
                      token_tile=12, accumulate_dtype=torch.float32, check_finite=True,
                      record_workspace=True):
        if not isinstance(compiled, CompiledQueryAdaptor) or not compiled._ready:
            raise TypeError('expected a ready CompiledQueryAdaptor')
        g, t = _positive(heads_per_group, 'heads_per_group'), _positive(token_tile, 'token_tile')
        if g > compiled.heads:
            raise ValueError('heads_per_group cannot exceed the number of heads')
        if accumulate_dtype not in (torch.float32, torch.float64):
            raise ValueError('accumulate_dtype must be torch.float32 or torch.float64')
        if type(check_finite) is not bool or type(record_workspace) is not bool:
            raise TypeError('check_finite and record_workspace must be bool')
        if compiled.compute_dtype != compiled._compiled_dtype:
            raise ValueError('compiled source was cast after composition')
        started = time.perf_counter()
        self = cls()
        for name in BUFFER_NAMES:
            value = getattr(compiled, name)
            self.register_buffer(name, None if value is None else value.detach(), persistent=True)
        for name in ('q_count','heads','head_dim','input_dim','output_dim','hq'):
            setattr(self, name, int(getattr(compiled, name)))
        self.hidden_size, self.num_q, self.num_heads = self.input_dim, self.q_count, self.heads
        self._compiled_dtype = compiled.compute_dtype
        self.heads_per_group, self.token_tile = g, t
        self.accumulate_dtype = accumulate_dtype
        self.check_finite, self.record_workspace = check_finite, record_workspace
        self._ready = True
        self.prepare_host_wall_ms = 1000*(time.perf_counter()-started)
        return self.eval()

    @property
    def input_dtype(self):
        return self.directions.dtype

    @property
    def input_device(self):
        return self.directions.device

    def metadata(self) -> dict[str, Any]:
        if not self._ready:
            raise RuntimeError('construct via StreamingQueryAdaptor.from_compiled')
        return dict(implementation='torch_online_softmax_pooled_original_features',
            heads_per_group=self.heads_per_group, token_tile=self.token_tile,
            score_and_accumulator_dtype=str(self.accumulate_dtype),
            pooled_rounding_and_projection_dtype=str(self._compiled_dtype),
            buffers=_storage_metadata([(name,getattr(self,name)) for name in BUFFER_NAMES]),
            persistent_request_workspace_bytes=0, source_module_retained=False,
            buffer_ownership='aliases frozen compiled storage; source mutations are shared',
            prepare_host_wall_ms=self.prepare_host_wall_ms,
            prepare_scope='buffer alias registration only; no device synchronization or composition',
            check_finite=self.check_finite, finite_check_scope='input once and output once; GPU reductions/host sync counted in forward',
            record_workspace=self.record_workspace,
            full_token_kv_materialized=False, score_storage_bounded_by_group_and_tile=True,
            sram_residency_proven=False, numerical_equivalence='real-arithmetic identity; changed reduction/rounding, no bitwise promise')

    def preparation_metadata(self):
        return self.metadata()

    def _check(self, x, group_sizes):
        if not self._ready:
            raise RuntimeError('construct via StreamingQueryAdaptor.from_compiled')
        if group_sizes is not None:
            raise ValueError('group_sizes is unsupported: this reference preserves every input token')
        if self.input_dtype != self._compiled_dtype or any(
            t.dtype != self._compiled_dtype or t.device != self.input_device for t in self.buffers()):
            raise RuntimeError('do not cast a composed streaming module; construct from the final model dtype')
        if (not isinstance(x, Tensor) or x.ndim != 3 or min(x.shape) < 1 or
                x.shape[-1] != self.input_dim or x.dtype != self.input_dtype or x.device != self.input_device):
            raise ValueError('expected nonempty [B,N,D] matching compiled dtype/device')
        if self.check_finite and not bool(torch.isfinite(x).all()):
            raise ValueError('input contains nonfinite values')

    @torch.no_grad()
    def forward(self, x: Tensor, group_sizes=None):
        self.last_workspace_receipt = None
        self._check(x, group_sizes)
        batch, length, width = x.shape
        q, h, d = self.q_count, self.heads, self.head_dim
        tracker = _Workspace(list(self.buffers()), x, self.record_workspace)
        groups, tiles = math.ceil(h/self.heads_per_group), math.ceil(length/self.token_tile)
        receipt = dict(completed=False, input_shape=list(x.shape), head_groups=groups,
            tiles_per_group=tiles, input_scans=groups, tokens_visited_per_batch=groups*length,
            max_score_tile_shape=[batch,min(h,self.heads_per_group)*q,min(length,self.token_tile)],
            max_accumulator_shape=[batch,min(h,self.heads_per_group)*q,width],
            logical_input_elements_visited=x.numel()*groups,
            logical_input_source_read_bytes=x.numel()*x.element_size()*groups,
            logical_reads_are_measured_dram=False,
            persistent_buffers=_storage_metadata([(name,getattr(self,name)) for name in BUFFER_NAMES]),
            caller_input_storage=_storage_metadata([('x',x)]),
            theoretical_pooling_sram=theoretical_pooling_sram_bytes(
                heads_per_group=self.heads_per_group, token_tile=self.token_tile,
                input_dim=width, queries=q, projected_width=h*d),
            theoretical_budget_matches_numeric_assumptions=(batch == 1 and x.dtype == torch.float16
                                                           and self.accumulate_dtype == torch.float32))
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                projected = x.new_empty((batch,q,h,d))
                for first in range(0,h,self.heads_per_group):
                    end = min(h,first+self.heads_per_group); count = (end-first)*q
                    directions = self.directions[first*q:end*q].to(self.accumulate_dtype)
                    bias = None if self.score_bias is None else self.score_bias[first*q:end*q].to(self.accumulate_dtype)
                    accum = torch.zeros((batch,count,width),dtype=self.accumulate_dtype,device=x.device)
                    maximum = torch.full((batch,count,1),-float('inf'),dtype=self.accumulate_dtype,device=x.device)
                    normalizer = torch.zeros_like(maximum)
                    for offset in range(0,length,self.token_tile):
                        tile = x[:,offset:offset+self.token_tile,:].to(self.accumulate_dtype)
                        scores = F.linear(tile,directions,bias).transpose(1,2)
                        block_max = scores.amax(-1,keepdim=True)
                        next_max = torch.maximum(maximum,block_max)
                        alpha = (maximum-next_max).exp()
                        scores.sub_(next_max).exp_()
                        block_sum = scores.sum(-1,keepdim=True)
                        tracker.observe('online_tile',projected=projected,directions=directions,bias=bias,
                            accum=accum,maximum=maximum,normalizer=normalizer,tile=tile,
                            scores_probabilities=scores,block_max=block_max,next_max=next_max,alpha=alpha,block_sum=block_sum)
                        accum.mul_(alpha).baddbmm_(scores,tile)
                        normalizer.mul_(alpha).add_(block_sum)
                        maximum = next_max
                        del tile,scores,block_max,next_max,alpha,block_sum
                    accum.div_(normalizer)
                    # Free pooling-only tensors before projecting this group.
                    del directions,bias,maximum,normalizer
                    per_head_raw = accum.reshape(batch,end-first,q,width).permute(1,0,2,3).reshape(end-first,batch*q,width)
                    per_head = per_head_raw.to(self._compiled_dtype)
                    values = torch.bmm(per_head,self.value_heads[first:end].transpose(1,2))
                    if self.value_bias is not None:
                        values.add_(self.value_bias[first:end])
                    tracker.observe('value_projection',projected=projected,accum=accum,
                                    per_head_layout=per_head_raw,per_head_model=per_head,values=values)
                    projected[:,:,first:end,:].copy_(values.reshape(end-first,batch,q,d).permute(1,2,0,3))
                    del accum,per_head_raw,per_head,values
                output = F.linear(projected.reshape(batch,q,h*d),self.output_weight,self.output_bias)
                tracker.observe('output_projection',projected=projected,output=output)
                if self.check_finite and not bool(torch.isfinite(output).all()):
                    raise ValueError('online adaptor produced nonfinite output')
                receipt.update(completed=True,output_shape=list(output.shape),
                               output_storage=_storage_metadata([('output',output)]))
                return output
        finally:
            receipt['workspace'] = tracker.receipt()
            self.last_workspace_receipt = receipt
