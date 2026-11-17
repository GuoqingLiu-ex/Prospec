"""Exact, bounded KV payload commit; no model/runtime patch and no eager CUDA.

Only batch-one, independent contiguous FP16 K/V banks, D=128, trees <=30,
and protected-prompt recent storage that has not overflowed are supported.
Prepare once per round from the existing validated RingCommitPlan, then reuse
the result across layers. Metadata validity/positions remain the caller's job.
Logical byte counts are not measured DRAM. This is a baseline optimization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import operator


@dataclass(frozen=True, init=False, eq=False)
class DeltaCommitPlan:
    commit_plan: object = field(repr=False)
    prefix_length: int
    recent_count: int
    recent_capacity: int
    scratch_start: int
    tree_capacity: int
    tree_nodes: int
    destination_start: int
    positions: tuple
    moved_positions: tuple

    def __init__(self, *args, **kwargs):
        raise TypeError('use prepare_delta_commit')

    @property
    def accepted_tokens(self):
        return len(self.positions)

    @property
    def moved_tokens(self):
        return len(self.moved_positions)


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise TypeError(name+' must be an integer')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(name+' must be an integer') from exc
    if value < minimum:
        raise ValueError(name+' is out of range')
    return value


def prepare_delta_commit(commit_plan, *, prefix_length, recent_count,
                         recent_capacity, scratch_start, tree_capacity,
                         mode='protected_prompt'):
    """Use the already-validated host tuple; no tensor reads or allocations.

    The underlying RingCommitPlan and its owned index tensor are immutable by
    contract. Reuse of a plan for a different round is forbidden by the runtime.
    No GPU->CPU validation is repeated here. Supports an active-width scratch or
    a fixed scratch after the recent capacity; both must avoid committed history.
    """
    from ring_kv_cache import RingCommitPlan
    if not isinstance(commit_plan, RingCommitPlan):
        raise TypeError('a validated RingCommitPlan is required')
    if mode != 'protected_prompt':
        raise ValueError('only protected_prompt is supported')
    fields = dict(prefix_length=_integer(prefix_length, 'prefix_length', 1),
                  recent_count=_integer(recent_count, 'recent_count'),
                  recent_capacity=_integer(recent_capacity, 'recent_capacity', 1),
                  scratch_start=_integer(scratch_start, 'scratch_start'),
                  tree_capacity=_integer(tree_capacity, 'tree_capacity', 1))
    positions = commit_plan.positions
    nodes = commit_plan.tree_plan.nodes
    if not 1 <= len(positions) <= nodes <= fields['tree_capacity'] <= 30:
        raise ValueError('nonempty accepted path and tree_capacity <=30 required')
    if fields['recent_count'] + len(positions) > fields['recent_capacity']:
        raise ValueError('commit would overflow recent storage; eviction/wrap forbidden')
    destination = fields['prefix_length'] + fields['recent_count']
    if fields['scratch_start'] not in (destination,
                                      fields['prefix_length']+fields['recent_capacity']):
        raise ValueError('scratch must be active-width or fixed-capacity boundary')
    moved = tuple(i for i, source in enumerate(positions)
                  if fields['scratch_start']+source != destination+i)
    result = object.__new__(DeltaCommitPlan)
    for name, value in dict(commit_plan=commit_plan, tree_nodes=nodes,
                            destination_start=destination, positions=positions,
                            moved_positions=moved, **fields).items():
        object.__setattr__(result, name, value)
    return result


def _validate(keys, values, plan):
    import torch
    if not isinstance(plan, DeltaCommitPlan):
        raise TypeError('a prepared DeltaCommitPlan is required')
    if (not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor)
            or keys.ndim != 4 or keys.shape != values.shape
            or keys.shape[0] != 1 or keys.shape[1] < 1 or keys.shape[-1] < 1):
        raise ValueError('matching K/V [1,H,capacity,D] with positive H/D required')
    if (keys.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64)
            or values.dtype != keys.dtype
            or keys.device != values.device or keys.device.type not in ('cpu','cuda')
            or keys.layout != torch.strided or values.layout != torch.strided
            or keys.requires_grad or values.requires_grad):
        raise ValueError('matching inference-only floating strided CPU/CUDA banks required')
    if keys.device.type=='cuda' and (keys.dtype!=torch.float16 or keys.shape[-1]!=128):
        raise ValueError('CUDA path is restricted to FP16 D128')
    # Require canonical non-overlapping rows and heads, including singleton H.
    _, heads, capacity, dim = keys.shape
    expected = (heads*capacity*dim, capacity*dim, dim, 1)
    if (tuple(keys.stride()) != expected or tuple(values.stride()) != expected
            or not keys.is_contiguous() or not values.is_contiguous()):
        raise ValueError('canonical contiguous independent rows/heads required')
    if keys.untyped_storage().data_ptr() == values.untyped_storage().data_ptr():
        raise ValueError('keys and values must have independent backing storage')
    if (plan.scratch_start+plan.tree_nodes > capacity
            or plan.destination_start+plan.accepted_tokens > capacity):
        raise ValueError('source/destination exceeds bank capacity')
    indices = plan.commit_plan._indices
    if (indices.device != keys.device or indices.dtype != torch.long
            or indices.ndim != 1 or indices.numel() != plan.accepted_tokens
            or not indices.is_contiguous()):
        raise ValueError('prepared indices must match bank device and accepted count')
    return heads


def _integer_view(tensor):
    import torch
    dtype={2:torch.int16,4:torch.int32,8:torch.int64}[tensor.element_size()]
    return tensor.view(dtype)


def _receipt(keys, plan, backend, launches, *, reference=False):
    payload_per_row = 2*keys.shape[0]*keys.shape[1]*keys.shape[-1]*keys.element_size()
    moved_payload = payload_per_row*plan.moved_tokens
    accepted_payload = payload_per_row*plan.accepted_tokens
    return dict(backend=backend, accepted_tokens=plan.accepted_tokens,
                moved_tokens=plan.moved_tokens,
                skipped_in_place_tokens=plan.accepted_tokens-plan.moved_tokens,
                launches=launches,
                copy_read_bytes=2*accepted_payload if reference else moved_payload,
                copy_write_bytes=2*accepted_payload if reference else moved_payload,
                copy_total_bytes=4*accepted_payload if reference else 2*moved_payload,
                reference_gather_copy_total_bytes=4*accepted_payload,
                removed_logical_copy_bytes=0 if reference else 4*accepted_payload-2*moved_payload,
                scratch_payload_peak_bytes=accepted_payload if reference else 0,
                global_payload_temporary_bytes=accepted_payload if reference else 0,
                index_payload_bytes_read_per_head=0 if reference or not launches else 8*plan.accepted_tokens,
                metadata_updated=False, traffic_is_measured_dram=False,
                byte_scope='K/V payload only; index reads and caller metadata reported separately',
                source_start=plan.scratch_start,destination_start=plan.destination_start,
                source_rows=list(plan.positions),moved_destination_ordinals=list(plan.moved_positions),
                overlap_safe=True, payload_arithmetic=False,
                note=('CPU reference allocates temporary selected payload; no GPU performance claim'
                      if backend=='cpu_reference' else
                      'one CTA per head; volatile integer loads of all moved K/V before CTA barrier/stores'))


_KERNEL = None


def _get_kernel():
    """Lazy Triton import/definition; importing this module creates no CUDA state."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=['source_start','destination_start','accepted',
                                 'head_stride','row_stride'])
    def _delta_commit_kernel(K, V, Indices, source_start, destination_start,
                             accepted, head_stride, row_stride,
                             BLOCK_ROWS: tl.constexpr, D: tl.constexpr):
        head = tl.program_id(0)
        rows = tl.arange(0, BLOCK_ROWS)
        channels = tl.arange(0, D)
        source = tl.load(Indices+rows, mask=rows<accepted, other=0)
        moving = (rows<accepted) & (source_start+source != destination_start+rows)
        source_offset = head*head_stride+(source_start+source[:,None])*row_stride+channels[None,:]
        destination_offset = head*head_stride+(destination_start+rows[:,None])*row_stride+channels[None,:]
        # Both banks are integer-reinterpreted; NaN payloads/-0 never undergo a
        # floating conversion. Volatile loads plus CTA barrier prohibit moving
        # a store before another warp's source loads have finished. Each head's
        # source/destination region is owned by exactly this CTA.
        key_bits = tl.load(K+source_offset, mask=moving[:,None], other=0, volatile=True)
        value_bits = tl.load(V+source_offset, mask=moving[:,None], other=0, volatile=True)
        tl.debug_barrier()
        tl.store(K+destination_offset, key_bits, mask=moving[:,None])
        tl.store(V+destination_offset, value_bits, mask=moving[:,None])

    _KERNEL = _delta_commit_kernel
    return _KERNEL


def delta_commit_(keys, values, plan, *, backend='triton'):
    """Modify only K/V payload, return a JSON receipt; no silent fallback.

    CUDA path returns after enqueueing on the current stream. Caller handles
    synchronization/metadata and includes preparation/checks in request timing.
    CPU reference is explicitly selected and has a global gather temporary;
    only the Triton path claims no payload temporary. All-in-place does not
    import Triton or enqueue a payload kernel.
    """
    heads = _validate(keys, values, plan)
    if backend not in ('triton','cpu_reference'):
        raise ValueError('backend must be triton or cpu_reference')
    if backend=='triton' and keys.device.type!='cuda':
        raise ValueError('triton backend requires CUDA; use explicit CPU reference')
    if backend=='cpu_reference' and keys.device.type!='cpu':
        raise ValueError('CPU reference requires CPU tensors')
    if not plan.moved_tokens:
        return _receipt(keys,plan,backend,0)
    import torch
    if backend=='cpu_reference':
        with torch.inference_mode():
            sources = torch.tensor([plan.scratch_start+plan.positions[i]
                                    for i in plan.moved_positions],dtype=torch.long)
            targets = torch.tensor([plan.destination_start+i for i in plan.moved_positions],dtype=torch.long)
            k = _integer_view(keys).index_select(-2,sources)
            v = _integer_view(values).index_select(-2,sources)
            _integer_view(keys).index_copy_(-2,targets,k)
            _integer_view(values).index_copy_(-2,targets,v)
        out = _receipt(keys,plan,backend,0)
        temporary = 2*heads*plan.moved_tokens*keys.shape[-1]*keys.element_size()
        out.update(scratch_payload_peak_bytes=temporary,global_payload_temporary_bytes=temporary,
                   copy_read_bytes=2*temporary,copy_write_bytes=2*temporary,
                   copy_total_bytes=4*temporary,
                   removed_logical_copy_bytes=out['reference_gather_copy_total_bytes']-4*temporary,
                   note='CPU oracle only; temporary gather and index_copy; not CUDA logical cost')
        return out
    kernel = _get_kernel()
    kernel[(heads,)](keys.view(torch.int16),values.view(torch.int16),plan.commit_plan._indices,
                     plan.scratch_start,plan.destination_start,plan.accepted_tokens,
                     keys.stride(1),keys.stride(2),BLOCK_ROWS=32,D=128,num_warps=4,num_stages=1)
    return _receipt(keys,plan,backend,1)


def reference_commit_(keys, values, plan):
    """Independent original gather-before-copy oracle, CPU or CUDA, bit preserving."""
    _validate(keys,values,plan)
    import torch
    with torch.inference_mode():
        indices = plan.commit_plan._indices
        kb = _integer_view(keys); vb = _integer_view(values)
        selected_k = kb.narrow(-2,plan.scratch_start,plan.tree_nodes).index_select(-2,indices)
        selected_v = vb.narrow(-2,plan.scratch_start,plan.tree_nodes).index_select(-2,indices)
        kb.narrow(-2,plan.destination_start,plan.accepted_tokens).copy_(selected_k)
        vb.narrow(-2,plan.destination_start,plan.accepted_tokens).copy_(selected_v)
    return _receipt(keys,plan,'torch_gather_copy',4,reference=True)
