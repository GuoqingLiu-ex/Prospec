"""Strict one-time aliasing of identical frozen Target/Draft input embeddings.

Install after both models have reached their final device and dtype. This file
does not change loading, decoding, checkpoints, or any forward function. A
successful installation releases the independently stored Draft weight. It does
not imply a lower checkpoint-loading/startup peak or measured DRAM savings.
"""
from __future__ import annotations

import gc
import weakref

import torch
from torch import nn


def _storage_key(tensor):
    if tensor.device.type == 'meta' or tensor.layout != torch.strided:
        raise ValueError('Storage accounting requires materialized strided tensors')
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr()), storage.nbytes()


def _storage_summary(*modules):
    storages = {}
    occurrences = 0
    for module in modules:
        if not isinstance(module, nn.Module):
            raise TypeError('Target and Draft must be nn.Module instances')
        tensors = list(module.named_parameters(recurse=True, remove_duplicate=False))
        tensors += list(module.named_buffers(recurse=True, remove_duplicate=False))
        for _name, tensor in tensors:
            key, size = _storage_key(tensor)
            if key in storages and storages[key]['bytes'] != size:
                raise RuntimeError('Inconsistent byte size for a shared storage')
            entry = storages.setdefault(key, {'bytes':size, 'references':0})
            entry['references'] += 1
            occurrences += 1
    return {'unique_storage_bytes':sum(v['bytes'] for v in storages.values()),
            'unique_storages':len(storages), 'registered_tensor_occurrences':occurrences,
            'storages':storages}


def unique_model_storage_bytes(*modules) -> int:
    """Deduplicate registered parameter/buffer storage across every given model.

    Counts complete underlying storages, including view backing storage. Does
    not count unregistered caches, external tensor references, allocator reserve,
    temporary workspaces or CUDA-library allocations.
    """
    return _storage_summary(*modules)['unique_storage_bytes']


def _validate_embedding(embedding, label):
    if type(embedding) is not nn.Embedding:
        raise TypeError(f'{label} must be an unmodified nn.Embedding')
    if embedding.training:
        raise ValueError(f'{label} must be in eval mode')
    if ('forward' in embedding.__dict__ or embedding._forward_pre_hooks or
            embedding._forward_hooks or embedding._backward_hooks):
        raise ValueError(f'{label} has custom forward behavior/hooks')
    if (embedding.max_norm is not None or embedding.norm_type != 2.0 or
            embedding.scale_grad_by_freq or embedding.sparse):
        raise ValueError(f'{label} has unsupported special embedding semantics')
    weight = embedding.weight
    if not isinstance(weight, nn.Parameter) or weight.requires_grad:
        raise ValueError(f'{label} weight must be a frozen Parameter')
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f'{label} weight requires a supported floating dtype')
    if tuple(weight.shape) != (embedding.num_embeddings, embedding.embedding_dim):
        raise ValueError(f'{label} metadata does not match its weight shape')
    if weight.device.type == 'meta' or weight.layout != torch.strided or not weight.is_contiguous():
        raise ValueError(f'{label} requires a materialized contiguous weight')
    if weight.storage_offset() != 0 or weight.untyped_storage().nbytes() != weight.numel()*weight.element_size():
        raise ValueError(f'{label} weight must span its complete backing storage')


def _bitwise_equal_chunks(left, right, chunk_rows):
    chunks = 0
    for start in range(0, left.shape[0], chunk_rows):
        # Views preserve exact representation, including signed zero/NaN bits.
        a = left[start:start+chunk_rows].view(torch.uint8)
        b = right[start:start+chunk_rows].view(torch.uint8)
        if not torch.equal(a, b):
            raise ValueError(f'Embedding weights are not bitwise equal in rows {start}:{min(start+chunk_rows,left.shape[0])}')
        chunks += 1
    return chunks


@torch.no_grad()
def install_shared_embedding(target, draft, *, chunk_rows: int = 4096) -> dict:
    """Alias the Draft weight to Target only after exhaustive bitwise validation.

    Every successful return proves both the original Parameter and its storage
    were released (via weak references), and registered storage decreased by
    exactly the independent Draft embedding storage size. No checks are added
    to inference. Equality checks may synchronize CUDA, only during setup.

    A release failure raises after aliasing: an external reference may still own
    the original storage. The caller must abort that measurement process rather
    than claim a memory saving. Repeated installation also raises explicitly.
    """
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    if not isinstance(target, nn.Module) or not isinstance(draft, nn.Module):
        raise TypeError('Target and Draft must be nn.Module instances')
    if target.training or draft.training:
        raise ValueError('Target and Draft must both be in eval mode')
    if any(p.requires_grad for m in (target,draft) for p in m.parameters()):
        raise ValueError('All Target and Draft parameters must be frozen')
    if not callable(getattr(target, 'get_input_embeddings', None)):
        raise TypeError('Target lacks get_input_embeddings()')
    target_embedding = target.get_input_embeddings()
    draft_embedding = getattr(draft, 'embed_tokens', None)
    _validate_embedding(target_embedding, 'Target input embedding')
    _validate_embedding(draft_embedding, 'Draft input embedding')
    old_weight = draft_embedding.weight
    shared_weight = target_embedding.weight
    if old_weight is shared_weight:
        raise ValueError('Embedding is already the same Parameter; no independent storage to release')
    if old_weight.shape != shared_weight.shape:
        raise ValueError('Embedding shapes differ')
    if old_weight.dtype != shared_weight.dtype:
        raise ValueError('Embedding dtypes differ; installer never casts weights')
    if old_weight.device != shared_weight.device:
        raise ValueError('Embedding devices differ; installer never moves weights')
    if draft_embedding.padding_idx != target_embedding.padding_idx:
        raise ValueError('Embedding padding_idx differs')
    old_key, expected_bytes = _storage_key(old_weight)
    shared_key, _ = _storage_key(shared_weight)
    if old_key == shared_key:
        raise ValueError('Embedding storage is already shared through distinct Parameters')
    before = _storage_summary(target,draft)
    if before['storages'][old_key]['references'] != 1:
        raise ValueError('Original Draft embedding storage has another registered model reference')
    chunks = _bitwise_equal_chunks(old_weight, shared_weight, chunk_rows)
    old_storage = old_weight.untyped_storage()
    parameter_ref, storage_ref = weakref.ref(old_weight), weakref.ref(old_storage)
    shape, dtype, device = list(old_weight.shape), str(old_weight.dtype), str(old_weight.device)
    draft_embedding.weight = shared_weight
    del old_weight, old_storage
    gc.collect()
    parameter_released = parameter_ref() is None
    storage_released = storage_ref() is None
    if not parameter_released or not storage_released:
        raise RuntimeError('Embedding alias installed but original storage release failed: '
                           f'old_parameter_released={parameter_released}, old_storage_released={storage_released}; '
                           'remove external original-weight/view references and abort this measurement')
    after = _storage_summary(target,draft)
    actual_reduction = before['unique_storage_bytes']-after['unique_storage_bytes']
    if actual_reduction != expected_bytes:
        raise RuntimeError('Registered model storage reduction differs from original embedding storage')
    if draft_embedding.weight is not target_embedding.weight:
        raise RuntimeError('Same-Parameter alias postcondition failed')
    return dict(installed=True, bitwise_equal=True, equality_method='exhaustive chunked torch.equal on uint8 views',
                equality_chunks=chunks, chunk_rows=chunk_rows, weight_shape=shape, dtype=dtype, device=device,
                same_parameter=True, same_storage=True, old_parameter_released=parameter_released,
                old_storage_released=storage_released, expected_released_bytes=expected_bytes,
                actual_unique_storage_reduction_bytes=actual_reduction,
                unique_model_storage_bytes_before=before['unique_storage_bytes'],
                unique_model_storage_bytes_after=after['unique_storage_bytes'],
                unique_model_storages_before=before['unique_storages'], unique_model_storages_after=after['unique_storages'],
                storage_scope='Registered parameter and buffer backing storages; excludes external caches/workspaces/allocator reserve',
                installed_after_loading=True, startup_peak_reduction_claim=False,
                forward_modified=False, per_forward_validation=False, checkpoint_modified=False)
