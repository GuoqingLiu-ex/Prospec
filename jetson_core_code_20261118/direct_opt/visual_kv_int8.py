"""Isolated visual-KV quantization primitive; no model hooks and no eviction.

The first configuration is symmetric INT8, 32 contiguous channels per group,
FP32 scales, 64-token pages. All positions are retained. Source keys are already
RoPE-transformed; this module never changes or reapplies position embeddings.
CPU routines are references. CUDA packing is delegated to a fused Triton kernel
and never constructs a full floating-point dequantized cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

GROUP_SIZE = 32
PAGE_SIZE = 64


@dataclass(frozen=True)
class KVPartition:
    prefix_tokens: int
    visual: tuple[int, ...]
    nonvisual: tuple[int, ...]

    def text_positions(self, generated_and_tree_tokens: int = 0) -> tuple[int, ...]:
        if generated_and_tree_tokens < 0:
            raise ValueError("negative appended length")
        return self.nonvisual + tuple(range(self.prefix_tokens,
                                           self.prefix_tokens + generated_and_tree_tokens))


@dataclass(frozen=True)
class KVPositionMap:
    indices: torch.Tensor
    host_indices: tuple[int, ...]
    total_tokens: int
    visual_indices: tuple[int, ...]

    def __post_init__(self):
        maps = self.visual_indices + self.host_indices
        if (any(type(i) is not int for i in maps) or self.total_tokens < 0 or
            len(maps) != self.total_tokens or sorted(maps) != list(range(self.total_tokens))):
            raise ValueError("two setup maps must cover every original position exactly once")


def prepare_fp16_positions(partition: KVPartition, appended_tokens: int,
                           device: torch.device | str) -> KVPositionMap:
    """Prepare once per tree layout, shared across layers; no GPU-to-host read."""
    positions = partition.text_positions(appended_tokens)
    return KVPositionMap(torch.tensor(positions, dtype=torch.long, device=device),
                         positions, partition.prefix_tokens + appended_tokens, partition.visual)


def partition_prefix(prefix_tokens: int, visual_positions: Sequence[int]) -> KVPartition:
    if prefix_tokens < 0:
        raise ValueError("negative prefix length")
    positions = tuple(visual_positions)
    if any(type(p) is not int for p in positions):
        raise TypeError("visual positions must be host integers, not GPU scalars")
    if len(set(positions)) != len(positions) or any(p < 0 or p >= prefix_tokens for p in positions):
        raise ValueError("visual positions must be unique valid prefix indices")
    # Preserve supplied order; the attention mask follows the original indices.
    selected = set(positions)
    return KVPartition(prefix_tokens, positions, tuple(i for i in range(prefix_tokens) if i not in selected))


def padded_tokens(n: int, page_size: int = PAGE_SIZE) -> int:
    if n < 0 or page_size != PAGE_SIZE:
        raise ValueError("first configuration requires nonnegative length and page_size=64")
    return ((n + page_size - 1) // page_size) * page_size


@dataclass
class VisualKVBank:
    keys: torch.Tensor                 # [B,Hkv,Npad,D], int8 or fp16 control
    values: torch.Tensor
    positions: torch.Tensor            # [Nlogical], original mask-column indices
    logical_tokens: int
    key_scales: torch.Tensor | None = None   # [B,Hkv,Npad,D/32], fp32
    value_scales: torch.Tensor | None = None
    nonfinite_flag: torch.Tensor | None = None  # CUDA pack scalar; explicit diagnostic read only
    host_positions: tuple[int, ...] = ()  # setup receipt; maps are immutable during execution

    @property
    def quantized(self) -> bool:
        return self.keys.dtype == torch.int8

    def validate_metadata(self) -> None:
        if self.keys.ndim != 4 or self.keys.shape != self.values.shape:
            raise ValueError("KV banks require equal [B,H,Npad,D] shapes")
        if self.keys.dtype != self.values.dtype or self.keys.dtype not in (torch.int8, torch.float16):
            raise ValueError("bank dtype must be INT8 or FP16 control")
        if self.keys.device != self.values.device or self.positions.device != self.keys.device:
            raise ValueError("bank tensors must share a device")
        b, h, n, d = self.keys.shape
        if b < 1 or h < 1 or d != 128 or n != padded_tokens(self.logical_tokens):
            raise ValueError("first configuration requires D=128 and exact page-aligned capacity")
        if self.positions.dtype != torch.long or tuple(self.positions.shape) != (self.logical_tokens,):
            raise ValueError("positions must be int64 [Nlogical]")
        if self.positions.stride() != (1,) or len(self.host_positions) != self.logical_tokens:
            raise ValueError("positions require contiguous storage and a matching host receipt")
        if self.quantized:
            for scale in (self.key_scales, self.value_scales):
                if scale is None or scale.dtype != torch.float32 or scale.device != self.keys.device:
                    raise ValueError("INT8 bank requires same-device FP32 scales")
                if tuple(scale.shape) != (b, h, n, d // GROUP_SIZE):
                    raise ValueError("incorrect scale shape")
                if not scale.is_contiguous():
                    raise ValueError("scale arrays must be contiguous")
        elif self.key_scales is not None or self.value_scales is not None:
            raise ValueError("FP16 control must not carry quantization scales")

    def storage_report(self) -> dict:
        self.validate_metadata()
        tensors = [self.keys, self.values, self.positions, self.key_scales,
                   self.value_scales, self.nonfinite_flag]
        storages = {}
        for tensor in tensors:
            if tensor is not None:
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return {
            "quantized": self.quantized,
            "logical_tokens": self.logical_tokens,
            "padded_tokens": self.keys.shape[2],
            "kv_payload_bytes": self.keys.numel() * self.keys.element_size() * 2,
            "scale_bytes": 0 if not self.quantized else self.key_scales.numel() * 4 * 2,
            "position_bytes": self.positions.numel() * self.positions.element_size(),
            "unique_backing_bytes": sum(storages.values()),
            "group_size": GROUP_SIZE, "page_size": PAGE_SIZE,
            "full_cache_dequantization": False,
            "scope": "one-layer visual bank only; excludes text/scratch/model/temporary allocations",
        }


def validate_source(keys: torch.Tensor, values: torch.Tensor, partition: KVPartition) -> None:
    if keys.ndim != 4 or keys.shape != values.shape or keys.shape[-1] != 128:
        raise ValueError("source must be matching [B,H,N,128]")
    if keys.dtype != torch.float16 or values.dtype != torch.float16 or keys.device != values.device:
        raise ValueError("first configuration requires same-device FP16 K/V")
    if keys.shape[2] != partition.prefix_tokens or keys.shape[0] < 1 or keys.shape[1] < 1:
        raise ValueError("source and partition disagree")


def _quantize_cpu(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.device.type != "cpu":
        raise ValueError("CPU reference cannot materialize a CUDA dequantization buffer")
    x = values.float().reshape(*values.shape[:-1], 4, GROUP_SIZE)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("non-finite source KV")
    maximum = x.abs().amax(dim=-1)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)
    quant = torch.round(x / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return quant.reshape(values.shape), scale


def pack_visual_cpu(keys: torch.Tensor, values: torch.Tensor, partition: KVPartition,
                    *, quantized: bool = True) -> VisualKVBank:
    """Reference pack. Explicitly CPU-only; never used as a CUDA fallback."""
    validate_source(keys, values, partition)
    if keys.device.type != "cpu":
        raise ValueError("pack_visual_cpu is CPU-only")
    positions = torch.tensor(partition.visual, dtype=torch.long)
    b, h, _, d = keys.shape
    n = padded_tokens(len(partition.visual))
    dtype = torch.int8 if quantized else torch.float16
    out_k, out_v = (torch.zeros((b, h, n, d), dtype=dtype) for _ in range(2))
    scales = [torch.ones((b, h, n, 4), dtype=torch.float32) for _ in range(2)] if quantized else [None, None]
    for source, output, scale_out in zip((keys, values), (out_k, out_v), scales):
        selected = source.index_select(2, positions)
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("non-finite source KV")
        if quantized:
            data, scale = _quantize_cpu(selected)
            output[:, :, :len(positions)] = data
            scale_out[:, :, :len(positions)] = scale
        else:
            output[:, :, :len(positions)] = selected
    bank = VisualKVBank(out_k, out_v, positions, len(positions), *scales,
                        host_positions=partition.visual)
    bank.validate_metadata()
    return bank


def dequantize_cpu(bank: VisualKVBank) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 diagnostic reference only. No CUDA implementation is provided."""
    bank.validate_metadata()
    if bank.keys.device.type != "cpu":
        raise ValueError("global dequantization is confined to the CPU diagnostic reference")
    outputs = []
    for x, scale in ((bank.keys, bank.key_scales), (bank.values, bank.value_scales)):
        x = x[:, :, :bank.logical_tokens].double()
        if bank.quantized:
            x = (x.reshape(*x.shape[:-1], 4, 32) *
                 scale[:, :, :bank.logical_tokens].double().unsqueeze(-1)).reshape(x.shape)
            # The fused kernel converts each dequantized tile to FP16 before dot.
            x = x.half().double()
        outputs.append(x)
    return tuple(outputs)


def theoretical_visual_budget(tokens: int, *, layers: int = 28, batch: int = 1,
                              kv_heads: int = 4) -> dict:
    if min(layers, batch, kv_heads) < 1:
        raise ValueError("positive dimensions required")
    n = padded_tokens(tokens)
    elements_per_token = layers * batch * kv_heads * 2 * 128
    return {
        "native_fp16_logical_bytes": tokens * elements_per_token * 2,
        "int8_logical_payload_bytes": tokens * elements_per_token,
        "fp32_logical_scale_bytes": tokens * elements_per_token // 32 * 4,
        "int8_padded_payload_bytes": n * elements_per_token,
        "fp32_padded_scale_bytes": n * elements_per_token // 32 * 4,
        "int8_padded_total_bytes": n * elements_per_token * 9 // 8,
        "logical_quantization_read_bytes": tokens * elements_per_token * 2,
        "valid_quantization_output_bytes": tokens * elements_per_token * 9 // 8,
        "padded_pack_output_write_bytes": n * elements_per_token * 9 // 8,
        "excluded": ["position metadata", "nonvisual/text/scratch", "model weights", "allocator", "DRAM cache effects"],
    }
