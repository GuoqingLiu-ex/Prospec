"""Two fixed inference candidates for ViSpec's consecutive visual fusion linears.

img_fc([h,g]) = A h + B g + a; fc([e,y]) = E e + H y + b.
There is no activation between the two linears in the supported author path.

cached_image: cache B g + a, evaluate A h + cache, then the original fc linear.
folded_pair: prepare F=H A, R=H B, c=b+H a, then cache R g+c and evaluate
             linear(cat(e,h), cat(E,F), cache).

Both alter floating-point operation order; neither promises bitwise equality
with the original pair. Production folded weights are composed in FP32 and
cast once to model dtype. FP64 sources use FP64 composition for math testing.

Weights must remain frozen. Call reset_cache() for every new request and after
any in-place mutation of g. Cache identity is the Python Tensor object, not its
data pointer or values. A strong g reference prevents pointer reuse. No Tensor
values, mutation counters, or GPU-to-CPU synchronization are read on the hot path.
This module does not alter or patch any Target/Draft decoder code.
"""
from __future__ import annotations

import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _storage_info(named_tensors):
    stores, logical = {}, 0
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        key = str(tensor.device), storage.data_ptr()
        logical += tensor.numel() * tensor.element_size()
        stores.setdefault(key, {"device": key[0], "data_ptr": key[1],
                                "storage_bytes": storage.nbytes(), "names": []})["names"].append(name)
    return {"logical_bytes": logical, "unique_storage_bytes": sum(s["storage_bytes"] for s in stores.values()),
            "storages": list(stores.values())}


class FrozenVisualFusion(nn.Module):
    """forward(e,h,g), with e/h [N,D] or [1,N,D] and g [1,D] or [1,1,D].

Construct after loading/casting frozen source weights. cached_image aliases
source weight storage; folded_pair has independent composed weights. Neither
keeps source modules. Preparation is outside request timing; each cache miss
performs its global projection inside forward and must be counted in timing.
"""

    WEIGHTS = ("hidden_weight", "global_weight", "img_bias", "fc_weight", "fc_bias",
               "fused_weight", "fused_global_weight", "fused_bias")

    @torch.no_grad()
    def __init__(self, img_fc: nn.Linear, fc: nn.Linear, *, mode="cached_image"):
        super().__init__()
        if mode not in ("cached_image", "folded_pair"):
            raise ValueError("mode must be cached_image or folded_pair")
        if not isinstance(img_fc, nn.Linear) or not isinstance(fc, nn.Linear):
            raise TypeError("img_fc and fc must be nn.Linear")
        d = img_fc.out_features
        if d < 1 or img_fc.in_features != 2*d or fc.in_features != 2*d or fc.out_features != d:
            raise ValueError("requires img_fc and fc both Linear(2*D,D)")
        params = [("img_fc.weight", img_fc.weight), ("img_fc.bias", img_fc.bias),
                  ("fc.weight", fc.weight), ("fc.bias", fc.bias)]
        dtype, device = img_fc.weight.dtype, img_fc.weight.device
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError("weights must use a supported floating-point dtype")
        if any(t.dtype != dtype or t.device != device for _, t in params if t is not None):
            raise ValueError("all weights and biases must share dtype and device")
        self.mode, self.hidden_size = mode, d
        self._prepared_dtype = dtype
        self.composition_dtype = (torch.float64 if dtype == torch.float64 else torch.float32) if mode == "folded_pair" else None
        self.prepare_device = str(device)
        self.prepare_has_device_sync = device.type in ("cuda", "mps")
        self.composition_backend_policy = {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "autocast": False,
        }
        self.source_storage = _storage_info(params)
        for name in self.WEIGHTS:
            self.register_buffer(name, None, persistent=True)
        self.register_buffer("_cached_projection", None, persistent=False)
        object.__setattr__(self, "_cached_global", None)
        self.cache_hits = self.cache_misses = self.cache_resets = 0
        _sync(device)
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=False):
            a_weight, b_weight = img_fc.weight.detach().split(d, dim=-1)
            e_weight, h_weight = fc.weight.detach().split(d, dim=-1)
            if mode == "cached_image":
                self.hidden_weight, self.global_weight = a_weight, b_weight
                self.img_bias = None if img_fc.bias is None else img_fc.bias.detach()
                self.fc_weight = fc.weight.detach()
                self.fc_bias = None if fc.bias is None else fc.bias.detach()
            else:
                h = h_weight.to(self.composition_dtype)
                f = torch.mm(h, a_weight.to(self.composition_dtype))
                r = torch.mm(h, b_weight.to(self.composition_dtype))
                self.fused_weight = torch.cat((e_weight, f.to(dtype)), dim=-1)
                self.fused_global_weight = r.to(dtype)
                bias = None
                if img_fc.bias is not None:
                    bias = torch.mv(h, img_fc.bias.detach().to(self.composition_dtype))
                if fc.bias is not None:
                    source_bias = fc.bias.detach().to(self.composition_dtype)
                    bias = source_bias.clone() if bias is None else source_bias + bias
                self.fused_bias = None if bias is None else bias.to(dtype)
        _sync(device)
        self.prepare_time_ms = (time.perf_counter() - started) * 1000.0
        self.eval()

    @property
    def _weight(self):
        return self.hidden_weight if self.mode == "cached_image" else self.fused_weight

    def reset_cache(self):
        """Required at request boundaries and after modifying g in-place."""
        object.__setattr__(self, "_cached_global", None)
        self._cached_projection = None
        self.cache_resets += 1

    def _apply(self, fn, recurse=True):
        self.reset_cache()
        return super()._apply(fn, recurse=recurse)

    def _check_inputs(self, embed, hidden, global_visual):
        if any(not isinstance(x, Tensor) for x in (embed, hidden, global_visual)):
            raise TypeError("embed, hidden and global_visual must be Tensors")
        d = self.hidden_size
        if (embed.ndim not in (2, 3) or embed.shape != hidden.shape or embed.shape[-1] != d or
                (embed.ndim == 3 and embed.shape[0] != 1)):
            raise ValueError("embed/hidden require identical [N,D] or [1,N,D] shapes")
        if global_visual.shape not in ((1, d), (1, 1, d)):
            raise ValueError("global_visual must contain one vector: [1,D] or [1,1,D]")
        weight = self._weight
        if weight.dtype != self._prepared_dtype:
            raise RuntimeError("construct fusion after final dtype casting; do not cast prepared weights")
        if any(x.dtype != weight.dtype or x.device != weight.device for x in (embed, hidden, global_visual)):
            raise ValueError("all inputs must match prepared weight dtype and device")

    def _global_projection(self, global_visual):
        if global_visual is self._cached_global:
            self.cache_hits += 1
            return self._cached_projection
        g = global_visual.reshape(self.hidden_size)
        projection = (F.linear(g, self.global_weight, self.img_bias) if self.mode == "cached_image" else
                      F.linear(g, self.fused_global_weight, self.fused_bias))
        object.__setattr__(self, "_cached_global", global_visual)
        self._cached_projection = projection
        self.cache_misses += 1
        return projection

    @torch.no_grad()
    def forward(self, embed: Tensor, hidden: Tensor, global_visual: Tensor) -> Tensor:
        self._check_inputs(embed, hidden, global_visual)
        if embed.numel() == 0:
            return torch.empty_like(embed)
        with torch.autocast(device_type=embed.device.type, enabled=False):
            projected_global = self._global_projection(global_visual)
            if self.mode == "cached_image":
                y = F.linear(hidden, self.hidden_weight) + projected_global
                return F.linear(torch.cat((embed, y), dim=-1), self.fc_weight, self.fc_bias)
            return F.linear(torch.cat((embed, hidden), dim=-1), self.fused_weight, projected_global)

    def cache_info(self):
        projection = self._cached_projection
        retained = self._cached_global
        return {"hits": self.cache_hits, "misses": self.cache_misses, "resets": self.cache_resets,
                "populated": retained is not None,
                "projection_bytes": 0 if projection is None else projection.numel() * projection.element_size(),
                "retained_global_storage_bytes": 0 if retained is None else retained.untyped_storage().nbytes(),
                "identity_key": "Python Tensor object; strong reference; in-place updates require reset_cache"}

    def metadata(self):
        weights = _storage_info([(name, getattr(self, name)) for name in self.WEIGHTS])
        return {"mode": self.mode, "hidden_size": self.hidden_size,
                "model_dtype": str(self._prepared_dtype), "composition_dtype": str(self.composition_dtype),
                "composition_rule": "FP32 for production; FP64 for FP64 mathematical tests" if self.mode == "folded_pair" else "none",
                "prepare_time_ms": self.prepare_time_ms,
                "setup_scope": "weight views/composition and device completion; excludes runtime global projection; may include library initialization",
                "prepare_clock": "perf_counter; synchronize before start and after operations on CUDA/MPS; CPU synchronous wall",
                "prepare_device": self.prepare_device, "prepare_has_device_sync": self.prepare_has_device_sync,
                "composition_backend_policy": self.composition_backend_policy,
                "source_weights": self.source_storage, "deployed_weights": weights,
                "original_modules_retained": False, "aliases_source_weights": self.mode == "cached_image",
                "requires_frozen_weights": True, "bitwise_equivalence_claim": False,
                "theoretical_weight_storage_reduction_bytes": self.source_storage["unique_storage_bytes"] - weights["unique_storage_bytes"],
                "cache": self.cache_info(), "cache_miss_cost_in_forward": True,
                "timing_or_quality_validated_on_gpu": False,
                "startup_peak_saving_proven": False}

    def preparation_metadata(self):
        return self.metadata()
