"""Scoped precision setting for packed scout down-projection calls.

Weights, inputs and outputs keep their dtypes. Only PyTorch's optional reduced-
precision FP16 GEMM reduction is disabled while the original down_proj forward
runs. The caller must use this solely around its packed probe calls. No kernel
speed, bitwise equivalence on unseen inputs, or QA guarantee is implied.
"""
from contextlib import contextmanager
from types import MethodType


@contextmanager
def packed_downproj_accumulation(block, *, label):
    import torch
    module = block.get_submodule("mlp.down_proj")
    existed = "forward" in module.__dict__
    saved = module.__dict__.get("forward")
    original = module.forward
    metadata = dict(scope=f"Only packed {label} mlp.down_proj.forward", block=label,
        entry_flag=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        requested_flag=False, tensor_dtype_changed=False, down_proj_calls=[])

    def forward(self, *args, **kwargs):
        old = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        call = dict(flag_before=old, requested_flag=False)
        metadata["down_proj_calls"].append(call)
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            call["flag_during"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            return original(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old
            call["flag_after"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction

    module.forward = MethodType(forward, module)
    try:
        yield metadata
    finally:
        if existed:
            module.forward = saved
        else:
            delattr(module, "forward")
        metadata["exit_flag"] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        metadata["forward_restored"] = ((module.__dict__.get("forward") is saved) if existed
                                        else "forward" not in module.__dict__)
