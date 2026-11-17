"""Isolated FP32 math GQA candidate with stride-zero shared K/V batches.

Reference order: PyTorch v2.8.0 attention.cpp, lines 793--877:
https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/transformers/attention.cpp#L793-L877

This is NOT a fused backend or runtime integration. It preserves all original
keys, positions and additive mask values. Batch28 GEMMs become four batch7
GEMMs; numerical bits may change. Python views prove no explicit repeat tensor,
not absence of internal BLAS packing or DRAM rereads. GPU throughput, actual
copies, allocated peaks and complete-generation quality require separate tests.
"""
from __future__ import annotations

import math
import torch

HQ, HKV, GROUP, D = 28, 4, 7, 128


def _description(tensor):
    return dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),
        device=str(tensor.device),logical_bytes=tensor.numel()*tensor.element_size(),
        storage_bytes=tensor.untyped_storage().nbytes(),
        storage_ptr=tensor.untyped_storage().data_ptr(),data_ptr=tensor.data_ptr(),
        storage_offset=tensor.storage_offset(),contiguous=tensor.is_contiguous())


def _validate(q,k,v,mask,scale,debug):
    if any(not isinstance(t,torch.Tensor) for t in (q,k,v,mask)):
        raise ValueError('Q/K/V and original explicit FP32 additive mask tensors are required')
    if any(t.ndim!=4 for t in (q,k,v,mask)):
        raise ValueError('all operands must be four-dimensional')
    if q.shape[:2]!=(1,HQ) or q.shape[-1]!=D or not 1<=q.shape[-2]<=30:
        raise ValueError('supported query shape is [1,28,T,128] with 1<=T<=30')
    if k.shape!=v.shape or k.shape[:2]!=(1,HKV) or k.shape[-1]!=D or k.shape[-2]<1:
        raise ValueError('supported unique KV shape is [1,4,N,128] with N>0')
    if mask.shape!=(1,1,q.shape[-2],k.shape[-2]):
        raise ValueError('original additive mask must be [1,1,T,N]')
    if any(t.dtype!=torch.float16 for t in (q,k,v)) or mask.dtype!=torch.float32:
        raise ValueError('Q/K/V must be FP16 and original additive mask FP32')
    if any(t.device!=q.device for t in (k,v,mask)) or q.device.type not in ('cpu','cuda'):
        raise ValueError('all operands must share a CPU or CUDA device')
    if any(t.requires_grad for t in (q,k,v,mask)):
        raise ValueError('inference-only: requires_grad tensors are unsupported')
    if any(t.stride(-1)!=1 or any(s<0 for s in t.stride()) for t in (q,k,v,mask)):
        raise ValueError('last axis must be contiguous; unsupported strides are not packed')
    if type(debug) is not bool:
        raise TypeError('debug_intermediates must be bool')
    scale=1.0/math.sqrt(D) if scale is None else scale
    if isinstance(scale,bool) or not isinstance(scale,(float,int)) or not math.isfinite(scale) or scale<=0:
        raise ValueError('positive finite scalar scale required')
    if torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
        raise RuntimeError('reference requires reduced-precision math SDPA disabled')
    if q.device.type=='cuda' and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError('CUDA FP32 candidate requires TF32 disabled by the caller')
    return float(scale)


def logical_copy_accounting(query_tokens,key_tokens):
    """Explicit materialization traffic, NOT GEMM reads or measured DRAM.

    One logical source read per produced element is assumed even for repeated
    addresses. Source cache reuse is deliberately not interpreted as bus traffic.
    Common Q/softmax/PV/output costs are excluded from the *difference* table.
    """
    if type(query_tokens) is not int or not 1<=query_tokens<=30 or type(key_tokens) is not int or key_tokens<1:
        raise ValueError('positive supported query/key counts required')
    e=HKV*key_tokens*D
    native=dict(fp16_repeat_kv=dict(read_bytes=28*e,write_bytes=28*e),
        fp32_cast_kv=dict(read_bytes=28*e,write_bytes=56*e),
        fp32_scale_k=dict(read_bytes=28*e,write_bytes=28*e))
    candidate=dict(fp16_repeat_kv=dict(read_bytes=0,write_bytes=0),
        fp32_cast_kv=dict(read_bytes=4*e,write_bytes=8*e),
        fp32_scale_k=dict(read_bytes=4*e,write_bytes=4*e))
    def totals(items):
        return {name:sum(row[name] for row in items.values()) for name in ('read_bytes','write_bytes')}
    a,b=totals(native),totals(candidate)
    for row in (a,b):row['read_plus_write_bytes']=row['read_bytes']+row['write_bytes']
    return dict(native_hf_repeat_then_math=native,candidate_unique_kv=candidate,
        native_total=a,candidate_total=b,
        removed_explicit_component_bytes={name:a[name]-b[name] for name in a},
        actual_dram_measured=False,actual_blas_packing_measured=False,
        common_full_fp32_score_bytes=4*HQ*query_tokens*key_tokens,
        common_unused_fp16_probability_cast_write_bytes=2*HQ*query_tokens*key_tokens,
        attention_major_macs=2*HQ*query_tokens*key_tokens*D,
        attention_mac_reduction=0,kv_resident_capacity_reduction=0,
        source_read_assumption='one logical source read per produced element; cache/transactions not modeled',
        exclusions='GEMM operand rereads, softmax/internal temporaries, allocator/BLAS workspace and cache effects')


@torch.no_grad()
def stride0_gqa_math(query,key,value,mask,scale=None,*,debug_intermediates=False):
    """Return (contiguous HF [1,T,28,128] FP16 output, receipt, debug-or-None).

    Inputs are already RoPE-rotated. Mask values are passed directly to add_;
    there is no hard-mask conversion, causal triangle, selection or quantization.
    No data-dependent CPU synchronization, fallback or global flag mutation.
    Debug retains/clones intermediates and invalidates ordinary memory/timing
    interpretation. The caller must disable debug for performance measurement.
    """
    scale=_validate(query,key,value,mask,scale,debug_intermediates)
    t,n=query.shape[-2],key.shape[-2]
    receipt=dict(implementation='stride0_four_group_fp32_math',torch_version=torch.__version__,
        scale=scale,sqrt_scale=math.sqrt(scale),input={name:_description(x) for name,x in
            (('query',query),('key',key),('value',value),('mask',mask))},
        allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        reduced_precision_math_sdp=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()),
        blas_library=str(torch.backends.cuda.preferred_blas_library()),
        explicit_repeat_kv=False,groups=[],qk_bmm_calls=4,pv_bmm_calls=4,
        native_bmm_calls=2,extra_bmm_calls=6,head_mapping='KV g -> Q heads [7g,7g+7)',
        mask_same_object=True,mask_values_transformed=False,rotary_reapplied=False,
        debug_intermediates=debug_intermediates,debug_extra_copy_bytes=0,
        bitwise_equivalence_claim=False,actual_internal_copy_measured=False,
        hardware_benefit_requires='actual allocation/copy/DRAM evidence, independently from throughput',
        logical_copies=logical_copy_accounting(t,n))
    debug={} if debug_intermediates else None
    with torch.autocast(device_type=query.device.type,enabled=False):
        # Preserve native conversion and separate sqrt(scale) multiplications.
        # Cast/scale UNIQUE K/V before any zero-stride expansion.
        q32=query.to(torch.float32)
        k32=key.to(torch.float32)
        v32=value.to(torch.float32)
        q_scaled=q32*math.sqrt(scale)
        k_scaled_t=k32.transpose(-2,-1)*math.sqrt(scale)
        scores=torch.empty((1,HQ,t,n),dtype=torch.float32,device=query.device)
        for group in range(HKV):
            first,last=group*GROUP,(group+1)*GROUP
            qs=q_scaled[0,first:last]
            ks=k_scaled_t[0,group].unsqueeze(0).expand(GROUP,-1,-1)
            if ks.stride(0)!=0 or ks.untyped_storage().data_ptr()!=k_scaled_t.untyped_storage().data_ptr():
                raise RuntimeError('shared K view unexpectedly materialized')
            receipt['groups'].append(dict(kv_head=group,query_heads=[first,last],
                query=_description(qs),key_transposed_expanded=_description(ks),
                key_view_shares_unique_storage=True))
            torch.bmm(qs,ks,out=scores[0,first:last])
        if debug is not None:
            debug.update(q32=q32,k32=k32,v32=v32,q_scaled=q_scaled,k_scaled_transposed=k_scaled_t,
                         qk_scores=scores.clone())
            receipt['debug_extra_copy_bytes']=scores.numel()*scores.element_size()
        # Full [1,28,T,N] shape and original FP32 bias are retained for softmax.
        scores.add_(mask)
        probabilities=torch.ops.aten._safe_softmax.default(scores,-1,None)
        receipt['intermediate_tensors']={name:_description(x) for name,x in
            (('q32',q32),('k32',k32),('v32',v32),('q_scaled',q_scaled),
             ('k_scaled_transposed',k_scaled_t),('scores',scores),('probabilities',probabilities))}
        if debug is not None:
            debug.update(masked_scores=scores,probabilities=probabilities)
        del qs,ks,scores
        pv=torch.empty((1,HQ,t,D),dtype=torch.float32,device=query.device)
        for group in range(HKV):
            first,last=group*GROUP,(group+1)*GROUP
            vs=v32[0,group].unsqueeze(0).expand(GROUP,-1,-1)
            if vs.stride(0)!=0 or vs.untyped_storage().data_ptr()!=v32.untyped_storage().data_ptr():
                raise RuntimeError('shared V view unexpectedly materialized')
            receipt['groups'][group].update(value_expanded=_description(vs),value_view_shares_unique_storage=True)
            torch.bmm(probabilities[0,first:last],vs,out=pv[0,first:last])
        output_bhtd=pv.to(query.dtype)
        # The native math operator also returns a model-dtype attention tensor,
        # even when F.sdpa discards it. Retain its conversion in this first
        # candidate so that removing it is not attributed to shared KV views.
        unused_weights=probabilities.to(query.dtype)
        receipt['intermediate_tensors'].update(pv_fp32=_description(pv),
            output_bhtd_fp16=_description(output_bhtd),native_unused_weights_fp16=_description(unused_weights))
        receipt['native_unused_probability_cast_preserved']=True
        if debug is not None:
            debug.update(pv_fp32=pv,output_bhtd_fp16=output_bhtd,native_unused_weights_fp16=unused_weights)
        del unused_weights,vs,probabilities
        output=output_bhtd.transpose(1,2).contiguous()
        output_copied=output.untyped_storage().data_ptr()!=output_bhtd.untyped_storage().data_ptr()
        receipt.update(output=_description(output),hf_output_layout_copy=output_copied,
            hf_output_layout_copy_read_plus_write_bytes=2*output.numel()*output.element_size() if output_copied else 0,
            receipt_scope='Observed Python tensor layouts/storages and explicit logical copy formulas; '
                          'not simultaneous allocator peak, SRAM or actual DRAM. BLAS may internally pack/read repeatedly.')
    return output,receipt,debug
