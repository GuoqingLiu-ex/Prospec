"""Explicit GPU diagnostic entry point; never runs at import.

Input .pt keys: q [B,Hq,T,128], k/v [B,Hkv,N,128], mask
[B|1,Hq|1,T,N], prefix_tokens:int, visual_positions:list[int]. K/V are
already RoPE-transformed. Torch.load uses weights_only=True. Synthetic mode
tests the same 7:1 GQA/tree-mask contract but is NOT a model/QA benchmark.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import torch
from direct_opt.visual_kv_int8 import (partition_prefix,prepare_fp16_positions,
    pack_visual_cpu,dequantize_cpu)
from direct_opt.triton_mixed_attention import (pack_visual_cuda,mixed_attention,
    validate_attention_metadata)
from direct_opt import triton_mixed_attention as attention_module


def cpu_attention_reference(q, bank, fk, fv, positions, mask):
    """FP64 attention of the represented banks; global dequant confined to CPU."""
    if any(x.device.type != "cpu" for x in (q,fk,fv,mask)):
        raise ValueError("reference is CPU-only")
    validate_attention_metadata(q,bank,fk,fv,positions,mask)
    vk,vv = dequantize_cpu(bank)
    key = torch.cat((vk,fk.double()),dim=2)
    value = torch.cat((vv,fv.double()),dim=2)
    order = torch.cat((bank.positions,positions.indices))
    bias = mask.index_select(-1,order)
    group = q.shape[1]//key.shape[1]
    key = key.repeat_interleave(group,dim=1)
    value = value.repeat_interleave(group,dim=1)
    scores = q.double() @ key.transpose(-1,-2) / math.sqrt(128)
    if bias.dtype == torch.bool:
        scores = scores.masked_fill(~bias,-torch.inf)
    else:
        scores = scores+bias.double()
    probabilities = torch.softmax(scores,dim=-1)
    probabilities = torch.where(torch.isneginf(scores).all(-1,keepdim=True),0,probabilities)
    return probabilities @ value


def diagnostic(actual, expected):
    a,e = actual.double(),expected.double()
    difference = a-e
    return {
        "finite":bool(torch.isfinite(a).all()),
        "relative_l2":float(difference.norm()/e.norm().clamp_min(1e-30)),
        "max_abs_error":float(difference.abs().max()),
        "allclose_rtol_0.005_atol_0.005":bool(torch.allclose(a,e,rtol=0.005,atol=0.005)),
    }


def synthetic_case(seed=20261002):
    generator=torch.Generator().manual_seed(seed)
    prefix,t,b,h = 96,30,1,4
    n=prefix+t
    # Explicit tree: every node sees root; siblings do not see each other.
    mask=torch.zeros((1,1,t,n),dtype=torch.bool)
    mask[:,:,:,:prefix]=True
    for i in range(t):
        mask[0,0,i,prefix]=True
        mask[0,0,i,prefix+i]=True
        if i >= 4:
            mask[0,0,i,prefix+(i-1)//3]=True
    return {
        "q":torch.randn((b,h*7,t,128),generator=generator).half(),
        "k":torch.randn((b,h,n,256),generator=generator).half()[...,::2],
        "v":torch.randn((b,h,n,256),generator=generator).half()[...,1::2],
        "mask":mask,"prefix_tokens":prefix,
        "visual_positions":list(range(9,87)),
    }


def length_sweep_cases(case):
    """Three diagnostic contexts, preserving existing history and tree values.

    Added FP16 text precedes the existing tree and is visible to all queries.
    This tests dynamic lengths/mask strides, not genuine generated text or QA.
    NF includes an aligned value and its successor to expose scalar specialization.
    """
    q,k,v,mask=(case[x] for x in ("q","k","v","mask"))
    t=q.shape[2];history=k.shape[2]-t
    if history < case["prefix_tokens"] or mask.shape[-1] != k.shape[2]:
        raise ValueError("length diagnostic requires a final T-column verification tree")
    nf=k.shape[2]-len(case["visual_positions"])
    offset=(-nf)%16 or 16
    result=[]
    for extra in (0,offset,offset+1):
        generator=torch.Generator().manual_seed(20261002+extra)
        shape=(*k.shape[:2],extra,k.shape[-1])
        new_k=torch.randn(shape,generator=generator).to(k.dtype)
        new_v=torch.randn(shape,generator=generator).to(v.dtype)
        bias=torch.full((*mask.shape[:-1],extra),mask.dtype==torch.bool,dtype=mask.dtype)
        item=dict(case, added_diagnostic_text_tokens=extra)
        item.update(k=torch.cat((k[:,:,:history],new_k,k[:,:,history:]),dim=2),
                    v=torch.cat((v[:,:,:history],new_v,v[:,:,history:]),dim=2),
                    mask=torch.cat((mask[...,:history],bias,mask[...,history:]),dim=-1))
        result.append(item)
    return result


def _jit_cache_keys(kernel):
    # Triton 3.4: each existing device entry starts with the compiled-kernel map.
    # Iterating avoids defaultdict lookup, which would initialize a GPU binder.
    return {(str(device),str(key)) for device,entry in kernel.device_caches.items()
            for key in entry[0]}


def _initialize_binder(kernel,device_index):
    # Triton 3.4 creates self.compile only in create_binder, not __init__.
    # Do this after the caller selects/locks its device, before wrapping compile.
    # No kernel is compiled or launched by this dictionary lookup.
    existed=device_index in kernel.device_caches
    start=time.perf_counter()
    kernel.device_caches[device_index]
    return {"device_index":device_index,"created":not existed,
            "wall_s":time.perf_counter()-start,
            "scope":"Triton device binder setup; no kernel compilation or execution"}


@contextmanager
def _trace_compiler(kernel):
    """Diagnostic-only scoped compiler timing, including disk-cache retrieval."""
    original=kernel.compile
    sentinel=object();previous=kernel.__dict__.get("compile",sentinel)
    records=[]
    def compile_timed(*args,**kwargs):
        start=time.perf_counter();ok=False
        try:
            value=original(*args,**kwargs);ok=True
            return value
        finally:
            records.append({"wall_s":time.perf_counter()-start,"completed":ok})
    kernel.compile=compile_timed
    try:yield records
    finally:
        if previous is sentinel:del kernel.compile
        else:kernel.compile=previous


def _length_sweep(case,device,qg,banks,cpu_banks,transfer,trace):
    kernel=attention_module._mixed
    results=[]
    for item in length_sweep_cases(case):
        partition=partition_prefix(item["prefix_tokens"],item["visual_positions"])
        appended=item["k"].shape[2]-partition.prefix_tokens
        pcpu=prepare_fp16_positions(partition,appended,"cpu")
        fk=item["k"].index_select(2,pcpu.indices)
        fv=item["v"].index_select(2,pcpu.indices)
        # Different contiguous NF also exercises changing FP16-bank head strides;
        # production narrow views are supported by the same dynamic stride API.
        fkg,fvg,mg=(transfer(x) for x in (fk,fv,item["mask"]))
        positions=prepare_fp16_positions(partition,appended,device)
        for mode,bank in banks.items():
            reference=cpu_attention_reference(item["q"],cpu_banks[mode],fk,fv,pcpu,item["mask"])
            torch.cuda.synchronize(device)
            before=_jit_cache_keys(kernel);calls=len(trace)
            start=time.perf_counter()
            output=mixed_attention(qg,bank,fkg,fvg,positions,mg)
            torch.cuda.synchronize(device)
            wall=time.perf_counter()-start
            after=_jit_cache_keys(kernel)
            results.append({"mode":mode,"fp16_tokens":fk.shape[2],
                "logical_mask_columns":item["mask"].shape[-1],
                "mask_strides":list(mg.stride()),"fp16_strides":list(fkg.stride()),
                "added_diagnostic_text_tokens":item["added_diagnostic_text_tokens"],
                "first_length_call_wall_s_including_jit":wall,
                "compiler_invocations":len(trace)-calls,
                "compiler_or_disk_cache_load_wall_s":sum(x["wall_s"] for x in trace[calls:]),
                "new_executor_cache_entries":len(after-before),
                "check":diagnostic(output.cpu(),reference)})
    return {"scope":"same executor, fixed query/tree/visual bank; synthetic extra text; attention call wall excludes input setup and CPU checks",
            "expected_new_specializations":0,
            "passed":all(x["new_executor_cache_entries"]==0 and x["compiler_invocations"]==0 and
                         x["check"]["finite"] and x["check"]["allclose_rtol_0.005_atol_0.005"] for x in results),
            "rows":results}


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input",type=Path)
    source.add_argument("--synthetic",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--physical-gpu",type=int,choices=range(8),required=True)
    parser.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    parser.add_argument("--repeats",type=int,default=10)
    parser.add_argument("--warmup",type=int,default=2)
    args=parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    if not args.device.startswith("cuda"):
        parser.error("GPU probe requires a CUDA device; CPU references are unit-tested separately")
    return args


def _run_impl(args,trace,binder_setup):
    case=synthetic_case() if args.synthetic else torch.load(args.input,map_location="cpu",weights_only=True)
    required={"q","k","v","mask","prefix_tokens","visual_positions"}
    if not required.issubset(case):
        raise ValueError(f"missing input keys: {sorted(required-set(case))}")
    q,k,v,mask=(case[x] for x in ("q","k","v","mask"))
    if any(x.device.type != "cpu" for x in (q,k,v,mask)):
        raise ValueError("input tensors must load on CPU")
    partition=partition_prefix(case["prefix_tokens"],case["visual_positions"])
    appended=k.shape[2]-partition.prefix_tokens
    positions_cpu=prepare_fp16_positions(partition,appended,"cpu")
    fk_cpu=k.index_select(2,positions_cpu.indices)
    fv_cpu=v.index_select(2,positions_cpu.indices)
    cpu_banks={name:pack_visual_cpu(k[:,:,:partition.prefix_tokens],v[:,:,:partition.prefix_tokens],partition,
                                   quantized=name=="int8_visual")
               for name in ("fp16_control","int8_visual")}
    references={name:cpu_attention_reference(q,bank,fk_cpu,fv_cpu,positions_cpu,mask)
                for name,bank in cpu_banks.items()}
    device=torch.device(args.device)
    # Explicit preserved strides: transfer .to() may compact a source view, so make
    # strided destination storage for the real attention inputs, then copy once.
    def transfer_strided(x):
        out=torch.empty_strided(x.shape,x.stride(),device=device,dtype=x.dtype)
        out.copy_(x)
        return out
    qg,kg,vg,mg=(transfer_strided(x) for x in (q,k,v,mask))
    fkg,fvg=(transfer_strided(x) for x in (fk_cpu,fv_cpu))
    positions=prepare_fp16_positions(partition,appended,device)
    banks={}
    pack_times={}
    for name in cpu_banks:
        torch.cuda.synchronize(device)
        start=time.perf_counter()
        bank=pack_visual_cuda(kg[:,:,:partition.prefix_tokens],vg[:,:,:partition.prefix_tokens],partition,
                              quantized=name=="int8_visual")
        torch.cuda.synchronize(device)
        pack_times[name]=time.perf_counter()-start
        if bool(bank.nonfinite_flag.item()):
            raise ValueError("non-finite source KV detected by pack kernel")
        # Setup map integrity check, deliberately outside repeated attention timing.
        if tuple(bank.positions.cpu().tolist()) != bank.host_positions:
            raise ValueError("visual map upload mismatch")
        banks[name]=bank
    if tuple(positions.indices.cpu().tolist()) != positions.host_indices:
        raise ValueError("FP16 map upload mismatch")
    # Validate before benchmarking; failed numerical/pack checks are preserved
    # without running the repeated timing schedule.
    quant_pack={}
    for name,bank in banks.items():
        reference=cpu_banks[name]
        quant_pack[name]={"keys_equal_cpu":bool(torch.equal(bank.keys.cpu(),reference.keys)),
                          "values_equal_cpu":bool(torch.equal(bank.values.cpu(),reference.values))}
        if bank.quantized:
            quant_pack[name]["key_scales_equal_cpu"]=bool(torch.equal(bank.key_scales.cpu(),reference.key_scales))
            quant_pack[name]["value_scales_equal_cpu"]=bool(torch.equal(bank.value_scales.cpu(),reference.value_scales))
    checks={}
    cold_attention_times={}
    for name,bank in banks.items():
        torch.cuda.synchronize(device)
        start=time.perf_counter()
        first=mixed_attention(qg,bank,fkg,fvg,positions,mg)
        torch.cuda.synchronize(device)
        cold_attention_times[name]=time.perf_counter()-start
        checks[name]=diagnostic(first.cpu(),references[name])
    if not (all(x["allclose_rtol_0.005_atol_0.005"] and x["finite"] for x in checks.values()) and
            all(all(x.values()) for x in quant_pack.values())):
        args.out.write_text(json.dumps({"primitive_checks_passed":False,"checks":checks,
            "pack_checks":quant_pack,"timings_attention_only":[],
            "mixed_binder_setup":binder_setup,"mixed_compiler_calls":trace,
            "scope":"failed primitive diagnostic; repeated benchmark not run"},indent=2))
        raise RuntimeError("primitive validation failed before timing; output JSON preserved")
    length_sweep=_length_sweep(case,device,qg,banks,cpu_banks,transfer_strided,trace)
    if not length_sweep["passed"]:
        args.out.write_text(json.dumps({"primitive_checks_passed":False,"checks":checks,
            "pack_checks":quant_pack,"dynamic_length_validation":length_sweep,
            "mixed_binder_setup":binder_setup,"mixed_compiler_calls":trace,"timings_attention_only":[],
            "scope":"failed dynamic-length diagnostic; repeated benchmark not run"},indent=2))
        raise RuntimeError("dynamic-length numerical/cache validation failed; output JSON preserved")
    rows=[]
    outputs={}
    warmup_times={}
    for name,bank in banks.items():
        start=time.perf_counter()
        for _ in range(args.warmup):
            mixed_attention(qg,bank,fkg,fvg,positions,mg)
        torch.cuda.synchronize(device)
        warmup_times[name]=time.perf_counter()-start
    import random
    rng=random.Random(20261002)
    for repeat in range(args.repeats):
        modes=list(banks)
        rng.shuffle(modes)
        for mode in modes:
            begin=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            start=time.perf_counter()
            begin.record(torch.cuda.current_stream(device))
            output=mixed_attention(qg,banks[mode],fkg,fvg,positions,mg)
            end.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            rows.append({"mode":mode,"repeat":repeat,"wall_s":time.perf_counter()-start,
                         "cuda_event_ms":begin.elapsed_time(end)})
            outputs[mode]=output.detach().cpu()
    checks={name:diagnostic(outputs[name],references[name]) for name in banks}
    report={
        "scope":"single attention primitive only; no model/QA/e2e performance conclusion",
        "torch":torch.__version__,"device":str(device),"synthetic":args.synthetic,
        "physical_gpu":args.physical_gpu,
        "input":None if args.input is None else str(args.input.resolve()),
        "shape":{"q":list(q.shape),"kv":list(k.shape)},
        "source_strides":{"q":list(q.stride()),"k":list(k.stride()),"v":list(v.stride())},
        "group_size":32,"page_size":64,"block_m":32,"block_n":64,
        "checks":checks,"pack_checks":quant_pack,
        "dynamic_length_validation":length_sweep,
        "mixed_compiler_calls":trace,
        "mixed_binder_setup":binder_setup,
        "compiler_timing_scope":"Triton compiler entry wall, including disk cache lookup/load; not pure code generation time",
        "quantization_change_vs_fp16_reference":diagnostic(references["int8_visual"],references["fp16_control"]),
        "pack_cold_wall_s_including_jit":pack_times,
        "attention_first_diagnostic_wall_s_including_jit":cold_attention_times,
        "attention_warmup_wall_s_including_jit":warmup_times,
        "timings_attention_only":rows,
        "storage":{name:bank.storage_report() for name,bank in banks.items()},
        "probe_keeps_native_and_both_banks_resident":True,
        "deployment_residency_measured":False,
        "timing_excludes":["input transfer","pack/quantization (separately recorded)","CPU diagnostics","model forward"],
        "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ("visual_kv_int8.py","triton_mixed_attention.py","probe_mixed_attention.py")},
    }
    report["primitive_checks_passed"]=all(x["allclose_rtol_0.005_atol_0.005"] and x["finite"] for x in checks.values()) and all(all(x.values()) for x in quant_pack.values())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if not report["primitive_checks_passed"]:
        raise RuntimeError("primitive diagnostic failed; details preserved in output JSON")
    return report


def _run(args):
    kernel=getattr(attention_module,"_mixed",None)
    if kernel is None:
        raise RuntimeError("Triton is required; no fallback")
    setup=_initialize_binder(kernel,torch.cuda.current_device())
    with _trace_compiler(kernel) as trace:
        return _run_impl(args,trace,setup)


def main(argv=None):
    args=parse_args(argv)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    # Use the same project lock as request runners. No GPU operation occurs
    # until after the lock and CUDA visibility selection.
    import run_joint
    old_visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        with run_joint.cooperative_lock(args.physical_gpu,args.out.parent):
            os.environ["CUDA_VISIBLE_DEVICES"]=str(args.physical_gpu)
            with torch.cuda.device(torch.device(args.device)):
                return _run(args)
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES",None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]=old_visible


if __name__=="__main__":
    main()
