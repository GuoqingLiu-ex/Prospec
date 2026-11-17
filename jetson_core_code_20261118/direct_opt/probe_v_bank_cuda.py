"""Explicit, isolated CUDA numerical gate for partitioned V pack/fill.

Run when root schedules a GPU, for example:
 python probe_v_bank_cuda.py --gpu 6 --expected-gpu-uuid GPU-... --out NEW

Protocol: create a fresh failure-preserving directory; acquire the existing
cooperative flock; check physical UUID, empty compute, explicitly allowlisted
graphics and zero utilization twice, two seconds apart; only then initialize
CUDA and match its UUID. No retry, fallback, process termination or model load.
With explicit --allow-shared-validation, existing compute and nonzero utilization
are recorded and allowed. This correctness-only mode keeps UUID/query/graphics
checks, cooperative flock and a 1024 MiB free-memory guard; it gives no timing,
DRAM or exclusive-residency evidence. The memory guard is not a peak estimate.
All inputs here are synthetic, including the 2059/2016 shape case: the visual
positions are explicit test metadata, not guessed positions of a real prompt.

Both modes compare packed data and reconstructed FP32 operands byte-for-byte
against an independent CPU definition. Error injection directly calls kernels
ONLY to test device guards; it is not a production API or permission to mutate
owned banks. Fresh poisoned fixtures are discarded. Observed executor cache
keys establish compilation facts; source annotations alone are not evidence.

Default: no timing, allocator benchmark, throughput or DRAM claim. Host checks,
readbacks, route upload, flag initialization, JIT and monitoring are diagnostic
work. This module never changes precision flags or CPATH.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
import weakref

HERE = Path(__file__).resolve().parent
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CPATH', 'TRITON_CACHE_DIR',
    'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG',
    'TRITON_OVERRIDE_ARCH', 'TRITON_KERNEL_OVERRIDE', 'TRITON_OVERRIDE_DIR')
OVERRIDE_KEYS = ('TRITON_OVERRIDE_ARCH', 'TRITON_KERNEL_OVERRIDE', 'TRITON_OVERRIDE_DIR')
TRITON_VERSION = '3.4.0'
SHARED_MIN_FREE_BYTES = 1024 * 1024 * 1024
PROTOCOL = dict(schema='partitioned_v_cuda_probe_v1', synthetic_only=True,
    model_loaded=False, timing_measured=False, allocator_peak_measured=False,
    DRAM_measured=False, throughput_claim=False, no_retry=True, no_fallback=True,
    startup='flock -> two idle NVML/UUID guards separated by 2s -> CUDA UUID check',
    numerical='FP16 bytes; independent CPU FP32 RNE INT8 codes/scales; FP32 operand bytes',
    error_injection='direct kernel diagnostic only; not a supported production mutation path',
    compilation='actual executor cache keys before/after each S and identical-pointer replay; '
        'different alignment specializations are retained, not silently counted as length recompiles',
    custom_launch_accounting='one pack/fill custom kernel excludes flag initialization, route upload, '
        'explicit stack/ne/any/item status boundary and host metadata checks')


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def require(value, message):
    if not value: raise RuntimeError(message)


def environment(): return {key: os.environ.get(key) for key in ENV_KEYS}


def compiler_contract(env, installed_version):
    require(not any(env.get(key) for key in OVERRIDE_KEYS), 'Triton architecture/kernel/IR overrides are forbidden')
    require(installed_version==TRITON_VERSION,'installed Triton must be exactly '+TRITON_VERSION)
    return dict(installed_triton_version=installed_version, overrides={key:env.get(key) for key in OVERRIDE_KEYS},
                expected_target=dict(backend='cuda',arch=120,warp_size=32),expected_ptx_targets=['sm_120','sm_120a'])


def source_receipt():
    paths = [Path(__file__), HERE/'v_bank_cuda_ops.py', HERE/'partitioned_v_bank.py',
        HERE/'ring_kv_cache.py', HERE/'run_joint.py', HERE/'run_active_dram_campaign.py',
        HERE.parent/'hardware_metrics.py']
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def flags(torch):
    return dict(allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        reduced_precision_math_sdp=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    parser.add_argument('--expected-gpu-uuid', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-graphics-pid', type=int, choices=(6360,), action='append', default=[])
    parser.add_argument('--allow-shared-validation', action='store_true',
                        help='explicit correctness-only sharing; no performance or DRAM claim')
    args = parser.parse_args(argv)
    try:
        raw = args.expected_gpu_uuid
        if not raw.startswith('GPU-') or str(uuid.UUID(raw[4:])) != raw[4:]:
            raise ValueError('canonical NVML GPU-UUID required')
        if args.out.exists(): raise ValueError('fresh directory required; no retry/resume')
    except ValueError as exc: parser.error(str(exc))
    args.allow_graphics_pid = sorted(set(args.allow_graphics_pid))
    return args


def probe_observation(backend, args):
    from run_active_dram_campaign import observe
    row = observe(backend)
    try: row['gpu_uuid'] = backend.uuid()
    except Exception as exc: row['uuid_error'] = repr(exc)
    if args.allow_shared_validation:
        try: row['memory'] = backend.memory()
        except Exception as exc: row['memory_error'] = repr(exc)
    return row


def validate_observation(row, args, *, startup):
    from run_active_dram_campaign import verify_guard
    require(row.get('gpu_uuid') == args.expected_gpu_uuid, 'monitor UUID/query failure')
    if not args.allow_shared_validation:
        filtered = dict(row, compute=[p for p in row.get('compute', []) if startup or p['pid'] != os.getpid()])
        verify_guard(filtered, args.allow_graphics_pid, require_idle_util=startup)
        return
    require(not any(key.endswith('_error') for key in row), 'Cannot verify shared GPU state: '+str(row))
    require(isinstance(row.get('compute'), list) and isinstance(row.get('graphics'), list),
            'missing shared GPU process observations')
    require(all(p['pid'] in args.allow_graphics_pid for p in row['graphics']), 'unknown graphics process')
    memory = row.get('memory', {})
    used, total = memory.get('device_memory_used_bytes'), memory.get('device_memory_total_bytes')
    require(type(used) is int and type(total) is int and 0 <= used <= total and total > 0,
            'invalid shared GPU memory observation')
    require(total-used >= SHARED_MIN_FREE_BYTES, 'shared GPU has less than 1024 MiB free memory')


def cases():
    return [dict(name='mixed_odd_tail', P=9, visual=(1,3,6), extra=(0,1,7,8,9)),
        dict(name='all_visual_zero_prompt_text', P=3, visual=(0,1,2), extra=(0,1,8)),
        dict(name='synthetic_2059_2016', P=2059, visual=tuple(range(11,2027)), extra=(30,263,505))]


def cpu_values(torch, tokens, offset=0):
    """Exactly representable finite FP16 fixtures with zero groups and signed zero."""
    ids = torch.arange(4*tokens*128, dtype=torch.int64).reshape(1,4,tokens,128)
    value = ((ids*37+offset)%2047-1023).to(torch.float16)/32
    if tokens:
        value[0,0,0,:32] = 0.; value[0,0,0,0] = -0.
        value[0,1,0,:8] = torch.tensor([.5,1.5,2.5,-.5,-1.5,127.,-127.,0.], dtype=torch.float16)
        value[...,32:38] = torch.tensor([2**-24,-2**-24,2**-14,-2**-14,65504.,-65504.],dtype=torch.float16)
    return value


def cpu_pack(torch, values, visual_positions, mode):
    """Independent definition; deliberately does not call the CUDA/reference pack."""
    chosen = set(visual_positions)
    text_positions = [i for i in range(values.shape[2]) if i not in chosen]
    visual = values.index_select(2, torch.tensor(visual_positions, dtype=torch.long))
    text = values.index_select(2, torch.tensor(text_positions, dtype=torch.long))
    if mode == 'fp16_partition': return visual, None, text
    if mode != 'int8_visual': raise ValueError('unknown mode')
    grouped = visual.float().reshape(1,4,len(visual_positions),4,32)
    maximum = grouped.abs().amax(-1)
    scales = torch.where(maximum == 0, torch.ones_like(maximum), maximum/127.)
    codes = torch.round(grouped/scales[...,None]).clamp(-127,127).to(torch.int8)
    return codes.reshape_as(visual), scales, text


def cpu_operand(torch, P, positions, visual, scales, current_text):
    out = torch.empty((1,4,P+current_text.shape[2]-(P-len(positions)),128), dtype=torch.float32)
    visual32 = visual.float()
    if scales is not None:
        visual32 = (visual32.reshape(1,4,len(positions),4,32)*scales[...,None]).reshape_as(visual32)
    mapping = {pos:i for i,pos in enumerate(positions)}; text_index = 0
    for j in range(out.shape[2]):
        if j<P and j in mapping: out[:,:,j].copy_(visual32[:,:,mapping[j]])
        else: out[:,:,j].copy_(current_text[:,:,text_index]); text_index += 1
    return out


def byte_check(torch, actual, expected, label):
    host = actual.detach().cpu().contiguous(); expected = expected.contiguous()
    okay = host.shape == expected.shape and host.dtype == expected.dtype and torch.equal(
        host.view(torch.uint8), expected.view(torch.uint8))
    require(okay, 'byte mismatch: '+label)
    return dict(label=label, byte_equal=True, shape=list(host.shape), dtype=str(host.dtype),
        sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest())


def jit_cache_receipt(kernel, device=0):
    """Read existing Triton 3.4 executor maps; do not create a binder or launch."""
    entry = kernel.device_caches.get(device)
    cache = {} if entry is None else entry[0]
    result = []
    for key, compiled in cache.items():
        ptx = compiled.asm.get('ptx', '')
        meta=compiled.metadata
        target=meta.get('target') if isinstance(meta,dict) else getattr(meta,'target',None)
        target={name:(target.get(name) if isinstance(target,dict) else getattr(target,name,None))
                for name in ('backend','arch','warp_size')}
        require(target==dict(backend='cuda',arch=120,warp_size=32),'compiled metadata target is not CUDA SM120')
        targets=re.findall(r'^\s*\.target\s+(\w+)(?=\s|,|$)',ptx,re.MULTILINE)
        require(len(targets)==1 and targets[0] in ('sm_120','sm_120a'),'compiled PTX target is not sm_120[a]')
        result.append(dict(cache_key=str(key), target=target,ptx_target=targets[0],
                           ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest()))
    return sorted(result, key=lambda x:x['cache_key'])


def _cuda_strided(torch, host, device):
    value = torch.empty_strided(host.shape, host.stride(), dtype=host.dtype, device=device)
    return value.copy_(host)


def run_numerics(torch, device, report, persist, process_check):
    import triton
    require(triton.__version__==TRITON_VERSION,'imported Triton differs from pinned 3.4.0')
    import v_bank_cuda_ops as ops
    from partitioned_v_bank import VisualPartition
    pack_kernel, fill_kernel = ops._get_kernels()
    report['triton_version']=getattr(sys.modules.get('triton'),'__version__',None)
    require(not jit_cache_receipt(pack_kernel) and not jit_cache_receipt(fill_kernel),
            'fresh probe requires empty in-process pack/fill executor caches')
    for case in cases():
        for mode in ops.MODES:
            row = dict(case=case['name'], mode=mode, status='starting', fills=[])
            report['cases'].append(row); persist(); process_check()
            P=case['P']; positions=case['visual']; M=len(positions); R=P-M
            # Noncontiguous head stride, like a logical view into native cache.
            source_cpu = cpu_values(torch,P+5)[:,:,:P]
            visual_cpu, scales_cpu, text_cpu = cpu_pack(torch,source_cpu,positions,mode)
            partition = VisualPartition.prepare(P,positions,provenance={'synthetic_case':case['name']})
            route = ops.prepare_partition_route(partition,device)
            source = _cuda_strided(torch,source_cpu,device)
            source_ref=weakref.ref(source); source_storage=source.untyped_storage().data_ptr()
            bank,text = ops.pack_v_cuda(source,route,mode=mode)
            row['pack_status']=ops.validate_pack_status([bank])
            row['pack_bytes']=[byte_check(torch,bank.visual_values,visual_cpu,'visual'),
                              byte_check(torch,text,text_cpu,'compact prompt text')]
            if scales_cpu is not None:row['pack_bytes'].append(byte_check(torch,bank.scales,scales_cpu,'scales'))
            roots=[bank.visual_values,bank.error_flag,text,route.prefix_route]+([] if bank.scales is None else [bank.scales])
            require(all(t.untyped_storage().data_ptr()!=source_storage for t in roots if t.numel()),
                    'packed data aliases old full V')
            del roots,source;gc.collect()
            require(source_ref() is None,'full source V retained after pack')
            row.update(source_tensor_released=True,old_full_V_storage_unreferenced_by_bank=True,
                storage=bank.storage_receipt(),pack_plan=ops.pack_launch_plan(P,M,mode),
                pack_executor_cache=jit_cache_receipt(pack_kernel))
            require(bool(row['pack_executor_cache']),'no observed compiled pack executor')
            # Caller text backing has a deliberate head gap; never retain a full FP16 V.
            max_extra=max(case['extra'])
            text_backing_cpu=cpu_values(torch,R+max_extra+3,offset=19)
            text_backing_cpu[:,:,:R].copy_(text_cpu)
            text_backing=_cuda_strided(torch,text_backing_cpu,device)
            del text
            for extra in case['extra']:
                S=P+extra; current=text_backing[:,:,:R+extra]
                current_cpu=text_backing_cpu[:,:,:R+extra]
                expected=cpu_operand(torch,P,positions,visual_cpu,scales_cpu,current_cpu)
                out=torch.empty((1,4,S,128),dtype=torch.float32,device=device)
                before=jit_cache_receipt(fill_kernel)
                returned=ops.fill_value_fp32_(out,bank,current)
                require(returned is out,'fill did not return caller operand')
                ops.check_status([bank],stage='after_fill')
                first=byte_check(torch,out,expected,'FP32 operand')
                after=jit_cache_receipt(fill_kernel)
                ops.fill_value_fp32_(out,bank,current)
                ops.check_status([bank],stage='after_identical_pointer_replay')
                replay=byte_check(torch,out,expected,'FP32 replay')
                require(after==jit_cache_receipt(fill_kernel),'identical-pointer replay created a new executor key')
                out_ref=weakref.ref(out);del returned,out;gc.collect()
                require(out_ref() is None,'fill retained caller FP32 operand')
                row['fills'].append(dict(S=S,NF=R+extra,text_stride=list(current.stride()),
                    comparisons=[first,replay],caller_operand_released=True,
                    executor_keys_before=before,executor_keys_after=after,
                    same_pointer_replay_cache_reused=True,fill_plan=ops.fill_launch_plan(P,M,S,mode)))
                persist();process_check()
            row.update(status='passed',dynamic_S_cache_snapshots_recorded=True,
                dynamic_S_new_executor_counts=[len(x['executor_keys_after'])-len(x['executor_keys_before']) for x in row['fills']],
                cache_scope='Observed in-process compiled-key sets. Extra alignment/type specializations are preserved; '
                    'identical pointer/shape replay must reuse. No assumption all different S share one binary.')
            del current,text_backing,bank,route;persist()
    error_diagnostics(torch,device,report,persist,process_check)


def error_diagnostics(torch,device,report,persist,process_check):
    """Intentional raw kernel calls: test guards without mutating production metadata."""
    import v_bank_cuda_ops as ops
    from partitioned_v_bank import VisualPartition
    _,fill=ops._get_kernels()
    for mode in ops.MODES:
        kinds=('invalid_prefix_route','nonfinite_source','nonfinite_text')
        if mode=='int8_visual':kinds+=('invalid_scale',)
        for kind in kinds:
            row=dict(mode=mode,kind=kind,direct_kernel_diagnostic=True,status='starting')
            report['error_diagnostics'].append(row);persist();process_check()
            route=ops.prepare_partition_route(VisualPartition.prepare(9,(1,3,6)),device)
            source=cpu_values(torch,9).to(device)
            if kind=='nonfinite_source':source[0,0,0,0]=float('nan')
            bank,text=ops.pack_v_cuda(source,route,mode=mode)
            if kind=='nonfinite_source':
                try:ops.validate_pack_status([bank])
                except RuntimeError:pass
                else:raise RuntimeError('nonfinite pack was incorrectly accepted')
                require(bank.failed and not bank.validated,'nonfinite bank not poisoned')
                row.update(status='passed',error_flag=int(bank.error_flag.item()),pack_rejected=True)
                require(row['error_flag']&2,'nonfinite pack missing error bit')
                persist();continue
            ops.validate_pack_status([bank])
            current=torch.cat((text,cpu_values(torch,5).to(device)),2)
            bad_route=route.prefix_route.clone()
            kernel_scales=bank.scales if bank.scales is not None else bank.visual_values
            if kind=='invalid_prefix_route':bad_route[0]=-route.R-1
            elif kind=='invalid_scale':
                kernel_scales=bank.scales.clone();kernel_scales[0,0,0,0]=0.
            else:current[0,0,0,0]=float('inf')
            out=torch.empty((1,4,14,128),device=device,dtype=torch.float32)
            def raw(route_tensor):
                fill[(4,2)](bank.visual_values,kernel_scales,
                    current,route_tensor,bank.error_flag,out,9,3,6,14,11,*current.stride()[1:],
                    QUANT=mode=='int8_visual',BLOCK=8,num_warps=4,num_stages=1,enable_fp_fusion=False)
            raw(bad_route)
            bad_column=1 if kind=='invalid_scale' else 0
            require(bool(torch.isnan(out[0,0,bad_column]).all() if kind=='invalid_prefix_route'
                         else torch.isnan(out[0,0,bad_column,0])),'bad lane not marked NaN')
            require(int(bank.error_flag.item())&4,'fill missing persistent error bit')
            # Restore raw inputs only; the persistent flag MUST still poison the next fill.
            if kind=='nonfinite_text':current[0,0,0,0]=0.
            if kind=='invalid_scale':kernel_scales=bank.scales
            raw(route.prefix_route)
            require(bool(torch.isnan(out).all()),'persistent error did not poison later fill')
            try:ops.check_status([bank],stage='intentional_raw_kernel_guard_diagnostic')
            except RuntimeError:pass
            else:raise RuntimeError('device guard error incorrectly accepted')
            try:ops.fill_value_fp32_(out,bank,current)
            except RuntimeError:pass
            else:raise RuntimeError('production API reused a poisoned bank')
            row.update(status='passed',error_flag=int(bank.error_flag.item()),persistent_nan=True,
                       explicit_boundary_rejected=True,production_reuse_rejected=True)
            persist()


class ProcessMonitor:
    """Kind-preserving NVML log; no GPU work, changes or process termination."""
    def __init__(self,backend,args,path):
        self.backend=backend;self.args=args;self.path=path;self.rows=[]
        self.event=threading.Event();self.thread=None;self._result=None
    def sample(self):
        row=probe_observation(self.backend,self.args)
        self.rows.append(row)
        with self.path.open('a') as stream:stream.write(json.dumps(row)+'\n')
    def start(self):
        self.sample()
        def run():
            while not self.event.wait(.5):
                try:self.sample()
                except BaseException as exc:self.rows.append({'monitor_error':repr(exc)});return
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def check(self):
        for row in self.rows:
            validate_observation(row,self.args,startup=False)
    def stop(self):
        if self._result is not None:return self._result
        self.event.set()
        if self.thread:self.thread.join()
        self.sample()
        self._result=dict(sample_count=len(self.rows),interval_s=.5,observations=self.rows,
            scope='Discrete kind-preserving process observations; not proof of exclusive residency')
        return self._result


def run_probe(args,*,torch_module=None,lock_factory=None,backend_factory=None,
              workload=run_numerics,monitor_factory=ProcessMonitor,sleep=time.sleep,
              version_reader=importlib.metadata.version):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(PROTOCOL,status='starting',pid=os.getpid(),started=time.time(),
        gpu=args.gpu,expected_gpu_uuid=args.expected_gpu_uuid,allowed_graphics=args.allow_graphics_pid,
        guards=[],cases=[],error_diagnostics=[],environment_before=environment())
    report['allow_shared_validation']=args.allow_shared_validation
    report['resource_mode']='shared_device_correctness_only' if args.allow_shared_validation else 'idle_device_numerical_gate'
    if args.allow_shared_validation:
        report.update(startup='flock -> two shared NVML/UUID/memory guards separated by 2s -> CUDA UUID check',
            minimum_free_memory_bytes=SHARED_MIN_FREE_BYTES,
            memory_guard_scope='engineering resource guard, not an allocator peak or capacity guarantee')
    def persist():save(out/'summary.json',report)
    persist();torch=torch_module;backend=None;monitor=None
    old_visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        report['source_sha256']=source_receipt();persist()
        report['installed_triton_version']=version_reader('triton')
        report['compiler_contract']=compiler_contract(report['environment_before'],report['installed_triton_version'])
        persist()
        if torch is None:import torch
        require(not torch.cuda.is_initialized(),'CUDA initialized before lock/idle device checks')
        report['startup_math_flags']=flags(torch)
        if lock_factory is None:
            from run_joint import cooperative_lock
            lock_factory=cooperative_lock
        if backend_factory is None:
            sys.path.insert(0,str(HERE.parent))
            from hardware_metrics import _make_backend
            backend_factory=_make_backend
        with lock_factory(args.gpu,out) as lock:
            report['lock']=dict(lock);backend=backend_factory(args.gpu)
            for i in range(2):
                row=probe_observation(backend,args);row.update(index=i)
                report['guards'].append(row);persist()
                require(row['gpu_uuid']==args.expected_gpu_uuid,'NVML physical GPU UUID mismatch')
                validate_observation(row,args,startup=True)
                if i==0:sleep(2)
            os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
            report['effective_environment']=environment()
            props=torch.cuda.get_device_properties(0)
            actual_uuid='GPU-'+str(uuid.UUID(str(props.uuid).removeprefix('GPU-')))
            require(actual_uuid==args.expected_gpu_uuid,'CUDA logical device and NVML UUID mismatch')
            require((props.major,props.minor)==(12,0),'this candidate gate requires SM120')
            report.update(torch=torch.__version__,cuda_device=dict(uuid=actual_uuid,name=props.name,
                capability=[props.major,props.minor]),actual_math_flags_before=flags(torch))
            monitor=monitor_factory(backend,args,out/'process_observations.jsonl');monitor.start()
            with torch.cuda.device(0),torch.inference_mode():
                try:
                    workload(torch,'cuda:0',report,persist,monitor.check)
                finally:
                    # Do not release the cooperative lock with our own queued
                    # kernels still running after a diagnostic failure.
                    torch.cuda.synchronize()
            require(len(report['cases'])==6 and all(r.get('status')=='passed' for r in report['cases']),
                    'incomplete six-case numerical gate')
            require(len(report['error_diagnostics'])==7 and all(r.get('status')=='passed' for r in report['error_diagnostics']),
                    'incomplete seven-case device-guard diagnostics')
            report['process_monitor']=monitor.stop();monitor.check();monitor=None
            report['actual_math_flags_after']=flags(torch)
            require(report['actual_math_flags_after']==report['startup_math_flags'],'math flags changed')
            require(source_receipt()==report['source_sha256'],'executed source changed during probe')
            report.update(status='completed',numerical_gate_passed=True)
    except BaseException as exc:
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if monitor is not None:
            try:report['process_monitor']=monitor.stop()
            except BaseException as exc:report['monitor_stop_error']=repr(exc)
        if backend is not None:
            try:backend.close()
            except BaseException as exc:report.update(status='failed',backend_close_error=repr(exc))
        if old_visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=old_visible
        if torch is not None and 'startup_math_flags' in report:report['exit_math_flags']=flags(torch)
        report.update(environment_after=environment(),finished=time.time())
        if report['environment_after']!=report['environment_before']:
            report.update(status='failed',environment_error='unexpected inherited environment change')
        persist()
    require(report['status']=='completed','probe failed finalization')
    return report


if __name__=='__main__':run_probe(parse_args())
