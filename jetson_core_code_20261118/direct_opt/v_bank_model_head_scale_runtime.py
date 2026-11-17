"""Request-local fixed-V8 head-scale proxy collection; no model or precision selector.

Only the pack operation differs. Base policy remains production CUDA with
reference_only=False. CPU tests replace private device helpers explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import time
import weakref

import torch
import v_bank_cuda_ops as frozen_ops
import v_head_scale_cuda_ops as head_ops
from v_bank_model_runtime import v_bank_model_policy as _base_policy

MODE='fixed_int8_pack_head_scale'
LAYERS=28
_ACTIVE=False


def _allocate_score_slab(device,M):
    if torch.device(device).type!='cuda':raise ValueError('model head-scale diagnostic requires CUDA; no CPU fallback')
    with torch.inference_mode(False):
        return torch.zeros((LAYERS,4),device=device,dtype=torch.float32)


def _copy_scores_to_cpu(slab):
    # One blocking data transfer for all layers; no per-layer .cpu() or .item().
    return slab.detach().to(device='cpu',copy=True,non_blocking=False)


def _slab_metadata(slab):
    return (id(slab),tuple(slab.shape),tuple(slab.stride()),str(slab.device),str(slab.dtype),
            slab._version,slab.storage_offset(),slab.untyped_storage().data_ptr())


def _file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


class HeadScalePackFacade:
    """Explicit frozen delegates plus a single request diagnostic allocation."""
    prepare_partition_route=staticmethod(frozen_ops.prepare_partition_route)
    fill_value_fp32_=staticmethod(frozen_ops.fill_value_fp32_)
    check_status=staticmethod(frozen_ops.check_status)

    def __init__(self,*,cell,source_sha256,pack_validation):
        self.slab=None;self.cpu_scores=None;self.route=None;self.bank_ids=[];self.snapshot=None
        self.slab_ref=None;self.failed=False;self.closed=False
        self.record=dict(schema='model_visual_v_pack_head_scale_v1',diagnostic_mode=MODE,
            cell=copy.deepcopy(cell),source_sha256=dict(source_sha256),pack_validation=copy.deepcopy(pack_validation),
            pack_calls=0,aggregate_status_calls=0,d2h_calls=0,collection_complete=False,
            GPU_slab_released=False,CPU_array_released=False,slab_payload_bytes=0,slab_backing_bytes=0,
            allocation_wall_s=0.,pack_wall_s=0.,status_d2h_validation_wall_s=0.,artifact_write_hash_wall_s=0.,
            artifact=None,selector_implemented=False,task_quality_claim=False,actual_DRAM_measured=False,
            allocation_count=0,zero_initialization_count=0,overlay_implemented=False,failures=[],
            storage_scope='one temporary diagnostic slab counted once, separate from persistent V-bank owned-storage tables',
            timing_scope='allocation/pack/status/D2H/CPU checks inside conversion; artifact write/hash inside request finalization',
            disk_scope='diagnostic .pt writing/hash included in E2E; parent JSON/status serialization remains outside')
        json.dumps(self.record,allow_nan=False)

    def _check(self):
        if self.failed or self.closed:raise RuntimeError('head-scale diagnostic is failed or closed')
        if self.slab is None or _slab_metadata(self.slab)!=self.snapshot:
            raise RuntimeError('diagnostic slab identity/layout/version changed')
        self.route.validate_metadata()

    def _fail(self,stage,exc):
        self.failed=True;self.record['failures'].append(dict(stage=stage,error=repr(exc)))
        self.release_gpu()

    def release_gpu(self):
        self.slab=None;self.route=None;self.snapshot=None
        self.record['GPU_slab_released']=self.slab_ref is None or self.slab_ref() is None

    def pack_v_cuda(self,values,route,*,mode='int8_visual'):
        start=time.perf_counter()
        try:
            if mode!='int8_visual' or self.failed or self.closed or self.record['collection_complete']:
                raise RuntimeError('fixed INT8 head-scale pack diagnostic unavailable/reused')
            index=self.record['pack_calls']
            if index>=LAYERS:raise RuntimeError('more than 28 diagnostic packs')
            if index==0:
                route.validate_metadata();allocation=time.perf_counter()
                self.slab=_allocate_score_slab(values.device,route.M)
                self.slab_ref=weakref.ref(self.slab);self.route=route;self.snapshot=_slab_metadata(self.slab)
                if (self.slab.shape!=(LAYERS,4) or self.slab.dtype!=torch.float32
                        or self.slab.device!=values.device or not self.slab.is_contiguous() or self.slab.requires_grad):
                    raise RuntimeError('diagnostic allocator returned invalid slab')
                self.record.update(allocation_count=1,zero_initialization_count=1,allocation_wall_s=time.perf_counter()-allocation,
                    slab_payload_bytes=self.slab.numel()*4,slab_backing_bytes=self.slab.untyped_storage().nbytes(),
                    shape=list(self.slab.shape),layer_indices=list(range(LAYERS)),kv_head_indices=list(range(4)),
                    layer_index_source='frozen prefill enumerates actual cache.layers once in index order',
                    visual_row_to_prompt_position=list(route.partition.visual),host_route=list(route.partition.host_route),
                    partition=route.partition.receipt(),pack_plan_per_layer=head_ops.pack_head_scale_launch_plan(route.P,route.M),
                    d2h_logical_payload_bytes=self.slab.numel()*4,
                    total_head_score_atomic_operations=LAYERS*4*route.M,
                    score_rule='max valid group FP32 scale/2, zero-group contribution0; atomic_max over visual rows',
                    score_semantics='one zero initialization, no per-layer clearing; no selected head or overlay',
                    extra_atomic_and_reduction_costs_measured=False)
            self._check()
            if route is not self.route or values.device!=self.slab.device:
                raise RuntimeError('every layer must share the same actual route/device')
            bank,text=head_ops.pack_v_with_head_scale(values,route,self.slab[index],mode=mode)
            if type(bank) is not frozen_ops.CudaVisualVBank or bank.route is not route or bank.mode!='int8_visual':
                raise RuntimeError('pack did not return the frozen V8 bank for this route')
            if id(bank) in self.bank_ids:raise RuntimeError('pack reused an earlier layer bank')
            self.bank_ids.append(id(bank));self.record['pack_calls']+=1
            self._check()
            return bank,text
        except BaseException as exc:
            self._fail('pack',exc);raise
        finally:self.record['pack_wall_s']+=time.perf_counter()-start

    def validate_pack_status(self,banks):
        start=time.perf_counter()
        try:
            self._check();banks=tuple(banks)
            if (self.record['pack_calls']!=LAYERS or self.record['aggregate_status_calls']
                    or len(banks)!=LAYERS or [id(bank) for bank in banks]!=self.bank_ids
                    or any(bank.route is not self.route for bank in banks)):
                raise RuntimeError('complete ordered 28-layer coverage required before collection')
            receipt=frozen_ops.validate_pack_status(banks)
            self.record['aggregate_status_calls']+=1
            self._check()
            self.cpu_scores=_copy_scores_to_cpu(self.slab);self.record['d2h_calls']+=1
            self.release_gpu()
            cpu=self.cpu_scores
            if (cpu.device.type!='cpu' or cpu.dtype!=torch.float32 or list(cpu.shape)!=self.record['shape']
                    or not cpu.is_contiguous() or cpu.requires_grad or not bool(torch.isfinite(cpu).all())
                    or bool((cpu<0).any())):
                raise RuntimeError('copied head-score array is malformed, nonfinite or negative')
            if not self.record['GPU_slab_released']:raise RuntimeError('GPU score slab still retained after D2H')
            self.record['collection_complete']=True
            return receipt
        except BaseException as exc:
            self.cpu_scores=None;self._fail('status_collect',exc);raise
        finally:self.record['status_d2h_validation_wall_s']+=time.perf_counter()-start

    def close(self):
        self.release_gpu();self.closed=True

    def write_artifact(self,path,*,request_successful):
        if self.record['artifact'] is not None or self.cpu_scores is None:return
        path=Path(path);partial=path.with_suffix(path.suffix+'.partial')
        start=time.perf_counter()
        try:
            if not self.record['collection_complete'] or self.failed:raise RuntimeError('invalid diagnostic cannot be saved as collected')
            if path.exists() or partial.exists():raise FileExistsError('fresh diagnostic artifact required')
            path.parent.mkdir(parents=True,exist_ok=True)
            metadata=copy.deepcopy(self.record)
            metadata.update(request_successful=bool(request_successful),artifact_scope='actual model visual V head rounding-scale proxies; no source V, selector or QA-risk labels',
                score_arithmetic='FP32 max(where(group maxabs>0, scale, 0))/2; atomic_max accumulation',
                statistic_scope='one scalar per layer and KV head over actual visual partition; not per-token reconstruction error')
            with partial.open('xb') as stream:torch.save(dict(head_scores_fp32=self.cpu_scores,metadata=metadata),stream)
            partial.replace(path)
            self.record['artifact']=dict(path=str(path.resolve()),sha256=_file_sha(path),bytes=path.stat().st_size,
                tensor_key='head_scores_fp32',shape=list(self.cpu_scores.shape),dtype='torch.float32',request_successful=bool(request_successful))
        finally:
            self.cpu_scores=None;self.record['CPU_array_released']=True
            self.record['artifact_write_hash_wall_s']+=time.perf_counter()-start


class DiagnosticPolicy:
    def __init__(self,state,facade,artifact_path):
        self.state,self.facade,self.artifact_path=state,facade,Path(artifact_path)
        self.record=state.record

    def validate_result(self,result):
        self.state.validate_result(result)
        if (self.facade.failed or not self.facade.record['collection_complete']
                or self.facade.record['pack_calls']!=28 or self.facade.record['d2h_calls']!=1):
            raise RuntimeError('model result lacks a complete head-score collection')
        result.precision=dict(result.precision,diagnostic_mode=MODE,
            diagnostic_changes_fixed_v8_storage_format=False,diagnostic_base_v_storage='int8_visual',
            diagnostic_statistics_are_task_risk=False)

    def summary(self):
        info=self.state.summary()
        if not info.get('finalized'):raise RuntimeError('diagnostic summary requires policy finish/release')
        success=info.get('result_coverage_validated',False) and not info.get('failures')
        try:self.facade.write_artifact(self.artifact_path,request_successful=success)
        except BaseException as exc:
            self.facade.record['failures'].append(dict(stage='artifact',error=repr(exc)))
            if success:raise
        info['pack_head_scale_diagnostic']=copy.deepcopy(self.facade.record)
        return info


@contextmanager
def v_bank_head_scale_policy(target,mode='int8_visual',*,artifact_path,cell,source_sha256,pack_validation,
                        interface=None,bridge=None):
    global _ACTIVE
    if _ACTIVE:raise RuntimeError('nested pack-head-scale diagnostic policy')
    if mode!='int8_visual':raise ValueError('pack head-scale diagnostic requires fixed int8_visual')
    facade=HeadScalePackFacade(cell=cell,source_sha256=source_sha256,pack_validation=pack_validation)
    _ACTIVE=True
    try:
        with _base_policy(target,mode=mode,interface=interface,bridge=bridge) as state:
            if state.reference_only or state.ops is not frozen_ops:
                raise RuntimeError('diagnostic requires genuine production CUDA policy, no reference/fallback')
            old_ops=state.ops;state.ops=facade
            try:yield DiagnosticPolicy(state,facade,artifact_path)
            finally:
                facade.close();state.ops=old_ops
    finally:
        facade.close();_ACTIVE=False
