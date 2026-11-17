"""Shared-GPU exploratory comparison of two visual acceleration methods.

All speculative arms use the same frozen bridge, draft, and tree budget.
No shared-device energy reading is attributed to this process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import dataclasses
import datetime as dt
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
from scheme_context import SCHEMES, SchemeRuntime
OLD_RUN = Path('/data1/lgq2026/jev_spec/runs/decoder_vispec_mechanism_cursor1_20261001')
TARGET = '/data1/lgq2026/jev_spec/assets/models/Qwen2.5-VL-7B-Instruct'
ARMS = {name: (v['depth'], v['ratio']) for name, v in SCHEMES.items()}



def timestamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


@contextmanager
def cooperative_lock(gpu, out):
    """Honor the project's process lock while explicitly allowing shared hardware.

    Existing idle-only experiments see this lock and cannot run over this one.
    Other users' workloads are observed, never changed.
    """
    lock_path = Path('/data1/lgq2026/jev_spec/locks') / f'gpu{gpu}.lock'
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(OLD_RUN / 'code'))
        import gpu_lock
        gpu_lock._check_legacy_metadata(fd)
        snapshot, processes = gpu_lock._gpu_status(gpu)
        metadata = dict(protocol='jev-flock-v1', gpu=gpu, pid=os.getpid(),
                        owner='training_free_visual_schemes', task=str(out),
                        observation_scope='shared_gpu_exploratory',
                        initial_snapshot=snapshot, initial_processes=processes,
                        ts=timestamp())
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(metadata, indent=2) + '\n').encode())
        yield metadata
        os.ftruncate(fd, 0)
    finally:
        os.close(fd)


class StageTimes:
    """A few CUDA events; no stage synchronization or per-layer profiling."""
    def __init__(self, torch):
        self.torch = torch
        self.events = {}
        self.handles = []
        self.merge_wall_ms = 0.0

    def attach(self, module, name):
        pending = []
        def before(_module, _args):
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            pending.append(event)
        def after(_module, _args, _output):
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.setdefault(name, []).append((pending.pop(), event))
        self.handles.extend([module.register_forward_pre_hook(before),
                             module.register_forward_hook(after)])

    @contextmanager
    def merge(self, joint):
        original = joint.prepare_draft_visual_context
        def wrapped(*args, **kwargs):
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
            start = time.perf_counter()
            output = original(*args, **kwargs)
            self.merge_wall_ms += (time.perf_counter() - start) * 1000
            end_event.record()
            self.events.setdefault('merge', []).append((start_event, end_event))
            return output
        joint.prepare_draft_visual_context = wrapped
        try:
            yield
        finally:
            joint.prepare_draft_visual_context = original

    def values(self):
        result = {name + '_ms': sum(a.elapsed_time(b) for a, b in events)
                  for name, events in self.events.items()}
        result['merge_wall_ms'] = self.merge_wall_ms
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


class NativeDepthPolicy:
    """Baseline leaves the native visual forward and all blocks untouched."""
    def __init__(self, target):
        self.depth = len(target.model.visual.blocks)
        self.grid = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def prepare(self, grid, device=None):
        self.grid = grid

    def summary(self):
        grid = self.grid.detach().cpu().tolist()
        tokens = sum(t*h*w for t,h,w in grid)
        return dict(mode='native', calls=[], original_token_blocks=tokens*self.depth,
                    executed_token_blocks=tokens*self.depth, token_block_reduction=0.0)


def make_schedule(requests, arms, repeats, warmup, seed, include_ar):
    schedule = []
    for index in range(warmup):
        for arm in arms:
            schedule.append(dict(request_id=requests[0]['request_id'], arm=arm,
                                 phase='warmup', repeat=index))
    if include_ar:
        schedule.extend(dict(request_id=r['request_id'], arm='ar_reference',
                             phase='reference', repeat=0) for r in requests)
    rng = random.Random(seed)
    for repeat in range(repeats):
        req_order = list(requests)
        rng.shuffle(req_order)
        for request in req_order:
            order = list(arms)
            rng.shuffle(order)
            schedule.extend(dict(request_id=request['request_id'], arm=arm,
                                 phase='measure', repeat=repeat) for arm in order)
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--physical-gpu', type=int, required=True, choices=range(8))
    parser.add_argument('--requests', default=str(OLD_RUN / 'inputs/diagnostic4.jsonl'))
    parser.add_argument('--request-id', action='append')
    parser.add_argument('--arms', nargs='+', choices=list(ARMS),
                        default=['baseline', 'reorder', 'reorder_fp32', 'attention50', 'difference50', 'response50'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--include-ar', action='store_true')
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--thresholds', type=str, default=None,
                        help='JSON file recording predeclared adaptive thresholds')
    parser.add_argument('--anchor-stride', type=int, default=4)
    parser.add_argument('--skip-layers', type=int, nargs=2, default=[28, 29],
                        metavar=('FIRST', 'SECOND'))
    parser.add_argument('--prefix-layer', type=int, default=None)
    parser.add_argument('--local-thresholds')
    parser.add_argument('--difference-threshold', type=float)
    parser.add_argument('--collect-local', action='store_true')
    parser.add_argument('--capture-visual-dir')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if any(SCHEMES[a]['compression'] == 'difference_gate' for a in args.arms) and args.difference_threshold is None:
        parser.error('difference_gate requires a development-calibrated --difference-threshold')
    if args.prefix_layer is None:
        args.prefix_layer = args.skip_layers[0] - 1
    if any(SCHEMES[a]['depth'] == 'local' for a in args.arms) and (args.prefix_layer != 27 or tuple(args.skip_layers) != (28, 29)):
        parser.error('local depth fixes prefix 27 and candidate blocks 28/29')
    if any(SCHEMES[a]['depth'] == 'adaptive' for a in args.arms) and not args.thresholds:
        parser.error('adaptive comparison requires explicit frozen --thresholds')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    requests = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    if args.request_id:
        requests = [r for r in requests if r['request_id'] in args.request_id]
    if not requests:
        raise ValueError('no matching requests')
    thresholds = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    local_thresholds = json.loads(Path(args.local_thresholds).read_text()) if args.local_thresholds else None
    source_sha256 = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts}
    request_sha256 = hashlib.sha256(Path(args.requests).read_bytes()).hexdigest()
    if args.resume and (out / 'status.json').exists():
        old_status = json.loads((out / 'status.json').read_text())
        same_args = lambda values: {k:v for k,v in values.items() if k != 'resume'}
        if (same_args(old_status['args']) != same_args(vars(args)) or
                old_status.get('thresholds') != thresholds or
                old_status.get('local_thresholds') != local_thresholds or
                old_status.get('source_sha256') != source_sha256 or
                old_status.get('request_sha256') != request_sha256):
            raise ValueError('Resume requires identical arguments, thresholds, requests, and code; use a new output directory')
    schedule = make_schedule(requests, args.arms, args.repeats, args.warmup,
                             args.seed, args.include_ar)
    completed = set()
    results_path = out / 'results.jsonl'
    if args.resume and results_path.exists():
        completed = {(r['request_id'], r['arm'], r['phase'], r['repeat'])
                     for r in map(json.loads, results_path.read_text().splitlines())}
    status = dict(status='starting', started_at=timestamp(), pid=os.getpid(),
                  args=vars(args), total_cells=len(schedule), completed_cells=len(completed),
                  target=TARGET, draft=str(OLD_RUN / 'models/ViSpec-Qwen2.5-VL-7B-Instruct'),
                  thresholds=thresholds, local_thresholds=local_thresholds, schemes=SCHEMES, methodology='server_gpu_exploratory',
                  source_sha256=source_sha256, request_sha256=request_sha256,
                  quality_claim='No original-Target losslessness claim after visual skipping.',
                  adaptive_claim='Prefix-state heuristic; final QA error prediction unvalidated.')
    save_json(out / 'status.json', status)
    save_json(out / 'schedule.json', schedule)
    save_json(out / 'requests.json', requests)
    try:
        with cooperative_lock(args.physical_gpu, out) as lock:
            status['lock'] = lock
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),
                              TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            sys.path.insert(0, str(HERE / 'vendor'))
            import torch
            import cv2
            import transformers
            import hf517_vispec_adapter as bridge
            import vispec_joint_adapter as joint
            from vispec_operators import new_tree_budget
            from vision_depth import depth_policy
            from local_depth import local_depth_policy
            from hardware_metrics import HardwareSampler
            bridge._native()
            from jev_spec.runtime import prepare_request
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
            start = time.perf_counter()
            target = bridge.load_target(TARGET, 'cuda:0', dtype='float16',
                                       output_head_dtype='float32', target_sdpa_backend='math')
            processor = target._hf517_processor
            spec = bridge.load_spec(target, status['draft'], total_token=48, depth=3,
                                    top_k=8, num_q=2, capacity=4096)
            budget = new_tree_budget(30, 3, 8)
            setup_start = time.perf_counter()
            runtime = SchemeRuntime(spec.spec_layer, difference_threshold=args.difference_threshold)
            torch.cuda.synchronize()
            status['experimental_setup_seconds'] = time.perf_counter() - setup_start
            capture_cache = {}
            if args.capture_visual_dir:
                capture_dir = Path(args.capture_visual_dir)
                capture_dir.mkdir(parents=True, exist_ok=True)
                a = spec.spec_layer.imadpt
                torch.save({'state_dict': a.state_dict(), 'hidden_size': a.hidden_size,
                            'num_heads': a.num_heads, 'num_q': a.num_q}, capture_dir / 'adaptor.pt')
            torch.cuda.synchronize()
            status.update(status='running', load_seconds=time.perf_counter()-start,
                          torch=torch.__version__, transformers=transformers.__version__,
                          device_name=torch.cuda.get_device_name(0))
            save_json(out / 'status.json', status)
            request_map = {r['request_id']: r for r in requests}
            with results_path.open('a') as writer:
                for index, cell in enumerate(schedule):
                    key = (cell['request_id'], cell['arm'], cell['phase'], cell['repeat'])
                    if key in completed:
                        continue
                    status.update(current=cell, current_index=index, updated_at=timestamp())
                    save_json(out / 'status.json', status)
                    mode, ratio = ARMS[cell['arm']]
                    captured = []
                    capture_handle = None
                    if args.capture_visual_dir and cell['arm'] == 'baseline' and cell['request_id'] not in capture_cache:
                        def capture_hook(_module, inputs):
                            if not captured:
                                captured.append(inputs[0].detach().clone())
                        capture_handle = spec.spec_layer.imadpt.register_forward_pre_hook(capture_hook)
                    spec.spec_layer.reset_kv()
                    spec.spec_layer.last_img_hidden = None
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    timer = StageTimes(torch)
                    timer.attach(target.model.visual, 'vision')
                    timer.attach(spec.spec_layer.imadpt, 'adaptor')
                    sampler = HardwareSampler(device_index=args.physical_gpu, sample_interval_s=0.1,
                                              process_interval_s=0.5, known_shared=False)
                    sampler.start()
                    started = timestamp()
                    start = time.perf_counter()
                    try:
                        policy_context = (local_depth_policy(target, thresholds=local_thresholds,
                                          anchor_stride=args.anchor_stride, collect_only=args.collect_local) if mode == 'local' else
                                          NativeDepthPolicy(target) if mode == 'fixed0' else
                                          depth_policy(target, mode=mode, thresholds=thresholds,
                                                       anchor_stride=args.anchor_stride,
                                                       skip_layers=args.skip_layers,
                                                       prefix_layer=args.prefix_layer))
                        with torch.inference_mode(), policy_context as policy, runtime.activate(joint, SCHEMES[cell['arm']]) as scheme_state, timer.merge(joint):
                            packed, sampled = prepare_request(processor, request_map[cell['request_id']], 'cuda:0')
                            prepared_s = time.perf_counter() - start
                            policy.prepare(packed['video_grid_thw'], device=target.model.visual.device)
                            torch.cuda.synchronize()
                            model_start = time.perf_counter()
                            if cell['arm'] == 'ar_reference':
                                result = bridge.generate_ar(target, packed, max_new_tokens=args.max_new_tokens,
                                                            eos_token_ids=[151645, 151643])
                            else:
                                result = joint.generate_joint(
                                    spec, packed, max_new_tokens=args.max_new_tokens,
                                    visual_m=ratio, tree_budget=budget, merge_operator='cosine',
                                    context_extractor='new', diagnostic_actual_imadpt=False,
                                    eos_token_ids=[151645, 151643])
                            output_text = processor.tokenizer.decode(result.token_ids, skip_special_tokens=True)
                            torch.cuda.synchronize()
                            end = time.perf_counter()
                            hardware = sampler.stop()
                            depth_stats = policy.summary()
                        peak_allocated = torch.cuda.max_memory_allocated()
                        peak_reserved = torch.cuda.max_memory_reserved()
                        stages = timer.values()
                        rounds = [r for r in result.rounds if r.get('kind') == 'tree']
                        effective = [sum(token not in {151645, 151643} for token in r['submitted'][1:]) for r in rounds]
                        record = dict(**cell, started_at=started, physical_gpu=args.physical_gpu,
                                      depth_mode=mode, visual_keep_ratio=ratio, scheme=SCHEMES[cell['arm']],
                                      compression=scheme_state['compression'],
                                      capture_enabled=bool(args.capture_visual_dir),
                                      effective_draft_tokens=sum(effective),
                                      mean_effective_draft_per_verify=sum(effective)/len(effective) if effective else None,
                                      e2e_wall_s=end-start, model_wall_s=end-model_start,
                                      prepare_wall_s=prepared_s, **stages,
                                      output_token_ids=result.token_ids, num_tokens=len(result.token_ids),
                                      text=output_text, stop_reason=result.stop_reason,
                                      truncated=result.stop_reason == 'limit',
                                      mean_accepted_draft=sum(r['accepted_draft'] for r in rounds)/len(rounds) if rounds else None,
                                      mean_committed_per_verify=sum(len(r['submitted']) for r in rounds)/len(rounds) if rounds else None,
                                      verify_rounds=len(rounds), stats=depth_stats,
                                      hardware=hardware, peak_allocated_bytes=peak_allocated,
                                      peak_reserved_bytes=peak_reserved,
                                      input_shapes={k:list(v.shape) for k,v in packed.items() if torch.is_tensor(v)},
                                      video_grid_thw=packed['video_grid_thw'].detach().cpu().tolist(),
                                      timestamps=list(sampled.timestamps),
                                      result=dataclasses.asdict(result),
                                      valid_timing=False,
                                      timing_note='Exploratory server GPU; inspect process observations before energy attribution. Captured-input runs are diagnostic only.')
                        record['memory_scope'] = 'Target and Draft both resident; AR is a quality reference, not standalone AR memory'
                        writer.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                        writer.flush()
                        completed.add(key)
                        print(json.dumps({k:record[k] for k in ['arm','request_id','phase','repeat','e2e_wall_s','vision_ms','num_tokens','stop_reason']}, ensure_ascii=False), flush=True)
                        status.update(completed_cells=len(completed), updated_at=timestamp())
                        save_json(out / 'status.json', status)
                        if captured:
                            torch.save({'x': captured[0].cpu(), 'grid_thw': record['video_grid_thw'],
                                        'request_id': cell['request_id']}, capture_dir / (cell['request_id'] + '.pt'))
                            capture_cache[cell['request_id']] = True
                        del result, packed, record
                    finally:
                        if capture_handle is not None:
                            capture_handle.remove()
                        timer.close()
                        sampler.stop()
            status.update(status='completed', completed_at=timestamp(), current=None)
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                      failed_at=timestamp())
        raise
    finally:
        save_json(out / 'status.json', status)


if __name__ == '__main__':
    main()
