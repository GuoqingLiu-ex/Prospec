"""Diagnostic Nsight Compute wrapper around the real ring-layout executor.

No experiment starts on import. Pass the normal run_ring_layout arguments plus
--profile-scope generation (default) or maintenance. The runner's --warmup
(default 1) supplies the unprofiled call count; all remaining scheduled calls
are profiled unless --profile-calls is explicit. Each selected generate_joint
call has one cudaProfilerStart/Stop range. Loading, input preparation, text
decoding and context-manager cleanup are outside that range.

Maintenance uses the same outer profiler range and NVTX push/pop KV_MAINT
ranges. Filter these with ncu --nvtx --nvtx-include 'KV_MAINT/'. This marks only
Target cache conversion, native post-prefill append/compact, and ring
append/commit (including shared commit-index preparation). Tree-mask creation
and attention reads are outside maintenance ranges but inside generation.
Profiled timing/allocator readings remain diagnostic, never performance data.
Use a fresh --out; multi-pass application replay must arrange fresh outputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
from pathlib import Path


def parse_args(argv=None):
    """Parse without importing torch, transformers, or the model runners."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--profile-scope', choices=('generation', 'maintenance'), default='generation')
    parser.add_argument('--profile-warmup-calls', type=int)
    parser.add_argument('--profile-calls', type=int)
    args, rest = parser.parse_known_args(argv)
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument('--warmup', type=int, default=1)
    probe.add_argument('--resume', action='store_true')
    runner, _ = probe.parse_known_args(rest)
    if runner.resume:
        parser.error('profiling requires a fresh output directory; --resume is unsupported')
    if args.profile_warmup_calls is None:
        args.profile_warmup_calls = runner.warmup
    if args.profile_warmup_calls < 0 or runner.warmup < 0:
        parser.error('warmup call counts must be nonnegative')
    if args.profile_calls is not None and args.profile_calls < 1:
        parser.error('--profile-calls must be positive')
    return args, rest


@contextmanager
def patched(obj, name, value):
    """Restore the exact attribute/descriptor, including on failed requests."""
    previous = vars(obj).get(name)
    existed = name in vars(obj)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


class GenerationProfiler:
    def __init__(self, cuda, *, scope, warmup_calls, profile_calls=None):
        self.cuda, self.scope = cuda, scope
        self.warmup_calls, self.profile_calls = warmup_calls, profile_calls
        self.calls = 0
        self.active = False
        self.maintenance_depth = 0
        self.current_cell = None
        self.ranges = []
        self.maintenance_counts = {}

    def observe_save(self, path, value):
        name = Path(path).name
        if name == 'schedule.json' and self.profile_calls is None:
            self.profile_calls = max(0, len(value) - self.warmup_calls)
        elif name == 'status.json' and value.get('current') is not None:
            self.current_cell = dict(value['current'])

    def metadata(self):
        complete = sum(r['completed'] for r in self.ranges)
        return dict(under_ncu=True, ncu_attachment_verified=False,
                    profiler_scope=self.scope, profiler_api='cudaProfilerStart/Stop',
                    profiled_wall_is_performance_evidence=False,
                    profiler_warmup_calls=self.warmup_calls, profiler_requested_calls=self.profile_calls,
                    generation_calls=self.calls, profiler_completed_ranges=complete,
                    profiler_capture_complete=(self.profile_calls is not None and complete == self.profile_calls),
                    profiler_ranges=[dict(r) for r in self.ranges],
                    maintenance_nvtx_name='KV_MAINT' if self.scope == 'maintenance' else None,
                    maintenance_calls=dict(self.maintenance_counts),
                    profile_scope_note=('generate_joint only; loading/input preparation/output decoding/runtime cleanup excluded; '
                                        'all profiled rows are diagnostic, including wall/energy/memory values'))

    def wrap(self, original):
        @wraps(original)
        def generate(*args, **kwargs):
            if self.active:
                raise RuntimeError('nested generate_joint profiling is unsupported')
            ordinal = self.calls
            self.calls += 1
            if self.profile_calls is None:
                raise RuntimeError('runner schedule must be observed before generation')
            selected = self.warmup_calls <= ordinal < self.warmup_calls + self.profile_calls
            if not selected:
                return original(*args, **kwargs)
            record = dict(call_index=ordinal, range_index=len(self.ranges),
                          cell=dict(self.current_cell) if self.current_cell else None, completed=False)
            self.ranges.append(record)
            # Drain input preparation before starting collection. Drain the
            # actual generation before Stop; do not add per-layer syncs.
            self.cuda.synchronize()
            self.cuda.profiler.start()
            self.active = True
            try:
                result = original(*args, **kwargs)
                self.cuda.synchronize()
                record['completed'] = True
                return result
            finally:
                try:
                    if not record['completed']:
                        self.cuda.synchronize()
                finally:
                    self.active = False
                    self.cuda.profiler.stop()
        return generate

    @contextmanager
    def maintenance(self, name, enabled):
        selected = self.active and self.scope == 'maintenance' and enabled
        if not selected:
            yield
            return
        outer = self.maintenance_depth == 0
        if outer:
            self.cuda.nvtx.range_push('KV_MAINT')
            self.maintenance_counts[name] = self.maintenance_counts.get(name, 0) + 1
        self.maintenance_depth += 1
        try:
            yield
        finally:
            self.maintenance_depth -= 1
            if outer:
                self.cuda.nvtx.range_pop()


@contextmanager
def maintenance_patches(profiler, bridge, runtime, primitives, dynamic_layer):
    """Install BEFORE ring_cache_policy snapshots bridge._compact.

    Full-mode class-level update is filtered by exact active Target layer
    identity, after prefill. Candidate methods use exact active store identity.
    No Draft operation, prefill append, or nested maintenance range is tagged.
    """
    def active():
        return runtime._ACTIVE

    def store_selected(store):
        state = active()
        return state is not None and state.mode != 'full' and any(store is s for s in state.stores)

    original_update = dynamic_layer.update
    @wraps(original_update)
    def update(layer, *args, **kwargs):
        state = active()
        enabled = (state is not None and state.mode == 'full' and state.cache is not None
                   and state.current_phase == 'decode' and state.in_forward
                   and any(layer is item for item in state.cache.layers))
        with profiler.maintenance('target_full_append', enabled):
            return original_update(layer, *args, **kwargs)

    original_compact = bridge._compact
    @wraps(original_compact)
    def compact(cache, *args, **kwargs):
        state = active()
        with profiler.maintenance('target_full_compact', state is not None and
                                  state.mode == 'full' and cache is state.cache):
            return original_compact(cache, *args, **kwargs)

    cls = primitives.RingKVLayer
    original_from = cls.from_prefill.__func__
    @wraps(original_from)
    def from_prefill(the_cls, keys, values, *args, **kwargs):
        state = active()
        enabled = (state is not None and state.mode != 'full' and state.cache is not None
                   and state.current_phase == 'conversion' and any(
                       keys is layer.keys and values is layer.values for layer in state.cache.layers))
        with profiler.maintenance('target_ring_conversion', enabled):
            return original_from(the_cls, keys, values, *args, **kwargs)

    def method(name):
        original = getattr(cls, name)
        @wraps(original)
        def call(store, *args, **kwargs):
            with profiler.maintenance('target_ring_' + name, store_selected(store)):
                return original(store, *args, **kwargs)
        return call

    original_prepare = primitives.prepare_commit
    @wraps(original_prepare)
    def prepare(plan, *args, **kwargs):
        state = active()
        with profiler.maintenance('target_ring_prepare_commit', state is not None and
                                  state.mode != 'full' and plan is state.plan):
            return original_prepare(plan, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patched(dynamic_layer, 'update', update))
        stack.enter_context(patched(bridge, '_compact', compact))
        stack.enter_context(patched(cls, 'from_prefill', classmethod(from_prefill)))
        stack.enter_context(patched(cls, 'append_tree', method('append_tree')))
        stack.enter_context(patched(cls, 'commit_prepared', method('commit_prepared')))
        stack.enter_context(patched(primitives, 'prepare_commit', prepare))
        yield


def main(argv=None):
    args, rest = parse_args(argv)
    # Imports do not initialize CUDA; the underlying runner acquires its lock
    # and sets CUDA_VISIBLE_DEVICES before making any CUDA calls.
    import run_joint
    import run_ring_layout
    import vispec_joint_adapter as joint
    import torch

    profiler = GenerationProfiler(torch.cuda, scope=args.profile_scope,
                                  warmup_calls=args.profile_warmup_calls,
                                  profile_calls=args.profile_calls)
    original_save = run_joint.save_json

    def save(path, value):
        profiler.observe_save(path, value)
        if Path(path).name == 'status.json':
            value = dict(value, **profiler.metadata())
        return original_save(path, value)

    with ExitStack() as stack:
        stack.enter_context(patched(run_joint, 'save_json', save))
        stack.enter_context(patched(joint, 'generate_joint', profiler.wrap(joint.generate_joint)))
        if args.profile_scope == 'maintenance':
            import hf517_vispec_adapter as bridge
            import ring_cache_runtime as runtime
            import ring_kv_cache as primitives
            from transformers.cache_utils import DynamicLayer
            stack.enter_context(maintenance_patches(profiler, bridge, runtime, primitives, DynamicLayer))
        run_ring_layout.main(rest)
        if not profiler.metadata()['profiler_capture_complete'] or not profiler.profile_calls:
            raise RuntimeError('profiling finished without the requested complete generation ranges')
    return profiler.metadata()


if __name__ == '__main__':
    main()
