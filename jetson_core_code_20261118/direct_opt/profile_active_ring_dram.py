"""Profile complete generation with the independent active-width ring runner.

Default: one unprofiled warmup, two measured requests. Only generate_joint is
inside cudaProfilerStart/Stop; model loading, packed-input preparation, output
decoding and runtime cleanup are excluded. NCU wall/energy/allocator values
are diagnostic. No CUDA work occurs on import. Use a fresh output directory.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

from profile_ring_dram import GenerationProfiler, patched

VARIANT = 'active_width'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--profile-scope', choices=('generation',), default='generation')
    parser.add_argument('--profile-calls', type=int)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    args, rest = parser.parse_known_args(argv)
    if args.warmup < 0 or args.repeats < 1 or (args.profile_calls is not None and args.profile_calls < 1):
        parser.error('warmup must be nonnegative; repeats and profile-calls must be positive')
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument('--resume', action='store_true')
    options, _ = probe.parse_known_args(rest)
    if options.resume:
        parser.error('profiling requires a fresh output directory; --resume is unsupported')
    # Explicit forwarding keeps the profiler and underlying schedule defaults
    # aligned. The profile call count otherwise comes from schedule.json.
    return args, [*rest, '--warmup', str(args.warmup), '--repeats', str(args.repeats)]


def main(argv=None):
    args, rest = parse_args(argv)
    import run_joint
    import run_active_ring_layout
    import vispec_joint_adapter as joint
    import torch

    profiler = GenerationProfiler(torch.cuda, scope='generation', warmup_calls=args.warmup,
                                  profile_calls=args.profile_calls)
    original_save = run_joint.save_json

    def save(path, value):
        profiler.observe_save(path, value)
        if Path(path).name == 'status.json':
            if value.get('ring_variant') != VARIANT:
                raise RuntimeError('active-width status identity missing or changed')
            value = dict(value, **profiler.metadata(), profiler_ring_variant=VARIANT)
        return original_save(path, value)

    with ExitStack() as stack:
        stack.enter_context(patched(run_joint, 'save_json', save))
        stack.enter_context(patched(joint, 'generate_joint', profiler.wrap(joint.generate_joint)))
        run_active_ring_layout.main(rest)
        info = profiler.metadata()
        if not info['profiler_capture_complete'] or not profiler.profile_calls:
            raise RuntimeError('profiling finished without the requested complete generation ranges')
    return dict(info, profiler_ring_variant=VARIANT)


if __name__ == '__main__':
    main()
