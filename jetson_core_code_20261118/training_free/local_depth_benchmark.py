"""Extend the existing visual microbenchmark with the frozen local policy.

Input preparation stays outside timing; policy installation, metadata,
statistics, actual bypass and output completion are inside wall timing.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vision_microbench
import vision_depth
from local_depth import local_depth_policy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--local-thresholds', required=True)
    parser.add_argument('--out', required=True)
    args, remaining = parser.parse_known_args()
    thresholds = json.loads(Path(args.local_thresholds).read_text())
    original = vision_depth.depth_policy

    def route(target, mode, **kwargs):
        if mode == 'local':
            kwargs['thresholds'] = thresholds
            return local_depth_policy(target, **kwargs)
        return original(target, mode=mode, **kwargs)

    vision_depth.depth_policy = route
    vision_microbench.MODES = ('native', 'fixed1', 'adaptive', 'local')
    sys.argv = [sys.argv[0], '--out', args.out, *remaining]
    try:
        vision_microbench.main()
    finally:
        out = Path(args.out)
        if out.exists():
            (out/'local_policy_receipt.json').write_text(json.dumps({
                'local_thresholds': thresholds, 'source': args.local_thresholds,
                'policy': 'local_window_statistics_v1', 'trained': False,
                'modes': list(vision_microbench.MODES)}, indent=2)+'\n')
        vision_depth.depth_policy = original


if __name__ == '__main__':
    main()
