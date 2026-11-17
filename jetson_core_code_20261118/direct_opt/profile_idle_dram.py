"""Small diagnostic to bound observed idle DRAM background, not subtract it."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranges", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()
    import torch
    out = Path(args.out)
    out.mkdir(exist_ok=False, parents=True)
    x = torch.ones(1024, device="cuda", dtype=torch.float32)
    for _ in range(3):
        x.add_(0.0)
    torch.cuda.synchronize()
    records = []
    for index in range(args.ranges):
        torch.cuda.profiler.start()
        started = time.perf_counter()
        x.add_(0.0)
        torch.cuda.synchronize()
        time.sleep(args.sleep_seconds)
        x.add_(0.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        torch.cuda.profiler.stop()
        records.append({"range_index": index, "elapsed_wall_s": elapsed,
                        "payload_bytes": x.numel() * x.element_size()})
    (out / "ranges.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
