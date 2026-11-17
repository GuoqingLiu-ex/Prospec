"""Compare stored pilot timestamps with a fresh sample of the same videos.

h264 mmco messages are decoder warnings. This check does not treat a finished
13/13 run as a crash; it reports whether the saved frame indices still match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--temporal-patch-size", type=int, default=2)
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()
    requests = {}
    for line in Path(args.requests).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            requests[row["request_id"]] = row
    rows = []
    for line in Path(args.results).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") == "measure" and row.get("arm") == "baseline":
            rows.append(row)
    report = []
    for row in rows:
        req = requests[row["request_id"]]
        nframes = int(req["nframes"])
        timestamps = [float(v) for v in row.get("timestamps") or []]
        grid = row["video_grid_thw"]
        item = {
            "request_id": row["request_id"],
            "nframes": nframes,
            "stored_timestamps": len(timestamps),
            "grid_t": int(grid[0][0]) if grid else None,
            "temporal_patches_match": bool(grid) and int(grid[0][0]) * args.temporal_patch_size == nframes,
            "timestamps_match_count": len(timestamps) == nframes,
            "timestamps_nondecreasing": all(b >= a for a, b in zip(timestamps, timestamps[1:])),
        }
        if args.resample:
            from jev_spec.video.sampling import sample_video
            sampled = sample_video(req["video_path"], nframes)
            fresh = [float(v) for v in sampled.timestamps]
            item["resampled_timestamps_match"] = fresh == timestamps
            item["resampled_indices"] = list(sampled.metadata["frames_indices"])
            item["source_timestamps_present"] = sampled.provenance.get("source_timestamps_s") is not None
        report.append(item)
    print(json.dumps({
        "measured": len(report),
        "temporal_ok": sum(item["temporal_patches_match"] for item in report),
        "timestamp_count_ok": sum(item["timestamps_match_count"] for item in report),
        "resample_ok": None if not args.resample else sum(bool(item.get("resampled_timestamps_match")) for item in report),
        "rows": report,
    }, indent=2))


if __name__ == "__main__":
    main()
