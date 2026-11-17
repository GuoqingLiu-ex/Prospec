"""Offline first-profile notes from the already-published adaptor microbench."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(raw_path: Path):
    rows = json.loads(raw_path.read_text())
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["n"], row["method"])].append(row["wall_ms"])
    out = []
    for (n, method), values in sorted(grouped.items()):
        values = sorted(values)
        mid = statistics.median(values)
        native = grouped[(n, "baseline")]
        native_mid = statistics.median(native)
        paired = []
        # repeats are interleaved; compare by order is not paired here.
        out.append({
            "n": n, "method": method, "median_wall_ms": mid,
            "vs_native_median_pct": 100.0 * (native_mid - mid) / native_mid,
            "count": len(values),
        })
    return out


def main():
    raw = Path(__file__).resolve().parent.parent / "training_free/results/adaptor_microbench_gpu7/raw.json"
    rows = summarize(raw)
    interesting = [r for r in rows if r["method"] in {"baseline", "reorder", "response50", "reorder_response50"}]
    print(json.dumps(interesting, indent=2))


if __name__ == "__main__":
    main()
