"""Three-arm development diagnostic, using the unchanged native ViSpec adaptor.

No experiment starts on import. The initial default is the three previously
length-sensitive development videos, one repeat, zero warmup, 512-token limit.
EOS behavior, prompt, Target/Draft, and tree verification are inherited intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PILOT_REQUESTS = tuple("direct_quality_vdc_v_" + name for name in
                       ("Kofjs0rkwxc", "syfyM1Pble8", "EZyLBq2jJtE"))
FROZEN_THRESHOLDS = dict(update1=0.19756050035357475, update2=0.16683463007211685,
                         temporal1=0.7460239082574844, temporal2=0.6907299906015396)
SCHEMES = {
    "baseline": dict(depth="fixed0", adaptor="native", compression=None, ratio=1.0),
    "direct_one_identity": dict(depth="direct", adaptor="native", compression=None,
                                ratio=1.0, residual_mode="identity"),
    "direct_one_reuse": dict(depth="direct", adaptor="native", compression=None,
                             ratio=1.0, residual_mode="reuse"),
}


class NativeRuntime:
    """No alternate adaptor, cached wrapper, compression, or forward patch."""
    def __init__(self, draft, *, difference_threshold=None):
        if difference_threshold is not None or not hasattr(draft, "imadpt"):
            raise ValueError("Require the original native imadpt without merging")

    @contextmanager
    def activate(self, joint, scheme):
        if scheme["adaptor"] != "native" or scheme["compression"] is not None or scheme["ratio"] != 1.0:
            raise ValueError("Residual diagnostic requires native imadpt and every visual token")
        yield dict(compression={"method": "identity", "actual_ratio": 1.0}, adaptor="native",
                   preparation={"mode": "native", "other_scheme_runtimes_constructed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int, choices=range(8))
    parser.add_argument("--requests", required=True)
    parser.add_argument("--local-thresholds", required=True)
    parser.add_argument("--request-id", action="append", help="Explicit later development subset; default is the fixed three-video pilot")
    parser.add_argument("--repeats", type=int, choices=[1], default=1)
    parser.add_argument("--warmup", type=int, choices=[0], default=0)
    parser.add_argument("--max-new-tokens", type=int, choices=[512], default=512)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    record = json.loads(Path(args.local_thresholds).read_text())
    if record.get("name") != "budget_first_ge110" or record.get("thresholds") != FROZEN_THRESHOLDS:
        parser.error("Require the exact old budget_first_ge110 thresholds; this diagnostic does not recalibrate")
    selected = args.request_id or list(PILOT_REQUESTS)
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    ids = [row["request_id"] for row in rows]
    if len(ids) != len(set(ids)) or len(selected) != len(set(selected)) or not set(selected) <= set(ids):
        parser.error("Request IDs must be unique and every selected request must exist")

    import run_joint
    from residual_reuse_depth import ResidualReuseDepthPolicy
    saved = {name: getattr(run_joint, name) for name in
             ("SCHEMES", "ARMS", "SchemeRuntime", "direct_policy_for_scheme", "save_json")}
    old_argv = sys.argv

    def policy(target, scheme, request, thresholds, anchor_stride=4):
        if thresholds != FROZEN_THRESHOLDS or scheme.get("residual_mode") not in {"identity", "reuse"}:
            raise ValueError("Unexpected diagnostic policy or thresholds")
        return ResidualReuseDepthPolicy(target, thresholds, residual_mode=scheme["residual_mode"],
                                        anchor_stride=anchor_stride)

    def save_json(path, value):
        if Path(path).name == "status.json":
            value = dict(value, methodology="prefix27_residual_reuse_development_diagnostic",
                residual_coefficient=1, gate_rule="Old budget_first_ge110 actions clamped to 0/1; old anchors",
                threshold_source_sha256=hashlib.sha256(Path(args.local_thresholds).read_bytes()).hexdigest(),
                pilot_request_ids=list(PILOT_REQUESTS), selected_request_ids=selected,
                independent_holdout=False, quality_pass=False, length_pass=False, speed_pass=False,
                acceptance_requirement="Every answer natural EOS, token count <=1.10 times paired baseline, and factual review before expansion",
                quality_claim="Residual extrapolation can change Target conditioning and text; no quality or length guarantee",
                adaptive_claim="No new score or threshold; fixed coefficient 1 from already computed block27 delta")
        saved["save_json"](path, value)

    forwarded = ["--out", args.out, "--physical-gpu", str(args.physical_gpu),
                 "--requests", args.requests, "--local-thresholds", args.local_thresholds,
                 "--repeats", str(args.repeats), "--warmup", str(args.warmup),
                 "--max-new-tokens", str(args.max_new_tokens), "--seed", str(args.seed),
                 "--anchor-stride", "4", "--skip-layers", "28", "29", "--prefix-layer", "27"]
    for request_id in selected:
        forwarded.extend(["--request-id", request_id])
    if args.resume:
        forwarded.append("--resume")
    try:
        run_joint.SCHEMES = {name: dict(scheme) for name, scheme in SCHEMES.items()}
        run_joint.ARMS = {name: (scheme["depth"], 1.0) for name, scheme in SCHEMES.items()}
        run_joint.SchemeRuntime = NativeRuntime
        run_joint.direct_policy_for_scheme = policy
        run_joint.save_json = save_json
        sys.argv = [str(Path(__file__)), *forwarded, "--arms", *SCHEMES]
        run_joint.main()
    finally:
        sys.argv = old_argv
        for name, original in saved.items():
            setattr(run_joint, name, original)


if __name__ == "__main__":
    main()
