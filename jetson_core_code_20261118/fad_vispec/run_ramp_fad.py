"""Buffered depth controller and reusable feedback buffer on the ordinary runner."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:sys.path.insert(0,str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import run_full_fad_cross_gpu as cross
full=cross.full
_ACTIVE=False


def parse_args(argv=None):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--controller-variant',choices=('ramp','old_full_fast'),default='ramp')
    p.add_argument('--ramp-config')
    opt,forwarded=p.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if opt.controller_variant=='ramp':full.require(opt.ramp_config is not None,'ramp config required')
    cross.parse_args(forwarded)
    return opt,forwarded


def main(argv=None):
    global _ACTIVE
    opt,forwarded=parse_args(argv)
    from combined_visual_experiment.fad_vispec.ramp_controller import RampConfig,RampController
    cfg=RampConfig(**full.read_json(opt.ramp_config)) if opt.controller_variant=='ramp' else None
    config_digest=full.digest(opt.ramp_config) if cfg is not None else None
    full.require(not _ACTIVE,'nested ramp entry forbidden')
    factory=full.ModelOps;_ACTIVE=True
    class OptimizedOps(factory):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            if cfg is not None:
                self.controller=RampController(config=cfg)
                self.metadata.update(policy=self.controller.policy_receipt(),actual_cost_table=None,calibration_used_by_current_policy=False)
            self.metadata.update(controller_variant=opt.controller_variant,feedback_buffer='request_reused_int64_20',
                ramp_config_sha256=config_digest,ramp_entry_sha256=full.digest(__file__),
                controller_compute_and_feedback_included_in_generation=True)
        def executor(self,packed,request):
            from combined_visual_experiment.fad_vispec.fast_fad_executor import FastFADExecutor
            return FastFADExecutor(self.spec,packed,request_key=request['request_id'],max_new_tokens=512,
                tree_budget=self.budget,eos_token_ids=[151645,151643])
    full.ModelOps=OptimizedOps
    try:
        result=cross.main(forwarded)
        if cfg is not None:full.require(full.digest(opt.ramp_config)==config_digest,'ramp config changed during execution')
        return result
    finally:
        full.ModelOps=factory;_ACTIVE=False


if __name__=='__main__':main()
