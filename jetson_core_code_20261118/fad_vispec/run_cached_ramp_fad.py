"""Run Ramp/Fast with immutable source correspondence checked once at setup."""
from __future__ import annotations
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import run_ramp_fad as previous

cross, full = previous.cross, previous.full
parse_args = previous.parse_args
_ACTIVE = False


def main(argv=None):
    global _ACTIVE
    opt, forwarded = parse_args(argv)
    from combined_visual_experiment.fad_vispec.ramp_controller import RampConfig, RampController
    cfg = RampConfig(**full.read_json(opt.ramp_config)) if opt.controller_variant == 'ramp' else None
    config_digest = full.digest(opt.ramp_config) if cfg is not None else None
    full.require(not _ACTIVE, 'nested cached ramp entry forbidden')
    factory = full.ModelOps
    _ACTIVE = True

    class OptimizedOps(factory):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            from combined_visual_experiment.fad_vispec.cached_setup_fad_executor import prepare_cached_setup
            self.cached_setup = prepare_cached_setup()
            if cfg is not None:
                self.controller = RampController(config=cfg)
                self.metadata.update(policy=self.controller.policy_receipt(), actual_cost_table=None,
                                     calibration_used_by_current_policy=False)
            self.metadata.update(controller_variant=opt.controller_variant,
                feedback_buffer='request_reused_int64_20', ramp_config_sha256=config_digest,
                ramp_entry_sha256=full.digest(__file__),
                source_AST_validation='once_at_model_setup; source_and_callable_rechecked_each_request',
                controller_compute_and_feedback_included_in_generation=True)

        def executor(self, packed, request):
            from combined_visual_experiment.fad_vispec.cached_setup_fad_executor import CachedSetupFastFADExecutor
            return CachedSetupFastFADExecutor(self.spec, packed, request_key=request['request_id'],
                max_new_tokens=512, tree_budget=self.budget, eos_token_ids=[151645, 151643])

    full.ModelOps = OptimizedOps
    try:
        result = cross.main(forwarded)
        if cfg is not None:
            full.require(full.digest(opt.ramp_config) == config_digest, 'ramp config changed during execution')
        return result
    finally:
        full.ModelOps = factory
        _ACTIVE = False


if __name__ == '__main__':
    main()
