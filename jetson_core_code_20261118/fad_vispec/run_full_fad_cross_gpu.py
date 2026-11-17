"""Formal cross-GPU FAD child: the frozen warm1 + two E24 repeats, fused packet.

The user's protocol compares this GPU6 child with the independently running
GPU0 original child. All ordinary preparation/generation/cleanup stays in the
frozen runner; this entry only selects its full arm and validated packet backend.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:sys.path.insert(0,str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import run_original_full_fad_abba as ordinary
from combined_visual_experiment.fad_vispec import run_full_fad_fused as fused_wrapper
full=ordinary.full
require,digest=full.require,full.digest
PROTOCOL='formal_cross_GPU_original_vs_full_FAD_E24'
_ACTIVE=False


def parse_args(argv=None):
    tokens=list(sys.argv[1:] if argv is None else argv)
    require(not any(t.split('=',1)[0]=='--validate-feedback' for t in tokens),'old CPU-reference --validate-feedback is forbidden')
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--fused-validation-report',required=True);p.add_argument('--fused-validation-report-sha256',required=True)
    wrapper,forwarded=p.parse_known_args(tokens);args=ordinary.parse_args(forwarded)
    require(args.child_arm in (None,'full'),'formal FAD entry requires the full child')
    if args.child_arm is None:forwarded+=['--child-arm','full'];args.child_arm='full'
    return wrapper,args,forwarded


def main(argv=None):
    global _ACTIVE
    wrapper,args,forwarded=parse_args(argv)
    proof=fused_wrapper.validation_receipt(wrapper.fused_validation_report,wrapper.fused_validation_report_sha256,args.expected_gpu_uuid)
    require(not _ACTIVE,'nested formal FAD installation forbidden')
    original_factory=full.ModelOps;patches=[];_ACTIVE=True
    class FusedFormalOps(original_factory):
        def __init__(self,*a,**kw):
            # The ordinary child has already taken flock, completed both idle
            # polls and bound its expected CUDA UUID before reaching ModelOps.
            from combined_visual_experiment.fad_vispec import production_fad_executor as executor
            from combined_visual_experiment.fad_vispec import full_feedback_fused as candidate
            require(digest(candidate.__file__)==fused_wrapper.FUSED_SHA,'validated candidate source changed')
            require(not patches,'one model factory per formal child required')
            patches.append((executor,executor.build_feedback_packet));executor.build_feedback_packet=candidate.build_feedback_packet
            super().__init__(*a,**kw)
            self.metadata.update(formal_protocol=PROTOCOL,feedback_backend='fused_cuda_single_packet',
                fused_source_sha256=fused_wrapper.FUSED_SHA,fused_GPU_validation=proof,fused_CPU_or_Torch_fallback=False,
                formal_entry_sha256=digest(__file__),ordinary_entry_sha256=digest(ordinary.__file__),
                complete_generation_entry_sha256=digest(full.__file__),feedback_selector_and_receipt_costs_in_generation=True,
                hardware_single_card_equivalence_claim=False)
    full.ModelOps=FusedFormalOps;failure=None
    try:return ordinary.main(forwarded)
    except BaseException as exc:failure=exc;raise
    finally:
        for module,old in reversed(patches):module.build_feedback_packet=old
        full.ModelOps=original_factory;_ACTIVE=False
        if failure is None:
            require(fused_wrapper.validation_receipt(wrapper.fused_validation_report,wrapper.fused_validation_report_sha256,args.expected_gpu_uuid)==proof,
                'fused validation changed during formal generation')


if __name__=='__main__':main()
