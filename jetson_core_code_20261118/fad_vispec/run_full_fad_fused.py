"""Thin full-generation runner with an explicitly GPU-validated fused packet.

The frozen run_full_fad owns guarding, loading, generation, timing and output.
Only its ModelOps factory and the packet callable are scoped replacements. The
CUDA-only candidate has no fallback and is imported after the original guards.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:sys.path.insert(0,str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import run_full_fad as full
require,digest,read_json=full.require,full.digest,full.read_json
VALIDATION_SCHEMA='fad_full_feedback_fused_device_validation_v1'
FUSED_SHA='e858f721cc5fc6909b0a69300174eecc1eb9067f53fd61c6478a1d83906f4c01'
FUSED_SOURCE=PROJECT/'fad_vispec/full_feedback_fused.py'
_ACTIVE=False


def validation_receipt(path,expected_sha256,expected_uuid):
    require(digest(path)==expected_sha256,'fused GPU validation report bytes differ')
    r=read_json(path)
    require(r['schema']==VALIDATION_SCHEMA and r['passed'] is True,'completed fused GPU validation required')
    require(r['device_uuid']==expected_uuid and r['device_actual']=='cuda:0','fused validation device UUID differs')
    require(r['fused_source_sha256']==FUSED_SHA and digest(FUSED_SOURCE)==FUSED_SHA,'fused GPU-validated source differs')
    return dict(path=str(Path(path).resolve()),sha256=expected_sha256,schema=r['schema'],device_uuid=r['device_uuid'],
        device_actual=r['device_actual'],fused_source_sha256=FUSED_SHA,passed=True)


def parse_args(argv=None):
    tokens=list(sys.argv[1:] if argv is None else argv)
    require(not any(t.split('=',1)[0]=='--validate-feedback' for t in tokens),
        '--validate-feedback uses the original validator and is forbidden under the CUDA-only fused wrapper')
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--fused-validation-report',required=True)
    p.add_argument('--fused-validation-report-sha256',required=True)
    wrapper,forwarded=p.parse_known_args(tokens)
    args=full.parse_args(forwarded)
    return wrapper,args,forwarded


def main(argv=None):
    global _ACTIVE
    wrapper,args,forwarded=parse_args(argv)
    proof=validation_receipt(wrapper.fused_validation_report,wrapper.fused_validation_report_sha256,args.expected_gpu_uuid)
    require(not _ACTIVE,'nested fused runner installation is forbidden')
    original_factory=full.ModelOps;packet_patch=[];_ACTIVE=True
    class FusedModelOps(original_factory):
        def __init__(self,*a,**kw):
            # Frozen main reaches this factory only after flock, two idle polls,
            # CUDA UUID binding and the existing precision scope.
            from combined_visual_experiment.fad_vispec import production_fad_executor as executor
            from combined_visual_experiment.fad_vispec import full_feedback_fused as fused
            require(digest(fused.__file__)==FUSED_SHA,'fused source changed before model construction')
            require(not packet_patch,'one guarded ModelOps construction required')
            packet_patch.append((executor,executor.build_feedback_packet))
            executor.build_feedback_packet=fused.build_feedback_packet
            super().__init__(*a,**kw)
            self.metadata.update(feedback_backend='fused_cuda_single_packet',fused_source_sha256=FUSED_SHA,
                fused_feedback_active=self.mode!='fixed4_p0',fused_GPU_validation=proof,
                fused_wrapper_source_sha256=digest(__file__),fused_wrapper_delegates_to_source_sha256=digest(full.__file__),
                fused_CPU_or_Torch_fallback=False,feedback_selector_and_receipt_costs_in_generation=True)
    full.ModelOps=FusedModelOps
    failure=None
    try:
        return full.main(forwarded)
    except BaseException as exc:
        failure=exc;raise
    finally:
        for module,old in reversed(packet_patch):module.build_feedback_packet=old
        full.ModelOps=original_factory;_ACTIVE=False
        if failure is None:
            require(validation_receipt(wrapper.fused_validation_report,wrapper.fused_validation_report_sha256,args.expected_gpu_uuid)==proof,
                'fused validation changed during generation')


if __name__=='__main__':main()
