"""Read-only post-hoc audit of active-width NCU artifacts.

The supervisor's historical campaign.json and all raw files remain untouched.
Workload completion, range/cell/PID binding and monitoring attribution are
revalidated independently, including when the supervisor receipt check failed.
Writes only a new --out JSON. No CUDA/NVML calls or model imports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_active_dram_campaign import ARM, REQUEST, workload_receipt
from validate_active_ring import load_run, bind_ncu


def classify_observations(observations, *, workload_pid, allowed_graphics):
    external, residents, errors = [], [], []
    allowed = set(allowed_graphics)
    for row in observations:
        compute = {p['pid'] for p in row.get('compute', [])}
        graphics = {p['pid'] for p in row.get('graphics', [])}
        # This PID is taken from the verified workload status AND NCU report,
        # never from the supervisor's potentially incomplete failure receipt.
        unknown = (compute | (graphics-allowed))-{workload_pid}
        resident = (graphics & allowed)-{workload_pid}
        if unknown:
            external.append({'time_unix': row['time_unix'], 'pids': sorted(unknown)})
        if resident:
            residents.append({'time_unix': row['time_unix'], 'pids': sorted(resident)})
        if any(key.endswith('_error') for key in row):
            errors.append(row)
    return dict(process_observation_count=len(observations), external_process_observations=external,
                resident_graphics_observations=residents, observation_errors=errors,
                allowed_resident_graphics_pids=sorted(allowed),
                attribution=('external_work_or_observation_error' if external or errors or not observations else
                             'resident_graphics_background_not_excluded' if residents else
                             'no_foreign_context_observed_not_proven_exclusive'),
                background_subtracted=False,
                observation_scope='whole campaign; not range-aligned; 0.5 second polling cannot exclude shorter contexts')


def audit(folder):
    folder = Path(folder)
    original = json.loads((folder/'campaign.json').read_text())
    arm = folder/ARM
    exit_receipt = json.loads((arm/'exit.json').read_text())
    if exit_receipt['returncode'] != 0:
        raise ValueError('NCU/application process did not exit successfully')
    request_id = original.get('protocol', {}).get('request_id', REQUEST)
    workload_receipt(arm, request_id)
    status, rows, _ = load_run(arm/'workload', active=True)
    metrics = bind_ncu(status, rows, arm/'metrics.csv')
    if len(rows) != 3 or sorted(r['phase'] for r in rows) != ['measure', 'measure', 'warmup']:
        raise ValueError('Expected one warmup and two measured result cells')
    observations = [json.loads(line) for line in (folder/'gpu_observations.jsonl').read_text().splitlines() if line.strip()]
    process_audit = classify_observations(observations, workload_pid=status['pid'],
                                         allowed_graphics=original.get('allowed_resident_graphics_pids', []))
    monitor_error = original.get('monitor_error')
    if monitor_error:
        process_audit['attribution'] = 'external_work_or_observation_error'
        process_audit['supervisor_monitor_error'] = monitor_error
    return {'source': str(folder), 'audit_kind': 'independent post-hoc artifact validation',
            'original_campaign_unchanged': True,
            'original_campaign_status': original['status'], 'original_campaign_error': original.get('error'),
            'application_returncode': exit_receipt['returncode'],
            'workload_status': status['status'], 'workload_variant': status['ring_variant'],
            'request_id': request_id, 'validated_rows_including_warmup': len(rows),
            'workload_pid': status['pid'], 'pid_bound_to_ncu': True, 'range_binding_verified': True,
            'ranges': metrics, 'process_audit': process_audit,
            'limits': ['A successful artifact audit does not rewrite the original supervisor failure.',
                       'Allowed resident graphics traffic remains part of the counter attribution uncertainty.',
                       'Output equivalence to the native reference must be checked with validate_active_ring.',
                       'NCU ranges measure generation; they are not ordinary latency or KV-only traffic.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    folder, out = Path(args.campaign), Path(args.out)
    if out.resolve() == (folder/'campaign.json').resolve():
        parser.error('the original campaign.json must remain unchanged')
    report = audit(folder)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('original_campaign_status', 'workload_status',
        'validated_rows_including_warmup', 'range_binding_verified', 'workload_pid')}, indent=2))
    return report


if __name__ == '__main__':
    main()
