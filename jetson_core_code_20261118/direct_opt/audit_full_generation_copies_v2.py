"""Kineto v2: CPU generation marker plus optional GPU annotation mirrors.

Read-only, streaming CPU analysis. Uses frozen v1 correlation/source/result/loss
checks unchanged. GPU annotations describe a range, not an additional workload.
The original trace and sidecar are never rewritten. Software counts are not DRAM.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import tempfile

import audit_full_generation_copies as legacy
from audit_full_generation_copies import (TraceFormatError, _Reasons, stream_trace,
    file_hash, _finite, _identity, _load_bound_sidecar, _empty_counts, _finish_counts,
    _count, _work_api_family, operator_role, compare_audits, CPU_CATEGORIES,
    GPU_CATEGORIES)

LEGACY_SHA256='318fb9d1e57b63152f63fb96021ad19c39a9b5451d8c8ae81b5a5f9590d9fef8'
# A new local set; never modify the imported v1 globals.
IGNORED_CATEGORIES=legacy.IGNORED_CATEGORIES | {'gpu_user_annotation'}

def analyze_trace(trace, *, sidecar=None, calibration=False, evidence_root=None, chunk_size=65536):
    """Return raw/scoped counts and explicit validity; malformed JSON raises, defects retain raw."""
    if file_hash(legacy.__file__) != LEGACY_SHA256:
        raise ValueError("v2 requires the audited frozen v1 dependency")
    trace = Path(trace)
    trace_receipt = {'sha256': file_hash(trace), 'bytes': trace.stat().st_size}
    metadata, categories, reasons, markers = {}, Counter(), _Reasons(), []
    gpu_annotation_names, gpu_marker_annotations = Counter(), []
    actual_sidecar = json.loads(Path(sidecar).read_text()) if sidecar else {}
    if not isinstance(actual_sidecar, dict):
        raise ValueError('sidecar must be an object')
    source_checks, bound_result = [], None
    if sidecar:
        source_checks, bound_result = _load_bound_sidecar(actual_sidecar, sidecar, evidence_root, trace_receipt, reasons)
    else:
        reasons.append('sidecar_missing')
    if calibration:
        reasons.append('calibration_only_not_full_generation')
    with tempfile.TemporaryDirectory(prefix='kineto-audit-') as temp:
        conn = sqlite3.connect(str(Path(temp) / 'events.sqlite'))
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA temp_store=FILE')
        conn.execute('CREATE TABLE events (id INTEGER PRIMARY KEY, cat TEXT, ph TEXT, name TEXT, pid TEXT, tid TEXT, ts REAL, dur REAL, ext TEXT, corr TEXT, device TEXT, context TEXT, args TEXT, role TEXT)')
        for kind, item in stream_trace(trace, chunk_size=chunk_size):
            if kind == 'metadata':
                key, value = item
                metadata[key] = value
                if any(word in key.lower() for word in ('dropped', 'overflow', 'warning', 'error')) and value not in (None, False, 0, [], ''):
                    reasons.append('trace_integrity_metadata:' + key)
                continue
            event = item
            cat, ph, name = event.get('cat'), event.get('ph'), event.get('name', '')
            categories[str(cat) + '/' + str(ph)] += 1
            if not isinstance(name, str):
                raise TraceFormatError('event name is not a string')
            if re.search(r'dropped[ _]*events|buffer[ _]*overflow|cupti[ _]*(?:error|warning)', name, re.I):
                reasons.append('trace_integrity_event:' + name)
            if name.startswith('FULL_GENERATION/'):
                if cat == 'user_annotation':
                    # Keep every CPU marker, including a wrong-PID marker. A
                    # second actual CPU marker must not disappear by filtering.
                    markers.append(event)
                elif cat == 'gpu_user_annotation':
                    gpu_marker_annotations.append(event)
                else:
                    reasons.append('full_generation_name_on_non_annotation:' + str(cat))
            if cat == 'gpu_user_annotation':
                gpu_annotation_names[name] += 1
                annotation_args = event.get('args', {})
                if not isinstance(annotation_args, dict):
                    raise TraceFormatError('GPU annotation args is not an object')
                if (ph != 'X' or not _finite(event.get('ts')) or not _finite(event.get('dur'))
                        or event['dur'] < 0 or _identity(event.get('pid')) is None or _identity(event.get('tid')) is None):
                    reasons.append('gpu_annotation_shape')
                for key, value in annotation_args.items():
                    if any(word in key.lower() for word in ('dropped', 'overflow', 'warning', 'error')) and value not in (None, False, 0, [], ''):
                        reasons.append('trace_integrity_event_arg:' + key)
            if cat in CPU_CATEGORIES | GPU_CATEGORIES and ph != 'X':
                reasons.append('unsupported_duration_phase:' + str(cat) + '/' + str(ph))
            if cat not in CPU_CATEGORIES | GPU_CATEGORIES | IGNORED_CATEGORIES:
                reasons.append('unknown_event_category:' + str(cat))
            if cat not in CPU_CATEGORIES | GPU_CATEGORIES or ph != 'X':
                continue
            args = event.get('args', {})
            if not isinstance(args, dict):
                raise TraceFormatError('event args is not an object')
            for key, value in args.items():
                if any(word in key.lower() for word in ('dropped', 'overflow', 'warning', 'error')) and value not in (None, False, 0, [], ''):
                    reasons.append('trace_integrity_event_arg:' + key)
            valid_time = _finite(event.get('ts')) and _finite(event.get('dur')) and event['dur'] >= 0
            if not valid_time:
                reasons.append('event_invalid_timestamp')
            conn.execute('INSERT INTO events(cat,ph,name,pid,tid,ts,dur,ext,corr,device,context,args) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                         (cat, ph, name, _identity(event.get('pid')), _identity(event.get('tid')),
                          event.get('ts') if valid_time else None, event.get('dur') if valid_time else None,
                          _identity(args.get('External id')), _identity(args.get('correlation')),
                          _identity(args.get('device')), _identity(args.get('context')), json.dumps(args)))
        conn.commit()
        conn.executescript('CREATE INDEX corr_idx ON events(corr,cat); CREATE INDEX external_idx ON events(pid,ext,cat); CREATE INDEX order_idx ON events(cat,pid,tid,ts);')
        duplicates = conn.execute("SELECT COUNT(*) FROM (SELECT 1 FROM events WHERE cat IN ('kernel','gpu_memcpy','gpu_memset') GROUP BY cat,pid,tid,ts,dur,corr,name,args HAVING COUNT(*)>1)").fetchone()[0]
        if duplicates:
            reasons.append('duplicate_gpu_activity_records')
        # Actual CPU nesting is computed in timestamp order; only a thread's live stack resides in memory.
        stack, thread = [], None
        for event in conn.execute("SELECT * FROM events WHERE cat='cpu_op' ORDER BY pid,tid,ts,dur DESC,id"):
            ident = (event['pid'], event['tid'])
            if ident != thread:
                stack, thread = [], ident
            if event['ts'] is None:
                continue
            end = event['ts'] + event['dur']
            while stack and not (stack[-1][0] <= event['ts'] and stack[-1][1] >= end):
                stack.pop()
            role = operator_role(event['name'], stack[-1][2] if stack else None)
            conn.execute('UPDATE events SET role=? WHERE id=?', (role, event['id']))
            stack.append((event['ts'], end, role))
        conn.commit()
        marker = None
        if len(markers) != 1:
            reasons.append('full_generation_marker_count:' + str(len(markers)))
        else:
            candidate = markers[0]
            if (candidate.get('ph') == 'X' and candidate.get('cat') == 'user_annotation'
                    and _finite(candidate.get('ts')) and _finite(candidate.get('dur')) and candidate['dur'] > 0
                    and isinstance(candidate.get('args', {}), dict)):
                marker = candidate
            else:
                reasons.append('full_generation_marker_shape')
        pid = _identity(actual_sidecar.get('process', {}).get('os_pid'))
        gpu = actual_sidecar.get('gpu', {})
        cpu_marker_candidate = marker
        if marker:
            identity_valid = True
            if marker['name'] != 'FULL_GENERATION/' + str(actual_sidecar.get('nonce')):
                reasons.append('marker_nonce_mismatch')
                identity_valid = False
            if pid is None or _identity(marker.get('pid')) != pid:
                reasons.append('marker_cpu_pid_mismatch')
                identity_valid = False
            if not identity_valid:
                marker = None
        # Mirrors can span multiple GPU streams. They cannot establish a CPU
        # generation scope and never enter launch/activity correlation tables.
        for annotation in gpu_marker_annotations:
            if annotation['name'] != 'FULL_GENERATION/' + str(actual_sidecar.get('nonce')):
                reasons.append('gpu_marker_nonce_mismatch')
            if cpu_marker_candidate is None:
                reasons.append('gpu_marker_without_unique_cpu_marker')
            else:
                cpu_ext = cpu_marker_candidate.get('args', {}).get('External id')
                gpu_ext = annotation.get('args', {}).get('External id')
                if cpu_ext is not None and gpu_ext is not None and _identity(cpu_ext) != _identity(gpu_ext):
                    reasons.append('gpu_marker_external_id_conflict')
        properties = metadata.get('deviceProperties')
        property_row = [x for x in properties or [] if isinstance(x, dict) and x.get('id') == gpu.get('logical_device')] if isinstance(properties, list) else []
        if len(property_row) != 1:
            reasons.append('trace_device_properties_binding')
        trace_uuid = property_row[0].get('uuid') if len(property_row) == 1 else None
        if trace_uuid is not None and trace_uuid != gpu.get('uuid'):
            reasons.append('trace_gpu_uuid_mismatch')
        if not calibration and (metadata.get('record_shapes') != 0 or metadata.get('profile_memory') != 0):
            reasons.append('trace_diagnostic_configuration')
        use_range = marker is not None and not calibration
        start, stop = (marker['ts'], marker['ts'] + marker['dur']) if use_range else (None, None)

        def inside(row):
            return row['ts'] is not None and (not use_range or (start <= row['ts'] and row['ts'] + row['dur'] <= stop))

        raw, scoped = _empty_counts(), _empty_counts()
        kernel_roles, classified_names, binding_counts = Counter(), Counter(), Counter()
        issue_examples = []

        def defect(reason, event):
            reasons.append(reason)
            binding_counts[reason] += 1
            if len(issue_examples) < 20:
                issue_examples.append({'reason': reason, 'event_id': event['id'], 'name': event['name'], 'correlation': event['corr']})

        for event in conn.execute('SELECT * FROM events ORDER BY id'):
            _count(raw, event)
            if event['cat'] in CPU_CATEGORIES:
                selected = inside(event)
                if use_range and event['ts'] is not None and event['ts'] < stop and event['ts'] + event['dur'] > start and not selected:
                    defect('cpu_event_crosses_marker', event)
                if selected:
                    _count(scoped, event)
                    if pid is not None and event['pid'] != pid:
                        defect('range_cpu_pid_mismatch', event)
                continue
            # Correlations are process launch IDs, never Chrome GPU lane PIDs.
            launches = list(conn.execute("SELECT * FROM events WHERE corr=? AND cat IN ('cuda_runtime','cuda_driver')", (event['corr'],))) if event['corr'] is not None else []
            if not launches:
                defect('gpu_missing_launch_correlation', event)
                if not use_range or inside(event):
                    _count(scoped, event)
                if event['cat'] == 'kernel' and (not use_range or inside(event)):
                    kernel_roles['unclassified'] += 1
                continue
            selected_launches = [row for row in launches if inside(row)]
            if use_range and not selected_launches:
                binding_counts['gpu_outside_marker_launch'] += 1
                continue
            _count(scoped, event)
            binding_counts['selected_gpu_activities'] += 1
            launch_pids = {row['pid'] for row in selected_launches}
            ext_ids = {row['ext'] for row in selected_launches if row['ext'] is not None}
            valid_binding = True
            families = [_work_api_family(row['name'], row['cat']) for row in selected_launches]
            if any(family is None for family in families):
                defect('gpu_correlated_api_not_supported_work', event)
                valid_binding = False
            elif any(family != event['cat'] for family in families):
                defect('gpu_activity_api_family_mismatch', event)
                valid_binding = False
            # Runtime->driver nesting is legal; repeated or unrelated launches sharing an ID are not.
            ambiguous = len({row['cat'] for row in selected_launches}) != len(selected_launches)
            if len(selected_launches) == 2:
                a, b = selected_launches
                nested = (a['ts'] <= b['ts'] and a['ts']+a['dur'] >= b['ts']+b['dur']) or (b['ts'] <= a['ts'] and b['ts']+b['dur'] >= a['ts']+a['dur'])
                ambiguous |= not nested or a['tid'] != b['tid']
            if ambiguous:
                defect('ambiguous_launch_correlation', event)
                valid_binding = False
            if len(launch_pids) != 1 or (pid is not None and launch_pids != {pid}):
                defect('gpu_launch_cpu_pid_mismatch', event)
                valid_binding = False
            if len(ext_ids) > 1 or (event['ext'] is not None and ext_ids and event['ext'] not in ext_ids):
                defect('gpu_external_id_conflict', event)
                valid_binding = False
            if use_range and not inside(event):
                defect('gpu_activity_outside_completed_marker', event)
                valid_binding = False
            if sidecar and (event['device'] != _identity(gpu.get('logical_device')) or event['context'] not in set(map(str, gpu.get('contexts', [])))):
                defect('gpu_device_or_context_mismatch', event)
                valid_binding = False
            external = event['ext'] if event['ext'] is not None else next(iter(ext_ids), None)
            cpu = list(conn.execute("SELECT * FROM events WHERE cat='cpu_op' AND pid=? AND ext=?", (next(iter(launch_pids), None), external))) if external is not None else []
            role = 'unclassified'
            operator_name = None
            if len(cpu) == 1 and inside(cpu[0]) and valid_binding:
                role, operator_name = cpu[0]['role'] or 'unclassified', cpu[0]['name']
            elif len(cpu) > 1:
                defect('ambiguous_cpu_external_id', event)
            if event['cat'] == 'kernel':
                kernel_roles[role] += 1
                classified_names[(role, operator_name or 'unknown', event['name'])] += 1
        # Reverse coverage catches a launch whose GPU activity was lost. API counts are
        # still separate from activities; one launch can legitimately have many records.
        for call in conn.execute("SELECT * FROM events WHERE cat IN ('cuda_runtime','cuda_driver')"):
            if not inside(call):
                continue
            name = call['name']
            family = _work_api_family(name, call['cat'])
            if name.startswith(('cudaGraphLaunch', 'cuGraphLaunch')):
                defect('unsupported_graph_launch_coverage', call)
                continue
            if name.startswith(('cudaLaunch', 'cuLaunch', 'cudaMemcpy', 'cuMemcpy', 'cudaMemset', 'cuMemset')) and family is None and name not in {'cudaLaunchHostFunc','cuLaunchHostFunc'}:
                defect('unsupported_gpu_work_api_coverage', call)
                continue
            if family is None:
                continue
            activities = conn.execute("SELECT COUNT(*) FROM events WHERE corr=? AND cat=?", (call['corr'], family)).fetchone()[0] if call['corr'] is not None else 0
            if not activities:
                defect('recorded_gpu_work_api_without_activity', call)
        if not scoped['gpu_kernel']:
            reasons.append('no_scoped_gpu_kernel_activity')
        if not scoped['cpu_aten']:
            reasons.append('no_scoped_aten_activity')
        conn.close()
    return {
        'schema_version': 2, 'kind': 'kineto_software_activity_audit',
        'parser_dependency': {'path': str(Path(legacy.__file__).resolve()), 'sha256': LEGACY_SHA256}, 'trace': {'path': str(trace), **trace_receipt},
        'full_generation_valid': not reasons, 'validity_reasons': sorted(set(reasons)),
        'scope': 'marker_bound' if use_range else 'whole_trace_unbound', 'calibration_only': calibration,
        'identity': {k: actual_sidecar.get(k) for k in ('nonce', 'run_id', 'arm', 'request_id', 'repeat', 'phase')},
        'process': actual_sidecar.get('process'), 'gpu': gpu, 'source_checks': source_checks,
        'gpu_uuid_binding': 'trace_and_sidecar' if trace_uuid else 'sidecar_receipt_only_not_independent_trace_uuid',
        'marker': marker, 'cpu_marker_candidates': markers,
        'gpu_annotations': {'counts': dict(gpu_annotation_names), 'full_generation_labels': gpu_marker_annotations,
            'scope': 'Range labels only; not CPU scope markers, kernels, memory activities or DRAM commands'},
        'trace_timestamp_unit': 'microseconds (Kineto ts/dur; displayTimeUnit is presentation only)', 'metadata': metadata, 'event_categories': dict(sorted(categories.items())),
        'raw_counts': _finish_counts(raw), 'scoped_counts': _finish_counts(scoped),
        'kernel_classification': {'counts': dict(kernel_roles), 'evidence': [dict(role=k[0], cpu_operator=k[1], kernel=k[2], count=v) for k,v in sorted(classified_names.items())],
                                  'method': 'launch correlation plus External id plus CPU nesting; never kernel-name classification'},
        'binding_counts': dict(binding_counts), 'binding_issue_examples': issue_examples,
        'bound_result': bound_result,
        'result_run_id_binding': ('row_and_sidecar' if bound_result is not None and 'run_id' in bound_result else 'sidecar_nonce_and_artifact_hash_only_row_has_no_run_id'),
        'limitations': ['ATen and API nesting is counted separately; no sum is an independent transfer count.',
                       'GPU memcpy payload and software calls are not DRAM bytes, requests, sectors or transactions.',
                       'Profiler timing and memory cannot establish ordinary throughput or peak memory.',
                       'Completeness uses marker, collector receipt and result; cannot rule out unreported profiler loss.',
                       'Source/UUID receipts bind supplied artifacts, not an independent attestation of the collector.'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--calibration', action='store_true')
    parser.add_argument('--compare-audit', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    audit = analyze_trace(args.trace, sidecar=args.sidecar, calibration=args.calibration, evidence_root=args.evidence_root)
    if args.compare_audit:
        audit['comparison'] = compare_audits(json.loads(args.compare_audit.read_text()), audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'full_generation_valid': audit['full_generation_valid'], 'reasons': audit['validity_reasons'], 'counts': audit['scoped_counts']['totals']}, ensure_ascii=False))
    return 0 if args.calibration or audit['full_generation_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
