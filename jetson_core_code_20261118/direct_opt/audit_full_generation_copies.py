"""CPU-only, streaming Kineto event accounting; software events are not DRAM transfers.

A versioned sidecar binds a future full-generation collection. Existing module traces
are usable only with --calibration. No torch/CUDA import, collector, or runner here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile


class TraceFormatError(ValueError):
    pass


class _Reasons(set):
    append = set.add


class _JSONStream:
    def __init__(self, handle, chunk_size=65536, max_value_bytes=16 * 1024 * 1024):
        self.handle, self.chunk_size, self.limit = handle, chunk_size, max_value_bytes
        self.buffer, self.pos, self.eof = '', 0, False
        self.decoder = json.JSONDecoder(parse_constant=lambda s: (_ for _ in ()).throw(TraceFormatError('nonfinite JSON: ' + s)))

    def _read(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0
        chunk = self.handle.read(self.chunk_size)
        self.buffer += chunk
        self.eof = not chunk

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ''
            self._read()

    def expect(self, char):
        if self.peek() != char:
            raise TraceFormatError('expected ' + repr(char))
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            if len(self.buffer) - self.pos > self.limit:
                raise TraceFormatError('individual JSON value exceeds configured bound')
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                # A scalar number may end exactly at an input chunk boundary.
                if end == len(self.buffer) and not self.eof:
                    self._read()
                    continue
                if end < len(self.buffer) and self.buffer[end] not in ' \t\r\n,:]}':
                    if self.eof:
                        raise TraceFormatError('invalid JSON value delimiter')
                    self._read()
                    continue
                self.pos = end
                return value
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise TraceFormatError('invalid or truncated JSON') from exc
                self._read()


def stream_trace(path, *, chunk_size=65536, max_value_bytes=16 * 1024 * 1024):
    """Yield ('event', dict) or ('metadata', (key,value)); never load traceEvents whole."""
    with Path(path).open(encoding='utf-8') as handle:
        reader = _JSONStream(handle, chunk_size, max_value_bytes)
        reader.expect('{')
        seen = set()
        if reader.peek() == '}':
            raise TraceFormatError('missing traceEvents')
        while True:
            key = reader.value()
            if not isinstance(key, str) or key in seen:
                raise TraceFormatError('invalid or duplicate top-level field')
            seen.add(key)
            reader.expect(':')
            if key == 'traceEvents':
                reader.expect('[')
                if reader.peek() != ']':
                    while True:
                        event = reader.value()
                        if not isinstance(event, dict):
                            raise TraceFormatError('trace event is not an object')
                        yield 'event', event
                        if reader.peek() == ']':
                            break
                        reader.expect(',')
                reader.expect(']')
            else:
                yield 'metadata', (key, reader.value())
            if reader.peek() == '}':
                break
            reader.expect(',')
        reader.expect('}')
        if reader.peek() or 'traceEvents' not in seen:
            raise TraceFormatError('trailing input or missing traceEvents')


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _identity(value):
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else None


COPY_OPS = {'aten::copy_', 'aten::clone', 'aten::contiguous'}
CAST_OPS = {'aten::to', 'aten::_to_copy', 'aten::type_as'}
GATHER_OPS = {'aten::index_select', 'aten::gather', 'aten::index', 'aten::take', 'aten::take_along_dim'}
CONCAT_OPS = {'aten::cat', 'aten::concat', 'aten::concatenate'}
GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}
CPU_CATEGORIES = {'cpu_op', 'cuda_runtime', 'cuda_driver'}
IGNORED_CATEGORIES = {'ac2g', 'cpu_instant_event', 'user_annotation', 'Trace', 'overhead', 'cuda_event', None}


def operator_role(name, inherited=None):
    """CPU operator semantics, not a kernel-name heuristic or inferred byte count."""
    if name in CAST_OPS:
        return 'cast_or_conversion'
    if name in GATHER_OPS:
        return 'gather_or_index'
    if name in CONCAT_OPS:
        return 'concat'
    if name in COPY_OPS:
        return inherited if inherited in {'cast_or_conversion', 'gather_or_index', 'concat'} else 'copy_or_cast'
    return 'unclassified'


def _work_api_family(name, category):
    """Recognized launch/memory API families; correlation alone is not a launch.

    Keep this list explicit. A future unrecognized API remains partial until
    actual trace/schema evidence establishes its activity family.
    """
    if category == 'cuda_runtime':
        kernel = r'cuda(?:LaunchKernel(?:ExC)?|LaunchCooperativeKernel)'
        memcpy = r'cudaMemcpy(?:(?:2D)?(?:ArrayToArray|FromArray|ToArray)|FromSymbol|ToSymbol|3DPeer|Peer|2D|3D)?(?:Async)?'
        memset = r'cudaMemset(?:2D|3D)?(?:Async)?'
    elif category == 'cuda_driver':
        kernel = r'cu(?:LaunchKernel(?:Ex)?|LaunchCooperativeKernel)'
        memcpy = r'cuMemcpy(?:DtoH|HtoD|DtoD|HtoA|AtoH|AtoD|DtoA|AtoA|2DUnaligned|2D|3DPeer|3D|Peer)?(?:Async)?'
        memset = r'cuMemset(?:D2D|D)(?:8|16|32)(?:Async)?'
    else:
        return None
    for family, pattern in (('kernel', kernel), ('gpu_memcpy', memcpy), ('gpu_memset', memset)):
        if re.fullmatch(pattern + r'(?:_v[0-9]+)?(?:_ptsz|_ptds)?', name):
            return family
    return None


def _empty_counts():
    return {'cpu_aten': Counter(), 'cpu_other_ops': Counter(), 'runtime_api': Counter(),
            'driver_api': Counter(), 'runtime_memcpy_api': Counter(), 'driver_memcpy_api': Counter(),
            'runtime_launch_api': Counter(), 'driver_launch_api': Counter(),
            'gpu_kernel': Counter(), 'gpu_memcpy': Counter(), 'gpu_memset': Counter(),
            'gpu_memcpy_directions': Counter(), 'gpu_memcpy_known_payload_bytes': 0,
            'gpu_memcpy_unknown_payload_events': 0}


def _count(counts, event):
    cat, name = event['cat'], event['name']
    if cat == 'cpu_op':
        counts['cpu_aten' if name.startswith('aten::') else 'cpu_other_ops'][name] += 1
    elif cat in {'cuda_runtime', 'cuda_driver'}:
        kind = 'runtime' if cat == 'cuda_runtime' else 'driver'
        counts[kind + '_api'][name] += 1
        if name.startswith('cudaMemcpy' if kind == 'runtime' else 'cuMemcpy'):
            counts[kind + '_memcpy_api'][name] += 1
        if name.startswith(('cudaLaunch','cudaGraphLaunch') if kind == 'runtime' else ('cuLaunch','cuGraphLaunch')):
            counts[kind + '_launch_api'][name] += 1
    elif cat in GPU_CATEGORIES:
        counts['gpu_kernel' if cat == 'kernel' else cat][name] += 1
        if cat == 'gpu_memcpy':
            args = json.loads(event['args'])
            direction = args.get('copy kind', args.get('Memcpy kind', args.get('direction', 'unknown')))
            counts['gpu_memcpy_directions'][str(direction)] += 1
            values = [args[k] for k in ('bytes', 'Bytes') if k in args]
            if values and len(set(map(str, values))) == 1 and isinstance(values[0], int) and not isinstance(values[0], bool) and values[0] >= 0:
                counts['gpu_memcpy_known_payload_bytes'] += values[0]
            else:
                counts['gpu_memcpy_unknown_payload_events'] += 1


def _finish_counts(counts):
    out = {k: dict(sorted(v.items())) if isinstance(v, Counter) else v for k, v in counts.items()}
    out['totals'] = {k: sum(v.values()) for k, v in counts.items() if isinstance(v, Counter)}
    return out


def result_evidence(row):
    """Support the saved run_joint row plus an explicitly nested diagnostic fixture."""
    payload = row.get('result', {})
    ids = row.get('output_token_ids', payload.get('token_ids', payload.get('output_token_ids')))
    return {'output_token_ids': ids, 'stop_reason': row.get('stop_reason', payload.get('stop_reason')),
            'truncated': row.get('truncated', payload.get('truncated')),
            **{k: payload.get(k) for k in ('rounds', 'forward_counts', 'precision')}}


def _load_bound_sidecar(sidecar, path, root, actual_trace, reasons):
    def check(ok, why):
        if not ok:
            reasons.append(why)
    check(sidecar.get('schema_version') == 1, 'sidecar_schema_version')
    check(sidecar.get('scope') == 'generate_joint', 'sidecar_scope')
    for key in ('nonce', 'run_id', 'arm', 'request_id'):
        check(isinstance(sidecar.get(key), str) and bool(sidecar[key]), 'sidecar_' + key)
    check(sidecar.get('phase') == 'measure' and isinstance(sidecar.get('repeat'), int), 'sidecar_cell')
    process, gpu, capture = (sidecar.get(k, {}) for k in ('process', 'gpu', 'capture'))
    check(isinstance(process.get('os_pid'), int) and process['os_pid'] > 0, 'sidecar_os_pid')
    check(isinstance(gpu.get('uuid'), str) and gpu['uuid'].startswith('GPU-'), 'sidecar_gpu_uuid')
    check(all(isinstance(gpu.get(k), int) and gpu[k] >= 0 for k in ('physical_device', 'logical_device')), 'sidecar_device_ordinals')
    check(isinstance(gpu.get('contexts'), list) and bool(gpu['contexts']) and all(_identity(v) is not None for v in gpu['contexts']), 'sidecar_contexts')
    for key in ('started', 'stopped', 'generation_returned', 'synchronized_inside_marker'):
        check(capture.get(key) is True, 'capture_' + key)
    for key in ('warmup_calls', 'profiled_calls', 'formal_call_index'):
        check(capture.get(key) == 1, 'capture_' + key)
    check(capture.get('dropped_events') == 0, 'capture_dropped_events')
    check(capture.get('warnings') == [], 'capture_warnings')
    check(isinstance(sidecar.get('trace'), dict) and all(sidecar['trace'].get(k) == v for k,v in actual_trace.items()), 'trace_hash_or_bytes_binding')
    evidence_root = Path(root) if root else Path(path).parent

    def resolve(name):
        p = Path(name)
        return p if p.is_absolute() else evidence_root / p

    sources = sidecar.get('sources')
    checks = []
    if not isinstance(sources, dict) or not sources:
        reasons.append('source_receipts_missing')
    else:
        for name, digest in sources.items():
            try:
                found = file_hash(resolve(name))
                ok = bool(re.fullmatch('[0-9a-f]{64}', str(digest))) and found == digest
                checks.append({'path': name, 'sha256': found, 'matches': ok})
                check(ok, 'source_hash_mismatch:' + name)
            except (OSError, TypeError, ValueError):
                reasons.append('source_unavailable:' + str(name))
    result = None
    binding = sidecar.get('result', {})
    try:
        result_path = resolve(binding['path'])
        check(file_hash(result_path) == binding['sha256'], 'result_hash_mismatch')
        index = binding['row_index']
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError('row_index')
        with result_path.open() as handle:
            for i, line in enumerate(handle):
                if i == index:
                    result = json.loads(line)
                    break
        if not isinstance(result, dict):
            raise ValueError('row_missing')
        for key in ('arm', 'request_id', 'repeat', 'phase'):
            check(result.get(key) == sidecar.get(key), 'result_cell_' + key)
        # Frozen run_joint rows do not carry run_id. Do not invent one in their raw data.
        if 'run_id' in result:
            check(result['run_id'] == sidecar.get('run_id'), 'result_run_id')
        payload = result_evidence(result)
        ids = payload['output_token_ids']
        check(isinstance(ids, list) and bool(ids) and all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in ids), 'result_output_ids_missing')
        for ids_key in ('token_ids', 'output_token_ids'):
            if ids_key in result.get('result', {}):
                check(result['result'][ids_key] == ids, 'result_output_ids_disagree')
        if 'stop_reason' in result and 'stop_reason' in result.get('result', {}):
            check(result['stop_reason'] == result['result']['stop_reason'], 'result_stop_reason_disagrees')
        if 'num_tokens' in result:
            check(isinstance(ids,list) and result['num_tokens'] == len(ids), 'result_output_length_disagrees')
        check(payload.get('stop_reason') == 'eos' and payload.get('truncated') is False, 'result_not_natural_eos')
        check(isinstance(payload.get('rounds'), list) and bool(payload['rounds']), 'result_rounds_missing')
        check(isinstance(payload.get('forward_counts'), dict) and bool(payload['forward_counts']), 'result_forward_counts_missing')
    except (KeyError, OSError, ValueError, TypeError):
        reasons.append('result_artifact_unavailable_or_invalid')
    return checks, result


def analyze_trace(trace, *, sidecar=None, calibration=False, evidence_root=None, chunk_size=65536):
    """Return raw/scoped counts and explicit validity; malformed JSON raises, defects retain raw."""
    trace = Path(trace)
    trace_receipt = {'sha256': file_hash(trace), 'bytes': trace.stat().st_size}
    metadata, categories, reasons, markers = {}, Counter(), _Reasons(), []
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
                markers.append(event)
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
                    and _finite(candidate.get('ts')) and _finite(candidate.get('dur')) and candidate['dur'] > 0):
                marker = candidate
            else:
                reasons.append('full_generation_marker_shape')
        pid = _identity(actual_sidecar.get('process', {}).get('os_pid'))
        gpu = actual_sidecar.get('gpu', {})
        if marker:
            if marker['name'] != 'FULL_GENERATION/' + str(actual_sidecar.get('nonce')):
                reasons.append('marker_nonce_mismatch')
            if _identity(marker.get('pid')) != pid:
                reasons.append('marker_cpu_pid_mismatch')
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
        'schema_version': 1, 'kind': 'kineto_software_activity_audit', 'trace': {'path': str(trace), **trace_receipt},
        'full_generation_valid': not reasons, 'validity_reasons': sorted(set(reasons)),
        'scope': 'marker_bound' if use_range else 'whole_trace_unbound', 'calibration_only': calibration,
        'identity': {k: actual_sidecar.get(k) for k in ('nonce', 'run_id', 'arm', 'request_id', 'repeat', 'phase')},
        'process': actual_sidecar.get('process'), 'gpu': gpu, 'source_checks': source_checks,
        'gpu_uuid_binding': 'trace_and_sidecar' if trace_uuid else 'sidecar_receipt_only_not_independent_trace_uuid',
        'marker': marker, 'trace_timestamp_unit': 'microseconds (Kineto ts/dur; displayTimeUnit is presentation only)', 'metadata': metadata, 'event_categories': dict(sorted(categories.items())),
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


def compare_audits(left, right):
    """Keep raw counts; never repair a changed answer by length normalization."""
    a, b = left.get('bound_result'), right.get('bound_result')
    equality = {}
    input_keys = ('input', 'input_fingerprint', 'input_shapes', 'video_grid_thw', 'timestamps')
    if isinstance(a, dict) and isinstance(b, dict):
        equality['request_id'] = a.get('request_id') == b.get('request_id')
        ae, be = result_evidence(a), result_evidence(b)
        for key in ('output_token_ids', 'rounds', 'forward_counts', 'precision', 'stop_reason', 'truncated'):
            av, bv = ae.get(key), be.get(key)
            equality[key] = av == bv if av is not None and bv is not None else None
        # Input schema may differ across collectors; missing proof never becomes equality.
        for key in input_keys:
            equality[key] = a[key] == b[key] if key in a and key in b else None
        for key in ('input', 'input_fingerprint'):
            if key in a and key in b and a[key] in (None, '', {}, []):
                equality[key] = None if a[key] == b[key] else False
    conflicts = [key for key in input_keys if equality.get(key) is False]
    strict = not conflicts and bool(equality) and all(equality.get(k) is True for k in ('request_id','output_token_ids','rounds','forward_counts','precision','stop_reason','truncated')) and any(equality.get(k) is True for k in ('input','input_fingerprint'))
    return {'same_workload_proven': strict, 'full_count_comparison_eligible': strict and left.get('full_generation_valid') is True and right.get('full_generation_valid') is True,
            'equality': equality, 'input_evidence_conflicts': conflicts, 'left_raw_counts': left['raw_counts'], 'right_raw_counts': right['raw_counts'],
            'left_scoped_counts': left['scoped_counts'], 'right_scoped_counts': right['scoped_counts'],
            'physical_dram_transfer_count': None, 'normalization_or_extrapolation': None}


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
