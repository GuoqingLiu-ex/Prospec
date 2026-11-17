"""Explicit ctypes binding and loss-aware parser for the CUDA12.8 copy recorder.

Counts GPU memcpy/memset activities, including memcpy copyCount. It does not
count copy/cast/gather kernels, DRAM commands, or API calls and never supplies
throughput/energy evidence. No CUDA/library initialization happens on import.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path

SCHEMA = 'cupti_copy_activity_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CopyActivity:
    def __init__(self, library, raw_path):
        self.library = Path(library).resolve()
        self.raw_path = Path(raw_path).resolve()
        if self.raw_path.exists():
            raise FileExistsError('fresh activity output required')
        self.lib = ctypes.CDLL(str(self.library))
        self.lib.vcp_initialize.argtypes = [ctypes.c_char_p]
        self.lib.vcp_begin.argtypes = self.lib.vcp_end.argtypes = [ctypes.c_uint64]
        self.lib.vcp_close.argtypes = []
        for name in ('vcp_initialize', 'vcp_begin', 'vcp_end', 'vcp_close'):
            getattr(self.lib, name).restype = ctypes.c_int
        self.active = None
        self.closed = False
        self.initialized = False
        self.source = dict(library_sha256=digest(self.library), binding_sha256=digest(__file__))

    def initialize(self):
        if self.initialized or self.closed:
            raise RuntimeError('collector already initialized or closed')
        # On failure, close may still preserve the C recorder's partial file.
        result = self.lib.vcp_initialize(str(self.raw_path).encode())
        self.initialized = True
        if result:
            raise RuntimeError(f'CUPTI initialize failed: {result}')

    def begin(self, torch, identity):
        if type(identity) is not int or identity <= 0 or not self.initialized or self.active is not None or self.closed:
            raise ValueError('positive identity and idle initialized collector required')
        torch.cuda.synchronize()
        result = self.lib.vcp_begin(identity)
        if result:
            raise RuntimeError(f'CUPTI begin failed: {result}')
        self.active = identity

    def end(self, torch):
        if self.active is None or self.closed:
            raise RuntimeError('no open activity range')
        torch.cuda.synchronize()
        identity = self.active
        result = self.lib.vcp_end(identity)
        self.active = None
        if result:
            raise RuntimeError(f'CUPTI end failed: {result}')

    def close(self):
        if self.closed:
            return
        if not self.initialized:
            self.closed = True
            return
        result = self.lib.vcp_close()
        self.closed = True
        if result:
            raise RuntimeError(f'CUPTI close failed: {result}')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_records(rows, *, expected_pid, expected_device=0):
    """All loss/error counters and synchronized timestamp bounds must be valid.

    This validates only the enabled activity kinds. It cannot infer unrecorded
    copy kernels or prove total DRAM traffic from memcpy payload.
    """
    _require(rows and all(isinstance(r, dict) for r in rows), 'nonempty activity records required')
    manifests = [r for r in rows if r.get('kind') == 'manifest']
    summaries = [r for r in rows if r.get('kind') == 'summary']
    _require(len(manifests) == len(summaries) == 1, 'one manifest and one final summary required')
    m, summary = manifests[0], summaries[0]
    _require(m.get('pid') == expected_pid and m.get('cupti_api_version') == 26
             and m.get('memcpy_layout') == 'Memcpy6' and m.get('memset_layout') == 'Memset4',
             'process or CUPTI record layout mismatch')
    _require(rows[-1] is summary and rows[0] is m, 'manifest/final-summary ordering invalid')
    for name in ('dropped_records', 'drop_query_errors', 'api_errors', 'parse_errors',
                 'invalid_timestamps', 'unknown_records', 'allocation_errors', 'write_errors'):
        _require(type(summary.get(name)) is int and summary[name] == 0, 'invalid recorder counter: ' + name)
    _require(type(summary.get('drop_queries')) is int and summary['drop_queries'] > 0,
             'drop counter was not queried')
    _require(type(summary.get('buffers_requested')) is int and summary['buffers_requested'] > 0
             and type(summary.get('buffers_completed')) is int
             and summary['buffers_requested'] == summary.get('buffers_completed'),
             'activity buffers were not all delivered')
    _require(all(r.get('kind') in ('manifest', 'summary', 'begin', 'end', 'memcpy', 'memset') for r in rows),
             'unknown/error activity record')
    boundaries = [r for r in rows if r.get('kind') in ('begin', 'end')]
    _require(boundaries and len(boundaries) % 2 == 0, 'missing range boundary')
    ranges, used_ids = [], set()
    for begin, end in zip(boundaries[::2], boundaries[1::2]):
        identity = begin.get('id')
        _require(begin['kind'] == 'begin' and end['kind'] == 'end' and type(identity) is int
                 and identity > 0 and identity == end.get('id') and identity not in used_ids,
                 'range identities/order invalid')
        _require(type(begin.get('timestamp')) is int and type(end.get('timestamp')) is int
                 and 0 < begin['timestamp'] < end['timestamp'] and end.get('flushed') is True,
                 'range endpoint/flush invalid')
        _require(type(begin.get('context')) is int and begin['context'] > 0
                 and begin['context'] == end.get('context') and
                 begin.get('device') == end.get('device') == expected_device,
                 'range context/device invalid')
        if ranges:
            _require(ranges[-1]['end'] <= begin['timestamp'], 'ranges overlap')
        used_ids.add(identity)
        ranges.append(dict(id=identity, begin=begin['timestamp'], end=end['timestamp'],
                           context=begin['context'], device=expected_device,
                           memcpy_records=0, memcpy_operations=0, memcpy_bytes=0,
                           memset_records=0, memset_bytes=0, directions={}))
    outside = 0
    for record in rows:
        kind = record['kind']
        if kind not in ('memcpy', 'memset'):
            continue
        start, end = record.get('start'), record.get('end')
        _require(type(start) is int and type(end) is int and 0 < start <= end,
                 'invalid activity timestamp')
        _require(type(record.get('bytes')) is int and record['bytes'] >= 0, 'invalid payload size')
        matches = [r for r in ranges if r['begin'] <= start and end <= r['end']]
        _require(not any(start < r['begin'] < end or start < r['end'] < end for r in ranges),
                 'activity crosses synchronized range boundary')
        if not matches:
            outside += 1
            continue
        _require(len(matches) == 1, 'ambiguous activity range')
        r = matches[0]
        _require(record.get('context') == r['context'] and record.get('device') == r['device'],
                 'activity belongs to another context/device')
        if kind == 'memcpy':
            count = record.get('copy_count')
            _require(type(count) is int and count > 0, 'missing memcpy copyCount')
            direction = record.get('copy_kind')
            _require(type(direction) is int and direction in range(1, 11), 'unknown memcpy direction')
            d = r['directions'].setdefault(str(direction), dict(records=0, operations=0, bytes=0))
            d['records'] += 1; d['operations'] += count; d['bytes'] += record['bytes']
            r['memcpy_records'] += 1; r['memcpy_operations'] += count; r['memcpy_bytes'] += record['bytes']
        else:
            r['memset_records'] += 1; r['memset_bytes'] += record['bytes']
    return dict(schema=SCHEMA, enabled_activity_complete=True, pid=expected_pid,
                recorded_ranges=ranges, outside_range_activity_records=outside, counters=summary,
                GPU_memcpy_scope='CUPTI memcpy records and copyCount; kernel-based copies excluded',
                DRAM_measured=False, throughput_measured=False)


def audit_file(path, **kwargs):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    result = audit_records(rows, **kwargs)
    return dict(result, raw_sha256=hashlib.sha256(raw).hexdigest())
