"""Offline grouped/no-discarded-weights request audit; no Torch or GPU imports.

One warmup, two formal repeats, one or twelve requests, natural EOS/max512.
All negative observations survive in JSON. Only two method-specific forward
counter names are excluded from cross-arm equality. Logical avoided tensor
payload is never a GPU kernel count, DRAM measurement or capacity saving.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path

from assess_visual_kv import generation
from audit_grouped_generation import background as legacy_background, coverage as grouped_coverage
from audit_native_delta_abba import aggregate
from audit_stride0_pilot import FLAGS, digest

METHOD = 'grouped_no_discarded_weights'
METHOD_COUNTS = {'target_grouped_attention_calls',
                 'target_grouped_no_discarded_weights_attention_calls'}
COMMON_SOURCES = ('direct_opt/run_joint.py', 'direct_opt/run_memory_layout.py',
    'direct_opt/run_deployed_adaptor.py', 'direct_opt/compiled_adaptor.py',
    'direct_opt/shared_embedding.py', 'direct_opt/stride0_gqa_runtime.py',
    'direct_opt/stride0_gqa_math.py', 'direct_opt/grouped_gqa_runtime.py',
    'direct_opt/grouped_gqa_math.py', 'direct_opt/run_grouped_gqa_layout.py',
    'vendor/hf517_vispec_adapter.py', 'vendor/vispec_joint_adapter.py')
NEW_SOURCES = ('direct_opt/run_grouped_no_weights_layout.py',
    'direct_opt/grouped_gqa_no_weights_runtime.py',
    'direct_opt/grouped_gqa_no_weights_math.py')
TIMES = ('e2e_wall_s', 'generation_wall_s', 'model_wall_s', 'prepare_wall_s',
         'policy_prepare_wall_s', 'output_decode_wall_s', 'runtime_finalize_wall_s')
BASE = Path(__file__).resolve().parent.parent
# Coarse, predeclared coverage checks for the frozen 0.5-second process sampler.
# Passing these checks is not evidence of exclusive execution between samples.
PROCESS_ENDPOINT_LIMIT_S, PROCESS_GAP_LIMIT_S = 1.5, 2.0


def integer(value):
    return type(value) is int and value >= 0


def finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)


def sample_coverage(row):
    """Reconcile actual old-schema query records, never trust totals alone.

    A query is a sample with process_query_complete exactly True or False.
    Missing/null marks ordinary telemetry, not an additional process query.
    """
    h=row.get('hardware',{});samples=h.get('samples');duration=h.get('measurement_duration_s')
    checks={};first=last=gap=None;queries=[];observed=set()
    checks['samples_present']=isinstance(samples,list) and len(samples)>=2
    if not isinstance(samples,list):samples=[]
    checks['sample_objects']=all(isinstance(s,dict) for s in samples)
    samples=[s for s in samples if isinstance(s,dict)]
    elapsed=[s.get('elapsed_s') for s in samples]
    checks['finite_strict_sample_times']=bool(elapsed) and all(finite_number(t) and t>=0 for t in elapsed) and all(b>a for a,b in zip(elapsed,elapsed[1:]))
    checks['frozen_sampling_intervals']=h.get('sample_interval_s')==0.1 and h.get('process_interval_s')==0.5
    checks['finite_duration']=finite_number(duration) and duration>0
    checks['sample_range']=checks['finite_duration'] and checks['finite_strict_sample_times'] and all(t<=duration for t in elapsed)
    checks['query_markers_valid']=all(s.get('process_query_complete') is None or type(s.get('process_query_complete')) is bool for s in samples)
    checks['foreign_lists_only_at_queries']=all('foreign_processes' not in s or type(s.get('process_query_complete')) is bool for s in samples)
    queries=[s for s in samples if type(s.get('process_query_complete')) is bool]
    checks['at_least_two_queries']=len(queries)>=2
    checks['query_count_matches']=integer(h.get('process_query_count')) and h['process_query_count']==len(queries)
    failures=sum(s['process_query_complete'] is False for s in queries)
    checks['failure_count_matches']=integer(h.get('process_query_failures')) and h['process_query_failures']==failures
    checks['all_queries_complete']=bool(queries) and failures==0
    lists=[s.get('foreign_processes') for s in queries]
    lists_ok=all(isinstance(ps,list) and all(isinstance(p,dict) and integer(p.get('pid')) and p['pid']>0 for p in ps) for ps in lists)
    checks['query_process_lists_valid']=lists_ok
    if lists_ok:
        observed={p['pid'] for ps in lists for p in ps}
    summary=h.get('foreign_processes')
    summary_ok=isinstance(summary,list) and all(isinstance(p,dict) and integer(p.get('pid')) and p['pid']>0 for p in summary)
    checks['foreign_summary_matches_samples']=lists_ok and summary_ok and observed=={p['pid'] for p in summary}
    checks['optional_sample_identity']=all(
        ('gpu_uuid' not in s or s['gpu_uuid']==h.get('device',{}).get('uuid')) and
        ('physical_device_index' not in s or s['physical_device_index']==h.get('physical_device_index')) for s in samples)
    qtimes=[s.get('elapsed_s') for s in queries]
    if len(qtimes)>=2 and checks['sample_range']:
        first=qtimes[0];last=duration-qtimes[-1];gap=max(b-a for a,b in zip(qtimes,qtimes[1:]))
    checks['query_time_coverage']=first is not None and first<=PROCESS_ENDPOINT_LIMIT_S and last<=PROCESS_ENDPOINT_LIMIT_S and gap<=PROCESS_GAP_LIMIT_S
    try:
        row_start=datetime.fromisoformat(row['started_at']).timestamp()
        sampler_start=h['started_unix_s'];wall=row['e2e_wall_s']
        # The row timestamp has whole-second resolution. Sampler start precedes
        # its forced initial query, and sampler stop follows request completion.
        checks['request_interval_bound']=(finite_number(sampler_start) and finite_number(wall) and wall>0 and
            checks['finite_duration'] and -1.0<=row_start-sampler_start<=PROCESS_ENDPOINT_LIMIT_S and
            -1e-6<=duration-wall<=PROCESS_ENDPOINT_LIMIT_S)
    except (KeyError,TypeError,ValueError,OverflowError):
        checks['request_interval_bound']=False
    return dict(passed=all(checks.values()),checks=checks,sample_count=len(samples),actual_query_count=len(queries),
        actual_failed_queries=failures,first_query_gap_s=first,last_query_gap_s=last,max_query_gap_s=gap,
        endpoint_limit_s=PROCESS_ENDPOINT_LIMIT_S,internal_gap_limit_s=PROCESS_GAP_LIMIT_S,
        observed_query_foreign_pids=sorted(observed),
        scope='Coarse discrete coverage of the sampler interval surrounding the request; not proof of exclusive residency.')


def background(row,own_pid,known_graphics):
    receipt=legacy_background(row,own_pid,known_graphics)
    coverage=sample_coverage(row)
    receipt['sample_coverage']=coverage
    receipt['sampled_background_clear']=bool(receipt['sampled_background_clear'] and coverage['passed'])
    return receipt


def common_counts(row):
    counts=row.get('result',{}).get('forward_counts')
    return {k:v for k,v in counts.items() if k not in METHOD_COUNTS} if isinstance(counts,dict) else None


def compare(a, b):
    fields = ('output_token_ids', 'text', 'stop_reason', 'truncated', 'num_tokens',
              'input_shapes', 'video_grid_thw', 'timestamps')
    checks = {k:k in a and k in b and a[k] == b[k] for k in fields}
    x, y = a.get('result', {}), b.get('result', {})
    for key in ('rounds', 'prompt_tokens', 'target_cache_length', 'budget_trace', 'visual_context'):
        checks['result.'+key] = key in x and key in y and x[key] == y[key]
    checks['common_forward_counts'] = (isinstance(x.get('forward_counts'), dict) and
        isinstance(y.get('forward_counts'), dict) and common_counts(a) == common_counts(b))
    return checks


def scalar_tree(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(k,str) and scalar_tree(v) for k,v in value.items())
    return isinstance(value,list) and all(scalar_tree(v) for v in value)


def _candidate_coverage(row):
    info = row['adaptor_preparation']['tree_attention']
    counts = row['result']['forward_counts']
    trees = [r for r in row['result']['rounds'] if r['kind']=='tree']
    rounds = info['rounds']; n = len(trees)
    elements = 28 * 28 * sum(t['tree_nodes']*(t['target_cache_before']+t['tree_nodes']) for t in trees)
    expected = dict(elements=elements, read_bytes=4*elements, write_bytes=2*elements,
                    read_plus_write_bytes=6*elements, counterfactual_software_cast_calls=28*n)
    # The inherited table compares grouped with native repeat-KV math, NOT with
    # the grouped control. Keep it separate from the new discarded-cast table.
    e = 28*4*128*sum(t['target_cache_before']+t['tree_nodes'] for t in trees)
    q = 28*4*28*128*sum(t['tree_nodes'] for t in trees)
    kv = dict(read_bytes=76*e, write_bytes=100*e, read_plus_write_bytes=176*e)
    qp = dict(read_bytes=q, write_bytes=q, read_plus_write_bytes=2*q)
    net = {k:kv[k]-qp[k] for k in kv}
    inherited = info['logical_copy_totals']
    removed = info['logical_discarded_probability_copy_totals']
    precision = dict(target='float16', output_head='float32', draft='float16', draft_sdpa='auto',
        target_sdpa='math_prefill+grouped_no_discarded_weights_fp32_math_tree',
        target_prefill_attention_backend='math',
        target_verify_attention_backend='grouped_two_bmm_no_discarded_weights_fp32_math',
        target_tree_attention_mode=METHOD, target_tree_math_precision=FLAGS,
        target_cache_policy='native_full_dynamic_cache', draft_attention_backend='unchanged',
        tree_attention_discarded_probability_cast_removed=True,
        tree_attention_bitwise_equivalence_claim=False)
    rep = info['representative_receipt']
    checks = dict(
        method=info['mode']==info['method']==info['implementation']==METHOD==row['adaptor_preparation']['tree_attention_mode'],
        primitive_identity=info['primitive_implementation']=='grouped_two_bmm_no_discarded_weights' and info['discarded_probability_cast_removed'] is True,
        finalized=info['finalized'] is True and info['failures']==[] and info['fallback_calls']==0,
        coverage_validated=info['result_coverage_validated'] is True,
        native_prefill_once=info['prefill_calls']==info['prefill_forward_calls']==1 and info['prefill_layer_calls']==[1]*28,
        all_verifications=info['verify_forward_calls']==counts['target_verify_calls']==n and info['verify_query_tokens']==counts['target_verify_query_tokens'],
        all_layers=info['layer_calls']==[n]*28 and info['selected_calls']==info['completed_calls']==28*n==counts['target_tree_attention_policy_calls'],
        candidate_calls=info['candidate_calls']==counts['target_grouped_no_discarded_weights_attention_calls']==28*n and
            info['native_tree_calls']==0 and 'target_grouped_attention_calls' not in counts and counts.get('target_stride0_attention_calls',0)==0,
        every_round=len(rounds)==n and all(r['cache_before']==t['target_cache_before'] and
            r['tree_nodes']==t['tree_nodes']==30 and r['cache_after']==r['cache_before']+r['tree_nodes'] and
            r['layer_calls']==[1]*28 and r['mask_shape']==[1,1,r['tree_nodes'],r['cache_after']] and
            r['position_shape']==[3,1,r['tree_nodes']] and r['mask_dtype']=='torch.float32' and
            r['position_dtype']=='torch.int64' for r,t in zip(rounds,trees)),
        full_cache=info['target_cache_policy']=='native_full_dynamic_cache' and info['new_kv_policy_installed'] is False and
            info['cache_functions_changed'] is False and info['native_cache_type']=='transformers.cache_utils.DynamicCache' and
            info['native_cache_layer_types']==['transformers.cache_utils.DynamicLayer']*28,
        mask_positions_rotary=info['original_mask_reused'] is True and info['position_ids_passed_unchanged'] is True and info['rotary_reapplied'] is False,
        precision=row['result']['precision']==precision and info['precision']==FLAGS,
        no_debug_or_measurement_claim=info['debug_intermediates'] is False and info['actual_dram_measured'] is False and info['actual_kernel_count_measured'] is False,
        first_only_receipt=info['primitive_receipt_calls']==int(n>0) and info['ordinary_primitive_record_receipt'] is False and
            info['maximum_representative_receipts_per_request']==1 and info['first_tree_call_receipt_in_request'] is True and
            info['summary_dictionary_prepared_in_runtime_finalization'] is True and bool(rep)==bool(n),
        inherited_grouped_copy_formula=inherited['kv_only_removed_explicit_bytes']==kv and inherited['added_query_pack_bytes']==qp and
            inherited['net_removed_explicit_bytes']==net and inherited['query_pack_calls']==28*n,
        inherited_copy_scope=inherited['actual_dram_measured'] is False and inherited['blas_internal_copy_measured'] is False and
            inherited['attention_mac_reduction']==inherited['kv_resident_capacity_reduction']==0,
        discarded_copy_formula=all(removed.get(k)==v for k,v in expected.items()),
        discarded_copy_scope=all(removed[k] is False for k in ('actual_kernel_count_measured','actual_dram_measured','throughput_measured')) and
            removed['attention_major_mac_reduction']==removed['kv_resident_capacity_reduction']==0)
    if rep:
        t, s = trees[0]['tree_nodes'], trees[0]['target_cache_before']+trees[0]['tree_nodes']
        rq = rep['input']['query']; rk = rep['input']['key']; rv = rep['input']['value']; rm = rep['input']['mask']
        checks['first_receipt_layout'] = (scalar_tree(rep) and rep['policy_method']==METHOD and
            rep['implementation']=='grouped_two_bmm_no_discarded_weights' and rep['query_layout_checked_by_policy'] is True and
            rep['query_pack_copy'] is True and rq['shape']==[1,28,t,128] and rq['stride']==[28*t*128,128,28*128,1] and
            rq['dtype']=='torch.float16' and rq['contiguous'] is False and rk['shape']==rv['shape']==[1,4,s,128] and
            rk['dtype']==rv['dtype']=='torch.float16' and rm['shape']==[1,1,t,s] and rm['dtype']=='torch.float32' and
            rep['original_mask_object'] is True and rep['mask_values_transformed'] is False and rep['qk_bmm_calls']==rep['pv_bmm_calls']==1 and
            rep['explicit_repeat_kv'] is False and rep['debug_intermediates'] is False and
            rep['native_unused_probability_cast_preserved'] is False and rep['discarded_probability_materialization'] is False)
        checks['first_receipt_removed_copy'] = all(rep['removed_discarded_probability_copy'][k]==v for k,v in
            dict(elements=28*t*s,read_bytes=4*28*t*s,write_bytes=2*28*t*s,read_plus_write_bytes=6*28*t*s).items())
    return dict(passed=all(checks.values()), checks=checks, verify_calls=n, attention_layer_calls=28*n,
        recomputed_removed_probability_copy=expected, logical_discarded_probability_copy_totals=removed,
        inherited_grouped_copy_totals=inherited)


def coverage(row, mode):
    try:
        if mode==METHOD:
            return _candidate_coverage(row)
        if mode!='grouped':
            raise ValueError('unknown mode')
        result=grouped_coverage(row,'grouped')
        result['checks']['no_new_method_counter'] = 'target_grouped_no_discarded_weights_attention_calls' not in row['result']['forward_counts']
        result['passed']=all(result['checks'].values())
        return result
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return dict(passed=False, checks={}, error=f'{type(exc).__name__}: {exc}')


def safe_aggregate(rows):
    try:
        if not rows or any(not integer(r.get('num_tokens')) or not all(
            type(r.get(k)) in (int,float) and math.isfinite(r[k]) and r[k]>0 for k in ('e2e_wall_s','generation_wall_s')) for r in rows):
            raise ValueError('missing/invalid formal tokens or times')
        return aggregate(rows)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return dict(error=str(exc))


def read_run(path, mode, known_graphics=(), source_root=BASE):
    path=Path(path)
    status=json.loads((path/'status.json').read_text())
    rows=[json.loads(x) for x in (path/'results.jsonl').read_text().splitlines() if x.strip()]
    requests=json.loads((path/'requests.json').read_text());schedule=json.loads((path/'schedule.json').read_text())
    terminal_path=Path(str(path)+'.exit.json')
    terminal=json.loads(terminal_path.read_text())
    args=status.get('args',{}); arm='memory_shared_embedding_compiled_fullkv_tree_'+mode+'_math'
    key=lambda r:tuple(r.get(k) for k in ('request_id','arm','phase','repeat'))
    protocol=dict(terminal=status.get('status')=='completed' and terminal.get('returncode')==0,
        process_identity=integer(status.get('pid')) and status['pid']>0,
        lock_identity=status.get('lock',{}).get('pid')==status.get('pid') and status.get('lock',{}).get('gpu')==args.get('physical_gpu'),
        fixed_protocol=args.get('warmup')==1 and args.get('repeats')==2 and args.get('max_new_tokens')==512 and args.get('resume') is False,
        request_count=len(requests) in (1,12) and len({r['request_id'] for r in requests})==len(requests),
        cell_count=len(rows)==status.get('completed_cells')==status.get('total_cells')==1+2*len(requests),
        unique_schedule=len({key(r) for r in rows})==len(rows) and list(map(key,rows))==list(map(key,schedule)),
        arms={r.get('arm') for r in rows}=={arm},
        warmup=bool(rows and requests) and rows[0].get('phase')=='warmup' and rows[0].get('repeat')==0 and rows[0].get('request_id')==requests[0]['request_id'],
        formal_coverage=all(r.get('phase')=='measure' for r in rows[1:]) and
            {(r.get('request_id'),r.get('repeat')) for r in rows[1:]}=={(r['request_id'],i) for r in requests for i in range(2)},
        deployment=status.get('tree_attention_mode')==mode and status.get('memory_layout')=='shared_embedding' and
            status.get('deployment_adaptor')=='compiled' and status.get('target_cache_policy')=='native_full_dynamic_cache' and
            status.get('new_kv_policy_installed') is False and status.get('tree_attention_precision')==FLAGS,
        no_fallback_quantization_eviction=status.get('attention_backend_fallback') is False and
            status.get('kv_quantized') is False and status.get('target_history_eviction') is False)
    if mode==METHOD:
        protocol['new_status_identity']=(status.get('discarded_probability_cast_removed') is True and
            status.get('methodology')=='same_shared_compiled_fullkv_grouped_vs_grouped_no_discarded_weights' and
            status.get('actual_dram_measured') is False)
    sources=COMMON_SOURCES+(NEW_SOURCES if mode==METHOD else ())
    source_checks={}
    for name in sources:
        expected=status.get('source_sha256',{}).get(name)
        try:
            actual=digest(Path(source_root)/name)
            source_checks[name]=dict(expected=expected,actual=actual,passed=bool(expected) and expected==actual)
        except OSError as exc:
            source_checks[name]=dict(expected=expected,passed=False,error=str(exc))
    details=[]
    for row in rows:
        try:
            gen=generation(row,status);cov=coverage(row,mode)
            components=sum(row[k] for k in ('generation_wall_s','output_decode_wall_s','runtime_finalize_wall_s'))
            timing=dict(finite_nonnegative=all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0 for k in TIMES),
                model_includes_finalization=math.isclose(row['model_wall_s'],components,abs_tol=1e-6),
                e2e_includes_all_phases=math.isclose(row['e2e_wall_s'],components+row['prepare_wall_s']+row['policy_prepare_wall_s'],abs_tol=1e-3))
            scheme=row['scheme'];recipe=(scheme==dict(depth='fixed0',adaptor='compiled',compression=None,ratio=1.0,
                tree_attention=mode,target_cache_policy='native_full_dynamic_cache') and row['capture_enabled'] is False and row.get('under_ncu',False) is False)
            hw=background(row,status['pid'],known_graphics)
            peaks={k:row.get(k) for k in ('peak_allocated_bytes','peak_reserved_bytes')}
            details.append(dict(request_id=row['request_id'],phase=row['phase'],repeat=row['repeat'],generation=gen,
                coverage=cov,timing_checks=timing,recipe_pass=recipe,timing={k:row[k] for k in TIMES},
                whole_request_peaks=peaks,peak_receipts_valid=all(integer(v) and v>0 for v in peaks.values()),hardware=hw,
                row_physical_gpu_matches=row.get('physical_gpu')==args.get('physical_gpu')==row['hardware'].get('physical_device_index'),
                lock_uuid_matches=bool(row['hardware'].get('device',{}).get('uuid')) and row['hardware']['device']['uuid']==
                    status.get('lock',{}).get('initial_snapshot',{}).get('gpu_uuid')))
        except (KeyError,TypeError,ValueError,IndexError) as exc:
            details.append(dict(request_id=row.get('request_id'),phase=row.get('phase'),repeat=row.get('repeat'),error=f'{type(exc).__name__}: {exc}'))
    within=[]
    for request in requests:
        same=[r for r in rows if r.get('request_id')==request['request_id']]
        within.extend(dict(request_id=request['request_id'],phase=r.get('phase'),repeat=r.get('repeat'),checks=compare(same[0],r)) for r in same[1:])
    mechanical=(all(protocol.values()) and all(x['passed'] for x in source_checks.values()) and all(
        'error' not in d and d['generation']['receipt_valid'] and d['generation']['natural_eos'] and d['coverage']['passed'] and
        d['recipe_pass'] and d['peak_receipts_valid'] and d['row_physical_gpu_matches'] and d['lock_uuid_matches'] and all(d['timing_checks'].values()) for d in details))
    hw=[d['hardware'] for d in details if 'hardware' in d]
    review=dict(mode=mode,directory=str(path.resolve()),protocol_checks=protocol,source_checks=source_checks,
        mechanical_execution_pass=bool(mechanical),within_run_repeat_checks=within,
        all_within_run_repeats_exact=bool(within) and all(all(x['checks'].values()) for x in within),details=details,
        formal_aggregate=safe_aggregate([r for r in rows if r.get('phase')=='measure']),terminal=terminal,
        hardware_summary=dict(sampled_background_clear=len(hw)==len(rows) and bool(hw) and all(h['sampled_background_clear'] for h in hw),
            all_sample_coverage_pass=len(hw)==len(rows) and bool(hw) and all(h['sample_coverage']['passed'] for h in hw),
            raw_query_count=sum(h['sample_coverage']['actual_query_count'] for h in hw),
            endpoint_limit_s=PROCESS_ENDPOINT_LIMIT_S,internal_gap_limit_s=PROCESS_GAP_LIMIT_S,
            all_observed_foreign_pids=sorted({p for h in hw for p in h['all_observed_foreign_pids']}),
            foreign_except_known_graphics=sorted({p for h in hw for p in h['foreign_except_known_graphics']}),
            raw_valid_timing_counts=dict(Counter(str(h['valid_timing']) for h in hw)),known_graphics=list(known_graphics)),
        input_hashes={k:digest(path/k) for k in ('status.json','results.jsonl','requests.json','schedule.json')},
        terminal_sha256=digest(terminal_path))
    return dict(status=status,rows=rows,requests=requests,review=review)


def _safe_read(path,mode,known_graphics,source_root):
    try:
        return read_run(path,mode,known_graphics,source_root)
    except (OSError,ValueError,TypeError,KeyError,IndexError) as exc:
        # Never invent zero counters or omit a failed deployment.
        return dict(review=dict(mode=mode,directory=str(Path(path).resolve()),mechanical_execution_pass=False,
            input_error=f'{type(exc).__name__}: {exc}'))


def audit(control,candidate,known_graphics=(),source_root=BASE):
    a=_safe_read(control,'grouped',known_graphics,source_root)
    b=_safe_read(candidate,METHOD,known_graphics,source_root)
    report=dict(schema='no_weights_generation_audit_v1',tool_sha256=digest(__file__),
        runs=[x['review'] for x in (a,b)],mechanical_exact_comparison_pass=False,
        quality_pass=None,actual_dram_measured=False,actual_kernel_count_measured=False,
        common_counts_excluded=sorted(METHOD_COUNTS),comparison=None,
        limitations=[
            'One or twelve distinct videos, two formal repeats each; repeats are not independent videos.',
            'Two sequential deployments are exploratory; loading/setup are excluded, request finalization included.',
            'Only saved input metadata/manifest are available; decoded pixel byte identity is not proven.',
            'Exact IDs/text/full trajectories reproduce these requests, not a general task-quality guarantee.',
            'Raw valid_timing flags and all sampled foreign processes are retained; sampling cannot prove exclusivity.',
            'Actual query records/counts must agree, endpoints <=1.5s and internal process-query gaps <=2.0s; coarse coverage is not isolation.',
            'HardwareSampler combines compute/graphics PIDs; supplied graphics identity is external evidence.',
            '6 bytes per removed probability element is logical payload, not GPU kernel count or DRAM traffic.',
            'Whole-request allocator peaks include prefill; no persistent KV or major matrix-MAC reduction is claimed.'])
    if any('rows' not in x for x in (a,b)):
        return report
    config={k:a['status'].get(k) is not None and a['status'].get(k)==b['status'].get(k) for k in
        ('target','draft','torch','transformers','request_sha256','device_name')}
    config.update({'args.'+k:a['status']['args'].get(k)==b['status']['args'].get(k) for k in
        ('seed','max_new_tokens','warmup','repeats')})
    config['same_manifest']=a['requests']==b['requests']
    config.update({'source.'+k:bool(a['status'].get('source_sha256',{}).get(k)) and
        a['status'].get('source_sha256',{}).get(k)==b['status'].get('source_sha256',{}).get(k) for k in COMMON_SOURCES})
    cell_key=lambda row:tuple(row.get(k) for k in ('request_id','phase','repeat'))
    left={cell_key(r):r for r in a['rows']};right={cell_key(r):r for r in b['rows']}
    config['same_cell_keys']=left.keys()==right.keys()
    pairs=[]
    for key in sorted(left.keys()|right.keys()):
        x,y=left.get(key),right.get(key)
        if x is None or y is None:
            pairs.append(dict(request_id=key[0],phase=key[1],repeat=key[2],missing='control' if x is None else 'candidate',checks={'both_present':False}));continue
        checks=compare(x,y);ratio=y['num_tokens']/x['num_tokens'] if integer(x.get('num_tokens')) and x['num_tokens'] else None
        pairs.append(dict(request_id=key[0],phase=key[1],repeat=key[2],checks=checks,
            control_tokens=x.get('num_tokens'),candidate_tokens=y.get('num_tokens'),length_ratio=ratio,
            within_110_percent=ratio is not None and ratio<=1.10,
            common_counts_control=common_counts(x),common_counts_candidate=common_counts(y),
            control_text=x.get('text'),candidate_text=y.get('text')))
    exact=all(config.values()) and all(r['review']['mechanical_execution_pass'] and r['review']['all_within_run_repeats_exact'] for r in (a,b)) and all(all(p['checks'].values()) for p in pairs)
    uuids=lambda run:{r.get('hardware',{}).get('device',{}).get('uuid') for r in run['rows']}
    au,bu=uuids(a),uuids(b)
    same_device=len(au)==len(bu)==1 and None not in au and au==bu and a['status']['args'].get('physical_gpu')==b['status']['args'].get('physical_gpu')
    ta,tb=a['review']['terminal'],b['review']['terminal']
    sequential=(a['status'].get('pid')!=b['status'].get('pid') and all(type(t.get(k)) in (int,float) and math.isfinite(t[k]) for t,k in
        ((ta,'started'),(ta,'ended'),(tb,'started'),(tb,'ended')))) and ta['started']<=ta['ended']<=tb['started']<=tb['ended']
    background_clear=all(r['review']['hardware_summary']['sampled_background_clear'] for r in (a,b))
    observed={name:r['review']['formal_aggregate'] for name,r in (('grouped',a),(METHOD,b))}
    change=None
    if all('error' not in v for v in observed.values()):
        change={phase:dict(throughput_change_pct=100*(observed[METHOD][phase]['tokens_per_second']/observed['grouped'][phase]['tokens_per_second']-1),
            latency_change_pct=100*(observed[METHOD][phase]['total_seconds']/observed['grouped'][phase]['total_seconds']-1)) for phase in ('e2e','generation')}
    eligible=bool(exact and same_device and sequential and background_clear and change is not None)
    report['mechanical_exact_comparison_pass']=bool(exact)
    report['comparison']=dict(configuration_checks=config,cells=pairs,same_physical_device=same_device,
        sequential_distinct_processes=bool(sequential),sampled_background_clear=background_clear,
        raw_formal_aggregates=observed,raw_observed_change=change,
        exploratory_timing_comparison_eligible=eligible,conditioned_exploratory_change=change if eligible else None,
        throughput_claim=False)
    report['helper_sha256']={name:digest(Path(__file__).with_name(name)) for name in
        ('assess_visual_kv.py','summarize_dram_campaign.py','audit_grouped_generation.py','audit_native_delta_abba.py','audit_stride0_pilot.py')}
    return report


def markdown(report):
    lines=['# No-weights 完整请求审计','',f"输出与完整轨迹机械验收：**{report['mechanical_exact_comparison_pass']}**。",'']
    for r in report['runs']:
        lines.extend([f"## {r['mode']}",'',f"机械执行：{r['mechanical_execution_pass']}；目录：`{r['directory']}`。"])
        if 'input_error' in r:
            lines.extend([r['input_error'],'']);continue
        lines.extend([f"正式原始汇总：`{r['formal_aggregate']}`。",
            f"背景：`{r['hardware_summary']}`。原始 valid_timing 未更改。",''])
    c=report['comparison']
    if c:
        lines.extend(['## 两臂比较','',f"同卡：{c['same_physical_device']}；顺序独立进程：{c['sequential_distinct_processes']}。",
            f"原始变化（不代表合格性能证据）：`{c['raw_observed_change']}`。",
            f"条件化探索性变化：`{c['conditioned_exploratory_change']}`；资格：{c['exploratory_timing_comparison_eligible']}。",''])
        for p in c['cells']:
            if p['phase']=='measure':
                lines.append(f"- {p['request_id']} / repeat {p['repeat']}：全部比较 {all(p['checks'].values())}；≤110%长度 {p.get('within_110_percent')}。")
    lines.extend(['','## 口径','']+['- '+s for s in report['limitations']])
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--known-graphics-pid',type=int,action='append',default=[])
    parser.add_argument('--source-root',type=Path,default=BASE)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():parser.error('fresh output directory required')
    report=audit(args.control,args.candidate,args.known_graphics_pid,args.source_root)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (args.out/'audit.md').write_text(markdown(report))
    print(json.dumps({'mechanical_exact_comparison_pass':report['mechanical_exact_comparison_pass'],
        'tool_sha256':report['tool_sha256'],'out':str(args.out)}))
    return 0 if report['mechanical_exact_comparison_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())
