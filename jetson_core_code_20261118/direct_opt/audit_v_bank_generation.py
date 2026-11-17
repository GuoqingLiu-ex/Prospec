"""Offline fixed V16/V8 model receipts; standard library only, no GPU actions.

One warmup and two repeats of one or twelve requests. Shared-GPU runs establish
bounded numerical/storage observations, never clean performance/energy claims.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from assess_visual_kv import generation
from audit_grouped_generation import coverage as grouped_coverage
from audit_no_weights_generation import (background, integer, finite_number,
    safe_aggregate, COMMON_SOURCES as PARENT_COMMON_SOURCES, TIMES)
from audit_stride0_pilot import FLAGS, digest

BASE = Path(__file__).resolve().parent.parent
METHOD = 'full_k_partitioned_visual_v_grouped_math'
MODES = ('grouped_control', 'fp16_partition', 'int8_visual')
METHOD_COUNTS = {'target_grouped_attention_calls', 'target_v_bank_attention_calls'}
COMMON_SOURCES = PARENT_COMMON_SOURCES+('direct_opt/ring_kv_cache.py',)
COMMON_COUNT_FIELDS = {
    'target_prefill_calls','target_prefill_query_tokens','target_verify_calls','target_verify_query_tokens',
    'target_decode_calls','target_decode_query_tokens','target_visual_calls','draft_tree_calls',
    'draft_model_forward_calls','draft_model_input_hidden_tokens','draft_decoder_query_tokens',
    'target_compaction_calls','target_compaction_selected_tokens','imadpt_calls','imadpt_input_tokens',
    'imadpt_overridden_calls','budget_switch_calls','merge_visual_tokens_in','merge_visual_tokens_out',
    'target_tree_attention_policy_calls'}
NEW_SOURCES = tuple('direct_opt/'+n for n in ('run_v_bank_layout.py',
    'v_bank_model_runtime.py', 'grouped_v_bank_math.py', 'partitioned_v_bank.py', 'v_bank_cuda_ops.py'))
RECEIPT_SOURCES = {'run_v_bank_layout.py','v_bank_model_runtime.py','grouped_v_bank_math.py',
    'partitioned_v_bank.py','v_bank_cuda_ops.py','stride0_gqa_runtime.py','run_grouped_gqa_layout.py',
    'run_memory_layout.py','run_deployed_adaptor.py','run_joint.py'}


def arm_name(mode):
    return ('memory_shared_embedding_compiled_fullkv_tree_grouped_math' if mode=='grouped_control'
        else 'memory_shared_embedding_compiled_fullk_v_'+mode+'_grouped_math')


def checked(fn, *args):
    try:
        return fn(*args)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        return dict(passed=False, error=f'{type(exc).__name__}: {exc}')


def common_counts(row):
    counts=row.get('result',{}).get('forward_counts')
    return {k:v for k,v in counts.items() if k not in METHOD_COUNTS} if isinstance(counts,dict) else None


def count_contract(row,mode):
    counts=row['result']['forward_counts'];visual=row['result']['visual_context']
    trees=[r for r in row['result']['rounds'] if r['kind']=='tree'];n=len(trees)
    method='target_grouped_attention_calls' if mode=='grouped_control' else 'target_v_bank_attention_calls'
    checks=dict(exact_fields=set(counts)==COMMON_COUNT_FIELDS|{method},
        nonnegative_integers=all(integer(x) for x in counts.values()),
        no_separate_target_decode=counts['target_decode_calls']==counts['target_decode_query_tokens']==0,
        visual_once=counts['target_visual_calls']==1,
        draft_calls=counts['draft_tree_calls']==counts['budget_switch_calls']==n and
            counts['draft_model_forward_calls']==sum(r['budget']['d']+1 for r in trees),
        unchanged_tokens=counts['merge_visual_tokens_in']==counts['merge_visual_tokens_out']==visual['original_count']==visual['kept_count'],
        overridden_adaptor_counter=counts['imadpt_calls']==counts['imadpt_input_tokens']==counts['imadpt_overridden_calls']==0,
        input_decoder_relation=counts['draft_model_input_hidden_tokens']-counts['draft_decoder_query_tokens']==visual['original_count']-1)
    return dict(passed=all(checks.values()),checks=checks)


def compare(a,b):
    fields=('output_token_ids','text','stop_reason','truncated','num_tokens',
            'input_shapes','video_grid_thw','timestamps')
    checks={k:k in a and k in b and a[k]==b[k] for k in fields}
    for k in ('input_fingerprint','input'):
        if k in a or k in b:checks[k]=k in a and k in b and a[k]==b[k]
    x,y=a.get('result',{}),b.get('result',{})
    for k in ('rounds','prompt_tokens','target_cache_length','budget_trace','visual_context'):
        checks['result.'+k]=k in x and k in y and x[k]==y[k]
    checks['common_forward_counts']=common_counts(a) is not None and common_counts(a)==common_counts(b)
    return checks


def expected_precision(mode):
    return dict(target='float16',output_head='float32',draft='float16',draft_sdpa='auto',
        target_sdpa='math_prefill+grouped_v_bank_fp32_math_tree',target_prefill_attention_backend='math',
        target_verify_attention_backend='grouped_two_bmm_v_bank_fp32_math',target_tree_attention_mode='grouped_v_bank',
        target_tree_math_precision=FLAGS,target_cache_policy=METHOD,
        target_key_storage='full_fp16_native_cat_index_select',
        target_visual_value_storage='int8_group32_fp32_scale' if mode=='int8_visual' else 'fp16_partition',
        target_nonvisual_value_storage='compact_fp16_native_cat_index_select',
        target_value_operand='one_direct_fill_fp32_four_kv_heads',v_storage=mode,
        draft_attention_backend='unchanged',tree_attention_discarded_probability_cast_removed=False,
        tree_attention_bitwise_equivalence_claim=False)


def storage_snapshot(s,mode,P,M,S):
    shapes={'keys':([1,4,S,128],'torch.float16',2),
        'text_values':([1,4,S-M,128],'torch.float16',2),
        'visual_values':([1,4,M,128],'torch.int8' if mode=='int8_visual' else 'torch.float16',1 if mode=='int8_visual' else 2),
        'error_flag':([],'torch.int32',4)}
    if mode=='int8_visual':shapes['scales']=([1,4,M,4],'torch.float32',4)
    checks={'layer_count':s['layer_count']==28 and len(s['layers'])==28}
    logical=backing=0
    for i,layer in enumerate(s['layers']):
        checks[f'layer_{i}_fields']=set(layer)==set(shapes)
        for name,(shape,dtype,size) in shapes.items():
            item=layer[name];payload=math.prod(shape)*size
            checks[f'layer_{i}_{name}']=(item['shape']==shape and item['dtype']==dtype and
                item['logical_bytes']==payload and integer(item['backing_bytes']) and item['backing_bytes']>=payload)
            logical+=payload;backing+=item['backing_bytes'] if payload else 0
    checks.update(independent_known_backings=s['unique_layer_backing_bytes']==backing,
        shared_route=s['shared_device_route_bytes']==s['host_partition_route_bytes']==4*P and s['route_counted_once'] is True,
        total_backing=s['owned_plus_shared_device_bytes']==backing+4*P,
        source_release=s['original_value_tensor_refs_alive']==0 and s['original_full_V_known_backing_retained'] is False,
        allocator_separate=s['allocator_reserved_is_separate'] is True)
    return dict(passed=all(checks.values()),checks=checks,known_owned_plus_shared_bytes=s['owned_plus_shared_device_bytes'],
        known_layer_backing_bytes=s['unique_layer_backing_bytes'],logical_layer_payload_bytes=logical,
        shared_device_route_bytes=s['shared_device_route_bytes'],
        full_target_kv_same_length_payload_reference_bytes=28*2*4*S*128*2,
        scope='Observed known bank roots including flags/scales and route once. Full Target KV reference is theoretical payload, not observed control backing or allocator peak.')


def memory_evidence(row,info):
    phases=info['memory_phase_peaks'];checks={};values={}
    for kind in ('allocated','reserved'):
        name='peak_'+kind+'_bytes';p=[phases[x][name] for x in ('prefill','conversion','decode')]
        valid=all(integer(x) and x>0 for x in p)
        checks[kind+'_phase_max']=valid and phases['full_request_'+name]==max(p)
        checks[kind+'_whole_includes_phases']=valid and integer(row[name]) and row[name]>=max(p)
        for label in ('prefill','conversion','decode'):
            at=phases[label][kind+'_at_boundary_bytes']
            checks[label+'_'+kind+'_boundary']=integer(at) and at<=phases[label][name]
        values[kind]=dict(phases=p,phase_max=phases['full_request_'+name],whole_request_peak=row[name])
    checks['reserved_not_below_allocated']=all(phases[p]['peak_reserved_bytes']>=phases[p]['peak_allocated_bytes'] for p in ('prefill','conversion','decode')) and row['peak_reserved_bytes']>=row['peak_allocated_bytes']
    return dict(passed=all(checks.values()),checks=checks,values=values,
        cleanup_scope='Parent source merges cleanup CUDA peak with policy phase maxima; cleanup-only peak is not independently recorded. Rows can verify inclusion, not reconstruct the unrecorded cleanup peak.')


def logical_io_from_rounds(P,M,mode,trees):
    """Exactly the frozen policy's ideal payload ledger; not memory transactions."""
    out={}
    def add(name,calls,read,write):
        v=out.setdefault(name,dict(calls=0,read_bytes=0,write_bytes=0,read_plus_write_bytes=0))
        v['calls']+=calls;v['read_bytes']+=read;v['write_bytes']+=write;v['read_plus_write_bytes']+=read+write
    visual=4*M*128*(1 if mode=='int8_visual' else 2)+(4*M*4*4 if mode=='int8_visual' else 0)
    add('route_upload',1,4*P,4*P)
    add('prefill_pack',28,28*4*P*128*2,28*(visual+4*(P-M)*128*2))
    for t in trees:
        T=t['tree_nodes'];S=t['target_cache_before']+T;C=t['target_cache_after']
        add('round_guard_host_payload',1,4*T*T+8*3*T,4*T*T+8*3*T)
        for name,length in (('key_cat',S),('compact_value_cat',S-M),('key_compact',C),('compact_value_compact',C-M)):
            size=28*4*length*128*2;add(name,28,size,size)
        add('value_operand_fill',28,28*(visual+4*(S-M)*128*2),28*4*S*128*4)
        add('shared_commit_index_upload',1,8*(C+C-M),8*(C+C-M))
    return out


def candidate_coverage(row,mode):
    info=row['adaptor_preparation']['v_bank'];result=row['result'];counts=result['forward_counts']
    trees=[r for r in result['rounds'] if r['kind']=='tree'];n=len(trees);P=result['prompt_tokens']
    part=info['partition'];M=part['visual_tokens'];R=part['nonvisual_prompt_tokens'];route=info['route']
    checks=dict(identity=info['variant']==info['target_cache_policy']==METHOD and info['mode']==mode==row['adaptor_preparation']['v_storage'],
        finalized=info['finalized'] is True and info['result_coverage_validated'] is True and info['failures']==[] and info['fallback_calls']==0,
        actual_cuda=info['reference_only'] is False,
        precision=info['precision']==FLAGS and result['precision']==expected_precision(mode),
        prefill_once=info['prefill_calls']==info['prefill_forward_calls']==1 and info['prefill_layer_calls']==[1]*28,
        all_verify=info['verify_forward_calls']==counts['target_verify_calls']==n and info['verify_query_tokens']==counts['target_verify_query_tokens'],
        all_layers=info['layer_calls']==[n]*28 and info['selected_calls']==info['completed_calls']==info['candidate_calls']==28*n==counts['target_tree_attention_policy_calls']==counts['target_v_bank_attention_calls'],
        all_mutations=info['update_calls']==info['fill_calls']==28*n and info['commit_calls']==n and info['native_tree_calls']==0,
        no_other_candidate='target_grouped_attention_calls' not in counts and counts.get('target_stride0_attention_calls',0)==0,
        status_once=info['setup_status_calls']==info['result_status_calls']==1,
        original_k_mask_positions=info['original_length_unchanged'] is True and info['full_k_native_cat_index_select'] is True and info['original_mask_reused'] is True and info['position_ids_passed_unchanged'] is True and info['rotary_reapplied'] is False,
        installed=info['new_kv_policy_installed'] is True and info['cache_functions_changed'] is True,
        no_full_v=info['full_fp16_value_reconstruction'] is False and info['native_unused_probability_cast_preserved'] is True,
        source_released=info['original_value_tensor_refs_alive']==0,
        no_extra_claim=info['actual_dram_measured'] is False and info['debug_intermediates'] is False,
        representative_once=info['primitive_receipt_calls']==int(n>0),
        native_layer_types=info['native_cache_type']=='transformers.cache_utils.DynamicCache' and info['native_cache_layer_types']==['transformers.cache_utils.DynamicLayer']*28,
        partition_counts=integer(M) and M>0 and integer(R) and P==part['prefix_tokens']==M+R and part['shared_route_bytes']==4*P and part['count_once_per_request'] is True and part['route_dtype']=='int32',
        route_once=route['P']==P and route['M']==M and route['R']==R and route['device_route_backing_bytes']==4*P and route['route_uploaded_once_per_request'] is True and route['host_partition']==part and str(route['device']).startswith('cuda'),
        visual_provenance=part['provenance']['visual_counts']==info['visual_counts'] and sum(info['visual_counts'].values())==M and
            isinstance(part['provenance']['input_ids_sha256'],str) and len(part['provenance']['input_ids_sha256'])==64 and
            part['provenance']['grid_thw']['video']==row['video_grid_thw'])
    prov=part['provenance'];merge=prov['spatial_merge_size']
    checks['visual_grid_counts']=integer(merge) and merge>0 and all(
        sum(math.prod(g) for g in (prov['grid_thw'][kind] or []))==info['visual_counts'][kind]*merge*merge and
        integer(prov['visual_token_ids'][kind]) for kind in ('video','image'))
    for field,stage in (('setup_status','after_pack_before_fill'),('result_status','model_result_boundary')):
        s=info[field];checks[field]=s['stage']==stage and s['banks']==28 and s['status_valid'] is True and s['explicit_host_read'] is True
    rounds=info['rounds'];checks['rounds']=len(rounds)==n and all(
        r['cache_before']==t['target_cache_before'] and r['tree_nodes']==t['tree_nodes'] and
        r['cache_after']==r['cache_before']+r['tree_nodes'] and r['layer_calls']==r['update_counts']==[1]*28 and
        r['compact_value_tokens']==r['cache_after']-M and r['selected_tree_indices']==t['selected_tree_indices'] and
        r['cache_after_commit']==t['target_cache_after'] and r['compact_value_after_commit']==r['cache_after_commit']-M
        for r,t in zip(rounds,trees))
    rep=info['representative_receipt'];checks['representative_exists']=bool(rep)==bool(n)
    if n:
        checks['primitive_contract']=all(rep[k]==v for k,v in dict(implementation='grouped_two_bmm_partitioned_v_fp32_math',
            mode=mode,original_mask_object=True,fp32_operand_count=1,full_fp16_value_reconstruction=False,
            native_unused_probability_cast_preserved=True,debug_intermediates=False,qk_bmm_calls=1,pv_bmm_calls=1).items())
        checks['primitive_shape']=rep['shape']==dict(query_tokens=trees[0]['tree_nodes'],
            logical_key_tokens=trees[0]['target_cache_before']+trees[0]['tree_nodes'],
            compact_value_tokens=trees[0]['target_cache_before']+trees[0]['tree_nodes']-M)
    storage={name:checked(storage_snapshot,info[name],mode,P,M,S) for name,S in
        (('storage_after_conversion',P),('storage_at_result',result['target_cache_length']),('final_storage',result['target_cache_length']))}
    checks['storage']=all(x['passed'] for x in storage.values())
    checks['final_storage_stable']=info['storage_at_result']==info['final_storage']
    io=info['logical_io'];checks['logical_io_arithmetic']=bool(io) and all(
        all(integer(v[k]) for k in ('calls','read_bytes','write_bytes','read_plus_write_bytes')) and
        v['read_plus_write_bytes']==v['read_bytes']+v['write_bytes'] for v in io.values())
    checks['logical_io_actual_trajectory']=io==logical_io_from_rounds(P,M,mode,trees)
    peaks=checked(memory_evidence,row,info);checks['memory_phases']=peaks['passed']
    return dict(passed=all(checks.values()),checks=checks,storage=storage,memory=peaks,
        partition=part,route=route,logical_io=io,logical_io_scope=info['logical_io_scope'],
        status_scope=info['status_scope'],verify_calls=n,attention_layer_calls=info['completed_calls'])


def terminal_checks(terminal,status,mode):
    cmd=terminal.get('command');options={}
    if isinstance(cmd,list):
        for option in ('--v-storage','--out','--physical-gpu'):
            at=[i for i,v in enumerate(cmd) if v==option]
            options[option]=cmd[at[0]+1] if len(at)==1 and at[0]+1<len(cmd) else None
    return dict(success=terminal.get('status')=='completed' and terminal.get('returncode')==0 and status.get('status')=='completed',
        child=integer(status.get('pid')) and status['pid']>0 and terminal.get('child_pid')==status['pid'],
        wrapper=integer(terminal.get('wrapper_pid')) and terminal['wrapper_pid']>0 and terminal['wrapper_pid']!=status.get('pid'),
        command=isinstance(cmd,list) and len(cmd)>1 and Path(cmd[1]).name=='run_v_bank_layout.py' and options.get('--v-storage')==mode and options.get('--out')==status['args'].get('out') and options.get('--physical-gpu')==str(status['args'].get('physical_gpu')),
        interval=finite_number(terminal.get('started')) and finite_number(terminal.get('ended')) and terminal['ended']>terminal['started'])


def read_run(path,mode,source_root=BASE,known_graphics=()):
    path=Path(path);status=json.loads((path/'status.json').read_text());args=status['args']
    rows=[json.loads(x) for x in (path/'results.jsonl').read_text().splitlines() if x.strip()]
    requests=json.loads((path/'requests.json').read_text());schedule=json.loads((path/'schedule.json').read_text())
    terminal=json.loads(Path(str(path)+'.exit.json').read_text())
    key=lambda r:tuple(r.get(k) for k in ('request_id','arm','phase','repeat'))
    protocol=terminal_checks(terminal,status,mode)
    protocol.update(request_count=len(requests) in (1,12) and len({r['request_id'] for r in requests})==len(requests),
        fixed_protocol=args['warmup']==1 and args['repeats']==2 and args['max_new_tokens']==512 and args['resume'] is False,
        cells=len(rows)==status['completed_cells']==status['total_cells']==1+2*len(requests),
        schedule=len({key(r) for r in rows})==len(rows) and list(map(key,rows))==list(map(key,schedule)),
        arms={r.get('arm') for r in rows}=={arm_name(mode)},
        warmup=bool(rows) and rows[0]['phase']=='warmup' and rows[0]['repeat']==0 and rows[0]['request_id']==requests[0]['request_id'],
        formal=all(r['phase']=='measure' for r in rows[1:]) and {(r['request_id'],r['repeat']) for r in rows[1:]}=={(r['request_id'],i) for r in requests for i in (0,1)},
        deployment=status['memory_layout']=='shared_embedding' and status['deployment_adaptor']=='compiled' and status['tree_attention_precision']==FLAGS,
        no_fallback_eviction=status['attention_backend_fallback'] is False and status['target_history_eviction'] is False,
        lock=status['lock']['pid']==status['pid'] and status['lock']['gpu']==args['physical_gpu'])
    candidate=mode!='grouped_control'
    if candidate:
        protocol['candidate_identity']=(status['v_storage']==mode and status['target_cache_policy']==METHOD and status['tree_attention_mode']=='grouped_v_bank' and status['new_kv_policy_installed'] is True and status['kv_quantized']==(mode=='int8_visual') and status['native_unused_probability_cast_preserved'] is True and status['actual_dram_measured'] is False)
        last=status['v_bank_last_request'];protocol['last_request']=last['summary_available'] is True and last['mode']==mode and last['finalized'] is True and last['result_coverage_validated'] is True and last['failures']==[] and bool(rows) and last['memory_phase_peaks']==rows[-1]['adaptor_preparation']['v_bank']['memory_phase_peaks']
    else:
        protocol['control_identity']=status['tree_attention_mode']=='grouped' and status['target_cache_policy']=='native_full_dynamic_cache' and status['new_kv_policy_installed'] is False and status['kv_quantized'] is False
    sources=tuple(dict.fromkeys(COMMON_SOURCES+('direct_opt/run_v_bank_layout.py',)+(NEW_SOURCES if candidate else ())))
    source_checks={}
    for name in sources:
        expected=status['source_sha256'].get(name)
        try:actual=digest(Path(source_root)/name)
        except OSError:actual=None
        source_checks[name]=dict(expected=expected,actual=actual,passed=bool(actual) and expected==actual)
    if candidate:
        protocol['dependency_receipt']=set(status['v_bank_source_sha256'])==RECEIPT_SOURCES and all(status['source_sha256'].get('direct_opt/'+k)==v for k,v in status['v_bank_source_sha256'].items())
    details=[]
    for row in rows:
        gen=checked(generation,row,status)
        counters=checked(count_contract,row,mode)
        cov=checked(candidate_coverage,row,mode) if candidate else checked(grouped_coverage,row,'grouped')
        scheme=dict(depth='fixed0',adaptor='compiled',compression=None,ratio=1.0,
            tree_attention='grouped_v_bank' if candidate else 'grouped',target_cache_policy=METHOD if candidate else 'native_full_dynamic_cache')
        if candidate:scheme['v_storage']=mode
        hw=checked(background,row,status['pid'],known_graphics)
        timing={k:row.get(k) for k in TIMES}
        timing_checks={'values':all(finite_number(x) and x>=0 for x in timing.values())}
        if timing_checks['values']:
            timing_checks.update(model=math.isclose(row['model_wall_s'],row['generation_wall_s']+row['output_decode_wall_s']+row['runtime_finalize_wall_s'],abs_tol=1e-6),
                whole=math.isclose(row['e2e_wall_s'],row['model_wall_s']+row['prepare_wall_s']+row['policy_prepare_wall_s'],abs_tol=1e-3))
        peaks={k:row.get(k) for k in ('peak_allocated_bytes','peak_reserved_bytes')}
        good_peaks=all(integer(x) and x>0 for x in peaks.values()) and peaks['peak_reserved_bytes']>=peaks['peak_allocated_bytes']
        recipe=(row.get('scheme')==scheme and row.get('capture_enabled') is False and row.get('under_ncu',False) is False)
        identity=(row.get('physical_gpu')==args['physical_gpu']==row.get('hardware',{}).get('physical_device_index') and
            bool(row.get('hardware',{}).get('device',{}).get('uuid')) and row['hardware']['device']['uuid']==status['lock']['initial_snapshot']['gpu_uuid'])
        details.append(dict(request_id=row['request_id'],phase=row['phase'],repeat=row['repeat'],generation=gen,coverage=cov,
            forward_count_contract=counters,
            recipe_pass=recipe,device_identity_pass=identity,timing=timing,timing_checks=timing_checks,
            peaks=peaks,peaks_valid=good_peaks,background=hw,raw_valid_timing=row.get('valid_timing'),
            raw_timing_reasons=row.get('timing_reasons'),raw_energy={k:v for k,v in row.get('hardware',{}).items() if 'energy' in k},
            passed=bool(gen.get('receipt_valid') and gen.get('natural_eos') and cov.get('passed') and counters['passed'] and recipe and identity and good_peaks and all(timing_checks.values()))))
    within=[]
    for req in requests:
        same=[r for r in rows if r['request_id']==req['request_id']]
        within.extend(dict(request_id=req['request_id'],phase=r['phase'],repeat=r['repeat'],checks=compare(same[0],r)) for r in same[1:])
    review=dict(mode=mode,directory=str(path.resolve()),protocol_checks=protocol,source_checks=source_checks,
        mechanical_execution_pass=all(protocol.values()) and all(s['passed'] for s in source_checks.values()) and bool(details) and all(d['passed'] for d in details),
        within_run_checks=within,within_run_exact=bool(within) and all(all(x['checks'].values()) for x in within),
        details=details,formal_aggregate=safe_aggregate([r for r in rows if r['phase']=='measure']),terminal=terminal,
        raw_valid_timing_counts=dict(Counter(str(r.get('valid_timing')) for r in rows)),
        all_observed_foreign_pids=sorted({p for d in details for p in d['background'].get('all_observed_foreign_pids',[])}),
        raw_hashes={k:digest(path/k) for k in ('status.json','results.jsonl','requests.json','schedule.json')},
        terminal_sha256=digest(str(path)+'.exit.json'))
    return dict(status=status,rows=rows,requests=requests,review=review)


def audit(control,candidate,mode,source_root=BASE,known_graphics=()):
    if mode not in MODES[1:]:raise ValueError('candidate must be fp16_partition or int8_visual')
    runs=[]
    for path,m in ((control,'grouped_control'),(candidate,mode)):
        try:runs.append(read_run(path,m,source_root,known_graphics))
        except (OSError,KeyError,TypeError,ValueError,IndexError) as exc:runs.append(dict(review=dict(mode=m,directory=str(path),mechanical_execution_pass=False,input_error=f'{type(exc).__name__}: {exc}')))
    report=dict(schema='v_bank_generation_audit_v1',tool_sha256=digest(__file__),runs=[r['review'] for r in runs],
        mechanical_execution_pass=False,fp16_exact_gate_pass=False,comparison=None,
        semantic_quality_status='not established by this mechanical audit',
        clean_timing_eligible=False,energy_savings_eligible=False,actual_dram_measured=False,
        common_count_exclusions=sorted(METHOD_COUNTS),
        limitations=['Shared-GPU pilot: timings/energy are raw observations, with no clean performance or energy-saving conclusion.',
            'Two repeats are not independent videos; sample coverage does not establish exclusive GPU execution.',
            'Input manifest/IDs provenance and metadata are checked; decoded pixel bytes are not independently available.',
            'FP16 exact gate is bounded to these recorded requests. V8 text changes require fact review; length alone is not quality.',
            'Known persistent bank backing, theoretical KV payload and whole allocator peaks have different scopes.',
            'Cleanup-only peak is not separately saved; whole peak must include all phase peaks, but exact cleanup max is not reconstructable.',
            'Pack/fill status validates device error flags, not finiteness of all model arithmetic. Logical IO is not measured DRAM.'])
    if any('rows' not in r for r in runs):return report
    a,b=runs
    config={k:a['status'].get(k) is not None and a['status'].get(k)==b['status'].get(k) for k in ('target','draft','torch','transformers','request_sha256')}
    config.update({'args.'+k:a['status']['args'].get(k)==b['status']['args'].get(k) for k in ('seed','max_new_tokens','warmup','repeats')})
    config['manifest']=a['requests']==b['requests']
    config.update({'source.'+k:bool(a['status']['source_sha256'].get(k)) and a['status']['source_sha256'].get(k)==b['status']['source_sha256'].get(k) for k in COMMON_SOURCES+('direct_opt/run_v_bank_layout.py',)})
    key=lambda r:(r['request_id'],r['phase'],r['repeat'])
    left={key(r):r for r in a['rows']};right={key(r):r for r in b['rows']};config['cell_keys']=left.keys()==right.keys()
    pairs=[]
    for k in sorted(left.keys()|right.keys()):
        x,y=left.get(k),right.get(k)
        if x is None or y is None:pairs.append(dict(request_id=k[0],phase=k[1],repeat=k[2],checks={'both_present':False}));continue
        ratio=y['num_tokens']/x['num_tokens'] if integer(x.get('num_tokens')) and x['num_tokens'] else None
        pairs.append(dict(request_id=k[0],phase=k[1],repeat=k[2],checks=compare(x,y),
            control_tokens=x.get('num_tokens'),candidate_tokens=y.get('num_tokens'),length_ratio=ratio,
            within_110_percent=ratio is not None and ratio<=1.10,control_text=x.get('text'),candidate_text=y.get('text'),
            control_common_counts=common_counts(x),candidate_common_counts=common_counts(y),
            control_peak_allocated_bytes=x.get('peak_allocated_bytes'),candidate_peak_allocated_bytes=y.get('peak_allocated_bytes'),
            control_peak_reserved_bytes=x.get('peak_reserved_bytes'),candidate_peak_reserved_bytes=y.get('peak_reserved_bytes')))
    structural=all(config.values()) and all(r['review']['mechanical_execution_pass'] for r in runs)
    exact=structural and all(r['review']['within_run_exact'] for r in runs) and bool(pairs) and all(all(p['checks'].values()) for p in pairs)
    report.update(mechanical_execution_pass=bool(structural),fp16_exact_gate_pass=mode=='fp16_partition' and bool(exact),
        comparison=dict(configuration_checks=config,cells=pairs,all_cells_exact=bool(exact),
            all_cells_within_110_percent=bool(pairs) and all(p.get('within_110_percent',False) for p in pairs),
            same_full_source_map=a['status']['source_sha256']==b['status']['source_sha256'],
            raw_source_counts=[len(r['status']['source_sha256']) for r in runs],
            quality_fact_review_needed=mode=='int8_visual' and not exact))
    return report


def markdown(report):
    lines=['# V-only 模型原始记录审计','',f"机械执行验收：**{report['mechanical_execution_pass']}**；FP16 分区逐位/轨迹门槛：**{report['fp16_exact_gate_pass']}**。",
        '', '共享 GPU：不形成干净时延、节能或 DRAM 结论。']
    for run in report['runs']:
        lines += ['', '## '+run['mode'],'',f"机械记录：{run['mechanical_execution_pass']}；目录：`{run['directory']}`。"]
        if 'input_error' in run:lines.append(run['input_error']);continue
        lines += [f"原始正式汇总：`{run['formal_aggregate']}`。",f"观察到的其他 PID：{run['all_observed_foreign_pids']}；原 valid_timing：{run['raw_valid_timing_counts']}。"]
        for d in run['details']:
            if d['phase']!='measure':continue
            lines.append(f"- {d['request_id']} / r{d['repeat']}：验收 {d['passed']}；whole peaks `{d['peaks']}`。")
            if 'storage' in d['coverage']:
                s=d['coverage']['storage'];lines.append('  已记录常驻 bank bytes：'+', '.join(f"{k}={v.get('known_owned_plus_shared_bytes')}" for k,v in s.items())+'；不等同 allocator peak。')
    if report['comparison']:
        lines += ['', '## 逐请求比较','']
        for p in report['comparison']['cells']:
            if p['phase']=='measure':lines.append(f"- {p['request_id']} / r{p['repeat']}：全部比较 {all(p['checks'].values())}；长度 {p.get('control_tokens')}→{p.get('candidate_tokens')}；≤110% {p.get('within_110_percent')}。")
    lines += ['', '## 限制','']+['- '+x for x in report['limitations']]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--mode',choices=MODES[1:],required=True);p.add_argument('--source-root',type=Path,default=BASE)
    p.add_argument('--known-graphics-pid',type=int,action='append',default=[]);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():p.error('fresh audit output required')
    report=audit(args.control,args.candidate,args.mode,args.source_root,args.known_graphics_pid)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (args.out/'audit.md').write_text(markdown(report))
    ok=report['fp16_exact_gate_pass'] if args.mode=='fp16_partition' else report['mechanical_execution_pass'] and bool(report['comparison']) and report['comparison']['all_cells_within_110_percent']
    print(json.dumps(dict(mechanical_execution_pass=report['mechanical_execution_pass'],fp16_exact_gate_pass=report['fp16_exact_gate_pass'],out=str(args.out))))
    return 0 if ok else 2


if __name__=='__main__':raise SystemExit(main())
