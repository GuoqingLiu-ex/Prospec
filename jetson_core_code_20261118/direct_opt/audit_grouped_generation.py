"""CPU-only native/grouped complete-request audit, including single-arm review.

Supports the frozen max512, one-warmup, two-repeat protocol for one or twelve
requests. Different GPU/process batches may establish numerical reproduction;
they never silently become a throughput comparison. Raw records are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from assess_visual_kv import generation
from audit_native_delta_abba import aggregate, hardware_evidence
from audit_stride0_pilot import FLAGS, check, digest, coverage as native_coverage

EXTRA_COUNTS = {'target_tree_attention_policy_calls', 'target_stride0_attention_calls',
                'target_grouped_attention_calls'}
COMMON_SOURCES = ('direct_opt/run_joint.py', 'direct_opt/run_memory_layout.py',
    'direct_opt/run_deployed_adaptor.py', 'direct_opt/compiled_adaptor.py',
    'direct_opt/shared_embedding.py', 'direct_opt/stride0_gqa_runtime.py',
    'vendor/hf517_vispec_adapter.py', 'vendor/vispec_joint_adapter.py')
REFERENCE = Path(__file__).resolve().parents[2]/'coordination/direct_opt_20261002/memory_shared_compiled_vdc12_r1'


def common_counts(row):
    return {k:v for k,v in row['result']['forward_counts'].items() if k not in EXTRA_COUNTS}


def compare(a,b):
    checks = {k:a[k] == b[k] for k in ('output_token_ids','text','input_shapes','video_grid_thw','timestamps')}
    checks.update({'result.'+k:a['result'][k] == b['result'][k] for k in
        ('rounds','prompt_tokens','target_cache_length','budget_trace','visual_context')})
    checks['common_forward_counts'] = common_counts(a) == common_counts(b)
    return checks


def round_changes(a,b):
    changes=[]
    for i in range(max(len(a),len(b))):
        left=a[i] if i<len(a) else None;right=b[i] if i<len(b) else None
        if left==right:continue
        fields={k:dict(native=left.get(k),grouped=right.get(k)) for k in sorted(left.keys()|right.keys())
                if left.get(k)!=right.get(k)} if left is not None and right is not None else dict(native=left,grouped=right)
        changes.append(dict(round_index=i,fields=fields))
    return dict(changed_round_count=len(changes),changes=changes)


def coverage(row,mode):
    if mode == 'native': return native_coverage(row,'native')
    check(mode == 'grouped','unknown attention mode')
    info=row['adaptor_preparation']['tree_attention'];counts=row['result']['forward_counts']
    trees=[r for r in row['result']['rounds'] if r['kind']=='tree'];rounds=info['rounds'];n=len(trees)
    # This deployed protocol submits thirty nodes per verification. Native
    # token-major Q then needs a full small-Q pack on every selected layer.
    e=sum(28*4*r['cache_after']*128 for r in rounds)
    q=sum(28*4*28*r['tree_nodes']*128 for r in rounds)
    removed=dict(read_bytes=76*e,write_bytes=100*e,read_plus_write_bytes=176*e)
    added=dict(read_bytes=q,write_bytes=q,read_plus_write_bytes=2*q)
    net={key:removed[key]-added[key] for key in removed}
    copied=info['logical_copy_totals']
    expected_precision=dict(target='float16',output_head='float32',draft='float16',draft_sdpa='auto',
        target_sdpa='math_prefill+grouped_fp32_math_tree',target_prefill_attention_backend='math',
        target_verify_attention_backend='grouped_two_bmm_fp32_math',target_tree_attention_mode='grouped',
        target_tree_math_precision=FLAGS,target_cache_policy='native_full_dynamic_cache',
        draft_attention_backend='unchanged',tree_attention_bitwise_equivalence_claim=False)
    rep=info['representative_receipt']
    checks=dict(mode=info['mode']=='grouped'==row['adaptor_preparation']['tree_attention_mode'] and info['implementation']=='grouped_two_bmm_fp32_math',
        finalized_no_failure=info['finalized'] is True and info['failures']==[] and info['fallback_calls']==0,
        coverage_validated=info['result_coverage_validated'] is True,
        native_prefill_once=info['prefill_calls']==info['prefill_forward_calls']==1 and info['prefill_layer_calls']==[1]*28,
        all_verifications=info['verify_forward_calls']==counts['target_verify_calls']==n and
            info['verify_query_tokens']==counts['target_verify_query_tokens'],
        all_layers=info['layer_calls']==[n]*28 and info['selected_calls']==info['completed_calls']==28*n==counts['target_tree_attention_policy_calls'],
        mode_calls=info['candidate_calls']==counts['target_grouped_attention_calls']==28*n and info['native_tree_calls']==0 and counts.get('target_stride0_attention_calls',0)==0,
        every_round_shape_and_precommit_boundary=len(rounds)==n and all(
            r['cache_before']==t['target_cache_before'] and r['tree_nodes']==t['tree_nodes']==30 and
            r['cache_after']==r['cache_before']+r['tree_nodes'] and r['layer_calls']==[1]*28 and
            r['mask_shape']==[1,1,r['tree_nodes'],r['cache_after']] and r['position_shape']==[3,1,r['tree_nodes']] and
            r['mask_dtype']=='torch.float32' and r['position_dtype']=='torch.int64' for r,t in zip(rounds,trees)),
        full_native_cache=info['target_cache_policy']=='native_full_dynamic_cache' and info['new_kv_policy_installed'] is False and
            info['cache_functions_changed'] is False and info['native_cache_type']=='transformers.cache_utils.DynamicCache' and
            info['native_cache_layer_types']==['transformers.cache_utils.DynamicLayer']*28,
        mask_and_rotary=info['original_mask_reused'] is True and info['position_ids_passed_unchanged'] is True and info['rotary_reapplied'] is False,
        expected_precision=row['result']['precision']==expected_precision and info['precision']==FLAGS,
        no_debug_or_dram_claim=info['debug_intermediates'] is False and info['actual_dram_measured'] is False,
        representative_receipt_once=info['primitive_receipt_calls']==int(n>0) and
            info['ordinary_primitive_record_receipt'] is False and info['maximum_representative_receipts_per_request']==1 and
            info['first_tree_call_receipt_in_request'] is True,
        representative_receipt=bool(rep) == bool(n),
        logical_copy_formula=copied['kv_only_removed_explicit_bytes']==removed and copied['added_query_pack_bytes']==added and
            copied['net_removed_explicit_bytes']==net and copied['query_pack_calls']==28*n,
        copy_scope=copied['actual_dram_measured'] is False and copied['blas_internal_copy_measured'] is False and
            copied['attention_mac_reduction']==copied['kv_resident_capacity_reduction']==0)
    if rep:
        checks['representative_layout']=all(rep[k]==v for k,v in dict(implementation='grouped_two_bmm_fp32_math',
            qk_bmm_calls=1,pv_bmm_calls=1,explicit_repeat_kv=False,unique_kv_heads=4,query_pack_copy=True,
            query_pack_logical_read_bytes=430080,query_pack_logical_write_bytes=430080,
            debug_intermediates=False,native_unused_probability_cast_preserved=True).items()) and \
            rep['input']['query']['stride']==[107520,128,3584,1]
    return dict(passed=all(checks.values()),checks=checks,verify_calls=n,attention_layer_calls=info['completed_calls'],
        logical_copy_totals=copied,recomputed_kv_only_removed=removed,recomputed_added_query_pack=added,
        recomputed_net_removed=net,logical_copy_scope=info['logical_copy_totals_scope'])


def background(row,own_pid,known_graphics):
    h=hardware_evidence(row,own_pid,known_graphics);raw=row['hardware']
    observed=sorted({p['pid'] for s in raw.get('samples',[]) for p in s.get('foreign_processes',[])})
    all_pids=sorted(set(h['foreign_pids'])|set(observed));unknown=sorted(set(all_pids)-set(known_graphics))
    h.update(raw_sample_foreign_pids=observed,all_observed_foreign_pids=all_pids,
        foreign_except_known_graphics=unknown,energy_attribution=raw.get('energy_attribution'),
        sample_process_failures=sum(s.get('process_query_complete') is False for s in raw.get('samples',[])),
        sampled_background_clear=bool(h['available'] and h['own_pid_allowlisted'] and
            h['process_visibility_complete'] and h['process_query_count']>0 and h['process_query_failures']==0 and
            not h['errors'] and not unknown and not any(s.get('process_query_complete') is False for s in raw.get('samples',[]))))
    return h


def read_run(path,mode,known_graphics=()):
    path=Path(path);status=json.loads((path/'status.json').read_text())
    rows=[json.loads(x) for x in (path/'results.jsonl').read_text().splitlines() if x.strip()]
    requests=json.loads((path/'requests.json').read_text());schedule=json.loads((path/'schedule.json').read_text())
    terminal=json.loads(Path(str(path)+'.exit.json').read_text());args=status['args']
    check(status['status']=='completed' and terminal['returncode']==0,'run not successfully terminal')
    check(len(requests) in (1,12) and len({r['request_id'] for r in requests})==len(requests),'one or twelve unique requests required')
    check(args['warmup']==1 and args['repeats']==2 and args['max_new_tokens']==512,'frozen 1+2-repeat/max512 protocol')
    check(len(rows)==status['completed_cells']==status['total_cells']==1+2*len(requests),'missing/extra cells')
    key=lambda r:tuple(r[k] for k in ('request_id','arm','phase','repeat'))
    check(len({key(r) for r in rows})==len(rows) and list(map(key,rows))==list(map(key,schedule)),'duplicate/changed schedule')
    check({r['arm'] for r in rows}=={'memory_shared_embedding_compiled_fullkv_tree_'+mode+'_math'},'wrong arm')
    check(rows[0]['phase']=='warmup' and rows[0]['repeat']==0 and rows[0]['request_id']==requests[0]['request_id'],'wrong warmup')
    check(all(r['phase']=='measure' for r in rows[1:]) and
        {(r['request_id'],r['repeat']) for r in rows[1:]}=={(r['request_id'],i) for r in requests for i in range(2)},'formal request/repeat coverage')
    check(status['tree_attention_mode']==mode and status['memory_layout']=='shared_embedding' and
        status['deployment_adaptor']=='compiled' and status['tree_attention_precision']==FLAGS and
        status['target_cache_policy']=='native_full_dynamic_cache' and status['new_kv_policy_installed'] is False,'deployment identity')
    details=[]
    for row in rows:
        gen=generation(row,status);cov=coverage(row,mode)
        components=sum(row[k] for k in ('generation_wall_s','output_decode_wall_s','runtime_finalize_wall_s'))
        times=('e2e_wall_s','generation_wall_s','model_wall_s','prepare_wall_s','policy_prepare_wall_s','output_decode_wall_s','runtime_finalize_wall_s')
        timing=dict(finite_nonnegative=all(math.isfinite(row[k]) and row[k]>=0 for k in times),
            finalization_in_model=math.isclose(row['model_wall_s'],components,abs_tol=1e-6),
            e2e_includes_all_phases=math.isclose(row['e2e_wall_s'],components+row['prepare_wall_s']+row['policy_prepare_wall_s'],abs_tol=1e-3))
        check(row['scheme']['depth']=='fixed0' and row['scheme']['compression'] is None and row['scheme']['ratio']==1.0 and
            row['capture_enabled'] is False and row.get('under_ncu',False) is False,'changed visual/measurement recipe')
        details.append(dict(request_id=row['request_id'],phase=row['phase'],repeat=row['repeat'],num_tokens=row['num_tokens'],
            verify_rounds=row['verify_rounds'],generation=gen,coverage=cov,timing_checks=timing,
            timing={k:row[k] for k in times},peak_allocated_bytes=row['peak_allocated_bytes'],peak_reserved_bytes=row['peak_reserved_bytes'],
            hardware=background(row,status['pid'],known_graphics)))
    selected_sources=COMMON_SOURCES+('direct_opt/run_stride0_gqa_layout.py',) if mode=='native' else COMMON_SOURCES+(
        'direct_opt/run_grouped_gqa_layout.py','direct_opt/grouped_gqa_runtime.py','direct_opt/grouped_gqa_math.py')
    source_checks={k:bool(status['source_sha256'].get(k)) and status['source_sha256'][k]==digest(Path(__file__).parent.parent/k)
                   for k in selected_sources}
    within=[]
    for request in requests:
        same=[r for r in rows if r['request_id']==request['request_id']]
        within.extend(dict(request_id=request['request_id'],phase=r['phase'],repeat=r['repeat'],checks=compare(same[0],r)) for r in same[1:])
    valid=all(source_checks.values()) and all(d['generation']['receipt_valid'] and d['generation']['natural_eos'] and
        d['coverage']['passed'] and all(d['timing_checks'].values()) for d in details)
    hw=[d['hardware'] for d in details]
    review=dict(mode=mode,directory=str(path.resolve()),terminal=terminal,mechanical_execution_pass=valid,
        source_checks=source_checks,source_sha256={k:status['source_sha256'].get(k) for k in selected_sources},
        request_count=len(requests),formal_count=len(rows)-1,within_run_repeat_checks=within,
        all_within_run_repeats_exact=all(all(x['checks'].values()) for x in within),formal_aggregate=aggregate(rows[1:]),
        details=details,hardware_summary=dict(sampled_background_clear=all(h['sampled_background_clear'] for h in hw),
            all_observed_foreign_pids=sorted({p for h in hw for p in h['all_observed_foreign_pids']}),
            foreign_except_known_graphics=sorted({p for h in hw for p in h['foreign_except_known_graphics']}),
            supplied_known_graphics=list(known_graphics),process_query_count=sum(h['process_query_count'] for h in hw),
            process_query_failures=sum(h['process_query_failures'] for h in hw),
            all_visibility_complete=all(h['process_visibility_complete'] for h in hw),
            raw_valid_timing_counts=dict(Counter(str(h['valid_timing']) for h in hw))),
        hashes={k:digest(path/k) for k in ('status.json','results.jsonl','requests.json','schedule.json')},
        terminal_sha256=digest(str(path)+'.exit.json'))
    return dict(status=status,rows=rows,requests=requests,review=review)


def timing_eligibility(a,b,exact):
    uuid=lambda run:{r['hardware']['device']['uuid'] for r in run['rows']}
    same_device=uuid(a)==uuid(b) and a['status']['args']['physical_gpu']==b['status']['args']['physical_gpu']
    sequential=a['status']['pid']!=b['status']['pid'] and a['review']['terminal']['ended']<=b['review']['terminal']['started']
    eligible=exact and same_device and sequential and all(run['review']['hardware_summary']['sampled_background_clear'] for run in (a,b))
    return bool(eligible),same_device,sequential


def historical_lineage(runs,reference):
    path=Path(reference);status=json.loads((path/'status.json').read_text())
    rows=[json.loads(x) for x in (path/'results.jsonl').read_text().splitlines() if x.strip()]
    check(status['status']=='completed' and status['memory_layout']=='shared_embedding' and
          status['deployment_adaptor']=='compiled','wrong/incomplete historical compiled reference')
    mapped={}
    for row in rows:
        if row['phase']=='measure':
            check(row['request_id'] not in mapped,'ambiguous historical reference')
            mapped[row['request_id']]=row
    cells=[]
    for run in runs:
        for row in run['rows']:
            check(row['request_id'] in mapped,'request absent from historical reference')
            checks=compare(mapped[row['request_id']],row)
            checks['same_models']=all(status[k]==run['status'][k] for k in ('target','draft'))
            cells.append(dict(mode=run['review']['mode'],request_id=row['request_id'],phase=row['phase'],repeat=row['repeat'],checks=checks))
    return dict(directory=str(path.resolve()),results_sha256=digest(path/'results.jsonl'),cells=cells,
        exact_output_and_trajectory=all(all(c['checks'].values()) for c in cells),
        scope='Historical numerical reproduction only; historical time/GPU values never enter throughput comparisons.')


def audit(native,candidate=None,known_graphics=(),reference=REFERENCE):
    a=read_run(native,'native',known_graphics);runs=[a];comparison=None
    if candidate is not None:
        b=read_run(candidate,'grouped',known_graphics);runs.append(b)
        config={k:a['status'][k]==b['status'][k] for k in ('target','draft','torch','transformers','request_sha256')}
        config.update({'args.'+k:a['status']['args'][k]==b['status']['args'][k] for k in ('max_new_tokens','seed')})
        config.update({'source.'+k:a['status']['source_sha256'].get(k)==b['status']['source_sha256'].get(k) for k in COMMON_SOURCES})
        request_map={r['request_id']:r for r in a['requests']}
        config['candidate_manifest_matches_native_subset']=all(request_map.get(r['request_id'])==r for r in b['requests'])
        native_by_key={(r['request_id'],r['phase'],r['repeat']):r for r in a['rows']}
        pairs=[];matched_native=[];matched_candidate=[]
        for row in b['rows']:
            key=(row['request_id'],row['phase'],row['repeat'])
            check(key in native_by_key,'candidate cell has no corresponding native cell')
            baseline=native_by_key[key];checks=compare(baseline,row)
            pairs.append(dict(request_id=row['request_id'],phase=row['phase'],repeat=row['repeat'],checks=checks,
                natural_eos=row['stop_reason']=='eos',length_ratio=row['num_tokens']/baseline['num_tokens'],
                within_110_percent=row['num_tokens']<=1.10*baseline['num_tokens'],
                changed_rounds=round_changes(baseline['result']['rounds'],row['result']['rounds']),
                native_tokens=baseline['num_tokens'],candidate_tokens=row['num_tokens']))
            if row['phase']=='measure':matched_native.append(baseline);matched_candidate.append(row)
        exact=all(config.values()) and all(run['review']['mechanical_execution_pass'] and
            run['review']['all_within_run_repeats_exact'] for run in runs) and all(all(p['checks'].values()) for p in pairs)
        timing_eligible,same_device,sequential=timing_eligibility(a,b,exact)
        aggregates=dict(native=aggregate(matched_native),grouped=aggregate(matched_candidate))
        changes={phase:dict(throughput_change_pct=100*(aggregates['grouped'][phase]['tokens_per_second']/
            aggregates['native'][phase]['tokens_per_second']-1),latency_change_pct=100*(aggregates['grouped'][phase]['total_seconds']/
            aggregates['native'][phase]['total_seconds']-1)) for phase in ('e2e','generation')} if timing_eligible else None
        comparison=dict(configuration_checks=config,exact_output_and_full_trajectory_pass=exact,cells=pairs,
            same_physical_device=same_device,sequential_distinct_processes=sequential,
            exploratory_timing_comparison_eligible=timing_eligible,matched_formal_aggregate=aggregates,
            exploratory_change=changes,throughput_claim=False)
    lineage=historical_lineage(runs,reference)
    if comparison is not None:
        comparison['strict_comparison_including_historical_lineage']=comparison['exact_output_and_full_trajectory_pass'] and lineage['exact_output_and_trajectory']
    return dict(schema='grouped_full_generation_audit_v1',tool_sha256=digest(__file__),
        runs=[r['review'] for r in runs],comparison=comparison,historical_compiled_reference=lineage,quality_pass=None,actual_dram_measured=False,
        limitations=['Natural EOS and exact outputs/trajectories on these requests are reproduction evidence, not a general quality guarantee.',
            'Twelve requests with two repeats are twelve distinct videos, not twenty-four independent videos.',
            'Input metadata and manifests are compared; no saved pixel-tensor hash exists to prove decoded input byte identity.',
            'Raw valid_timing flags remain unchanged. No foreign compute observed is a sampled statement; graphics and between-sample activity remain possible.',
            'Different GPU UUIDs disable timing deltas. Same-GPU sequential processes still give exploratory observations, not stable throughput evidence.',
            'Logical copy savings subtract added Q packing but exclude GEMM reads and BLAS internals; they are not NCU DRAM counters.',
            'Whole-request allocated/reserved peaks include prefill; attention-module savings are not whole-model or resident-KV savings.'])


def markdown(report):
    lines=['# Grouped 完整请求独立审计','']
    for run in report['runs']:
        h=run['hardware_summary'];agg=run['formal_aggregate']
        lines += [f"## {run['mode']}",'',
            f"机械执行验收：**{run['mechanical_execution_pass']}**；{run['request_count']} 题、{run['formal_count']} 次正式请求（预热排除）。同请求重复的输出和完整轨迹全部一致：{run['all_within_run_repeats_exact']}。",
            f"正式输出 {agg['output_tokens_including_eos']} tokens，验证 {agg['verify_rounds']} 轮。所有原始结果与逐条层覆盖、精度、EOS、峰值保存在 JSON。",
            f"全过程采样仅含已指定图形进程或自身：**{h['sampled_background_clear']}**；观察到外部 PID {h['all_observed_foreign_pids']}，未列为已知图形进程者 {h['foreign_except_known_graphics']}。查询 {h['process_query_count']} 次，失败 {h['process_query_failures']} 次；完整可见性 {h['all_visibility_complete']}。",
            f"原 valid_timing 标记计数 {h['raw_valid_timing_counts']}，未改写。目录：`{run['directory']}`。",'']
    c=report['comparison']
    lines += ['## 历史 compiled 对照','',
        f"所有本次请求与原 compiled 对照的输出和完整轨迹一致：**{report['historical_compiled_reference']['exact_output_and_trajectory']}**。历史耗时不进入本次比较。",'']
    if c is None:
        lines+=['## 当前结论','', '当前只验收 native。若所有背景检查通过，可保留为同卡探索性对照；尚无候选吞吐改善结论。后来出现的外部进程不会自动证明历史请求受污染，采样也不能证明绝对独占。','']
    else:
        formal=[p for p in c['cells'] if p['phase']=='measure']
        lines += ['## 两臂对照','',
            f"输出和完整轨迹严格一致验收：**{c['exact_output_and_full_trajectory_pass']}**。正式输出 ID 相同 {sum(p['checks']['output_token_ids'] for p in formal)}/{len(formal)}；完整 rounds 相同 {sum(p['checks']['result.rounds'] for p in formal)}/{len(formal)}；≤110% 长度 {sum(p['within_110_percent'] for p in formal)}/{len(formal)}。",
            f"同物理卡：{c['same_physical_device']}；可计算同卡探索性耗时差：{c['exploratory_timing_comparison_eligible']}。",
            f"探索性变化：`{c['exploratory_change']}`。不合格时为 null，不能拿跨卡或污染数据宣称吞吐收益。",'']
        for run in report['runs']:
            formal_rows=[d for d in run['details'] if d['phase']=='measure']
            lines += [f"- {run['mode']} 正式整请求 allocated 峰值：{[d['peak_allocated_bytes'] for d in formal_rows]} B；reserved：{[d['peak_reserved_bytes'] for d in formal_rows]} B。"]
        grouped=report['runs'][1]['details'][1]['coverage']
        lines += ['',f"候选首个正式请求按自身实际轨迹核算：KV-only 避免逻辑读写 {grouped['recomputed_kv_only_removed']['read_plus_write_bytes']:,} B；新增 Q pack {grouped['recomputed_added_query_pack']['read_plus_write_bytes']:,} B；净避免的显式逻辑读写 {grouped['recomputed_net_removed']['read_plus_write_bytes']:,} B。实际每条明细另存 JSON，均不是实测 DRAM。",'']
    lines+=['## 限制','']+['- '+s for s in report['limitations']]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--candidate',type=Path)
    p.add_argument('--known-graphics-pid',type=int,action='append',default=[])
    p.add_argument('--compiled-reference',type=Path,default=REFERENCE)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args(argv)
    report=audit(args.native,args.candidate,args.known_graphics_pid,args.compiled_reference);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'audit.md').write_text(markdown(report))
    print(json.dumps(dict(runs=[{k:r[k] for k in ('mode','mechanical_execution_pass','formal_aggregate','hardware_summary')}
        for r in report['runs']],comparison=None if report['comparison'] is None else
        {k:report['comparison'][k] for k in ('exact_output_and_full_trajectory_pass','exploratory_timing_comparison_eligible','exploratory_change')}),indent=2))


if __name__=='__main__':main()
