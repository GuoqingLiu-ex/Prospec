"""Offline ABBA audit: original/native and complete grouped deployment.

Four independent processes, twelve videos, two repeats each. Raw package
measurements survive output/trajectory differences. Same-output EOS and full
trajectory subsets are separate, with paired background exclusions. Bootstrap
resamples videos with all their process/repeat observations kept together.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from assess_visual_kv import generation
from audit_native_delta_abba import aggregate,hardware_evidence
from audit_grouped_generation import compare,common_counts,round_changes,coverage,background
from run_original_grouped_timing import PUBLIC_ENV,SCHEMA

HERE=Path(__file__).resolve().parent
ROLES=('original','grouped','grouped','original')
NATIVE_PRECISION=dict(target='float16',output_head='float32',target_sdpa='math',draft='float16',draft_sdpa='auto')
FLAGS=dict(allow_tf32=False,float32_matmul_precision='highest',reduced_precision_math_sdp=False)
SOURCES=('direct_opt/run_original_grouped_timing.py','direct_opt/run_joint.py',
    'direct_opt/run_memory_layout.py','direct_opt/run_deployed_adaptor.py','direct_opt/shared_embedding.py',
    'direct_opt/compiled_adaptor.py','direct_opt/run_grouped_gqa_layout.py','direct_opt/grouped_gqa_runtime.py',
    'direct_opt/grouped_gqa_math.py','direct_opt/stride0_gqa_runtime.py','direct_opt/stride0_gqa_math.py',
    'vendor/hf517_vispec_adapter.py','vendor/vispec_joint_adapter.py')
TIMES=('e2e_wall_s','generation_wall_s','model_wall_s','prepare_wall_s','policy_prepare_wall_s',
       'output_decode_wall_s','runtime_finalize_wall_s')


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):return tuple(row[k] for k in ('request_id','arm','phase','repeat'))


def flags_valid(value):
    return (isinstance(value,dict) and set(value)==set(FLAGS) and type(value['allow_tf32']) is bool
        and type(value['reduced_precision_math_sdp']) is bool
        and value['float32_matmul_precision'] in ('highest','high','medium'))


def precision_receipt(folder,status,rows,schedule,role):
    receipt=json.loads((folder/'precision_observation.json').read_text())
    require(receipt.get('schema')==SCHEMA and receipt.get('status')=='completed','missing/incomplete precision observation')
    require(receipt.get('comparison_arm')==role and receipt.get('pid')==status['pid'],'precision mode/PID mismatch')
    require(isinstance(receipt.get('run_id'),str) and bool(receipt['run_id']),'missing precision invocation identity')
    require(receipt.get('smoke_only') is False,'smoke observations cannot enter the ABBA result')
    require(receipt.get('flags_are_read_only') is True and receipt.get('precision_setters_added') is False,'precision observer changed flags')
    require(receipt.get('source_sha256')==digest(HERE/'run_original_grouped_timing.py'),'precision observer source mismatch')
    require(status['source_sha256'].get('direct_opt/run_original_grouped_timing.py')==receipt['source_sha256'],'child source differs from observer')
    require(receipt.get('manifest_sha256')==status['request_sha256'],'precision manifest binding mismatch')
    require(receipt.get('abbreviated_recipe')==dict(warmup=1,repeats=2,max_new_tokens=512,seed=20261001),'observation recipe mismatch')
    require(len(receipt.get('calls',[]))==len(rows),'missing/extra precision observations')
    require(set(receipt.get('public_environment',{}))==set(PUBLIC_ENV) and
        receipt.get('public_environment_after')==receipt['public_environment'],'public environment changed')
    require(flags_valid(receipt.get('default_startup_flags')) and flags_valid(receipt.get('exit_flags')),'invalid default/exit precision')
    require(receipt.get('final_file_sha256')=={name:digest(folder/name) for name in
        ('status.json','results.jsonl','requests.json','schedule.json')},'precision sidecar file binding mismatch')
    for i,(call,row,cell) in enumerate(zip(receipt['calls'],rows,schedule)):
        require(call.get('call_index')==call.get('cell_index')==i and call.get('cell')==cell and key(row)==key(cell),'precision cell order mismatch')
        require(call.get('generation_completed') is True and 'error' not in call,'failed generation observation')
        require(flags_valid(call.get('before')) and flags_valid(call.get('after')),'invalid actual generation precision')
    return receipt



def process_coverage(row):
    """Conservative offline qualification; samples never prove exclusivity."""
    h=row['hardware'];samples=h.get('samples',[])
    queried=[s for s in samples if 'process_query_complete' in s]
    times=[s.get('elapsed_s') for s in queried]
    duration=h.get('measurement_duration_s');interval=h.get('process_interval_s');sampling=h.get('sample_interval_s')
    numeric=all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in [duration,interval,sampling,*times])
    if not numeric:return dict(passed=False,reason='missing/invalid sampler coverage timestamps')
    gap_limit=2*interval+2*sampling;edge_limit=interval+2*sampling
    gaps=[b-a for a,b in zip(times,times[1:])]
    checks=dict(at_least_two=len(times)>=2,declared_query_count=h.get('process_query_count')==len(queried),
        declared_failures=h.get('process_query_failures')==sum(s['process_query_complete'] is False for s in queried),
        all_complete=all(s['process_query_complete'] is True for s in queried),
        positive_intervals=interval>0 and sampling>0,ordered=all(g>0 for g in gaps),within_duration=all(0<=t<=duration for t in times),
        start_observed=bool(times) and times[0]<=edge_limit,
        end_observed=bool(times) and duration-times[-1]<=edge_limit,
        bounded_gap=bool(gaps) and max(gaps)<=gap_limit,
        request_duration_covered=duration+2*sampling>=row['e2e_wall_s'])
    return dict(passed=all(checks.values()),checks=checks,queried_samples=len(queried),
        measurement_duration_s=duration,first_s=times[0] if times else None,last_s=times[-1] if times else None,
        largest_gap_s=max(gaps) if gaps else None,gap_limit_s=gap_limit,edge_limit_s=edge_limit,
        threshold_scope='offline predeclared 2*process_interval+2*sample_interval gap and process_interval+2*sample_interval endpoints')


def read_run(path,role,known_graphics):
    path=Path(path);status=json.loads((path/'status.json').read_text());args=status['args']
    rows=[json.loads(line) for line in (path/'results.jsonl').read_text().splitlines() if line.strip()]
    requests=json.loads((path/'requests.json').read_text());schedule=json.loads((path/'schedule.json').read_text())
    terminal=json.loads(Path(str(path)+'.exit.json').read_text())
    require(status['status']=='completed' and terminal['returncode']==0,'run not successfully terminal')
    require(type(status.get('pid')) is int and status['pid']>0,'invalid workload PID')
    require(all(type(terminal.get(k)) in (int,float) and math.isfinite(terminal[k]) for k in ('started','ended'))
        and terminal['ended']>=terminal['started'],'invalid process time interval')
    require(len(requests)==len({r['request_id'] for r in requests})==12,'twelve unique requests required')
    require(args['warmup']==1 and args['repeats']==2 and args['max_new_tokens']==512 and args['seed']==20261001,'wrong fixed recipe')
    require(args.get('request_id') is None and not args.get('resume') and not args.get('include_ar'),'full manifest/fresh speculative run required')
    require(len(rows)==status['completed_cells']==status['total_cells']==25,'missing/extra result cells')
    require(len(set(map(key,rows)))==25 and list(map(key,rows))==list(map(key,schedule)),'duplicate/mismatched scheduled cells')
    require(rows[0]['phase']=='warmup' and rows[0]['repeat']==0 and rows[0]['request_id']==requests[0]['request_id'],'wrong warmup cell')
    require(all(r['phase']=='measure' for r in rows[1:]) and {(r['request_id'],r['repeat']) for r in rows[1:]}==
        {(r['request_id'],i) for r in requests for i in (0,1)},'formal request/repeat coverage differs')
    arm='memory_native_native' if role=='original' else 'memory_shared_embedding_compiled_fullkv_tree_grouped_math'
    require(args['arms']==[arm] and {r['arm'] for r in rows}=={arm},'wrong comparison arm')
    require(status['memory_layout']==('native' if role=='original' else 'shared_embedding') and
        status['deployment_adaptor']==('native' if role=='original' else 'compiled'),'wrong deployment layout')
    require('ring_variant' not in status and 'ring_policy' not in status,'unexpected KV policy')
    precision=precision_receipt(path,status,rows,schedule,role)
    require(precision['selected_request_ids']==[r['request_id'] for r in requests],'precision selected-request order differs')
    source_checks={name:status['source_sha256'].get(name)==digest(HERE.parent/name) for name in SOURCES}
    details=[]
    for i,row in enumerate(rows):
        require(row['scheme']['depth']=='fixed0' and row['scheme']['compression'] is None and row['scheme']['ratio']==1.0
            and row['depth_mode']=='fixed0' and row['visual_keep_ratio']==1.0 and row['capture_enabled'] is False and row.get('under_ncu',False) is False,'changed visual/profiling recipe')
        gen=generation(row,status)
        if role=='original':
            require(row['result']['precision']==NATIVE_PRECISION and 'tree_attention' not in row['adaptor_preparation']
                and 'ring_kv' not in row['adaptor_preparation'],'original contains a candidate policy')
            cov=dict(passed=True,scope='native entry/source identity; no new attention policy installed')
        else:cov=coverage(row,'grouped')
        phases=row['generation_wall_s']+row['output_decode_wall_s']+row['runtime_finalize_wall_s']
        checks=dict(finite_positive=all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0 for k in TIMES)
            and row['generation_wall_s']>0 and row['e2e_wall_s']>0,
            model_sum=math.isclose(row['model_wall_s'],phases,abs_tol=1e-6),
            e2e_sum=math.isclose(row['e2e_wall_s'],phases+row['prepare_wall_s']+row['policy_prepare_wall_s'],abs_tol=1e-3))
        hw=background(row,status['pid'],known_graphics)
        hw['coverage']=process_coverage(row)
        hw['sampled_background_clear']=hw['sampled_background_clear'] and hw['coverage']['passed']
        call=precision['calls'][i]
        precision_stable=call['before']==call['after']
        if role=='grouped':precision_stable=precision_stable and call['before']==FLAGS
        details.append(dict(request_id=row['request_id'],phase=row['phase'],repeat=row['repeat'],
            generation=gen,coverage=cov,timing_checks=checks,hardware=hw,
            actual_precision=dict(before=call['before'],after=call['after']),precision_stable=precision_stable,
            mechanical_valid=gen['receipt_valid'] and cov['passed'] and all(checks.values()) and precision_stable))
    within=[]
    for request in requests:
        same=[r for r in rows if r['request_id']==request['request_id']]
        within.extend(dict(request_id=request['request_id'],phase=r['phase'],repeat=r['repeat'],checks=compare(same[0],r)) for r in same[1:])
    return dict(path=str(path.resolve()),role=role,status=status,rows=rows,requests=requests,schedule=schedule,
        terminal=terminal,precision=precision,source_checks=source_checks,details=details,
        within_run=within,within_run_exact=all(all(x['checks'].values()) for x in within),
        mechanical_pass=all(source_checks.values()) and all(d['mechanical_valid'] for d in details),
        hashes={name:digest(path/name) for name in ('status.json','results.jsonl','requests.json','schedule.json','precision_observation.json')})


def aggregate_extra(rows):
    if not rows:return None
    value=aggregate(rows)
    effective=sum(r.get('effective_draft_tokens',0) for r in rows)
    rounds=value['verify_rounds']
    value.update(effective_draft_tokens=effective,effective_draft_tokens_per_verify=effective/rounds if rounds else None,
        common_forward_counts={k:sum(common_counts(r).get(k,0) for r in rows) for k in sorted({k for r in rows for k in common_counts(r)})},
        natural_eos=sum(r['stop_reason']=='eos' and not r['truncated'] for r in rows),
        max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
        max_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows))
    return value


def rate_change(a,b):
    return {phase:100*(b[phase]['tokens_per_second']/a[phase]['tokens_per_second']-1) for phase in ('generation','e2e')}


def percentile(values,p):
    values=sorted(values);f=(len(values)-1)*p;i=int(f);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(f-i)


def video_bootstrap(pairs,iterations,seed):
    groups=defaultdict(list)
    for pair in pairs:groups[pair['request_id']].append(pair)
    videos=sorted(groups)
    if not videos:return None
    # Each cluster retains every available AB/BA x repeat observation.
    rng=random.Random(seed);values={phase:[] for phase in ('generation','e2e')}
    for _ in range(iterations):
        selected=[p for _ in videos for p in groups[rng.choice(videos)]]
        nt=sum(p['_original']['num_tokens'] for p in selected);gt=sum(p['_grouped']['num_tokens'] for p in selected)
        for phase in values:
            na=sum(p['_original'][phase+'_wall_s'] for p in selected);gb=sum(p['_grouped'][phase+'_wall_s'] for p in selected)
            values[phase].append(100*((gt/gb)/(nt/na)-1))
    return dict(unit='video; all process/repeat pairs in a sampled video retained together',distinct_videos=len(videos),
        pairs=len(pairs),iterations=iterations,seed=seed,
        confidence_level=.95,throughput_change_pct_ci={phase:[percentile(v,.025),percentile(v,.975)] for phase,v in values.items()})


def summarize_pairs(pairs,iterations,seed):
    if not pairs:return dict(pair_count=0,video_count=0,original=None,grouped=None,throughput_change_pct=None,bootstrap=None)
    a=aggregate_extra([p['_original'] for p in pairs]);b=aggregate_extra([p['_grouped'] for p in pairs])
    per=[]
    for vid in sorted({p['request_id'] for p in pairs}):
        subset=[p for p in pairs if p['request_id']==vid]
        av=aggregate_extra([p['_original'] for p in subset]);bv=aggregate_extra([p['_grouped'] for p in subset])
        per.append(dict(request_id=vid,pair_count=len(subset),throughput_change_pct=rate_change(av,bv),
            median_paired_latency_saving_pct={phase:statistics.median(100*(1-p['_grouped'][phase+'_wall_s']/p['_original'][phase+'_wall_s']) for p in subset)
                for phase in ('generation','e2e')}))
    saved=a['e2e']['total_seconds']-b['e2e']['total_seconds']
    prepare_saved=a['phase_seconds']['prepare_wall_s']-b['phase_seconds']['prepare_wall_s']
    return dict(pair_count=len(pairs),video_count=len(per),original=a,grouped=b,throughput_change_pct=rate_change(a,b),
        per_video=per,macro_median_paired_latency_saving_pct={phase:statistics.median(r['median_paired_latency_saving_pct'][phase] for r in per) for phase in ('generation','e2e')},
        prepare_wall_saved_seconds=prepare_saved,prepare_share_of_e2e_saved=prepare_saved/saved if saved else None,
        all_pairs_same_math_flags=all(p['precision_matched'] for p in pairs),
        all_pairs_input_metadata_equal=all(p['input_metadata_equal'] for p in pairs),
        all_pairs_mechanically_valid=all(p['mechanical_valid'] for p in pairs),
        all_pairs_sampled_background_clear=all(p['background_clear'] for p in pairs),
        bootstrap=video_bootstrap(pairs,iterations,seed))


def audit(paths,known_graphics=(),bootstrap_samples=2000,bootstrap_seed=20261003):
    require(len(paths)==4,'four ABBA process directories required')
    require(type(bootstrap_samples) is int and bootstrap_samples>=2,'at least two bootstrap samples required')
    runs=[read_run(path,role,known_graphics) for path,role in zip(paths,ROLES)];first=runs[0]
    config={key:all(r['status'][key]==first['status'][key] for r in runs) for key in ('target','draft','torch','transformers','device_name','request_sha256')}
    config.update({'args.'+k:all(r['status']['args'][k]==first['status']['args'][k] for r in runs) for k in ('physical_gpu','seed','max_new_tokens')})
    config.update(full_manifest=all(r['requests']==first['requests'] for r in runs),four_distinct_pids=len({r['status']['pid'] for r in runs})==4,
        four_distinct_observation_ids=len({r['precision']['run_id'] for r in runs})==4,
        sequential=all(runs[i]['terminal']['ended']<=runs[i+1]['terminal']['started'] for i in range(3)),
        same_public_environment=all(r['precision']['public_environment']==first['precision']['public_environment'] for r in runs),
        observed_precision_restored=all(r['precision']['exit_flags']==r['precision']['default_startup_flags'] for r in runs))
    devices=[{row['hardware']['device']['uuid'] for row in r['rows']} for r in runs]
    config['same_physical_uuid']=len(devices[0])==1 and all(d==devices[0] for d in devices)
    config['same_schedule_order']=all([(x['request_id'],x['phase'],x['repeat']) for x in r['schedule']]==
        [(x['request_id'],x['phase'],x['repeat']) for x in first['schedule']] for r in runs)
    config.update({'source.'+name:len({r['status']['source_sha256'].get(name) for r in runs})==1 for name in SOURCES})
    maps=[{(row['request_id'],row['repeat']):(row,d) for row,d in zip(r['rows'],r['details']) if row['phase']=='measure'} for r in runs]
    pairs=[]
    for label,ni,gi in (('AB',0,1),('BA',3,2)):
        require(maps[ni].keys()==maps[gi].keys(),'paired cells differ')
        for request_id,repeat in sorted(maps[ni]):
            a,ad=maps[ni][request_id,repeat];b,bd=maps[gi][request_id,repeat]
            checks=compare(a,b);same_eos=checks['output_token_ids'] and ad['generation']['natural_eos'] and bd['generation']['natural_eos']
            inputs=all(checks[k] for k in ('input_shapes','video_grid_thw','timestamps','result.prompt_tokens'))
            clean=ad['hardware']['sampled_background_clear'] and bd['hardware']['sampled_background_clear']
            reasons=[]
            for role,d in (('original',ad),('grouped',bd)):
                if not d['hardware']['sampled_background_clear']:reasons.append(dict(arm=role,hardware=d['hardware']))
            pairs.append(dict(request_id=request_id,repeat=repeat,block_pair=label,checks=checks,
                same_output_eos=bool(same_eos),full_trajectory_exact=bool(same_eos and all(checks.values())),
                input_metadata_equal=inputs,precision_matched=ad['actual_precision']==bd['actual_precision'],
                mechanical_valid=ad['mechanical_valid'] and bd['mechanical_valid'],background_clear=bool(clean),
                excluded_background_reasons=reasons,round_changes=round_changes(a['result']['rounds'],b['result']['rounds']),
                _original=a,_grouped=b))
    require(len(pairs)==48,'twelve videos x two repeats x AB/BA must produce 48 pairs')
    subsets={};filters=(('all_package',lambda p:True),('same_output_eos',lambda p:p['same_output_eos']),
        ('full_trajectory_exact',lambda p:p['full_trajectory_exact']))
    for i,(label,select) in enumerate(filters):
        selected=[p for p in pairs if select(p)]
        subsets[label]=summarize_pairs(selected,bootstrap_samples,bootstrap_seed+i)
        subsets[label+'_background_filtered']=summarize_pairs([p for p in selected if p['background_clear']],bootstrap_samples,bootstrap_seed+i)
    within=[]
    for role,indices in (('original',(0,3)),('grouped',(1,2))):
        for request in first['requests']:
            cells=[row for i in indices for row in runs[i]['rows'] if row['request_id']==request['request_id']]
            within.append(dict(arm=role,request_id=request['request_id'],cells=len(cells),
                output_ids_stable=all(row['output_token_ids']==cells[0]['output_token_ids'] for row in cells),
                full_trajectory_stable=all(all(compare(cells[0],row).values()) for row in cells)))
    mechanical=all(config.values()) and all(r['mechanical_pass'] for r in runs) and all(p['input_metadata_equal'] for p in pairs)
    precision_same=all(p['precision_matched'] for p in pairs)
    block_summaries=[dict(role=r['role'],directory=r['path'],pid=r['status']['pid'],terminal=r['terminal'],hashes=r['hashes'],
        source_checks=r['source_checks'],mechanical_pass=r['mechanical_pass'],precision=r['precision'],
        within_run_exact=r['within_run_exact'],formal_aggregate=aggregate_extra(r['rows'][1:]),
        details=r['details'],raw_valid_timing_counts=dict(Counter(str(row['valid_timing']) for row in r['rows']))) for r in runs]
    return dict(schema='original_grouped_abba_v1',tool_sha256=digest(__file__),
        helper_sha256={name:digest(HERE/name) for name in ('assess_visual_kv.py','audit_native_delta_abba.py','audit_grouped_generation.py','run_original_grouped_timing.py')},
        configuration_checks=config,mechanical_package_comparison_pass=mechanical,math_flags_match_across_arms=precision_same,
        pure_execution_attribution_eligible=mechanical and precision_same and all(p['full_trajectory_exact'] and p['background_clear'] for p in pairs),
        observed_pair_count=48,distinct_videos=12,subsets=subsets,
        order_effect={label:summarize_pairs([p for p in pairs if p['block_pair']==label],bootstrap_samples,bootstrap_seed) for label in ('AB','BA')},
        pairs=[{k:v for k,v in p.items() if not k.startswith('_')} for p in pairs],
        paired_background_exclusions=[{k:p[k] for k in ('request_id','repeat','block_pair','excluded_background_reasons')} for p in pairs if not p['background_clear']],
        within_arm_reproducibility=within,blocks=block_summaries,quality_pass=None,actual_dram_measured=False,
        limitations=[
            'This is the project HF5.17 ViSpec compatibility path versus shared embedding + compiled adaptor + grouped tree attention; no individual component attribution.',
            'All-package observed throughput is retained even when answers or accepted paths change; it is not a same-quality or same-workload claim.',
            'Actual math flags are read-only observations. A mismatch permits an as-configured package measurement, not a pure execution comparison.',
            'Primary generation throughput excludes preparation. E2E includes input preparation, output decode and runtime finalization; loading, installation and warmup are excluded.',
            'ABBA reduces order drift but does not remove video filesystem/decoder cache effects; preparation differences are separately reported.',
            'Bootstrap clusters by twelve videos, preserving process/repeat observations; this is sequential batch-one throughput, not a concurrent serving-capacity experiment.',
            'Background exclusions remove both sides of each affected pair. Observed-clean samples do not prove strict exclusivity; original valid_timing fields are unchanged.',
            'Ordinary HardwareSampler merges compute/graphics by PID. A known-graphics exemption requires separately confirmed process identity; these records cannot prove that PID never held a compute context.',
            'Input metadata and manifests are checked; no captured input pixel hashes prove byte-identical decoded video.',
            'Module memory or logical-copy results are not whole-request peak or actual DRAM measurements.'])


def markdown(report):
    lines=['# 原版与 grouped：普通请求 ABBA 审计','',
        f"机械协议通过：{report['mechanical_package_comparison_pass']}；两臂实际 math flags 相同：{report['math_flags_match_across_arms']}。",
        f"全体配对同时满足同精度、同工作量及背景采样资格：{report['pure_execution_attribution_eligible']}；否则下表仅保留原始整套配置测量，不能解释成纯执行收益。",
        '12 个视频，四独立进程，48 对正式请求；预热排除。','',
        '| 子集 | 配对数 | 视频数 | generation token/s 变化 | E2E token/s 变化 |','|---|---:|---:|---:|---:|']
    for label,info in report['subsets'].items():
        rate=info['throughput_change_pct']
        lines.append(f"| {label} | {info['pair_count']} | {info['video_count']} | {rate['generation']:.4f}% | {rate['e2e']:.4f}% |" if rate else f"| {label} | 0 | 0 | — | — |")
    lines+=['','原始全部结果始终保留；同输出 EOS、完整轨迹相同和成对背景过滤单列。生成吞吐为主，E2E 的输入准备变化不能归因于 attention。',
        f"成对背景排除数：{len(report['paired_background_exclusions'])}。逐视频汇总、视频聚类 bootstrap、AB/BA 顺序差、接受贡献与全部 precision 回执见 JSON。",'','## 限制','']
    lines+=['- '+text for text in report['limitations']]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs=4,type=Path,required=True,metavar=('A_FIRST','B_FIRST','B_SECOND','A_LAST'))
    p.add_argument('--known-graphics-pid',type=int,action='append',default=[])
    p.add_argument('--bootstrap-samples',type=int,default=2000);p.add_argument('--bootstrap-seed',type=int,default=20261003)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    report=audit(a.runs,a.known_graphics_pid,a.bootstrap_samples,a.bootstrap_seed)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(a.out/'audit.md').write_text(markdown(report))
    print(json.dumps(dict(mechanical_pass=report['mechanical_package_comparison_pass'],math_flags_match=report['math_flags_match_across_arms'],
        all_package=report['subsets']['all_package']['throughput_change_pct'],same_eos=report['subsets']['same_output_eos']['pair_count'])))
    return report


if __name__=='__main__':main()
