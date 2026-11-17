"""Offline three-arm NCU audit. Reads files only; never launches a workload.

Original = native weights/adaptor and untouched full cache. Control/grouped
share embedding, compiled adaptor and full DynamicCache. Only their paired
comparison isolates tree-attention execution. Raw counters and logical copy
formulas remain separate, including the added grouped Q-pack cost.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import profile_implicit_kv_dram as original_profile
import profile_grouped_gqa_dram as tree_profile
from summarize_dram_campaign import equal, require
from assess_visual_kv import generation
from audit_grouped_generation import common_counts, compare, digest, round_changes

ARMS=('original','control','grouped')
HERE=Path(__file__).resolve().parent
OPTIONS={'--config-file':'off','--target-processes':'application-only','--replay-mode':'app-range',
         '--cache-control':'none','--clock-control':'none','--pipeline-boost-state':'dynamic',
         '--metrics':'dram__bytes_read.sum,dram__bytes_write.sum','--page':'raw'}
MODE_PRECISION={'target_sdpa','target_prefill_attention_backend','target_verify_attention_backend',
    'target_tree_attention_mode','target_tree_math_precision','target_cache_policy',
    'draft_attention_backend','tree_attention_bitwise_equivalence_claim'}
# Declared before reviewing this campaign's counters. These are coarse sampling
# coverage tolerances, not a guarantee against activity between observations.
POLL_INTERVAL_S=.5
ENDPOINT_TOLERANCE_S=3*POLL_INTERVAL_S
MAX_SAMPLE_GAP_S=4*POLL_INTERVAL_S


def read(path):return json.loads(Path(path).read_text())


def option(command,flag):
    equal(command.count(flag),1,'single command option '+flag)
    index=command.index(flag)
    require(index+1<len(command),'missing option argument '+flag)
    return command[index+1]


def command_checks(command,arm,contract):
    equal(command[0],contract['ncu'],'recorded NCU executable')
    for flag,value in OPTIONS.items():equal(option(command,flag),value,'profiler option '+flag)
    for flag in ('--csv','--disable-extra-suffixes'):equal(command.count(flag),1,'profiler flag '+flag)
    require(not any(x in command for x in ('--profile-from-start','--nvtx','--resume','--ring-policy')),
            'unsupported profile filtering/runtime option')
    for flag,value in (('--physical-gpu',str(contract['gpu'])),('--requests',contract['request_path']),
                       ('--request-id',contract['request_id'])):
        equal(option(command,flag),value,'workload command '+flag)
    script='profile_implicit_kv_dram.py' if arm=='original' else 'profile_grouped_gqa_dram.py'
    positions=[i for i,x in enumerate(command) if Path(x).name==script]
    equal(len(positions),1,'one expected profiling wrapper')
    equal(command[positions[0]-1],contract['python'],'recorded Python executable')
    if arm=='original':
        for flag,value in (('--profile-mode','native'),('--profile-scope','generation'),('--profile-calls','2')):
            equal(option(command,flag),value,'original generation option')
        require('--tree-attention' not in command,'original must not install tree policy')
    else:
        equal(option(command,'--tree-attention'),'native' if arm=='control' else 'grouped','tree mode')
        require('--profile-mode' not in command,'control/grouped wrapper mismatch')
    out=Path(option(command,'--out'))
    require(out.name=='workload' and out.parent.name==arm,'arm-specific workload directory')
    equal(option(command,'--log-file'),str(out.parent/'metrics.csv'),'same-arm raw CSV')
    equal(option(command,'--export'),str(out.parent/'dram'),'same-arm NCU binary report')
    return script


def finite_time(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def observation_log_review(observations,expected_count,declared_poll_interval_s,monitor_error=None):
    """Background defects remove interpretation eligibility, not raw counters."""
    times=[r.get('time_unix') if isinstance(r,dict) else None for r in observations]
    valid_times=all(finite_time(t) for t in times)
    ordered=valid_times and all(a<b for a,b in zip(times,times[1:]))
    count_matches=type(expected_count) is int and expected_count==len(observations)
    reasons=[]
    if not observations:reasons.append('empty_or_missing_observation_log')
    if not count_matches:reasons.append('campaign_observation_count_mismatch_or_missing')
    if not valid_times:reasons.append('missing_or_nonfinite_observation_time')
    if not ordered:reasons.append('observation_times_not_strictly_increasing')
    if declared_poll_interval_s!=POLL_INTERVAL_S:reasons.append('poll_interval_differs_from_frozen_0.5s_protocol')
    if monitor_error:reasons.append('campaign_monitor_error')
    return dict(raw_observation_count=len(observations),campaign_observation_count=expected_count,
        count_matches=count_matches,finite_times=valid_times,strictly_increasing_times=ordered,
        declared_poll_interval_s=declared_poll_interval_s,monitor_error=monitor_error,
        sufficient=not reasons,reasons=reasons)


def read_observations(path):
    """Retain malformed-line evidence so a background problem cannot hide counts."""
    if not path.exists():return []
    rows=[]
    for index,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except ValueError as exc:rows.append(dict(observation_parse_error=str(exc),line_number=index))
    return rows


def process_review(observations,own_pids,allowed_graphics,start=None,end=None,*,log_review=None):
    if log_review is None:
        log_review=observation_log_review(observations,len(observations),POLL_INTERVAL_S)
    valid=[r for r in observations if isinstance(r,dict) and finite_time(r.get('time_unix'))]
    interval_source='child_exit_receipt' if start is not None and end is not None else 'observed_log_span'
    if start is None:start=valid[0]['time_unix'] if valid else None
    if end is None:end=valid[-1]['time_unix'] if valid else None
    interval_valid=finite_time(start) and finite_time(end) and start<=end
    rows=[r for r in valid if start<=r['time_unix']<=end] if interval_valid else []
    times=[r['time_unix'] for r in rows]
    start_gap=times[0]-start if times else None
    end_gap=end-times[-1] if times else None
    max_gap=max((b-a for a,b in zip(times,times[1:])),default=None)
    reasons=list(log_review['reasons'])
    if not interval_valid:reasons.append('invalid_observation_interval')
    if len(rows)<2:reasons.append('fewer_than_two_in_interval_observations')
    if start_gap is None or start_gap>ENDPOINT_TOLERANCE_S:reasons.append('start_endpoint_not_covered')
    if end_gap is None or end_gap>ENDPOINT_TOLERANCE_S:reasons.append('end_endpoint_not_covered')
    if max_gap is not None and max_gap>MAX_SAMPLE_GAP_S:reasons.append('internal_sampling_gap_too_large')
    coverage=dict(sufficient=not reasons,reasons=reasons,interval_source=interval_source,
        interval_start_unix_s=start,interval_end_unix_s=end,start_gap_s=start_gap,end_gap_s=end_gap,
        maximum_internal_gap_s=max_gap,minimum_samples=2,poll_interval_s=POLL_INTERVAL_S,
        endpoint_tolerance_s=ENDPOINT_TOLERANCE_S,maximum_sample_gap_s=MAX_SAMPLE_GAP_S,
        log_integrity=log_review)
    unknown=[];residents=[];errors=[]
    for r in rows:
        schema_ok=all(isinstance(r.get(k),list) and all(isinstance(p,dict) and type(p.get('pid')) is int
            for p in r[k]) for k in ('compute','graphics'))
        if not schema_ok:
            errors.append(dict(time_unix=r['time_unix'],error='missing_or_invalid_process_observation_fields'))
            continue
        compute={p['pid'] for p in r['compute']};graphics={p['pid'] for p in r['graphics']}
        # A graphics allowlist never exempts a PID appearing as compute.
        foreign=(compute|(graphics-set(allowed_graphics)))-set(own_pids)
        if foreign:unknown.append(dict(time_unix=r['time_unix'],pids=sorted(foreign)))
        resident=(graphics&set(allowed_graphics))-set(own_pids)
        if resident:residents.append(dict(time_unix=r['time_unix'],pids=sorted(resident)))
        if any(k.endswith('_error') for k in r):errors.append(r)
    return dict(observations=len(rows),unknown_context_observations=unknown,resident_graphics_observations=residents,
        query_errors=errors,sampled_no_unknown_context=bool(rows) and not unknown and not errors,
        coverage=coverage,background_review_clear=coverage['sufficient'] and not unknown and not errors,
        exclusive_process_attribution_proven=False,background_subtracted=False)


def load_arm(root,arm,campaign,observations,log_review):
    folder=root/arm;contract=campaign['contract']
    script=command_checks(read(folder/'command.json'),arm,contract)
    launch=read(folder/'launch.json');terminal=read(folder/'exit.json')
    equal(terminal['returncode'],0,'successful NCU child exit')
    require(terminal['started_unix_s']<=terminal['finished_unix_s'],'child interval')
    env=launch['child_environment_overrides']
    equal(env['CPATH'],contract['headers']['CPATH'],'exact launched header paths')
    equal(env['CUDA_VISIBLE_DEVICES'],str(contract['gpu']),'launched physical device')
    profile,mode=(original_profile,'native') if arm=='original' else (tree_profile,'native' if arm=='control' else 'grouped')
    status,rows=profile.validate_workload(folder/'workload',mode,contract['request_id'])[:2]
    equal(status['pid'],campaign['workload_pids'][arm],'campaign/child PID')
    equal(read(folder/'pid_receipt.json')['workload_pid'],status['pid'],'retained workload PID')
    require(isinstance(status['profiler_run_id'],str) and status['profiler_run_id'],'actual invocation UUID')
    equal(status['args']['physical_gpu'],contract['gpu'],'workload physical GPU')
    equal(status['request_sha256'],contract['request_sha256'],'request manifest hash')
    equal(status['profiler_effective_cpath'],env['CPATH'],'effective child CPATH')
    equal(status['profiler_source_sha256'],contract['source_sha256']['direct_opt/'+script],'executed wrapper hash')
    for path,sha in contract['source_sha256'].items():equal(status['source_sha256'].get(path),sha,'executed source '+path)
    require(status['profile_scope_note'].startswith('generate_joint only;'),'unexpected collection boundary')
    equal(status['profiler_api'],'cudaProfilerStart/Stop','explicit generation boundaries')
    key=lambda r:tuple(r[k] for k in ('request_id','arm','phase','repeat'))
    schedule=read(folder/'workload/schedule.json')
    equal(list(map(key,rows)),list(map(key,schedule)),'workload and schedule cells')
    requests=read(folder/'workload/requests.json')
    equal([r['request_id'] for r in requests],[contract['request_id']],'singleton request manifest')
    for row in rows:
        equal(row['profiler_run_id'],status['profiler_run_id'],'row UUID binding')
        equal(row.get('profiler_mode'),mode,'row profile mode')
        for flag in ('valid_timing','valid_memory','valid_energy'):equal(row.get(flag),False,'diagnostic '+flag)
        equal(row['hardware']['device']['uuid'],campaign['gpu_uuid'],'physical UUID binding')
        gen=generation(row,status)
        require(gen['receipt_valid'] and gen['natural_eos'],'invalid generation/EOS: '+str(gen['trace_validation_error']))
    csv=folder/'metrics.csv'
    require(sum(line.startswith('"ID",') for line in csv.read_text().splitlines())==1,'single raw NCU CSV header')
    records=profile.bind_ncu(status,rows,csv) # byte units, one pass, range order, PID and phase/repeat all verified
    binding=read(folder/'range_binding.json')
    equal((binding['verified'],binding['range_count'],binding['workload_pid'],binding['profiler_run_id']),
          (True,2,status['pid'],status['profiler_run_id']),'saved range receipt identity')
    equal(binding['verified_child_source_sha256'],contract['source_sha256'],'saved source contract')
    measured=[r for r in rows if r['phase']=='measure']
    for record,saved,row in zip(records,binding['records'],measured):
        for key,value in record.items():equal(saved.get(key),value,'saved vs raw range '+key)
        record.update({k:row[k] for k in ('input_shapes','timestamps','video_grid_thw')})
    equal(len(binding['records']),2,'saved raw range count')
    require(binding['binary_report']['bytes']>0,'nonempty original binary report receipt')
    binary=folder/'dram.ncu-rep'
    if binary.exists():equal(binary.stat().st_size,binding['binary_report']['bytes'],'local binary size')
    process=process_review(observations,[status['pid']],contract['allowed_graphics'],
        terminal['started_unix_s'],terminal['finished_unix_s'],log_review=log_review)
    return dict(arm=arm,rows=rows,status=status,records=records,requests=requests,
        output_stable=all(r['output_token_ids']==rows[0]['output_token_ids'] for r in rows),
        rounds_stable=all(r['result']['rounds']==rows[0]['result']['rounds'] for r in rows),
        terminal=terminal,process_review=process,
        binary_local=binary.exists(),binary_recorded_bytes=binding['binary_report']['bytes'],
        source_sha256=contract['source_sha256'],hashes={name:digest(folder/name) for name in
            ('metrics.csv','command.json','launch.json','exit.json','pid_receipt.json','range_binding.json',
             'workload/status.json','workload/results.jsonl','workload/schedule.json','workload/requests.json')})


def compare_arms(left,right,isolated):
    protocol=all(left['status'][k]==right['status'][k] for k in ('target','draft','request_sha256')) and all(
        left['status']['args'][k]==right['status']['args'][k] for k in ('seed','max_new_tokens')) and left['requests']==right['requests']
    pairs=[]
    for a,b,ra,rb in zip(left['rows'][1:],right['rows'][1:],left['records'],right['records']):
        equal((a['request_id'],a['repeat']),(b['request_id'],b['repeat']),'paired cells')
        checks=compare(a,b)
        checks['unchanged_precision_fields']={k:v for k,v in a['result']['precision'].items() if k not in MODE_PRECISION}=={
            k:v for k,v in b['result']['precision'].items() if k not in MODE_PRECISION}
        if isolated:checks['same_explicit_math_flags']=a['result']['precision']['target_tree_math_precision']==b['result']['precision']['target_tree_math_precision']
        pairs.append(dict(repeat=b['repeat'],checks=checks,changed_rounds=round_changes(a['result']['rounds'],b['result']['rounds']),
            raw_reference={k:ra[k] for k in ('read_bytes','write_bytes','total_bytes')},
            raw_candidate={k:rb[k] for k in ('read_bytes','write_bytes','total_bytes')},
            observed_reduction_pct={k:100*(1-rb[k]/ra[k]) if ra[k] else None for k in ('read_bytes','write_bytes','total_bytes')}))
    stable=left['output_stable'] and left['rounds_stable'] and right['output_stable'] and right['rounds_stable']
    matched=protocol and stable and all(all(p['checks'].values()) for p in pairs)
    medians={name:{k:median(r[k] for r in arm['records']) for k in ('read_bytes','write_bytes','total_bytes')}
        for name,arm in (('reference',left),('candidate',right))}
    change={k:100*(1-medians['candidate'][k]/medians['reference'][k]) if medians['reference'][k] else None for k in medians['reference']}
    background_clear=left['process_review']['background_review_clear'] and right['process_review']['background_review_clear']
    coverage_sufficient=left['process_review']['coverage']['sufficient'] and right['process_review']['coverage']['sufficient']
    return dict(reference=left['arm'],candidate=right['arm'],isolated_attention_comparison=isolated,
        same_protocol=protocol,all_calls_stable=stable,strict_same_workload=matched,pairs=pairs,
        median_actual_dram_bytes=medians,observed_reduction_pct=change,
        workload_conditioned_reduction_pct=change if matched and background_clear else None,
        background_review_clear=background_clear,background_coverage_sufficient=coverage_sufficient,
        exclusive_process_attribution_proven=False,
        scope='Only tree attention differs within shared/compiled/full-cache deployment' if isolated else
              'Combined deployment package; changed tree trajectories retained, no isolated attention attribution')


def audit(root,source_root):
    root=Path(root);campaign=read(root/'campaign.json');contract=campaign['contract']
    equal(campaign['status'],'completed','complete three-arm campaign')
    equal(campaign['completed'],list(ARMS),'complete fixed arm order')
    equal(campaign['scope'],'generation','whole generation scope')
    equal((contract['warmup'],contract['repeats'],contract['max_new_tokens']),(1,2,512),'frozen protocol')
    require(campaign['actual_counters_bound'] and campaign['parent_environment_unchanged'],'completed counter/environment receipts')
    for path,sha in contract['source_sha256'].items():equal(digest(source_root/path),sha,'local reviewed source '+path)
    observations=read_observations(root/'gpu_observations.jsonl')
    log_review=observation_log_review(observations,campaign.get('process_observation_count'),
        campaign.get('process_poll_interval_s'),campaign.get('monitor_error'))
    arms={arm:load_arm(root,arm,campaign,observations,log_review) for arm in ARMS}
    equal(len({a['status']['pid'] for a in arms.values()}),3,'three distinct workload PIDs')
    equal(len({a['status']['profiler_run_id'] for a in arms.values()}),3,'three distinct invocation UUIDs')
    require(all(arms[a]['terminal']['finished_unix_s']<=arms[b]['terminal']['started_unix_s'] for a,b in zip(ARMS,ARMS[1:])),
            'serial independent process intervals')
    comparisons={name:compare_arms(arms[a],arms[b],isolated) for name,a,b,isolated in
        (('control_vs_original','original','control',False),('grouped_vs_original','original','grouped',False),
         ('grouped_vs_control','control','grouped',True))}
    grouped_logical=[dict(repeat=row['repeat'],**row['adaptor_preparation']['tree_attention']['logical_copy_totals'])
                     for row in arms['grouped']['rows'][1:]]
    for item in grouped_logical:
        equal(item['net_removed_explicit_bytes'],{k:item['kv_only_removed_explicit_bytes'][k]-item['added_query_pack_bytes'][k]
              for k in ('read_bytes','write_bytes','read_plus_write_bytes')},'net logical Q-pack subtraction')
    return dict(schema='independent_grouped_dram_audit_v2',validated_counter_bindings=True,
        strict_control_grouped_workload_pass=comparisons['grouped_vs_control']['strict_same_workload'],
        tool_sha256=digest(__file__),campaign_sha256=digest(root/'campaign.json'),run=str(root.resolve()),
        campaign_attribution_record=campaign.get('attribution'),comparisons=comparisons,
        arms={name:{k:arm[k] for k in ('arm','records','output_stable','rounds_stable','terminal','process_review',
            'binary_local','binary_recorded_bytes','source_sha256','hashes')} for name,arm in arms.items()},
        logical_copy_estimates_separate_from_dram=grouped_logical,
        observation_log_review=log_review,
        background_all=process_review(observations,campaign['workload_pids'].values(),contract['allowed_graphics'],log_review=log_review),
        profile_scope='generate_joint: vision/prefill, adaptor, Draft/Target decoding and cache operations; '
            'excludes loading/install, input preparation, tokenizer decoding and runtime finalization',
        no_background_subtraction=True,ncu_time_memory_energy_are_performance_evidence=False,
        quality_pass=None,throughput_pass=None,
        limitations=['Two measured app-ranges per arm on one request; no statistical or cross-workload generalization.',
            'Raw DRAM counters may include other GPU contexts. Resident graphics and between-sample activity are not excluded.',
            'Background eligibility requires the recorded raw observation count, finite strictly ordered timestamps, at least two samples in each child interval, endpoint gaps <=1.5 s and internal gaps <=2.0 s under the frozen 0.5 s polling protocol. These coarse tolerances do not prove exclusivity.',
            'Original comparison changes the deployment package and can change speculative trajectories; preserve differences and do not isolate attention savings from it.',
            'Control/grouped requires full IDs, rounds, common counts, inputs and precision agreement for matched-workload interpretation.',
            'Logical copy bytes count explicit tensor-operation payloads, including Q pack; they do not predict actual bus transactions or replace NCU counters.',
            'One-pass CSV and PID/range binding establish observed counters; they do not make profiled wall, energy or allocator values ordinary performance measurements.'])


def markdown(r):
    percent=lambda value:'null' if value is None else f'{value:.4f}%'
    lines=['# Grouped 三臂 generation DRAM 独立验收','',
        f"计数绑定通过：**{r['validated_counter_bindings']}**；control→grouped 严格同工作量：**{r['strict_control_grouped_workload_pass']}**。",'',
        '| 对照 | 同工作量 | 背景覆盖充分 | 背景检查通过 | read 观测减少 | write 观测减少 | total 观测减少 | 条件化 total |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for name,c in r['comparisons'].items():
        d=c['observed_reduction_pct']
        conditional=c['workload_conditioned_reduction_pct']
        lines.append(f"| {name} | {c['strict_same_workload']} | {c['background_coverage_sufficient']} | {c['background_review_clear']} | {percent(d['read_bytes'])} | {percent(d['write_bytes'])} | {percent(d['total_bytes'])} | {percent(None if conditional is None else conditional['total_bytes'])} |")
    lines+=['','original 两项仅是整套部署的原始计数比较；轨迹差异在 JSON 逐轮保留。control→grouped 只有在工作量严格匹配且背景覆盖、进程检查均通过时，才给出条件化比较。背景不足时仅保留原始观测差；workload_conditioned_reduction_pct 为 null。即使通过，也未证明 GPU 独占或排除观测间活动。','',
        '## 背景采样覆盖','',
        f"原始观察数 / campaign 记录数：{r['observation_log_review']['raw_observation_count']} / {r['observation_log_review']['campaign_observation_count']}。全日志完整性通过：{r['observation_log_review']['sufficient']}；原因：{r['observation_log_review']['reasons']}。",
        '固定 0.5 秒采样协议：每臂真实子进程起止区间内至少 2 点，首尾间隙各 ≤1.5 秒，内部最大间隙 ≤2.0 秒；不使用区间外观察替代端点。全日志数量、有限且严格递增时间或监测状态异常，使所有臂失去条件化比较资格。','',
        '| 臂 | 区间内观察数 | 起点间隙 s | 终点间隙 s | 最大内部间隙 s | 覆盖充分 | 失败原因 |',
        '|---|---:|---:|---:|---:|---|---|']
    for name,arm in r['arms'].items():
        process=arm['process_review'];c=process['coverage']
        lines.append(f"| {name} | {process['observations']} | {c['start_gap_s']} | {c['end_gap_s']} | {c['maximum_internal_gap_s']} | {c['sufficient']} | {', '.join(c['reasons'])} |")
    lines+=['',
        '## 实测与理论分别记账','',
        '- 三臂各 2 条 byte 单位、1 pass、完整 app-range 计数，重新绑定 child PID、run UUID、arm/phase/repeat、源码哈希和原始 CSV；原 range_binding 摘要不是唯一数据源。',
        '- grouped 的 KV-only 避免显式搬运、新增 Q pack、二者净差另列于 logical_copy_estimates_separate_from_dram；不混入实测 read/write/total。',
        '- 范围：'+r['profile_scope']+'。',
        '- 原始二进制报告若未同步本地，JSON 会明确 binary_local=false；仍核远端非空回执与本地 CSV，不声称重新解析二进制。','',
        '## 限制','']+['- '+x for x in r['limitations']]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-root',type=Path,default=HERE.parent)
    args=p.parse_args(argv);result=audit(args.run,args.source_root);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'audit.md').write_text(markdown(result))
    print(json.dumps({k:result[k] for k in ('validated_counter_bindings','strict_control_grouped_workload_pass')},indent=2))


if __name__=='__main__':main()
