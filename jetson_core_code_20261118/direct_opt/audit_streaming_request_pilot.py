"""CPU-only bounded long-request audit; does not assign a numerical tolerance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_dram_campaign import equal, require, validate_output

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'coordination/direct_opt_20261002'
RID='direct_quality_vdc_v_sPK-sbHIb3w'
METHOD='streaming_online_softmax_fp32'
ARM='memory_shared_embedding_streaming_g28_t48_fp32_fullkv'
PRECISION=dict(target='float16',output_head='float32',target_sdpa='math',draft='float16',draft_sdpa='auto')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(directory):
    status=json.loads((directory/'status.json').read_text())
    rows=[json.loads(x) for x in (directory/'results.jsonl').read_text().splitlines() if x.strip()]
    equal(status['status'],'completed','run completed')
    equal(len(rows),status['completed_cells'],'record coverage')
    equal(status['completed_cells'],status['total_cells'],'planned cell coverage')
    require(len({(r['request_id'],r['arm'],r['phase'],r['repeat'])for r in rows})==len(rows),'duplicate cells')
    for r in rows:validate_output(r,status)
    return status,rows


def validate_streaming(row):
    equal(row['request_id'],RID,'bounded pilot request')
    equal(row['arm'],ARM,'streaming arm')
    equal(row['scheme'],dict(depth='fixed0',adaptor=METHOD,compression=None,ratio=1.0,
        streaming_heads_per_group=28,streaming_token_tile=48),'actual scheme')
    equal((row['num_tokens'],row['verify_rounds'],row['stop_reason'],row['truncated']),(478,164,'eos',False),'natural complete request')
    equal(row['stats']['mode'],'native','complete vision mode')
    equal(row['stats']['calls'],[],'no visual skipping')
    equal(row['stats']['executed_token_blocks'],row['stats']['original_token_blocks'],'full visual depth')
    expected=dict(PRECISION,draft_visual_adaptor=METHOD,draft_visual_pool_dtype='float32',
        draft_visual_projection_dtype='float16',visual_adaptor_bitwise_equivalence_claim=False,streaming_allow_tf32=False)
    equal(row['result']['precision'],expected,'actual computation precision metadata')
    prep=row['adaptor_preparation']
    equal(prep['mode'],METHOD,'installed mode')
    equal(prep['streaming_allow_tf32'],False,'request TF32 disabled')
    require('ring_kv' not in prep and 'fusion_mode' not in prep,'no unrelated runtime/fusion override')
    for key in ('installed','bitwise_equal','same_parameter','same_storage'):
        equal(prep['memory_installation'][key],True,'frozen embedding alias '+key)
    c=prep['streaming_request']
    for k,v in dict(started_calls=1,completed_calls=1,input_lengths=[2016],input_shapes=[[1,2016,3584]],
        input_tokens=2016,failed=False,dense_input_finite_check=False,live_workspace_snapshots=False,
        target_cache_policy='native_full_dynamic_cache',new_kv_policy_installed=False).items():
        equal(c[k],v,'actual module receipt '+k)
    equal(len(c['receipts']),1,'one completed adaptor receipt')
    r=c['receipts'][0]
    for k,v in dict(completed=True,input_shape=[1,2016,3584],output_shape=[1,2,3584],
        head_groups=1,tiles_per_group=42,input_scans=1,tokens_visited_per_batch=2016,
        logical_input_source_read_bytes=14450688,logical_reads_are_measured_dram=False).items():
        equal(r[k],v,'streaming forward '+k)
    equal(r['workspace']['enabled'],False,'no timed workspace diagnostics')
    fc=row['result']['forward_counts']
    equal(fc['streaming_imadpt_actual_calls'],1,'actual adaptor call count')
    equal(fc['streaming_imadpt_actual_input_tokens'],2016,'actual adaptor input tokens')
    equal(fc['merge_visual_tokens_in'],2016,'full visual tokens in')
    equal(fc['merge_visual_tokens_out'],2016,'full visual tokens out')
    p=row['result']['prompt_tokens']
    for t in row['result']['rounds']:
        if t['kind']!='tree':continue
        sel=t['selected_tree_indices']
        equal(t['tree_nodes'],30,'fixed tree nodes')
        require(sel and sel[0]==0 and len(set(sel))==len(sel) and all(type(i)is int and 0<=i<30 for i in sel),'valid selected path indices')
        equal(len(sel),len(t['submitted']),'path and committed count')
        equal(t['target_cache_before'],p,'semantic cache chain')
        p+=len(t['submitted'])
        equal(t['target_cache_after'],p,'semantic postcommit')
    equal(p,row['result']['target_cache_length'],'final cache')
    return dict(phase=row['phase'],repeat=row['repeat'],tokens=478,verify_rounds=164,natural_eos=True,
        output_sha256=digest(row['output_token_ids']),full_rounds_sha256=digest(row['result']['rounds']),
        actual_adaptor_calls=1,actual_adaptor_input_tokens=2016,fp32_pooling=True,tf32=False,
        direct_finite_check_during_request=False,finite_intermediate_tensors_verified=None,
        measured_e2e_wall_s=row['e2e_wall_s'],generation_wall_s=row['generation_wall_s'],
        peak_allocated_bytes=row['peak_allocated_bytes'],peak_reserved_bytes=row['peak_reserved_bytes'],
        hardware={k:row['hardware'].get(k) for k in ('sharing','energy_attribution','foreign_processes','process_visibility_complete')})


def compare(candidate, baseline):
    for k in ('request_id','input_shapes','video_grid_thw','timestamps'):
        equal(candidate[k],baseline[k],'same input '+k)
    for k,v in PRECISION.items():equal(baseline['result']['precision'][k],v,'baseline '+k)
    a,b=baseline['result']['rounds'],candidate['result']['rounds']
    differences=[]
    for i in range(max(len(a),len(b))):
        x=a[i] if i<len(a) else None;y=b[i] if i<len(b) else None
        if x!=y:
            changed={k:dict(baseline=x.get(k),candidate=y.get(k)) for k in set(x)|set(y) if x.get(k)!=y.get(k)} if x is not None and y is not None else dict(baseline=x,candidate=y)
            differences.append(dict(round_index_zero_based=i,changed=changed))
    cf,bf=candidate['result']['forward_counts'],baseline['result']['forward_counts']
    common=set(cf)&set(bf)
    countdiff={k:dict(baseline=bf[k],candidate=cf[k]) for k in sorted(common) if bf[k]!=cf[k]}
    return dict(same_output=candidate['output_token_ids']==baseline['output_token_ids'],
        both_natural_eos=candidate['stop_reason']==baseline['stop_reason']=='eos',
        same_full_rounds=a==b,changed_round_count=len(differences),round_differences=differences,
        common_forward_counts_equal=not countdiff,common_forward_count_differences=countdiff,
        candidate_only_count_keys=sorted(set(cf)-set(bf)),baseline_only_count_keys=sorted(set(bf)-set(cf)),
        baseline_arm=baseline['arm'],baseline_repeat=baseline['repeat'],baseline_phase=baseline['phase'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=BASE/'streaming_adaptor_pilot478_r1')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    status,rows=read(args.run)
    equal([(r['phase'],r['repeat'])for r in rows],[('warmup',0),('measure',0),('measure',1)],'one warmup two measures')
    for k,v in dict(streaming_method=METHOD,streaming_heads_per_group=28,streaming_token_tile=48,
        streaming_allow_tf32=False,target_cache_policy='native_full_dynamic_cache',new_kv_policy_installed=False,
        actual_sram_residency=False).items():equal(status[k],v,'status '+k)
    validated=[validate_streaming(r) for r in rows]
    comparisons={}
    for name in ('throughput_native_delta_abba_r1/00_native','throughput_native_delta_abba_r1/03_native',
                 'memory_shared_compiled_vdc12_r1','deploy_compiled_vdc12_r1'):
        folder=BASE/name;bs,br=read(folder)
        for k in ('request_sha256','target','draft'):equal(bs[k],status[k],'comparison protocol '+k)
        equal(bs['args']['max_new_tokens'],status['args']['max_new_tokens'],'comparison generation limit')
        matched=[r for r in br if r['request_id']==RID and r['phase']=='measure']
        equal(len(matched),1,'one existing reference cell')
        comparisons[name]=dict(source=str(folder.resolve()),
            baseline_seed=bs['args']['seed'],candidate_seed=status['args']['seed'],
            seed_matches=bs['args']['seed']==status['args']['seed'],
            source_results_sha256=hashlib.sha256((folder/'results.jsonl').read_bytes()).hexdigest(),
            pairs=[dict(candidate_phase=c['phase'],candidate_repeat=c['repeat'],**compare(c,matched[0]))for c in rows])
    within=[compare(r,rows[0])for r in rows[1:]]
    result=dict(scope='One independent long-answer request, one warmup plus two measured repeats; no broad quality or speed claim',
        run=str(args.run.resolve()),status_sha256=hashlib.sha256((args.run/'status.json').read_bytes()).hexdigest(),
        results_sha256=hashlib.sha256((args.run/'results.jsonl').read_bytes()).hexdigest(),
        source_sha256=status['source_sha256'],validated_cells=validated,comparisons=comparisons,
        all_three_outputs_and_rounds_equal=all(c['same_output'] and c['same_full_rounds']for c in within),
        finite_evidence='Full-request finite diagnostics were intentionally disabled. Prior four-capture module audit (including N2016) supplies finite adaptor-output evidence; this run does not directly verify every intermediate tensor.',
        cross_batch_speed_comparison=False,quality_claim='This request preserves the exact baseline text. Repeats do not establish 12-video quality.',
        recommendation='Proceed with the already planned 12 long-answer quality evaluation, using the same G28/B48/FP32 scheme. No post-hoc tolerance or parameter selection is justified by this one request.')
    lines=['# G28/B48 单长回答 pilot 独立审计','',
        '**可以按原计划扩到12个长回答。** 本次一个请求的一次暖机、两次测量均自然EOS，478 token、164轮；三次输出与完整round记录彼此一致。尚未证明跨视频质量。','',
        '| 对照 | 三次输出逐token一致 | 自然EOS | 共同forward计数一致 | 每次不同完整round数 |','|---|---:|---:|---:|---:|']
    for name,c in comparisons.items():
        pairs=c['pairs'];lines.append(f"| {name} | {sum(x['same_output']for x in pairs)}/3 | {sum(x['both_natural_eos']for x in pairs)}/3 | {sum(x['common_forward_counts_equal']for x in pairs)}/3 | {sorted(set(x['changed_round_count']for x in pairs))} |")
    lines+=['','原版对照有2轮selected_tree_indices不同（从0编号的59、122）；compiled对照有5轮路径不同。输出token、每轮提交数和全部共同forward_counts相同，但不能把完整验证轨迹写成一致。具体路径差值保存在JSON。','',
        '## 实际运行合同','',
        '- 每次适配器实际调用1次，输入[1,2016,3584]、输出[1,2,3584]；G28/B48，一遍输入、42个tile。原imadpt_calls=0是旧override计数，独立hooks提供真实调用数。',
        '- 全视觉深度、全部视觉token；Target/Draft及其普通DynamicCache保留。只有视觉适配器采用FP32评分/在线归约，池化结果转回FP16后执行V/O。Target/Draft主体FP16、共享LM head FP32，Target math SDPA，TF32关闭。',
        '- 计时路径的finite检查和workspace诊断关闭。先前N2016模块诊断输出finite；本次未直接检查全部中间张量finite，不能从自然EOS反推这一点。',
        '- 每次采样都观察到图形PID6360；energy_attribution=unavailable_shared_device。CPU文件未修改原始结果。',
        '- 模型路径、请求manifest、输入形状/时间点及512上限相同。ABBA native与新pilot seed均20261001；两个较早compiled对照seed为20261002，因此明确是历史结果的描述性核对，不能称完全相同运行条件的重新配对实验。','',
        '## 有界结论','',
        '当前请求的回答文字与原版、compiled都一致，支持继续按原计划验证12题；不新增误差容差，也不因本题长度相同就跳过跨视频事实检查。扩展保持参数固定，核自然EOS、逐token与完整轨迹，若文字变化再按现有参考关键事实审查。','',
        '两次测量只是同题重复。时间和峰值原始观测保存在JSON，未比较跨批次加速；此Torch功能参考没有实测SRAM或DRAM。','']
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'STREAMING_LONG_PILOT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.out/'STREAMING_LONG_PILOT_AUDIT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(validated_cells=len(validated),all_three_outputs_and_rounds_equal=result['all_three_outputs_and_rounds_equal'],comparisons={k:v['pairs'][0]['changed_round_count']for k,v in comparisons.items()}),indent=2))


if __name__=='__main__':main()
