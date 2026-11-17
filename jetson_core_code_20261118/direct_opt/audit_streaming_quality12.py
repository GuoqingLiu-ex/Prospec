"""Offline twelve-request audit; preserves the separate, frozen pilot auditor.

Changed answers are exported for factual review, never silently accepted by an
error threshold. This entry does not load models, access a GPU or compare speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_streaming_request_pilot import BASE, METHOD, ARM, PRECISION, read, compare, digest
from summarize_dram_campaign import equal, require


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within_110(candidate_tokens, baseline_tokens):
    require(baseline_tokens > 0 and candidate_tokens > 0, 'nonempty output length')
    return 10*candidate_tokens <= 11*baseline_tokens


def validate_candidate(row):
    equal(row['arm'],ARM,'streaming identity')
    equal(row['scheme'],dict(depth='fixed0',adaptor=METHOD,compression=None,ratio=1.0,
        streaming_heads_per_group=28,streaming_token_tile=48),'fixed declared scheme')
    equal(row['stats']['mode'],'native','full visual mode')
    equal(row['stats']['calls'],[],'no visual depth policy')
    equal(row['stats']['executed_token_blocks'],row['stats']['original_token_blocks'],'full visual depth')
    equal(row['result']['precision'],dict(PRECISION,draft_visual_adaptor=METHOD,
        draft_visual_pool_dtype='float32',draft_visual_projection_dtype='float16',
        visual_adaptor_bitwise_equivalence_claim=False,streaming_allow_tf32=False),'audited precision')
    prep=row['adaptor_preparation']
    equal(prep['mode'],METHOD,'installed adaptor')
    equal(prep['streaming_allow_tf32'],False,'TF32 disabled')
    require('ring_kv' not in prep and 'fusion_mode' not in prep,'no unrelated cache/fusion patch')
    for key in ('installed','bitwise_equal','same_parameter','same_storage'):
        equal(prep['memory_installation'][key],True,'exact embedding alias '+key)
    counts=row['result']['forward_counts']
    n=counts['merge_visual_tokens_in']
    equal(counts['merge_visual_tokens_out'],n,'all visual tokens kept')
    require(n>0 and counts['draft_tree_calls']>0,'this bounded long-answer audit expects a processed visual span/tree')
    require(len(row['video_grid_thw'])==1,'single video grid')
    t,h,w=row['video_grid_thw'][0]
    equal(t*h*w//4,n,'actual visual grid vs adaptor input')
    require(h%2==w%2==0,'valid visual spatial merge')
    c=prep['streaming_request']
    for key,value in dict(started_calls=1,completed_calls=1,input_lengths=[n],input_shapes=[[1,n,3584]],
        input_tokens=n,failed=False,dense_input_finite_check=False,live_workspace_snapshots=False,
        target_cache_policy='native_full_dynamic_cache',new_kv_policy_installed=False).items():
        equal(c[key],value,'actual adaptor receipt '+key)
    equal(len(c['receipts']),1,'one observed adaptor call')
    forward=c['receipts'][0]
    for key,value in dict(completed=True,input_shape=[1,n,3584],output_shape=[1,2,3584],head_groups=1,
        tiles_per_group=math.ceil(n/48),input_scans=1,tokens_visited_per_batch=n,
        logical_input_source_read_bytes=n*3584*2,logical_reads_are_measured_dram=False).items():
        equal(forward[key],value,'adaptor forward '+key)
    equal(forward['workspace']['enabled'],False,'workspace diagnostics disabled')
    equal(counts['streaming_imadpt_actual_calls'],1,'actual hook call count')
    equal(counts['streaming_imadpt_actual_input_tokens'],n,'actual hook input count')
    base=row['result']['prompt_tokens']
    tree_count=0
    for tree in row['result']['rounds']:
        require(tree['kind'] in ('tree','terminal_target_token'),'known round kind')
        if tree['kind']!='tree':continue
        tree_count+=1
        selected=tree['selected_tree_indices']
        equal(tree['tree_nodes'],30,'fixed tree nodes')
        equal(tree['budget'],dict(B=30,d=3,K=8,total_tokens=29),'fixed tree budget')
        require(selected and selected[0]==0 and len(set(selected))==len(selected)
            and all(type(i)is int and 0<=i<30 for i in selected),'valid selected path')
        equal(len(selected),len(tree['submitted']),'accepted path count')
        equal(tree['target_cache_before'],base,'semantic cache before')
        base+=len(tree['submitted'])
        equal(tree['target_cache_after'],base,'semantic cache after')
    equal(base,row['result']['target_cache_length'],'final target cache')
    equal(tree_count,row['verify_rounds'],'tree count')
    equal(counts['target_compaction_calls'],tree_count,'actual compaction count')
    equal(counts['target_compaction_selected_tokens'],base-row['result']['prompt_tokens'],'committed token counter')
    equal(counts['target_visual_calls'],1,'one full visual call')
    equal(counts['target_prefill_calls'],1,'one target prefill')
    return dict(request_id=row['request_id'],phase=row['phase'],repeat=row['repeat'],
        tokens=row['num_tokens'],verify_rounds=row['verify_rounds'],stop_reason=row['stop_reason'],
        natural_eos=row['stop_reason']=='eos' and not row['truncated'],
        output_sha256=digest(row['output_token_ids']),full_rounds_sha256=digest(row['result']['rounds']),
        actual_adaptor_calls=1,actual_adaptor_input_tokens=n,input_scans=1,fp32_pooling=True,tf32=False,
        finite_intermediates_verified=None,direct_finite_diagnostic_enabled=False,
        observed_e2e_wall_s=row['e2e_wall_s'],peak_allocated_bytes=row['peak_allocated_bytes'],
        peak_reserved_bytes=row['peak_reserved_bytes'],
        hardware={k:row['hardware'].get(k) for k in ('sharing','energy_attribution','foreign_processes','process_visibility_complete')})


def audit(run, baseline_root):
    status,raw=read(run)
    equal(status['args']['arms'],[ARM],'single streaming arm')
    for key,value in dict(repeats=1,warmup=1,max_new_tokens=512,seed=20261001,request_id=None,
        capture_visual_dir=None,collect_local=False,include_ar=False,resume=False).items():
        equal(status['args'][key],value,'fixed run argument '+key)
    for key,value in dict(streaming_method=METHOD,streaming_heads_per_group=28,streaming_token_tile=48,
        streaming_allow_tf32=False,target_cache_policy='native_full_dynamic_cache',new_kv_policy_installed=False,
        actual_sram_residency=False).items():equal(status[key],value,'streaming status '+key)
    source_verified={}
    for name in ('direct_opt/streaming_adaptor.py','direct_opt/run_streaming_adaptor_layout.py',
                 'direct_opt/compiled_adaptor.py','direct_opt/run_joint.py','vendor/vispec_joint_adapter.py'):
        expected=status['source_sha256'][name]
        equal(sha(Path(__file__).resolve().parent.parent/name),expected,'recorded runtime source '+name)
        source_verified[name]=expected
    validated=[validate_candidate(row)for row in raw]
    measured=[row for row in raw if row['phase']=='measure']
    equal(len(measured),12,'twelve measured requests')
    equal(len(raw)-len(measured),1,'one excluded warmup')
    equal({row['repeat']for row in measured},{0},'one repeat per request')
    require(len({row['request_id']for row in measured})==12,'twelve independent request IDs')
    candidate={row['request_id']:row for row in measured}
    comparisons,reference_rows={},{}
    for name in ('00_native','03_native'):
        directory=baseline_root/name;bs,br=read(directory)
        for key in ('target','draft','request_sha256'):
            equal(bs[key],status[key],'same model/manifest '+key)
        for key in ('seed','max_new_tokens'):
            equal(bs['args'][key],status['args'][key],'same protocol '+key)
        equal(bs['args']['arms'],['memory_native_native'],'native comparison arm')
        refs={r['request_id']:r for r in br if r['phase']=='measure'}
        equal(len(refs),12,'native request coverage')
        equal(set(refs),set(candidate),'complete matched requests')
        reference_rows[name]=refs
        pairs=[]
        for rid,c in sorted(candidate.items()):
            b=refs[rid]
            for key in ('adapter_version','joint_version','author_revision'):
                equal(c['result'][key],b['result'][key],'generation source identity '+key)
            item=compare(c,b)
            item.update(request_id=rid,baseline_tokens=b['num_tokens'],candidate_tokens=c['num_tokens'],
                baseline_verify_rounds=b['verify_rounds'],candidate_verify_rounds=c['verify_rounds'],
                length_ratio=c['num_tokens']/b['num_tokens'],length_at_most_110_percent=within_110(c['num_tokens'],b['num_tokens']),
                first_different_output_token=next((i for i,(x,y)in enumerate(zip(b['output_token_ids'],c['output_token_ids']))if x!=y),
                    min(b['num_tokens'],c['num_tokens'])if b['output_token_ids']!=c['output_token_ids']else None),
                factual_review_required=c['output_token_ids']!=b['output_token_ids'])
            item['input_metadata_equal']=True
            item['input_tensor_byte_equality_verified']=False
            item['changed_round_fields']=sorted({k for r in item['round_differences']for k in r['changed']})
            pairs.append(item)
        comparisons[name]=dict(source=str(directory.resolve()),results_sha256=sha(directory/'results.jsonl'),pairs=pairs,
            same_outputs=sum(p['same_output']for p in pairs),same_full_rounds=sum(p['same_full_rounds']for p in pairs),
            common_counts_equal=sum(p['common_forward_counts_equal']for p in pairs),
            natural_eos_pairs=sum(p['both_natural_eos']for p in pairs),
            lengths_at_most_110_percent=sum(p['length_at_most_110_percent']for p in pairs))
    references_consistent=all(reference_rows['00_native'][rid]['output_token_ids']==reference_rows['03_native'][rid]['output_token_ids']for rid in candidate)
    factual=[]
    for rid,c in sorted(candidate.items()):
        if any(c['output_token_ids']!=reference_rows[name][rid]['output_token_ids']for name in reference_rows):
            factual.append(dict(request_id=rid,candidate_text=c['text'],candidate_output_ids=c['output_token_ids'],
                baseline_text={name:refs[rid]['text']for name,refs in reference_rows.items()},
                baseline_output_ids={name:refs[rid]['output_token_ids']for name,refs in reference_rows.items()},
                status='Needs reference/video fact review; omission is not automatically a contradiction'))
    logs={}
    for name,path in [('candidate',run.with_name(run.name+'.launch.log')),
                      ('00_native',baseline_root/'00_native.log'),('03_native',baseline_root/'03_native.log')]:
        if path.exists():
            text=path.read_text(errors='replace')
            logs[name]=dict(path=str(path.resolve()),sha256=sha(path),
                h264_mmco_unref_short_failure_count=text.count('mmco: unref short failure'),
                traceback_count=text.count('Traceback (most recent call last)'),
                cuda_error_count=text.count('CUDA error'))
        else:logs[name]=None
    exit_path=run.with_name(run.name+'.exit.json')
    exit_receipt=json.loads(exit_path.read_text())if exit_path.exists()else None
    if exit_receipt is not None:equal(exit_receipt['returncode'],0,'supervisor terminal exit')
    return dict(run=str(run.resolve()),status_sha256=sha(run/'status.json'),results_sha256=sha(run/'results.jsonl'),
        source_sha256=status['source_sha256'],verified_runtime_sources=source_verified,
        validated_cells=validated,comparisons=comparisons,log_evidence=logs,exit_receipt=exit_receipt,
        measured_requests=12,excluded_warmup=1,independent_requests=12,repeats_per_request=1,
        all_native_reference_outputs_consistent=references_consistent,
        candidate_tokens=sum(r['num_tokens']for r in measured),candidate_verify_rounds=sum(r['verify_rounds']for r in measured),
        candidate_natural_eos=sum(r['stop_reason']=='eos' and not r['truncated']for r in measured),
        factual_review_required=factual,automatic_factual_quality_pass=None,
        full_request_finite_intermediate_check=False,previous_module_finite_evidence_only=True,
        input_tensor_byte_equality_verified=False,
        input_evidence='All saved input shapes, video grid, timestamps and manifest agree; pixel/input tensor content hashes were not saved, so byte equality cannot be established from these records.',
        timing_comparison=False,actual_sram_measurement=False,actual_dram_measurement=False)


def markdown(report):
    lines=['# G28/B48 12题独立审计','',
        f"共12个独立请求，每题1次测量；另1次暖机排除。自然EOS {report['candidate_natural_eos']}/12，合计{report['candidate_tokens']} token、{report['candidate_verify_rounds']}轮验证。",'',
        '| 对照 | 逐token输出相同 | 完整rounds相同 | 共同调用计数相同 | 双方自然EOS | 长度≤110% |',
        '|---|---:|---:|---:|---:|---:|']
    for name,c in report['comparisons'].items():
        lines.append(f"| {name} | {c['same_outputs']}/12 | {c['same_full_rounds']}/12 | {c['common_counts_equal']}/12 | {c['natural_eos_pairs']}/12 | {c['lengths_at_most_110_percent']}/12 |")
    lines+=['','## 逐题（对native 00）','',
        '| 请求 | token原版→候选 | 验证轮次原版→候选 | 输出相同 | 不同round数 | 长度≤110% | 需要事实审查 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in report['comparisons']['00_native']['pairs']:
        lines.append(f"| {p['request_id'].removeprefix('direct_quality_vdc_v_')} | {p['baseline_tokens']}→{p['candidate_tokens']} | {p['baseline_verify_rounds']}→{p['candidate_verify_rounds']} | {p['same_output']} | {p['changed_round_count']} | {p['length_at_most_110_percent']} | {p['factual_review_required']} |")
    changed=[p['request_id']for p in report['comparisons']['00_native']['pairs']if
             any(k!='selected_tree_indices'for k in p['changed_round_fields'])]
    lines+=['','完整round差异已保留。以下请求还出现每轮提交/接受时点差异，不能把全部差异概括为selected索引重编号：',
            *['- '+rid for rid in changed]]
    lines+=['','## 质量边界','']
    if report['factual_review_required']:
        lines.append('以下文字改变，需按现有参考关键事实和视频输入审查；本机械审计不自动判错，也不自动通过：')
        lines += ['- '+r['request_id']for r in report['factual_review_required']]
    else:
        lines.append('这12题相对两份原版记录均保留逐token输出，因此没有本轮新增回答文字需要事实差异审查。这只支持本批输出一致，不证明更广数据质量，也不把原版本身的事实错误判为正确。')
    lines+=['','## 运行和计量边界','',
        '- 候选固定G28/B48、FP32评分/在线归约、FP16 V/O，TF32关闭；全视觉深度和全部token保留；普通DynamicCache，无ring/量化/新解码策略。',
        '- 实际hooks逐题核1次适配器调用、对应真实N、一遍X扫描与ceil(N/48)个tile；普通旧imadpt_calls不被当作真实调用数。',
        '- 两份native对照与候选的模型路径、请求manifest、输入形状/采样时间、seed20261001和max512均相同。输出、完整rounds、共同计数分别比较，不能互相替代。',
        '- 新候选与native00/03日志均有一次H.264 mmco: unref short failure；候选exit0且没有CUDA error/Traceback。现有输入shapes/grid/timestamps全部一致，但没有保存pixel/input tensor内容hash，所以不能声称输入tensor逐位相同。warning不被误记为CUDA失败，内容一致性未核验这一限制也不省略。',
        '- 长度≤110%是已要求的描述性限制，不是事实质量分数；任何改变的文字仍待事实审查。',
        '- 完整请求关闭finite与workspace诊断，未直接证明所有中间张量finite。之前模块有限值检查是独立证据。',
        '- 这是不同批次记录的数值/轨迹比较，不给跨批次加速、能耗或吞吐结论。每条原始峰值/时间/foreign PID保留在JSON，不等同SRAM或DRAM。',
        '- 原pilot审计脚本和历史输出均未修改；所有逐轮差异及待审文本见JSON。','']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=BASE/'streaming_adaptor_vdc12_r1')
    p.add_argument('--baseline-root',type=Path,default=BASE/'throughput_native_delta_abba_r1')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    report=audit(args.run,args.baseline_root)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'STREAMING_QUALITY12_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (args.out/'STREAMING_QUALITY12_AUDIT.md').write_text(markdown(report))
    print(json.dumps({name:{k:v for k,v in c.items()if k not in ('source','results_sha256','pairs')}for name,c in report['comparisons'].items()},indent=2))


if __name__=='__main__':main()
