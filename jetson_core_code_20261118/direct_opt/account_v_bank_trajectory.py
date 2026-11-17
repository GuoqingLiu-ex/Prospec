"""Offline V-bank trajectory-dependent major matrices and selected logical IO.

No model, checkpoint, GPU, subprocess or remote action. Uses the frozen counter's
source reconstruction; distinguishes unrecorded Draft K shapes from actual Target
cache receipts. Software ledger calls/bytes are not physical DRAM transactions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from account_request_macs import count_request, require, equal, EXCLUDED
from account_exact_layout_macs import load_configs
import audit_v_bank_generation as v_audit
import audit_original_grouped_abba as original_audit

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'coordination/direct_opt_20261002'
L,H,HK,D=28,28,4,128


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest_value(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def add(out,name,calls,read,write):
    e=out.setdefault(name,dict(calls=0,read_bytes=0,write_bytes=0,read_plus_write_bytes=0))
    for k,v in dict(calls=calls,read_bytes=read,write_bytes=write,read_plus_write_bytes=read+write).items():e[k]+=v


def known_io(row,mode):
    trees=[t for t in row['result']['rounds'] if t['kind']=='tree']
    P=row['result']['prompt_tokens'];M=row['result']['visual_context']['original_count']
    out={}
    if mode!='grouped_control':
        recorded=row['adaptor_preparation']['v_bank']['logical_io']
        rebuilt=v_audit.logical_io_from_rounds(P,M,mode,trees)
        equal(recorded,rebuilt,'recorded policy ledger vs actual trajectory')
        out=json.loads(json.dumps(rebuilt))
    else:
        # Frozen DynamicLayer.update torch.cat and bridge._compact index_select.
        # This excludes common native prefill KV construction and index metadata.
        for t in trees:
            S=t['target_cache_before']+t['tree_nodes'];C=t['target_cache_after']
            for name,size in [('key_cat',S),('full_value_cat',S),('key_compact',C),('full_value_compact',C)]:
                n=L*HK*size*D*2;add(out,name,L,n,n)
            add(out,'value_operand_cast',L,L*HK*S*D*2,L*HK*S*D*4)
    common={}
    for t in trees:
        T=t['tree_nodes'];S=t['target_cache_before']+T
        add(common,'key_fp16_to_fp32_cast',L,L*HK*S*D*2,L*HK*S*D*4)
        add(common,'query_fp32_pack',L,L*H*T*D*4,L*H*T*D*4)
        add(common,'preserved_unused_probability_cast',L,L*H*T*S*4,L*H*T*S*2)
    maintenance=[k for k in out if k.endswith('_cat') or k.endswith('_compact')]
    operand='value_operand_cast' if mode=='grouped_control' else 'value_operand_fill'
    extras=[k for k in out if k not in maintenance and k!=operand]
    def selected(keys):return {field:sum(out[k][field] for k in keys) for field in ('calls','read_bytes','write_bytes','read_plus_write_bytes')}
    return dict(ledger=out,kv_maintenance=selected(maintenance),value_fp32_operand=out[operand],
        other_recorded_policy_payload=selected(extras),common_selected_attention_copies=common,
        scope='Selected ideal payload read/write, once per named source operation. Not complete logical IO, observed GPU activities, allocator storage or DRAM. Guard/index ledger calls aggregate several operations.')


def count_row(row,mode,configs):
    target,draft,vision,prov=configs
    require(row['phase']=='measure','formal rows only')
    equal(row['result']['author_revision'],prov['draft_author_revision'],'author revision')
    equal(row['result']['joint_version'],'vispec_joint_m_budget_v1','joint version')
    equal(row['result']['adapter_version'],'hf517_vispec_tree_v1','bridge version')
    equal(row['scheme']['adaptor'],'compiled','fixed compiled adaptor')
    coverage=v_audit.grouped_coverage(row,'grouped') if mode=='grouped_control' else v_audit.candidate_coverage(row,mode)
    require(coverage['passed'],'full per-mode math/cache coverage')
    require(v_audit.count_contract(row,mode)['passed'],'complete frozen counter contract')
    # Dispatch alias changes no field consumed by actual shape reconstruction.
    counted=count_request(dict(row,arm='deployment_compiled'),target,draft,vision)
    counted.pop('observed_e2e_wall_s');counted.pop('output_token_ids')
    components=counted['components_macs'];pairs=counted['query_reconstruction']
    trees=[t for t in row['result']['rounds'] if t['kind']=='tree']
    actual_products=sum(t['tree_nodes']*(t['target_cache_before']+t['tree_nodes']) for t in trees)
    equal(pairs['target_sum_query_times_key'],row['result']['prompt_tokens']**2+actual_products,'Target actual widths')
    return dict(request_id=row['request_id'],repeat=row['repeat'],mode=mode,num_tokens=row['num_tokens'],
        output_sha256=digest_value(row['output_token_ids']),rounds_sha256=digest_value(row['result']['rounds']),
        forward_counts=row['result']['forward_counts'],prompt_tokens=row['result']['prompt_tokens'],
        visual_tokens=row['result']['visual_context']['original_count'],major=counted,
        actual_target_tree=dict(query_key_products_per_layer=actual_products,attention_macs=2*H*D*L*actual_products,
            layer_attention_calls=L*len(trees),qk_bmm_calls=L*len(trees),pv_bmm_calls=L*len(trees)),
        recorded_draft_key_shapes=None,recorded_shape_draft_attention_macs=None,
        source_reconstructed_draft_attention_macs=components['draft_attention'],
        major_macs_excluding_unrecorded_draft_attention=counted['total_major_macs']-components['draft_attention'],
        io=known_io(row,mode))


def sum_ledger(ledgers):
    out={}
    for ledger in ledgers:
        for k,v in ledger.items():add(out,k,v['calls'],v['read_bytes'],v['write_bytes'])
    return out


def totals(rows):
    comps={k:sum(r['major']['components_macs'][k] for r in rows) for k in rows[0]['major']['components_macs']}
    result=dict(formal_requests=len(rows),unique_videos=len({r['request_id'] for r in rows}),
        output_tokens=sum(r['num_tokens'] for r in rows),verify_rounds=sum(r['major']['verify_rounds'] for r in rows),
        components_macs=comps,total_major_macs=sum(comps.values()),
        major_macs_excluding_unrecorded_draft_attention=sum(comps.values())-comps['draft_attention'],
        actual_target_tree={k:sum(r['actual_target_tree'][k] for r in rows) for k in rows[0]['actual_target_tree']},
        recorded_draft_key_shapes=None,recorded_shape_draft_attention_macs=None,
        source_reconstructed_draft_attention_macs=comps['draft_attention'],
        io_ledger=sum_ledger([r['io']['ledger'] for r in rows]),
        common_selected_attention_copies=sum_ledger([r['io']['common_selected_attention_copies'] for r in rows]))
    for cat in ('kv_maintenance','value_fp32_operand','other_recorded_policy_payload'):
        result[cat]={k:sum(r['io'][cat][k] for r in rows) for k in rows[0]['io'][cat]}
    return result


def pairs(baseline,candidate):
    out=[]
    for key,b in baseline.items():
        c=candidate[key];br=b['result']['rounds'];cr=c['result']['rounds']
        require(len(br)==len(cr),'same-output comparison requires complete round pairing, no synthetic padding')
        changed=[]
        for index,(x,y) in enumerate(zip(br,cr)):
            fields={k:dict(control=x.get(k),candidate=y.get(k)) for k in x.keys()|y.keys() if x.get(k)!=y.get(k)}
            if fields:changed.append(dict(round_index_zero_based=index,fields=fields))
        out.append(dict(request_id=key[0],repeat=key[1],output_ids_same=b['output_token_ids']==c['output_token_ids'],
            input_metadata_same=all(b[k]==c[k] for k in ('input_shapes','video_grid_thw','timestamps')),
            common_counts_same=v_audit.common_counts(b)==v_audit.common_counts(c),full_rounds_same=br==cr,
            changed_rounds=changed))
    return out


def original_batch(path,configs):
    records={};metadata={};source_maps=[]
    for name,role in [('00_original','original'),('01_grouped','grouped'),('02_grouped','grouped'),('03_original','original')]:
        data=original_audit.read_run(path/name,role,[6360]);require(data['mechanical_pass'],'original same-package run failed mechanical review')
        source_maps.append(data['status']['source_sha256']);rows=[]
        target,draft,vision,prov=configs
        for row in data['rows']:
            if row['phase']!='measure':continue
            require(row['result']['author_revision']==prov['draft_author_revision'],'original author mismatch')
            alias='deployment_native' if role=='original' else 'deployment_compiled'
            c=count_request(dict(row,arm=alias),target,draft,vision);c.pop('observed_e2e_wall_s')
            rows.append(c)
        records[name]=rows
        metadata[name]=dict(raw_hashes=data['hashes'],source_hashes_valid=all(data['source_checks'].values()),
            role=role,formal_requests=len(rows),outputs=sum(r['num_tokens'] for r in rows),
            components_macs={k:sum(r['components_macs'][k] for r in rows) for k in rows[0]['components_macs']},
            total_major_macs=sum(r['total_major_macs'] for r in rows),original_full_source_count=len(source_maps[-1]))
    require(all(x==source_maps[0] for x in source_maps),'original/grouped reference is not one frozen package')
    pair=[]
    for a,b in [('00_original','01_grouped'),('03_original','02_grouped')]:
        aa={(r['request_id'],r['repeat']):r for r in records[a]};bb={(r['request_id'],r['repeat']):r for r in records[b]}
        equal(set(aa),set(bb),'original ABBA requests')
        pair.append(dict(original=a,grouped=b,pairs=len(aa),same_output=sum(aa[k]['output_token_ids']==bb[k]['output_token_ids'] for k in aa),
            original_major_macs=metadata[a]['total_major_macs'],grouped_major_macs=metadata[b]['total_major_macs'],
            reduction_pct=100*(1-metadata[b]['total_major_macs']/metadata[a]['total_major_macs']),
            component_delta={k:metadata[b]['components_macs'][k]-v for k,v in metadata[a]['components_macs'].items()}))
    return dict(scope='Separate previously completed original/shared+compiled+grouped ABBA; no V-bank arm in this package and no timing comparison to current batch.',
        source_package_all_equal=True,runs=metadata,pairs=pair)


def run(directory,configdir,visionpath,original=None):
    configs=load_configs(configdir,visionpath);raw={};counted={};summary={};provenance={}
    for name,mode in [('control','grouped_control'),('fp16_partition','fp16_partition'),('int8_visual','int8_visual')]:
        data=v_audit.read_run(directory/name,mode,known_graphics=[6360])
        require(data['review']['mechanical_execution_pass'],name+' complete execution/source validation failed')
        require(len(data['requests'])==12,'requires full12 protocol')
        status=data['status'];prov=configs[3]
        for key in ('target','draft'):equal(status[key],str(Path(prov['sources'][key+'_config']).parent),'checkpoint config source '+key)
        equal(status['transformers'],'5.17.0','Target runtime');equal(status['torch'],'2.8.0+cu128','Torch runtime')
        raw[name]={(r['request_id'],r['repeat']):r for r in data['rows'] if r['phase']=='measure'}
        counted[name]=[count_row(r,mode,configs) for r in raw[name].values()]
        summary[name]=totals(counted[name]);provenance[name]=dict(raw_hashes=data['review']['raw_hashes'],
            terminal_sha256=data['review']['terminal_sha256'],source_sha256=status['source_sha256'],excluded_warmup_rows=1,
            physical_gpu=status['args']['physical_gpu'],foreign_pids=data['review']['all_observed_foreign_pids'])
    equal(set(raw['control']),set(raw['fp16_partition']),'V16 request keys');equal(set(raw['control']),set(raw['int8_visual']),'V8 request keys')
    require(provenance['control']['source_sha256']==provenance['fp16_partition']['source_sha256']==provenance['int8_visual']['source_sha256'],'current three-arm frozen source packages differ')
    comparisons={n:pairs(raw['control'],raw[n]) for n in ('fp16_partition','int8_visual')}
    b,c=summary['fp16_partition'],summary['int8_visual']
    counterfactual=[]
    for r in raw['int8_visual'].values():
        P=r['result']['prompt_tokens'];M=r['result']['visual_context']['original_count']
        counterfactual.append(v_audit.logical_io_from_rounds(P,M,'fp16_partition',[t for t in r['result']['rounds'] if t['kind']=='tree']))
    cf=sum_ledger(counterfactual)
    decompose={k:{field:dict(actual_v8_minus_v16=c['io_ledger'][k][field]-v[field],
        v8_storage_on_v8_trajectory=c['io_ledger'][k][field]-cf[k][field],
        changed_trajectory_at_v16_storage=cf[k][field]-v[field]) for field in ('calls','read_bytes','write_bytes','read_plus_write_bytes')}
        for k,v in b['io_ledger'].items()}
    return dict(schema='v_bank_trajectory_major_matrix_and_selected_logical_io_v1',current_dataset=str(directory.resolve()),
        formulas=dict(target_prefill_attention_macs='2*28*3584*P*P',target_tree_attention_macs='sum_round 2*28*3584*T*(base+T)',
            draft_attention_source_reconstruction='2*3584*sum(syncQ*stableK + sum_j 8*(stableK+8*(j+1)))',
            kv_fp16_bytes_per_token_per_layer=4*128*2,v8_visual_bytes_per_token_per_layer=4*128+4*4*4,
            value_fp32_operand_bytes_per_token_per_layer=4*128*4),
        source_and_raw_provenance=provenance,
        config_hashes={str(p):digest(p) for p in [configdir/'target_config.json',configdir/'draft_config.json',configdir/'config_provenance.json',visionpath]},
        accounting_source_hashes={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('account_request_macs.py'),
            Path(__file__).with_name('account_exact_layout_macs.py'),Path(__file__).with_name('audit_v_bank_generation.py')]},
        totals=summary,requests=counted,comparisons_to_grouped_control=comparisons,
        v8_minus_v16_logical_io_decomposition=decompose,original_same_package_reference=original_batch(original,configs) if original else None,
        claims=dict(recorded_draft_per_forward_key_shapes=None,complete_scalar_flops=None,complete_logical_io=None,
            actual_dram_bytes=None,physical_transfer_count=None,timing_improvement=None,quality_pass=None),
        exclusions=EXCLUDED+['Draft physical cache traffic/shape telemetry is absent',
            'logical ledgers exclude matrix operand rereads/BLAS packing, softmax/score traffic, flags/route lookup repeats and most metadata allocations/scalar sync payloads',
            'common initial native prefill KV construction and temporary allocator behavior',
            'setup matrix folding is separate: each compiled process 2*3584^2+2*3584=25697280 MAC'],
        precision='Target/Draft body FP16; LM heads FP32; current grouped Target tree QK/PV FP32; V8 only storage plus FP32 elementwise reconstruction, not INT8 matrix multiplication')


def markdown(r):
    t=r['totals'];lines=['# V-only 三臂：逐轮计算与选定逻辑 IO','',
        '24 次正式请求 = 12 视频 × 2 重复；三臂各一条预热已排除。全量按各自真实轮次重建，不用输出长度替代计算量。',
        'grouped_control 是 shared embedding + compiled adaptor + grouped FP32 attention 对照，**不是原版 ViSpec**。本报告没有测速或 DRAM 实测。','',
        '## 主要矩阵 MAC','',
        '1 MAC 为一次乘加；TMAC=10¹² MAC。总计含按已审源码重建的 Draft attention，原始记录未逐调用保存 Draft K 形状；该实测形状字段留空。',
        '| 臂 | tokens / verify | 总主要 TMAC | Target attention TMAC | Draft attention GMAC（源码重建） |',
        '|---|---:|---:|---:|---:|']
    for name,a in t.items():lines.append(f"| {name} | {a['output_tokens']} / {a['verify_rounds']} | {a['total_major_macs']/1e12:.12f} | {a['components_macs']['target_attention']/1e12:.12f} | {a['components_macs']['draft_attention']/1e9:.9f} |")
    lines+=['','| 矩阵组件 | control = V16 合计 TMAC | V8−control MAC |','|---|---:|---:|']
    for name,value in t['control']['components_macs'].items():
        lines.append(f"| {name} | {value/1e12:.12f} | {t['int8_visual']['components_macs'][name]-value:,} |")
    delta=t['int8_visual']['total_major_macs']-t['control']['total_major_macs']
    lines+=['',f"V16 与 control 的矩阵账一致。V8 多 {delta:,} MAC（{100*delta/t['control']['total_major_macs']:.10f}%）；变化来自每轮已提交前缀不同，Target attention 和 Draft attention 的矩阵长宽随之改变。线性层、LM head、视觉塔、适配器与视觉融合的矩阵 MAC 一致。",'',
        '| V8 变化题目 | repeat | 差异轮数 | 影响 MAC 的提交边界变化轮（从 0 起） |','|---|---:|---:|---|']
    for p in r['comparisons_to_grouped_control']['int8_visual']:
        changes=[x['round_index_zero_based'] for x in p['changed_rounds'] if any(k in x['fields'] for k in ('submitted','target_cache_before','target_cache_after'))]
        if changes:lines.append(f"| {p['request_id']} | {p['repeat']} | {len(p['changed_rounds'])} | {changes} |")
    lines+=['','仅 selected_tree_indices 不同且提交数/前缀长度不变，不改变 dense 矩阵形状账；仍不能称完整执行轨迹一致。所有组的输入 shape/grid/timestamps、输出和公共计数核对见 JSON，不能替代输入像素内容哈希。','',
        '## 选定 KV 逻辑读写','',
        'GiB=2³⁰ bytes；每格为读+写。KV 维护覆盖 verify cat 与 accepted index_select；V32 operand 覆盖逐层填充/转换。候选附加项含一次 route/prefill pack、每轮 guard 与共享索引账；不是全请求 IO。',
        '| 臂 | KV 维护 GiB / 源算子调用 | V32 operand GiB / 调用 | 其它已记录 policy GiB |','|---|---:|---:|---:|']
    for name,a in t.items():
        k,v,e=[a[x] for x in ('kv_maintenance','value_fp32_operand','other_recorded_policy_payload')]
        extra='无 V-bank ledger；原生 metadata 未计' if name=='control' else f"{e['read_plus_write_bytes']/2**30:.9f}"
        lines.append(f"| {name} | {k['read_plus_write_bytes']/2**30:.9f} / {k['calls']} | {v['read_plus_write_bytes']/2**30:.9f} / {v['calls']} | {extra} |")
    lines+=['','调用数是上述源函数/账目次数，不是 kernel、CUDA memcpy、DRAM request/sector/burst。guard 等账目一次可包含多个操作；各类次数不合成所谓物理搬运次数。',
        '本三臂没有减少上述维护或 V32 生成的源算子调用次数：分区降低每次搬运的 V payload，V8 再降低填充输入的表示大小。control 的候选专属 ledger 为空不等于原生 metadata 没有工作。',
        'V8 填充 read_bytes 按每个 codes/scale/text payload 条目读一次计理想大小；同一 scale 被多个 lane 使用、route 查表和实际缓存行为不由本账推断。',
        'K32 cast、Q pack、仍保留的 FP16 概率 cast 单列在 JSON，不与 KV 维护混为一项。矩阵实际读取、softmax/score、scale 重读、路由/错误标志和标量同步等未计，不能把本表当净 DRAM。','',
        '## V8 与 V16：存储变换和轨迹变化拆分','',
        '以下为明确反事实：在 V8 实际轮次上计算 V16 表示的相同 payload 公式，然后分别相减；不是额外执行或测量。',
        '| 项目 | 实际 V8−V16 bytes | 固定 V8 轨迹的表示变化 bytes | 固定 V16 表示的轨迹变化 bytes |','|---|---:|---:|---:|']
    for k,v in r['v8_minus_v16_logical_io_decomposition'].items():
        d=v['read_plus_write_bytes']
        if any(d.values()):lines.append(f"| {k} | {d['actual_v8_minus_v16']:,} | {d['v8_storage_on_v8_trajectory']:,} | {d['changed_trajectory_at_v16_storage']:,} |")
    orig=r['original_same_package_reference']
    if orig:
        lines+=['','## 独立的既有原版同包 ABBA','',
            '只用已完成 ordinary_original_grouped_abba_gpu3_r1 的 234 文件同包结果；此批没有 V-only 臂。不把它拼成当前三臂的同批原版比较，也不跨批比时延。',
            '| 原版 / grouped 进程 | 请求对 / 同输出 | 原版 TMAC | grouped TMAC | MAC 减少% |','|---|---:|---:|---:|---:|']
        for p in orig['pairs']:lines.append(f"| {p['original']} / {p['grouped']} | {p['pairs']} / {p['same_output']} | {p['original_major_macs']/1e12:.9f} | {p['grouped_major_macs']/1e12:.9f} | {p['reduction_pct']:.9f} |")
        lines+=['','每24对的适配器重排减少1,188,828,741,632 MAC，另有实际轨迹导致的 Target attention −12,042,240、Draft attention −344,064 MAC。共享 embedding 和 grouped GQA 本身不减少本账中的语言矩阵乘加。']
    lines+=['','## 保留边界','',
        '- 全部源路径、配置/原始文件 SHA、逐请求矩阵组件、实际 Target key width 积和逻辑 IO 分项均在 JSON。',
        '- Draft K 长度没有逐调用原始遥测；依据固定树预算和已审作者缓存规则重建，并校验 forward/input/decoder 行数。实际 Draft cache 流量留空。',
        '- V8 不是 INT8 matmul：QK/softmax/PV 仍为 FP32。量化/反量化标量算术未并入 MAC，完整标量 FLOPs 留空。',
        '- 同 MAC 不代表同硬件成本。不同精度、形状、访存、同步及 launch 都影响时延；共享 GPU 无干净速度/能耗结论。',
        '- 没有新 GPU 操作，也没有修改 frozen execution sources 或原始记录。','']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--configs',type=Path,default=BASE/'full_request_mac_accounting')
    p.add_argument('--vision-config',type=Path,default=ROOT/'combined_visual_experiment/vision_config.json')
    p.add_argument('--original-abba',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);require(not a.out.exists(),'fresh accounting directory required')
    result=run(a.run,a.configs,a.vision_config,a.original_abba)
    a.out.mkdir(parents=True);(a.out/'accounting.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (a.out/'REPORT.md').write_text(markdown(result))
    print(json.dumps({k:v['total_major_macs'] for k,v in result['totals'].items()}))
    return result


if __name__=='__main__':main()
