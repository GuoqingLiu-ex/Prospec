"""Offline major-matrix MAC accounting for explicitly audited exact KV layouts.

Raw records are never edited. Active-width equality is established before an
arm-only dispatch alias reaches the existing request counter. No GPU/model load.
This is a fixed experimental-protocol report, not a generic unknown-arm counter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from account_request_macs import count_request, equal, require, EXCLUDED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = 'memory_native_native'
DELTA = 'memory_shared_embedding_compiled_ring_active_width_protected_prompt_512_delta_commit'
IMPLICIT = DELTA + '_implicit_layout'
VARIANTS = {DELTA: 'active_width_delta_commit', IMPLICIT: 'active_width_delta_commit_implicit_layout'}
COMMON_SOURCES = ['compute_cost.py', 'training_free/factorized_adaptor.py',
    'vendor/hf517_vispec_adapter.py', 'vendor/vispec_joint_adapter.py',
    'direct_opt/run_joint.py', 'direct_opt/run_memory_layout.py',
    'direct_opt/run_deployed_adaptor.py']
LAYOUT_SOURCES = ['direct_opt/compression.py', 'direct_opt/compiled_adaptor.py',
    'direct_opt/active_ring_kv_cache.py', 'direct_opt/active_ring_runtime.py',
    'direct_opt/ring_cache_runtime.py', 'direct_opt/delta_commit_runtime.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_configs(directory, vision_path):
    target, draft = [json.loads((directory/(name+'_config.json')).read_text()) for name in ('target', 'draft')]
    vision = json.loads(vision_path.read_text())
    provenance = json.loads((directory/'config_provenance.json').read_text())
    for key, value in target['vision_config'].items():
        key = 'in_channels' if key == 'in_chans' else key
        require(key not in vision or vision[key] == value, 'checkpoint/runtime vision mismatch: '+key)
    for config, layers, kv in ((target, 28, 4), (draft, 1, 28)):
        equal((config['hidden_size'], config['num_hidden_layers'], config['num_attention_heads'],
               config['num_key_value_heads'], config['vocab_size']),
              (3584, layers, 28, kv, 152064), 'audited language configuration')
        equal(config['use_sliding_window'], False, 'full language attention')
    equal(vision['depth'], 32, 'full visual depth')
    equal(draft['qkv_bias'], True, 'folding setup bias')
    return target, draft, vision, provenance


def validate_round_paths(row):
    result = row['result']
    trees = [r for r in result['rounds'] if r['kind'] == 'tree']
    for round_ in trees:
        selected, nodes = round_['selected_tree_indices'], round_['tree_nodes']
        require(selected and selected[0] == 0 and len(set(selected)) == len(selected), 'invalid selected path root/duplicates')
        require(all(type(i) is int and 0 <= i < nodes for i in selected), 'selected tree index out of bounds')
        equal(len(selected), len(round_['submitted']), 'selected path vs submitted tokens')
        equal(round_['visual_m'], 1.0, 'uncompressed visual context')
    return trees


def validate_layout(row, target_layers=28):
    """Prove physical attention widths equal the old counter's logical widths."""
    arm, prep = row['arm'], row['adaptor_preparation']
    require(arm in {NATIVE, DELTA, IMPLICIT}, 'unreviewed exact-layout arm')
    mode = 'native' if arm == NATIVE else 'compiled'
    expected = dict(depth='fixed0', adaptor=mode, compression=None, ratio=1.0)
    if 'cache_layout' in row['scheme']:
        require(arm != NATIVE, 'native cannot have a cache-layout override')
        expected['cache_layout'] = 'implicit' if arm == IMPLICIT else 'delta'
    equal(row['scheme'], expected, 'exact supported scheme')
    equal(row['depth_mode'], 'fixed0', 'depth mode')
    equal(row['visual_keep_ratio'], 1.0, 'visual keep ratio')
    equal(row['stats']['mode'], 'native', 'visual execution mode')
    equal(row['stats']['calls'], [], 'no visual skip/routing calls')
    equal(prep['mode'], mode, 'installed adaptor mode')
    equal(prep['other_scheme_runtimes_constructed'], False, 'single adaptor installation')
    equal(prep['checkpoint_modified'], False, 'frozen checkpoint')
    require(prep.get('fusion_mode') in (None, 'native'), 'unreviewed visual fusion')
    equal(row['result']['precision'], dict(target='float16', output_head='float32',
        target_sdpa='math', draft='float16', draft_sdpa='auto'), 'audited attention and body precision')
    trees = validate_round_paths(row)
    if arm == NATIVE:
        equal(prep['memory_layout'], 'native', 'native memory layout')
        require(not prep.get('ring_kv') and not prep.get('ring_variant'), 'native cache receipt unexpected')
        equal(prep['preparation'], None, 'native no folded setup')
        return dict(kind='native_dynamic', tree_rounds=len(trees), active_width_matches_logical=True)
    equal(prep['memory_layout'], 'shared_embedding', 'compiled/shared installation')
    for name in ('installed', 'bitwise_equal', 'same_parameter', 'same_storage'):
        equal(prep['memory_installation'][name], True, 'shared embedding '+name)
    for name, value in dict(prepare_count=1, precision='model', pooling='bmm', value_projection='bmm').items():
        equal(prep['preparation'][name], value, 'compiled adaptor '+name)
    ring = prep['ring_kv']
    variant = VARIANTS[arm]
    equal(prep['ring_variant'], variant, 'preparation ring variant')
    for name, value in dict(variant=variant, mode='protected_prompt', recent_capacity=512, tree_capacity=30,
        finalized=True, failed=False, prefill_calls=1, completed_verifications=len(trees),
        target_history_eviction=False, attention_backend_changed=False, kv_quantized=False,
        draft_cache_policy='unchanged_full_history', attention_layout='filled_history_plus_current_tree',
        zero_history_packing=True, commit_method='skip_inplace_rows_and_fuse_moved_rows').items():
        equal(ring[name], value, 'ring '+name)
    if arm == IMPLICIT:
        for name, value in dict(metadata_tensor_storage_bytes=0, metadata_update_write_bytes=0,
            caller_attention_mask_reused=True, mask_rematerialization_bytes=0,
            per_layer_metadata_gpu_operations=0, round_tree_path_mrope_validation_retained=True).items():
            equal(ring[name], value, 'implicit '+name)
    p = row['result']['prompt_tokens']
    equal(ring['prompt_tokens'], p, 'ring prompt')
    equal(ring['bytes_per_token_all_target_layers'], 57344, 'Target KV payload shape')
    equal(len(ring['rounds']), len(trees), 'ring tree receipt count')
    equal(ring['prefill_cache']['semantic_length'], p, 'prefill semantic length')
    equal(ring['prefill_cache']['layer_count'], target_layers, 'prefill layer count')
    base, key_sum, product_sum, queries = p, 0, 0, 0
    for i, (trace, r) in enumerate(zip(trees, ring['rounds'])):
        nodes, accepted = trace['tree_nodes'], len(trace['submitted'])
        width, after = base+nodes, base+accepted
        require(after-p <= 512, 'capacity overflow/eviction would change this accounting')
        for key, value in dict(logical_base=base, nodes=nodes, active_attention_key_tokens=width,
            committed_tokens=accepted, semantic_length_after=after,
            semantic_rope_base=trace['semantic_rope_base'], reserved_physical_capacity=p+512+30,
            excluded_padding_key_tokens=p+512+30-width).items():
            equal(r[key], value, 'ring round '+key)
        equal(r['all_layer_commit_reports_equal'], True, 'all-layer commit equality')
        require(r['layer_commit_reports'], 'missing layer commit report')
        for commit in r['layer_commit_reports']:
            for key, value in dict(accepted_tokens=accepted, stored_tokens=accepted,
                evicted_recent_tokens=0, next_position=after, recent_count=after-p).items():
                equal(commit[key], value, 'commit '+key)
        for key, length, active, commits, next_pos in (
                ('cache_before_commit', width, nodes, i, base),
                ('cache_after_commit', after, 0, i+1, after)):
            cache = r[key]
            equal(cache['layer_count'], target_layers, key+' layer count')
            equal(cache['semantic_length'], length, key+' length')
            equal(cache['active_attention_key_tokens'], length, key+' active width')
            layer = cache['first_layer']
            equal(layer['next_position'], next_pos, key+' absolute position')
            equal(layer['active_tree_nodes'], active, key+' scratch nodes')
            equal(layer['operation_counts'], dict(prefill=1, append_tree=i+1, commit=commits), key+' calls')
        base = after
        queries += nodes
        key_sum += width
        product_sum += nodes*width
    equal(base, row['result']['target_cache_length'], 'final committed cache length')
    equal(ring['final_cache']['semantic_length'], base, 'final ring semantic length')
    equal(len(ring['final_layer_metadata']), target_layers, 'all final layer receipts')
    for layer in ring['final_layer_metadata']:
        for key, value in dict(prefix_length=p, next_position=base, active_tree_nodes=0,
            recent_count=base-p, scratch_start=base, rotated_keys_modified=False,
            active_attention_tokens=base).items():
            equal(layer[key], value, 'final layer '+key)
        equal(layer['operation_counts'], dict(prefill=1, append_tree=len(trees), commit=len(trees)), 'final layer calls')
        shape = layer['attention_shape_totals']
        for key, value in dict(verification_calls=len(trees), query_tokens=queries,
            active_key_tokens=key_sum, active_query_key_pairs=product_sum).items():
            equal(shape[key], value, 'final layer attention '+key)
    return dict(kind=variant, tree_rounds=len(trees), active_width_matches_logical=True,
        actual_target_tree_query_tokens=queries, actual_target_tree_query_key_pairs=product_sum,
        no_evictions=True, all_layer_lifecycle_verified=True)


def validate_status(status, raw, provenance):
    equal(status['status'], 'completed', 'run completed')
    equal(status['completed_cells'], status['total_cells'], 'all planned cells completed')
    equal(len(raw), status['total_cells'], 'all raw rows present')
    require({r['phase'] for r in raw} <= {'warmup', 'measure'}, 'unreviewed phase')
    require(len({(r['arm'], r['request_id'], r['repeat'], r['phase']) for r in raw}) == len(raw), 'duplicate raw cell')
    args = status['args']
    for key, value in dict(max_new_tokens=512, include_ar=False, collect_local=False,
        capture_visual_dir=None, resume=False).items():
        equal(args[key], value, 'run argument '+key)
    equal(status['methodology'], 'independent_deployment_exact_embedding_layout', 'audited deployment loader')
    equal(status['transformers'], '5.17.0', 'audited Target implementation')
    equal(status['torch'], '2.8.0+cu128', 'audited Torch version')
    for name in ('target', 'draft'):
        equal(status[name], str(Path(provenance['sources'][name+'_config']).parent), name+' config provenance')
    arms = set(args['arms'])
    require(arms <= {NATIVE, DELTA, IMPLICIT} and arms, 'unsupported status arms')
    equal({r['arm'] for r in raw}, arms, 'raw vs planned arms')
    sources = list(COMMON_SOURCES)
    if arms != {NATIVE}:
        sources += LAYOUT_SOURCES
    if IMPLICIT in arms:
        sources += ['direct_opt/implicit_kv_runtime.py', 'direct_opt/run_implicit_kv_layout.py']
        equal(arms, {IMPLICIT, DELTA}, 'audited paired implicit experiment')
        equal(status['paired_delta_baseline'], True, 'implicit paired baseline')
    checked = {}
    for name in sources:
        recorded = status['source_sha256'][name]
        equal(sha(HERE.parent/name), recorded, 'audited local source vs run: '+name)
        checked[name] = recorded
    return checked


def input_signature(row):
    return digest({k:row[k] for k in ('request_id', 'input_shapes', 'video_grid_thw', 'timestamps')})


def read_dataset(path, target, draft, vision, provenance, *, pair_repeat=None, abba_index=None):
    status = json.loads((path/'status.json').read_text())
    raw = [json.loads(x) for x in (path/'results.jsonl').read_text().splitlines() if x.strip()]
    sources = validate_status(status, raw, provenance)
    counted = []
    for r in raw:
        if r['phase'] != 'measure':
            continue
        context = f"{path.name}/{r['request_id']}/{r['arm']}/r{r['repeat']}"
        try:
            for key, value in dict(author_revision=provenance['draft_author_revision'],
                adapter_version='hf517_vispec_tree_v1', joint_version='vispec_joint_m_budget_v1').items():
                equal(r['result'][key], value, key)
            layout = validate_layout(r)
            # A new dict changes only dispatch identity after the real path was
            # checked. Every round/count/output/input field stays as recorded.
            dispatch = dict(r, arm='deployment_native' if r['arm'] == NATIVE else 'deployment_compiled')
            out = count_request(dispatch, target, draft, vision)
        except Exception as exc:
            raise ValueError(context+': '+str(exc)) from exc
        out.pop('observed_e2e_wall_s')
        out.update(arm=r['arm'], source_repeat=r['repeat'],
            pair_repeat=r['repeat'] if pair_repeat is None else pair_repeat,
            dataset=path.name, abba_process_index=abba_index, source_results=str((path/'results.jsonl').resolve()),
            input_signature_sha256=input_signature(r), full_rounds_sha256=digest(r['result']['rounds']),
            full_rounds=r['result']['rounds'], forward_counts=r['result']['forward_counts'], layout_validation=layout)
        if pair_repeat is not None:
            equal(r['repeat'], 0, 'ABBA single-process original repeat')
        counted.append(out)
    setup_count = int(status['deployment_adaptor'] == 'compiled')
    d, q = draft['hidden_size'], 2
    metadata = dict(path=str(path.resolve()), results_sha256=sha(path/'results.jsonl'),
        status_sha256=sha(path/'status.json'), source_sha256=sources,
        measured_rows=len(counted), excluded_warmup_rows=len(raw)-len(counted),
        manifest_sha256=status['request_sha256'], target=status['target'], draft=status['draft'],
        seed=status['args']['seed'], max_new_tokens=status['args']['max_new_tokens'],
        preparation_contractions_per_process=setup_count, preparation_matrix_macs=setup_count*(q*d*d+q*d))
    return counted, metadata


def totals(rows):
    require(rows, 'empty accounting group')
    components = {k:sum(r['components_macs'][k] for r in rows) for k in rows[0]['components_macs']}
    queries = {k:sum(r['query_reconstruction'][k] for r in rows) for k in rows[0]['query_reconstruction']}
    return dict(request_repeats=len(rows), independent_requests=len({r['request_id'] for r in rows}),
        tokens=sum(r['num_tokens'] for r in rows), verify_rounds=sum(r['verify_rounds'] for r in rows),
        total_major_macs=sum(components.values()), components_macs=components, query_totals=queries)


def compare(baseline, candidates):
    bmap = {(r['request_id'], r['pair_repeat']):r for r in baseline}
    cmap = {(r['request_id'], r['pair_repeat']):r for r in candidates}
    equal(len(bmap), len(baseline), 'unique baseline pair keys')
    equal(len(cmap), len(candidates), 'unique candidate pair keys')
    equal(set(cmap), set(bmap), 'complete paired request coverage')
    pairs = []
    for key, c in sorted(cmap.items()):
        b = bmap[key]
        equal(c['input_signature_sha256'], b['input_signature_sha256'], 'paired input shapes/timestamps')
        equal(c['precision'], b['precision'], 'paired precision')
        components = {name:c['components_macs'][name]-b['components_macs'][name] for name in b['components_macs']}
        pairs.append(dict(request_id=key[0], pair_repeat=key[1],
            baseline_dataset=b['dataset'], candidate_dataset=c['dataset'],
            same_output=c['output_token_ids'] == b['output_token_ids'],
            same_full_rounds=c['full_rounds_sha256'] == b['full_rounds_sha256'],
            baseline_tokens=b['num_tokens'], candidate_tokens=c['num_tokens'],
            baseline_verify_rounds=b['verify_rounds'], candidate_verify_rounds=c['verify_rounds'],
            baseline_major_macs=b['total_major_macs'], candidate_major_macs=c['total_major_macs'],
            saving_percent=100*(1-c['total_major_macs']/b['total_major_macs']),
            component_delta_candidate_minus_baseline_macs=components,
            adaptor_saving_macs=-components['adaptor'],
            actual_trajectory_extra_macs=sum(v for name,v in components.items() if name != 'adaptor')))
    subsets = {}
    for label, selected in [('all', pairs), ('same_output', [p for p in pairs if p['same_output']]),
                            ('same_full_rounds', [p for p in pairs if p['same_full_rounds']])]:
        if not selected:
            subsets[label] = None
            continue
        bm, cm = (sum(p[k+'_major_macs'] for p in selected) for k in ('baseline', 'candidate'))
        subsets[label] = dict(pairs=len(selected), independent_requests=len({p['request_id'] for p in selected}),
            baseline_major_macs=bm, candidate_major_macs=cm,
            ratio_of_totals_saving_percent=100*(1-cm/bm),
            mean_pair_saving_percent=mean(p['saving_percent'] for p in selected),
            adaptor_saving_macs=sum(p['adaptor_saving_macs'] for p in selected),
            actual_trajectory_extra_macs=sum(p['actual_trajectory_extra_macs'] for p in selected))
    return dict(same_output_pairs=sum(p['same_output'] for p in pairs),
        same_full_round_pairs=sum(p['same_full_rounds'] for p in pairs), subsets=subsets, pairs=pairs)


def markdown(report):
    primary = report['comparisons']['implicit_vs_native_abba']
    primary_total = primary['subsets']['all']
    lines = ['# 原版与 implicit 布局的完整请求主要矩阵 MAC', '',
        '只读重建；没有加载模型或执行 GPU。每组是 12 个不同请求各 2 次，共 24 次测量。暖机排除。', '',
        f"**本批整请求主要矩阵 MAC 减少 {primary_total['ratio_of_totals_saving_percent']:.6f}%，几乎全部来自已有 compiled 适配器。** implicit 与同批 delta 的矩阵 MAC 和完整 round 记录均相同；implicit 去除元数据/复制开销的效果属于另一份 IO 与时间账。", '',
        '**这里计算主要卷积/矩阵乘加，不是全部标量 FLOPs、GPU 指令或 DRAM。跨批次只比较工作量，不比较时延。**', '',
        '## 合计', '', '| 组 | EOS 请求数 | token（含EOS） | 验证轮次 | 总 TMAC |', '|---|---:|---:|---:|---:|']
    for name, s in report['groups'].items():
        lines.append(f"| {name} | {s['request_repeats']} | {s['tokens']} | {s['verify_rounds']} | {s['total_major_macs']/1e12:.9f} |")
    lines += ['', '## 配对', '', '正省算率表示 candidate 主要 MAC 减少，主列使用合计之比。', '',
        '| 比较 | 同输出/完整rounds | 合计省算% | 适配器节省 GMAC | 其余实际轨迹增量 GMAC |', '|---|---:|---:|---:|---:|']
    for name, c in report['comparisons'].items():
        s = c['subsets']['all']
        lines.append(f"| {name} | {c['same_output_pairs']}/{c['same_full_round_pairs']}（共{s['pairs']}对） | {s['ratio_of_totals_saving_percent']:.6f} | {s['adaptor_saving_macs']/1e9:.6f} | {s['actual_trajectory_extra_macs']/1e9:+.6f} |")
    main = report['comparisons']['implicit_vs_native_abba']
    lines += ['', '同输出并不保证验证树/接受率轨迹相同，逐请求都按自己的实际调用计数。新增布局只改变复制和元数据，不直接减少矩阵 MAC；共享 embedding 也不改变 MAC。', '',
        '## 主比较分项（合计 TMAC）', '', '| 部分 | native ABBA24 | implicit24 |', '|---|---:|---:|']
    for k, v in report['groups']['native_abba24']['components_macs'].items():
        lines.append(f"| {k} | {v/1e12:.9f} | {report['groups']['implicit24']['components_macs'][k]/1e12:.9f} |")
    lines += ['', '## Query 与 attention 形状重建（24次合计）', '',
        '| 量 | native ABBA24 | implicit24 |', '|---|---:|---:|']
    for k, v in report['groups']['native_abba24']['query_totals'].items():
        lines.append(f"| {k} | {v} | {report['groups']['implicit24']['query_totals'][k]} |")
    same = primary['subsets']['same_full_rounds']
    lines += ['', f"输出24/24逐token相同，完整rounds仅{primary['same_full_round_pairs']}/24相同。完整rounds相同的{same['pairs']}对（{same['independent_requests']}个视频）省算{same['ratio_of_totals_saving_percent']:.6f}%，其中非适配器矩阵增量为0。",
        '其余轨迹差异仍逐项计入：只有 EZyLBq2jJtE 的两次重复使 attention 的 Q×K 合计略变，Target少60、Draft少48，合计少12,386,304 MAC。其余已改变的selected路径没有改变矩阵形状。', '']
    lines += ['', '## 逐请求主比较', '', '| 请求 | repeat配对 | token native→implicit | 验证 native→implicit | 输出/完整rounds一致 | 总 MAC省算% |', '|---|---:|---:|---:|---:|---:|']
    for p in main['pairs']:
        lines.append(f"| {p['request_id'].removeprefix('direct_quality_vdc_v_')} | {p['pair_repeat']} | {p['baseline_tokens']}→{p['candidate_tokens']} | {p['baseline_verify_rounds']}→{p['candidate_verify_rounds']} | {p['same_output']}/{p['same_full_rounds']} | {p['saving_percent']:.6f} |")
    lines += ['', '## 公式和范围', '',
        '- 复用 account_request_macs.py、compute_cost.py 和已核验的 checkpoint 配置。原始 source SHA 必须与所审本地源码一致；未知臂直接拒绝。输入 patch/文本形状、EOS、Target/Draft 所有记录的 query/call 数、提交路径和缓存链全部校验。',
        '- 每层语言线性为 Q(2D²+2D·Dkv+3D·I)，注意力为 2D·Q·K；D=3584、I=18944，Target 28层/Dkv512，Draft 1层/Dkv3584。词表每行为 D×152064。',
        '- Target prefill Q=K=P；验证 Q=30、K=logical_base+30。逐层 active shape totals 与每轮宽度重新核对，固定预留容量/被排除空槽不作为 attention K。',
        '- Draft 首轮压缩prompt S=P−visual_tokens+1；后续同步query等于上一轮提交数，再有3次各8query分支。LM head 为每轮25行；两层视觉融合为4D²×(Draft decoder query数−1)。',
        '- 原生适配器的主 MAC 为2ND²+2NqD+qD²。compiled 为2NHqD+2qD²，H28/q2；N2016为0.860618752 GMAC。其一次方向构造 qD²+qD=25,697,280 MAC 每进程另计。',
        '- Target/Draft 主体 FP16，共享 LM head 实际 FP32。相同 MAC 不能视为相同硬件成本；dense attention 形状计数包含树mask位置，不能解释为内核实际执行指令。',
        '- 容量、复制、metadata写入、掩码、softmax、归一化、激活、RoPE、top-k、逐元素操作、CPU往返、视频解码与准备均不计入本矩阵账。没有把 IO 收益伪装成 MAC 节省。',
        '- native ABBA00/03的原repeat均为0；按进程顺序映射到配对repeat0/1。delta ABBA01/02亦如此。implicit同批保留原repeat0/1。原始文件不变，JSON保留source_repeat与ABBA索引。',
        '- 配置、输入grid/timestamps、manifest、seed、精度一致。native与implicit来自不同批次，完整round记录哈希及差异保留；本报告不做跨批次时延、吞吐、能量或质量结论。',
        '', '逐请求全分项、query重建、完整rounds与来源哈希见 request_macs.jsonl；所有配对子集与准备量见 summary.json。', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    base = ROOT/'coordination/direct_opt_20261002'
    p.add_argument('--abba', type=Path, default=base/'throughput_native_delta_abba_r1')
    p.add_argument('--implicit', type=Path, default=base/'implicit_paired12_r1')
    p.add_argument('--config-dir', type=Path, default=base/'full_request_mac_accounting')
    p.add_argument('--vision-config', type=Path, default=HERE.parent/'vision_config.json')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    target, draft, vision, provenance = load_configs(args.config_dir, args.vision_config)
    abba = json.loads((args.abba/'status.json').read_text())
    equal(abba['status'], 'completed', 'ABBA supervisor completion')
    entries = abba['completed']
    equal([(e['index'], e['arm'], e['returncode']) for e in entries],
          [(0,'native',0),(1,'delta',0),(2,'delta',0),(3,'native',0)], 'ABBA process binding')
    groups = {k:[] for k in ('native_abba24', 'delta_abba24', 'delta_within_implicit24', 'implicit24')}
    datasets = {}
    for entry in entries:
        directory = args.abba/Path(entry['out']).name
        rows, info = read_dataset(directory, target, draft, vision, provenance,
            pair_repeat=int(entry['index'] in (2,3)), abba_index=entry['index'])
        expected_arm = NATIVE if entry['arm'] == 'native' else DELTA
        require(all(r['arm'] == expected_arm for r in rows), 'ABBA command/row arm binding')
        groups['native_abba24' if expected_arm == NATIVE else 'delta_abba24'].extend(rows)
        datasets[directory.name] = info
    rows, info = read_dataset(args.implicit, target, draft, vision, provenance)
    datasets[args.implicit.name] = info
    for row in rows:
        groups['implicit24' if row['arm'] == IMPLICIT else 'delta_within_implicit24'].append(row)
    for name, rows in groups.items():
        equal(len(rows), 24, name+' planned measured count')
        equal(set(Counter(r['request_id'] for r in rows).values()), {2}, name+' request repetition')
        equal({r['pair_repeat'] for r in rows}, {0,1}, name+' comparison repeats')
    first = next(iter(datasets.values()))
    for data in datasets.values():
        for k in ('manifest_sha256','target','draft','seed','max_new_tokens'):
            equal(data[k], first[k], 'cross-dataset '+k)
    comparisons = {name:compare(groups[b], groups[c]) for name,b,c in (
        ('implicit_vs_native_abba','native_abba24','implicit24'),
        ('delta_abba_vs_native_abba','native_abba24','delta_abba24'),
        ('implicit_vs_same_run_delta','delta_within_implicit24','implicit24'))}
    report = dict(scope='Analytic major convolution/matrix MACs; not all scalar FLOPs, hardware instruction counts, DRAM or timing',
        convention='1 MAC is one multiply-accumulate; conventional arithmetic FLOPs=2*MAC',
        timing_comparison=False, quality_claim=None, excluded=EXCLUDED,
        pairing='ABBA process index0(native)/1(delta) -> pair_repeat0; index3(native)/2(delta) -> pair_repeat1; implicit original repeats retained',
        configs=dict(target=target,draft=draft,vision=vision), config_provenance=provenance,
        config_sha256={str(x.resolve()):sha(x) for x in [args.config_dir/'target_config.json',args.config_dir/'draft_config.json',args.vision_config]},
        counter_sha256={str(x):sha(x) for x in [Path(__file__),HERE/'account_request_macs.py']},
        datasets=datasets, groups={k:totals(v) for k,v in groups.items()}, comparisons=comparisons)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (args.out/'request_macs.jsonl').write_text(''.join(json.dumps(dict(r,group=name),ensure_ascii=False)+'\n' for name,rows in groups.items() for r in rows))
    (args.out/'REPORT.md').write_text(markdown(report))
    print(json.dumps({k:v['subsets']['all'] for k,v in comparisons.items()},indent=2))


if __name__ == '__main__':
    main()
