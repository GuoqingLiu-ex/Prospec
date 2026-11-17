"""Offline receipt audit of the four-feature streaming-adaptor module pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def require(value, message):
    if not value:
        raise ValueError(message)


def equal(actual, expected, message):
    require(actual == expected, f'{message}: {actual!r} != {expected!r}')


def audit(status):
    equal(status['status'], 'completed', 'module run completion')
    equal(status['completed_features'], 4, 'completed feature count')
    equal(status['expected_features'], 4, 'planned feature count')
    equal(len(status['results']), 4, 'all feature records')
    equal(sorted(r['shape'][1] for r in status['results']), [468, 792, 884, 2016], 'actual captured N')
    equal(status['actual_sram_residency'], False, 'no SRAM measurement claim')
    equal(status['allow_tf32'], False, 'FP32 GEMM precision')
    equal(status['timing_diagnostics'], dict(check_finite=False, record_workspace=False), 'timed diagnostic flags')
    configs = {'stream_g7_b12':(7,12), 'stream_g14_b24':(14,24), 'stream_g28_b48':(28,48)}
    equal(set(status['candidate_metadata']), set(configs), 'candidate identities')
    for name, (g,b) in configs.items():
        m = status['candidate_metadata'][name]
        for key,value in dict(heads_per_group=g, token_tile=b, score_and_accumulator_dtype='torch.float32',
            pooled_rounding_and_projection_dtype='torch.float16', check_finite=False,
            record_workspace=False, full_token_kv_materialized=False,
            sram_residency_proven=False).items():
            equal(m[key], value, name+' '+key)
    rows = []
    for feature in status['results']:
        _, n, d = feature['shape']
        equal(feature['shape'], [1,n,3584], 'input shape')
        equal(feature['dtype'], 'torch.float16', 'input precision')
        equal(set(feature['candidates']), set(configs), 'feature candidates')
        equal(set(feature['measurements']), {'native','compiled',*configs}, 'all module measurements')
        for control in ('compiled_vs_native', 'dense_fp32_vs_compiled'):
            equal(feature[control]['finite'], True, control+' finite')
        timing = {}
        for name,m in feature['measurements'].items():
            equal(len(m['rows']), status['args']['repeats'], 'all timing repeats')
            require(status['args']['warmup'] >= 1 and len(m['rows']) >= 2, 'warmup/repetition protocol')
            for i,r in enumerate(m['rows']):
                equal(r['repeat'], i, 'timing repeat order')
                equal(r['output_bytes'], 1*2*d*2, 'timed output payload')
                require(math.isfinite(r['wall_ms']) and r['wall_ms'] > 0, 'positive finite module wall')
                require(r['incremental_peak_bytes'] >= r['output_bytes'], 'peak must include output')
            equal(m['median_wall_ms'], median(r['wall_ms'] for r in m['rows']), 'recomputed median wall')
            equal(m['median_incremental_peak_bytes'], median(r['incremental_peak_bytes'] for r in m['rows']), 'recomputed median peak')
            timing[name] = dict(median_wall_ms=m['median_wall_ms'],
                median_incremental_peak_bytes=m['median_incremental_peak_bytes'],
                raw_wall_ms=[r['wall_ms'] for r in m['rows']])
        candidates = {}
        for name,(g,b) in configs.items():
            c = feature['candidates'][name]
            w = c['workspace_receipt']
            for key,value in dict(completed=True,input_shape=[1,n,d],output_shape=[1,2,d],
                head_groups=math.ceil(28/g),tiles_per_group=math.ceil(n/b),input_scans=math.ceil(28/g),
                tokens_visited_per_batch=n*math.ceil(28/g),
                logical_input_elements_visited=n*d*math.ceil(28/g),
                logical_input_source_read_bytes=n*d*2*math.ceil(28/g),
                logical_reads_are_measured_dram=False,theoretical_budget_matches_numeric_assumptions=True).items():
                equal(w[key], value, name+' '+key)
            equal(w['workspace']['enabled'], True, 'separate diagnostic workspace receipt')
            theoretical = w['theoretical_pooling_sram']
            expected = 6*g*2*d + 4*b*d + 4*g*2*b + 10*g*2 + 2*2*d + 32768
            equal(theoretical['total_bytes'], expected, 'independent planning budget sum')
            equal(sum(theoretical['terms_bytes'].values()), expected, 'planning budget terms')
            equal(theoretical['actual_sram_measured'], False, 'theory vs SRAM')
            errors = {key:c[key] for key in ('vs_native','vs_compiled','vs_dense_fp32')}
            for key,e in errors.items():
                equal(e['finite'], True, name+' '+key+' finite')
                require(all(math.isfinite(e[x]) and e[x] >= 0 for x in ('max_abs','rmse','relative_l2')), 'finite nonnegative error stats')
            candidates[name] = dict(errors=errors, logical_input_scans=w['input_scans'],
                logical_input_source_read_bytes=w['logical_input_source_read_bytes'],
                theoretical_pooling_budget_bytes=expected,
                explicit_torch_workspace_peak_bytes=w['workspace']['peak_explicit_live_workspace_storage_bytes'])
        rows.append(dict(feature=feature['feature'], shape=feature['shape'],
            compiled_vs_native=feature['compiled_vs_native'], dense_fp32_vs_compiled=feature['dense_fp32_vs_compiled'],
            candidates=candidates, measurements=timing, measurement_order=feature['measurement_order']))
    return dict(validated=True, actual_sram_measurement=False, rows=rows,
        hardware_context={key:status.get('hardware',{}).get(key) for key in
            ('backend','physical_device_index','sharing','energy_attribution','foreign_processes')},
        numerical_threshold=None, numerical_quality_pass=None,
        recommendation='A bounded G28/B48 full-request numerical/quality pilot is warranted if these receipts validate. Finite module outputs and proximity to a dense FP32 control are diagnostic evidence, not an acceptance tolerance or a quality pass.',
        limitations=['Only captured-feature module evaluation; no full-request token/fact quality evidence.',
            'Torch explicit-storage snapshots are not SRAM and exclude internal operator/BLAS and validation temporaries.',
            'Incremental CUDA allocated peak includes output; all arms/weights and caller X remain resident. Not deployment or full-request memory.',
            'Sequential per-arm repeats with feature-level order rotation; module wall includes Python/host dispatch. Not paired end-to-end speed evidence.',
            'Logical group input reads do not measure DRAM and omit weight traffic and Torch temporary/cast traffic.',
            'The dense FP32 control uses the same already-rounded folded parameters and model-dtype V/O; it is not exact real arithmetic or proof against the native model.',
            'No post-hoc error threshold is introduced; long-answer output/trajectory and reference-fact checks remain necessary.'])


def markdown(report):
    lines=['# Streaming 适配器四输入模块审计', '',
        '收据完整性校验通过。下面是功能参考的模块观测；Torch workspace 不能解释为片上 SRAM。', '',
        '## 模块 wall / CUDA 增量 allocated peak', '',
        '| N | 方法 | wall中位 ms | 增量峰值 MiB |', '|---:|---|---:|---:|']
    for row in report['rows']:
        for name,m in row['measurements'].items():
            lines.append(f"| {row['shape'][1]} | {name} | {m['median_wall_ms']:.6f} | {m['median_incremental_peak_bytes']/2**20:.6f} |")
    lines += ['', '## 原版与控制数值差异', '',
        '| N | compiled对native maxabs / relL2 | denseFP32对compiled maxabs / relL2 |', '|---:|---:|---:|']
    for row in report['rows']:
        values=['%.8g / %.8g'%(row[k]['max_abs'],row[k]['relative_l2']) for k in
            ('compiled_vs_native','dense_fp32_vs_compiled')]
        lines.append(f"| {row['shape'][1]} | "+' | '.join(values)+' |')
    lines += ['', '各控制和候选输出均finite；当前保存的全部误差比较都不是逐位相同。compiled本身已经改变舍入，不能把对compiled的偏差都归因于流式归约。']
    lines += ['', '## 数值与显式 workspace', '',
        '| N | 方法 | 对native maxabs / relL2 | 对compiled maxabs / relL2 | 对denseFP32 maxabs / relL2 | Torch workspace MiB | X扫描次数 / 逻辑读MiB |', '|---:|---|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        for name,c in row['candidates'].items():
            errors = ['%.8g / %.8g'%(c['errors'][k]['max_abs'],c['errors'][k]['relative_l2']) for k in ('vs_native','vs_compiled','vs_dense_fp32')]
            lines.append(f"| {row['shape'][1]} | {name} | "+' | '.join(errors)+f" | {c['explicit_torch_workspace_peak_bytes']/2**20:.6f} | {c['logical_input_scans']} / {c['logical_input_source_read_bytes']/2**20:.6f} |")
    lines += ['', '## 建议', '',
        '可以做 G28/B48 的有界完整长回答数值与质量 pilot。四输入 finite、shape 与计量收据通过，仅说明值得继续验证；没有设置事后容差，也不构成原版逐位一致或回答质量通过。先比较自然 EOS、逐token输出、完整轮次路径和参考关键事实；若输出变化，应保留变化并评估事实，不能仅按长度或模块误差判定质量。', '',
        '它仍是普通 Torch 功能参考：模块 wall 的 Python 循环开销不能直接推到硬件流水线；反过来也不能用理论 SRAM 预算证明服务器加速或片上可实现性。', '', '## 口径', '',
        '- 评分和在线归约确为 FP32；池化结果舍入到 FP16 后执行模型 V/O，TF32关闭。所有候选输出[1,2,3584]且finite。',
        '- timed pass 的 check_finite/record_workspace 均关闭；数值CPU拷贝和workspace诊断在计时外。普通shape检查、storage收据和Python循环仍在forward计时内。',
        '- workspace快照不含caller X、持久权重、内部BLAS/临时和validation张量；CUDA增量峰值含输出。所有臂权重同时驻留，不能推部署显存。',
        '- 每组完整扫描X；G7/G14/G28分别4/2/1遍。列出的源特征逻辑读取不含后续V/O权重、casts与其它读写，不是DRAM测量。',
        '- denseFP32控制与候选共用已舍入folded参数和FP16 V/O；并非对native的高精度真值。',
        '- 本次各臂顺序批量重复，按feature轮换顺序；表为探索性模块wall，没有端到端加速/节能结论。',
        '- 硬件采样记录到常驻图形PID6360，energy_attribution=unavailable_shared_device；本报告不从整卡能量推候选能耗。', '']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    report=audit(json.loads(args.status.read_text()))
    report['source_status']=str(args.status.resolve())
    report['source_status_sha256']=hashlib.sha256(args.status.read_bytes()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'STREAMING_MODULE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (args.out/'STREAMING_MODULE_AUDIT.md').write_text(markdown(report))
    print(json.dumps(dict(validated=report['validated'],features=[r['shape'][1] for r in report['rows']]),indent=2))


if __name__=='__main__':
    main()
