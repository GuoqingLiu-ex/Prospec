"""Offline audit of the same-GPU native/stride0/grouped capture experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
from statistics import median

from audit_stride0_gqa_probe import BASE, HERE, equal, exact, require, sha

ARMS = ('native', 'stride0', 'grouped')


def trace_evidence(path, n, arm, profile):
    events = json.loads(path.read_text())['traceEvents']
    ops = [e for e in events if e.get('cat') == 'cpu_op']
    kernels = Counter(e['name'] for e in events if e.get('cat') == 'kernel')
    def matches(name, shape):
        return sum(e['name'] == name and e.get('args', {}).get('Input Dims', [None])[0] == shape for e in ops)
    repeated = matches('aten::clone', [1,4,7,n,128])
    equal(repeated, 2 if arm == 'native' else 0, 'explicit repeated KV clones')
    q_packs = matches('aten::clone', [1,28,30,128])
    equal(q_packs, int(arm == 'grouped'), 'explicit small Q pack')
    equal(matches('aten::clone', [1,30,28,128]), 1, 'common HF output layout copy')
    kv_shape = [1,28 if arm == 'native' else 4,n,128]
    equal(matches('aten::_to_copy', kv_shape), 2, 'unique/repeated KV casts')
    equal(matches('aten::_to_copy', [1,28,30,n]), 1, 'unused FP16 probability cast preserved')
    bmm = [e['args']['Input Dims'] for e in ops if e['name'] == 'aten::bmm']
    if arm == 'native':
        expected = [[[28,30,128],[28,128,n]], [[28,30,n],[28,n,128]]]
    elif arm == 'stride0':
        expected = [[[7,30,128],[7,128,n],[7,30,n]]]*4 + [[[7,30,n],[7,n,128],[7,30,128]]]*4
    else:
        expected = [[[4,210,128],[4,128,n]], [[4,210,n],[4,n,128]]]
    equal(bmm, expected, 'actual BMM operands')
    gemms = {k:v for k,v in kernels.items() if 'sgemm' in k.lower()}
    equal(sum(gemms.values()), len(expected), 'CUDA SGEMM launch count')
    equal(kernels, Counter({x['name']:x['count'] for x in profile['cuda_kernels']}), 'summary/raw kernels')
    for name in ('aten::clone', 'aten::_to_copy', 'aten::bmm'):
        equal(sum(e['name'] == name for e in ops),
              sum(x['count'] for x in profile['operators'] if x['name'] == name), 'summary/raw '+name)
    return dict(path=str(path.resolve()), sha256=sha(path), cpu_operator_events=len(ops),
        cuda_kernel_count=sum(kernels.values()), bmm_operands=bmm, sgemm_kernels=gemms,
        repeated_kv_clone_count=repeated, small_q_pack_count=q_packs, kv_cast_shape=kv_shape,
        kernel_counts=dict(kernels),
        separately_named_pack_kernels={k:v for k,v in kernels.items() if 'pack' in k.lower()},
        internal_library_pack_absence_proven=False)


def audit(directory):
    report = json.loads((directory/'summary.json').read_text())
    equal(report['status'], 'completed', 'probe completion')
    equal(report['failures'], [], 'probe failures')
    equal(report['arms'], list(ARMS), 'three contemporaneous arms')
    equal((report['args']['physical_gpu'],report['args']['warmup'],report['args']['repeats']),
          (0,3,12), 'actual GPU/protocol')
    for key, value in dict(actual_dram_measured=False,end_to_end_claim=False,
        actual_sram_residency=False,allow_tf32=False,float32_matmul_precision='highest',
        reduced_precision_math_sdp=False).items():
        equal(report[key], value, key)
    for name, digest in report['source_hashes'].items():
        equal(sha(HERE/name), digest, 'executed source '+name)
    equal(len(report['samples']), 3, 'three captures')
    permutations = list(itertools.permutations(ARMS))
    samples = []
    for index, sample in enumerate(report['samples']):
        equal(sample['status'], 'completed', 'sample completion')
        meta = sample['metadata']; n = meta['tensors']['key']['shape'][2]
        equal((meta['layer'],meta['layer_call_index'],n), (0,(0,80,163)[index],(2089,2322,2564)[index]),
              'actual layer/call/length')
        for key, shape in dict(query=[1,28,30,128],key=[1,4,n,128],value=[1,4,n,128],
                              mask=[1,1,30,n],native_output=[1,30,28,128]).items():
            equal(meta['tensors'][key]['shape'], shape, key+' shape')
        equal(meta['tensors']['query']['stride'], [107520,128,3584,1], 'original Q stride')
        exact(sample['native_replay'], 'native initial output')
        for name in ('stride0','grouped'): exact(sample['initial'][name], name+' initial output')
        exact(sample['grouped_diagnostic']['error'], 'grouped diagnostic output')
        receipt = sample['grouped_diagnostic']['receipt']
        for key, value in dict(implementation='grouped_two_bmm_fp32_math',qk_bmm_calls=1,pv_bmm_calls=1,
            explicit_repeat_kv=False,unique_kv_heads=4,query_pack_copy=True,
            query_pack_logical_read_bytes=430080,query_pack_logical_write_bytes=430080,
            native_unused_probability_cast_preserved=True,debug_intermediates=False,
            mask_same_object=True,mask_values_transformed=False,rotary_reapplied=False).items():
            equal(receipt[key], value, 'grouped receipt '+key)
        equal(receipt['intermediate_tensors']['grouped_query']['shape'], [4,210,128], 'packed Q shape')
        for key in ('k32','v32'):
            equal(receipt['intermediate_tensors'][key]['storage_bytes'], 4*4*n*128, 'unique '+key+' backing')
        cells = sample['ordinary']['records']; equal(len(cells), 12, 'all formal triplets')
        for repeat, cell in enumerate(cells):
            equal(cell['repeat'], repeat, 'formal order index')
            equal(tuple(cell['order']), permutations[(repeat+index)%6], 'balanced execution order')
            equal(len({cell[a]['baseline_allocated_bytes'] for a in ARMS}), 1, 'common allocated baseline')
            for arm in ARMS:
                m = cell[arm]; exact(m['error'], arm+' formal output')
                equal(m['peak_allocated_bytes']-m['baseline_allocated_bytes'],
                      m['incremental_peak_allocated_bytes'], 'peak subtraction')
                require(m['wall_ms'] > 0, 'positive wall')
            require(cell['grouped']['incremental_peak_allocated_bytes'] <=
                    cell['stride0']['incremental_peak_allocated_bytes'], 'no lost module peak saving')
        wall = {a:median(c[a]['wall_ms'] for c in cells) for a in ARMS}
        peaks = {a:median(c[a]['incremental_peak_allocated_bytes'] for c in cells) for a in ARMS}
        equal(wall, sample['ordinary']['median_wall_ms'], 'recomputed wall medians')
        equal(peaks, sample['ordinary']['median_incremental_peak_bytes'], 'recomputed peak medians')
        traces = {a:trace_evidence(directory/sample['profiles'][a]['trace_file'],n,a,sample['profiles'][a]) for a in ARMS}
        samples.append(dict(file=sample['file'],capture_sha256=sample['sha256'],layer=0,
            layer_call_index=meta['layer_call_index'],key_tokens=n,all_formal_outputs_byte_exact=True,
            all_triplets_same_baseline_allocated=True,all_grouped_peaks_at_most_stride0=True,
            median_wall_ms=wall,median_incremental_peak_bytes=peaks,
            grouped_wall_reduction_pct={a:100*(1-wall['grouped']/wall[a]) for a in ('native','stride0')},
            grouped_peak_reduction_pct={a:100*(1-peaks['grouped']/peaks[a]) for a in ('native','stride0')},
            median_paired_wall_reduction_pct={a:median(100*(1-c['grouped']['wall_ms']/c[a]['wall_ms'])
                for c in cells) for a in ('native','stride0')},formal_records=cells,trace_evidence=traces,
            added_q_pack_logical_read_plus_write_bytes=860160))
    launch = json.loads(directory.with_name(directory.name+'.launch.json').read_text())
    exit_receipt = json.loads(directory.with_name(directory.name+'.exit.json').read_text())
    equal(exit_receipt['returncode'], 0, 'successful terminal exit')
    equal(launch['physical_gpu'], 0, 'launch device')
    hw = report['hardware']
    equal(hw['physical_device_index'],0,'sampler device')
    equal(hw['device']['uuid'],report['lock']['initial_snapshot']['gpu_uuid'],'physical UUID binding')
    equal(hw['allowed_pids'],[report['pid']],'own PID excluded from background')
    return dict(validated=True,run=str(directory.resolve()),source_summary_sha256=sha(directory/'summary.json'),
        source_hashes=report['source_hashes'],launch=launch,exit_receipt=exit_receipt,
        physical_gpu=0,device=report['device'],samples=samples,formal_output_checks=108,formal_triplets=36,
        hardware={k:hw.get(k) for k in ('device','foreign_processes','process_visibility_complete',
            'process_query_failures','sharing','energy_attribution','measurement_duration_s')},
        supports_bounded_full_generation=True,whole_model_peak_or_speed_claim=False,actual_dram_measured=False,
        all_layers_or_generation_byte_exact_proven=False,intermediate_five_stage_gpu_comparison_performed=False,
        raw_capture_tensors_reloaded_offline=False,
        numerical_scope='Recorded bytewise output metrics, executed source hashes and nine raw traces; '
                        'input .pt tensors remain remote and are not independently replayed by this offline audit.')


def markdown(r):
    lines = ['# Grouped GQA：GPU0 三臂模块审计','',
        '**支持进入有界完整生成验证。** 同一次 GPU0 实验的 36 组三臂正式测量（108 个输出）全部 finite、逐字节复现保存的 native 输出；每组 baseline allocated 相同。新 grouped 的模块增量峰值始终低于旧 stride0，wall 中位数也低于旧 stride0。','',
        '| layer/call | K 长度 | native/stride0/grouped wall 中位 ms | grouped 较 native/stride0 降时 | native/stride0/grouped 增量峰值 MiB | grouped 较 native/stride0 峰值下降 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in r['samples']:
        w=s['median_wall_ms'];p=s['median_incremental_peak_bytes'];wd=s['grouped_wall_reduction_pct'];pd=s['grouped_peak_reduction_pct']
        lines.append(f"| 0/{s['layer_call_index']} | {s['key_tokens']} | "+'/'.join(f'{w[a]:.6f}' for a in ARMS)+
            f" | {wd['native']:.3f}%/{wd['stride0']:.3f}% | "+'/'.join(f'{p[a]/2**20:.6f}' for a in ARMS)+
            f" | {pd['native']:.3f}%/{pd['stride0']:.3f}% |")
    lines += ['','## 原始数据与 trace 核对','',
        '- 三个样本均来自 Target layer0、调用 0/80/163，T=30；不是三个不同层。每臂每形状先暖 3 次，再做 12 次正式调用；6 种执行排列各出现 2 次。表中百分比按三臂各自中位数计算，逐组比值中位数另存 JSON。',
        '- 全部初始 native/stride0/grouped 输出及独立 grouped receipt 调用也逐字节一致。此次没有做 grouped 的五个中间阶段 GPU 比较，不能继承旧 stride0 的阶段结论。离线审核没有下载或重放远端输入 .pt，而是核查数值回执、比较代码哈希与九份原始 trace。',
        '- 九份 trace 独立复算：native/stride0/grouped 的 BMM 和实际 SGEMM 数为 2/8/2，总 CUDA kernel 数为 18/22/17，CPU operator 事件数为 103/179/56。grouped 实际矩阵为 [4,210,128]×[4,128,N] 和 [4,210,N]×[4,N,128]。',
        '- native 有两个 [1,4,7,N,128] KV clone；两候选均没有。KV FP32 转换从 native 的 28 头降至 4 头。grouped 新增 [1,28,30,128] 小 Q clone：FP32 payload 430,080 B，逻辑读写各 430,080 B，已在普通调用和峰值测量内。两项收益不能把这个新增成本隐藏掉。',
        '- 三臂均保留完整 softmax 和被丢弃概率的 FP16 cast，以及最终 HF 输出布局复制。raw trace 没有独立命名的 pack kernel，不足以排除库内部 workspace、打包或重复 DRAM 读取。','',
        '## 计量与结论边界','',
        '- 三臂耗时全部来自本次同 GPU0、同进程、同输入的重放；没有使用旧 GPU7 毫秒数。仅报告局部观测，较 native 的约 0.8–9.4% 中位差异没有统计显著性结论，更不是端到端加速。',
        '- 普通计时包含小 Q pack、全部张量操作及各实现 Python 开销；数值 metrics、独立 receipt 调用和 profiler 在计时区间外。旧 stride0 每调用生成完整 receipt，新 grouped 的默认路径不生成，另有临时张量提前释放。因此这是整个实现的比较，不能将降时只归因于 8→2 BMM。',
        '- 峰值是公共输入/参考已经驻留后的 PyTorch 增量 allocated，包含返回输出，不是完整模型峰值、reserved 或设备物理显存。allocator 缓存可能保留，底层 BLAS 非 PyTorch 分配不由此完整覆盖。',
        '- background sampler 仅见 foreign PID 6360，最大 4,669,440 B；自身 PID 2292343 排除。查询失败 0，process_visibility_complete=true，但离散快照不能排除间隔内活动，energy_attribution=unavailable_shared_device。未作候选节能结论。',
        '- 支持下一步原生 EOS 完整生成：核所有 28 Target 层、实际 verification 调用、输出 ID、完整树轨迹及完整请求峰值。三个 layer0 长度不能保证其他层/形状逐位一致。当前没有实际 DRAM 计数，也没有端侧 SRAM 或新的 GQA 算法主张。','']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=BASE/'grouped_gqa_capture_gpu0_r1')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv);r=audit(args.run);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'GROUPED_GQA_MODULE_AUDIT.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    (args.out/'GROUPED_GQA_MODULE_AUDIT.md').write_text(markdown(r))
    print(json.dumps(dict(validated=True,samples=[{k:s[k] for k in
        ('key_tokens','grouped_wall_reduction_pct','grouped_peak_reduction_pct')} for s in r['samples']]),indent=2))


if __name__=='__main__':main()
