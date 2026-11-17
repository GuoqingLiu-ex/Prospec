"""Offline audit of actual three-capture stride-zero GQA receipts and traces."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median


HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/'coordination/direct_opt_20261002'


def require(value,message):
    if not value:raise ValueError(message)


def equal(actual,expected,message):
    require(actual==expected,f'{message}: {actual!r} != {expected!r}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(error,context):
    equal(error,dict(finite=True,bitwise_equal=True,relative_l2=0.0,max_abs=0.0),context)


def trace_evidence(path,n,arm,profile):
    trace=json.loads(path.read_text())
    events=trace['traceEvents']
    ops=[e for e in events if e.get('cat')=='cpu_op']
    kernels=Counter(e['name']for e in events if e.get('cat')=='kernel')
    def matching(name,shape):
        return sum(e['name']==name and e.get('args',{}).get('Input Dims',[None])[0]==shape for e in ops)
    repeat_clones=matching('aten::clone',[1,4,7,n,128])
    equal(repeat_clones,2 if arm=='native'else 0,'explicit repeated KV clone count')
    kv_casts=matching('aten::_to_copy',[1,28 if arm=='native'else 4,n,128])
    equal(kv_casts,2,'two KV conversion operators')
    equal(matching('aten::clone',[1,30,28,128]),1,'shared HF output layout copy')
    bmm=[e for e in ops if e['name']=='aten::bmm']
    equal(len(bmm),2 if arm=='native'else 8,'raw BMM count')
    expected=[[[28,30,128],[28,128,n]],[[28,30,n],[28,n,128]]] if arm=='native'else \
        [[[7,30,128],[7,128,n],[7,30,n]]]*4+[[[7,30,n],[7,n,128],[7,30,128]]]*4
    equal([e['args']['Input Dims']for e in bmm],expected,'actual BMM operand shapes')
    gemms={k:v for k,v in kernels.items()if 'sgemm' in k.lower()}
    equal(sum(gemms.values()),2 if arm=='native'else 8,'actual CUDA SGEMM kernel count')
    recorded=Counter({e['name']:e['count']for e in profile['cuda_kernels']})
    equal(kernels,recorded,'summary vs raw CUDA kernels')
    for name in ('aten::clone','aten::_to_copy','aten::bmm'):
        equal(sum(e['count']for e in profile['operators']if e['name']==name),
              sum(e['name']==name for e in ops),'summary vs raw operator '+name)
    return dict(path=str(path.resolve()),sha256=sha(path),cpu_operator_events=len(ops),
        cuda_kernel_count=sum(kernels.values()),repeated_kv_clone_count=repeat_clones,
        kv_cast_count=kv_casts,kv_cast_shape=[1,28 if arm=='native'else 4,n,128],
        bmm_count=len(bmm),sgemm_kernels=gemms,
        separately_named_pack_kernels={k:v for k,v in kernels.items()if 'pack' in k.lower()},
        kernel_counts=dict(kernels),internal_library_pack_absence_proven=False)


def audit(directory):
    report=json.loads((directory/'summary.json').read_text())
    equal(report['status'],'completed','probe completion')
    equal(report['failures'],[],'no probe failure')
    equal((report['args']['warmup'],report['args']['repeats']),(2,5),'formal sampling protocol')
    for key,value in dict(actual_dram_measured=False,fallback_allowed=False,allow_tf32=False,
        float32_matmul_precision='highest',reduced_precision_math_sdp=False).items():
        equal(report[key],value,'probe '+key)
    equal(len(report['samples']),3,'three captures')
    equal([r['metadata']['tensors']['key']['shape'][2]for r in report['samples']],[2089,2322,2564],'captured lengths')
    for name,value in report['source_hashes'].items():equal(sha(HERE/name),value,'executed source '+name)
    rows=[]
    for index,row in enumerate(report['samples']):
        equal(row['status'],'completed','capture completion')
        meta=row['metadata'];n=meta['tensors']['key']['shape'][2]
        equal(meta['layer'],0,'captured layer')
        equal(meta['layer_call_index'],(0,80,163)[index],'captured call index')
        equal(meta['tensors']['query']['shape'],[1,28,30,128],'query shape')
        equal(meta['tensors']['key']['shape'],[1,4,n,128],'unique key shape')
        equal(meta['tensors']['value']['shape'],[1,4,n,128],'unique value shape')
        equal(meta['tensors']['mask']['shape'],[1,1,30,n],'original tree mask shape')
        equal(row['source_stride_restored'],True,'source stride replay')
        equal(row['original_mask_unchanged'],True,'original mask preservation')
        exact(row['native_replay'],'native initial output')
        exact(row['candidate_replay'],'candidate initial output')
        inter=row['intermediates']
        exact(inter['output'],'diagnostic candidate output')
        exact(inter['dense_order_vs_saved_native'],'diagnostic dense reference output')
        equal(set(inter['stages']),{'q_scaled','qk_scores','masked_scores','probabilities','pv_fp32'},'five intermediate stages')
        for name,error in inter['stages'].items():exact(error,'diagnostic '+name)
        receipt=row['storage_receipt']
        for key,value in dict(explicit_repeat_kv=False,qk_bmm_calls=4,pv_bmm_calls=4,extra_bmm_calls=6,
            mask_same_object=True,mask_values_transformed=False,rotary_reapplied=False,
            debug_intermediates=False,debug_extra_copy_bytes=0,native_unused_probability_cast_preserved=True,
            allow_tf32=False,reduced_precision_math_sdp=False).items():equal(receipt[key],value,'receipt '+key)
        equal(len(receipt['groups']),4,'four unique KV heads')
        key_pointers=set();value_pointers=set()
        for group,g in enumerate(receipt['groups']):
            equal(g['query_heads'],[7*group,7*(group+1)],'group-to-query mapping')
            equal(g['key_transposed_expanded']['stride'],[0,1,128],'shared K stride')
            equal(g['value_expanded']['stride'],[0,128,1],'shared V stride')
            equal(g['key_view_shares_unique_storage'],True,'K alias')
            equal(g['value_view_shares_unique_storage'],True,'V alias')
            key_pointers.add(g['key_transposed_expanded']['storage_ptr'])
            value_pointers.add(g['value_expanded']['storage_ptr'])
        equal(len(key_pointers),1,'all K groups share unique bank backing')
        equal(len(value_pointers),1,'all V groups share unique bank backing')
        pairs=row['ordinary']['pairs'];equal(len(pairs),5,'all formal pairs')
        for i,pair in enumerate(pairs):
            equal(pair['repeat'],i,'formal repeat index')
            equal(pair['order'],['native','candidate']if(i+index)%2==0 else['candidate','native'],'alternating order')
            equal(pair['native']['baseline_allocated_bytes'],pair['candidate']['baseline_allocated_bytes'],'equal resident fixture')
            for arm in ('native','candidate'):
                m=pair[arm];exact(m['error'],arm+' formal output')
                equal(m['peak_allocated_bytes']-m['baseline_allocated_bytes'],m['incremental_peak_allocated_bytes'],'allocated peak subtraction')
                equal(m['output_bytes'],215040,'HF output payload')
                require(m['wall_ms']>0,'positive ordinary wall')
        wall={arm:median(p[arm]['wall_ms']for p in pairs)for arm in ('native','candidate')}
        peak={arm:median(p[arm]['incremental_peak_allocated_bytes']for p in pairs)for arm in ('native','candidate')}
        equal(wall,row['ordinary']['median_wall_ms'],'ordinary wall medians')
        equal(peak,row['ordinary']['median_incremental_peak_allocated_bytes'],'ordinary peak medians')
        traces={arm:trace_evidence(directory/row['profiles'][arm]['trace_file'],n,arm,row['profiles'][arm])for arm in ('native','candidate')}
        rows.append(dict(file=row['file'],capture_sha256=row['sha256'],layer=0,layer_call_index=meta['layer_call_index'],key_tokens=n,
            all_initial_and_five_formal_pairs_byte_exact=True,five_intermediate_stages_byte_exact=True,
            ordinary_wall_ms=wall,ordinary_incremental_peak_allocated_bytes=peak,
            incremental_peak_reduction_percent=100*(1-peak['candidate']/peak['native']),
            ratio_of_median_wall_candidate_over_native=wall['candidate']/wall['native'],
            ordinary_pairs=pairs,raw_trace_evidence=traces,
            stage_dump_stored_on_server=inter['saved_file'],stage_dump_reloaded_in_offline_audit=False,
            logical_component_only=receipt['logical_copies']))
    exit_path=directory.with_name(directory.name+'.exit.json')
    exit_receipt=json.loads(exit_path.read_text());equal(exit_receipt['returncode'],0,'terminal child exit')
    hw=report['hardware']
    equal(hw['allowed_pids'],[report['pid']],'sampler own PID exclusion')
    return dict(validated=True,run=str(directory.resolve()),source_summary_sha256=sha(directory/'summary.json'),
        source_hashes=report['source_hashes'],exit_receipt=exit_receipt,samples=rows,
        hardware={k:hw.get(k)for k in ('foreign_processes','process_visibility_complete','process_query_failures','sharing','energy_attribution','measurement_duration_s')},
        supports_bounded_full_generation=True,whole_model_peak_or_dram_claim=False,
        all_layer_bitwise_equivalence_proven=False,internal_blas_pack_absence_proven=False,
        numerical_scope='Recorded bytewise metrics and actual trace receipts reviewed; no raw remote intermediate tensor files were reloaded offline.',
        recommendation='One bounded natural-EOS full-generation test is supported; keep prefill/Draft/native mask and positions unchanged, verify all 28 Target verification layers are covered, preserve any output/trajectory differences. Module slowdown is retained, not converted into end-to-end throughput.')


def markdown(r):
    lines=['# Stride-zero GQA 实测模块独立审计','',
        '**支持进入一次有界完整生成验证。** 三个真实capture的原生重放和候选输出均逐字节一致；每个capture的5对正式测量也全部finite且逐字节一致。单层模块增量峰值明显下降，但模块wall约为原生2.2–2.3倍。','',
        '| layer/call | K长度 | native增量峰值 MiB | candidate增量峰值 MiB | 峰值减少 | native/candidate wall中位ms | wall倍数 |','|---|---:|---:|---:|---:|---:|---:|']
    for s in r['samples']:
        p=s['ordinary_incremental_peak_allocated_bytes'];w=s['ordinary_wall_ms']
        lines.append(f"| {s['layer']}/{s['layer_call_index']} | {s['key_tokens']} | {p['native']/2**20:.6f} | {p['candidate']/2**20:.6f} | {s['incremental_peak_reduction_percent']:.4f}% | {w['native']:.6f}/{w['candidate']:.6f} | {s['ratio_of_median_wall_candidate_over_native']:.4f}× |")
    lines+=['','## 数值与内存计量','',
        '- 三个capture均来自Target layer0，调用0/80/163；不是三个不同层，也未覆盖全部28层。Q均为[1,28,30,128]，unique K/V为[1,4,N,128]。',
        '- q_scaled、QK、masked_scores、softmax概率、PV五个诊断阶段各3/3记录为逐字节一致，dense-order诊断输出也复现原生输出。没有使用容差通过。',
        '- 全部15对正式测量在每对内baseline allocated相等；输出与数值检查在wall/峰值区间之后，debug/profiler另做。峰值是相同输入和参考已驻留后的PyTorch增量allocated，含输出；不是完整模型峰值、驻留KV压缩或DRAM。',
        '- 本离线审核核对summary数值回执、比较函数的uint8逐字节实现与六份原始trace。远端intermediates.pt未下载，未在本地重新计算这些数组差值。','',
        '## 六份trace的直接证据','',
        '- native每次有2个[1,4,7,N,128]的aten::clone及对应copy；candidate没有这两个七份KV物化。两项KV的_to_copy形状从[1,28,N,128]变为[1,4,N,128]。',
        '- candidate四个head组的K/V都是batch stride=0，并分别共享同一个4头bank backing。mask原值、原对象和已旋转位置保留；未改变FP32 math精度，TF32关闭。',
        '- native每次2个SGEMM kernel，candidate为4个QK+4个PV，共8个；三组总CUDA kernel数均18→22。两边保留最后HF布局clone和原生未使用概率的FP16转换，未将它们的省略算进候选收益。',
        '- raw trace未出现单独命名的packing kernel，也未出现候选七份KV的ATen clone。这支持“显式大张量物化消失”；不能排除库内部打包、workspace或kernel内部重复读取，不能宣称K/V仅从DRAM读一次。','',
        '## 背景与下一步边界','',
        '- sampler记录的唯一foreign PID为6360（约4.45MiB常驻图形进程）；自身PID2792717被排除。process_visibility_complete=true、查询失败0，energy_attribution=unavailable_shared_device。采样仍不能证明严格独占。',
        '- 保存的进程约14秒墙钟含启动/文件等；sampler约2.52秒包含普通重放、profiler及CPU保存。整段能量不归因到某个attention调用。',
        '- 新实现的主要attention MAC和常驻KV容量未减少；这是当前HF math重复物化的执行改进，不是新的GQA模型方法。',
        '- 可以做完整自然EOS pilot，核28层×实际验证调用覆盖、输出ID、完整树路径、接受/提交计数及完整请求峰值。layer0三个长度逐位一致不能替代全层/全长度验证。模块变慢记录保留；按当前硬件目标，它不单独否决容量/搬运方向。',
        '- 本次没有NCU DRAM计数，也不推出整机峰值减少76–77%或端到端加速。','']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=BASE/'stride0_gqa_capture_r1')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv);r=audit(args.run);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'STRIDE0_GQA_MODULE_AUDIT.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    (args.out/'STRIDE0_GQA_MODULE_AUDIT.md').write_text(markdown(r))
    print(json.dumps(dict(validated=True,samples=[{k:s[k]for k in ('key_tokens','incremental_peak_reduction_percent','ratio_of_median_wall_candidate_over_native')}for s in r['samples']]),indent=2))


if __name__=='__main__':main()
