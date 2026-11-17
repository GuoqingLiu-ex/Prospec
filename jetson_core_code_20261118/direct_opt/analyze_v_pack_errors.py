"""CPU-only distributions and fixed-budget hypothetical V protection.

Does not read answers or choose a winning policy. Index/alignment are an explicit
proposed layout budget, not measured cache allocation, DRAM or model quality.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import torch

FRACTIONS=(Fraction(0),Fraction(1,8),Fraction(1,4),Fraction(1,2),Fraction(1))
LAYERS,HEADS,DIM,BLOCK,ALIGN=28,4,128,8,128


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_artifact(data):
    if not isinstance(data,dict) or set(data)!={'errors_fp32','metadata'}:raise ValueError('diagnostic tensor/metadata artifact required')
    e,meta=data['errors_fp32'],data['metadata']
    if (not isinstance(e,torch.Tensor) or e.device.type!='cpu' or e.dtype!=torch.float32
            or e.layout!=torch.strided or e.ndim!=3 or e.shape[:2]!=(LAYERS,HEADS) or e.shape[2]<1
            or not e.is_contiguous() or e.requires_grad or not bool(torch.isfinite(e).all()) or bool((e<0).any())):
        raise ValueError('finite nonnegative contiguous CPU FP32 [28,4,M] required')
    if (meta.get('schema')!='model_visual_v_pack_error_v1' or meta.get('diagnostic_mode')!='fixed_int8_pack_error'
            or meta.get('collection_complete') is not True or meta.get('pack_calls')!=28
            or meta.get('aggregate_status_calls')!=1 or meta.get('d2h_calls')!=1
            or meta.get('GPU_slab_released') is not True or meta.get('failures')
            or meta.get('shape')!=list(e.shape) or meta.get('layer_indices')!=list(range(28))
            or meta.get('kv_head_indices')!=list(range(4)) or type(meta.get('request_successful')) is not bool):
        raise ValueError('completed actual model collection metadata required')
    M=e.shape[2];part=meta.get('partition',{});P=part.get('prefix_tokens');positions=meta.get('visual_row_to_prompt_position')
    if (type(P) is not int or P<M or part.get('visual_tokens')!=M or part.get('nonvisual_prompt_tokens')!=P-M
            or not isinstance(positions,list) or len(positions)!=M
            or any(type(p) is not int or not 0<=p<P for p in positions)
            or positions!=sorted(set(positions))):raise ValueError('invalid visual-to-original prompt positions')
    expected=[];vis={p:i for i,p in enumerate(positions)};text=0
    for p in range(P):
        if p in vis:expected.append(vis[p])
        else:expected.append(-text-1);text+=1
    if meta.get('host_route')!=expected:raise ValueError('host route does not cover original prompt exactly')
    route_hash=hashlib.sha256(json.dumps(tuple(expected)).encode()).hexdigest()
    if part.get('route_sha256')!=route_hash:raise ValueError('route provenance hash mismatch')
    provenance=part.get('provenance',{});input_hash=provenance.get('input_ids_sha256')
    if (not isinstance(input_hash,str) or len(input_hash)!=64 or any(c not in '0123456789abcdef' for c in input_hash)
            or not isinstance(provenance.get('grid_thw'),dict)):
        raise ValueError('actual input ID hash and grid provenance required')
    return e,meta


def describe(values):
    values=torch.as_tensor(values,dtype=torch.float64).reshape(-1)
    if not values.numel():return dict(count=0,min=None,mean=None,max=None,p50=None,p90=None,p95=None,p99=None,zero_count=0,sum=0.)
    qs=torch.quantile(values,torch.tensor([.5,.9,.95,.99],dtype=torch.float64)).tolist()
    return dict(count=values.numel(),min=float(values.min()),mean=float(values.mean()),max=float(values.max()),
        p50=qs[0],p90=qs[1],p95=qs[2],p99=qs[3],zero_count=int((values==0).sum()),sum=float(values.sum()))


def rank_rows(values):
    xs=[float(x) for x in values]
    return sorted(range(len(xs)),key=lambda i:(-xs[i],i))


def rank_blocks(values):
    xs=[float(x) for x in values]
    blocks=[dict(index=i//BLOCK,start=i,stop=min(i+BLOCK,len(xs)),score=max(xs[i:i+BLOCK])) for i in range(0,len(xs),BLOCK)]
    return sorted(blocks,key=lambda b:(-b['score'],b['index']))


def allocation(values,fraction,kind):
    if fraction not in FRACTIONS or kind not in ('token','block8'):raise ValueError('only predeclared fractions/rules')
    M=len(values);target=(fraction.numerator*M+fraction.denominator-1)//fraction.denominator
    if kind=='token':selected=rank_rows(values)[:target];block_ids=[]
    else:
        selected=[];block_ids=[]
        for block in rank_blocks(values):
            if len(selected)>=target:break
            selected.extend(range(block['start'],block['stop']));block_ids.append(block['index'])
    return dict(selected_rows=sorted(selected),selected_block_ids=sorted(block_ids),requested_rows=target,
        actual_rows=len(selected),excess_rows=len(selected)-target)


def _align(n):return (n+ALIGN-1)//ALIGN*ALIGN if n else 0


def capacity(M,P,protected,kind):
    """Explicit mixed-layout hypothesis, not the existing runtime's allocation."""
    if (type(M) is not int or type(P) is not int or M<1 or P<M or len(protected)!=LAYERS*HEADS
            or any(type(n) is not int or not 0<=n<=M for n in protected) or kind not in ('token','block8')):
        raise ValueError('invalid capacity inputs')
    rows=LAYERS*HEADS*M;n=sum(protected)
    fp16=256*rows;v8=144*rows;ideal=256*n+144*(rows-n)
    uniform=all(x==0 for x in protected) or all(x==M for x in protected)
    bank_pad=lookup_bytes=lookup_pad=offsets=counts=table_pad=0
    if not uniform:
        bank_pad=sum(_align(256*p)-256*p+_align(128*(M-p))-128*(M-p)
            +_align(16*(M-p))-16*(M-p) for p in protected)
        entries=M if kind=='token' else math.ceil(M/BLOCK)
        lookup_bytes=4*entries*LAYERS*HEADS
        lookup_pad=(_align(4*entries)-4*entries)*LAYERS*HEADS
        offsets=3*8*LAYERS*HEADS;counts=4*LAYERS*HEADS
        table_pad=_align(offsets)-offsets+_align(counts)-counts
    metadata=lookup_bytes+lookup_pad+offsets+counts+table_pad
    layout=ideal+bank_pad+metadata
    prefix_baseline=LAYERS*HEADS*P*DIM*4
    return dict(visual_rows=rows,protected_rows=n,actual_protected_fraction=n/rows,
        baseline_visual_V_fp16_bytes=fp16,all_V8_code_scale_bytes=v8,all_V8_ideal_saving_bytes=fp16-v8,
        mixed_ideal_visual_V_bytes=ideal,ideal_saving_vs_visual_fp16_bytes=fp16-ideal,
        extra_payload_vs_all_V8_bytes=ideal-v8,
        hypothetical_alignment_bytes=ALIGN,bank_padding_bytes=bank_pad,
        lookup_payload_bytes=lookup_bytes,lookup_padding_bytes=lookup_pad,
        bank_offset_table_bytes=offsets,protected_count_table_bytes=counts,table_padding_bytes=table_pad,
        incremental_metadata_total_bytes=metadata,hypothetical_layout_total_visual_bytes=layout,
        hypothetical_saving_vs_visual_fp16_bytes=fp16-layout,
        baseline_full_prefix_KV_payload_bytes=prefix_baseline,
        hypothetical_full_prefix_KV_payload_plus_new_metadata_bytes=prefix_baseline-fp16+layout,
        full_prefix_KV_payload_saving_fraction=(fp16-layout)/prefix_baseline,
        shared_prefix_route_unchanged_bytes=4*P,unchanged_layer_error_flags_bytes=4*LAYERS,
        baseline_fp16_partition_known_prefix_bytes=prefix_baseline+4*P+4*LAYERS,
        hypothetical_known_prefix_with_shared_route_flags_bytes=prefix_baseline-fp16+layout+4*P+4*LAYERS,
        payload_only_field_exclusions='full_prefix_KV_payload fields exclude shared route and layer flags; known_prefix fields add both',
        known_prefix_scope='logical payload plus declared metadata/alignment, not actual tensor backing or allocator resident memory',
        uniform_format_reused_without_mixed_lookup=uniform,
        diagnostic_array_bytes=4*rows,diagnostic_array_included_in_deployment_capacity=False,
        allocator_or_DRAM_measured=False)


def error_after_protection(row,selected,original):
    selected_set=set(selected);remaining=[x for i,x in enumerate(row) if i not in selected_set]
    effective=[0. if i in selected_set else x for i,x in enumerate(row)]
    protected_sum=sum(row[i] for i in selected)
    tails={}
    for key in ('p95','p99'):
        indices=[i for i,x in enumerate(row) if x>original[key]]
        covered=sum(i in selected_set for i in indices)
        tails[key]=dict(rule='strictly above original interpolated quantile',threshold=original[key],
            original_count=len(indices),protected_count=covered,remaining_count=len(indices)-covered,
            covered_fraction=covered/len(indices) if indices else None)
    return dict(unprotected_distribution=describe(remaining),hypothetical_effective_distribution=describe(effective),
        protected_error_sum=protected_sum,protected_error_sum_fraction=protected_sum/original['sum'] if original['sum'] else None,
        original_tail_coverage=tails)


def global_budget_results(e,P):
    """Fixed total quota across all layers/heads; a distinct offline comparator."""
    M=e.shape[-1];flat=e.double().reshape(-1).tolist();original=describe(flat)
    token_order=rank_rows(flat);blocks=[]
    for head_row in range(LAYERS*HEADS):
        for block in rank_blocks(flat[head_row*M:(head_row+1)*M]):
            blocks.append(dict(start=head_row*M+block['start'],stop=head_row*M+block['stop'],score=block['score']))
    blocks.sort(key=lambda b:(-b['score'],b['start']))
    reports=[]
    for kind in ('token','block8'):
        for fraction in FRACTIONS:
            target=(fraction.numerator*len(flat)+fraction.denominator-1)//fraction.denominator
            if kind=='token':selected=token_order[:target]
            else:
                selected=[]
                for block in blocks:
                    if len(selected)>=target:break
                    selected.extend(range(block['start'],block['stop']))
            protected=[0]*(LAYERS*HEADS)
            for index in selected:protected[index//M]+=1
            reports.append(dict(kind=kind,fraction=str(fraction),quota_scope='global_across_layer_head_rows',
                requested_protected_rows=target,actual_protected_rows=len(selected),
                extra_rows_due_to_blocks=len(selected)-target,
                protected_rows_by_layer_head=[protected[i:i+HEADS] for i in range(0,len(protected),HEADS)],
                **error_after_protection(flat,selected,original),capacity=capacity(M,P,protected,kind)))
    return reports


def analyze(data):
    e,meta=validate_artifact(data);M=e.shape[2];P=meta['partition']['prefix_tokens']
    per_head=[];aggregate={(kind,str(f)):dict(protected=[],effective=[],remaining=[],requested=0,actual=0,excess=0)
        for kind in ('token','block8') for f in FRACTIONS}
    for layer in range(LAYERS):
        for head in range(HEADS):
            row=e[layer,head].double().tolist();original=describe(row);rules=[]
            for kind in ('token','block8'):
                for fraction in FRACTIONS:
                    choice=allocation(row,fraction,kind);selected=choice.pop('selected_rows')
                    metrics=error_after_protection(row,selected,original)
                    rules.append(dict(kind=kind,fraction=str(fraction),**choice,**metrics))
                    acc=aggregate[kind,str(fraction)];acc['protected'].append(len(selected))
                    chosen=set(selected);acc['effective'].extend(0. if i in chosen else x for i,x in enumerate(row))
                    acc['remaining'].extend(x for i,x in enumerate(row) if i not in chosen)
                    for name in ('requested','actual','excess'):acc[name]+=choice[name+'_rows']
            per_head.append(dict(layer=layer,kv_head=head,original_distribution=original,rules=rules))
    totals=[]
    for (kind,f),acc in aggregate.items():
        totals.append(dict(kind=kind,fraction=f,requested_protected_rows=acc['requested'],
            actual_protected_rows=acc['actual'],extra_rows_due_to_blocks=acc['excess'],
            unprotected_distribution=describe(acc['remaining']),hypothetical_effective_distribution=describe(acc['effective']),
            capacity=capacity(M,P,acc['protected'],kind)))
    by_position=[dict(visual_row=i,original_prompt_position=p,distribution_across_layer_heads=describe(e[:,:,i]))
        for i,p in enumerate(meta['visual_row_to_prompt_position'])]
    return dict(schema='offline_fixed_budget_v_error_v1',artifact_metadata=meta,
        request_successful=meta['request_successful'],shape=list(e.shape),fractions=[str(f) for f in FRACTIONS],
        arithmetic='FP32 input errors; FP64 descriptive sums and interpolated quantiles',
        rule='primary: per layer/head descending e or block-max; ties by original compact row/block index; ceil fraction*M',
        quota_comparison='global comparator uses ceil fraction*(28*4*M), allowing unequal allocation across layers/heads; not a deployed selector',
        index_alignment_hypothesis=dict(alignment_bytes=ALIGN,
            token_lookup='signed int32 per original compact visual row encodes precision/compact row index',
            block_lookup='signed int32 per original compact 8-row block encodes precision/compact start row; tail uses actual length',
            head_tables='3 int64 offsets + 1 int32 protected count; tables aligned to128B',
            bank_segments='FP16/code/FP32scale head segments individually aligned to128B',
            uniform_endpoints='reuse existing uniform formats; no mixed routing overhead',
            unmeasured=['allocator granularity/reserve','kernel argument handling and lookup traffic',
                'selection/compaction/conversion work','diagnostic collection and model effects']),
        scope='Numerical V-reconstruction errors only. Hypothetical original-FP16 protection, no deployed mixed bank or semantic/task-quality guarantee.',
        actual_DRAM_measured=False,quality_pass=None,selected_winner=None,threshold_fitted=False,
        original_distribution=describe(e),per_layer_head=per_head,per_prompt_position=by_position,capacity_rows=totals,
        global_quota_rows=global_budget_results(e,P),
        M2016_reference=capacity(2016,2016,[0]*112,'token'))


def write_rows(path,e,positions):
    with Path(path).open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['layer','kv_head','visual_row','original_prompt_position','error_fp32',
            'token_descending_rank_0based','block8_index','block_descending_rank_0based'])
        for layer in range(LAYERS):
            for head in range(HEADS):
                row=e[layer,head].tolist();ranks={i:r for r,i in enumerate(rank_rows(row))}
                blocks={b['index']:r for r,b in enumerate(rank_blocks(row))}
                for i,x in enumerate(row):writer.writerow([layer,head,i,positions[i],x,ranks[i],i//BLOCK,blocks[i//BLOCK]])


def markdown(report):
    lines=['# 固定预算 V 误差离线分析','',
        '这是数值误差和假设容量分析；未运行混合精度模型，不能据此判断语义重要性或回答质量。',
        f"输入形状：{report['shape']}；原请求成功：{report['request_successful']}。",'',
        '## 每层、每个 KV head 分别分配相同比例','',
        '| 规则 | 目标保留比例 | 实际保留比例 | 剩余误差 max | 理想节省 MiB | 扣假设索引/对齐后 MiB |','|---|---:|---:|---:|---:|---:|']
    for row in report['capacity_rows']:
        c=row['capacity'];maximum=row['hypothetical_effective_distribution']['max']
        lines.append(f"| {row['kind']} | {row['fraction']} | {c['actual_protected_fraction']:.4f} | {maximum:.8g} | "
            f"{c['ideal_saving_vs_visual_fp16_bytes']/2**20:.6f} | {c['hypothetical_saving_vs_visual_fp16_bytes']/2**20:.6f} |")
    lines += ['', '## 跨全部层/head 的全局固定总配额','',
        '这是另一种离线配额口径：相同总比例可以偏向某些层/head。块边界仍不跨层/head，未据此选择生产策略。','',
        '| 规则 | 目标保留比例 | 实际保留比例 | 剩余误差 max | 理想节省 MiB | 扣假设索引/对齐后 MiB |','|---|---:|---:|---:|---:|---:|']
    for row in report['global_quota_rows']:
        c=row['capacity'];maximum=row['hypothetical_effective_distribution']['max']
        lines.append(f"| {row['kind']} | {row['fraction']} | {c['actual_protected_fraction']:.4f} | {maximum:.8g} | "
            f"{c['ideal_saving_vs_visual_fp16_bytes']/2**20:.6f} | {c['hypothetical_saving_vs_visual_fp16_bytes']/2**20:.6f} |")
    lines += ['', '容量含每组32的 FP32 scale。索引预算假设为 signed-int32 查找表、三个 int64 bank offset 和一个 int32 count；128 B 对齐。f=0/1 复用统一格式。表中结果不是已实现分配，也不是实际 DRAM。',
        '完整 K 和非视觉 V 保留；全文本增长、模型权重、临时 attention operand、allocator reserve 不在此容量表内。诊断数组单列，不能藏入部署节省。',
        'M=2016 的全 V8 理想节省为 25,288,704 B（24.1171875 MiB）。连续8行块只代表存储顺序，不代表视频时间窗口。',
        'e 是 FP32 重建误差，非严格 FP64 上界；sum(e) 覆盖没有注意力或语义权重。未根据答案拟合阈值或选出赢家。',
        '详表在 evidence.json；每层/head/原位置及稳定排名在 per_row.csv。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--expected-sha256')
    args=p.parse_args(argv)
    if args.out.exists():raise FileExistsError('fresh analysis directory required')
    digest=_sha(args.artifact)
    if args.expected_sha256 is not None and digest!=args.expected_sha256:raise ValueError('artifact hash differs from request receipt')
    data=torch.load(args.artifact,map_location='cpu',weights_only=True)
    report=analyze(data)
    report['artifact_source']=dict(path=str(args.artifact.resolve()),sha256=digest,
        expected_hash_verified=args.expected_sha256 is not None)
    report['analyzer_source_sha256']=_sha(__file__)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_rows(args.out/'per_row.csv',data['errors_fp32'],data['metadata']['visual_row_to_prompt_position'])
    (args.out/'REPORT.md').write_text(markdown(report))
    return report


if __name__=='__main__':main()
