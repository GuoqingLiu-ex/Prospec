"""Offline fixed14 overlay accounting: actual trajectories, major MACs and payload.

No GPU/model/remote action. Does not change old accounting or execution modules.
All source-call/ideal-byte counts are separate from hardware traffic and timing.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import account_v_bank_trajectory as old
from account_request_macs import count_request, require, equal

ROOT,BASE=old.ROOT,old.BASE
L,H,HK,D=28,28,4,128
UNITS=L*HK
K=14
VARIANT='full_k_v8_physical_head_overlay14_grouped_math'
ARM='memory_shared_embedding_compiled_fullk_v_head_overlay14_grouped_math'
METHOD_COUNT='target_head_overlay_attention_calls'


def trees(row):return [t for t in row['result']['rounds'] if t['kind']=='tree']


def normalized_counts(row):
    counts=row['result']['forward_counts']
    equal(set(counts),old.v_audit.COMMON_COUNT_FIELDS|{METHOD_COUNT},'overlay exact counter fields')
    copy=dict(row,result=dict(row['result'],forward_counts=dict(counts)))
    copy['result']['forward_counts']['target_v_bank_attention_calls']=copy['result']['forward_counts'].pop(METHOD_COUNT)
    require(old.v_audit.count_contract(copy,'int8_visual')['passed'],'overlay counter values')
    return {k:v for k,v in counts.items() if k!=METHOD_COUNT}


def overlay_ledger(P,M,rounds,budget=K):
    equal(budget,K,'only frozen fixed14 is supported')
    require(0<M<P,'visual and nonvisual prompt partition')
    out=old.v_audit.logical_io_from_rounds(P,M,'int8_visual',rounds)
    old.add(out,'score_slab_zero',1,0,UNITS*4)
    copy_bytes=K*M*D*2
    old.add(out,'original_head_overlay_copy',L,copy_bytes,copy_bytes)
    out['value_operand_fill']=dict(calls=0,read_bytes=0,write_bytes=0,read_plus_write_bytes=0)
    for t in rounds:
        S=t['target_cache_before']+t['tree_nodes']
        require(S>=P,'attention cannot omit prompt history')
        read=K*M*D*2+(UNITS-K)*M*(D+4*4)+UNITS*(S-M)*D*2
        old.add(out,'value_operand_fill',L,read,UNITS*S*D*4)
        old.add(out,'overlay_fill_route_logical_reads',L,L*(4*P*4+4*((S+7)//8)*4),0)
    return out


def overlay_io(row):
    info=row['adaptor_preparation']['head_overlay'];P=row['result']['prompt_tokens']
    M=info['partition']['visual_tokens'];tt=trees(row)
    ledger=overlay_ledger(P,M,tt,info['overlay_budget_heads'])
    equal(info['logical_io'],ledger,'recorded overlay ledger vs actual trajectory')
    equal(info['counterfactual_protected_V8_scale_read_bytes_avoided'],K*M*(D+16)*len(tt),
          'avoided duplicate representation component')
    # Reuse only common attention-copy formulas, not a false claim this is native.
    common=old.known_io(row,'grouped_control')['common_selected_attention_copies']
    maintenance=[n for n in ledger if n.endswith('_cat') or n.endswith('_compact')]
    extras=[n for n in ledger if n not in maintenance and n!='value_operand_fill']
    def total(names):return {f:sum(ledger[n][f] for n in names) for f in ('calls','read_bytes','write_bytes','read_plus_write_bytes')}
    return dict(ledger=ledger,kv_maintenance=total(maintenance),value_fp32_operand=ledger['value_operand_fill'],
        other_recorded_policy_payload=total(extras),common_selected_attention_copies=common,
        fill_visual_breakdown=dict(protected_head_units=K,unprotected_head_units=UNITS-K,
            protected_original_fp16_read_bytes=K*M*D*2*len(tt),
            unprotected_V8_codes_read_bytes=(UNITS-K)*M*D*len(tt),
            unprotected_scales_read_bytes=(UNITS-K)*M*4*4*len(tt),
            text_fp16_read_bytes=sum(UNITS*(t['target_cache_before']+t['tree_nodes']-M)*D*2 for t in tt),
            counterfactual_duplicate_V8_scale_component_not_read=info['counterfactual_protected_V8_scale_read_bytes_avoided']),
        scope='Selected source-operation ideal payload. Calls are not kernel/CUDA memcpy/DRAM commands; scale reuse and flags are not physical traffic counts.')


def storage(info,P,M,S,name):
    s=info[name];require(s.get('partial') is not True,'incomplete storage receipt')
    equal(s['layer_count'],L,'all installed layers');equal(len(s['layers']),L,'layer snapshots')
    shapes={'keys':([1,4,S,128],'torch.float16',2),'text_values':([1,4,S-M,128],'torch.float16',2),
        'visual_values':([1,4,M,128],'torch.int8',1),'scales':([1,4,M,4],'torch.float32',4),
        'error_flag':([],'torch.int32',4)}
    total=0
    for layer in s['layers']:
        equal(set(layer),set(shapes),'storage tensor set')
        for key,(shape,dtype,element) in shapes.items():
            n=1
            for x in shape:n*=x
            equal(layer[key],dict(shape=shape,dtype=dtype,logical_bytes=n*element,backing_bytes=n*element),key+' backing')
            total+=n*element
    overlay=K*M*D*2+UNITS*4
    equal(s['unique_layer_backing_bytes'],total,'base full V8 retained')
    equal(s['shared_device_route_bytes'],P*4,'shared prefix route')
    equal(s['unique_overlay_head_route_backing_bytes'],overlay,'physical overlay+head route')
    equal(s['owned_plus_shared_device_bytes'],total+P*4+overlay,'known capacity is not allocator peak')
    for key in ('original_value_tensor_refs_alive','transient_score_slab_known_backing_bytes','transient_retained_original_V_known_backing_bytes'):
        equal(s[key],0,'released '+key)
    equal(s['original_full_V_known_backing_retained'],False,'original full V release')
    o=s['overlay_storage']
    for key,value in dict(budget_heads=K,complete_V8_codes_bytes=UNITS*M*D,complete_V8_scales_bytes=UNITS*M*16,
            original_FP16_overlay_bytes=K*M*D*2,head_route_bytes=UNITS*4,completed_layer_copies=L,
            owns_original_source=False,owns_score_or_sort_buffers=False,owns_FP32_operand=False,validated=True,failed=False).items():
        equal(o[key],value,'overlay '+key)
    return dict(known_owned_plus_shared_bytes=total+P*4+overlay,full_V8_base_bytes=UNITS*M*(D+16),
        original_FP16_overlay_bytes=K*M*D*2,head_route_bytes=UNITS*4,original_full_V_retained=False,
        saving_vs_same_shape_V16_bytes=UNITS*M*D*2-UNITS*M*(D+16)-overlay,
        extra_vs_same_shape_V8_bytes=overlay)


def validate_row(row):
    equal(row['arm'],ARM,'overlay arm')
    equal(row['scheme'],dict(depth='fixed0',adaptor='compiled',compression=None,ratio=1.0,v_storage='head_overlay14',
        target_cache_policy=VARIANT,tree_attention='grouped_head_overlay'),'fixed scheme')
    counts=normalized_counts(row);info=row['adaptor_preparation']['head_overlay'];r=row['result'];tt=trees(row);n=len(tt)
    P=r['prompt_tokens'];M=r['visual_context']['original_count']
    for key,value in dict(variant=VARIANT,target_cache_policy=VARIANT,overlay_budget_heads=K,head_pack_variant='validated_row',
            finalized=True,result_coverage_validated=True,failures=[],fallback_calls=0,reference_only=False,
            row_head_pack_calls=L,original_head_copy_calls=L,overlay_ready_status_calls=1,setup_status_calls=1,
            result_status_calls=1,score_zero_initializations=1,original_value_tensor_refs_alive=0,score_tensor_refs_alive=0,
            full_k_native_cat_index_select=True,original_mask_reused=True,position_ids_passed_unchanged=True,
            rotary_reapplied=False,full_fp16_value_reconstruction=False,native_unused_probability_cast_preserved=True,
            source_full_V_stacked=False,all_original_V_retained_until_copy_ready=True,actual_dram_measured=False).items():
        equal(info[key],value,'overlay coverage '+key)
    for key,value in dict(prefill_calls=1,prefill_forward_calls=1,prefill_layer_calls=[1]*L,verify_forward_calls=n,
            verify_query_tokens=sum(t['tree_nodes'] for t in tt),layer_calls=[n]*L,selected_calls=L*n,
            completed_calls=L*n,candidate_calls=L*n,update_calls=L*n,fill_calls=L*n,commit_calls=n).items():
        equal(info[key],value,'actual coverage '+key)
    equal(row['result']['forward_counts'][METHOD_COUNT],L*n,'method attention counter')
    equal(counts['target_tree_attention_policy_calls'],L*n,'shared attention counter')
    equal(info['precision'],old.v_audit.FLAGS,'actual math flags')
    equal(r['precision']['target_tree_math_precision'],old.v_audit.FLAGS,'result math flags')
    equal(r['precision']['target_verify_attention_backend'],'grouped_two_bmm_head_overlay_fp32_math','backend')
    equal(r['precision']['tree_attention_discarded_probability_cast_removed'],False,'original unused cast remains')
    equal(info['partition']['prefix_tokens'],P,'partition P');equal(info['partition']['visual_tokens'],M,'partition M')
    equal(info['partition']['nonvisual_prompt_tokens'],P-M,'nonvisual prompt')
    equal(info['partition']['provenance']['grid_thw']['video'],row['video_grid_thw'],'real visual grid')
    equal(len(info['rounds']),n,'full runtime rounds')
    for a,t in zip(info['rounds'],tt):
        for key,value in dict(cache_before=t['target_cache_before'],tree_nodes=t['tree_nodes'],
                cache_after=t['target_cache_before']+t['tree_nodes'],compact_value_tokens=t['target_cache_before']+t['tree_nodes']-M,
                layer_calls=[1]*L,update_counts=[1]*L,selected_tree_indices=t['selected_tree_indices'],
                cache_after_commit=t['target_cache_after'],compact_value_after_commit=t['target_cache_after']-M).items():
            equal(a[key],value,'round receipt '+key)
    for key in ('setup_status','overlay_ready_status','result_status'):
        equal(info[key]['banks'],L,key+' banks');equal(info[key]['status_valid'],True,key+' success')
    require(old.v_audit.memory_evidence(row,info)['passed'],'phase and full allocator peaks')
    equal(info['storage_at_result'],info['final_storage'],'final storage stable')
    return {key:storage(info,P,M,S,key) for key,S in [('storage_after_conversion',P),('storage_at_result',r['target_cache_length'])]}


def scalar_setup(row):
    info=row['adaptor_preparation']['head_overlay'];M=info['partition']['visual_tokens'];P=info['partition']['prefix_tokens']
    plan=info['row_head_pack_plan'];equal(plan['extra_atomic_max_operations'],4*M,'per-layer row score atomics')
    equal(plan['route_logical_reads_bytes'],4*P*4,'per-layer pack route payload')
    equal(info['overlay_preparation']['stable_descending_GPU_argsort_calls'],1,'ranking source calls')
    return dict(pack_source_calls=L,visual_quantized_elements=UNITS*M*D,quantization_groups32=UNITS*M*4,
        row_score_atomic_max_operations=UNITS*M,score_initialization_write_bytes=UNITS*4,
        score_atomic_naive_read_plus_write_byte_model=UNITS*M*8,
        pack_route_logical_read_bytes=L*4*P*4,existing_flag_zero_write_bytes=L*4,
        GPU_argsort_source_calls=1,sorted_score_elements=UNITS,sort_comparisons=None,
        additional_pack_scans_for_score=0,additional_pack_source_calls_for_score=0,
        visual_dequantized_elements=(UNITS-K)*M*D*len(trees(row)),
        protected_fp16_promoted_elements=K*M*D*len(trees(row)),complete_scalar_FLOPs=None,
        scope='Scalar dimensions/source calls, not instruction or launch counts. Atomic bytes are a naive logical RMW model, not DRAM; not added to payload totals.')


def count_overlay(row,configs):
    snapshots=validate_row(row);io=overlay_io(row)
    target,draft,vision,prov=configs
    equal(row['result']['author_revision'],prov['draft_author_revision'],'author revision')
    equal(row['result']['joint_version'],'vispec_joint_m_budget_v1','joint version')
    equal(row['result']['adapter_version'],'hf517_vispec_tree_v1','adapter version')
    c=count_request(dict(row,arm='deployment_compiled'),target,draft,vision)
    c.pop('observed_e2e_wall_s');c.pop('output_token_ids')
    products=sum(t['tree_nodes']*(t['target_cache_before']+t['tree_nodes']) for t in trees(row))
    equal(c['query_reconstruction']['target_sum_query_times_key'],row['result']['prompt_tokens']**2+products,'actual Target widths')
    return dict(request_id=row['request_id'],repeat=row['repeat'],mode='head_overlay14',num_tokens=row['num_tokens'],
        major=c,actual_target_tree=dict(query_key_products_per_layer=products,attention_macs=2*H*D*L*products,
            layer_attention_calls=L*len(trees(row)),qk_bmm_calls=L*len(trees(row)),pv_bmm_calls=L*len(trees(row))),
        io=io,storage=snapshots,scalar_setup=scalar_setup(row),recorded_draft_key_shapes=None,
        recorded_shape_draft_attention_macs=None,source_reconstructed_draft_attention_macs=c['components_macs']['draft_attention'])


def read_overlay(directory,reference_status,configs):
    path=directory/'result';status=json.loads((path/'status.json').read_text());args=status['args']
    raw=[json.loads(x) for x in (path/'results.jsonl').read_text().splitlines() if x.strip()]
    terminal=json.loads((directory/'launch/wrapper_exit.json').read_text())
    equal(status['status'],'completed','overlay completed');equal(terminal['status'],'completed','wrapper completed')
    equal(terminal['returncode'],0,'wrapper exit');equal(terminal['child_pid'],status['pid'],'actual child')
    equal(Path(terminal['command'][1]).name,'run_v_head_overlay_layout.py','entry')
    equal(len(raw),25,'warm1 plus 24 formal');equal(status['completed_cells'],25,'completed cells');equal(status['total_cells'],25,'total cells')
    for key,value in dict(v_storage='head_overlay14',target_cache_policy=VARIANT,overlay_protected_heads=K,
            overlay_head_units=UNITS,target_history_eviction=False,attention_backend_fallback=False,
            memory_layout='shared_embedding',deployment_adaptor='compiled',actual_dram_measured=False).items():
        equal(status[key],value,'status '+key)
    for key,value in dict(warmup=1,repeats=2,max_new_tokens=512,seed=20261001,resume=False).items():equal(args[key],value,'protocol '+key)
    schedule=json.loads((path/'schedule.json').read_text());requests=json.loads((path/'requests.json').read_text())
    key=lambda r:tuple(r[k] for k in ('request_id','arm','phase','repeat'))
    equal([key(r) for r in raw],[key(r) for r in schedule],'actual schedule')
    require(len({key(r) for r in raw})==25 and len(requests)==12,'unique cohort')
    equal(raw[0]['phase'],'warmup','warmup first');require(all(r['phase']=='measure' for r in raw[1:]),'formal rows')
    equal({(r['request_id'],r['repeat']) for r in raw[1:]},{(r['request_id'],i) for r in requests for i in (0,1)},'12 x 2 requests')
    freeze=json.loads((directory/'source_freeze.json').read_text());equal(freeze,status['source_sha256'],'actual source manifest')
    require(reference_status['source_sha256'].keys()<=status['source_sha256'].keys(),'complete prior frozen package preserved')
    common=reference_status['source_sha256'].keys() & status['source_sha256'].keys()
    require(common,'shared source provenance')
    for name in common:equal(status['source_sha256'][name],reference_status['source_sha256'][name],'shared frozen source '+name)
    require({'run_v_head_overlay_layout.py','v_head_overlay_model_runtime.py','grouped_head_overlay_math.py',
        'v_head_scale_cuda_ops.py','v_head_overlay_cuda_ops.py','v_bank_model_runtime.py',
        'grouped_v_bank_math.py','partitioned_v_bank.py','v_bank_cuda_ops.py','ring_kv_cache.py',
        'run_joint.py'}<=status['head_overlay_source_sha256'].keys(),'all accounting-relevant execution sources')
    for name,sha in status['head_overlay_source_sha256'].items():
        equal(freeze['direct_opt/'+name],sha,'overlay source receipt '+name)
        equal(old.digest(ROOT/'combined_visual_experiment/direct_opt'/name),sha,'locally reviewed source '+name)
    for key in ('target','draft','torch','transformers'):equal(status[key],reference_status[key],'common model/runtime '+key)
    for row in raw:
        count_overlay(row,configs)  # Validate warmup too, exclude it only from totals.
    return raw,dict(source_sha256=status['source_sha256'],common_frozen_files_identical=len(common),
        raw_hashes={str(p.relative_to(directory)):old.digest(p) for p in [path/'results.jsonl',path/'status.json',directory/'source_freeze.json',directory/'launch/wrapper_exit.json']},
        overlay_gate_receipts=status['primitive_gates'],shared_GPU_correctness_only=True)


def run(overlay,reference,configdir,visionpath):
    configs=old.load_configs(configdir,visionpath)
    existing=old.run(reference,configdir,visionpath)
    status=json.loads((reference/'control/status.json').read_text())
    raw,provenance=read_overlay(overlay,status,configs)
    formal={(r['request_id'],r['repeat']):r for r in raw if r['phase']=='measure'}
    counted=[count_overlay(r,configs) for r in formal.values()]
    totals=dict(existing['totals'],head_overlay14=old.totals(counted))
    comparisons={};decomposition={}
    for mode in ('control','fp16_partition','int8_visual'):
        refs={(r['request_id'],r['repeat']):r for line in (reference/mode/'results.jsonl').read_text().splitlines()
              if (r:=json.loads(line))['phase']=='measure'}
        equal(set(refs),set(formal),'matched '+mode+' cohort')
        comparisons[mode]=[]
        for key,b in refs.items():
            c=formal[key]
            for field in ('output_token_ids','text','stop_reason','truncated','num_tokens','input_shapes','video_grid_thw','timestamps'):
                equal(c[field],b[field],'paired '+field)
            equal(normalized_counts(c),old.v_audit.common_counts(b),'paired common counts')
            if mode!='control':
                equal(c['adaptor_preparation']['head_overlay']['partition'],b['adaptor_preparation']['v_bank']['partition'],
                    'same input-ID and grid-derived visual partition')
            changed=[];equal(len(b['result']['rounds']),len(c['result']['rounds']),'complete paired rounds')
            for i,(x,y) in enumerate(zip(b['result']['rounds'],c['result']['rounds'])):
                fields={k:dict(reference=x.get(k),overlay=y.get(k)) for k in x.keys()|y.keys() if x.get(k)!=y.get(k)}
                if fields:changed.append(dict(round_index_zero_based=i,fields=fields))
            comparisons[mode].append(dict(request_id=key[0],repeat=key[1],output_ids_same=True,common_counts_same=True,
                full_rounds_same=b['result']['rounds']==c['result']['rounds'],changed_rounds=changed,
                shape_change_rounds=[d['round_index_zero_based'] for d in changed if set(d['fields'])&{'submitted','target_cache_before','target_cache_after'}]))
        # Counterfactual old layout on OVERLAY's own trajectory, not an extra run.
        cf=[]
        for r in formal.values():
            if mode=='control':cf.append(old.known_io(r,'grouped_control')['ledger'])
            else:cf.append(old.v_audit.logical_io_from_rounds(r['result']['prompt_tokens'],r['result']['visual_context']['original_count'],mode,trees(r)))
        cf=old.sum_ledger(cf);base=totals[mode]['io_ledger'];actual=totals['head_overlay14']['io_ledger']
        details={}
        for a,b in [('key_cat','key_cat'),('compact_value_cat','full_value_cat' if mode=='control' else 'compact_value_cat'),
                ('key_compact','key_compact'),('compact_value_compact','full_value_compact' if mode=='control' else 'compact_value_compact'),
                ('value_operand_fill','value_operand_cast' if mode=='control' else 'value_operand_fill')]:
            details[a]={f:dict(actual_overlay_minus_reference=actual[a][f]-base[b][f],
                layout_on_overlay_trajectory=actual[a][f]-cf[b][f],trajectory_at_reference_layout=cf[b][f]-base[b][f])
                for f in ('calls','read_bytes','write_bytes','read_plus_write_bytes')}
        decomposition[mode]=details
    summary=totals['head_overlay14']
    summary['scalar_setup']={k:sum(r['scalar_setup'][k] for r in counted) for k,v in counted[0]['scalar_setup'].items() if type(v) is int}
    summary['scalar_setup'].update(sort_comparisons=None,complete_scalar_FLOPs=None)
    summary['fill_visual_breakdown']={k:sum(r['io']['fill_visual_breakdown'][k] for r in counted)
        for k in counted[0]['io']['fill_visual_breakdown'] if k.endswith('bytes') or k.endswith('not_read')}
    summary['fixed_head_budget']=dict(protected=K,unprotected=UNITS-K,units=UNITS)
    return dict(schema='head_overlay_actual_trajectory_accounting_v1',formal_requests=24,unique_videos=12,
        totals=totals,overlay_requests=counted,comparisons=comparisons,logical_io_decomposition=decomposition,
        source_and_raw_provenance=dict(overlay=provenance,references=existing['source_and_raw_provenance']),
        config_hashes=existing['config_hashes'],accounting_source_hashes={p.name:old.digest(p) for p in
            [Path(__file__),Path(old.__file__),Path(__file__).with_name('account_request_macs.py'),Path(__file__).with_name('audit_v_bank_generation.py')]},
        scope='Same 12 requests x2 protocol with unchanged common source files, executed in separate shared-GPU processes. No cross-batch timing claim. grouped_control is not original ViSpec.',
        claims=dict(complete_scalar_FLOPs=None,complete_logical_IO=None,actual_DRAM=None,physical_transfer_count=None,timing_improvement=None,
            recorded_draft_key_shapes=None,quality_beyond_same_12_outputs=None),
        exclusions=existing['exclusions']+['Overlay route ranking/check/status temporaries, error flags/atomics, scale-lane rereads and unrecorded copies are not a complete IO inventory.'],
        formulas=dict(target_tree_MAC='sum 2*28*28*128*T*(base+T)',overlay_fill_read='14*M*128*2+98*M*(128+16)+112*(S-M)*128*2',
            overlay_fill_write='112*S*128*4',original_overlay_copy_each_direction='14*M*128*2',
            complete_visual_backing='112*M*(128+16)+14*M*128*2+112*4',draft_attention='source reconstructed, not per-call shape telemetry'))


def markdown(r):
    t=r['totals'];o=t['head_overlay14'];lines=['# Head-overlay14：实际逐轮计算与逻辑读写','',
        '24 次正式请求＝12 视频×2；各臂预热不计。control 是共享 embedding、compiled adaptor、grouped FP32 attention 对照，**不是原版 ViSpec**。同问题/配置/共同源码，分进程执行；共享 GPU 不作跨批速度或能耗结论。','',
        '## 主要矩阵 MAC','',
        '| 臂 | tokens / verify | 主要 TMAC | 相对 control 的 MAC 差 | 完整轨迹与 overlay 相同 |','|---|---:|---:|---:|---:|']
    for name,a in t.items():
        same='—' if name=='head_overlay14' else str(sum(p['full_rounds_same'] for p in r['comparisons'][name]))+'/24'
        lines.append(f"| {name} | {a['output_tokens']} / {a['verify_rounds']} | {a['total_major_macs']/1e12:.12f} | {a['total_major_macs']-t['control']['total_major_macs']:+,} | {same} |")
    lines+=['','全部输出 IDs/文本/EOS 和公共调用计数一致；局部接受路径与提交边界仍可改变每轮 K 宽度。只改变节点编号而不改变提交数/前缀长度，不改变本 dense 矩阵账。','',
        '| 组件 | control MAC | overlay−control MAC |','|---|---:|---:|']
    for k,v in t['control']['components_macs'].items():lines.append(f"| {k} | {v:,} | {o['components_macs'][k]-v:+,} |")
    lines+=['','Draft attention 按冻结作者源码和实际提交轨迹重建；没有逐调用 Draft K 形状遥测，实际 Draft cache 流量留空。主要 MAC 不是完整标量 FLOPs，也不是等成本芯片操作。','',
        '| overlay 相对 control 的形状变化题目 | repeat | 轮（0 起） |','|---|---:|---|']
    for p in r['comparisons']['control']:
        if p['shape_change_rounds']:lines.append(f"| {p['request_id']} | {p['repeat']} | {p['shape_change_rounds']} |")
    lines+=['','## 选定逻辑读写与源调用','',
        '| 臂 | KV cat+compact GiB / 源调用 | V32 operand GiB / 源调用 |','|---|---:|---:|']
    for name,a in t.items():
        k,v=a['kv_maintenance'],a['value_fp32_operand'];lines.append(f"| {name} | {k['read_plus_write_bytes']/2**30:.9f} / {k['calls']} | {v['read_plus_write_bytes']/2**30:.9f} / {v['calls']} |")
    lines+=['','每次 fill 明确读取 **14 个原 FP16 头＋98 个 V8/scale 头＋全部非视觉 FP16 V**，再写一份四唯一 KV 头 FP32 operand。保护的 14 头不再读取其 V8 表示，但完整 112 头 V8 base 仍驻留。','',
        '| overlay 单列账目（24 请求合计） | 源调用/账目次数 | 逻辑读 bytes | 逻辑写 bytes |','|---|---:|---:|---:|']
    for name,a in o['io_ledger'].items():lines.append(f"| {name} | {a['calls']} | {a['read_bytes']:,} | {a['write_bytes']:,} |")
    lines+=['','GiB=2³⁰ bytes。这里调用为冻结源码算子/账目次数：guard 一次包含多个操作，不能求和冒充 CUDA kernel、memcpy 或 DRAM request/sector/burst。原生 control 没有候选 guard ledger，空缺不代表没有原生 metadata 工作。',
        '原 V 在全 28 层 overlay copy-ready 后释放，结果处弱引用为 0；仍须承担 prefill→转换重叠和排序/状态同步。JSON 单列已知持久 backing 与 allocator 峰值校验，不将逻辑 payload 等同显存峰值。','',
        '## 标量/选择开销（不并入矩阵 MAC）','']
    s=o['scalar_setup']
    lines += [f"- 量化视觉元素 {s['visual_quantized_elements']:,}，32-channel groups {s['quantization_groups32']:,}；原 row-head score atomic-max {s['row_score_atomic_max_operations']:,} 次。",
        f"- stable GPU argsort 源调用 {s['GPU_argsort_source_calls']} 次，每请求排序 112 个 score；实际比较/指令/kernel 数未知。",
        f"- score atomic 的朴素逻辑 RMW 模型 {s['score_atomic_naive_read_plus_write_byte_model']:,} bytes，pack route 源逻辑读取 {s['pack_route_logical_read_bytes']:,} bytes，均独立列示且不加入所谓净 DRAM。",
        '- 112 头完整 V8 base 都先量化，包含以后受保护的 14 头；量化量相对原 V8 基线没有减少。score 在已有 pack 的寄存器结果上计算，没有额外扫描原 V，但存在归约、原子竞争、排序和状态检查成本。','',
        '## 分解与边界','',
        'JSON 分解每个对照的变化：固定 overlay 实际轨迹仅换布局的 payload 差，再加固定对照布局下的轨迹差。两项严格相加等于实际差；这是反事实公式，不是新运行。',
        'K32 cast、Q pack、仍保留的未用 FP16 probability cast 已单列；权重/矩阵操作数重复读取、softmax/score、route/scale lane 复读、flags/atomic 和部分临时分配未完整统计。',
        '**没有实际 DRAM、物理搬运次数或速度新结论。** 本报告不把旧原版 ABBA 与本 cohort 拼成新的原版对照，也不把这些 12 个视频等同跨数据集质量证明。','']
    selected=[x['storage']['storage_at_result'] for x in r['overlay_requests']]
    low=lambda key:min(x[key] for x in selected)
    high=lambda key:max(x[key] for x in selected)
    lines+=['## 持久容量与填充成本的交换','',
        f"- 相同最终 K/文本长度下，overlay 已知 backing 比 V16 少 {low('saving_vs_same_shape_V16_bytes'):,}–{high('saving_vs_same_shape_V16_bytes'):,} bytes；比完整 V8 多 {low('extra_vs_same_shape_V8_bytes'):,}–{high('extra_vs_same_shape_V8_bytes'):,} bytes。包含新增 head route 的 448 bytes，不包含 allocator rounding。",
        '- 固定相同轨迹，每次 fill 相对 V8 多读 `14*M*112` bytes，相对 V16 少读 `98*M*112` bytes；FP32 operand 写入大小相同。上述实际合计另含各自轨迹的微小宽度差。',
        '- 源 V 释放是对象/backing 观察；全请求峰值可能仍由原生 prefill 主导，不能用这项 KV 容量差替代整模型峰值结果。','']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    home=BASE/'memory_layout_audit'
    p.add_argument('--overlay',type=Path,default=home/'v_head_overlay_model_shared_gpu3_vdc12_r1')
    p.add_argument('--reference',type=Path,default=home/'v_bank_model_shared_gpu3_vdc12_r1')
    p.add_argument('--config-dir',type=Path,default=BASE/'full_request_mac_accounting')
    p.add_argument('--vision-config',type=Path,default=ROOT/'combined_visual_experiment/vision_config.json')
    p.add_argument('--out',type=Path,default=home/'head_overlay_trajectory_accounting')
    a=p.parse_args(argv);r=run(a.overlay,a.reference,a.config_dir,a.vision_config)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    (a.out/'REPORT.md').write_text(markdown(r))
    print(json.dumps({k:{f:v[f] for f in ('formal_requests','output_tokens','verify_rounds','total_major_macs')} for k,v in r['totals'].items()},indent=2))


if __name__=='__main__':main()
