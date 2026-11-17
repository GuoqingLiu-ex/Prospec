"""Offline accounting of major convolution/matrix MACs for recorded ViSpec runs.

Never loads a checkpoint/model or executes a GPU workload. Unsupported paths or
inconsistent counters fail instead of receiving an invented zero cost.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "training_free"))
from compute_cost import calculate_vision_cost, cost_from_depth_stats
from factorized_adaptor import adaptor_macs


EXCLUDED = ["normalization", "softmax/logsoftmax", "activation", "elementwise bias/scale",
            "RoPE", "argmax/topk/sort", "embedding lookup", "masks", "KV compaction",
            "tensor copies/casts/layout/concatenation", "gating/routing/CPU work",
            "video decoding and input/output preparation", "memory traffic/DRAM",
            "allocator and kernel-launch costs"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, name):
    require(actual == expected, f"{name}: recorded {actual!r}, reconstructed {expected!r}")


def language_linear_per_query(config):
    d, heads, kv = (config[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads"))
    require(d % heads == 0, "non-integral language head dimension")
    return 2*d*d + 2*d*(d//heads*kv) + 3*d*config["intermediate_size"]


def residual_extra_cost(row, vision):
    """Validate the two audited block28-only arms; keep pointwise/IO separate."""
    modes = {'direct_one_identity':'identity', 'direct_one_reuse':'reuse'}
    if row['arm'] not in modes:
        return None
    mode = modes[row['arm']]
    equal(row['scheme'], dict(depth='direct', adaptor='native', compression=None,
                              ratio=1.0, residual_mode=mode), 'residual native scheme')
    stats = row['stats']
    equal(stats['policy'], 'direct_one_prefix_residual_v1', 'residual policy')
    equal(stats['residual_mode'], mode, 'residual summary mode')
    equal(len(stats['calls']), 1, 'residual visual call count')
    record = stats['calls'][0]
    equal(record['residual_mode'], mode, 'residual record mode')
    equal(record['prefix_layer'], 27, 'residual prefix layer')
    equal(record['skip_layers'], [28,29], 'residual candidate layers')
    equal(record['residual_coefficient'], int(mode=='reuse'), 'residual coefficient')
    actions = record['actions']
    require(all(type(a) is int and a in (0,1) for a in actions), 'residual actions must be only 0/1')
    original = record['original_gate_actions']
    require(all(type(a) is int and a in (0,1,2) for a in original), 'invalid original gate actions')
    equal(actions, [int(a>0) for a in original], 'clamped residual actions')
    t,h,w = row['video_grid_thw'][0]
    equal(len(actions), t, 'residual action count')
    n = t*h*w
    selected_times = [i for i,a in enumerate(actions) if a]
    selected_rows = len(selected_times)*h*w
    equal(record['selected_time_blocks'], selected_times, 'selected temporal blocks')
    equal(record['selected_rows'], selected_rows, 'selected residual rows')
    equal(record['executed_tokens_per_layer'], [n-selected_rows if i==28 else n for i in range(vision['depth'])],
          'residual per-layer execution (28 skip / 29 full)')
    equal(record['protect_first'], True, 'residual protected first block')
    equal(record['anchor_stride'], 4, 'residual anchor stride')
    require(all(actions[i]==0 for i in range(0,t,4)), 'residual anchor incorrectly skipped')
    d, element_bytes = vision['hidden_size'], 2  # Body FP16 is checked by count_request.
    elements = selected_rows*d if mode=='reuse' else 0
    materialized = elements*element_bytes if mode=='reuse' and len(selected_times)>1 else 0
    expected = dict(residual_elements=elements, residual_elementwise_ops=2*elements,
                    retained_prefix_logical_bytes=n*d*element_bytes if mode=='reuse' else 0,
                    selected_prefix_materialized_bytes=materialized,
                    delta_temporary_bytes=elements*element_bytes,
                    residual_logical_io_bytes=6*elements*element_bytes+2*materialized,
                    residual_applied=bool(elements))
    for key,value in expected.items():
        equal(record[key], value, 'residual '+key)
    require(record['retained_prefix_storage_bytes'] >= expected['retained_prefix_logical_bytes'],
            'retained prefix storage smaller than its logical tensor')
    if mode=='identity':
        equal(record['retained_prefix_storage_bytes'], 0, 'identity retained storage')
    return dict(mode=mode, selected_rows=selected_rows, selected_time_blocks=selected_times,
                **expected, retained_prefix_storage_bytes=record['retained_prefix_storage_bytes'],
                matrix_macs=0, scope='Additional residual subtract/add and selected-prefix cat only; no MAC or DRAM attribution. Shared routing/gating not included.')


def count_request(row, target, draft, vision):
    """Count the supported complete single-video, fixed-tree-budget execution."""
    require(row['arm'] in {'ar_reference','baseline','frozen_bmm','native_direct','frozen_direct',
                          'deployment_native','deployment_retained_bmm','deployment_compiled',
                          'direct_one_identity','direct_one_reuse'},
            "unreviewed arm/fusion implementation; add its actual matrix path before accounting")
    preparation = row.get('adaptor_preparation') or {}
    require(preparation.get('fusion_mode') in (None,'native'), "modified fusion requires its own matrix accounting")
    result, shapes, stats = row["result"], row["input_shapes"], row["stats"]
    counts, grid = result["forward_counts"], row["video_grid_thw"]
    equal(result["precision"]["target"], "float16", "Target body precision")
    equal(result["precision"]["draft"], "float16", "Draft body precision")
    equal(result["precision"]["output_head"], "float32", "shared output-head precision")
    require(len(grid) == 1, "only recorded single-video requests are supported")
    t, h, w = grid[0]
    n = t*h*w
    merge = vision["spatial_merge_size"]
    require(h % merge == 0 and w % merge == 0, "invalid merged visual grid")
    visual_tokens = n//merge**2
    p = result["prompt_tokens"]
    equal(shapes["input_ids"], [1, p], "prompt input shape")
    equal(shapes["attention_mask"], [1, p], "attention mask shape")
    equal(shapes["mm_token_type_ids"], [1, p], "multimodal type shape")
    equal(shapes["video_grid_thw"], [1, 3], "video grid shape")
    patch_width = vision["in_channels"]*vision["temporal_patch_size"]*vision["patch_size"]**2
    equal(shapes["pixel_values_videos"], [n, patch_width], "patch input shape")
    equal(counts["target_visual_calls"], 1, "visual forward count")
    equal(counts["target_prefill_calls"], 1, "Target prefill calls")
    equal(counts["target_prefill_query_tokens"], p, "Target prefill queries")
    equal(result["token_ids"], row["output_token_ids"], "output IDs")
    equal(result["output_length"], len(result["token_ids"]), "result output length")
    equal(row["num_tokens"], result["output_length"], "row output length")
    require(row["stop_reason"] == result["stop_reason"] == "eos" and not row["truncated"],
            "this accounting report requires natural EOS; limit paths need separate validation")
    require(result["token_ids"][-1] in (151645, 151643), "missing recorded EOS")
    require(not any(x in (151645, 151643) for x in result["token_ids"][:-1]), "early EOS in result")
    require(row["scheme"]["compression"] is None and row["scheme"]["ratio"] == 1.0,
            "token-merging paths require additional accounting")
    require(not row.get("capture_enabled"), "instrumented capture is not a normal request")
    require(not row.get("graph_setup"), "graph capture would add extra request computation")
    if stats["calls"]:
        equal(len(stats["calls"]), 1, "depth records")
        equal(stats["calls"][0]["grid_thw"], grid, "depth grid")
        visual = cost_from_depth_stats(stats, vision)["calls"][0]
    else:
        require(stats["mode"] == "native", "missing execution records for non-native vision")
        visual = calculate_vision_cost(grid, vision, actions=None, skip_layers=(28, 29))
    equal(stats["original_token_blocks"], visual["full"]["token_blocks"], "full visual token-blocks")
    equal(stats["executed_token_blocks"], visual["executed"]["token_blocks"], "executed visual token-blocks")
    residual_extra = residual_extra_cost(row, vision)
    vv = visual["executed"]
    components = dict(vision_patch=vv["patch_embedding_macs"], vision_merger=vv["patch_merger_macs"],
                      vision_linear=sum(x["qkv_projection_macs"]+x["output_projection_macs"]+
                                        x["gated_mlp_projection_macs"] for x in vv["layers"]),
                      vision_attention=sum(x["attention_qk_macs"]+x["attention_av_macs"] for x in vv["layers"]))
    dt, dd = target["hidden_size"], draft["hidden_size"]
    equal(dt, dd, "shared head hidden width")
    equal(target["vocab_size"], draft["vocab_size"], "shared head vocabulary")
    vt = target["vocab_size"]
    target_queries = p
    target_qk_products = p*p
    target_head_rows = 1
    rounds = [r for r in result["rounds"] if r["kind"] == "tree"]
    is_ar = row["arm"] == "ar_reference"
    adaptor_parts = None
    if is_ar:
        equal(result["rounds"], [], "AR tree records")
        for key in ("target_verify_calls", "target_verify_query_tokens", "draft_tree_calls",
                    "draft_model_forward_calls", "draft_model_input_hidden_tokens",
                    "draft_decoder_query_tokens", "target_compaction_calls", "target_compaction_selected_tokens"):
            equal(counts[key], 0, "AR "+key)
        decode = row["num_tokens"]-1
        equal(counts["target_decode_calls"], decode, "AR decode calls")
        equal(counts["target_decode_query_tokens"], decode, "AR decode query rows")
        target_queries += decode
        target_qk_products += sum(p+i for i in range(1, decode+1))
        target_head_rows += decode
        draft_queries = draft_qk_products = draft_head_rows = fusion_rows = 0
        components.update(adaptor=0, draft_linear=0, draft_attention=0, draft_lm_head=0, visual_fusion=0)
        # These are structural zeros: AR executes none of these Draft operators.
    else:
        require(rounds, "speculative request terminated before a Draft tree; unsupported here")
        equal(counts["target_decode_calls"], 0, "spec Target decode calls")
        equal(counts["target_decode_query_tokens"], 0, "spec Target decode queries")
        equal(counts["merge_visual_tokens_in"], visual_tokens, "visual adaptor input tokens")
        equal(counts["merge_visual_tokens_out"], visual_tokens, "no-merge visual tokens")
        equal(counts["draft_tree_calls"], len(rounds), "Draft tree calls")
        equal(counts["target_verify_calls"], len(rounds), "Target verify calls")
        equal(row["verify_rounds"], len(rounds), "row verify rounds")
        equal(counts["target_compaction_calls"], len(rounds), "Target compaction calls")
        equal(counts["target_compaction_selected_tokens"], sum(len(r["submitted"]) for r in rounds), "compaction selected rows")
        equal([x for r in result["rounds"] for x in r["submitted"]], result["token_ids"], "round token reconstruction")
        require(all(r["kind"] in ("tree", "terminal_target_token") for r in result["rounds"]), "unknown round type")
        compact_prompt = p-visual_tokens+1
        require(compact_prompt > 1, "empty/nonstandard text prompt")
        committed_before = 0
        draft_queries = draft_qk_products = draft_head_rows = forward_calls = 0
        input_hidden_rows = 0
        previous_committed = None
        for i, r in enumerate(rounds):
            budget = r["budget"]
            equal((budget["B"], budget["d"], budget["K"]), (30, 3, 8), "validated tree budget")
            b, depth, beam = budget["B"], budget["d"], budget["K"]
            equal(r["tree_nodes"], b, "tree node count")
            equal(r["target_cache_before"], p+committed_before, "Target cache before verify")
            equal(r["target_cache_after"], p+committed_before+len(r["submitted"]), "Target compacted cache")
            target_queries += b
            target_qk_products += b*(p+committed_before+b)
            target_head_rows += b
            sync_q = compact_prompt if i == 0 else previous_committed
            stable_k = compact_prompt+committed_before
            draft_queries += sync_q+depth*beam
            input_hidden_rows += (p if i == 0 else previous_committed)+depth*beam
            draft_qk_products += sync_q*stable_k+sum(beam*(stable_k+beam*(j+1)) for j in range(depth))
            draft_head_rows += 1+depth*beam
            forward_calls += 1+depth
            previous_committed = len(r["submitted"])
            committed_before += previous_committed
        equal(counts["target_verify_query_tokens"], target_queries-p, "Target verify query reconstruction")
        equal(counts["draft_model_forward_calls"], forward_calls, "Draft forward reconstruction")
        equal(counts["draft_model_input_hidden_tokens"], input_hidden_rows, "Draft input-hidden row reconstruction")
        equal(counts["draft_decoder_query_tokens"], draft_queries, "Draft decoder query reconstruction")
        fusion_rows = draft_queries-1  # Only the single compressed visual sequence token bypasses these two Linear layers.
        equal(fusion_rows, input_hidden_rows-visual_tokens, "two independent fusion-row reconstructions")
        a = adaptor_macs(num_tokens=visual_tokens, in_features=dd,
                         num_heads=draft["num_attention_heads"], head_dim=dd//draft["num_attention_heads"], num_queries=2)
        name = row["scheme"]["adaptor"]
        if name == "native":
            adaptor_parts = a["original"]
        elif name in ("frozen_bmm", "retained_bmm", "compiled"):
            adaptor_parts = a["factorized"]
        else:
            raise ValueError("unsupported adaptor path: "+str(name))
        components.update(adaptor=sum(adaptor_parts.values()),
                          draft_linear=draft_queries*language_linear_per_query(draft)*draft["num_hidden_layers"],
                          draft_attention=2*dd*draft_qk_products*draft["num_hidden_layers"],
                          draft_lm_head=draft_head_rows*dd*vt, visual_fusion=4*fusion_rows*dd*dd)
    components.update(target_linear=target_queries*language_linear_per_query(target)*target["num_hidden_layers"],
                      target_attention=2*dt*target_qk_products*target["num_hidden_layers"],
                      target_lm_head=target_head_rows*dt*vt)
    equal(result["target_cache_length"], p+row["num_tokens"]-len(result["pending_tokens"]), "final Target cache")
    require(len(result["pending_tokens"]) in (0, 1), "unexpected pending tokens")
    return dict(request_id=row["request_id"], arm=row["arm"], repeat=row["repeat"], phase=row["phase"],
                num_tokens=row["num_tokens"], stop_reason=row["stop_reason"], prompt_tokens=p, visual_tokens=visual_tokens,
                observed_e2e_wall_s=row["e2e_wall_s"], precision=result["precision"],
                patch_tokens=n, output_token_ids=row["output_token_ids"],
                validated=True, components_macs=components, total_major_macs=sum(components.values()),
                adaptor_parts_macs=adaptor_parts, adaptor_call_count_inferred_from_source=0 if is_ar else 1,
                adaptor_counter_note="Non-diagnostic imadpt_calls=0 is not evidence of zero work; one visual span is processed once.",
                query_reconstruction=dict(target=target_queries, target_head=target_head_rows,
                    draft=draft_queries, draft_head=draft_head_rows, visual_fusion=fusion_rows,
                    target_sum_query_times_key=target_qk_products, draft_sum_query_times_key=draft_qk_products),
                vision_saved_major_macs=visual["saved_macs"], verify_rounds=len(rounds),
                residual_extra=residual_extra, setup_included=False)


def summarize(rows, baseline_arm):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    baseline = {(r["request_id"], r["repeat"]): r for r in grouped[baseline_arm]}
    summary = {}
    for arm, group in sorted(grouped.items()):
        savings, same_savings, baseline_total = [], [], 0
        for r in group:
            key = (r["request_id"], r["repeat"])
            require(key in baseline, "missing paired baseline")
            b = baseline[key]
            baseline_total += b["total_major_macs"]
            savings.append(100*(1-r["total_major_macs"]/b["total_major_macs"]))
            if r["output_token_ids"] == b["output_token_ids"]:
                same_savings.append(savings[-1])
        totals = {k: sum(r["components_macs"][k] for r in group) for k in group[0]["components_macs"]}
        total = sum(totals.values())
        summary[arm] = dict(request_repeats=len(group), independent_videos=len({r["request_id"] for r in group}),
                            token_total=sum(r["num_tokens"] for r in group), total_major_macs=total,
                            mean_request_major_macs=total/len(group), component_totals_macs=totals,
                            component_share_percent={k:100*v/total for k,v in totals.items()},
                            paired_mean_saving_percent=mean(savings), paired_median_saving_percent=median(savings),
                            paired_total_saving_percent=100*(1-total/baseline_total),
                            same_output_pairs=len(same_savings),
                            same_output_mean_saving_percent=mean(same_savings) if same_savings else None)
    return summary


def ar_comparison(rows):
    ar = {(r['request_id'],r['repeat']):r for r in rows if r['arm']=='ar_reference'}
    baseline = [r for r in rows if r['arm']=='baseline']
    require(ar and len(ar)==len(baseline), "AR comparison lacks complete pairs")
    results = {}
    for label, same_only in [('all_pairs',False),('same_output_eos',True)]:
        pairs = [(r,ar[(r['request_id'],r['repeat'])]) for r in baseline]
        if same_only:
            pairs = [(b,a) for b,a in pairs if b['output_token_ids']==a['output_token_ids']]
        require(pairs, "empty AR comparison subset")
        bmac, amac = (sum(x[i]['total_major_macs'] for x in pairs) for i in (0,1))
        bwall, awall = (sum(x[i]['observed_e2e_wall_s'] for x in pairs) for i in (0,1))
        results[label] = dict(pairs=len(pairs), independent_videos=len({b['request_id'] for b,a in pairs}),
            baseline_major_macs=bmac, ar_major_macs=amac, baseline_to_ar_mac_ratio=bmac/amac,
            baseline_more_macs_percent=100*(bmac/amac-1), observed_baseline_e2e_s=bwall,
            observed_ar_e2e_s=awall, observed_e2e_total_saving_percent=100*(1-bwall/awall),
            timing_scope="Existing raw paired process observations; no new timing experiment or energy attribution")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--vision-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    target, draft, vision = (json.loads(p.read_text()) for p in (args.target_config, args.draft_config, args.vision_config))
    for k, value in target["vision_config"].items():
        if k == "in_chans": k = "in_channels"
        require(k not in vision or vision[k] == value, "checkpoint/runtime visual config disagreement: "+k)
    require(target["num_hidden_layers"] == 28 and draft["num_hidden_layers"] == 1,
            "only audited 28-layer Target / 1-layer Draft supported")
    require(not target["use_sliding_window"] and not draft["use_sliding_window"], "sliding cache unsupported")
    check = adaptor_macs(num_tokens=2016, in_features=3584, num_heads=28, head_dim=128, num_queries=2)
    equal(check["factorized_total"], 860618752, "verified BMM N2016 regression")
    datasets, all_rows = {}, []
    for directory in args.run_dir:
        status = json.loads((directory/"status.json").read_text())
        require(status["status"] == "completed", "incomplete run: "+str(directory))
        raw = [json.loads(line) for line in (directory/"results.jsonl").read_text().splitlines() if line.strip()]
        rows = []
        for r in raw:
            if r["phase"] != "measure": continue
            try:
                counted = count_request(r, target, draft, vision)
            except Exception as exc:
                raise ValueError(f"{directory.name}/{r['arm']}/{r['request_id']}/repeat{r['repeat']}: {exc}") from exc
            counted["dataset"] = directory.name
            rows.append(counted)
        require(rows, "no measured requests")
        require(len({(r['request_id'],r['arm'],r['repeat']) for r in rows}) == len(rows), "duplicate request key")
        all_rows.extend(rows)
        d, q = draft["hidden_size"], 2
        # Existing multi-arm runtime constructs 2 FactorizedAdaptor + 1 scoring
        # direction + 2 FrozenQueryAdaptor preparations; deployed runtime 0/1.
        mode = status.get("deployment_adaptor")
        require(mode in (None, "native", "retained_bmm", "compiled"), "unknown deployed setup")
        is_residual_run = status['methodology']=='prefix27_residual_reuse_development_diagnostic'
        require(mode is not None or status["methodology"] == "direct_optimization_paired_long_answers" or is_residual_run,
                "unknown multi-arm setup")
        if is_residual_run:
            require(mode is None and {r['arm'] for r in rows} <= {'baseline','direct_one_identity','direct_one_reuse'},
                    'unexpected residual run implementation')
            for r in raw:
                equal(r['adaptor_preparation']['other_scheme_runtimes_constructed'], False, 'residual no alternate setup')
        setups = 0 if mode == "native" or is_residual_run else 1 if mode in ("retained_bmm", "compiled") else 5
        require(draft.get("qkv_bias") is True, "setup bias contraction assumption differs")
        datasets[directory.name] = dict(path=str(directory.resolve()), measured_rows=len(rows),
            excluded_non_measure_rows=len(raw)-len(rows),
            preparation_matrix_macs= setups*(q*d*d+q*d), preparation_direction_contractions=setups,
            preparation_note="Once per process, not per request; matrix contractions only, not loading/casting/allocation cost.",
            rows=rows)
    grouped_datasets = {}
    deployed = []
    for name, data in datasets.items():
        if name.startswith("deploy_"):
            deployed.extend(data["rows"])
        else:
            require(any(r["arm"] == "baseline" for r in data["rows"]), "no baseline for dataset")
            grouped_datasets[name] = summarize(data["rows"], "baseline")
    if deployed:
        grouped_datasets["independent_deployed_processes"] = summarize(deployed, "deployment_native")
    ar_baseline = {name:ar_comparison(data['rows']) for name,data in datasets.items()
                   if any(r['arm']=='ar_reference' for r in data['rows'])}
    report = dict(convention="1 MAC is one multiplication-plus-accumulation; 2 arithmetic FLOPs per MAC by convention",
        scope="Analytic major convolution and dense-matrix shape count; neither total scalar FLOPs nor executed hardware instructions nor DRAM/energy",
        attention_note="Dense QK and PV shape accounting including masked positions; fused kernels may execute differently. Full visual attention is within each temporal block.",
        precision_note="Both Target and Draft calls share the FP32 output-head wrapper. Model bodies are FP16; setup includes model/FP32 contractions. MACs of different precision/operators are not equal-cost hardware work.",
        excluded=EXCLUDED, configs=dict(target=target, draft=draft, vision=vision),
        source_configs=[str(p.resolve()) for p in (args.target_config,args.draft_config,args.vision_config)],
        validated_request_repeats=len(all_rows), datasets={k:{a:b for a,b in v.items() if a!='rows'} for k,v in datasets.items()},
        summaries=grouped_datasets, baseline_vs_ar=ar_baseline)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"request_macs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in all_rows))
    (args.out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    lines=["# 完整请求的主要矩阵 MAC 账", "", f"已逐请求校验 {len(all_rows)} 条测量记录。1 MAC 记一次乘加；下表是解析计算量，不是硬件实测 FLOPs、净 DRAM 或能耗。", "",
           f"统计覆盖自然 EOS 的完整请求，共 {len({r['request_id'] for r in all_rows})} 个不同视频。每条使用自己的实际输出轨迹；输出不同的计算量差不能直接称为同质量省算，重复次数不增加独立视频数。", ""]
    categories={"视觉":"vision_patch vision_merger vision_linear vision_attention", "适配器":"adaptor", "Target线性":"target_linear", "Target注意力":"target_attention", "Target词表":"target_lm_head", "Draft线性":"draft_linear", "Draft注意力":"draft_attention", "Draft词表":"draft_lm_head", "视觉融合":"visual_fusion"}
    for dataset, summary in grouped_datasets.items():
        lines.extend(["## "+dataset,"", "单位：每请求平均 TMAC（10¹² MAC）。省算率以对应原版为对照，正数表示减少。主结论看合计比率；逐对均值给每个请求相同权重，两者可不同。", "",
                      "| 臂 | 请求次数 | 总计 | "+" | ".join(categories)+" | 合计省算% | 逐对平均省算% | 同输出对数 |", "|"+"---|"*(len(categories)+6)])
        for arm,s in summary.items():
            values=[sum(s['component_totals_macs'][k] for k in keys.split())/s['request_repeats']/1e12 for keys in categories.values()]
            lines.append(f"| {arm} | {s['request_repeats']} | {s['mean_request_major_macs']/1e12:.6f} | "+" | ".join(f"{v:.6f}" for v in values)+f" | {s['paired_total_saving_percent']:.5f} | {s['paired_mean_saving_percent']:.5f} | {s['same_output_pairs']}/{s['request_repeats']} |")
        lines.append("")
    if ar_baseline:
        lines.extend(["## 原版 ViSpec 与同批 AR", "", "| 数据集/子集 | 请求对/视频 | ViSpec / AR 主要 MAC | ViSpec 多算% | 已记录整请求总时延减少% |", "|---|---:|---:|---:|---:|"])
    for name,subsets in ar_baseline.items():
        for subset,a in subsets.items():
            lines.append(f"| {name} / {subset} | {a['pairs']}/{a['independent_videos']} | {a['baseline_to_ar_mac_ratio']:.6f}× | {a['baseline_more_macs_percent']:.3f} | {a['observed_e2e_total_saving_percent']:.3f} |")
    if ar_baseline:
        lines.extend(["", "投机解码可通过并行验证减少串行等待，同时计算更多候选 token。此处实际数据体现了这一点；MAC、wall 和能耗不是同一个指标。时延列仅重述同批原始观测，无新增测速、独占 GPU 或节能结论。", ""])
    if any(r['residual_extra'] is not None for r in all_rows):
        lines.extend(["## 残差实验逐请求明细", "", "正数表示相对同请求 baseline 增加。下表未把逐元素减加折成 MAC。", "",
                      "| 视频 | 臂 | token（含EOS） | 验证轮次 | 总 TMAC | 总 MAC 变化% | 视觉省 GMAC | 残差减加 M次 | 残差逻辑读写 MiB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        bases={(r['dataset'],r['request_id'],r['repeat']):r for r in all_rows if r['arm']=='baseline'}
        for r in sorted(all_rows,key=lambda x:(x['request_id'],x['arm'])):
            b=bases[(r['dataset'],r['request_id'],r['repeat'])]
            extra=r['residual_extra']
            ops=extra['residual_elementwise_ops'] if extra else 0
            traffic=extra['residual_logical_io_bytes'] if extra else 0
            lines.append(f"| {r['request_id'].removeprefix('direct_quality_vdc_v_')} | {r['arm']} | {r['num_tokens']} | {r['verify_rounds']} | {r['total_major_macs']/1e12:.6f} | {100*(r['total_major_macs']/b['total_major_macs']-1):+.3f} | {r['vision_saved_major_macs']/1e9:.6f} | {ops/1e6:.6f} | {traffic/2**20:.6f} |")
        lines.extend(["", "复用仅作用于被选行：每个元素一次减法、一次加法。逻辑读写按两次逐元素操作各二读一写，加选中前缀 cat 的一读一写计算（6×元素数×字节 + 2×cat物化字节）。这是显式操作的逻辑流量，不是 DRAM 实测；缓存和内核实现会改变实际流量。原本的门控、路由、重排不包含在新增残差项中。", "",
                      "逐请求还保存 retained_prefix 的逻辑/底层存储字节、选中前缀与 delta 临时字节。保留引用不等于新增复制，临时张量大小也不等于整请求峰值。所有 residual 臂逐层核验仅跳 block28、block29 全执行，动作只允许 0/1。", ""])
        lines.extend(["### 长度与验证计算", ""])
        for r in sorted((r for r in all_rows if r['arm']=='direct_one_reuse'),key=lambda x:x['request_id']):
            b=bases[(r['dataset'],r['request_id'],r['repeat'])]
            increase=r['total_major_macs']-b['total_major_macs']
            visual_saved=r['vision_saved_major_macs']
            lines.append(f"- {r['request_id'].removeprefix('direct_quality_vdc_v_')}：token {b['num_tokens']}→{r['num_tokens']}（{100*(r['num_tokens']/b['num_tokens']-1):+.3f}%），验证 {b['verify_rounds']}→{r['verify_rounds']} 轮（{100*(r['verify_rounds']/b['verify_rounds']-1):+.3f}%）。视觉省 {visual_saved/1e9:.3f} GMAC，但其余矩阵多算 {(increase+visual_saved)/1e9:.3f} GMAC，整请求净增 {increase/1e9:.3f} GMAC（{100*increase/b['total_major_macs']:.3f}%）。")
        lines.extend(["", "这些请求的输出均已改变；较短或较接近原版长度本身不能证明事实质量相同。验证轮次决定额外候选计算，不能只用最终 token 长度预测整请求工作量。", ""])
    lines.extend(["## 口径与校验", "", "- 复用 compute_cost 的视觉计数，逐层校验实际动作与 token-block 数；原生全深度按实际 grid 重建。窗口边界单独计算，全空间注意力只跨同一时间块的空间位置。",
        "- 校验输入 patch/文本形状、Target prefill/verify/decode、Draft forward/输入/decoder 行数、KV 边界和完整 token 轨迹。缺失、不支持或不一致直接报错。",
        "- Target 的每个验证节点都计词表投影；Draft 每轮另计 1+3×8 行。共享 LM-head 权重不意味着这些调用免费。",
        "- 日志已逐请求验证：Target/Draft 主体为 FP16，共享的词表投影为 FP32。不同精度和算子的 MAC 不是等成本硬件操作，不能据此直接推时延或能耗。",
        "- 单视觉片段只调用一次 imadpt。非诊断日志里的 imadpt_calls=0 是未启用该计数，不能解释为没有计算。",
        "- 原始视觉融合计两次 [N,2D]→[N,D] Linear；没有把尚未在这些结果中运行的 cached/folded 候选省算算入。",
        "- BMM 重排四项：评分 NHQD、原始特征聚合 NHQD、池化后 V 投影 QD²、O 投影 QD²；N=2016 时总计 0.860618752 GMAC。",
        "- 注意力按 dense QK/PV 的形状计数，含 mask 位置；不是 Tensor Core 指令或内核实际跳过位置的硬件计数。", "",
        "## 重建公式与来源", "",
        "令文本宽度 D=3584、中间宽度 I=18944。每层 Q 行线性部分为 Q(2D²+2DDkv+3DI)；Target 的 Dkv=512、28 层，Draft 的 Dkv=3584、1 层。两注意力矩阵为 2DQK。词表投影每行为 D×152064。",
        "",
        "Target prefill 使用 Q=K=P；每轮验证使用 Q=B、K=target_cache_before+B。Draft 压缩后 prompt 长度 S=P−V+1，首轮同步 Q=K=S；以后同步 Q 为上一轮提交数，K 为 S 加此前累计提交数。每轮三个分支各 8 行，K 在同步长度上增加 8、16、24。只有单个视觉序列 token 不走两层视觉融合，融合行数为 Draft decoder 总行数减 1。",
        "",
        "实现依据：vendor/hf517_vispec_adapter.py（实际 prefill/head/counter）、vendor/vispec_joint_adapter.py（树预算、提交和 KV 边界）、作者 cnets_ours.py（Draft/ImgAdaptor/img_fc/fc/topK）、direct_opt/compression.py、direct_opt/integration.py、training_free/scheme_context.py，以及残差两臂的 direct_opt/residual_reuse_depth.py 和 run_residual_reuse.py。真实 checkpoint config 的只读副本路径在 summary.json 的 source_configs 中；原始路径记录在配置副本旁的 config_provenance.json。视觉矩阵结构核对了服务器 Transformers 5.17 的 modeling_qwen2_5_vl.py。未知新融合臂直接拒绝，不能沿用原两Linear费用。", "",
        "## 启动准备（另计）", "", "| 数据集/进程 | 方向构造次数 | 准备 GMAC |", "|---|---:|---:|"])
    for name,data in datasets.items():
        lines.append(f"| {name} | {data['preparation_direction_contractions']} | {data['preparation_matrix_macs']/1e9:.9f} |")
    lines.extend(["", "旧通用多臂进程构造 5 组方向（两种精度、评分方向和两个冻结后端）；单独 native 和本次 NativeRuntime 残差实验为 0，retained/compiled 为 1。计入 QWk 和 Q·Kbias 收缩；加载、类型转换与分配成本不在该数中。", "", "## 未计项", "", ", ".join(EXCLUDED)+"。这些操作和 I/O 仍影响时延，MAC 降幅不能替代完整 wall、内存或能耗测试；若有残差实验，其明确减加及逻辑 IO 已在独立栏报告，不进入主要 MAC 合计。", ""])
    (args.out/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"validated_request_repeats":len(all_rows), "out":str(args.out.resolve())}))


if __name__ == "__main__":
    main()
