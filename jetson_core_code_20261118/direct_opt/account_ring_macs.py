"""Offline dense-matrix MAC accounting for completed ring-layout requests.

Uses each arm's actual trajectory and physical attention shape, including masked
padding. Does not load a model, execute GPU work, or modify the original counter.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from account_request_macs import (EXCLUDED, adaptor_macs, calculate_vision_cost,
                                  count_request, equal, language_linear_per_query, require)


def count_ring_request(row, target, draft, vision, *, max_new_tokens):
    result, counts = row["result"], row["result"]["forward_counts"]
    preparation = row["adaptor_preparation"]
    ring = preparation["ring_kv"]
    mode, cap, tree_cap = (ring[k] for k in ("mode", "recent_capacity", "tree_capacity"))
    require(mode in ("full", "protected_prompt", "pure_recent"), "unknown ring mode")
    equal(row["arm"], f"memory_shared_embedding_compiled_ring_{mode}_{cap}", "ring arm")
    equal(row["scheme"], dict(depth="fixed0", adaptor="compiled", compression=None, ratio=1.0), "ring scheme")
    require(preparation.get("fusion_mode") in (None, "native"), "modified visual fusion unsupported")
    equal(preparation["mode"], "compiled", "deployed adaptor")
    require(not row.get("capture_enabled") and not row.get("graph_setup"), "capture/graph requests unsupported")
    equal((target["num_hidden_layers"], draft["num_hidden_layers"]), (28, 1), "audited layer counts")
    require(not target["use_sliding_window"] and not draft["use_sliding_window"], "model sliding attention unsupported")
    for name, expected in (("target", "float16"), ("draft", "float16"), ("output_head", "float32"), ("target_sdpa", "math")):
        equal(result["precision"][name], expected, name + " precision/backend")
    p, tokens = result["prompt_tokens"], result["token_ids"]
    equal(tokens, row["output_token_ids"], "output token IDs")
    equal((len(tokens), result["output_length"]), (row["num_tokens"], row["num_tokens"]), "output length")
    require(tokens, "empty output unsupported")
    equal(result["stop_reason"], row["stop_reason"], "termination")
    eos = (151645, 151643)
    if row["stop_reason"] == "eos":
        require(not row["truncated"] and tokens[-1] in eos and not any(t in eos for t in tokens[:-1]), "invalid natural EOS")
    elif row["stop_reason"] == "limit":
        require(row["truncated"] and len(tokens) == max_new_tokens and not any(t in eos for t in tokens), "invalid token-limit termination")
    else:
        raise ValueError("unsupported termination")
    grid, shapes = row["video_grid_thw"], row["input_shapes"]
    require(len(grid) == 1, "one video required")
    t, h, w = grid[0]
    n, merge = t*h*w, vision["spatial_merge_size"]
    require(h % merge == 0 and w % merge == 0, "invalid merged grid")
    visual_tokens = n//merge**2
    for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
        equal(shapes[name], [1, p], name + " shape")
    equal(shapes["video_grid_thw"], [1, 3], "grid shape")
    equal(shapes["pixel_values_videos"], [n, vision["in_channels"]*vision["temporal_patch_size"]*vision["patch_size"]**2], "patch shape")
    equal(row["stats"]["mode"], "native", "full visual depth")
    equal(row["stats"]["calls"], [], "no depth policy")
    visual = calculate_vision_cost(grid, vision, actions=None, skip_layers=(28, 29))["executed"]
    equal(row["stats"]["executed_token_blocks"], visual["token_blocks"], "executed visual blocks")
    equal(row["stats"]["original_token_blocks"], visual["token_blocks"], "original visual blocks")
    rounds = [r for r in result["rounds"] if r["kind"] == "tree"]
    require(rounds and all(r["kind"] in ("tree", "terminal_target_token") for r in result["rounds"]), "unknown/empty tree execution")
    equal([x for r in result["rounds"] for x in r["submitted"]], tokens, "submitted output reconstruction")
    require(ring["finalized"] and not ring["failed"], "ring request not finalized")
    equal(ring["draft_cache_policy"], "unchanged_full_history", "Draft cache policy")
    equal((ring["prompt_tokens"], ring["prefill_calls"], ring["completed_verifications"]), (p, 1, len(rounds)), "ring call counts")
    equal(len(ring["rounds"]), len(rounds), "ring trace length")
    dt, dd, layers = target["hidden_size"], draft["hidden_size"], target["num_hidden_layers"]
    equal(dt, dd, "Target/Draft width")
    equal(target["vocab_size"], draft["vocab_size"], "vocabulary")
    kv_bytes_per_layer = 2*target["num_key_value_heads"]*(dt//target["num_attention_heads"])*2
    equal(ring["bytes_per_token_all_target_layers"], layers*kv_bytes_per_layer, "physical KV stride")
    physical_cap = None if mode == "full" else (p if mode == "protected_prompt" else 0) + cap + tree_cap

    def validate_cache(snapshot, semantic, *, active, before_commit):
        equal(snapshot["semantic_length"], semantic, "cache semantic length")
        if mode == "full":
            length = snapshot["occupied_tokens_per_layer"]
            equal(length, semantic, "full cache physical length")
        else:
            equal(snapshot["layer_count"], layers, "cache layer count")
            first = snapshot["first_layer"]
            equal(first["physical_capacity"], physical_cap, "physical capacity includes padding")
            equal(first["prefix_length"], p if mode == "protected_prompt" else 0, "protected prefix")
            equal((first["recent_capacity"], first["tree_capacity"]), (cap, tree_cap), "ring capacities")
            equal(first["active_tree_nodes"], active, "active tree rows")
            equal(first["next_position"] + active, semantic, "semantic base vs physical layout")
            equal(first["allocated_kv_storage_bytes"], physical_cap*kv_bytes_per_layer, "layer KV allocation")
            require(first["valid_history_tokens"] + active <= physical_cap, "valid rows exceed capacity")
            length = physical_cap
        equal(snapshot["allocated_kv_storage_bytes"], length*layers*kv_bytes_per_layer, "all-layer KV allocation")
        return length

    validate_cache(ring["prefill_cache"], p, active=0, before_commit=False)
    target_queries, physical_products, semantic_products = p, p*p, p*p
    draft_queries = draft_products = draft_head = input_hidden = committed = 0
    compact_prompt = p - visual_tokens + 1
    require(compact_prompt > 1, "invalid compressed text prompt")
    previous = None
    attention_rows = []
    for i, (r, rr) in enumerate(zip(rounds, ring["rounds"])):
        equal((r["budget"]["B"], r["budget"]["d"], r["budget"]["K"]), (30, 3, 8), "audited tree budget")
        b, selected = r["tree_nodes"], len(r["submitted"])
        equal(b, 30, "tree node count")
        equal((r["target_cache_before"], r["target_cache_after"]), (p+committed, p+committed+selected), "semantic KV trajectory")
        equal((rr["logical_base"], rr["nodes"], rr["committed_tokens"], rr["semantic_length_after"]),
              (p+committed, b, selected, p+committed+selected), "ring/decoder round agreement")
        equal(rr["semantic_rope_base"], r["semantic_rope_base"], "unchanged RoPE base")
        actual_k = validate_cache(rr["cache_before_commit"], p+committed+b, active=b, before_commit=True)
        validate_cache(rr["cache_after_commit"], p+committed+selected, active=0, before_commit=False)
        target_queries += b
        physical_products += b*actual_k
        semantic_products += b*(p+committed+b)
        attention_rows.append(dict(round=i, query_rows=b, physical_key_rows=actual_k,
                                   semantic_key_rows=p+committed+b, submitted=selected))
        sync = compact_prompt if i == 0 else previous
        stable = compact_prompt+committed
        draft_queries += sync+24
        input_hidden += (p if i == 0 else previous)+24
        draft_products += sync*stable + sum(8*(stable+8*(j+1)) for j in range(3))
        draft_head += 25
        previous = selected
        committed += selected
    pending = result["pending_tokens"]
    require(len(pending) in (0, 1), "unsupported pending token count")
    equal(result["target_cache_length"], p+len(tokens)-len(pending), "final semantic cache length")
    equal(committed, len(tokens)-len(pending), "committed vs emitted token count")
    validate_cache(ring["final_cache"], result["target_cache_length"], active=0, before_commit=False)
    if mode != "full":
        equal(len(ring["final_layer_metadata"]), layers, "final layer records")
        for item in ring["final_layer_metadata"]:
            equal(item["physical_capacity"], physical_cap, "all layers physical capacity")
            equal(item["operation_counts"]["append_tree"], len(rounds), "per-layer append count")
            equal(item["operation_counts"]["commit"], len(rounds), "per-layer commit count")
    expected_counts = dict(target_visual_calls=1, target_prefill_calls=1, target_prefill_query_tokens=p,
        target_verify_calls=len(rounds), target_verify_query_tokens=target_queries-p,
        target_decode_calls=0, target_decode_query_tokens=0, draft_tree_calls=len(rounds),
        draft_model_forward_calls=4*len(rounds), draft_model_input_hidden_tokens=input_hidden,
        draft_decoder_query_tokens=draft_queries, target_compaction_calls=len(rounds),
        target_compaction_selected_tokens=committed, merge_visual_tokens_in=visual_tokens,
        merge_visual_tokens_out=visual_tokens)
    for key, value in expected_counts.items():
        equal(counts[key], value, "actual forward count "+key)
    equal(row["verify_rounds"], len(rounds), "verify rounds")
    fusion_rows = draft_queries-1
    equal(fusion_rows, input_hidden-visual_tokens, "fusion row reconstruction")
    adaptor = adaptor_macs(num_tokens=visual_tokens, in_features=dd, num_heads=draft["num_attention_heads"],
                            head_dim=dd//draft["num_attention_heads"], num_queries=2)["factorized"]
    components = dict(vision_patch=visual["patch_embedding_macs"], vision_merger=visual["patch_merger_macs"],
        vision_linear=sum(x["qkv_projection_macs"]+x["output_projection_macs"]+x["gated_mlp_projection_macs"] for x in visual["layers"]),
        vision_attention=sum(x["attention_qk_macs"]+x["attention_av_macs"] for x in visual["layers"]),
        adaptor=sum(adaptor.values()), draft_linear=draft_queries*language_linear_per_query(draft)*draft["num_hidden_layers"],
        draft_attention=2*dd*draft_products*draft["num_hidden_layers"], draft_lm_head=draft_head*dd*target["vocab_size"],
        visual_fusion=4*fusion_rows*dd*dd, target_linear=target_queries*language_linear_per_query(target)*layers,
        target_attention=2*dt*physical_products*layers, target_lm_head=(1+target_queries-p)*dt*target["vocab_size"])
    if row["stop_reason"] == "eos":
        # Only the dispatch label is aliased; no token, output, count, or round
        # data is changed. This uses the same audited compiled/native-fusion
        # matrix path. The independent old counter assumes semantic Target K.
        reference = count_request(dict(row, arm="deployment_compiled"), target, draft, vision)
        for name, value in components.items():
            equal(value if name != "target_attention" else 2*dt*semantic_products*layers,
                  reference["components_macs"][name], "old-counter cross-check "+name)
    return dict(request_id=row["request_id"], repeat=row["repeat"], arm=row["arm"], mode=mode,
        validated=True, num_tokens=len(tokens), stop_reason=row["stop_reason"], truncated=row["truncated"],
        prompt_tokens=p, verify_rounds=len(rounds), physical_capacity=physical_cap,
        components_macs=components, total_major_macs=sum(components.values()),
        target_prefill_attention_macs=2*dt*p*p*layers,
        target_verification_attention_macs=2*dt*(physical_products-p*p)*layers,
        target_semantic_key_counterfactual_attention_macs=2*dt*semantic_products*layers,
        attention_layout_delta_same_trajectory_macs=2*dt*(physical_products-semantic_products)*layers,
        old_counter_cross_checked=row["stop_reason"] == "eos", setup_included=False,
        attention_rounds=attention_rows, output_token_ids=tokens,
        query_reconstruction=dict(target=target_queries, draft=draft_queries, draft_head=draft_head,
                                  visual_fusion=fusion_rows, target_sum_query_times_key=physical_products))


def summarize_requests(rows):
    groups = defaultdict(list)
    full = {(r["request_id"], r["repeat"]): r for r in rows if r["mode"] == "full"}
    for row in rows:
        groups[row["arm"]].append(row)
    summaries = {}
    for arm, group in groups.items():
        total = sum(r["total_major_macs"] for r in group)
        pairs = [(r, full[(r["request_id"], r["repeat"])]) for r in group
                 if (r["request_id"], r["repeat"]) in full]
        same = [(r, b) for r, b in pairs if r["output_token_ids"] == b["output_token_ids"]
                and r["stop_reason"] == b["stop_reason"] == "eos"]
        summaries[arm] = dict(request_repeats=len(group), independent_videos=len({r["request_id"] for r in group}),
            token_total=sum(r["num_tokens"] for r in group), verify_rounds_total=sum(r["verify_rounds"] for r in group),
            natural_eos=sum(r["stop_reason"] == "eos" for r in group), total_major_macs=total,
            component_totals_macs={k:sum(r["components_macs"][k] for r in group) for k in group[0]["components_macs"]},
            paired_total_mac_change_vs_full_percent=100*(total/sum(b["total_major_macs"] for _, b in pairs)-1) if len(pairs) == len(group) else None,
            same_output_natural_eos_pairs=len(same),
            same_output_natural_eos_mac_change_percent=100*(sum(r["total_major_macs"] for r, _ in same)/sum(b["total_major_macs"] for _, b in same)-1) if same else None,
            total_layout_attention_delta_same_trajectories_macs=sum(r["attention_layout_delta_same_trajectory_macs"] for r in group))
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    for name in ("target", "draft", "vision"):
        parser.add_argument(f"--{name}-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    configs = [json.loads(p.read_text()) for p in (args.target_config, args.draft_config, args.vision_config)]
    for key, value in configs[0]["vision_config"].items():
        key = "in_channels" if key == "in_chans" else key
        require(key not in configs[2] or configs[2][key] == value, "vision config disagreement: "+key)
    rows = []
    for directory in args.run_dir:
        status = json.loads((directory/"status.json").read_text())
        equal(status["status"], "completed", "run completion")
        for row in map(json.loads, (directory/"results.jsonl").read_text().splitlines()):
            if row["phase"] == "measure":
                counted = count_ring_request(row, *configs, max_new_tokens=status["args"]["max_new_tokens"])
                counted["dataset"] = directory.name
                rows.append(counted)
    require(rows, "no measured rows")
    require(len({(r['request_id'],r['repeat'],r['arm']) for r in rows}) == len(rows), "duplicate measured key")
    baseline = {(r["request_id"], r["repeat"]): r for r in rows if r["mode"] == "full"}
    for row in rows:
        base = baseline.get((row["request_id"], row["repeat"]))
        if base:
            row["same_output_as_full"] = row["output_token_ids"] == base["output_token_ids"]
            row["same_output_natural_eos_as_full"] = row["same_output_as_full"] and row["stop_reason"] == base["stop_reason"] == "eos"
            row["major_mac_change_vs_full_percent"] = 100*(row["total_major_macs"]/base["total_major_macs"]-1)
    report = dict(scope="Analytic dense-matrix/convolution MACs; 1 MAC is one multiplication-plus-accumulation; not hardware instructions, scalar FLOPs, DRAM, time or energy",
        attention_rule="Target prefill K=prompt; tree K=actual full physical capacity, including masked padding; full mode K=semantic base+tree nodes",
        trajectory_rule="Every arm uses its own actual forward counts and output/verification rounds; no baseline trajectory substitution",
        limit_rule="Limit outputs retain their true tokens/stop reason and are not treated as natural EOS or quality-equivalent",
        setup_note="Compiled-adaptor setup contractions and all allocation/copy/mask operations excluded from major matrix MACs; unchanged by cache layout",
        excluded=EXCLUDED, requests=rows, summaries=summarize_requests(rows),
        configs={k:str(getattr(args, k+"_config").resolve()) for k in ("target","draft","vision")})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    lines = ["# Ring 完整请求主要矩阵 MAC 账", "", "按每臂实际轨迹计算；含 masked padding 的 dense attention。MAC 不等于时延、能耗或 DRAM。", "",
        "## 合计", "", "| 臂 | 请求 | token | 验证轮 | 总 TMAC | 总 MAC 变化% | 同输出自然 EOS 对数 | 同输出子集 MAC 变化% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "无配对" if value is None else f"{value:+.5f}"
    for arm, summary in report["summaries"].items():
        lines.append(f"| {arm} | {summary['request_repeats']} | {summary['token_total']} | {summary['verify_rounds_total']} | {summary['total_major_macs']/1e12:.6f} | {percent(summary['paired_total_mac_change_vs_full_percent'])} | {summary['same_output_natural_eos_pairs']} | {percent(summary['same_output_natural_eos_mac_change_percent'])} |")
    lines += ["", "## 逐请求", "",
        "| 臂 | token / 停止 | 验证轮 | 总 TMAC | Target attention TMAC | 相对 full 总 MAC 变化 | 同输出自然 EOS |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        delta = row.get("major_mac_change_vs_full_percent")
        lines.append(f"| {row['arm']} | {row['num_tokens']} / {row['stop_reason']} | {row['verify_rounds']} | {row['total_major_macs']/1e12:.6f} | {row['components_macs']['target_attention']/1e12:.6f} | {delta:+.4f}% | {row.get('same_output_natural_eos_as_full')} |" if delta is not None else f"| {row['arm']} | {row['num_tokens']} / {row['stop_reason']} | {row['verify_rounds']} | {row['total_major_macs']/1e12:.6f} | {row['components_macs']['target_attention']/1e12:.6f} | 无配对 | 未比较 |")
    lines += ["", "- Prefill 和完整视觉计算保持原执行形状。Target linear/head、Draft 与融合均按真实执行行数重建并校验计数。",
              "- Ring 验证 attention 使用 P+C+T 或 C+T；未使用有效槽位数替代真实矩阵 K。full 模式使用每轮语义长度加 tree 节点数。",
              "- 每条自然 EOS 记录还与原 account_request_macs.py 交叉核验；limit 记录保留真实 token 与截断状态，未伪造 EOS。",
              "- 输出不同或截断的 MAC 差异不能解释为同质量省算。JSON 另列同一候选轨迹下物理 K 对 attention 的独立增减。",
              "- 减少 KV 复制不直接减少 MAC；GQA repeat、mask、copy/cast、softmax 和内核开销不在本矩阵账中。"]
    (args.out/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"validated_requests":len(rows), "out":str(args.out.resolve())}))


if __name__ == "__main__":
    main()
