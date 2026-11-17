"""Offline paired throughput and exact-commit receipts; no Torch/CUDA imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from assess_visual_kv import generation, memory_evidence

BASE = 'memory_shared_embedding_compiled_ring_active_width_protected_prompt_512'
DELTA = BASE + '_delta_commit'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commit_receipt(row):
    info = row['adaptor_preparation']['ring_kv']
    is_delta = row['arm'] == DELTA
    expected_variant = 'active_width_delta_commit' if is_delta else 'active_width'
    require(info['variant'] == expected_variant, 'Wrong per-cell cache variant')
    require(info['finalized'] and not info['failed'], 'Unfinalized/failed cache')
    require(info['mode'] == 'protected_prompt' and info['recent_capacity'] == 512,
            'Unexpected cache recipe')
    require(info['tree_capacity'] == 30, 'Unexpected tree capacity')
    trees = [r for r in row['result']['rounds'] if r['kind'] == 'tree']
    require(len(trees) == len(info['rounds']) == info['completed_verifications'], 'Uncovered verification')
    layers = len(info['final_layer_metadata'])
    require(layers == 28, 'Expected all 28 Target layers')
    per_token = info['bytes_per_token_all_target_layers']
    require(per_token == 57344, 'Unexpected Target FP16 KV payload/token')
    read = write = moved = accepted = launches = 0
    for tree, step in zip(trees, info['rounds']):
        selected = tree['selected_tree_indices']
        k, d = len(selected), sum(i != source for i, source in enumerate(selected))
        require(step['logical_base'] == tree['target_cache_before'], 'Runtime/generator base mismatch')
        require(step['semantic_length_after'] == tree['target_cache_after'], 'Runtime/generator commit mismatch')
        require(step['committed_tokens'] == k and step['nodes'] == tree['tree_nodes'], 'Runtime/generator tree mismatch')
        require(step['all_layer_commit_reports_equal'] is True, 'First-layer report not representative')
        report, = step['layer_commit_reports']
        require(report['accepted_tokens'] == report['stored_tokens'] == k, 'Stored accepted path differs')
        require(report['evicted_recent_tokens'] == 0, 'History eviction occurred')
        require(report['previous_scratch_start'] == tree['target_cache_before'], 'Scratch is not active-width')
        expected_each = (d if is_delta else 2*k) * per_token // layers
        require(report['copy_read_bytes'] == report['copy_write_bytes'] == expected_each,
                'Commit logical byte receipt disagrees with actual selected rows')
        require(report['copy_total_bytes'] == 2*expected_each, 'Bad commit byte total')
        require(report['traffic_is_measured_dram'] is False, 'Copy bytes mislabeled as measured DRAM')
        if is_delta:
            require(report['payload_backend'] == 'triton', 'Candidate silently used another payload backend')
            require(report['moved_tokens'] == d and report['payload_kernel_launches'] == int(d > 0),
                    'Candidate moved-row/kernel receipt differs')
            require(report['scratch_payload_peak_bytes'] == 0, 'Candidate retained a global gather temporary')
            launches += report['payload_kernel_launches'] * layers
        read += report['copy_read_bytes'] * layers
        write += report['copy_write_bytes'] * layers
        moved += d
        accepted += k
    metadata = info['final_layer_metadata']
    for layer in metadata:
        require(layer['next_position'] == row['result']['target_cache_length'], 'Final KV boundary mismatch')
        require(layer['recent_count'] == accepted <= 512 and layer['active_tree_nodes'] == 0,
                'Evicted or unfinished final cache')
        require(layer['operation_counts']['commit'] == len(trees), 'Missing layer commits')
        totals = layer['payload_copy_totals']
        expected_total_read = ((info['prompt_tokens'] + sum(t['tree_nodes'] for t in trees))
                               * per_token // layers + read // layers)
        require(totals['read_bytes'] == totals['write_bytes'] == expected_total_read,
                'Final prefill+append+commit byte ledger differs from round receipts')
        require(totals['total_bytes'] == 2*expected_total_read, 'Final payload total inconsistent')
    return dict(variant=expected_variant, layers=layers, committed_kv_tokens=accepted,
                already_in_place_tokens=accepted-moved, moved_tokens=moved,
                copy_read_bytes=read, copy_write_bytes=write, copy_total_bytes=read+write,
                recorded_payload_kernel_launches=launches if is_delta else None,
                baseline_payload_kernel_launches_not_recorded=not is_delta,
                owned_storage_bytes=info['final_cache']['owned_storage_bytes'],
                all_maintenance_payload_read_bytes=sum(x['payload_copy_totals']['read_bytes'] for x in metadata),
                all_maintenance_payload_write_bytes=sum(x['payload_copy_totals']['write_bytes'] for x in metadata),
                scope='Target accepted-path commit payload only; logical operator bytes, not DRAM')


def audit(directory, known_graphics_pids=()):
    directory = Path(directory)
    status = json.loads((directory/'status.json').read_text())
    schedule = json.loads((directory/'schedule.json').read_text())
    rows = [json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines() if line.strip()]
    require(status['status'] == 'completed', 'Run incomplete')
    require(status['paired_active_baseline'] is True, 'Not a paired active/delta run')
    require(status['args']['repeats'] == 3 and status['args']['warmup'] == 1, 'Unexpected repeat/warmup recipe')
    require(status['args']['max_new_tokens'] == 512 and status['memory_layout'] == 'shared_embedding'
            and status['deployment_adaptor'] == 'compiled', 'Wrong frozen model recipe')
    key = lambda r: (r['request_id'], r['arm'], r['phase'], r['repeat'])
    require(len(rows) == len({key(r) for r in rows}), 'Duplicate result cell')
    require([key(r) for r in rows] == [key(r) for r in schedule], 'Results do not match scheduled order')
    require(len(rows) == status['total_cells'] == status['completed_cells'] == 8, 'Expected 2 warmups and 6 measured cells')
    require({r['arm'] for r in rows} == {BASE, DELTA}, 'Unexpected/missing arm')
    require(len({r['request_id'] for r in rows}) == 1, 'This audit expects the single 478-token request')
    details = []
    for row in rows:
        gen = generation(row, status)
        require(gen['receipt_valid'] and gen['natural_eos'], 'Invalid trace or no natural EOS: '+str(gen['trace_validation_error']))
        require(row['num_tokens'] == 478, 'Expected the fixed 478-token workload')
        require(math.isfinite(row['e2e_wall_s']) and row['e2e_wall_s'] > 0, 'Invalid complete request duration')
        receipt = commit_receipt(row)
        memory = memory_evidence(row, row['adaptor_preparation']['ring_kv'])
        require(memory['consistent_complete_peaks'], 'Incomplete/inconsistent whole-request peaks')
        hardware = row['hardware']
        foreign = sorted({entry['pid'] for entry in hardware.get('foreign_processes', [])})
        details.append(dict(request_id=row['request_id'], arm=row['arm'], phase=row['phase'], repeat=row['repeat'],
            num_tokens=row['num_tokens'], verify_rounds=row['verify_rounds'], e2e_wall_s=row['e2e_wall_s'],
            output_tokens_per_e2e_s=row['num_tokens']/row['e2e_wall_s'], commit=receipt, memory=memory,
            valid_timing=row['valid_timing'], timing_reasons=row.get('timing_reasons'),
            runtime_finalize_wall_s=row.get('runtime_finalize_wall_s'), hardware={
                'available':hardware.get('available'), 'device':hardware.get('device'),
                'physical_device_index':hardware.get('physical_device_index'),
                'allowed_pids':hardware.get('allowed_pids'), 'foreign_pids':foreign,
                'foreign_except_known_graphics':sorted(set(foreign)-set(known_graphics_pids)),
                'process_query_count':hardware.get('process_query_count'),
                'process_query_failures':hardware.get('process_query_failures'),
                'process_visibility_complete':hardware.get('process_visibility_complete'),
                'process_interval_s':hardware.get('process_interval_s'),
                'telemetry_summary':hardware.get('telemetry_summary')}))
    measured = [r for r in rows if r['phase'] == 'measure']
    require(all(sum(r['phase']=='warmup' and r['arm']==arm for r in rows)==1 for arm in (BASE,DELTA)), 'Wrong warmup allocation')
    require(all({r['repeat'] for r in measured if r['arm']==arm}=={0,1,2} for arm in (BASE,DELTA)), 'Missing formal repeats')
    pairs = []
    for repeat in range(3):
        pair = [r for r in measured if r['repeat']==repeat]
        require(len(pair)==2, 'Bad matched repeat')
        b, c = (next(r for r in pair if r['arm']==arm) for arm in (BASE,DELTA))
        checks = {field:b[field]==c[field] for field in ['output_token_ids','input_shapes','video_grid_thw','timestamps','text']}
        checks.update({field:b['result'][field]==c['result'][field]
                       for field in ['rounds','forward_counts','precision','prompt_tokens','target_cache_length']})
        require(all(checks.values()), 'Paired exact semantics/input metadata differ: '+str(checks))
        pairs.append(dict(repeat=repeat, first_arm=pair[0]['arm'], equality=checks,
            baseline_e2e_wall_s=b['e2e_wall_s'], delta_e2e_wall_s=c['e2e_wall_s'],
            wall_reduction_pct=100*(1-c['e2e_wall_s']/b['e2e_wall_s']),
            throughput_increase_pct=100*(b['e2e_wall_s']/c['e2e_wall_s']-1)))
    arms = {}
    for arm in (BASE,DELTA):
        sample = [r for r in measured if r['arm']==arm]
        walls = [r['e2e_wall_s'] for r in sample]
        tokens = sum(r['num_tokens'] for r in sample)
        arms[arm] = dict(formal_count=len(sample), total_output_tokens=tokens,
            total_e2e_wall_s=sum(walls), total_tokens_per_total_e2e_s=tokens/sum(walls),
            median_e2e_wall_s=statistics.median(walls), min_e2e_wall_s=min(walls), max_e2e_wall_s=max(walls),
            median_per_request_tokens_per_s=statistics.median(r['num_tokens']/r['e2e_wall_s'] for r in sample))
    b, c = arms[BASE], arms[DELTA]
    return dict(source=str(directory.resolve()), source_sha256={name:digest(directory/name)
        for name in ['results.jsonl','status.json','schedule.json','requests.json']},
        tool_sha256=digest(__file__), complete_paired_mechanical_pass=True, pairs=pairs, arms=arms,
        aggregate_throughput_increase_pct=100*(c['total_tokens_per_total_e2e_s']/b['total_tokens_per_total_e2e_s']-1),
        ratio_of_median_wall_reduction_pct=100*(1-c['median_e2e_wall_s']/b['median_e2e_wall_s']),
        all_eight_output_ids_identical=all(r['output_token_ids']==rows[0]['output_token_ids'] for r in rows),
        rows=details, pid=status.get('pid'), known_graphics_pids=list(known_graphics_pids),
        limits=['One independent video; three paired repeats are not three videos.',
            'Throughput is sequential single-request output tokens divided by complete request wall time; not concurrent serving throughput.',
            'Output counts include EOS; warmups and model loading are excluded from throughput.',
            'Input token values were not saved: same manifest, prompt count, grid and timestamps do not independently prove equal input tensors.',
            'Raw valid_timing remains unchanged; process polling cannot establish exclusive GPU use.',
            'Copy receipts are logical payload operations, not measured DRAM. No new task-quality pass from string equality.'])


def render(report):
    b, c = report['arms'][BASE], report['arms'][DELTA]
    lines = ['# Delta commit 配对吞吐审计', '',
        '只统计正式 3 对；两臂各 1 次预热剔除。模型常驻、串行单请求吞吐，token 数包含 EOS。', '',
        '| 指标 | Active 原 commit | Delta commit |', '|---|---:|---:|',
        f"| 总输出 token | {b['total_output_tokens']} | {c['total_output_tokens']} |",
        f"| 总整请求 wall，秒 | {b['total_e2e_wall_s']:.6f} | {c['total_e2e_wall_s']:.6f} |",
        f"| 总 token / 总 wall，token/s | {b['total_tokens_per_total_e2e_s']:.4f} | {c['total_tokens_per_total_e2e_s']:.4f} |",
        f"| 整请求 wall 中位数，秒 | {b['median_e2e_wall_s']:.6f} | {c['median_e2e_wall_s']:.6f} |", '',
        f"观察到的总吞吐变化：{report['aggregate_throughput_increase_pct']:+.3f}%。未据此自动认定稳定加速。", '',
        '| Repeat | 先运行 | wall 下降 | 吞吐上升 |', '|---|---|---:|---:|']
    for p in report['pairs']:
        lines.append(f"| {p['repeat']} | {'delta' if p['first_arm']==DELTA else 'active'} | {p['wall_reduction_pct']:+.3f}% | {p['throughput_increase_pct']:+.3f}% |")
    lines += ['', '正式配对完整 output IDs、result.rounds、forward counts、precision、输入网格与时间戳逐项相同。',
              '', '## 提交回执和硬件观察', '']
    measured = [r for r in report['rows'] if r['phase']=='measure']
    bc, dc = (next(r['commit'] for r in measured if r['arm']==arm) for arm in (BASE,DELTA))
    lines += [f"每次提交 {dc['committed_kv_tokens']} 个 KV token，其中 {dc['already_in_place_tokens']} 行原位、{dc['moved_tokens']} 行需要移动。",
        f"Target commit 逻辑读写由 {bc['copy_total_bytes']/2**20:.6f} MiB 降为 {dc['copy_total_bytes']/2**20:.6f} MiB，少 {(bc['copy_total_bytes']-dc['copy_total_bytes'])/2**20:.6f} MiB。",
        f"Delta 每请求回执记录 {dc['recorded_payload_kernel_launches']} 次 payload kernel；原 commit 未记录实际 launch 数，未凭源码推断冒充实测。",
        f"两臂最终 owned KV storage 均 {bc['owned_storage_bytes']} bytes；这轮不减少固定驻留容量。", '',
        f"正式各 cell 进程观察次数：{[r['hardware']['process_query_count'] for r in measured]}；错误次数：{[r['hardware']['process_query_failures'] for r in measured]}。",
        f"观测 foreign PID 并集：{sorted({pid for r in measured for pid in r['hardware']['foreign_pids']})}；已知图形 PID 之外：{sorted({pid for r in measured for pid in r['hardware']['foreign_except_known_graphics']})}。",
        f"原始 valid_timing：{[r['valid_timing'] for r in measured]}，未改写。已知图形 PID={report['known_graphics_pids']} 是人工提供的既有身份，仍未扣除其活动。",
        '单视频三次配对均出现小幅提高，但不能称严格独占或稳定跨视频收益。', '',
        '详见 JSON 的逐条计账、时钟/温度观察、原 timing_reasons 与三阶段/整请求峰值。', '', '## 范围限制', '']
    lines += ['- '+limit for limit in report['limits']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--known-graphics-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    report = audit(args.run, args.known_graphics_pid)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.out/'audit.md').write_text(render(report))
    print(json.dumps({'throughput_change_pct':report['aggregate_throughput_increase_pct'],
                     'mechanical_pass':True, 'pairs':report['pairs']}, indent=2))


if __name__ == '__main__':
    main()
