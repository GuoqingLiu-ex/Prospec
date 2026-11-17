"""FAD actual-model neutral check and prespecified same-parent depth screen.

Stdlib import only. All snapshots, trace exports, losing trials and Python
feedback are paid diagnostics. Shared-card data is never ordinary throughput.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from contextlib import contextmanager
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))
from combined_visual_experiment.focusdraft import run_g1_real_system as real

g0, old_g1 = real.g0, real.old_g1
require, save, digest, read_json, jsonl = real.require, real.save, real.digest, real.read_json, real.jsonl
SCHEMA = 'fad_actual_same_parent_depth_screen_v1'
REAL_KIND = 'actual_fad_depth_diagnostics'
STATE_SHA = '1e777bbe4e517552b7cec94d52a7916bbc1873caaa32966cd5ab4eb203f8f9c2'
NOTE = ('Actual-model diagnostic only: full native KV copies, trace exports, '
        'all depth trials and CPU feedback included. No ordinary TPS, DRAM, '
        'energy or deployment-memory conclusion.')


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def token_prefix_sha(ids):
    # Exact sampler representation, not json_sha's sorted/compact convention.
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def bind_state_manifest(path, expected_sha, requests, reference):
    require(expected_sha == STATE_SHA and digest(path) == STATE_SHA, 'state manifest freeze differs')
    m = read_json(path)
    require(m['schema'] == 'fad_prespecified_same_state_sampling_v1'
            and m['D80_sha256'] == old_g1.D80_SHA and m['FocusDraft_MANIFEST_sha256'] == g0.MANIFEST_SHA,
            'state manifest source bindings differ')
    require(m['C0_artifact_hashes'] == reference['hashes'], 'state C0 hashes differ')
    require(m['method_outcomes_used_for_sampling'] is False and m['GPU_execution'] is False,
            'only original prespecified state locators supported')
    require(m['depths'] == list(range(1, 9)) and m['beam_width'] == 8
            and m['verify_nodes_including_root_cap'] == 30
            and m['actual_verify_nodes'] == {str(d): 9 if d == 1 else 30 for d in range(1, 9)},
            'depth or actual node contract differs')
    require([r['request_id'] for r in m['rows']] == [r['request_id'] for r in requests]
            and m['requests'] == 80 and m['states'] == 998, 'state membership differs')
    bound = {}
    count = 0
    for req, row in zip(requests, m['rows']):
        saved = reference['selected'][req['request_id']]['row']
        rounds, ids = saved['result']['rounds'], saved['output_token_ids']
        require((row['dataset'], row['split']) == (req['dataset'], req['split']), 'state dataset/split differs')
        require(row['reference_output_length'] == len(ids) and row['reference_stop'] == saved['stop_reason']
                and row['reference_tree_rounds'] == sum(r['kind'] == 'tree' for r in rounds),
                'state reference outcome differs')
        indices = [s['round_index'] for s in row['sampled_states']]
        require(indices == sorted(set(indices)) and 1 <= len(indices) <= 13, 'invalid sampled round indices')
        states = {}
        for s in row['sampled_states']:
            i = s['round_index']
            require(type(i) is int and 0 <= i < len(rounds) and rounds[i]['kind'] == 'tree', 'state locator is not a tree')
            r = rounds[i]
            offset = sum(len(p['submitted']) for p in rounds[:i])
            require(s['tree_round_index'] == sum(p['kind'] == 'tree' for p in rounds[:i])
                    and s['output_prefix_tokens'] == offset and s['output_prefix_sha256'] == token_prefix_sha(ids[:offset])
                    and s['expected_root_token'] == r['submitted'][0]
                    and s['target_cache_before'] == r['target_cache_before']
                    and s['semantic_rope_base'] == r['semantic_rope_base']
                    and s['reference_accepted_draft'] == r['accepted_draft']
                    and s['reference_tree_submitted_including_root'] == len(r['submitted']),
                    'state locator differs from raw C0 round/prefix')
            states[i] = deepcopy(s)
        bound[req['request_id']] = states
        count += len(states)
    require(count == 998, 'state count differs')
    return bound


def validate_live_locator(parent, locator):
    c = parent.controls
    require(len(c['generated']) == locator['output_prefix_tokens']
            and token_prefix_sha(c['generated']) == locator['output_prefix_sha256']
            and c['root'] == locator['expected_root_token']
            and parent.cache_length == locator['target_cache_before']
            and parent.cache_length + c['delta'] == locator['semantic_rope_base'],
            'live parent is not the prespecified C0 state')
    return dict(prefix_exact=True, cache_boundary_exact=True, root_exact=True,
                semantic_position_exact=True,
                scope='Live native snapshot boundary plus original C0 prefix; KV bytes are owned by the snapshot, not this JSON locator.')


def compare_branch_output(trial, prefix_len, reference_ids):
    expected = reference_ids[prefix_len:prefix_len + trial['all_new_output_tokens']]
    actual = [t for r in trial['step_rounds'] for t in r['submitted']]
    require(len(actual) == trial['all_new_output_tokens'], 'trial output ledger differs')
    require(trial['output_ids'][prefix_len:] == actual, 'trial full IDs do not reconstruct its branch')
    return dict(reference_prefix_ids_exact=actual == expected, actual_ids=actual,
                reference_ids=expected, automatic_near_tie_pass=False,
                scope='One actual branch from the same parent, not a full adaptive generation.')


def compare_trace_prefixes(trials):
    """Observed search-prefix check; neither a future guarantee nor Target test."""
    require([t['actual_depth'] for t in trials] == list(range(1, 9)), 'all eight actual depths required')
    result = []
    for shallow, deeper in zip(trials, trials[1:]):
        a, b = shallow['trace'], deeper['trace']
        n = len(a['token_ids'])
        dtype_exact = a['score_dtype'] == b['score_dtype']
        bytes_exact = dtype_exact and b['scores_bytes_hex'].startswith(a['scores_bytes_hex'])
        checks = {k: a[k] == b[k][:n] for k in ('token_ids', 'parent_ids', 'scores', 'generation_depth')}
        checks['checkpoints'] = a['checkpoints'] == b['checkpoints'][:shallow['actual_depth']]
        checks['score_dtype'] = dtype_exact; checks['score_bytes'] = bytes_exact
        checks['root'] = a['root_token'] == b['root_token']
        result.append(dict(shallow=shallow['actual_depth'], deeper=deeper['actual_depth'],
                           passed=all(checks.values()), checks=checks))
    return dict(passed=all(r['passed'] for r in result), pairs=result,
                target_prefix_invariance_measured=False,
                scope='Measured candidate/score/checkpoint prefixes only. Target numerical equivalence is separate.')


def aggregate(cases, requests, mode):
    exact = (len(cases) == len(requests) and [c['request_id'] for c in cases] == [r['request_id'] for r in requests]
             and all(c.get('completed') is True and c.get('input_exact') is True and c.get('flags_exact') is True
                     and c.get('c0_comparison', {}).get('exact') is True
                     and c.get('neutral_comparison', {}).get('exact') is True
                     and c.get('execution_kind') == REAL_KIND for c in cases))
    states = sum(c.get('screen_states', 0) for c in cases)
    result = dict(expected_requests=len(requests), recorded_requests=len(cases), complete_exact_cases=exact,
                  states=states, failed_request_ids=[c['request_id'] for c in cases if not c.get('completed')],
                  deployment_hardware_eligible=False, scope=NOTE)
    if mode == 'neutral_e24':
        from collections import Counter
        result['neutral_passed'] = bool(exact and len(requests) == 24
             and Counter(r['dataset'] for r in requests) == {'nextqa': 12, 'videodetailcaption': 12})
    else:
        result['screen_complete'] = bool(exact and len(requests) == 80 and states == 998)
        result['branch_reference_prefix_failures'] = sum(c.get('branch_reference_prefix_failures', 0) for c in cases)
        result['search_prefix_failures'] = sum(c.get('search_prefix_failures', 0) for c in cases)
        result['FAD_gain_accepted'] = False
    return result


def validate_neutral_execution(receipt, neutral, request_key, execution_sha, actual_flags):
    require(receipt['schema'] == 'fad_actual_state_depth_executor_v1'
            and receipt['source_receipt'] == execution_sha
            and receipt['precision_receipt'] == json_sha(actual_flags)
            and receipt['request_key'] == request_key
            and receipt['all_trial_coverage_validated'] is True
            and not receipt['poisoned'] and not receipt['failures'],
            'neutral executor source/precision/health binding differs')
    common = {k:v for k,v in neutral['result']['forward_counts'].items()
              if k not in ('target_tree_attention_policy_calls', 'target_grouped_attention_calls')}
    public_rounds = [p for p in neutral['result']['rounds'] if p['kind'] != 'prefill_terminal']
    require(receipt['all_trial_forward_counts'] == common
            and receipt['all_trial_rounds'] == public_rounds,
            'neutral all-trial counts/rounds differ from actual public result')
    trees = [p for p in public_rounds if p['kind'] == 'tree']
    history = receipt['depth_trial_history']
    require(len(history) == len(trees) == common['target_verify_calls']
            and receipt['actual_depths'] == [h['actual_depth'] for h in history]
            and all(h['actual_depth'] == 4 and h['tree_nodes'] == p['tree_nodes'] == 30
                    for h,p in zip(history,trees))
            and receipt['actual_verify_nodes'] == sum(h['tree_nodes'] for h in history)
                    == common['target_verify_query_tokens'],
            'neutral actual depth history/node totals differ from public verification')
    return dict(passed=True, actual_trees=len(trees), actual_verify_nodes=receipt['actual_verify_nodes'])


@contextmanager
def preserve_failure_receipt(out, executor, policy):
    """Save evidence before the executor and attention contexts close."""
    try:
        yield
    except BaseException as exc:
        record = dict(error=repr(exc), execution=executor.execution_receipt(),
                      policy=policy.summary(), all_trial_coverage_accepted=False)
        save(Path(out) / 'failed_execution.json', record)
        raise


def validate_neutral_gate(directory, requests, reference, execution):
    base = Path(directory); report = read_json(base / 'FAD_SCREEN_REPORT.json')
    require(report['schema'] == SCHEMA and report['mode'] == 'neutral_e24' and report['status'] == 'completed',
            'actual FAD E24 neutral is incomplete')
    require(report == read_json(base / 'status.json') and report['cases'] == jsonl(base / 'cases.jsonl'), 'neutral final records differ')
    require(report['entry_sha256'] == digest(__file__)
            and report['execution_source_freeze'] == report['execution_source_freeze_after'] == execution
            and report['source_recheck']['passed'] is True, 'neutral execution source binding differs')
    require(report['request_sha256'] == g0.E24_SHA and report['manifest_sha256'] == g0.MANIFEST_SHA
            and report['c0_artifacts_sha256'] == reference['hashes'], 'neutral data binding differs')
    for i, (case, req) in enumerate(zip(report['cases'], requests)):
        f = base / 'cases' / f'{i:02d}'
        require(case == read_json(f / 'case.json') and case['request_id'] == req['request_id'], 'neutral identity differs')
        c0, neutral = read_json(f / 'c0_rerun.json'), read_json(f / 'neutral.json')
        g0.validate_result(c0); g0.validate_result(neutral)
        saved = reference['selected'][req['request_id']]
        require(g0.compare_c0(c0, saved['row']) == case['c0_comparison']
                and g0.compare_c0(neutral, c0) == case['neutral_comparison'], 'neutral raw C0 comparisons differ')
        inp, obs = read_json(f / 'input.json'), read_json(f / 'observations.json')
        require(all(inp[k] == saved['input'][k] for k in ('input_signature','timestamps','frame_indices',
                    'sampled_metadata','sampled_provenance')), 'neutral raw input differs')
        flags = {k:saved['precision'][k] for k in ('before','after')}
        require(obs['c0']['flags'] == obs['neutral']['flags'] == flags, 'neutral raw flags differ')
        receipt = read_json(f / 'neutral_execution.json')
        require(receipt == obs['neutral']['execution'],
                'neutral all-trial receipt differs')
        validate_neutral_execution(receipt,neutral,req['request_id'],execution['sha256'],flags['before'])
    gate = aggregate(report['cases'], requests, 'neutral_e24')
    require(gate == report['gate'] and gate['neutral_passed'], 'all24 raw neutral exact required')
    return dict(passed=True, report_sha256=digest(base / 'FAD_SCREEN_REPORT.json'), gate=gate)


class DepthOps(real.RealSystemOps):
    execution_kind = REAL_KIND

    def __init__(self, args, reference, requests):
        super().__init__(args, reference, requests)
        from combined_visual_experiment.fad_vispec.state_depth_executor import StateDepthExecutor
        self.executor_type = StateDepthExecutor
        self.metadata.update(execution_kind=REAL_KIND, focus_pool_installed=False,
                             actual_depth_actions=list(range(1, 9)), diagnostic_only=True)

    def executor(self, packed, request):
        return self.executor_type(self.spec, packed, request_key=request['request_id'],
            source_receipt=self.args.execution_source_freeze_sha256,
            precision_receipt=json_sha(self.flags()), max_new_tokens=512,
            tree_budget=self.budget, eos_token_ids=[151645, 151643])

    def neutral(self, packed, request, out):
        self.reset(); memory = self.begin_memory(); before = self.flags()
        with self.grouped() as policy:
            with self.executor(packed, request) as executor, preserve_failure_receipt(out, executor, policy):
                parent = executor.begin()
                while parent.controls['stop_reason'] is None:
                    parent, trial = executor.step(parent, actual_depth=4)
                    # Stream each full trace rather than retaining a second history.
                    with (Path(out) / 'neutral_trace.jsonl').open('a') as f:
                        f.write(json.dumps(trial, ensure_ascii=False, allow_nan=False)+'\n')
                joint = executor.finish(parent); top2 = deepcopy(parent.controls['top2'])
                executor.validate_coverage(policy); self.complete_result(joint, policy)
                receipt = executor.execution_receipt()
                del parent
        save(Path(out) / 'neutral_execution.json', receipt)
        return self.result_row(joint), dict(flags=dict(before=before, after=self.flags()), policy=policy.summary(),
            execution=receipt, top2=top2, memory=self.end_memory(memory), diagnostic_only=True)

    def screen(self, packed, request, out, states, reference_ids):
        self.reset(); memory = self.begin_memory(); before = self.flags()
        counts = dict(screen_states=0, branch_reference_prefix_failures=0, search_prefix_failures=0)
        with self.grouped() as policy:
            with self.executor(packed, request) as executor, preserve_failure_receipt(out, executor, policy):
                parent = executor.begin(); round_index = 0
                while parent.controls['stop_reason'] is None:
                    if round_index in states:
                        locator = states[round_index]
                        binding = validate_live_locator(parent, locator)
                        trials, keep = [], None
                        for depth in range(1, 9):
                            child, trial = executor.step(parent, actual_depth=depth)
                            trial['reference_comparison'] = compare_branch_output(trial,
                                len(parent.controls['generated']), reference_ids)
                            trials.append(trial)
                            # Preserve every successful branch even if a later D fails.
                            with (Path(out) / 'depth_trial_events.jsonl').open('a') as f:
                                f.write(json.dumps(dict(request_id=request['request_id'],
                                    round_index=round_index, locator=locator, live_binding=binding,
                                    trial=trial, diagnostic_only=True),
                                    ensure_ascii=False, allow_nan=False)+'\n')
                            if depth == 4: keep = child
                            del child
                        prefix = compare_trace_prefixes(trials)
                        record = dict(request_id=request['request_id'], round_index=round_index,
                            locator=locator, live_binding=binding,
                            trials=trials, search_prefix_comparison=prefix, diagnostic_only=True)
                        with (Path(out) / 'depth_trials.jsonl').open('a') as f:
                            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                        counts['screen_states'] += 1
                        counts['branch_reference_prefix_failures'] += sum(not t['reference_comparison']['reference_prefix_ids_exact'] for t in trials)
                        counts['search_prefix_failures'] += not prefix['passed']
                        require(prefix['passed'], 'actual candidate/score search prefixes differ across depths; diagnostic saved')
                        require(keep is not None, 'canonical D4 branch missing')
                        parent = keep
                        del keep, trials, record, trial
                    else:
                        parent, _ = executor.step(parent, actual_depth=4)
                    round_index += 1
                require(counts['screen_states'] == len(states), 'prespecified states not all reached')
                joint = executor.finish(parent); top2 = deepcopy(parent.controls['top2'])
                executor.validate_coverage(policy)
                # Coverage counts all eight trials; output counts only canonical D4.
                joint.precision = dict(joint.precision, target_sdpa='math_prefill+grouped_fp32_math_tree',
                    target_prefill_attention_backend='math', target_verify_attention_backend='grouped_two_bmm_fp32_math',
                    target_tree_attention_mode='grouped', target_tree_math_precision=self.flags(),
                    target_cache_policy='native_full_dynamic_cache', draft_attention_backend='unchanged',
                    tree_attention_bitwise_equivalence_claim=False)
                calls = 28 * joint.forward_counts['target_verify_calls']
                joint.forward_counts.update(target_tree_attention_policy_calls=calls, target_grouped_attention_calls=calls)
                receipt = executor.execution_receipt(); del parent
        save(Path(out) / 'screen_execution.json', receipt)
        return self.result_row(joint), dict(flags=dict(before=before, after=self.flags()), policy=policy.summary(),
            execution=receipt, top2=top2, memory=self.end_memory(memory), **counts)


def run_case(request, saved, out, ops, mode, states=None):
    if mode == 'neutral_e24':
        return real.run_case(request, saved, out, ops, mode)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    case = dict(request_id=request['request_id'],dataset=request['dataset'],split=request['split'],completed=False,
                stage='prepare',execution_kind=ops.execution_kind,diagnostic_only=True,throughput_eligible=False)
    sampler = None
    try:
        sampler = ops.sampler(); sampler.start(); ops.reset()
        packed, sampled = ops.prepare(request)
        inp = dict(input_signature=ops.signature(packed,list(sampled.timestamps)),timestamps=list(sampled.timestamps),
            frame_indices=list(sampled.metadata['frames_indices']),sampled_metadata=dict(sampled.metadata),
            sampled_provenance=dict(sampled.provenance),signature_logical_payload=ops.signature_payload(packed))
        save(out/'input.json',inp)
        require(all(inp[k]==saved['input'][k] for k in ('input_signature','timestamps','frame_indices','sampled_metadata','sampled_provenance')), 'screen input differs')
        case['input_exact']=True;case['stage']='original_c0'
        c0,cobs=ops.original(packed);save(out/'c0_rerun.json',c0);g0.validate_result(c0)
        case['c0_comparison']=g0.compare_c0(c0,saved['row']);require(case['c0_comparison']['exact'],'C0 rerun differs')
        case['stage']='same_parent_depths'
        neutral,nobs=ops.screen(packed,request,out,states,c0['output_token_ids'])
        save(out/'neutral.json',neutral);save(out/'observations.json',dict(c0=cobs,neutral=nobs));g0.validate_result(neutral)
        case['neutral_comparison']=g0.compare_c0(neutral,c0);require(case['neutral_comparison']['exact'],'canonical D4 continuation differs from C0')
        flags={k:saved['precision'][k] for k in ('before','after')}
        case['flags_exact']=cobs['flags']==nobs['flags']==flags;require(case['flags_exact'],'actual precision differs')
        case.update({k:nobs[k] for k in ('screen_states','branch_reference_prefix_failures','search_prefix_failures')})
        case.update(completed=True,stage='completed',rates={'neutral':real.rate(neutral)})
    except Exception as exc:
        case.update(error=repr(exc),traceback=g0.traceback.format_exc())
    finally:
        try: ops.synchronize()
        except Exception as exc: case.update(completed=False,synchronize_error=repr(exc))
        try: case['hardware']=sampler.stop() if sampler else {'unavailable':True}
        except Exception as exc: case['hardware']=dict(unavailable=True,error=repr(exc))
        save(out/'case.json',case)
    return case


def write_report(out, report):
    save(Path(out)/'status.json',report);save(Path(out)/'FAD_SCREEN_REPORT.json',report)
    gate=report.get('gate',{})
    passed = report.get('status') == 'completed' and (gate.get('neutral_passed')
        if report['mode']=='neutral_e24' else gate.get('screen_complete'))
    lines=['# FAD 真实诊断：'+('通过' if passed else '未完成或未通过'),'',
        f"模式 {report['mode']}；记录 {len(report['cases'])}/{gate.get('expected_requests','?')}。",'',
        '真实同父状态 Draft 深度诊断；固定 K8、含root验证上限30。D1真实9个节点，其余真实30。',
        '全快照、trace、未选择分支和CPU反馈均为诊断成本；共享卡不下TPS、DRAM、能耗或总显存结论。']
    if report.get('error'):lines+=['','错误：'+report['error']]
    (Path(out)/'FAD_SCREEN_REPORT.md').write_text('\n'.join(lines)+'\n')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',choices=('neutral_e24','screen_d80'),required=True)
    for name in ('out','requests','requests-sha256','c0-dir','expected-gpu-uuid','execution-source-freeze','execution-source-freeze-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--physical-gpu',type=int,choices=(3,6),required=True)
    p.add_argument('--c0-project');p.add_argument('--fad-neutral-dir');p.add_argument('--state-manifest');p.add_argument('--state-manifest-sha256')
    a=p.parse_args(argv)
    require(a.requests_sha256==(g0.E24_SHA if a.mode=='neutral_e24' else old_g1.D80_SHA),'frozen E24 or D80 only')
    require(a.mode!='screen_d80' or (a.fad_neutral_dir and a.state_manifest and a.state_manifest_sha256==STATE_SHA),'actual FAD neutral and state manifest required')
    require(not Path(a.out).exists(),'fresh output directory required')
    return a


def main(argv=None):
    args=parse_args(argv)
    execution=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
    sys.path.insert(0,str(g0.HERE))
    from run_c0_control import read_requests,observe,verify_shared_guard
    requests=read_requests(args.requests,args.requests_sha256);md=Path(args.requests).parent
    require(digest(md/'MANIFEST.json')==g0.MANIFEST_SHA,'manifest differs')
    e24=read_requests(md/'E24.jsonl',g0.E24_SHA)
    if args.mode=='neutral_e24':
        require(requests==e24,'complete frozen E24 required');reference=g0.load_reference(args.c0_dir,e24);gate=None;states={}
    else:
        reference=old_g1.bind_d80(args.c0_dir,requests,e24)
        gate=validate_neutral_gate(args.fad_neutral_dir,e24,dict(reference,selected=reference['g0_selected']),execution)
        states=bind_state_manifest(args.state_manifest,args.state_manifest_sha256,requests,reference)
    manifest=read_json(md/'MANIFEST.json');c0_project=g0.resolve_c0_project(args.c0_dir,args.c0_project)
    sources=g0.check_sources(reference,manifest,c0_project=c0_project);sources.update(execution['source_sha256'])
    lock,lock_receipt=g0.load_lock_without_torch(g0.PROJECT/'direct_opt/run_joint.py')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,mode=args.mode,run_id=uuid.uuid4().hex,pid=os.getpid(),status='starting',
        args=vars(args),request_sha256=digest(args.requests),manifest_sha256=g0.MANIFEST_SHA,
        c0_artifacts_sha256=reference['hashes'],source_sha256=sources,c0_project=str(c0_project),
        execution_source_freeze=execution,lock_source=lock_receipt,entry_sha256=digest(__file__),
        gate0=gate,cases=[],shared_guards=[],scope=NOTE)
    previous={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','MKL_NUM_THREADS')};failed=False
    try:
        for path in (g0.PROJECT,g0.PROJECT/'vendor',g0.PROJECT/'direct_opt',g0.PROJECT/'training_free'):sys.path.insert(0,str(path))
        from hardware_metrics import _make_backend
        with lock(args.physical_gpu,out) as metadata:
            report['lock']=metadata;backend=_make_backend(args.physical_gpu)
            try:
                for i in range(2):
                    obs=observe(backend);report['shared_guards'].append(obs)
                    verify_shared_guard(obs,args.expected_gpu_uuid,32*1024**3,os.getpid())
                    if i==0:g0.time.sleep(2)
            finally:backend.close()
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            import torch
            import run_joint
            require(digest(run_joint.__file__)==lock_receipt['sha256'],'lock source differs')
            for name,expected in lock_receipt['ast_sha256'].items():
                fn=ast.parse(inspect.getsource(getattr(run_joint,name))).body[0]
                require(hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()==expected,'lock AST differs')
            from grouped_gqa_runtime import math_precision_scope
            with math_precision_scope(),torch.inference_mode():
                ops=DepthOps(args,reference,requests);report.update(status='running',setup=ops.metadata)
                for i,request in enumerate(requests):
                    report['current']=dict(index=i,request_id=request['request_id']);write_report(out,report)
                    case=run_case(request,reference['selected'][request['request_id']],out/'cases'/f'{i:02d}',ops,args.mode,states.get(request['request_id']))
                    report['cases'].append(case)
                    with (out/'cases.jsonl').open('a') as f:f.write(json.dumps(case,ensure_ascii=False,allow_nan=False)+'\n')
                    report['gate']=aggregate(report['cases'],requests,args.mode);write_report(out,report)
                    require(case['completed'],'actual FAD diagnostic contract failed; saved raw data, later requests paused')
                report.update(status='completed',current=None);report['gate']=aggregate(report['cases'],requests,args.mode)
    except BaseException as exc:
        failed=True;report.update(status='failed',error=repr(exc),traceback=g0.traceback.format_exc());raise
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        source_error=None
        try:
            after=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
            report['execution_source_freeze_after']=after;require(after==execution,'execution sources changed')
            final=g0.check_sources(reference,manifest,c0_project=c0_project);final.update(after['source_sha256'])
            require(final==sources and digest(md/'MANIFEST.json')==g0.MANIFEST_SHA,'source/manifest changed')
            report['source_recheck']=dict(passed=True,execution_inventory_exact=True)
        except BaseException as exc:
            source_error=exc;report.update(status='failed',source_guard_error=repr(exc),source_recheck=dict(passed=False))
            report.setdefault('gate', {})['neutral_passed' if args.mode=='neutral_e24' else 'screen_complete'] = False
        report['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat();write_report(out,report)
        if source_error and not failed:raise RuntimeError('terminal FAD source guard failed') from source_error
    return report


if __name__=='__main__':
    main()
