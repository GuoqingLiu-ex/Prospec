"""Build provisional, A-only FAD priors/costs from accepted ordinary results.

No B selection, model training, interpolation, or invented production overhead.
The new feedback/controller overhead must still be measured in full generation.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

CONTEXT_EDGES=(1024,2048,4096)
ACCEPTANCE_EDGES=(1,3,5)
ALLOWED=tuple(range(2,9))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical_hash(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def aggregate(samples, *, bindings):
    costs=defaultdict(lambda:defaultdict(list));priors=defaultdict(lambda:defaultdict(list))
    shapes={};seen=set();requests=set()
    for row in samples:
        if row['split']!='D80-A':raise ValueError('calibration accepts A only')
        identity=(row['request_id'],row['round_index'],row['depth'])
        if identity in seen:raise ValueError('duplicate state/depth')
        seen.add(identity);requests.add(row['request_id'])
        depth=row['depth']
        if type(depth) is not int or depth not in ALLOWED:raise ValueError('allowed depth')
        v,q=row['visual_tokens'],row['query_tokens']
        if type(v) is not int or v<0 or type(q) is not int or q<1:raise ValueError('shape metadata')
        if type(row['first_tree']) is not bool or type(row['context_length']) is not int or row['context_length']<1:
            raise ValueError('first/context metadata')
        cost,g=row['cost'],row['G_tree']
        if type(cost) not in (int,float) or not math.isfinite(cost) or cost<=0:raise ValueError('actual measured cost required')
        if type(g) is not int or not 1<=g<=depth+1 or type(row['censored']) is not bool:raise ValueError('actual G/censor')
        shape=f'v{v}_q{q}'
        shapes[shape]=dict(visual_tokens=v,query_tokens=q)
        key=f's{shape}|f{int(row["first_tree"])}|c{bisect_right(CONTEXT_EDGES,row["context_length"])}'
        costs[key][str(depth)].append(float(cost))
        if not row['censored']:priors[key][str(depth)].append(g)
    if not costs:raise ValueError('no A measurements')
    for key,ds in costs.items():
        if set(ds)!={str(d) for d in ALLOWED}:raise ValueError('incomplete measured depth bucket')
    return dict(schema='fad_full_calibration_v1',split='D80-A',kind='measured_device_round_seconds',unit='seconds',
        **bindings,context_edges=list(CONTEXT_EDGES),acceptance_edges=list(ACCEPTANCE_EDGES),shape_metadata=shapes,
        cost_scope='draft_plus_target_verify_and_commit_round',estimated_additional_cost_s=0.,deployment_cost_validated=False,
        costs={k:{d:dict(cost=math.fsum(v)/len(v),samples=2*len(v)) for d,v in ds.items()} for k,ds in costs.items()},
        actual_priors={k:{d:dict(sum_G=sum(v),samples=len(v)) for d,v in ds.items()} for k,ds in priors.items()},
        provenance=dict(requests=len(requests),state_depth_measurements=len(seen),A_only=True,B_used=False,T200_used=False,
            costs_include_first_tree_adaptor=True,costs_exclude_new_controller_and_feedback=True,
            query_tokens_scope='raw selected_hidden.shape[1], before first-tree visual contraction',
            source_sha256_scope='ordinary measurement runner, not new full FAD implementation',
            aggregation='mean of actual per-state mean of two timed repeats; observed priors count each state once',
            hyperparameters_selected_by_holdout=False,typed_priors_supplied=False))


def build(analysis_path, mirror):
    analysis_path=Path(analysis_path);mirror=Path(mirror)
    analysis=read(analysis_path)
    if analysis.get('cost_table_eligible') is not True or analysis.get('states')!=998 or analysis.get('measured_intervals')!=15968:
        raise ValueError('complete accepted ordinary cost analysis required')
    states_path=analysis_path.parent/'validated_states.jsonl'
    if sha(states_path)!=analysis['validated_states_sha256']:raise ValueError('validated states changed')
    states=[json.loads(s) for s in states_path.read_text().splitlines()]
    states=[s for s in states if s['split']=='D80-A']
    report=read(mirror/'cost_diagnostics/DEPTH_COST_REPORT.json')
    if report['status']!='completed' or not report['source_recheck']['passed']:raise ValueError('terminal source not accepted')
    indexed={c['request_id']:(i,c) for i,c in enumerate(report['cases'])}
    by_request=defaultdict(list)
    for state in states:by_request[state['request_id']].append(state)
    samples=[];flags_seen=[];raw_receipts={}
    for request_id,ss in by_request.items():
        index,case=indexed[request_id];folder=mirror/f'cost_diagnostics/cases/{index:02d}'
        for name in ('ordinary/records.jsonl','ordinary/output.json','ordinary/precision.json'):
            if sha(folder/name)!=case['artifact_sha256'][name]:raise ValueError('raw measurement changed: '+name)
            raw_receipts[f'{index:02d}/{name}']=sha(folder/name)
        output=read(folder/'ordinary/output.json')['result']
        visual=output['forward_counts']['merge_visual_tokens_in']
        precision=read(folder/'ordinary/precision.json')
        if precision['before']!=precision['after']:raise ValueError('precision changed')
        flags_seen.append(precision['before'])
        rows=defaultdict(list)
        for line in (folder/'ordinary/records.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['phase']=='measure':rows[(r['round_index'],r['D'])].append(r)
        for state in ss:
            for depth in ALLOWED:
                rr=rows[(state['round_index'],depth)]
                if len(rr)!=2 or {x['repeat'] for x in rr}!={0,1}:raise ValueError('two formal repeats required')
                if rr[0]['shape']!=rr[1]['shape'] or rr[0]['censored']!=rr[1]['censored']:raise ValueError('repeat shape/censor mismatch')
                shape=rr[0]['shape'];measured=state['depths'][str(depth)]
                mean=math.fsum(r['wall_s'] for r in rr)/2
                if mean!=measured['C_core_s']:raise ValueError('cost differs from audited state')
                samples.append(dict(split='D80-A',request_id=request_id,round_index=state['round_index'],depth=depth,
                    visual_tokens=visual,query_tokens=shape['selected_hidden_shape'][1],first_tree=shape['first_tree'],
                    context_length=shape['cache_before'],cost=mean,G_tree=measured['G_tree'],censored=rr[0]['censored']))
    if not flags_seen or any(f!=flags_seen[0] for f in flags_seen):raise ValueError('mixed precision flags')
    bindings=dict(device_uuid=report['cuda_device']['uuid'],source_sha256=report['entry_sha256'],
        precision_sha256=canonical_hash(flags_seen[0]),measurement_receipt_sha256=sha(analysis_path),
        prior_receipt_sha256=sha(states_path))
    table=aggregate(samples,bindings=bindings)
    table['precision_flags']=flags_seen[0]
    table['provenance'].update(A_raw_file_sha256=raw_receipts,validated_A_states=len(states),actual_policy_selection_pending=True)
    return table


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--mirror',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();table=build(a.analysis,a.mirror)
    out=Path(a.out)
    with out.open('x') as f:json.dump(table,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(out),sha256=sha(out),buckets=len(table['costs']),requests=table['provenance']['requests'],
        states=table['provenance']['validated_A_states'],deployment_cost_validated=False)))


if __name__=='__main__':main()
