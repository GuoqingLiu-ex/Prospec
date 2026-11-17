"""Idle-device FAD core parity/cost runner; stdlib-only import.

parity_gate reconstructs separate diagnostic and ordinary trajectories and hashes
actual device tensor bytes outside all timing. cost_d80 requires that raw gate.
One warmup and two measured permutations per frozen state are supplied in an
explicit hash-bound plan. Only D4 continues the request. Core cost excludes
snapshot restore/capture, source/parity exports, controller and feedback; it is
neither dynamic TPS nor whole deployment allocated/reserved acceptance.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT.parent) not in sys.path: sys.path.insert(0, str(PROJECT.parent))
from combined_visual_experiment.fad_vispec import run_depth_screen as screen
real, g0 = screen.real, screen.g0
require, save, digest, read_json, jsonl = screen.require, screen.save, screen.digest, screen.read_json, screen.jsonl
SCHEMA = 'fad_ordinary_depth_cost_v1'
PLAN_SCHEMA = 'fad_ordinary_depth_cost_plan_v1'
CORE_SCOPE = 'draft_plus_target_verify_and_commit_round'
ENDPOINT_S, MAX_GAP_S, PROCESS_S = 1.5, 2.0, .5
MINIMUM_FREE_BYTES = 32 * 1024**3
INPUT_KEYS = ('input_signature','timestamps','frame_indices','sampled_metadata','sampled_provenance')
NOTE = 'C_core only; excludes production controller/feedback and outside-core snapshots. No dynamic TPS, DRAM, energy or deployment whole-memory acceptance.'


# BEGIN REVIEWED PREDECESSOR HELPERS
PREDECESSOR_SCHEMA = 'fad_ordinary_parity_predecessor_binding_v1'
PREDECESSOR_ENTRY_SHA = '5124484cdccba368564ed6271ec7ff1a05ca8490f9b71a323eee829499b4d611'
PREDECESSOR_TEST_SHA = '45e23520bca6260172966dc41977727a325283b626970db0f898c8fb323faff0'
PREDECESSOR_FREEZE_SHA = 'c6fc3ac40eb2dd24e5c5110ab89554b5a8a63e53bd10bb612b5ba79436b55479'
ENTRY_REL = 'fad_vispec/run_depth_cost.py'
TEST_REL = 'fad_vispec/test_run_depth_cost.py'
# Exact reviewed edits, including main. No execution function/whole-main whitelist.
PREDECESSOR_SOURCE_EDITS = (("                and draft['tree_mask'] is None and not generated,'first-tree Draft state differs')",
  "                and not generated,'first-tree Draft state differs')\n"
  '        # Native prior request may leave this exact mask; author reset() clears it\n'
  '        # before the first Draft forward. Preserve its bytes/alias for parity.\n'
  "        if draft['tree_mask'] is not None:\n"
  '            '
  "tensor(draft['tree_mask'],'state/draft/tree_mask',dtype='torch.float32',shape=[1,1,8,24],device='cuda')"),
 ('def validate_parity_gate(directory, *, execution, plan_sha, reference, request_ids, states, '
  'expected_uuid):\n'
  "    base=Path(directory);report=read_json(base/'DEPTH_COST_REPORT.json')",
  'def validate_parity_gate(directory, *, execution, plan_sha, reference, request_ids, states, expected_uuid,\n'
  '                         predecessor_binding=None, predecessor_binding_sha256=None):\n'
  "    require((predecessor_binding is None)==(predecessor_binding_sha256 is None),'predecessor binding and "
  "SHA required together')\n"
  '    predecessor=validate_predecessor_binding(predecessor_binding,predecessor_binding_sha256,\n'
  '        directory=directory,execution=execution) if predecessor_binding is not None else None\n'
  "    expected_execution=predecessor['predecessor_execution'] if predecessor else execution\n"
  '    expected_entry=PREDECESSOR_ENTRY_SHA if predecessor else digest(__file__)\n'
  "    base=Path(directory);report=read_json(base/'DEPTH_COST_REPORT.json')"),
 ("and report['entry_sha256']==digest(__file__),'ordinary parity run is incomplete/different')",
  "and report['entry_sha256']==expected_entry,'ordinary parity run is incomplete/different')"),
 ("report['execution_source_freeze']==report['execution_source_freeze_after']==execution\n",
  "report['execution_source_freeze']==report['execution_source_freeze_after']==expected_execution\n"),
 ("return dict(passed=True,raw_compared=total,report_sha256=digest(base/'DEPTH_COST_REPORT.json'),\n"
  '        scope=',
  "return dict(passed=True,raw_compared=total,report_sha256=digest(base/'DEPTH_COST_REPORT.json'),\n"
  '        predecessor_binding=predecessor,\n'
  '        scope='),
 ('    a=p.parse_args(argv)\n',
  "    p.add_argument('--parity-predecessor-binding');p.add_argument('--parity-predecessor-binding-sha256')\n"
  '    a=p.parse_args(argv)\n'
  '    require((a.parity_predecessor_binding is None)==(a.parity_predecessor_binding_sha256 is '
  "None),'predecessor binding and SHA required together')\n"
  '    if a.parity_predecessor_binding is not None:\n'
  "        require(a.mode=='cost_d80' and a.parity_dir is not None,'predecessor only supports explicit cost "
  "parity')\n"
  '        require(Path(a.parity_dir).is_absolute() and Path(a.parity_predecessor_binding).is_absolute()\n'
  "                and hex_sha(a.parity_predecessor_binding_sha256),'absolute predecessor paths/valid SHA "
  "required')\n"),
 ("        reference=reference,request_ids=[r['request_id'] for r in "
  "requests],states=states,expected_uuid=args.expected_gpu_uuid) if args.mode=='cost_d80' else None",
  "        reference=reference,request_ids=[r['request_id'] for r in "
  'requests],states=states,expected_uuid=args.expected_gpu_uuid,\n'
  '        predecessor_binding=args.parity_predecessor_binding,\n'
  "        predecessor_binding_sha256=args.parity_predecessor_binding_sha256) if args.mode=='cost_d80' else "
  'None'),
 ("            report['source_recheck']=dict(passed=True,execution_inventory_exact=True)",
  "            if gate is not None and gate.get('predecessor_binding') is not None:\n"
  '                '
  "report['parity_predecessor_recheck']=recheck_predecessor_binding(gate['predecessor_binding'])\n"
  "            report['source_recheck']=dict(passed=True,execution_inventory_exact=True)"))


def predecessor_source_compatibility(old_source, new_source):
    require(hashlib.sha256(old_source.encode()).hexdigest()==PREDECESSOR_ENTRY_SHA,
            'unreviewed predecessor runner source')
    expected=old_source
    for before,after in PREDECESSOR_SOURCE_EDITS:
        require(expected.count(before)==1,'predecessor edit anchor differs')
        expected=expected.replace(before,after)
    begin='# BEGIN REVIEWED PREDECESSOR HELPERS'+'\n'
    end='# END REVIEWED PREDECESSOR HELPERS'+'\n\n'
    require(new_source.count(begin)==new_source.count(end)==1,'predecessor helper boundary differs')
    start=new_source.index(begin);stop=new_source.index(end,start)+len(end)
    require(new_source[:start]+new_source[stop:]==expected,
            'unreviewed source edit outside predecessor helpers (including main/timing/math)')
    return dict(passed=True,comparison='exact text after enumerated CPU edits and bounded new helper block',
        enumerated_edits=len(PREDECESSOR_SOURCE_EDITS),execution_functions_unchanged=True,
        main_only_exact_reviewed_insertions=True)


def _predecessor_absolute(value):
    require(type(value) is str and Path(value).is_absolute(),'predecessor paths must be explicit absolute paths')
    p=Path(value)
    require(str(p.resolve())==value,'predecessor paths must be canonical, without symlink/parent aliases')
    return p


def make_predecessor_binding(*, predecessor_project, predecessor_freeze, parity_directory,
                             parity_archive_receipt, current_execution, evidence=()):
    """CPU metadata builder for accessible paths; returns JSON, writes nothing.

    Deployers may render the identical schema using remote absolute paths and
    frozen hashes. validate_predecessor_binding must still read all live files.
    """
    project=_predecessor_absolute(str(predecessor_project));base=_predecessor_absolute(str(parity_directory))
    freeze=_predecessor_absolute(str(predecessor_freeze));archive=_predecessor_absolute(str(parity_archive_receipt))
    current=Path(current_execution['project'])
    return dict(schema=PREDECESSOR_SCHEMA,predecessor_project=str(project),predecessor_freeze=str(freeze),
        predecessor_freeze_sha256=digest(freeze),predecessor_entry_sha256=digest(project/ENTRY_REL),
        predecessor_test_sha256=digest(project/TEST_REL),parity_directory=str(base),
        parity_report_sha256=digest(base/'DEPTH_COST_REPORT.json'),parity_archive_receipt=str(archive),
        parity_archive_receipt_sha256=digest(archive),current_entry_sha256=digest(current/ENTRY_REL),
        current_test_sha256=digest(current/TEST_REL),current_execution_freeze_sha256=current_execution['sha256'],
        evidence=[dict(path=str(_predecessor_absolute(str(p))),sha256=digest(p)) for p in evidence])


def validate_predecessor_binding(path, sha256, *, directory, execution):
    """Rehash old/new inventories and old archive. This does NOT replace raw parity validation."""
    path=_predecessor_absolute(str(path));require(hex_sha(sha256) and digest(path)==sha256,'predecessor binding SHA differs')
    binding=read_json(path)
    keys={'schema','predecessor_project','predecessor_freeze','predecessor_freeze_sha256',
        'predecessor_entry_sha256','predecessor_test_sha256','parity_directory','parity_report_sha256',
        'parity_archive_receipt','parity_archive_receipt_sha256','current_entry_sha256','current_test_sha256',
        'current_execution_freeze_sha256','evidence'}
    require(type(binding) is dict and set(binding)==keys and binding['schema']==PREDECESSOR_SCHEMA,
            'predecessor binding schema/fields differ')
    require(binding['predecessor_freeze_sha256']==PREDECESSOR_FREEZE_SHA
        and binding['predecessor_entry_sha256']==PREDECESSOR_ENTRY_SHA
        and binding['predecessor_test_sha256']==PREDECESSOR_TEST_SHA,'predecessor must be the reviewed immutable 376-source run')
    old_project=_predecessor_absolute(binding['predecessor_project'])
    old_freeze=_predecessor_absolute(binding['predecessor_freeze'])
    base=_predecessor_absolute(binding['parity_directory']);root=old_project.parents[1]
    require(old_project==root/'code/combined_visual_experiment' and old_freeze==root/'EXECUTION_SOURCE_FREEZE.json'
        and base==root/'parity_diagnostics' and base==_predecessor_absolute(str(directory)), 'predecessor root/parity path binding differs')
    old=g0.check_execution_sources(old_freeze,PREDECESSOR_FREEZE_SHA,project=old_project)
    current=g0.check_execution_sources(execution['path'],execution['sha256'],project=execution['project'])
    require(current==execution and Path(current['project'])/ENTRY_REL==Path(__file__).resolve(),
            'current execution inventory/entry differs')
    old_inventory=read_json(old_freeze);new_inventory=read_json(current['path'])
    require(old['source_count']==current['source_count']==376 and set(old_inventory)==set(new_inventory),
            'predecessor/current inventories must contain the same complete 376 sources')
    changed={name for name in old_inventory if old_inventory[name]!=new_inventory[name]}
    require(changed=={ENTRY_REL,TEST_REL},'only reviewed runner/test may differ; other 374 sources must match')
    require(old_inventory[ENTRY_REL]==PREDECESSOR_ENTRY_SHA and old_inventory[TEST_REL]==PREDECESSOR_TEST_SHA
        and binding['current_entry_sha256']==new_inventory[ENTRY_REL]
        and binding['current_test_sha256']==new_inventory[TEST_REL]
        and binding['current_execution_freeze_sha256']==current['sha256'],'new/old entry/test/freeze binding differs')
    compatibility=predecessor_source_compatibility((old_project/ENTRY_REL).read_text(),Path(__file__).read_text())
    report_path=base/'DEPTH_COST_REPORT.json';require(digest(report_path)==binding['parity_report_sha256'],'predecessor parity report drift')
    source_map=read_json(report_path)['source_sha256']
    require(type(source_map) is dict and bool(source_map),'predecessor report source inventory missing')
    for name,expected_sha in source_map.items():
        # Old raw reports retain historical absolute labels, including symlinks.
        # Preserve those labels; validate their resolved file and exact bytes.
        require(type(name) is str and Path(name).is_absolute(),'historical dependency label must be absolute')
        resolved=Path(name).resolve(strict=True)
        require(resolved.is_file() and hex_sha(expected_sha) and digest(resolved)==expected_sha,
                'predecessor report dependency drift: '+name)
    archive_path=_predecessor_absolute(binding['parity_archive_receipt'])
    require(digest(archive_path)==binding['parity_archive_receipt_sha256'],'predecessor archive receipt drift')
    archive=read_json(archive_path)
    require(archive['schema']=='fad_ordinary_actual_terminal_readonly_archive_v1'
        and archive['ready'] is True and archive['mode']=='parity_gate' and archive['status']=='completed'
        and archive['source_count']==376 and archive['source_bindings_match'] is True
        and archive['execution']==old and archive['root']==str(root)
        and archive['raw_report_sha256']==binding['parity_report_sha256'],'predecessor archive source/report binding differs')
    archived=_predecessor_absolute(archive['archive'])
    require(archived.parent==root and archived.stat().st_size==archive['archive_bytes']
        and digest(archived)==archive['archive_sha256'],'predecessor archive bytes differ')
    files=archive['files'];require(type(files) is list and len(files)==archive['member_count'],'archive file inventory incomplete')
    names=set()
    for row in files:
        name=row['path'];item=(root/name).resolve()
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
            and item.is_relative_to(root) and name not in names,'archive unsafe/duplicate member')
        require(item.is_file() and item.stat().st_size==row['bytes'] and digest(item)==row['sha256'],'predecessor archived file drift: '+name)
        names.add(name)
    mandatory={str(p.relative_to(root)) for p in base.rglob('*') if p.is_file()}
    mandatory|={'EXECUTION_SOURCE_FREEZE.json','source_deployment_contract.json',
        'launch_parity/wrapper_exit.json','launch_parity/wrapper_status.json'}
    require(mandatory<=names,'archive omits parity/source/terminal artifacts')
    require(digest(root/'source_deployment_contract.json')==archive['contract_sha256'],'predecessor deployment contract drift')
    exit_=read_json(root/'launch_parity/wrapper_exit.json')
    require(exit_==read_json(root/'launch_parity/wrapper_status.json')==archive['wrapper_exit']
        and exit_['status']=='completed' and type(exit_['returncode']) is int and exit_['returncode']==0
        and read_json(report_path)['pid']==exit_['child_pid'],'predecessor wrapper/child terminal binding differs')
    command=exit_['command']
    require(command[1]==str(old_project/ENTRY_REL) and command.count('--mode')==1
        and command[command.index('--mode')+1]=='parity_gate' and command.count('--out')==1
        and command[command.index('--out')+1]==str(base),'predecessor actual entry/command differs')
    evidence=binding['evidence'];require(type(evidence) is list,'evidence must be an explicit list')
    for item in evidence:
        require(type(item) is dict and set(item)=={'path','sha256'} and hex_sha(item['sha256'])
            and digest(_predecessor_absolute(item['path']))==item['sha256'],'auxiliary independent evidence drift')
    return dict(schema=PREDECESSOR_SCHEMA,passed=True,binding_path=str(path),binding_sha256=sha256,
        binding=binding,predecessor_execution=old,current_execution=current,source_compatibility=compatibility,
        other_project_sources_unchanged=374,archive_file_count=len(files),
        scope='source compatibility and archived artifact hashes only; caller must still recompute all 7984 raw parity comparisons')


def recheck_predecessor_binding(receipt):
    """Terminal rehash; a failure revokes the run's source/cost eligibility."""
    after=validate_predecessor_binding(receipt['binding_path'],receipt['binding_sha256'],
        directory=receipt['binding']['parity_directory'],execution=receipt['current_execution'])
    require(after==receipt,'terminal predecessor binding changed')
    return after
# END REVIEWED PREDECESSOR HELPERS

def append(path, value):
    with Path(path).open('a') as stream: stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def validate_plan(path, sha, states):
    require(digest(path)==sha, 'measurement plan SHA differs')
    p=read_json(path)
    require(p.get('schema')==PLAN_SCHEMA and p.get('D80_sha256')==screen.old_g1.D80_SHA
            and p.get('state_manifest_sha256')==screen.STATE_SHA, 'plan source bindings differ')
    require(p.get('warmup_per_depth')==1 and p.get('measure_repeats')==2
            and p.get('depths')==list(range(1,9)) and p.get('max_new_tokens')==512
            and p.get('scope')==CORE_SCOPE and p.get('production_feedback_included') is False,
            'explicit fixed ordinary measurement protocol differs')
    expected=[(key,i) for key,by_round in states.items() for i in by_round]
    rows=p.get('schedule',[])
    require([(r.get('request_id'),r.get('round_index')) for r in rows]==expected,
            'plan must preserve complete prespecified state membership/order')
    by_key={}; positions=Counter()
    for row in rows:
        warm=row.get('warmup_order'); orders=row.get('measure_orders')
        require(type(warm) is list and sorted(warm)==list(range(1,9))
                and all(type(x) is int for x in warm), 'warmup order must contain every D once')
        require(type(orders) is list and len(orders)==2 and all(type(x) is list for x in orders)
                and sorted(orders[0])==list(range(1,9)) and orders[1]==orders[0][::-1]
                and all(type(d) is int for order in orders for d in order), 'measure orders must be an exact reverse-balanced pair')
        for order in orders:
            for slot,d in enumerate(order): positions[(d,slot)]+=1
        by_key[(row['request_id'],row['round_index'])]=deepcopy(row)
    return p,by_key,dict(path=str(Path(path).resolve()),sha256=sha,states=len(rows),
        within_state_position_pair_balanced=True,position_counts=[dict(D=d,slot=s,count=n) for (d,s),n in sorted(positions.items())])


def interval_schedule(plan_row):
    rows=[dict(phase='warmup',repeat=0,D=d,order_index=i) for i,d in enumerate(plan_row['warmup_order'])]
    rows += [dict(phase='measure',repeat=r,D=d,order_index=i)
             for r,order in enumerate(plan_row['measure_orders']) for i,d in enumerate(order)]
    return rows


def fingerprint_snapshot(snapshot):
    """Paid diagnostic D2H of real snapshot tensors, not IDs-only validation.

    Compares logical bytes, dtype/layout, storage aliasing and relevant control
    state. Padding outside tensor views is excluded; no pointer equality across
    independent cache owners is required. Never called by cost hot intervals.
    """
    import torch
    graph=snapshot._graph  # Explicit frozen RealSnapshot diagnostic interface.
    require(len(graph['layers'])==28 and snapshot._draft_meta==dict(total_tokens=29,depth=3,top_k=8),
            'actual snapshot cache/canonical contract differs')
    keys=('generated','root','stop_reason','first_tree','ids','delta','visual_mask',
          'selected_hidden_waiting','first_features','pending_tokens')
    selected=dict(layers=graph['layers'],draft=graph['draft'],rope=graph['rope'],
                  controls={k:graph['controls'][k] for k in keys})
    aliases={}; tensors=[]; transferred=0
    def visit(x,path):
        nonlocal transferred
        if torch.is_tensor(x):
            storage=x.untyped_storage(); key=(str(x.device),storage.data_ptr())
            alias=aliases.setdefault(key,len(aliases))
            host=x.detach().contiguous().reshape(-1).view(torch.uint8).to('cpu',non_blocking=False)
            payload=host.numpy().tobytes()
            transferred += len(payload) if x.device.type=='cuda' else 0
            row=dict(path=path,device_type=x.device.type,shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),
                storage_offset=x.storage_offset(),storage_bytes=storage.nbytes(),alias=alias,
                logical_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            tensors.append(row);return row
        if isinstance(x,dict):return {str(k):visit(v,path+'/'+str(k)) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return dict(container=type(x).__name__,values=[visit(v,path+'/'+str(i)) for i,v in enumerate(x)])
        require(x is None or type(x) in (str,int,float,bool),'unknown snapshot metadata')
        return x
    plain=visit(selected,'state')
    return dict(schema='fad_actual_snapshot_byte_fingerprint_v1',sha256=screen.json_sha(plain),
        cache_length=snapshot.cache_length,canonical_constants=dict(snapshot._draft_meta),
        byte_structure=plain,tensors=tensors,device_to_host_logical_bytes=transferred,
        scope='actual named logical tensor bytes + storage/layout alias and semantic controls; excludes diagnostic top2/FAD IDs/counts and unused backing padding')


def byte_equal(a,b):
    return (a['schema']==b['schema']=='fad_actual_snapshot_byte_fingerprint_v1'
        and a['sha256']==b['sha256'] and a['byte_structure']==b['byte_structure'] and a['tensors']==b['tensors']
        and a['cache_length']==b['cache_length'] and a['canonical_constants']==b['canonical_constants'])


def trial_view(trial):
    fields=('tree_submitted_including_root','all_new_output_tokens','terminal_bonus_tokens',
            'accepted_draft','committed_draft','submitted','selected_tree_indices','stop_reason',
            'step_rounds','execution_counts_delta')
    return {k:deepcopy(trial[k]) for k in fields}


def compare_parity(reference,actual):
    checks=dict(parent_bytes=byte_equal(reference['parent'],actual['parent']),
        child_bytes=byte_equal(reference['child'],actual['child']),
        trial=reference['trial']==actual['trial'],binding=reference['binding']==actual['binding'],
        actual_depth=reference['D']==actual['D'],output_ids=reference['output_ids']==actual['output_ids'])
    return dict(passed=all(checks.values()),checks=checks)


def time_round(executor, working, D, *, torch, clock=time.perf_counter, unix=time.time):
    """Called only AFTER restore+sync. Event/diagnostic setup lies outside wall."""
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    started=unix();t0=clock()
    try:
        start.record()
        receipt=executor.run_round_inplace(working,actual_depth=D)
        end.record();torch.cuda.synchronize()
    except BaseException as original:
        # A device failure at synchronization must not erase the first error.
        failure=dict(completed=False, cost_table_eligible=False, error=repr(original),
            started_unix_s=started, event_s=None)
        try:torch.cuda.synchronize()
        except BaseException as sync_error:failure['synchronize_error']=repr(sync_error)
        failure.update(ended_unix_s=unix(),wall_s=clock()-t0)
        executor._ordinary_failed_timing=failure
        raise
    wall=clock()-t0;ended=unix()
    event_s=start.elapsed_time(end)/1000
    require(wall>0 and event_s>=0 and math.isfinite(wall) and math.isfinite(event_s),'invalid timer values')
    return receipt,dict(started_unix_s=started,ended_unix_s=ended,wall_s=wall,event_s=event_s,
        timer='synchronized ordinary round including host work; CUDA event is timeline envelope',
        snapshot_restore_included=False,production_feedback_included=False)


def observe_processes(backend):
    row=dict(time_unix=time.time(),uuid=None)
    for name,fn in (('uuid',backend.uuid),('compute',lambda:backend.processes('compute')),
                    ('graphics',lambda:backend.processes('graphics'))):
        try:row[name]=fn()
        except BaseException as exc:row[name+'_error']=repr(exc)
    try:
        memory=backend.memory();row['memory']=memory
        row['free_bytes']=memory['device_memory_total_bytes']-memory['device_memory_used_bytes']
    except BaseException as exc:row['memory_error']=repr(exc)
    row['graphics_names']={}
    for process in row.get('graphics',[]):
        try:row['graphics_names'][str(process['pid'])]=Path(f"/proc/{process['pid']}/comm").read_text().strip()
        except (OSError,KeyError) as exc:row['graphics_names'][str(process.get('pid'))]=None
    return row


def verify_idle(row, expected_uuid, allowed_graphics, own_pid=None, minimum_free_bytes=None):
    require(not any(k.endswith('_error') for k in row) and row.get('uuid')==expected_uuid,'GPU visibility/UUID unavailable')
    require(type(row.get('compute')) is list and type(row.get('graphics')) is list,'missing process visibility')
    require(all(type(p) is dict and type(p.get('pid')) is int and p['pid']>0
                for p in row['compute']+row['graphics']), 'invalid process identity')
    if minimum_free_bytes is not None:
        require(type(row.get('free_bytes')) is int and row['free_bytes']>=minimum_free_bytes,
                'free GPU memory unavailable or below fixed 32 GiB startup minimum')
    require(set(allowed_graphics).issubset({6360}), 'only explicitly verified Xorg PID 6360 may be allowlisted')
    require(all(p['pid']==own_pid or (p['pid'] in allowed_graphics and
                row.get('graphics_names',{}).get(str(p['pid']))=='Xorg') for p in row['graphics']),
                'graphics PID identity differs from Xorg')
    require(not [p for p in row['compute'] if p['pid']!=own_pid], 'foreign GPU compute observed')
    require(not [p for p in row['graphics'] if p['pid'] not in allowed_graphics and p['pid']!=own_pid],
            'unknown graphics process observed')


def prelaunch_checks(backend, args, observations, *, persist, sleep=time.sleep):
    """Caller already holds flock; both polls finish before importing Torch."""
    for i in range(2):
        row=observe_processes(backend);observations.append(row);persist()
        verify_idle(row,args.expected_gpu_uuid,args.allow_graphics_pid,
                    minimum_free_bytes=MINIMUM_FREE_BYTES)
        if i==0:sleep(2)


def bind_cuda_device(torch, args):
    require(not torch.cuda.is_initialized(),'CUDA was initialized before guarded device selection')
    props=torch.cuda.get_device_properties(0)
    actual_uuid=str(props.uuid)
    actual_uuid=actual_uuid if actual_uuid.startswith('GPU-') else 'GPU-'+actual_uuid
    require(actual_uuid==args.expected_gpu_uuid,'CUDA logical device UUID differs from physical NVML device')
    return dict(logical=0,physical=args.physical_gpu,uuid=actual_uuid,name=props.name)


class ProcessMonitor:
    """Kind-preserving background records; no CUDA calls or profiler."""
    def __init__(self,backend):
        self.backend=backend;self.samples=[];self.event=threading.Event();self.thread=None
    def sample(self):self.samples.append(observe_processes(self.backend))
    def start(self):
        self.sample()
        def loop():
            while not self.event.wait(PROCESS_S):self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
    def stop(self):
        self.event.set()
        if self.thread:self.thread.join()
        self.sample()
        return dict(schema='fad_kind_preserving_background_v1',sample_interval_s=PROCESS_S,
                    observation_count=len(self.samples),samples=self.samples)


def background_review(log, start, end, uuid_, allowed_graphics, own_pid):
    """Coarse bracket coverage for sub-0.5s rounds; thresholds unchanged 1.5/2s."""
    result=dict(eligible=False,endpoint_limit_s=ENDPOINT_S,max_gap_limit_s=MAX_GAP_S,
        bracketed_short_intervals=True,strict_exclusivity_proven=False)
    try:
        samples=log['samples'];times=[p['time_unix'] for p in samples]
        require(log['sample_interval_s']==PROCESS_S and log['observation_count']==len(samples), 'raw observation count/cadence differs')
        require(len(samples)>=2 and all(type(x) in (float,int) and math.isfinite(x) for x in times)
                and all(a<b for a,b in zip(times,times[1:])),'missing/nonmonotonic process samples')
        require(type(start) in (float,int) and type(end) in (float,int)
                and math.isfinite(start) and math.isfinite(end) and start<end,'invalid interval')
        before=[i for i,t in enumerate(times) if t<=start];after=[i for i,t in enumerate(times) if t>=end]
        require(before and after,'interval lacks bracketing process observations')
        left,right=before[-1],after[0];require(right>left,'interval needs two distinct observations')
        block=samples[left:right+1]; gaps=[b['time_unix']-a['time_unix'] for a,b in zip(block,block[1:])]
        result.update(start_gap_s=start-times[left],end_gap_s=times[right]-end,max_gap_s=max(gaps),sample_count=len(block))
        require(result['start_gap_s']<=ENDPOINT_S and result['end_gap_s']<=ENDPOINT_S
                and result['max_gap_s']<=MAX_GAP_S,'process coverage gap')
        # Any malformed observation anywhere invalidates the raw log, while
        # foreign processes affect only the bracketed interval.
        require(all(not any(k.endswith('_error') for k in row) and row.get('uuid')==uuid_
                    and type(row.get('compute')) is list and type(row.get('graphics')) is list for row in samples),
                    'incomplete visibility/UUID in process log')
        for row in block:verify_idle(row,uuid_,allowed_graphics,own_pid)
        result['eligible']=True
    except (ValueError,KeyError,TypeError,IndexError) as exc:result['reason']=str(exc)
    return result


class CostOps(screen.DepthOps):
    """Actual frozen loader/math with separate diagnostic/ordinary contexts."""
    execution_kind='actual_fad_ordinary_core'
    def __init__(self,args,reference,requests):
        super().__init__(args,reference,requests)
        from combined_visual_experiment.fad_vispec.ordinary_depth_round import OrdinaryDepthExecutor
        self.ordinary_type=OrdinaryDepthExecutor
        self.metadata.update(execution_kind=self.execution_kind,ordinary_core_scope=CORE_SCOPE,
                             production_feedback_included=False)
    def executor(self,packed,request,diagnostic=False):
        if diagnostic:return super().executor(packed,request)
        return self.ordinary_type(self.spec,packed,request_key=request['request_id'],
            max_new_tokens=512,tree_budget=self.budget,eos_token_ids=[151645,151643])
    def public_result(self,joint):
        # Coverage of all actual trials is validated separately. These public
        # two counters describe only the canonical D4 continuation, as screen.
        joint.precision=dict(joint.precision,target_sdpa='math_prefill+grouped_fp32_math_tree',
            target_prefill_attention_backend='math',target_verify_attention_backend='grouped_two_bmm_fp32_math',
            target_tree_attention_mode='grouped',target_tree_math_precision=self.flags(),
            target_cache_policy='native_full_dynamic_cache',draft_attention_backend='unchanged',
            tree_attention_bitwise_equivalence_claim=False)
        calls=28*joint.forward_counts['target_verify_calls']
        joint.forward_counts.update(target_tree_attention_policy_calls=calls,target_grouped_attention_calls=calls)
        return self.result_row(joint)


def trajectory(ops,packed,request,states,plans,out,*,diagnostic=False,reference=None,timed=False):
    """Sequential context; cannot share model hooks/cache owners with another pass."""
    ops.reset();flags=ops.flags();records=[];intervals=[];parities=[]
    path=Path(out);path.mkdir(parents=True,exist_ok=False)
    with ops.grouped() as policy:
        with ops.executor(packed,request,diagnostic) as executor,screen.preserve_failure_receipt(path,executor,policy):
            parent=executor.begin();round_index=0
            while parent.controls['stop_reason'] is None:
                if round_index not in states:
                    if diagnostic:parent,_=executor.step(parent,actual_depth=4)
                    else:
                        working=executor.restore(parent);executor.run_round_inplace(working,4)
                        parent=executor.capture_working(working)
                else:
                    binding=screen.validate_live_locator(parent,states[round_index]);keep=None
                    plan=plans[(request['request_id'],round_index)]
                    schedule=interval_schedule(plan) if timed else [dict(phase='parity',repeat=0,D=d,order_index=i)
                        for i,d in enumerate(plan['warmup_order'])]
                    for item in schedule:
                        row=dict(item,request_id=request['request_id'],round_index=round_index,binding=binding,
                                 completed=False,cost_table_eligible=False)
                        try:
                            # No child can be installed into the other context.
                            # Fingerprints bind each independently restored parent.
                            if diagnostic:
                                parent_bytes=fingerprint_snapshot(parent)
                                child,trial=executor.step(parent,actual_depth=item['D'])
                            else:
                                working=executor.restore(parent)
                                if timed:
                                    trial,timing=time_round(executor,working,item['D'],torch=ops.torch)
                                    row.update(timing)
                                else:
                                    parent_bytes=fingerprint_snapshot(parent)
                                    trial=executor.run_round_inplace(working,item['D'])
                                child=executor.capture_working(working) # after timer
                            trial=dict(trial,output_ids=list(child.controls['generated']))
                            row.update(completed=True,trial=trial_view(trial),output_ids=trial['output_ids'])
                            row['reference_comparison']=screen.compare_branch_output(trial,len(parent.controls['generated']),ops.reference_ids)
                            event=trial['step_rounds'][0]
                            row['censored']=bool(event['budget_or_eos_clipped'] or trial['stop_reason'] is not None)
                            row['shape']=dict(cache_before=parent.cache_length,first_tree=parent.controls['first_tree'],
                                selected_hidden_shape=list(parent.controls['selected_hidden_waiting'].shape),
                                actual_verify_nodes=event['tree_nodes'])
                            if not timed:
                                row.update(parent=parent_bytes,child=fingerprint_snapshot(child))
                                if reference is not None:
                                    expected=reference[(round_index,item['D'])]
                                    row['parity']=compare_parity(expected,row);parities.append(row['parity'])
                            if item['D']==4:keep=child
                            del child
                            append(path/'records.jsonl',row);records.append(row)
                            if timed:intervals.append(row)
                            if reference is not None:require(row['parity']['passed'],'actual parent/child bytes or trial parity failed')
                        except BaseException as exc:
                            if timed and hasattr(executor,'_ordinary_failed_timing'):
                                row['failed_timing']=deepcopy(executor._ordinary_failed_timing)
                            row.update(completed=False,error=repr(exc),failed_unix_s=time.time())
                            append(path/'failed_intervals.jsonl',row)
                            raise
                    require(keep is not None,'D4 continuation missing');parent=keep
                round_index+=1
            require({r['round_index'] for r in records}==set(states),'not all frozen states reached')
            result=executor.finish(parent);executor.validate_coverage(policy)
            public=ops.public_result(result);execution=executor.execution_receipt();del parent
    require(ops.flags()==flags,'math flags changed in trajectory')
    save(path/'output.json',public);save(path/'execution.json',execution)
    save(path/'precision.json',dict(before=flags,after=ops.flags(),policy=policy.summary()))
    return dict(output=public,records=records,intervals=intervals,parities=parities,execution=execution)


CORE_COUNTERS=('target_prefill_calls','target_prefill_query_tokens','target_verify_calls',
    'target_verify_query_tokens','target_decode_calls','target_decode_query_tokens',
    'target_visual_calls','draft_tree_calls','draft_model_forward_calls',
    'draft_model_input_hidden_tokens','draft_decoder_query_tokens','target_compaction_calls',
    'target_compaction_selected_tokens','imadpt_calls','imadpt_input_tokens',
    'imadpt_overridden_calls','budget_switch_calls','merge_visual_tokens_in','merge_visual_tokens_out')
FP_KEYS=('generated','root','stop_reason','first_tree','ids','delta','visual_mask',
    'selected_hidden_waiting','first_features','pending_tokens')
DTYPE_BYTES={'torch.float16':2,'torch.bfloat16':2,'torch.float32':4,'torch.float64':8,
    'torch.int64':8,'torch.int32':4,'torch.int16':2,'torch.int8':1,'torch.uint8':1,'torch.bool':1}


def integer(x,minimum=0):return type(x) is int and x>=minimum


def hex_sha(x):
    return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)


def sequence(x,kind=None):
    require(type(x) is dict and set(x)=={'container','values'} and x['container'] in ('list','tuple')
            and type(x['values']) is list and (kind is None or x['container']==kind),'missing structural sequence')
    return x['values']


def validate_fingerprint(record,*,cuda_required=True):
    """Validate the saved full structure, not two matching opaque hashes.

    Tensor byte hashes attest the producer's actual D2H reads; this consumer
    does not reconstruct the original tensor bytes from hashes.
    """
    require(record['schema']=='fad_actual_snapshot_byte_fingerprint_v1'
            and integer(record['cache_length'],1)
            and record['canonical_constants']==dict(total_tokens=29,depth=3,top_k=8),'fingerprint header differs')
    structure=record['byte_structure']
    require(type(structure) is dict and set(structure)=={'layers','draft','rope','controls'},'missing snapshot groups')
    flat=[];aliases={}
    def visit(x,path):
        if type(x) is dict and 'path' in x:
            require(set(x)=={'path','device_type','shape','dtype','stride','storage_offset','storage_bytes','alias','logical_bytes','sha256'},'tensor descriptor fields differ')
            require(x['path']==path and x['device_type'] in ('cpu','cuda') and x['dtype'] in DTYPE_BYTES
                    and hex_sha(x['sha256']),'invalid tensor identity/digest')
            shape,stride=x['shape'],x['stride'];width=DTYPE_BYTES[x['dtype']]
            require(type(shape) is list and type(stride) is list and len(shape)==len(stride)
                    and all(integer(n) for n in shape+stride)
                    and integer(x['storage_offset']) and integer(x['storage_bytes']) and integer(x['alias'])
                    and x['storage_bytes']%width==0 and x['logical_bytes']==math.prod(shape)*width,'invalid tensor byte/layout metadata')
            end=x['storage_offset']+(sum((n-1)*s for n,s in zip(shape,stride))+1 if math.prod(shape)>0 else 0)
            require(end*width<=x['storage_bytes'],'tensor view exceeds backing')
            group=(x['device_type'],x['storage_bytes'])
            if x['alias'] not in aliases:
                require(x['alias']==len(aliases),'alias numbering is not canonical');aliases[x['alias']]=group
            require(aliases[x['alias']]==group,'inconsistent shared storage metadata')
            flat.append(x);return
        if type(x) is dict:
            if 'container' in x:
                for i,item in enumerate(sequence(x)):visit(item,path+'/'+str(i))
            else:
                for key,value in x.items():visit(value,path+'/'+key)
        else:require(x is None or type(x) in (int,float,str,bool),'unsupported structure metadata')
    visit(structure,'state')
    require(flat==record['tensors'] and screen.json_sha(structure)==record['sha256'],'fingerprint structure/hash/list differ')
    require(record['device_to_host_logical_bytes']==sum(t['logical_bytes'] for t in flat if t['device_type']=='cuda'), 'fingerprint transfer bytes differ')
    tensors={r['path']:r for r in flat};require(len(tensors)==len(flat),'duplicate tensor path')
    def tensor(x,path,*,dtype=None,shape=None,device=None):
        require(type(x) is dict and tensors.get(path)==x,'required tensor missing: '+path)
        if dtype:require(x['dtype']==dtype,'tensor dtype differs: '+path)
        if shape is not None:require(x['shape']==shape,'tensor shape differs: '+path)
        if cuda_required and device=='cuda':require(x['device_type']=='cuda','actual CUDA tensor missing: '+path)
        return x
    layers=sequence(structure['layers'],'list');require(len(layers)==28,'all 28 layers required')
    for i,layer in enumerate(layers):
        require(set(layer)=={'keys','values','is_initialized'} and layer['is_initialized'] is True,'incomplete cache layer')
        for key in ('keys','values'):
            tensor(layer[key],f'state/layers/{i}/{key}',dtype='torch.float16',shape=[1,4,record['cache_length'],128],device='cuda')
    draft=structure['draft'];c=structure['controls']
    require(type(draft) is dict and set(draft)=={'stable_kv','last_img_hidden','tree_mask'}
            and type(c) is dict and set(c)==set(FP_KEYS),'Draft/controls fields missing')
    generated=sequence(c['generated'],'list');pending=sequence(c['pending_tokens'],'list')
    require(all(integer(t) for t in generated+pending) and len(pending)<=1
            and integer(c['root']) and type(c['delta']) is int and type(c['first_tree']) is bool
            and c['stop_reason'] in (None,'eos','limit'),'invalid semantic controls')
    tensor(c['ids'],'state/controls/ids',dtype='torch.int64',shape=[1,record['cache_length']],device='cuda')
    require(type(structure['rope']) is dict and set(structure['rope'])=={'rope_deltas'},'rope state missing')
    # Native prefill supplies explicit positions, so HF may leave this cache
    # attribute None. The separate integer controls.delta drives tree mRoPE.
    # Preserve None in the fingerprint instead of inventing a tensor value.
    if structure['rope']['rope_deltas'] is not None:
        tensor(structure['rope']['rope_deltas'],'state/rope/rope_deltas',dtype='torch.int64',device='cuda')
    if c['visual_mask'] is not None:tensor(c['visual_mask'],'state/controls/visual_mask',dtype='torch.bool',device='cuda')
    if c['stop_reason'] is None:
        t=tensor(c['selected_hidden_waiting'],'state/controls/selected_hidden_waiting',device='cuda')
        require(len(t['shape'])==3 and t['shape'][0]==1 and t['shape'][1]>0,'invalid selected hidden')
        require(not pending,'nonterminal pending token')
    else:require(c['selected_hidden_waiting'] is None,'terminal hidden must be released')
    if c['first_tree']:
        require(draft['stable_kv'] is None and draft['last_img_hidden'] is None
                and not generated,'first-tree Draft state differs')
        # Native prior request may leave this exact mask; author reset() clears it
        # before the first Draft forward. Preserve its bytes/alias for parity.
        if draft['tree_mask'] is not None:
            tensor(draft['tree_mask'],'state/draft/tree_mask',dtype='torch.float32',shape=[1,1,8,24],device='cuda')
        features=c['first_features'];require(type(features) is dict and {'hidden','inputs_embeds'}<=set(features),'first features missing')
        for key in ('hidden','inputs_embeds'):tensor(features[key],'state/controls/first_features/'+key,device='cuda')
    else:
        require(c['first_features'] is None,'stale first-tree features')
        kv=sequence(draft['stable_kv']);require(kv,'stable KV missing')
        for i,layer in enumerate(kv):
            pair=sequence(layer);require(len(pair)>=2,'Draft stable KV pair missing')
            for j in (0,1):tensor(pair[j],f'state/draft/stable_kv/{i}/{j}',device='cuda')
        tensor(draft['last_img_hidden'],'state/draft/last_img_hidden',device='cuda')
        if draft['tree_mask'] is not None:tensor(draft['tree_mask'],'state/draft/tree_mask',device='cuda')
    return c,tensors


def validate_trial_record(row,locator,*,counter_keys=CORE_COUNTERS,cuda_required=True,maximum_tokens=512):
    require(row['completed'] is True and row['phase']=='parity' and row['repeat']==0
            and integer(row['order_index']) and type(row['D']) is int and 1<=row['D']<=8,'invalid parity trial header')
    require(type(row['binding']) is dict and all(row['binding'].get(k) is True for k in
        ('prefix_exact','cache_boundary_exact','root_exact','semantic_position_exact')),'live binding incomplete')
    parent,pt=validate_fingerprint(row['parent'],cuda_required=cuda_required)
    child,ct=validate_fingerprint(row['child'],cuda_required=cuda_required)
    before=sequence(parent['generated']);after=sequence(child['generated']);base=row['parent']['cache_length']
    require(parent['stop_reason'] is None and len(before)==locator['output_prefix_tokens']
        and screen.token_prefix_sha(before)==locator['output_prefix_sha256']
        and parent['root']==locator['expected_root_token'] and base==locator['target_cache_before']
        and base+parent['delta']==locator['semantic_rope_base'],'snapshot parent differs from frozen locator')
    t=row['trial'];require(set(t)==set(trial_view(t)),'trial fields differ')
    rounds=t['step_rounds'];require(type(rounds) is list and len(rounds) in (1,2),'missing full step rounds')
    event=rounds[0];n=9 if row['D']==1 else 30
    event_keys={'kind','tree_nodes','budget_action','budget','accepted_draft','committed_draft','submitted',
        'target_cache_before','target_cache_after','discarded_tree_nodes','budget_or_eos_clipped',
        'pending_tokens','selected_tree_indices','semantic_rope_base','selectedH_waiting_sync','visual_m'}
    require(set(event)==event_keys and event['kind']=='tree' and event['tree_nodes']==n
        and event['budget']==dict(B=n,d=row['D']-1,K=8,total_tokens=n-1)
        and event['budget_action']==('new_B30_d3_K8' if row['D']==4 else 'fad_D'+str(row['D'])),'actual budget/event differs')
    submitted=event['submitted'];selected=event['selected_tree_indices'];k=len(submitted)
    require(type(submitted) is list and 1<=k<=n and all(integer(x) for x in submitted)
        and submitted[0]==parent['root'] and type(selected) is list and len(selected)==k
        and all(integer(x) and x<n for x in selected) and len(set(selected))==k and selected[0]==0,'invalid selected root/path')
    require(integer(event['accepted_draft']) and k-1<=event['accepted_draft']<=row['D']
        and event['committed_draft']==k-1 and event['target_cache_before']==base
        and event['target_cache_after']==base+k and event['discarded_tree_nodes']==n-k
        and type(event['budget_or_eos_clipped']) is bool
        and event['budget_or_eos_clipped']==(k!=event['accepted_draft']+1)
        and event['semantic_rope_base']==base+parent['delta'] and event['pending_tokens']==[]
        and event['selectedH_waiting_sync'] is False and event['visual_m']==1.0,'event commit/semantic ledger differs')
    bonus=[]
    if len(rounds)==2:
        r=rounds[1]
        require(set(r)=={'kind','submitted','target_cache_after','pending_tokens','selectedH_waiting_sync','known_root'}
            and r['kind']=='terminal_target_token' and len(r['submitted'])==1
            and r['submitted']==r['pending_tokens']==[r['known_root']] and integer(r['known_root'])
            and r['target_cache_after']==base+k and r['selectedH_waiting_sync'] is True,'terminal bonus differs')
        bonus=r['submitted']
    require(after==before+submitted+bonus==row['output_ids'] and len(after)<=maximum_tokens
        and sequence(child['pending_tokens'])==bonus and row['child']['cache_length']==base+k
        and child['delta']==parent['delta'] and child['first_tree'] is False,'child state/output/cache differs')
    require(t['submitted']==submitted and t['selected_tree_indices']==selected
        and t['tree_submitted_including_root']==k and t['all_new_output_tokens']==k+len(bonus)
        and t['terminal_bonus_tokens']==len(bonus) and t['accepted_draft']==event['accepted_draft']
        and t['committed_draft']==k-1 and t['stop_reason']==child['stop_reason'],'trial G/A/bonus/stop differs')
    if child['stop_reason']=='limit':require(len(after)==maximum_tokens,'limit before token budget')
    if cuda_required and child['stop_reason']=='eos':require(after[-1] in (151645,151643),'EOS token differs')
    if bonus:require(child['stop_reason'] is not None and child['root']==bonus[0],'bonus root differs')
    if child['stop_reason'] is None:
        require(ct['state/controls/selected_hidden_waiting']['shape'][1]==k,'next hidden length differs')
    mask=row['child']['byte_structure']['draft']['tree_mask']
    require((mask is None) if row['D']==1 else (mask is not None),'Draft expansion mask missing/differs')
    delta=t['execution_counts_delta']
    require(set(delta)==set(counter_keys) and all(integer(v) for v in delta.values()),'missing/invalid per-round counters')
    expected=dict(target_prefill_calls=0,target_prefill_query_tokens=0,target_visual_calls=0,
        target_decode_calls=0,target_decode_query_tokens=0,target_verify_calls=1,target_verify_query_tokens=n,
        draft_tree_calls=1,draft_model_forward_calls=row['D'],target_compaction_calls=1,
        target_compaction_selected_tokens=k,budget_switch_calls=1,imadpt_overridden_calls=0,
        merge_visual_tokens_in=0,merge_visual_tokens_out=0,
        draft_model_input_hidden_tokens=pt['state/controls/selected_hidden_waiting']['shape'][1]+8*(row['D']-1))
    require(all(delta[k]==v for k,v in expected.items()) and delta['draft_decoder_query_tokens']>=row['D'], 'actual D/counts differ')
    if not parent['first_tree']:require(delta['imadpt_calls']==delta['imadpt_input_tokens']==0,'later-tree adaptor count differs')
    require(type(row['censored']) is bool and row['censored']==bool(event['budget_or_eos_clipped'] or child['stop_reason'] is not None),'censor receipt differs')
    require(row['shape']==dict(cache_before=base,first_tree=parent['first_tree'],
        selected_hidden_shape=pt['state/controls/selected_hidden_waiting']['shape'],actual_verify_nodes=n),'shape receipt differs')
    return delta


def validate_execution_ledger(records,execution,public,precision,request_id):
    """Rebuild all branches from retained D4 public path plus every sampled D."""
    result=public['result'];counts=result['forward_counts']
    common={k:v for k,v in counts.items() if k not in ('target_grouped_attention_calls','target_tree_attention_policy_calls')}
    require(set(common)==set(CORE_COUNTERS) and all(integer(x) for x in common.values()),'public core counters differ')
    allcounts=dict(common);by_round={};reconstructed=[];cursor=0;round_index=0
    for row in records:by_round.setdefault(row['round_index'],[]).append(row)
    rounds=result['rounds']
    while cursor<len(rounds):
        event=rounds[cursor];require(event['kind']=='tree','unexpected standalone terminal round')
        length=1+(cursor+1<len(rounds) and rounds[cursor+1]['kind']=='terminal_target_token')
        current=rounds[cursor:cursor+length]
        if round_index in by_round:
            block=by_round[round_index];require(sorted(r['D'] for r in block)==list(range(1,9)),'missing full D block')
            require(all(byte_equal(block[0]['parent'],r['parent']) for r in block), 'depth branches do not share the same parent bytes')
            baseline=next(r for r in block if r['D']==4)
            require(baseline['trial']['step_rounds']==current,'selected D4 differs from public path')
            first=block[0]['trial']['execution_counts_delta']
            for r in block:
                d=r['trial']['execution_counts_delta'];reconstructed.extend(r['trial']['step_rounds'])
                for key in common:allcounts[key]+=d[key]
                require(d['draft_decoder_query_tokens']-8*(r['D']-1)==first['draft_decoder_query_tokens']-8*(block[0]['D']-1)
                    and d['imadpt_calls']==first['imadpt_calls'] and d['imadpt_input_tokens']==first['imadpt_input_tokens'],
                    'same-parent Draft/adaptor counts differ across D')
            for key in common:allcounts[key]-=baseline['trial']['execution_counts_delta'][key]
        else:reconstructed.extend(current)
        cursor+=length;round_index+=1
    require(execution['request_key']==request_id and execution['all_trial_coverage_validated'] is True
        and execution['poisoned'] is False and execution['failures']==[], 'execution health/request differs')
    require(execution['all_trial_rounds']==reconstructed and execution['all_trial_forward_counts']==allcounts,
        'all-trial rounds/counts fail reconstruction')
    trees=[r for r in reconstructed if r['kind']=='tree'];nodes=sum(r['tree_nodes'] for r in trees)
    require(execution['actual_operation_attempts']==dict(prefill=1,draft_tree=len(trees),target_verify=len(trees),compaction=len(trees))
        and execution['actual_verify_nodes']==nodes and allcounts['target_verify_query_tokens']==nodes,
        'all-trial attempts/actual nodes differ')
    depths=[r['budget']['d']+1 for r in trees]
    if 'ordinary_round_history' in execution:
        require(execution['ordinary_round_history']==[dict(actual_depth=d,tree_nodes=t['tree_nodes']) for d,t in zip(depths,trees)],'ordinary depth history differs')
    else:
        require(execution['actual_depths']==depths and len(execution['depth_trial_history'])==len(trees)
            and [(r['actual_depth'],r['tree_nodes']) for r in execution['depth_trial_history']]==[(d,t['tree_nodes']) for d,t in zip(depths,trees)],'diagnostic depth history differs')
    p=precision['policy']
    require(p['mode']=='grouped' and p['prefill_calls']==p['prefill_forward_calls']==1
        and p['prefill_layer_calls']==[1]*28 and p['verify_forward_calls']==len(trees)
        and p['failures']==[] and p['fallback_calls']==p['native_tree_calls']==0
        and p['original_mask_reused'] is True and p['position_ids_passed_unchanged'] is True
        and p['rotary_reapplied'] is False, 'policy scope/prefill/precision route differs')
    expected_policy_rounds=[dict(cache_before=t['target_cache_before'],tree_nodes=t['tree_nodes'],
        cache_after=t['target_cache_before']+t['tree_nodes'],layer_calls=[1]*28,
        mask_shape=[1,1,t['tree_nodes'],t['target_cache_before']+t['tree_nodes']],mask_dtype='torch.float32',
        position_shape=[3,1,t['tree_nodes']],position_dtype='torch.int64') for t in trees]
    require(p['rounds']==expected_policy_rounds, 'per-round 28-layer/mask/position coverage differs')
    require(p['finalized'] is True and p['completed_calls']==p['candidate_calls']==28*len(trees)
        and p['layer_calls']==[len(trees)]*28 and p['verify_query_tokens']==nodes,'28-layer coverage raw counters differ')
    return dict(trees=len(trees),verify_nodes=nodes,counters=allcounts)


def validate_parity_gate(directory, *, execution, plan_sha, reference, request_ids, states, expected_uuid,
                         predecessor_binding=None, predecessor_binding_sha256=None):
    require((predecessor_binding is None)==(predecessor_binding_sha256 is None),'predecessor binding and SHA required together')
    predecessor=validate_predecessor_binding(predecessor_binding,predecessor_binding_sha256,
        directory=directory,execution=execution) if predecessor_binding is not None else None
    expected_execution=predecessor['predecessor_execution'] if predecessor else execution
    expected_entry=PREDECESSOR_ENTRY_SHA if predecessor else digest(__file__)
    base=Path(directory);report=read_json(base/'DEPTH_COST_REPORT.json')
    require(report==read_json(base/'status.json') and report['schema']==SCHEMA
            and report['mode']=='parity_gate' and report['status']=='completed'
            and report['entry_sha256']==expected_entry,'ordinary parity run is incomplete/different')
    require(report['cuda_device']['uuid']==expected_uuid and report['setup']['execution_kind']=='actual_fad_ordinary_core',
            'parity actual device/loader identity differs')
    require(report['execution_source_freeze']==report['execution_source_freeze_after']==expected_execution
            and report['source_recheck']['passed'] is True and report['measurement_plan']['sha256']==plan_sha
            and report['c0_artifacts_sha256']==reference['hashes'],'parity source/plan/C0 binding differs')
    require([c['request_id'] for c in report['cases']]==request_ids and len(request_ids)==80,
            'complete frozen D80 parity required')
    require(report['cases']==jsonl(base/'cases.jsonl'),'parity case list mismatch')
    _,plan_rows,_=validate_plan(base/'measurement_plan.json',plan_sha,states)
    total=0
    for i,case in enumerate(report['cases']):
        folder=base/'cases'/f'{i:02d}'
        require(case==read_json(folder/'case.json') and case['completed'] and case['input_exact']
                and case['flags_exact'] and case['c0_comparison']['exact'] and case['continuation_comparison']['exact'],
                'parity raw case invalid')
        required={'input.json','c0_rerun.json','c0_observation.json'} | {role+'/'+name for role in ('diagnostic','ordinary')
            for name in ('records.jsonl','output.json','execution.json','precision.json')}
        require(required<=set(case['artifact_sha256']) and all(hex_sha(h) for h in case['artifact_sha256'].values()),'required parity artifact hashes missing')
        for name,sha in case['artifact_sha256'].items():
            artifact=(folder/name).resolve()
            require(artifact.is_relative_to(folder.resolve()) and not Path(name).is_absolute(), 'parity artifact path escape')
            require(digest(artifact)==sha,'parity artifact drift: '+name)
        a=jsonl(folder/'diagnostic/records.jsonl');b=jsonl(folder/'ordinary/records.jsonl')
        expected={(round_index,D) for round_index in states[case['request_id']] for D in range(1,9)}
        require(len(a)==len(b)==len(expected) and {(r['round_index'],r['D']) for r in a}==expected
                and {(r['round_index'],r['D']) for r in b}==expected,'parity state/depth coverage differs')
        saved=reference['selected'][case['request_id']]
        require(all(read_json(folder/'input.json')[k]==saved['input'][k] for k in INPUT_KEYS),'parity raw input differs')
        require(g0.compare_c0(read_json(folder/'c0_rerun.json'),saved['row'])['exact'],'parity raw C0 differs')
        for role in ('diagnostic','ordinary'):
            ex=read_json(folder/role/'execution.json');precision=read_json(folder/role/'precision.json')
            require(ex['all_trial_coverage_validated'] and not ex['poisoned'] and not ex['failures'], 'parity all-trial coverage/health failed')
            require(precision['before']==precision['after']==saved['precision']['before']==saved['precision']['after'], 'parity math flags differ')
            public=read_json(folder/role/'output.json')
            require(g0.compare_c0(public,saved['row'])['exact'],'parity canonical full trajectory differs')
            raw=a if role=='diagnostic' else b
            for row in raw:
                locator=states[case['request_id']][row['round_index']]
                validate_trial_record(row,locator)
                expected_comparison=screen.compare_branch_output(dict(row['trial'],output_ids=row['output_ids']),
                    locator['output_prefix_tokens'],saved['row']['output_token_ids'])
                require(row['reference_comparison']==expected_comparison,'branch/C0 difference receipt differs')
            for ri in states[case['request_id']]:
                block=[r for r in raw if r['round_index']==ri]
                require([r['D'] for r in block]==plan_rows[(case['request_id'],ri)]['warmup_order']
                    and [r['order_index'] for r in block]==list(range(8)),'actual parity order differs from frozen plan')
            validate_execution_ledger(raw,ex,public,precision,case['request_id'])
        for x,y in zip(a,b):
            require(x['completed'] and y['completed'] and x['phase']==y['phase']=='parity'
                    and x['request_id']==y['request_id']==case['request_id'] and x['round_index']==y['round_index']
                    and compare_parity(x,y)['passed'],'actual-byte parity failed')
            for row in (x,y):
                for kind in ('parent','child'):
                    tensors={t['path']:t for t in row[kind]['tensors']}
                    require(all(tensors.get(f'state/layers/{i}/{name}',{}).get('device_type')=='cuda'
                        for i in range(28) for name in ('keys','values')), 'real CUDA cache byte receipts required')
        total+=len(a)
        g0.validate_result(read_json(folder/'ordinary/output.json'))
        require(g0.compare_c0(read_json(folder/'ordinary/output.json'),read_json(folder/'c0_rerun.json'))['exact'],
                'parity continuation raw differs from C0')
    require(total==998*8,'parity coverage must contain all 998 x8 comparisons')
    return dict(passed=True,raw_compared=total,report_sha256=digest(base/'DEPTH_COST_REPORT.json'),
        predecessor_binding=predecessor,
        scope='raw actual-byte SHA receipts checked, tensors not replayed by this preflight')


def run_case(ops,request,saved,folder,states,plans,mode,backend):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    case=dict(request_id=request['request_id'],completed=False,mode=mode,stage='prepare',
              diagnostic_only=True,throughput_eligible=False,deployment_memory_eligible=False)
    monitor=None;log=None;run=None
    try:
        ops.reset();packed,sampled=ops.prepare(request)
        inp=dict(input_signature=ops.signature(packed,list(sampled.timestamps)),timestamps=list(sampled.timestamps),
            frame_indices=list(sampled.metadata['frames_indices']),sampled_metadata=dict(sampled.metadata),
            sampled_provenance=dict(sampled.provenance))
        save(folder/'input.json',inp)
        case['input_exact']=all(inp[k]==saved['input'][k] for k in INPUT_KEYS)
        require(case['input_exact'],'prepared tensors/grid/timestamps differ from actual C0')
        before=ops.flags();require(before==saved['precision']['before']==saved['precision']['after'],'C0 precision differs')
        case['stage']='c0_rerun';c0,cobs=ops.original(packed);save(folder/'c0_rerun.json',c0);save(folder/'c0_observation.json',cobs)
        g0.validate_result(c0);case['c0_comparison']=g0.compare_c0(c0,saved['row']);require(case['c0_comparison']['exact'],'actual C0 rerun differs')
        ops.reference_ids=c0['output_token_ids']
        if mode=='parity_gate':
            case['stage']='diagnostic_parity'
            ref=trajectory(ops,packed,request,states,plans,folder/'diagnostic',diagnostic=True)
            reference={(r['round_index'],r['D']):r for r in ref['records']}
            require(g0.compare_c0(ref['output'],c0)['exact'],'diagnostic D4 continuation differs')
            case['stage']='ordinary_parity'
            run=trajectory(ops,packed,request,states,plans,folder/'ordinary',reference=reference)
            case['actual_byte_parities']=len(run['parities'])
        else:
            case['stage']='ordinary_cost';monitor=ProcessMonitor(backend).start()
            run=trajectory(ops,packed,request,states,plans,folder/'ordinary',timed=True)
        g0.validate_result(run['output']);case['continuation_comparison']=g0.compare_c0(run['output'],c0)
        require(case['continuation_comparison']['exact'],'ordinary D4 continuation differs from actual C0')
        case['flags_exact']=ops.flags()==before;require(case['flags_exact'],'final precision differs')
        case.update(completed=True,stage='completed',states=len(states))
    except BaseException as exc:case.update(error=repr(exc),traceback=g0.traceback.format_exc())
    finally:
        try:ops.synchronize()
        except BaseException as exc:case.update(completed=False,sync_error=repr(exc))
        if monitor:
            try:log=monitor.stop()
            except BaseException as exc:
                log=dict(unavailable=True,error=repr(exc));case.update(completed=False,background_error=repr(exc))
            save(folder/'background.json',log)
            # Read partial rows even when later execution fails; never erase negatives.
            records=jsonl(folder/'ordinary/records.jsonl') if (folder/'ordinary/records.jsonl').exists() else []
            reviewed=[]
            for row in records:
                if 'started_unix_s' not in row:continue
                row['background']=background_review(log,row['started_unix_s'],row['ended_unix_s'],
                    ops.args.expected_gpu_uuid,ops.args.allow_graphics_pid,os.getpid())
                row['eligibility_requires_completed_report_and_source_recheck']=True
                row['cost_table_eligible']=bool(case.get('completed') and row['completed'] and row['background']['eligible'])
                reviewed.append(row)
            (folder/'ordinary').mkdir(exist_ok=True)
            save(folder/'ordinary/cost_intervals.json',reviewed)
            case['eligible_measured_intervals']=sum(r['phase']=='measure' and r['cost_table_eligible'] for r in reviewed)
            case['raw_measured_intervals']=sum(r['phase']=='measure' for r in reviewed)
        case['artifact_sha256']={str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='case.json'}
        save(folder/'case.json',case)
    return case


def aggregate(cases,mode):
    complete=len(cases)==80 and all(c.get('completed') for c in cases) and sum(c.get('states',0) for c in cases)==998
    return dict(complete=complete,parity_passed=bool(complete and mode=='parity_gate'
        and sum(c.get('actual_byte_parities',0) for c in cases)==998*8),
        measured_intervals=sum(c.get('raw_measured_intervals',0) for c in cases),
        eligible_measured_intervals=sum(c.get('eligible_measured_intervals',0) for c in cases),
        complete_cost_table_eligible=bool(complete and mode=='cost_d80' and
            sum(c.get('eligible_measured_intervals',0) for c in cases)==998*16),
        scope=CORE_SCOPE,production_feedback_included=False,dynamic_TPS_claim=False,deployment_memory_acceptance=False)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',choices=('parity_gate','cost_d80'),required=True)
    for name in ('out','requests','requests-sha256','c0-dir','expected-gpu-uuid','execution-source-freeze',
                 'execution-source-freeze-sha256','state-manifest','state-manifest-sha256','measurement-plan','measurement-plan-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--physical-gpu',type=int,choices=range(8),required=True)
    p.add_argument('--allow-graphics-pid',type=int,action='append',default=[])
    p.add_argument('--c0-project');p.add_argument('--parity-dir')
    p.add_argument('--parity-predecessor-binding');p.add_argument('--parity-predecessor-binding-sha256')
    a=p.parse_args(argv)
    require((a.parity_predecessor_binding is None)==(a.parity_predecessor_binding_sha256 is None),'predecessor binding and SHA required together')
    if a.parity_predecessor_binding is not None:
        require(a.mode=='cost_d80' and a.parity_dir is not None,'predecessor only supports explicit cost parity')
        require(Path(a.parity_dir).is_absolute() and Path(a.parity_predecessor_binding).is_absolute()
                and hex_sha(a.parity_predecessor_binding_sha256),'absolute predecessor paths/valid SHA required')
    require(a.requests_sha256==screen.old_g1.D80_SHA and a.state_manifest_sha256==screen.STATE_SHA,'frozen D80/998 required')
    require(not Path(a.out).exists(),'fresh output required; no resume/retry')
    require(a.mode!='cost_d80' or a.parity_dir,'actual complete parity gate required')
    require(a.expected_gpu_uuid.startswith('GPU-') and str(uuid.UUID(a.expected_gpu_uuid[4:]))==a.expected_gpu_uuid[4:], 'canonical GPU UUID required')
    require(set(a.allow_graphics_pid).issubset({6360}),'only verified graphics PID 6360 may be allowlisted')
    a.allow_graphics_pid=sorted(set(a.allow_graphics_pid));return a


def main(argv=None):
    args=parse_args(argv)
    execution=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
    requests=jsonl(args.requests)
    require(digest(args.requests)==args.requests_sha256,'D80 request bytes differ')
    md=Path(args.requests).parent;manifest=read_json(md/'MANIFEST.json')
    require(digest(md/'MANIFEST.json')==g0.MANIFEST_SHA,'manifest differs')
    e24=jsonl(md/'E24.jsonl');require(digest(md/'E24.jsonl')==g0.E24_SHA,'E24 differs')
    reference=screen.old_g1.bind_d80(args.c0_dir,requests,e24)
    states=screen.bind_state_manifest(args.state_manifest,args.state_manifest_sha256,requests,reference)
    plan,plans,plan_receipt=validate_plan(args.measurement_plan,args.measurement_plan_sha256,states)
    c0_project=g0.resolve_c0_project(args.c0_dir,args.c0_project)
    sources=g0.check_sources(reference,manifest,c0_project=c0_project);sources.update(execution['source_sha256'])
    gate=validate_parity_gate(args.parity_dir,execution=execution,plan_sha=args.measurement_plan_sha256,
        reference=reference,request_ids=[r['request_id'] for r in requests],states=states,expected_uuid=args.expected_gpu_uuid,
        predecessor_binding=args.parity_predecessor_binding,
        predecessor_binding_sha256=args.parity_predecessor_binding_sha256) if args.mode=='cost_d80' else None
    lock,lock_receipt=g0.load_lock_without_torch(PROJECT/'direct_opt/run_joint.py')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,mode=args.mode,status='starting',pid=os.getpid(),run_id=uuid.uuid4().hex,
        args=vars(args),entry_sha256=digest(__file__),source_sha256=sources,
        execution_source_freeze=execution,source_recheck=dict(passed=False),
        measurement_plan=plan_receipt,c0_artifacts_sha256=reference['hashes'],lock_source=lock_receipt,
        parity_gate=gate,cases=[],prelaunch=[],scope=NOTE)
    (out/'measurement_plan.json').write_bytes(Path(args.measurement_plan).read_bytes())
    def persist():
        report['updated_at']=dt.datetime.now(dt.timezone.utc).isoformat();report['gate']=aggregate(report['cases'],args.mode)
        if report['status']!='completed' or not report['source_recheck']['passed']:
            report['gate']['parity_passed']=report['gate']['complete_cost_table_eligible']=False
        save(out/'status.json',report);save(out/'DEPTH_COST_REPORT.json',report)
    persist();backend=None
    previous={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','MKL_NUM_THREADS')}
    try:
        # Only stdlib/setup hashes precede cooperative ownership.
        for path in (PROJECT,PROJECT/'vendor',PROJECT/'direct_opt',PROJECT/'training_free'):
            sys.path.insert(0,str(path))
        from hardware_metrics import _make_backend
        with lock(args.physical_gpu,out) as metadata:
            report['lock']=metadata;backend=_make_backend(args.physical_gpu)
            prelaunch_checks(backend,args,report['prelaunch'],persist=persist)
            os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            import torch
            report['cuda_device']=bind_cuda_device(torch,args)
            from grouped_gqa_runtime import math_precision_scope
            with math_precision_scope(),torch.inference_mode():
                ops=CostOps(args,reference,requests);report.update(status='running',setup=ops.metadata);persist()
                for i,request in enumerate(requests):
                    # Block a newly observed foreign job before preparing another request.
                    row=observe_processes(backend);report.setdefault('request_guards',[]).append(row)
                    verify_idle(row,args.expected_gpu_uuid,args.allow_graphics_pid,os.getpid())
                    report['current']=dict(index=i,request_id=request['request_id']);persist()
                    case=run_case(ops,request,reference['selected'][request['request_id']],out/'cases'/f'{i:02d}',
                        states[request['request_id']],plans,args.mode,backend)
                    report['cases'].append(case);append(out/'cases.jsonl',case);persist()
                    require(case['completed'],'case failed; raw negatives retained; later requests paused')
                report.update(status='completed',current=None)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=g0.traceback.format_exc());raise
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as exc:report.update(status='failed',backend_close_error=repr(exc))
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        error=None
        try:
            after=g0.check_execution_sources(args.execution_source_freeze,args.execution_source_freeze_sha256)
            report['execution_source_freeze_after']=after;require(after==execution,'execution source drift')
            final=g0.check_sources(reference,manifest,c0_project=c0_project);final.update(after['source_sha256'])
            require(final==sources and digest(md/'MANIFEST.json')==g0.MANIFEST_SHA,'old source/manifest drift')
            require(digest(args.measurement_plan)==args.measurement_plan_sha256
                    and digest(args.state_manifest)==args.state_manifest_sha256,'measurement/state plan drift')
            if gate is not None and gate.get('predecessor_binding') is not None:
                report['parity_predecessor_recheck']=recheck_predecessor_binding(gate['predecessor_binding'])
            report['source_recheck']=dict(passed=True,execution_inventory_exact=True)
        except BaseException as exc:
            error=exc;report.update(status='failed',source_recheck=dict(passed=False,error=repr(exc)))
        report['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat();persist()
        if error:raise RuntimeError('terminal ordinary-cost source recheck failed') from error
    require(report['status']=='completed','cost run did not complete; see retained negative report')
    return report


if __name__=='__main__':main()
