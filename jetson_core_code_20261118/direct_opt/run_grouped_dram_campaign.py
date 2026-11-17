"""Three-process actual-DRAM campaign: original, layout control, grouped.

No execution on import. Original uses native/native deployment with no KV
policy; control and grouped both use shared/compiled/native DynamicCache.
Every child owns the cooperative lock and profiles one warmup plus two real
generation ranges. No automatic retry, fallback, GPU mutation or installation.
Resume only revalidates existing successful arms and starts missing arms; an
existing invalid/partial arm directory is preserved and never executed again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from run_active_dram_campaign import observe,retain_workload_pid,save,verify_guard
from run_visual_kv_dram_campaign import header_environment
import profile_implicit_kv_dram as original_profile
import profile_grouped_gqa_dram as tree_profile

HERE=Path(__file__).resolve().parent
ARMS=('original','control','grouped')
REQUEST=original_profile.REQUEST
KNOWN_LOADER_SUFFIXES=(
    '/data1/lgq2026/jev_spec/assets/build/python_headers/extracted/usr/include/python3.12',
    '/data1/lgq2026/jev_spec/assets/build/python_headers/extracted/usr/include')
ENV_KEYS=('CUDA_VISIBLE_DEVICES','PYTHONUNBUFFERED','TOKENIZERS_PARALLELISM','CPATH')
INSTRUMENTATION_COUNTS={'target_tree_attention_policy_calls','target_stride0_attention_calls','target_grouped_attention_calls'}
MODE_PRECISION_KEYS={'target_sdpa','target_prefill_attention_backend','target_verify_attention_backend',
    'target_tree_attention_mode','target_tree_math_precision','target_cache_policy',
    'draft_attention_backend','tree_attention_bitwise_equivalence_claim'}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--out',required=True)
    p.add_argument('--gpu',type=int,choices=range(8),default=7)
    p.add_argument('--requests',default=str(HERE.parents[1]/'coord/quality_requests.jsonl'))
    p.add_argument('--request-id',default=REQUEST)
    p.add_argument('--scope',choices=('generation',),default='generation')
    p.add_argument('--warmup',type=int,choices=(1,),default=1)
    p.add_argument('--repeats',type=int,choices=(2,),default=2)
    p.add_argument('--ncu',default='/usr/local/cuda/bin/ncu')
    p.add_argument('--allow-graphics-pid',type=int,action='append',default=[])
    p.add_argument('--header-environment',default=str(HERE.parents[1]/'kv_int8_primitive_r3/header_environment.json'))
    p.add_argument('--resume',action='store_true',help='revalidate/skip valid existing arms; never rerun an existing directory')
    a=p.parse_args(argv)
    if any(pid<1 for pid in a.allow_graphics_pid):p.error('positive graphics PIDs required')
    a.allow_graphics_pid=sorted(set(a.allow_graphics_pid))
    return a


def route(arm):
    if arm=='original':return original_profile,'native','profile_implicit_kv_dram.py'
    if arm in ('control','grouped'):return tree_profile,'native' if arm=='control' else 'grouped','profile_grouped_gqa_dram.py'
    raise ValueError('unsupported campaign arm')


def command(args,folder,arm):
    _,mode,script=route(arm)
    cmd=[args.ncu,'--config-file','off','--target-processes','application-only',
        '--replay-mode','app-range','--cache-control','none','--clock-control','none',
        '--pipeline-boost-state','dynamic','--disable-extra-suffixes',
        '--metrics','dram__bytes_read.sum,dram__bytes_write.sum',
        '--export',str(folder/'dram'),'--csv','--page','raw','--log-file',str(folder/'metrics.csv'),
        sys.executable,str(HERE/script)]
    if arm=='original':cmd+=['--profile-mode',mode,'--profile-scope','generation','--profile-calls','2']
    else:cmd+=['--tree-attention',mode]  # this wrapper enforces its own 1+2/512 recipe
    return cmd+['--out',str(folder/'workload'),'--physical-gpu',str(args.gpu),
        '--requests',str(Path(args.requests).resolve()),'--request-id',args.request_id]


def campaign_headers(path):
    headers=header_environment(path)  # strict .13/version/import/hash/three-directory checks
    prefix=headers['CPATH'].split(':')
    if len(prefix)!=3 or len(set(prefix))!=3:raise ValueError('verified CPATH must have three distinct directories')
    if any(not Path(x).is_dir() for x in KNOWN_LOADER_SUFFIXES):
        raise ValueError('known frozen-loader CPATH suffix directory missing')
    effective=':'.join([*prefix,*KNOWN_LOADER_SUFFIXES])
    return dict(headers,verified_prefix_CPATH=headers['CPATH'],CPATH=effective,
        known_loader_suffixes=list(KNOWN_LOADER_SUFFIXES),
        environment_contract='Exact verified three-directory precedence plus the two frozen-loader suffixes; no other change accepted')


def source_hashes():
    direct=('run_grouped_dram_campaign.py','profile_grouped_gqa_dram.py','profile_implicit_kv_dram.py',
        'profile_ring_dram.py','run_active_dram_campaign.py','run_visual_kv_dram_campaign.py',
        'summarize_dram_campaign.py','run_grouped_gqa_layout.py','grouped_gqa_runtime.py','grouped_gqa_math.py',
        'run_memory_layout.py','run_deployed_adaptor.py','run_joint.py','compiled_adaptor.py','shared_embedding.py',
        'run_stride0_gqa_layout.py','stride0_gqa_runtime.py','stride0_gqa_math.py')
    paths=[HERE/name for name in direct]+[HERE.parent/'hardware_metrics.py',
        HERE.parent/'vendor/hf517_vispec_adapter.py',HERE.parent/'vendor/vispec_joint_adapter.py']
    return {str(path.relative_to(HERE.parent)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validate_arm(folder,arm,args,headers,gpu_uuid):
    profile,mode,script=route(arm)
    validated=profile.validate_workload(folder/'workload',mode,args.request_id)
    status,rows=validated[:2]
    if status['args']['physical_gpu']!=args.gpu or status.get('profiler_effective_cpath')!=headers['CPATH']:
        raise ValueError('child GPU or exact effective CPATH differs from launch')
    if status.get('profiler_source_sha256')!=hashlib.sha256((HERE/script).read_bytes()).hexdigest():
        raise ValueError('child profiler source differs from recorded wrapper')
    expected_sources=source_hashes()
    child_sources=status.get('source_sha256',{})
    if any(child_sources.get(path)!=digest for path,digest in expected_sources.items()):
        raise ValueError('child execution source receipt differs from launch source contract')
    if any(row.get('hardware',{}).get('device',{}).get('uuid')!=gpu_uuid for row in rows):
        raise ValueError('child row GPU UUID differs from supervisor')
    records=profile.bind_ncu(status,rows,folder/'metrics.csv')
    # Tree binder deliberately focuses on the execution receipt; enrich its
    # records using the already validated exact phase/request/repeat row.
    measured={(row['request_id'],row['repeat']):row for row in rows if row['phase']=='measure'}
    for record in records:
        source=measured[(record['request_id'],record['repeat'])]
        record.update({key:source[key] for key in ('input_shapes','timestamps','video_grid_thw')})
    validation=dict(verified=True,cells=len(rows),all_calls_same_output=all(
        row['output_token_ids']==rows[0]['output_token_ids'] for row in rows),
        all_calls_same_rounds=all(row['result']['rounds']==rows[0]['result']['rounds'] for row in rows),
        all_calls_natural_eos=all(row['stop_reason']=='eos' for row in rows))
    protocol={key:status[key] for key in ('target','draft','request_sha256')}
    protocol.update({key:status['args'][key] for key in ('seed','max_new_tokens')})
    binary=folder/'dram.ncu-rep'
    if not binary.is_file() or binary.stat().st_size==0:raise ValueError('missing raw NCU report')
    return dict(records=records,validation=validation,protocol=protocol,
        verified_child_source_sha256=expected_sources,
        workload_pid=status['pid'],profiler_run_id=status['profiler_run_id'],
        child_effective_environment={'CPATH':status['profiler_effective_cpath']},
        binary_report=dict(path=str(binary),bytes=binary.stat().st_size))


def compare_pair(left,right,*,isolated):
    keys=('output_token_ids','rounds','input_shapes','timestamps','video_grid_thw')
    if len(left['records'])!=2 or len(right['records'])!=2:raise ValueError('two bound ranges per arm required')
    pairs=[]
    for a,b in zip(left['records'],right['records']):
        if (a['request_id'],a['repeat'])!=(b['request_id'],b['repeat']):raise ValueError('paired range identity differs')
        checks={key:a[key]==b[key] for key in keys}
        checks['both_natural_eos']=a['stop_reason']==b['stop_reason']=='eos'
        checks['core_forward_counts']={k:v for k,v in a['forward_counts'].items() if k not in INSTRUMENTATION_COUNTS}=={
            k:v for k,v in b['forward_counts'].items() if k not in INSTRUMENTATION_COUNTS}
        checks['unchanged_precision_fields']={k:v for k,v in a['precision'].items() if k not in MODE_PRECISION_KEYS}=={
            k:v for k,v in b['precision'].items() if k not in MODE_PRECISION_KEYS}
        if isolated:
            checks['same_explicit_math_flags']=a['precision']['target_tree_math_precision']==b['precision']['target_tree_math_precision']
        pairs.append(dict(request_id=b['request_id'],repeat=b['repeat'],checks=checks,
            matched_workload=all(checks.values()),
            observed_reduction_pct={k:100*(1-b[k]/a[k]) if a[k] else None for k in ('read_bytes','write_bytes','total_bytes')}))
    protocol_same=left['protocol']==right['protocol']
    stable=all(all(info['validation'][key] for key in ('all_calls_same_output','all_calls_same_rounds','all_calls_natural_eos'))
               for info in (left,right))
    matched=protocol_same and stable and all(pair['matched_workload'] for pair in pairs)
    metrics=('read_bytes','write_bytes','total_bytes')
    medians={name:{k:statistics.median(r[k] for r in info['records']) for k in metrics}
             for name,info in (('reference',left),('candidate',right))}
    changes={key:100*(1-medians['candidate'][key]/medians['reference'][key]) if medians['reference'][key] else None for key in metrics}
    return dict(pairs=pairs,same_protocol=protocol_same,all_calls_stable=stable,matched_workload=matched,
        median_bytes=medians,observed_reduction_pct_of_medians=changes,
        workload_conditioned_reduction_pct=changes if matched else None,
        isolated_attention_comparison=isolated,
        scope='Same shared/compiled/full-cache control; only Target tree attention differs' if isolated else
              'Combined package comparison against original native/native; not individual attention attribution',
        attribution='Review campaign process observations; matching does not prove exclusive process attribution',
        actual_counters=True,logical_copy_bytes_used=False,quality_pass=None,performance_pass=None)


def compare_bound(bound):
    result=dict(counter_scope='Whole generate_joint app-ranges, one warmup excluded, two one-pass measurements',
        ncu_timing_memory_energy_are_performance_evidence=False,no_background_subtraction=True)
    for name,left,right,isolated in (('control_vs_original','original','control',False),
        ('grouped_vs_original','original','grouped',False),('grouped_vs_control','control','grouped',True)):
        if left in bound and right in bound:result[name]=compare_pair(bound[left],bound[right],isolated=isolated)
    return result


def run_campaign(args,*,make_backend,run_process=subprocess.run,sleep=time.sleep):
    manifest=Path(args.requests)
    if not manifest.is_file():raise FileNotFoundError('request manifest missing')
    headers=campaign_headers(args.header_environment);hashes=source_hashes()
    contract=dict(gpu=args.gpu,request_id=args.request_id,request_path=str(manifest.resolve()),
        request_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),warmup=1,repeats=2,max_new_tokens=512,
        ncu=args.ncu,python=sys.executable,allowed_graphics=args.allow_graphics_pid,
        headers=headers,source_sha256=hashes)
    out=Path(args.out)
    if args.resume:
        status=json.loads((out/'campaign.json').read_text())
        if status.get('contract')!=contract:raise ValueError('resume requires the identical source, environment and request contract')
        status.setdefault('history',[]).append({**{key:status.get(key) for key in ('status','error','updated_at')},
            'completed':list(status.get('completed',[]))})
        status.pop('error',None);status.pop('monitor_error',None)
    else:
        out.mkdir(parents=True,exist_ok=False)
        status=dict(status='starting',modes=list(ARMS),completed=[],workload_pids={},guards=[],history=[],
            started_at=time.time(),contract=contract,lock_owner='child runner only',process_poll_interval_s=.5,
            scope='generation',native_ring_full_instrumentation=False,actual_counters_bound=False,
            quality_pass=None,performance_pass=None)
    status.update(pid=os.getpid(),status='starting',updated_at=time.time())
    parent_before={key:os.environ.get(key) for key in ENV_KEYS}
    status['parent_environment_before']=parent_before
    save(out/'campaign.json',status)
    backend,thread=None,None;stop,ready=threading.Event(),threading.Event();observations=[];bound={}
    existing=out/'gpu_observations.jsonl'
    if existing.is_file():observations=[json.loads(line) for line in existing.read_text().splitlines() if line.strip()]

    def watch():
        try:
            with existing.open('a') as stream:
                while not stop.is_set():
                    row=observe(backend);observations.append(row);stream.write(json.dumps(row)+'\n');stream.flush();ready.set();stop.wait(.5)
        except BaseException as exc:status['monitor_error']=repr(exc);ready.set()

    def guards(arm,stage):
        for i in range(2):
            row=observe(backend);row.update(arm=arm,stage=stage,observation_index=i)
            status['guards'].append(row);save(out/'campaign.json',status)
            verify_guard(row,args.allow_graphics_pid,require_idle_util=stage=='before')
            if i==0:sleep(2)

    def retain(folder,arm):
        receipt={};retain_workload_pid(folder,receipt);save(folder/'pid_receipt.json',receipt)
        if 'workload_pid' in receipt:status['workload_pids'][arm]=receipt['workload_pid']

    def bind(folder,arm):
        info=validate_arm(folder,arm,args,headers,status['gpu_uuid']);bound[arm]=info
        save(folder/'workload_validation.json',info['validation'])
        save(folder/'range_binding.json',dict(verified=True,range_count=2,**info))
        if arm not in status['completed']:status['completed'].append(arm)
        save(out/'comparison.json',compare_bound(bound));save(out/'campaign.json',status)

    try:
        backend=make_backend(args.gpu)
        device=dict(gpu_uuid=backend.uuid(),gpu_name=backend.name(),driver=backend.driver_version())
        if args.resume and any(key in status and status[key]!=value for key,value in device.items()):
            raise ValueError('resume physical GPU/driver differs')
        status.update(device)
        thread=threading.Thread(target=watch,name='grouped-dram-nvml',daemon=True);thread.start()
        if not ready.wait(5) or status.get('monitor_error'):raise RuntimeError('GPU observation monitor failed to start')
        for arm in ARMS:
            folder=out/arm;cmd=command(args,folder,arm)
            if folder.exists():
                # Existing raw work is never relaunched, including failed runs.
                if not args.resume:raise FileExistsError(str(folder))
                retain(folder,arm)
                if json.loads((folder/'command.json').read_text())!=cmd:raise ValueError('existing arm command differs')
                if json.loads((folder/'exit.json').read_text()).get('returncode')!=0:raise RuntimeError('existing failed arm preserved; no retry')
                bind(folder,arm);continue
            if arm in status['completed']:raise ValueError('completed arm directory is missing')
            guards(arm,'before')
            if status.get('monitor_error'):raise RuntimeError('observation monitor failed')
            folder.mkdir();env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONUNBUFFERED='1',
                TOKENIZERS_PARALLELISM='false',CPATH=headers['CPATH'])
            save(folder/'command.json',cmd);save(folder/'header_environment.json',headers)
            save(folder/'launch.json',dict(started_unix_s=time.time(),working_directory=os.getcwd(),
                parent_environment={key:os.environ.get(key) for key in ENV_KEYS},
                child_environment_overrides={key:env[key] for key in ENV_KEYS}))
            status.update(status='running',current=arm,updated_at=time.time());save(out/'campaign.json',status)
            print('START '+arm,flush=True);exit_record=dict(started_unix_s=time.time(),returncode=None)
            try:
                with (folder/'application.log').open('w') as stream:
                    child=run_process(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env)
                exit_record['returncode']=child.returncode
            except BaseException as exc:exit_record['error']=repr(exc);raise
            finally:
                exit_record['finished_unix_s']=time.time();save(folder/'exit.json',exit_record);retain(folder,arm)
                save(folder/'environment_after.json',dict(parent={key:os.environ.get(key) for key in ENV_KEYS}))
            # Bind successful counters before the after-guard, so a foreign job
            # starting afterward cannot cause a valid arm to be executed twice.
            if exit_record['returncode']:raise RuntimeError('NCU/workload failed for '+arm)
            if source_hashes()!=hashes:raise ValueError('execution source changed during campaign')
            bind(folder,arm);guards(arm,'after');print('DONE '+arm,flush=True)
        status.update(status='completed',current=None,actual_counters_bound=True,updated_at=time.time())
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),updated_at=time.time());raise
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():status['monitor_error']='monitor did not terminate within five seconds'
        if backend is not None:
            try:backend.close()
            except Exception as exc:status['monitor_error']='backend close failed: '+repr(exc)
        own=set(status['workload_pids'].values());unknown=[];errors=[];residents=[]
        for row in observations:
            compute={p['pid'] for p in row.get('compute',[])};graphics={p['pid'] for p in row.get('graphics',[])}
            foreign=(compute|(graphics-set(args.allow_graphics_pid)))-own
            if foreign:unknown.append(dict(time_unix=row['time_unix'],pids=sorted(foreign)))
            resident=(graphics&set(args.allow_graphics_pid))-own
            if resident:residents.append(dict(time_unix=row['time_unix'],pids=sorted(resident)))
            if any(key.endswith('_error') for key in row):errors.append(row)
        status.update(process_observation_count=len(observations),external_process_observations=unknown,
            resident_graphics_observations=residents,observation_errors=errors,
            attribution='external_work_or_observation_error' if unknown or errors or status.get('monitor_error') else
                'resident_graphics_background_not_excluded' if residents else 'no_foreign_context_observed_not_proven_exclusive',
            parent_environment_after={key:os.environ.get(key) for key in ENV_KEYS},
            environment_observation_scope='Exact parent/launch overrides and child-reported effective CPATH; not a dump of private process environments')
        status['parent_environment_unchanged']=status['parent_environment_after']==parent_before
        if not status['parent_environment_unchanged']:
            status.update(status='failed',environment_error='parent measured environment changed unexpectedly')
        save(out/'campaign.json',status)
    return status


def main(argv=None):
    args=parse_args(argv);sys.path.insert(0,str(HERE.parent))
    from hardware_metrics import _make_backend
    return run_campaign(args,make_backend=_make_backend)


if __name__=='__main__':main()
