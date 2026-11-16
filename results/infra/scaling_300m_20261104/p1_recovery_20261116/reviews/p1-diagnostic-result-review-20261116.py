"""Independent stdlib reconstruction from collected small evidence, no remote IO."""
from pathlib import Path
import hashlib, json, subprocess, math, collections, datetime, tempfile
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/submit_job.py').is_file())
BASE=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_20261116'
E=BASE/'evidence/run_final'
R=json.loads((E/'report.json').read_text()); S=json.loads((E/'snapshot.json').read_text())
EV=[json.loads(x) for x in (E/'events.jsonl').read_text().splitlines()]
PLAN=json.loads((BASE/'submission_plan.json').read_text())
checks=[]
def ck(name,ok):
    checks.append({'name':name,'passed':bool(ok)})
    if not ok: raise AssertionError(name)
def sha(b):return hashlib.sha256(b).hexdigest()
def gitbytes(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
HARD=('dev','ino','size','mode','uid','gid','nlink');TIME=('mtime_ns','ctime_ns');STATE=HARD+TIME
for name,pin in S['files'].items():
    b=(E/name).read_bytes(); ck('collected_bytes_'+name,len(b)==pin['bytes'] and sha(b)==pin['sha256'] and b==pin['text'].encode())
ck('events_pin',sha((E/'events.jsonl').read_bytes())==R['events_sha256'])
ck('sacct_capture',S['commands']['sacct']['exit']==0 and S['commands']['sacct']['stdout']==(E/'sacct.txt').read_text())
ck('squeue_empty',S['commands']['squeue']['exit']==0 and S['commands']['squeue']['stdout']=='' and (E/'squeue.txt').read_bytes()==b'')
ck('events_sequence',[e['sequence'] for e in EV]==list(range(1,len(EV)+1)))
ck('events_monotonic',all(a['monotonic']<=b['monotonic'] for a,b in zip(EV,EV[1:])))
ck('final_event',EV[-1]['kind']=='report' and EV[-1]['details']=={k:R[k] for k in ('complete','outcome','readers_status')})
ck('successful_null',R['complete'] is True and R['outcome']=='NOT_REPRODUCED' and R['readers_status']=='READERS_CLEAR_IN_TEST' and R['errors']==[] and R['metadata_only_cases']==[] and R['automatic_p1_release'] is False)
ck('protocol',R['protocol']=='SCALING-P1-IDENTITY-DIAG-v1' and R['manifest']['protocol']==R['protocol'])
ck('code_commit',R['code_sha']==PLAN['code_sha']=='e4f615e8eb3c33e584d55c6e83cdd48ec9bac4a9')
mpath='results/infra/scaling_300m_20261104/p1_recovery_20261116/diagnostic_manifest.json';mb=gitbytes(R['code_sha'],mpath)
ck('manifest_immutable',sha(mb)==PLAN['manifest_sha256'] and json.loads(mb)==R['manifest'] and mb==(BASE/'diagnostic_manifest.json').read_bytes())
for path,digest in R['source_sha256'].items():ck('executed_source_'+path,sha(gitbytes(R['code_sha'],path))==digest)
oldreview=json.loads((BASE/'reviews/p1-diagnostic-independent-review-20261116.json').read_text())
ck('reviewed_helper_runner',all(oldreview['sha256'][p]==d for p,d in R['source_sha256'].items()))
ck('launcher_pin',sha(gitbytes(R['code_sha'],'scripts/slurm/scaling_p1_identity_diagnostic.sbatch'))==PLAN['diagnostic']['stdin_sha256']==oldreview['sha256']['scripts/slurm/scaling_p1_identity_diagnostic.sbatch'])
ck('frozen_pinset',R['frozen_source_sha256']==R['manifest']['frozen_sources'])
for path,digest in R['frozen_source_sha256'].items():ck('frozen_source_'+path,sha(gitbytes(R['manifest']['frozen_code_sha'],path))==digest)
ct=R['manifest']['candidate_tests'];ctb=gitbytes(R['code_sha'],ct['path']);cpu=json.loads(ctb)
ck('cpu_attestation',sha(ctb)==ct['sha256']==R['candidate_tests']['sha256'] and cpu['passed'] is True and cpu['complete'] is True and cpu['source_sha256']['scripts/scaling_output_identity.py']==R['source_sha256']['scripts/scaling_output_identity.py'])
M=[('fresh','A'),('fresh','F'),('fresh','S'),('aged','S'),('aged','F'),('aged','A')]
ck('matrix',[(x['age'],x['mode']) for x in R['cycles_completed']]==M)
fx=R['fixture_results']; ck('18_unique_fixtures',len(fx)==18 and len({x['path'] for x in fx})==18)
ck('fixed_file_order',[x['id'] for x in fx]==[f'{a}_{m}_{db}' for a,m in M for db in ('db2','db3','db6')])
ck('fixture_events',[x['details'] for x in EV if x['kind']=='fixture_complete']==fx)
ck('tiny_oracle_qualified',R['oracle_qualification']['passed'] is True and len(R['oracle_qualification']['sha256'])==64)
size_ref={'db2':1934763898,'db3':493636450,'db6':2191580650}
state_diffs=[];atime_diffs=0;aged=[];by_mode={}
def observations(a,b,digest,size,label):
    global atime_diffs
    ck(label+'_hashes',a['error'] is None and b['error'] is None and a['sha256']==b['sha256']==digest)
    ck(label+'_descriptor',b['kind']=='descriptor' and a['kind']=='legacy' and b['byte_count']==size and a['path']==b['path']==b['canonical_path'])
    states=[a['stat_before'],a['stat_after'],b['stat_before'],b['stat_after'],b['path_before'],b['path_after']]
    ck(label+'_integer_stats',all(type(s[k]) is int for s in states for k in STATE+('atime_ns',)))
    ck(label+'_size',all(s['size']==size for s in states))
    diffs=sorted({k for s in states for k in STATE if s[k]!=states[0][k]})
    ck(label+'_hard_and_mctime_stable',not diffs)
    ck(label+'_two_second_pause',b['started_monotonic']-a['finished_monotonic']>=2)
    atime_diffs+=sum(s['atime_ns']!=states[0]['atime_ns'] for s in states[1:])
    return states[0]
for x in fx:
    observations(x['first'],x['second'],x['sha256'],x['bytes'],x['id'])
    db=x['id'].split('_')[-1];ck(x['id']+'_size_tolerance',abs(x['bytes']-size_ref[db])<=.01*size_ref[db])
    ck(x['id']+'_candidate',x['candidate']['accepted'] is True and x['candidate']['original_guard_passed'] is True and x['candidate']['timestamp_transition'] is False and x['candidate']['reason']=='stable')
    ck(x['id']+'_npz',x['validation']['passed'] is True and x['validation_drift']==[] and x['legacy_changed_fields']==[])
    ck(x['id']+'_write_closed',x['closed_monotonic']<x['first']['started_monotonic'])
    ck(x['id']+'_age_consistency',abs((x['first']['started_monotonic']-x['closed_monotonic'])-x['age_at_primary_seconds'])<.05)
    if x['age']=='aged' and db!='db6':aged.append(x['age_at_primary_seconds']);ck(x['id']+'_aged_720',x['age_at_primary_seconds']>=720)
    ve=[e for e in EV if e['kind']=='validation_begin' and e['details']['path']==x['path']]
    ck(x['id']+'_validation_boundary',len(ve)==1 and x['first']['finished_monotonic']<ve[0]['monotonic']<x['second']['started_monotonic'])
    by_mode.setdefault(x['age']+'_'+x['mode'],[]).append(x['write_seconds'])
ck('aged_count',len(aged)==6)
ck('fixture_cap',sum(x['bytes'] for x in fx)==R['fixture_bytes_written']<R['limits']['fixture_cap_bytes']==32000000000)
ck('deadline',0<R['elapsed_seconds']<R['limits']['deadline_seconds']==1500)
ck('sync_exposure',all(x['global_sync_calls_before_primary']==2 for x in fx if x['age']=='aged' and x['mode']=='A'))
rr=R['reader_rounds'];pins={x['id']:x for x in R['manifest']['inputs']}; ck('two_reader_rounds',[x['phase'] for x in rr]==['before','after'])
readerev=[x['details'] for x in EV if x['kind']=='reader_complete'];ck('12_reader_events',readerev==[x for r in rr for x in r['inputs']] and len(readerev)==12)
reader_st={};readers=[]
for roundno,round in enumerate(rr):
    ck(f'reader_order_{roundno}',[x['id'] for x in round['inputs']]==list(pins))
    for x in round['inputs']:
        pin=pins[x['id']];digest=pin['sha256'] or x['first']['sha256'];label=f'reader_{roundno}_{x["id"]}'
        base=observations(x['first'],x['second'],digest,pin['bytes'],label)
        fr=x['frozen'];ci=fr['classification_identity'];ref=x['second']['stat_after']
        ck(label+'_guard_ok',not x['affected'] and not x['adoption'] and fr['errors']==[] and x['guard_boundary_drift']==[])
        ck(label+'_classification_pin',ci['sha256']==digest and ci['bytes']==pin['bytes'] and ci['path']==pin['path'])
        ck(label+'_classification_bridge',all(ci[k]==ref[j] for k,j in [('inode','ino'),('bytes','size'),('mtime_ns','mtime_ns'),('ctime_ns','ctime_ns')]))
        want=[ref[k] for k in ('dev','ino','size','mtime_ns','ctime_ns')]
        ck(label+'_pose_bridge',fr['pose_verify']=={pin['path']:want} and fr['pose_final_state']==want)
        if roundno:
            old=reader_st[x['id']];ck(label+'_cross_round',all(ref[k]==old['state'][k] for k in STATE) and digest==old['sha256'] and ci==old['ci'])
        else: reader_st[x['id']]={'state':ref,'sha256':digest,'ci':ci}
        readers.append({'phase':round['phase'],'id':x['id'],'sha256':digest,'historical_pin':pin['sha256'] is not None,'pin_kind':pin['pin_kind']})
rows=[line.split('|') for line in (E/'sacct.txt').read_text().splitlines()];byid={x[0]:x for x in rows}
for jid in ('64610004','64610181'):
    row=byid[jid];ck('completed_'+jid,row[1:3]==['COMPLETED','0:0'])
d=byid['64610181'];tres=dict(x.split('=',1) for x in d[5].split(','))
ck('allocation',d[4]=='8' and tres['cpu']=='8' and tres['gres/gpu']=='1' and tres['mem']=='32G' and d[10]=='00:27:00')
ck('application_within_slurm',R['elapsed_seconds']<=int(d[3])==977 and int(d[3])<1620)
ck('batch_completed',byid['64610181.batch'][1:3]==['COMPLETED','0:0'])
ck('logged_code',R['code_sha'] in (E/'scaling_p1_identity_diag_64610181.out').read_text() and R['code_sha'] in (E/'campaign_checkout_64610004.out').read_text())
ck('thread_env',all(x=='8' for x in R['environment']['thread_environment'].values()))
ck('node_identity',d[7]==R['environment']['allocation']['SLURM_JOB_NODELIST']=='lrdn0105' and R['environment']['allocation']['SLURM_JOB_ID']=='64610181')
account={'diagnostic_elapsed_seconds':977,'diagnostic_gpu_hours':977/3600,'diagnostic_core_hours':977*8/3600,'checkout_elapsed_seconds':int(byid['64610004'][3]),'checkout_core_hours':int(byid['64610004'][3])*int(byid['64610004'][4])/3600,'total_core_hours':(977*8+41*2)/3600,'maxrss_batch_gib':int(byid['64610181.batch'][6][:-1])/1024**2}
out={'verdict':'ESITO TECNICO CONFORME; diagnosi NOT_REPRODUCED, READERS_CLEAR_IN_TEST','all_checks_passed':True,'checks':checks,'source_report_sha256':sha((E/'report.json').read_bytes()),'source_events_sha256':sha((E/'events.jsonl').read_bytes()),'source_sacct_sha256':sha((E/'sacct.txt').read_bytes()),'fixture_count':18,'cycles':M,'fixture_bytes':R['fixture_bytes_written'],'aged_6_min_max_seconds':[min(aged),max(aged)],'write_seconds_min_max':[min(x['write_seconds'] for x in fx),max(x['write_seconds'] for x in fx)],'write_seconds_by_mode':by_mode,'atime_state_differences_observed':atime_diffs,'mtime_ctime_or_hard_drift_observed':0,'readers':readers,'frozen_source_pin_count':len(R['frozen_source_sha256']),'event_count':len(EV),'accounting':account,'interpretation':['Zero reproductions in 18 correlated synthetic fixtures. This neither proves a Lustre cause nor proves that barriers cure it.','All 12 sampled real-cache guard rounds match their available pins and each other; this does not guarantee 43-hour reader stability.','Two failed caches had no historical digest pin; their values are diagnostic baselines, not accepted scientific artifacts.','No third P1 inference or reader release follows automatically from this null diagnostic. The failure cause remains open.'],'limitations':['Review of collected small records only; no SSH, no read of large remote NPZs, no independent rerun of their hashes.','Byte-oracle and schema results independently cross-checked in the report/events, not recomputed from large fixtures.'],'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=(Path(tempfile.gettempdir())/'p1-diagnostic-result-review-20261116.json');p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('checks','readers')},indent=2));print('CHECKS',len(checks));print(p)
