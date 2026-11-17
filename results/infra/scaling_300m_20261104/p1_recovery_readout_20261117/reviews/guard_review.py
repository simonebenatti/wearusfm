"""Independent inspection of collected P1-r2 evidence; no remote IO or large cache loads."""
from pathlib import Path
import datetime, hashlib, json, math, re, stat, subprocess, sys
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-p1-recovery')
BASE=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_readout_20261117';E=BASE/'evidence'
S=json.loads((E/'snapshot.json').read_text());R=json.loads((E/'p1/receipt.json').read_text());M=json.loads((E/'release/recovery_manifest.json').read_text());P=json.loads((E/'release/extraction_plan.json').read_text());C=json.loads((E/'release/training_source_contract.json').read_text());EV=[json.loads(x) for x in (E/'p1/output_identity_events.jsonl').read_text().splitlines()]
checks=[]
def ck(name,ok):
 checks.append({'name':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def blob(path):return subprocess.check_output(['git','show',R['code_sha']+':'+path],cwd=ROOT)
HARD=('dev','ino','size','mode','uid','gid','nlink');TIME=('mtime_ns','ctime_ns');FIELDS=HARD+('atime_ns',)+TIME
ck('complete_receipt',R['complete'] is True and R['stage']=='extract-classification' and R['checkpoint_unchanged'] is True and R['postload_recipe_passed'] is True)
ck('fixed_release',R['slurm_job_id']=='64620434' and R['logical_job_id']=='SCALING-300M-eval19618-p1-r2-20261116' and R['code_sha']=='e4ebdbfa709c3f83cb5b1b11bdd261d132ccae3a')
ck('new_output_only',M['out'].endswith('/T100-G-300M/p1-r2') and R['failed_attempts_preserved']==M['failed_attempts'] and M['budget']['retry_authorized'] is False)
ck('criterion_and_null_preserved',R['release_criterion']==M['release_criterion']=='scaling-p1-single-recovery-after-clean-diagnostic-v2' and R['diagnostic_outcome']==M['diagnostic_outcome']=='NOT_REPRODUCED' and R['output_barrier']==M['output_barrier']=='none')
ck('manifest_pin',sha(E/'release/recovery_manifest.json')==R['recovery_manifest_sha256']=='1a2104c59e55f5f3c46e03a984bdcda163ef26634c69cc1431ce3319bf878ee3')
ck('original_plan_pin',sha(E/'release/extraction_plan.json')==R['plan_sha256']==M['extraction_plan']['sha256']=='3cb94b050fa351c01e86184954f542b01b6c939169efc663d869d134e93b4556')
ck('diagnostic_pinned',R['diagnostic_report']==M['diagnostic_report'] and R['diagnostic_report']['sha256']=='c2e32e9cfab36a76c79b05913e350b9ec7f15484f73e2a271552c06f6e980908')
ck('checkpoint_endpoint',R['arm']=='T100-G-300M' and R['which']=='teacher' and R['checkpoint_step']==19618 and R['checkpoint_sha256']==P['checkpoints']['T100-G-300M']['sha256'])
ck('contract_pin',sha(E/'release/training_source_contract.json')==R['training_source_contract_sha256'])
ck('helper_pin',R['recovery_source_sha256']['scripts/scaling_output_identity.py']=='717362c25cb99151ab058b4f212cc73611a36899f67251952ca613a71058e656')
ck('source_manifest_match',R['recovery_source_sha256']==M['source_sha256'])
for path,h in R['recovery_source_sha256'].items():
 ck('reviewed_source_'+path,hashlib.sha256(blob(path)).hexdigest()==h)
 ck('source_guard_'+path,R['source_states']['recovery_source_'+path]['sha256']==h)
for path,h in C['scientific_source_sha256'].items():
 ck('scientific_source_'+path,R['source_states']['scientific_source_'+path]['sha256']==h and hashlib.sha256(blob(path)).hexdigest()==h)
 if path in S['sources']:ck('remote_source_'+path,S['sources'][path]==h)
ck('scientific_count',len(C['scientific_source_sha256'])==103)
ck('helper_remote',S['sources']['scripts/scaling_output_identity.py']==M['source_sha256']['scripts/scaling_output_identity.py'])
ck('wrapper_remote',S['sources']['scripts/scaling_p1_recovery.py']==M['source_sha256']['scripts/scaling_p1_recovery.py'])
ck('checkout_current_clean',S['commands']['code_head']['exit']==0 and S['commands']['code_head']['stdout'].strip()==R['code_sha'] and S['commands']['code_status']['exit']==0 and S['commands']['code_status']['stdout']=='')
ck('commands_success',all(x['exit']==0 for x in S['commands'].values()))
ck('raw_input_guard',set(R['raw_inputs'])=={'p1'} and R['raw_inputs']['p1']['unchanged'] is True and len(R['raw_inputs']['p1']['file_states'])==360)
ck('one_producer_command',set(R['commands'])=={'p1'})
ck('runtime_same_as_p2',R['runtime']==json.loads((E/'p2/receipt.json').read_text())['runtime'])
ck('model_recipe',R['model_config']['dim']==896 and R['model_config']['tokenizer_activation']=='gelu' and R['model_config']['window_ms']==100 and R['model_config']['patch_ms']==25)
for key,pin in S['files'].items():
 p=E/key;b=p.read_bytes();ck('collected_'+key,len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['sha256'] and b==pin['text'].encode())
 ck('collection_stability_'+key,all(pin['before'][f]==pin['after'][f] for f in pin['before'] if f!='atime_ns'))
ck('event_file_pin',sha(E/'p1/output_identity_events.jsonl')==R['output_identity_events_sha256'])
DATASETS=['ninapro_db2','ninapro_db3','ninapro_db6'];ck('datasets_exact',list(R['outputs'])==DATASETS)
flat=[];summary=[];atime_changes=0
for ds in DATASETS:
 out=R['outputs'][ds];a=out['output_identity_audit'];pr=json.loads((E/'p1'/Path(out['provenance_path']).name).read_text());obs=a['observations'];flat.extend(obs)
 ck(ds+'_audit_sidecar',pr['output_identity_audit']==a and sha(E/'p1'/Path(out['provenance_path']).name)==out['provenance_sha256'])
 ck(ds+'_sidecar_pins',pr['cache']==out['path'] and pr['cache_sha256']==out['sha256'] and pr['cache_bytes']==out['bytes'])
 ck(ds+'_sidecar_shared',all(pr[k]==R[k] for k in ['protocol','arm','which','checkpoint_step','checkpoint_sha256','code_sha','plan_sha256','postload_recipe_passed','model_config','stats_sha256','training_source_contract_sha256','recovery_protocol','recovery_manifest_sha256','recovery_source_sha256','output_barrier','release_criterion','diagnostic_outcome','diagnostic_report']))
 ck(ds+'_cache_schema',pr['dataset']==ds and pr['task']=='p1' and pr['pool_dtype']=='<f4' and pr['pool_shape']==[pr['n_windows'],1792])
 ck(ds+'_audit_completed',a['complete'] is True and a['validator_started'] is True and a['validator_completed'] is True and a['policy']=='scaling-output-identity-20261116-v1')
 ck(ds+'_bounded_reads',type(a['extra_reads']) is int and a['extra_reads'] in (0,1) and len(obs)==2+a['extra_reads'])
 allstates=[]
 for i,o in enumerate(obs):
  lab=f'{ds}_observation_{i}';states=[o[x] for x in ('path_before','stat_before','stat_after','path_after')];allstates.extend(states)
  ck(lab+'_identity',o['kind']=='descriptor' and o['error'] is None and o['path']==o['canonical_path']==out['path']==M['out']+'/'+ds+'.npz')
  ck(lab+'_digest',o['sha256']==out['sha256'] and re.fullmatch('[0-9a-f]{64}',o['sha256']) is not None and type(o['byte_count']) is int and o['byte_count']==out['bytes'])
  ck(lab+'_integer_states',all(type(s.get(k)) is int for s in states for k in FIELDS))
  ck(lab+'_regular_file',all(stat.S_ISREG(s['mode']) and s['nlink']>=1 and s['size']==out['bytes'] for s in states))
  ck(lab+'_hard_stable',all(all(s[k]==states[0][k] for k in HARD) for s in states))
  ck(lab+'_duration',0<o['duration_seconds'] and math.isclose(o['duration_seconds'],o['finished_monotonic']-o['started_monotonic'],abs_tol=1e-9))
  ck(lab+'_changed_fields',[k for k in FIELDS if o['stat_before'][k]!=o['stat_after'][k]]==o['changed_fields'])
  atime_changes+=int(o['stat_before']['atime_ns']!=o['stat_after']['atime_ns'])
  if i:ck(lab+'_chronological',o['started_monotonic']>=obs[i-1]['finished_monotonic'])
 ck(ds+'_all_digests_identical',len({o['sha256'] for o in obs})==1)
 ck(ds+'_object_preserved',all(all(s[k]==allstates[0][k] for k in HARD) for s in allstates))
 time_drift=any(any(s[k]!=allstates[0][k] for k in TIME) for s in allstates)
 ck(ds+'_timestamp_flag_matches_evidence',a['timestamp_transition'] is time_drift)
 finalstates=[obs[-1][n] for n in ('path_before','stat_before','stat_after','path_after')]
 ck(ds+'_last_timestamp_stable',all(all(s[k]==finalstates[0][k] for k in TIME) for s in finalstates))
 firststates=[obs[0][n] for n in ('path_before','stat_before','stat_after','path_after')]
 initial_unstable=any(any(s[k]!=firststates[0][k] for k in TIME) for s in firststates)
 if a['extra_reads']==0:ck(ds+'_no_settling_needed',not time_drift and len(obs)==2)
 else:
  ck(ds+'_extra_read_requires_drift',time_drift)
  retryidx=1 if initial_unstable else 2
  ck(ds+'_retry_pause',obs[retryidx]['started_monotonic']-obs[retryidx-1]['finished_monotonic']>=2)
  if initial_unstable:
   later=[obs[j][n] for j in (1,2) for n in ('path_before','stat_before','stat_after','path_after')]
   ck(ds+'_no_second_instability',all(all(s[k]==later[0][k] for k in TIME) for s in later))
 current=S['cache_states'][out['path']];ck(ds+'_current_metadata_matches',not current['symlink'] and current['canonical']==out['path'] and all(current[k]==allstates[-1][k] for k in HARD+TIME))
 summary.append({'dataset':ds,'bytes':out['bytes'],'sha256':out['sha256'],'complete_hash_reads':len(obs),'extra_reads':a['extra_reads'],'timestamp_transition':time_drift,'validator_complete':True,'windows':pr['n_windows'],'users':pr['n_users'],'pool_shape':pr['pool_shape'],'hash_seconds':[o['duration_seconds'] for o in obs]})
ck('event_count_exact',len(EV)==2*len(flat))
for i,o in enumerate(flat):
 start,end=EV[2*i:2*i+2]
 ck('event_start_'+str(i),start=={'event':'identity_observation_start','kind':o['kind'],'path':o['path'],'started_utc':o['started_utc']})
 ck('event_end_'+str(i),end=={'event':'identity_observation_end','observation':o})
ck('six_normal_hashes',len(flat)==6 and len(EV)==12 and sum(x['extra_reads'] for x in summary)==0)
ck('validator_between_reads_from_frozen_code',b'details = validator(Path(path))' in blob('scripts/scaling_output_identity.py') and b'after = observe()' in blob('scripts/scaling_output_identity.py'))
ck('no_checkpoint_stop',S['checkpoint_stop_exists'] is False)
cpstate=R['source_states']['checkpoint'];cpnow=S['checkpoint_state'];ck('checkpoint_metadata_stable',cpstate['bytes']==cpnow['size']==P['checkpoints']['T100-G-300M']['bytes'] and cpstate['inode']==cpnow['ino'] and cpstate['mtime_ns']==cpnow['mtime_ns'] and cpstate['ctime_ns']==cpnow['ctime_ns'])
ck('sacct_capture',(E/'sacct.txt').read_text()==S['commands']['sacct']['stdout'])
rows={row[0]:row for l in (E/'sacct.txt').read_text().splitlines() if (row:=l.split('|'))};job=rows['64620434'];checkout=rows['64620427'];tres=dict(x.split('=',1) for x in job[5].split(','))
ck('job_completed',job[1:3]==['COMPLETED','0:0'] and rows['64620434.batch'][1:3]==['COMPLETED','0:0'])
ck('allocation',job[4]=='8' and tres['gres/gpu']=='1' and tres['mem']=='100G' and job[10]=='02:00:00')
ck('duration',int(job[3])==1772 and 0<R['elapsed_seconds']<1772<7200)
ck('checkout_completed',checkout[1:3]==['COMPLETED','0:0'] and checkout[3:5]==['34','2'])
ck('extract_log_complete','estrazione finita in 1709 s' in (E/'p1/p1.log').read_text() and all(ds in (E/'p1/p1.log').read_text() for ds in DATASETS))
account={'p1_elapsed_seconds':1772,'p1_gpu_hours':1772/3600,'p1_core_hours':1772*8/3600,'checkout_core_hours':34*2/3600,'p1_maxrss_gib':int(rows['64620434.batch'][6][:-1])/1024**2}
payload_path=E/'payload_stream_verification.json'
PV=json.loads(payload_path.read_text())
ck('payload_stream_complete',PV['complete'] is True and PV['hash_computed_on']=='Mac' and PV['cache_copies_retained'] is False)
ck('payload_three_exact',len(PV['records'])==3 and [r['dataset'] for r in PV['records']]==DATASETS)
for rec in PV['records']:
 ds=rec['dataset'];outrec=R['outputs'][ds];lab=ds+'_fresh_full_readback'
 ck(lab+'_digest',rec['accepted'] is True and rec['exit']==0 and rec['path']==outrec['path'] and rec['sha256']==outrec['sha256'] and rec['bytes']==outrec['bytes'])
 parsed=[json.loads(line[len('P1_TRANSFER '):]) for line in rec['stderr'].splitlines() if line.startswith('P1_TRANSFER ')]
 ck(lab+'_raw_events',parsed==rec['observations'] and [v['event'] for v in parsed]==['before','after'])
 states=[parsed[0]['value'][k] for k in ('path','descriptor')]+[parsed[1]['value'][k] for k in ('path','descriptor')]
 ck(lab+'_byte_count',parsed[1]['value']['bytes']==rec['bytes'])
 ref=outrec['output_identity_audit']['observations'][-1]['stat_after'];now=S['cache_states'][outrec['path']]
 ck(lab+'_protected_stats',all(type(st[k]) is int and st[k]==ref[k]==now[k] for st in states for k in HARD+TIME))
 ck(lab+'_regular_file',all(stat.S_ISREG(st['mode']) and st['nlink']>=1 for st in states))
ck('payload_total_bytes',PV['total_bytes']==sum(r['bytes'] for r in PV['records'])==4619980998)
out={'verdict':'P1-r2 guardia e rilettura integrale CONFORMI; tre cache finalizzate senza errori né stabilizzazioni' ,'all_checks_passed':True,'checks':checks,'receipt_sha256':sha(E/'p1/receipt.json'),'events_sha256':sha(E/'p1/output_identity_events.jsonl'),'snapshot_sha256':sha(E/'snapshot.json'),'sacct_sha256':sha(E/'sacct.txt'),'sources':{'code_sha':R['code_sha'],'helper_sha256':M['source_sha256']['scripts/scaling_output_identity.py'],'scientific_pins':103},'cache_audits':summary,'event_count':len(EV),'atime_only_changes_in_hash_reads':atime_changes,'total_cache_bytes':sum(x['bytes'] for x in summary),'accounting':account,'runtime':R['runtime'],'causal_reading':['Il run è completo e l’integrità richiesta dalla nuova guardia è attestata in receipt, sidecar ed eventi coerenti.','Nessuna cache ha richiesto la rilettura aggiuntiva o la tolleranza di mtime/ctime: quel ramo della policy non è stato esercitato.','Questo successo non dimostra che i fallimenti storici fossero causati da Lustre né che la stabilizzazione degli attributi li abbia risolti.','Atime è variato durante tutte le sei letture, come previsto; la guardia originale già lo escludeva.','La nuova guardia usa un descriptor e osservazioni diverse dalla vecchia: non si può concludere retrospettivamente che la vecchia avrebbe sicuramente superato questo stesso run.'],'payload_stream_sha256':sha(payload_path),'payload_full_readback':{'complete':True,'records':3,'bytes':PV['total_bytes'],'hash_computed_on':'Mac','all_protected_stats_unchanged':True},'reader_readiness':'Tre nuovi payload P1 riletti integralmente via stream, SHA verificati sul Mac e quattro stat protetti stabili rispetto ad audit e snapshot. Provenienze idonee alla composizione con pose/P2; il controller e i cancelli originali di validazione/benchmark dei lettori restano necessari prima dei fit.','limitations':['Il parent ha trasferito i payload e calcolato gli hash sul Mac; questo revisore ha ricalcolato i confronti su receipt, report di stream e stat grezzi, senza duplicare il trasferimento.','Nessun SSH, job, modifica di repository o test di implementazione.'],'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=Path('/private/tmp/p1-r2-guard-review-20261117.json');p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(p);print('checks',len(checks));print(json.dumps({k:v for k,v in out.items() if k not in ['checks','causal_reading','limitations','runtime']},indent=2))
