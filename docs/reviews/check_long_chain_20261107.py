#!/usr/bin/env python3
"""Read-only provenance audit of published WearUsFM long campaign c109bfe.
Usage: python3 audit.py --snapshot-root /path/to/c109bfe/worktree --out /path/outside/repo
No cluster access, scientific payload download, fitting, inference, or repo edit.
"""
import argparse,base64,collections,csv,hashlib,io,json,math,pathlib,shlex,subprocess,importlib.util,sys
sys.dontwritebytecode=True
P=argparse.ArgumentParser();P.add_argument('--snapshot-root','--repo',dest='snapshot_root',type=pathlib.Path,required=True);P.add_argument('--out',type=pathlib.Path,required=True);P.add_argument('--ledger',type=pathlib.Path,default=pathlib.Path.home()/'wearusfm_local/jobs');args=P.parse_args()
ROOT=args.snapshot_root.resolve();LEDGER=args.ledger.expanduser().resolve();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
BASE='results/infra/consolidation_long_20261105';CAMP=ROOT/BASE;E=CAMP/'execution';COMMIT='c109bfef5a3f80cc4b2334e3554c2171db938467'
assert OUT!=ROOT and ROOT not in OUT.parents
checks=[];evidence={};warnings=[]
def check(ok,name,detail=None):
 checks.append({'check':name,'passed':bool(ok),**({'detail':detail} if detail is not None else {})})
 if not ok: raise AssertionError(name+': '+repr(detail))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(x):return sha(json.dumps(x,sort_keys=True,allow_nan=False).encode())
def j(p):return json.loads(p.read_text())
def git(*argv):return subprocess.check_output(['git',*argv],cwd=ROOT)
def gblob(ref,path):return git('show',ref+':'+path)
def local_pin(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size}
def check_pin(pin,path,name):check(sha(path.read_bytes())==pin['sha256'] and ('bytes' not in pin or len(path.read_bytes())==pin['bytes']),name)
check(git('rev-parse','HEAD').decode().strip()==COMMIT,'pinned_delivery_commit')
check(not git('status','--porcelain').strip(),'clean_review_worktree_before')
plan=j(CAMP/'plan.json');launch=j(CAMP/'launch_plan.json');summary_plan=j(E/'summary_plan.json')
phase_shas={x['name'].split('consolidation-long-')[1].split('-20261105')[0]:x['code_sha'] for x in launch['phase_checkouts']}
expected_shas={'extract':'7d276c98fc54d66dab1226226eb6c5e447422c69','benchmark':'166fb8653b53439507183d76e3e36a79dd28d166','fit':'a70e29e4abb09e6e23fad34ca6b7b9e28874952a','aggregate':'d75106304d1592fbea6d81fc4c1220e669cffecf'}
check(phase_shas==expected_shas,'four_frozen_phase_commits',phase_shas)
# Hash file contents directly from immutable Git objects, using a single batch.
def source_map(ref):
 entries=[]
 for entry in git('ls-tree','-r','-z',ref).split(b'\0'):
  if not entry:continue
  meta,path=entry.split(b'\t');name=path.decode();kind=meta.split()[1]
  covered=(name.startswith('src/') and name.endswith('.py')) or (name.startswith('scripts/') and '/' not in name[len('scripts/'):] and name.endswith('.py')) or name in ('scripts/slurm/consolidation_long.sbatch','scripts/slurm/lib_code_dir.sh')
  if covered:
   check(kind==b'blob','covered_source_is_blob:'+ref[:7]+':'+name)
   entries.append((name,meta.split()[2].decode()))
 proc=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=('\n'.join(oid for _,oid in entries)+'\n').encode(),stdout=subprocess.PIPE,check=True)
 stream=io.BytesIO(proc.stdout);result={}
 for path,oid in entries:
  header=stream.readline().decode().strip().split();check(header[:2]==[oid,'blob'],'git_batch_blob:'+ref[:7]+':'+path)
  raw=stream.read(int(header[2]));check(stream.read(1)==b'\n','git_batch_separator:'+ref[:7]+':'+path);result[path]=sha(raw)
 return result
maps={phase:source_map(ref) for phase,ref in phase_shas.items()}
source_hashes={phase:digest(m) for phase,m in maps.items()};input_manifest=j(CAMP/'amendment_a1/input_compatibility.json')
check(maps['extract']==maps['benchmark']==input_manifest['source_files'],'original_sources_227_exact')
check(maps['fit']==maps['aggregate'],'amended_source_frozen_fit_through_aggregate')
check(source_hashes['extract']==input_manifest['source_sha256']=='1ca7aead5cd48a1fd9ff398338d80f2471a8d4041ccc0ac7fcdb3c3c03d0b1e1','original_source_hash')
check(source_hashes['fit']==plan['source_sha256']=='77e09626271b1aeded8d7f3df77ef9557115e903a9aca8e280e3f3259527ac88','amended_source_hash')
changed=sorted(p for p in maps['benchmark'] if maps['fit'].get(p)!=maps['benchmark'][p]);added=sorted(set(maps['fit'])-set(maps['benchmark']))
policy=plan['phase_release']['source_change_policy']['amendment_A1']
check(changed==sorted(policy['changed_paths']) and added==sorted(policy['added_paths']),'closed_source_file_change_list',{'changed':changed,'added':added})
spec=importlib.util.spec_from_file_location('audited_amendment_runtime',ROOT/'scripts/consolidation_long_amendment.py');amendment=importlib.util.module_from_spec(spec);spec.loader.exec_module(amendment)
validated=amendment.validate_inputs(input_manifest,maps['aggregate'],{path:(ROOT/path).read_text() for path in amendment.CHANGED_PATHS},digest)
check(validated==input_manifest,'A1_regions_order_launcher_and_28_input_validation')
scientific=[p for p in maps['benchmark'] if p.startswith('src/')]+['scripts/consolidation_long_classification.py','scripts/consolidation_long_pose.py','scripts/consolidation_long.py']
check(all(maps['benchmark'][p]==maps['fit'][p] for p in scientific),'scientific_leaf_and_src_byte_equality',{'files':len(scientific)})
phase_plan_hash={phase:sha(gblob(ref,BASE+'/plan.json')) for phase,ref in phase_shas.items()}
phase_plans={phase:json.loads(gblob(ref,BASE+'/plan.json')) for phase,ref in phase_shas.items()}
for phase,v in phase_plans.items():check(v['source_sha256']==source_hashes[phase] and v['recipe_sha256']==plan['recipe_sha256'],'phase_plan_identity:'+phase)
# Recorded allocations, logical identifiers, attempts, copied logs and receipts.
rows={};states={};receipts={};audit_receipts={};logpins={};accounting={}
for phase in phase_shas:
 sacct=list(csv.DictReader(io.StringIO((E/f'accounting/sacct_{phase}.tsv').read_text()),delimiter='\t'));gh=ch=0.
 for row in sacct:
  ident=row['JobName'];check(ident not in rows,'unique_sacct_logical_id:'+ident);rows[ident]=row
  check(row['State']=='COMPLETED' and row['ExitCode']=='0:0' and int(row['Restarts'])==0,'completed_once:'+ident)
  tres=dict(x.split('=',1) for x in row['AllocTRES'].split(','));gh+=int(row['ElapsedRaw'])*int(tres.get('gres/gpu','0'))/3600;ch+=int(row['ElapsedRaw'])*int(row['AllocCPUS'])/3600
 a=j(E/f'audit/receipts_{phase}.json');audit_receipts.update(a['receipts']);logs=j(E/f'audit/logs_{phase}.json')['logs'];logpins.update(logs)
 check(len(sacct)==a['accounting']['rows'] and abs(gh-a['accounting']['gpu_hours'])<5.1e-7 and abs(ch-a['accounting']['allocated_core_hours'])<5.1e-7,'phase_accounting_recomputed:'+phase)
 accounting[phase]={'allocations':len(sacct),'gpu_hours':gh,'allocated_core_hours':ch,'elapsed_allocation_seconds':sum(int(r['ElapsedRaw']) for r in sacct)}
for p in sorted((E/'submission').glob('*.json')):states[p.stem]=j(p)
for p in sorted((E/'receipts').glob('*.json')):receipts[p.stem]=j(p)
jobrows={r['id']:r for r in launch['jobs']};checkouts={r['id']:r for r in launch['phase_checkouts']}
check(len(rows)==len(states)==len(logpins)==107 and set(rows)==set(states)==set(logpins)==set(jobrows)|set(checkouts),'107_identical_allocation_sets')
check(len(receipts)==len(audit_receipts)==len(jobrows)==103 and set(receipts)==set(audit_receipts)==set(jobrows),'103_identical_receipt_sets')
check(len({r['JobIDRaw'] for r in rows.values()})==107,'107_distinct_slurm_ids')
check(all(r['id'] not in states for r in launch['superseded_jobs']) and len(launch['superseded_jobs'])==48,'48_superseded_ids_absent')
live_state_equal=0
for ident,state in states.items():
 row=rows[ident];item=jobrows.get(ident,checkouts.get(ident));phase=item['phase'] if ident in jobrows else ident.split('LONG-checkout-')[1].split('-20261105')[0]
 expected_code=phase_shas[phase]
 check(state['id']==ident and state['stato']=='sottomesso' and state['agente']=='claude' and str(state['job_slurm'])==row['JobIDRaw'],'submission_identity:'+ident)
 check([x['stato'] for x in state['storia']]==['prenotato','in_sottomissione','sottomesso'],'single_submission_attempt:'+ident)
 check(state['codice_sha']==expected_code,'submission_code:'+ident)
 argv=state['sbatch'];check('--no-requeue' in argv and '--job-name='+ident in argv and '--comment=wearusfm:'+ident in argv,'submission_guard:'+ident)
 check(any('EXPECTED_SHA='+expected_code in x for x in argv),'export_code:'+ident)
 live=LEDGER/ident/'state.json'
 if live.exists():check(j(live)==state,'current_local_ledger_equals_archived:'+ident);live_state_equal+=1
 log=E/'logs'/(ident+'.log');check_pin(logpins[ident],log,'log_hash:'+ident)
 if ident in checkouts:
  check(state['script_stdin_sha256']==sha(gblob(expected_code,'scripts/slurm/campaign_checkout.sbatch')),'checkout_stdin_hash:'+ident)
  check(expected_code in log.read_text() and 'checkout di campagna pronto' in log.read_text(),'checkout_log_provenance:'+ident)
  check(int(row['AllocCPUS'])<=2,'checkout_cpu_cap:'+ident)
 else:
  r=receipts[ident];check_pin(audit_receipts[ident],E/'receipts'/(ident+'.json'),'receipt_hash:'+ident)
  expected_plan=sha((E/'summary_plan.json').read_bytes()) if r['kind']=='summary' else phase_plan_hash[phase]
  check(r['complete'] is True and r['kind']==item['stage'] and r['code_sha']==expected_code and r['source_sha256']==source_hashes[phase] and r['recipe_sha256']==plan['recipe_sha256'] and r['plan_sha256']==expected_plan,'receipt_identity:'+ident)
  check(audit_receipts[ident]['remote_path']==item['out']+'/report.json','receipt_path:'+ident)
  check(int(row['AllocCPUS'])==item['allocation']['cpus'] and int(row['ElapsedRaw'])<=item['allocation']['seconds'],'allocation_duration_cpu:'+ident)
  events=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"event": "allocation_accepted"')]
  check(len(events)==1 and events[0]['allocation']==item['allocation'] and events[0]['stage']==item['stage'],'launcher_resource_acceptance:'+ident)
  if phase=='extract':check(r['checkpoint_unchanged'] is True and r['frozen_teacher_unchanged'] is True,'extraction_teacher_immutability:'+ident)
  else:check(r['checkpoint_unchanged'] is False and r['frozen_teacher_unchanged'] is False,'cpu_no_model_loaded_flags:'+ident)
  if phase in ('fit','aggregate'):check(r['amendment']=={'id':'A1','input_source_sha256':input_manifest['source_sha256'],'input_compatibility':plan['amendment']['input_compatibility']},'new_receipt_amendment:'+ident)
check(live_state_equal==107,'current_local_ledger_107_equal',live_state_equal)
# Pin identities as consumed by each frozen phase; no remote payload hashing claimed.
remote_to_id={p['remote_path']:ident for ident,p in audit_receipts.items()}
def receipt_for(pin,name):
 check(pin['path'] in remote_to_id,'listed_receipt_path:'+name);ident=remote_to_id[pin['path']];check_pin(pin,E/'receipts'/(ident+'.json'),'listed_receipt_pin:'+name);return receipts[ident]
for ent in input_manifest['receipts']:
 r=receipt_for(ent,ent['id']);phase=ent['phase'];check(r['source_sha256']==input_manifest['source_sha256'] and r['code_sha']==phase_shas[phase] and r['plan_sha256']==phase_plan_hash[phase] and all(r[k]==v for k,v in ent['fields'].items()),'bridge_original_fields:'+ent['id'])
check(len(input_manifest['receipts'])==28,'bridge_closed_28_original_receipts')
check_pin(plan['amendment']['input_compatibility'],CAMP/'amendment_a1/input_compatibility.json','bridge_manifest_pin')
for key in ('cache_manifest','classification_fit_manifest','pose_fit_manifest'):
 check_pin(plan[key],ROOT/plan[key]['path'],'final_plan_'+key)
check_pin(plan['benchmark_receipts']['classification'],ROOT/plan['benchmark_receipts']['classification']['path'],'benchmark_manifest_pin')
for key in ('cache_manifest',):check(phase_plans['benchmark'][key]==plan[key]==input_manifest[key],'cache_manifest_frozen_across_phases')
check(phase_plans['fit']['benchmark_receipts']==plan['benchmark_receipts'],'benchmark_pins_frozen_from_fit')
for key in ('classification_fit_manifest','pose_fit_manifest'):check(phase_plans['aggregate'][key]==plan[key],'fit_manifest_frozen_from_aggregate:'+key)
cache=j(ROOT/plan['cache_manifest']['path']);cache_count=0
for stage,groups in cache.items():
 if stage not in ('classification','pose'):continue
 for step,items in groups.items():
  for tag,entry in items.items():
   r=receipt_for(entry['receipt'],'cache:'+stage+':'+step+':'+tag);check(entry['cache'] in r['artifacts'].values() and r['checkpoint_step']==int(step),'cache_pin_bound_to_extraction:'+stage+':'+step+':'+tag);cache_count+=1
check(cache_count==21,'21_cache_pins_bound')
fit_manifest=j(ROOT/plan['classification_fit_manifest']['path']);fit_pose=j(ROOT/plan['pose_fit_manifest']['path']);bench_manifest=j(ROOT/plan['benchmark_receipts']['classification']['path'])
for ds,vs in bench_manifest['reports'].items():
 for view,pin in vs.items():r=receipt_for(pin,'benchmark:'+ds+':'+view);check(r['dataset']==ds and r['view']==view,'benchmark_manifest_cell:'+ds+':'+view)
receipt_for(plan['benchmark_receipts']['pose'],'pose_benchmark')
folds=collections.defaultdict(list);fit_ratios=[];numeric=set();allartifactpins={}
for pin in fit_manifest['reports']:
 r=receipt_for(pin,'classification_fit');key=f"{r['checkpoint_step']}/{r['view']}/{r['dataset']}/folds"+'-'.join(map(str,r['fold_indices']));table=plan['amendment']['fit_shards'][key];g=r['resource_gate']
 check(r['benchmark_manifest']==plan['benchmark_receipts']['classification'] and r['global_resource_gate_passed'] is True and g['passed'] is True and g['walltime_seconds']==table['walltime_seconds'] and abs(g['projected_seconds']-table['projected_seconds'])<=1e-6,'fit_resource_pins:'+key)
 check(r['checkpoint_sha256']==plan['endpoints'][str(r['checkpoint_step'])]['checkpoint']['sha256'],'fit_checkpoint:'+key)
 folds[(r['checkpoint_step'],r['view'],r['dataset'])].extend(r['fold_indices']);numeric.add(r['historical_numeric_match']['numeric_sha256'])
 ident=remote_to_id[pin['path']];fit_ratios.append({'id':ident,'elapsed_seconds':int(rows[ident]['ElapsedRaw']),'walltime_seconds':g['walltime_seconds'],'fraction':int(rows[ident]['ElapsedRaw'])/g['walltime_seconds']})
 for artifact,p in r['artifacts'].items():allartifactpins[p['path']]={**p,'kind':artifact,'receipt_id':ident}
check(len(fit_manifest['reports'])==69 and len(folds)==40 and all(sorted(fs)==[0,1,2,3] for fs in folds.values()),'69_fits_40_cells_160_folds_once')
check(len(numeric)==1,'one_new_classification_numeric_hash',list(numeric))
check(set(fit_pose['fits'])=={'19618','39237','49046'},'three_pose_doses')
for step,v in fit_pose['fits'].items():
 r=receipt_for(v['report'],'pose_fit:'+step);check(r['checkpoint_step']==int(step) and r['benchmark_receipt']==plan['benchmark_receipts']['pose'] and r['cache_manifest']==plan['cache_manifest'] and r['held_payload_opened_after_both_ridge_fits'] is True,'pose_fit_input_order:'+step)
 for artifact,p in r['artifacts'].items():allartifactpins[p['path']]={**p,'kind':artifact,'receipt_id':remote_to_id[v['report']['path']]}
# The two comparison receipts are embedded verbatim in the summary as parsed JSON.
s=receipts['LONG-SUMMARY-20261105'];check(s['inputs']==summary_plan['final_reports'],'summary_final_input_pins')
for name,pin in summary_plan['final_reports'].items():
 r=receipt_for(pin,'comparison:'+name);check(s['reports'][name]==r,'summary_embedded_receipt:'+name)
 raw=(json.dumps(s['reports'][name],indent=2,sort_keys=True,allow_nan=False)+'\n').encode();check(sha(raw)==pin['sha256'] and len(raw)==pin['bytes'],'summary_embedded_receipt_exact_bytes:'+name)
class_pin=s['reports']['classification']['artifacts']['comparison'];check(s['classification_result']['complete'] is True and s['classification_result']['kind']=='comparison' and s['classification_result']['code_sha']==phase_shas['aggregate'] and s['classification_result']['plan_sha256']==phase_plan_hash['aggregate'],'summary_embedded_classification_result_identity')
# Its raw original JSON key order was lost on sorted serialization of summary.
class_original_reconstructed=False
for sort_keys in (False,True):
 raw=(json.dumps(s['classification_result'],indent=2,sort_keys=sort_keys,allow_nan=False)+'\n').encode()
 class_original_reconstructed |= sha(raw)==class_pin['sha256']
# Decode wrapper only; never execute it.
wrapper=(E/'summary_materialize.sbatch').read_text();exports={}
for line in wrapper.splitlines():
 if line.startswith('export '):
  word=shlex.split(line)[1];key,value=word.split('=',1);exports[key]=value
raw=base64.b64decode(exports['LONG_PLAN_B64'],validate=True)
check(raw==(E/'summary_plan.json').read_bytes(),'summary_wrapper_payload_exact_final_plan')
check(sha(raw)==exports['LONG_PLAN_SHA256']==s['plan_sha256'] and len(raw)==int(exports['LONG_PLAN_BYTES']),'summary_wrapper_payload_sha_bytes')
check(exports['LONG_SUMMARY_PLAN']==plan['output_root']+'/manifests/summary_plan_'+sha(raw)+'.json','summary_external_plan_path')
check(states['LONG-SUMMARY-20261105']['script_stdin_sha256']==sha(wrapper.encode()),'summary_wrapper_ledger_hash')
last=shlex.split(wrapper.strip().splitlines()[-1]);check(last[0:2]==['exec','bash'] and '--expected-code' in last and last[last.index('--expected-code')+1]==phase_shas['aggregate'] and last[last.index('--plan')+1]==exports['LONG_SUMMARY_PLAN'],'summary_fourth_checkout_exec_binding')
check('--account=IscrB_WearUsFM' in states['LONG-SUMMARY-20261105']['sbatch'],'summary_stdin_account_explicit')
# Phase ordering from archived sacct, without querying live scheduler.
phase_ranges={}
for phase in phase_shas:
 ids=[ident for ident,row in jobrows.items() if row['phase']==phase]
 phase_ranges[phase]={'first_start':min(rows[i]['Start'] for i in ids),'last_end':max(rows[i]['End'] for i in ids)}
check(all(phase_ranges[a]['last_end']<=phase_ranges[b]['first_start'] for a,b in zip(('extract','benchmark','fit'),('benchmark','fit','aggregate'))),'sequential_phase_boundaries',phase_ranges)
check(max(rows[i]['End'] for i in rows if i.startswith('LONG-COMPARE-'))<=rows['LONG-SUMMARY-20261105']['Start'],'summary_after_two_compares')
pose_ids=[i for i in jobrows if jobrows[i]['stage']=='fit-pose'];class_ids=[i for i in jobrows if jobrows[i]['stage']=='fit-classification']
check(max(states[i]['creato'] for i in pose_ids)<min(states[i]['creato'] for i in class_ids),'three_pose_submissions_before_classification')
# Every artifact remains a pinned remote payload; no fresh remote verification made.
limits=['Review uses published local receipts, accounting exports, logs and Git objects, plus read-only current local ledger copies. No current remote state was queried.',
'Original cache/OOF/probe/audit/prediction payload bytes were not found in the examined local line2 collection and were not downloaded or independently rehashed. Runtime receipts and comparator code attest their remote verification at execution.',
'Classification comparison JSON is fully embedded in summary, but its original unsorted key order was lost; cannot reproduce its original raw file hash from sorted embedding alone. Both embedded comparison receipt byte hashes are independently reproduced.',
'No inference, refitting, held scoring or scientific CI recomputation performed in this provenance-only audit.']
if class_original_reconstructed:limits=[x for x in limits if not x.startswith('Classification comparison')]
check(not git('status','--porcelain').strip(),'clean_review_worktree_after')
result={'verdict':'PASS_LOCAL_PROVENANCE_WITH_PAYLOAD_LIMITS','delivery_commit':COMMIT,'checks_passed':len(checks),'checks':checks,'phase_commits':phase_shas,'phase_plan_sha256':phase_plan_hash,'phase_source_sha256':source_hashes,'source_file_counts':{k:len(v) for k,v in maps.items()},'amendment':{'closed_original_receipts':28,'manifest_pin':local_pin(CAMP/'amendment_a1/input_compatibility.json'),'changed_source_paths':changed,'added_source_paths':added,'scientific_files_byte_identical':len(scientific)},'counts':{'allocations':107,'checkout':4,'extract':12,'benchmark':16,'classification_fit':69,'pose_fit':3,'compare':2,'summary':1,'complete_receipts':103,'current_local_ledger_matches':live_state_equal,'single_submission_attempts':107,'cache_pins':21},'accounting':{'by_phase':accounting,'gpu_hours':sum(x['gpu_hours'] for x in accounting.values()),'allocated_core_hours':sum(x['allocated_core_hours'] for x in accounting.values()),'approved_caps':launch['caps'],'fit_max_walltime_fraction':max(fit_ratios,key=lambda x:x['fraction'])},'phase_ranges':phase_ranges,'summary':{'summary_receipt_pin':local_pin(E/'receipts/LONG-SUMMARY-20261105.json'),'external_plan_pin':local_pin(E/'summary_plan.json'),'wrapper_pin':local_pin(E/'summary_materialize.sbatch'),'class_comparison_original_pin':class_pin,'class_comparison_raw_hash_reproduced':class_original_reconstructed,'class_comparison_embedded_full_content':True,'embedded_pose_and_class_receipt_raw_hashes_reproduced':True},'remote_payload_artifact_pins':allartifactpins,'limits':limits}
(OUT/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
(OUT/'README.md').write_text('# Local provenance audit of long evaluation\n\nPASS with payload limitations. Delivery `'+COMMIT+'`.\n\n'+f"{len(checks)} checks passed; 107 allocations, 103 complete receipts, one attempt each; 4 frozen checkouts. Cost: {result['accounting']['gpu_hours']:.9f} GPUh and {result['accounting']['allocated_core_hours']:.9f} allocated coreh.\n\n"+'Reproduce: `python3 audit.py --snapshot-root /path/to/c109bfe/worktree --out /path/outside/repo [--ledger ~/wearusfm_local/jobs]`.\n\n'+('\n'.join('- '+x for x in limits))+'\n')
print(json.dumps({k:result[k] for k in ('verdict','checks_passed','counts','phase_source_sha256','limits')},indent=2))
print('ACCOUNTING',result['accounting']['gpu_hours'],result['accounting']['allocated_core_hours'],'maxfit',result['accounting']['fit_max_walltime_fraction'])
