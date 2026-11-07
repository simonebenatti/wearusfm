#!/usr/bin/env python3
"""Local, read-only verification of the archived preservation manifests.
No remote queries or payload reads. Git refs are pinned below; --repo may be any
local clone containing them. Writes only --out, which must be outside --repo.
"""
import argparse,hashlib,json,pathlib,re,subprocess
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--ledger',type=pathlib.Path,default=pathlib.Path.home()/'wearusfm_local/jobs');a=p.parse_args()
ROOT=a.repo.resolve();OUT=a.out.resolve();assert ROOT not in OUT.parents and OUT!=ROOT;OUT.mkdir(parents=True,exist_ok=True)
DELIVERY='424f0ecc5dc23120ac84a7f2e3e1a24fede2b250';SCRIPT='5b9b43c8205731deb004543e08c848e447d6cad3';INVENTORY='57c4d7e';BASE='results/infra/consolidation_long_20261105';PRES=BASE+'/preservation';TAG='wearusfm_runs_results_consolidation_long_20261105';SRC='/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105';DST='/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105'
checks=[]
def check(ok,name):
 checks.append({'check':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def get(ref,path):return git('show',ref+':'+path)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def pin(raw,path):return {'git_path':path,'sha256':sha(raw),'bytes':len(raw)}
initial_status=git('status','--porcelain').decode()  # Parent may edit its own worktree; audited inputs come from pinned Git objects.
raw={name:get(DELIVERY,PRES+'/'+TAG+'.'+name) for name in ('source.sizes','copy.sizes','source.dirs','copy.dirs','source.sha256','copy.sha256','receipt.txt')}
receipt=raw['receipt.txt'].decode();log=get(DELIVERY,PRES+'/preserve_long_to_work_63781257.log').decode();operator=json.loads(get(DELIVERY,PRES+'/inventory_check.json'))
for typ in ('sizes','dirs','sha256'):check(raw['source.'+typ]==raw['copy.'+typ],'source_copy_'+typ+'_byte_equal')
manifest_receipt_hashes={name:h for h,name in re.findall(r'^([a-f0-9]{64})  (\S+)$',receipt,re.M)}
check(set(manifest_receipt_hashes)=={TAG+'.'+k for k in raw if k!='receipt.txt'},'receipt_hashes_exact_six_manifests')
for name,h in manifest_receipt_hashes.items():check(sha(raw[name[len(TAG)+1:]])==h,'manifest_hash:'+name)
def normalize(path):
 check(path=='.' or path.startswith('./'),'relative_manifest_path:'+path)
 rel=path[2:] if path.startswith('./') else ''
 check(not pathlib.PurePosixPath(rel).is_absolute() and '..' not in pathlib.PurePosixPath(rel).parts,'safe_relative_manifest_path:'+path)
 return rel
sizes={};hashes={};dirs=[]
for line in raw['copy.sizes'].decode().splitlines():
 size,path=line.split(' ',1);name=normalize(path);check(name not in sizes,'unique_size_path:'+name);sizes[name]=int(size)
for line in raw['copy.sha256'].decode().splitlines():
 h,path=line.split('  ',1);check(bool(re.fullmatch('[a-f0-9]{64}',h)),'valid_sha:'+path);name=normalize(path);check(name not in hashes,'unique_hash_path:'+name);hashes[name]=h
for line in raw['copy.dirs'].decode().splitlines():dirs.append(normalize(line))
check(set(sizes)==set(hashes) and len(sizes)==354,'354_file_paths_sizes_hashes')
check(len(dirs)==len(set(dirs))==174 and '' in dirs,'174_directories_including_root')
check(sum(sizes.values())==21062138097,'total_bytes_21062138097')
check(all(('' if str(pathlib.PurePosixPath(f).parent)=='.' else str(pathlib.PurePosixPath(f).parent)) in dirs for f in sizes),'all_file_parent_directories_listed')
check(SRC in receipt and DST in receipt and SCRIPT in receipt and 'job: 63781257' in receipt,'receipt_expected_roots_code_job')
check('file: 354; byte: 21062138097; sha256 identici' in receipt and 'riuscite=1 problemi=0 su 1 voci' in log,'receipt_and_log_completion_claim')
# Inventory coverage uses frozen inventoried pins, not the operator's inventory_check.
inventory_raw=get(INVENTORY,BASE+'/gpt_review_20261107/preservation_inventory.json');inventory=json.loads(inventory_raw);wanted={};groups={}
for group,entries in inventory['groups'].items():
 count=bytecount=0
 for ent in entries:
  check(ent['path'].startswith(SRC+'/'),'inventory_under_original_root');rel=ent['path'][len(SRC)+1:];check(rel not in wanted,'inventory_no_duplicate:'+rel);wanted[rel]=ent
  check(rel in sizes and sizes[rel]==ent['bytes'] and hashes[rel]==ent['sha256'],'inventoried_file_preserved:'+rel);count+=1;bytecount+=ent['bytes']
 groups[group]={'files':count,'bytes':bytecount};check(count==inventory['summary'][group]['files'] and bytecount==inventory['summary'][group]['bytes'],'inventory_group_totals:'+group)
check(len(wanted)==352 and sum(x['bytes'] for x in wanted.values())==21059236232,'352_original_inventory_count_and_bytes')
extras=sorted(set(sizes)-set(wanted));expected_extra=['manifests/summary_plan_f7728778681431314849ff6f21ba55695ebab799d74f806347faf9eba3286a50.json','summary/report.json'];check(extras==expected_extra,'only_two_pinned_summary_extras')
extra_sources={expected_extra[0]:BASE+'/execution/summary_plan.json',expected_extra[1]:BASE+'/execution/receipts/LONG-SUMMARY-20261105.json'}
extra_details=[]
for rel,path in extra_sources.items():
 content=get('c109bfef5a3f80cc4b2334e3554c2171db938467',path);check(sizes[rel]==len(content) and hashes[rel]==sha(content),'extra_matches_frozen_git_bytes:'+rel)
 extra_details.append({'path':rel,'bytes':len(content),'sha256':sha(content),'git_reference_path':path,'git_reference_commit':'c109bfef5a3f80cc4b2334e3554c2171db938467'})
check(operator['receipt_manifest_hashes_match'] is True and operator['source_equals_copy']=={'sizes':True,'dirs':True,'sha256':True} and operator['copy_files']==len(sizes) and operator['copy_bytes']==sum(sizes.values()) and operator['copy_dirs']==len(dirs) and operator['inventory_files_missing_in_copy']==[] and operator['inventory_files_mismatching']==[] and operator['copy_files_not_in_inventory']==[{k:v for k,v in e.items() if k not in ('git_reference_path','git_reference_commit')} for e in extra_details],'operator_inventory_check_independently_reproduced')
# Submitted script is bound by its exact stdin hash, not merely EXPECTED_SHA in the text receipt.
script=get(SCRIPT,'scripts/slurm/preserve_long_to_work.sbatch');state_path=a.ledger.expanduser()/'LONG-PRESERVE-WORK-20261107/state.json';state_raw=state_path.read_bytes();state=json.loads(state_raw)
check(state['script_stdin_sha256']==sha(script) and state['codice_sha']==SCRIPT,'submitted_stdin_equals_pinned_script')
check(state['job_slurm']==63781257 and state['stato']=='sottomesso' and state['agente']=='claude' and state['campagna']=='consolidation_long_20261105' and [r['stato'] for r in state['storia']]==['prenotato','in_sottomissione','sottomesso'],'local_ledger_single_submission')
check('--no-requeue' in state['sbatch'] and '--cpus-per-task=1' in state['sbatch'] and '--mem=2G' in state['sbatch'] and '--time=00:40:00' in state['sbatch'] and '--export=ALL,PRESERVE_MODE=copy,EXPECTED_SHA='+SCRIPT in state['sbatch'],'submission_resources_and_copy_mode')
syntax=subprocess.run(['bash','-n'],input=script,stdout=subprocess.PIPE,stderr=subprocess.PIPE);check(syntax.returncode==0,'bash_syntax')
text=script.decode();check('cp -r --preserve=mode,timestamps -- "$src" "$dst"' in text,'copy_requests_mode_and_timestamps')
# Archived quota values: actual after-copy remainder, independently from projected bytes.
quotas=re.findall(r'^\s+(\d+)\s+(1073741824)\s+0\s+-',log,re.M);check(len(quotas)==2,'two_logged_quota_snapshots')
before_used,limit=map(int,quotas[0]);after_used,after_limit=map(int,quotas[1]);threshold=11244930;after_free=after_limit-after_used
check(after_free>=threshold,'logged_after_copy_project_quota_above_C1')
final_status=git('status','--porcelain').decode()  # No command in this script writes to ROOT.
limits=[
 'No current remote existence, disk contents, permissions, ownership, timestamps, symlinks, quota or scheduler state was queried. Findings concern immutable Git evidence and the local submitter ledger.',
 'File names, byte sizes, directory tree and every file SHA match between archived source and destination manifests. Payload bytes were not independently read or hashed in this review.',
 'The script requests preservation of mode and timestamps, but no stat metadata manifests are archived; independent equality of modes/mtime/atime is not proved. Reading files during hashing may update access times.',
 'The script has no explicit source writes, renames or removal. Original intact means copied content retained, not a forensic guarantee that all source filesystem metadata was unchanged.',
 'EXPECTED_SHA is logged by this standalone stdin script, not verified with a Git checkout. The source binding here comes from the local ledger script_stdin_sha256 matching the pinned script bytes.',
 'COMPLETED 0:0, ElapsedRaw=156s and AllocCPUS=2 were reported by the operator/root; no sacct export is included among these preservation artifacts. CPU cost below is arithmetic conditional on those reported values, not an independent scheduler verification.',
 'WORK destination avoids the documented SCRATCH 40-day purge policy; no claim of permanent protection from deletion or future project retention changes is made.'
]
result={'verdict':'PASS_ARCHIVED_PRESERVATION_CONTENT_EVIDENCE','delivery_commit':DELIVERY,'script_commit':SCRIPT,'inventory_commit':git('rev-parse',INVENTORY).decode().strip(),'check_count':len(checks),'checks':checks,'preservation':{'source':SRC,'destination':DST,'files':len(sizes),'directories_including_root':len(dirs),'bytes':sum(sizes.values()),'GiB':sum(sizes.values())/1024**3,'source_copy_manifests_byte_identical':True,'original_inventory_files':len(wanted),'original_inventory_bytes':sum(x['bytes'] for x in wanted.values()),'groups':groups,'two_extra_files':extra_details},'archived_manifest_pins':{name:pin(value,PRES+'/'+TAG+'.'+name) for name,value in raw.items()},'other_evidence_pins':{'log':pin(log.encode(),PRES+'/preserve_long_to_work_63781257.log'),'inventory':pin(inventory_raw,BASE+'/gpt_review_20261107/preservation_inventory.json'),'script':pin(script,'scripts/slurm/preserve_long_to_work.sbatch'),'local_submitter_state':{'sha256':sha(state_raw),'bytes':len(state_raw),'logical_id':state['id']}},'quota_from_archived_log':{'unit':'KiB','before_used':before_used,'after_used':after_used,'limit':limit,'after_free':after_free,'C1_minimum':threshold,'after_headroom_over_C1':after_free-threshold,'claim_scope':'historical log values only'},'accounting':{'job':63781257,'scheduler_reported_state':'COMPLETED 0:0','elapsed_seconds_operator_reported':156,'cpus_requested_verified_in_ledger':1,'cpus_allocated_operator_reported':2,'gpu_hours':0,'allocated_core_hours_conditional_on_reported_elapsed_and_cpus':156*2/3600,'requested_walltime_seconds':2400,'allocation_cap_core_hours_if2CPUs':2400*2/3600,'sacct_independently_read':False},'limits':limits}
(OUT/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
(OUT/'README.md').write_text('# Review locale della conservazione WearUsFM\n\n**Esito: manifest e copertura coerenti, nessuna discrepanza trovata.**\n\n'+f"Consegna `{DELIVERY}`, script `{SCRIPT}`. {len(sizes)} file, {len(dirs)} directory (radice inclusa), {sum(sizes.values()):,} byte. Le tre coppie di manifest sorgente/copia sono identiche; i sei hash coincidono con la ricevuta. Tutti i 352 file dell’inventario GPT sono presenti con SHA e byte uguali; i soli due aggiuntivi sono il piano esterno e il report della sintesi, verificati sui rispettivi byte Git congelati.\n\n"+'Il ledger locale lega l’unico invio allo script esatto tramite il suo hash stdin. `cp` richiede conservazione di modi e date, ma tali metadati non sono confrontati nei manifest: la verifica indipendente riguarda contenuti e struttura. Nessuna lettura remota eseguita.\n\n'+'Il costo 0,086667 coreh deriva da 156 s × 2 CPU allocate, valori riportati dall’operatore; il solo ledger conferma 1 CPU richiesta. Nessun export sacct è incluso nella consegna di conservazione.\n\n'+'Riprodurre con `python3 audit.py --repo <repo-con-gli-oggetti-Git> --out <directory-esterna> [--ledger <ledger-locale>]`.\n\n'+'Limiti dettagliati in `audit.json`.\n')
print(json.dumps({k:result[k] for k in ('verdict','check_count','preservation','quota_from_archived_log','accounting')},indent=2))
