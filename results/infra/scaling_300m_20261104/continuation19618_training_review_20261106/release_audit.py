import hashlib,json,re,shlex,subprocess
from pathlib import Path
from datetime import datetime,timedelta
ROOT=Path('/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-scaling-exec')
BASE=ROOT/'results/infra/scaling_300m_20261104/continuation19618_20261106'
DEPLOY='73524b63be534f81b3db4ade4b7fc7f47129a0f3'
TIP='92320d39646fd06c7b3b9954a6ca5b7e48455c86'
checks=[]
def ck(value,label):
    assert value,label
    checks.append(label)
def sha(b): return hashlib.sha256(b).hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)
plan=json.loads((BASE/'training_plan.json').read_text())
v=json.loads((BASE/'remote_verification.json').read_text())
rawhash=(BASE/'remote_sha256.txt').read_text(); rawlisting=(BASE/'remote_listing.txt').read_text()
ck(git('rev-parse','HEAD').decode().strip()==TIP,'tip')
ck(not git('status','--porcelain').strip(),'clean_worktree')
ck(sha((BASE/'training_plan.json').read_bytes())=='eded0922bd1022c417b8d70523c0c66d1ddb1bd8ad273925ec6d53fda5957d5f','plan_hash')
for name in ['training_plan.json','remote_verification.json','chain_plan.md','endpoint19618_reading.md']:
    ck((BASE/name).read_bytes()==git('show',DEPLOY+':'+str((BASE/name).relative_to(ROOT))),name+'_matches_deploy')
for rel,digest in {**plan['scientific_source_sha256'],**plan['execution_source_sha256']}.items():
    ck(sha((ROOT/rel).read_bytes())==digest and sha(git('show',DEPLOY+':'+rel))==digest,'source:'+rel)
for key in ['budget_freeze','source_contract']:
    pin=plan[key]; b=(ROOT/pin['path']).read_bytes()
    ck(sha(b)==pin['sha256'] and len(b)==pin['bytes'],key)
freeze=json.loads((ROOT/plan['budget_freeze']['path']).read_text())
ck(plan['allocation']['walltime_seconds']==53100 and 4*53100/3600==freeze['training']['gpu_hours_cap']==59,'train_budget')
ck(freeze['extractions']['gpu_hours_cap']==7 and freeze['readers']['cpu_only_core_hours_cap']==350,'remaining_budget')
previous=plan['previous']; remote=v['directories'][previous['run_dir']]
ck(len(previous['files'])==len(remote)==11,'11_previous_files')
for name,pin in previous['files'].items():
    ck(pin['sha256']==remote[name]['sha256'] and pin['bytes']==remote[name]['bytes'],'previous:'+name)
    ck(re.search(re.escape(pin['sha256'])+r'  '+re.escape(name)+r'\s',rawhash) is not None,'raw_previous:'+name)
ck(previous['files']['checkpoint.pt']==previous['files']['checkpoint_step0005000.pt'],'identical_checkpoint_copies')
for key in ['receipt_copy']:
    pin=previous[key]; b=(ROOT/pin['path']).read_bytes()
    ck(sha(b)==pin['sha256'] and len(b)==pin['bytes'],'previous_'+key)
ck(previous['receipt']['sha256']==previous['receipt_copy']['sha256'],'receipt_identity')
control=plan['control']; inventories=[]
for key in ['receipt','pose_inventory','classification_inventory']:
    pin=control[key]; b=(ROOT/pin['path']).read_bytes()
    ck(sha(b)==pin['sha256'] and len(b)==pin['bytes'],'control_'+key)
    inventories.append(json.loads(b))
def collect_pins(o):
    pins={}
    if isinstance(o,dict):
        if 'path' in o and 'sha256' in o: pins[o['path']]=o['sha256']
        for stem in ('provenance','receipt'):
            if stem+'_path' in o and stem+'_sha256' in o: pins[o[stem+'_path']]=o[stem+'_sha256']
        for x in o.values(): pins.update(collect_pins(x))
    elif isinstance(o,list):
        for x in o: pins.update(collect_pins(x))
    return pins
inventory_pins=collect_pins(inventories)
ck(len(control['artifacts'])==19,'19_control_pins')
for name,pin in control['artifacts'].items():
    q=v['control_caches_and_pose_references'][pin['path']]
    ck(q['sha256']==pin['sha256'] and q['bytes']>0 and pin.get('bytes',q['bytes'])==q['bytes'],'remote_control:'+name)
    ck(inventory_pins.get(pin['path'])==pin['sha256'],'inventory_control:'+name)
    ck(pin['sha256'] in rawhash,'raw_control:'+name)
ck(control['checkpoint']['sha256']==freeze['control19618']['checkpoint_sha256'],'control_checkpoint')
for name in ['fixed_eval_set.json','pose_panel_set.json','origin_sessions.json']:
    ck(remote[name]['sha256']==v['directories'][str(Path(control['checkpoint']['path']).parent)][name]['sha256'],'common_identity:'+name)
mt=[q['mtime_cet'] for q in v['control_caches_and_pose_references'].values()]+[v['directories'][str(Path(control['checkpoint']['path']).parent)]['checkpoint_step0019618.pt']['mtime_cet']]
expiry=min(datetime.strptime(x,'%Y-%m-%d %H:%M') for x in mt)+timedelta(days=40)
ck(expiry.strftime('%Y-%m-%d %H:%M')==v['control_earliest_scratch_expiry_cet']=='2026-12-08 01:27','retention')
ck(expiry>datetime(2026,11,22,1,0),'retention_horizon')
commands=[shlex.split(line) for line in (BASE/'training_submit_lines.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
ck(len(commands)==3,'three_commands')
ck(commands[0]==['git','-C',str(ROOT),'push','leonardo',DEPLOY+':refs/heads/deploy/scaling_300m_continuation19618_20261106_v1'],'deploy_exact_new_ref')
for i,logical in [(1,'SCALING-300M-cont19618-checkout-20261106'),(2,'SCALING-300M-train19618-20261106')]:
    argv=commands[i]; ck(argv[argv.index('--id')+1]==logical,'logical_id_'+str(i))
    ck(argv[argv.index('--agente')+1]=='claude' and argv[argv.index('--campagna')+1]=='scaling-300m-20261104','operator_campaign_'+str(i))
    ck('--dependency' not in ' '.join(argv),'no_dependency_'+str(i))
ck(commands[1][-1]=='/dev/stdin','checkout_stdin')
ck('--cpus-per-task=2' in commands[1] and '--mem=4G' in commands[1] and '--time=00:05:00' in commands[1],'checkout_resources')
ck(commands[1][commands[1].index('--gpu-ore')+1]=='0','checkout_gpu')
ck(commands[2][commands[2].index('--gpu-ore')+1]=='59','training_gpu')
ck(commands[2][-2]==str((BASE/'training_plan.json').relative_to(ROOT)) and commands[2][-1]==plan['receipt_dir'],'training_args')
ck(DEPLOY in commands[1][-2] and DEPLOY in commands[2][-4],'both_expected_sha')
report={'passed':len(checks),'checks':checks,'tip':TIP,'deploy':DEPLOY,'plan_sha256':sha((BASE/'training_plan.json').read_bytes()),'scientific_source_count':len(plan['scientific_source_sha256']),'execution_source_count':len(plan['execution_source_sha256']),'control_artifact_count':len(control['artifacts']),'earliest_retention_cet':str(expiry),'scope':'Published local files, no remote commands, no writes to repository; future live gates not attested','launcher_tests':{'passed':39,'seconds':40.50}}
Path('/private/tmp/wearusfm_continue_release_audit_20261106.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
