"""Collect bounded evidence and cache metadata read-only; no jobs or payload hashes on login."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

BASE=Path(__file__).resolve().parent
OUT=BASE/'evidence'
REMOTE=r'''
from pathlib import Path
import datetime,hashlib,json,os,subprocess
root=Path('/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/T100-G-300M')
code=Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_p1_recovery_20261116_v1')
old=Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_300m_evaluation19618_20261106_v1')
result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands={},files={},cache_states={},tasks={},sources={},directories={})
def state(p):
 s=p.lstat();return dict(path=str(p),canonical=str(p.resolve()),symlink=p.is_symlink(),**{k:int(getattr(s,'st_'+k)) for k in ('dev','ino','size','mode','uid','gid','nlink','atime_ns','mtime_ns','ctime_ns')})
def small(key,p):
 before=state(p)
 if before['size']>32*1024**2:raise RuntimeError('unexpected large metadata: '+str(p))
 b=p.read_bytes();after=state(p)
 for k in before:
  if k!='atime_ns' and before[k]!=after[k]:raise RuntimeError('metadata changed: '+str(p))
 result['files'][key]=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),text=b.decode(),before=before,after=after)
 return json.loads(b) if p.suffix=='.json' else None
commands={
 'sacct':['sacct','-n','-P','-j','64571999,64572000,64572001,64596915,64610004,64610181,64620427,64620434','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,MaxRSS,NodeList,Start,End,Timelimit'],
 'squeue':['squeue','-u','sbenatti','-h','-o','%i|%j|%T|%q|%R'],
 'code_head':['git','-C',str(code),'rev-parse','HEAD'],
 'code_status':['git','-C',str(code),'status','--porcelain','--untracked-files=all'],
 'old_code_head':['git','-C',str(old),'rev-parse','HEAD'],
 'old_code_status':['git','-C',str(old),'status','--porcelain','--untracked-files=all']}
for k,args in commands.items():
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
 result['commands'][k]=dict(argv=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
 if p.returncode:raise RuntimeError(k+': '+p.stderr)
for task,folder in [('p1','p1-r2'),('pose','pose'),('p2','p2')]:
 directory=root/folder;result['directories'][task]=sorted(p.name for p in directory.iterdir())
 receipt=small(task+'/receipt.json',directory/'receipt.json')
 result['tasks'][task]=dict(receipt_key=task+'/receipt.json',outputs={})
 for name,out in receipt['outputs'].items():
  pp=Path(out['provenance_path']);key=task+'/'+pp.name
  small(key,pp);result['tasks'][task]['outputs'][name]=dict(provenance_key=key,cache_path=out['path'])
  result['cache_states'][out['path']]=state(Path(out['path']))
for name,p in [('p1/output_identity_events.jsonl',root/'p1-r2/output_identity_events.jsonl'),('p1/p1.log',root/'p1-r2/p1.log'),
 ('p1/slurm.log',Path('/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_p1_recovery_64620434.out')),
 ('checkout/slurm.log',Path('/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_64620427.out')),
 ('release/recovery_manifest.json',code/'results/infra/scaling_300m_20261104/p1_recovery_release_20261116/recovery_manifest.json')]:small(name,p)
manifest=json.loads(result['files']['release/recovery_manifest.json']['text'])
plan=small('release/extraction_plan.json',Path(manifest['extraction_plan']['path']))
checkpoint=Path(plan['checkpoints']['T100-G-300M']['path']);result['checkpoint_state']=state(checkpoint)
result['checkpoint_stop_exists']=(checkpoint.parent/'STOP').exists()
receipt=json.loads(result['files']['p1/receipt.json']['text'])
contract=Path(receipt['source_states']['training_source_contract']['path'])
contract_obj=small('release/training_source_contract.json',contract)
for folder in ('scripts','src'):
 for p in sorted((code/folder).rglob('*.py')):
  result['sources'][str(p.relative_to(code))]=hashlib.sha256(p.read_bytes()).hexdigest()
for folder in ('p1','p1-r1'):
 result['directories'][folder+'_failed']=sorted(p.name for p in (root/folder).iterdir())
print(json.dumps(result))
'''

def main():
    if OUT.exists():raise SystemExit('new evidence directory required')
    p=subprocess.run(['ssh','leonardo','-o','ProxyCommand=none','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15','python3 -c '+shlex.quote(REMOTE)],capture_output=True,text=True,timeout=180)
    if p.returncode:raise SystemExit(p.stderr)
    snapshot=json.loads(p.stdout);OUT.mkdir(parents=True)
    (OUT/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    (OUT/'ssh.stderr.txt').write_text(p.stderr)
    for key,record in snapshot['commands'].items():(OUT/(key+'.txt')).write_text(record['stdout'])
    for key,record in snapshot['files'].items():
        data=record['text'].encode();assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
        path=OUT/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print(json.dumps(dict(evidence=str(OUT),cache_count=len(snapshot['cache_states']),small_files=len(snapshot['files']),p1_bytes=sum(snapshot['cache_states'][v['cache_path']]['size'] for v in snapshot['tasks']['p1']['outputs'].values())),indent=2))

if __name__=='__main__':main()
