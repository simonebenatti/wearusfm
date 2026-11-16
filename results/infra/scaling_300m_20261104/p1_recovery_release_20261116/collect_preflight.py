"""Read-only release snapshot, fixed paths and common-ledger identities, Mac only."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/submit_job.py').is_file())
BASE=Path(__file__).resolve().parent
IDS=['SCALING-P1-recovery-checkout-20261116','SCALING-300M-eval19618-p1-r2-20261116']
REMOTE=r'''
import datetime,hashlib,json,subprocess
from pathlib import Path
work=Path('/leonardo_work/IscrB_WearUsFM')
scratch=Path('/leonardo_scratch/large/userexternal/sbenatti')
old=work/'wearusfm_campaigns/scaling_300m_evaluation19618_20261106_v1'
root=scratch/'wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/T100-G-300M'
out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':{},'new_paths':{},'files':{},'failed_directories':{}}
commands={
 'squeue':['squeue','-u','sbenatti','-h','-o','%i|%j|%T|%q|%R'],
 'sacct':['sacct','-n','-P','-j','64571999,64572000,64572001,64596915,64610004,64610181','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,Start,End,Timelimit'],
 'config':['scontrol','show','config'],
 'quota':['lfs','quota','-hp','20147512',str(work)],
 'old_code_head':['git','-C',str(old),'rev-parse','HEAD'],
 'old_code_status':['git','-C',str(old),'status','--porcelain','--untracked-files=all'],
 'deploy_ref':['git','-C',str(work/'wearusfm'),'for-each-ref','--format=%(refname)|%(objectname)','refs/heads/deploy/scaling_p1_recovery_20261116_v1'],
 'deploy_configuration':['git','-C',str(work/'wearusfm'),'config','--list'],
}
for key,args in commands.items():
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=90)
 out['commands'][key]={'argv':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
for p in [root/'p1-r2',work/'wearusfm_campaigns/scaling_p1_recovery_20261116_v1']:
 out['new_paths'][str(p)]={'exists':p.exists(),'is_symlink':p.is_symlink(),'canonical':str(p.resolve())}
for task in ('p1','p1-r1'):
 out['failed_directories'][task]=sorted(p.name for p in (root/task).iterdir())
small=[root/'pose/receipt.json',root/'p2/receipt.json',scratch/'wearusfm_runs/diag/scaling_p1_identity_20261116_v1/report.json']
checkpoint=scratch/'wearusfm_runs/runs/scaling_300m_20261104/T100-G-300M-19618-from5000-v3/checkpoint_step0019618.pt'
for p in small+[checkpoint]:
 s=p.stat(); item={'path':str(p),'canonical':str(p.resolve()),'bytes':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
 if p in small:
  if s.st_size>16000000:raise ValueError('unexpectedly large metadata')
  data=p.read_bytes();item.update(sha256=hashlib.sha256(data).hexdigest(),text=data.decode())
 out['files'][str(p)]=item
out['hooks']={}
for name in ('pre-receive','post-receive','post-update','update'):
 p=work/'wearusfm/.git/hooks'/name
 if p.exists():out['hooks'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps(out))
'''

def main():
    dest=BASE/'evidence/preflight'
    if dest.exists():raise SystemExit('snapshot already exists: do not overwrite')
    ledger={name:{'unused':not (Path.home()/'wearusfm_local/jobs'/name).exists()} for name in IDS}
    if not all(x['unused'] for x in ledger.values()):raise SystemExit('logical ID already reserved')
    r=subprocess.run(['ssh','leonardo','-o','ProxyCommand=none','-o','ControlMaster=no','-o','ControlPath=none',
                      '-o','ConnectTimeout=15','python3 -c '+shlex.quote(REMOTE)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
    if r.returncode:raise SystemExit(r.stderr)
    value=json.loads(r.stdout);value['local_ledger']=ledger
    dest.mkdir(parents=True)
    (dest/'remote.json').write_text(json.dumps(value,indent=2)+'\n')
    (dest/'ssh.stderr.txt').write_text(r.stderr)
    for name,cmd in value['commands'].items():(dest/(name+'.txt')).write_text(cmd['stdout'])
    checks={
      'all_reads_succeeded':all(type(x['exit']) is int and x['exit']==0 for x in value['commands'].values()),
      'new_paths_absent':all(not v['exists'] and not v['is_symlink'] and v['canonical']==p for p,v in value['new_paths'].items()),
      'new_deploy_ref_absent':value['commands']['deploy_ref']['stdout']=='',
      'old_code_head_exact':value['commands']['old_code_head']['stdout'].strip()=='df2de35c05381e7a5a69edde409171b210a3b5d8',
      'old_code_clean':value['commands']['old_code_status']['stdout']=='',
      'scheduler_margin_180s':'OverTimeLimit           = 3 min' in value['commands']['config']['stdout'],
      'no_failed_receipt_adoption':all('receipt.json' not in x for x in value['failed_directories'].values()),
      'logical_ids_unused':all(x['unused'] for x in ledger.values()),
    }
    old=json.loads((ROOT/'results/infra/scaling_300m_20261104/p1_recovery_20261116/evidence/preparation/remote_metadata.json').read_text())
    for path,item in value['files'].items():
        if path in old['files'] and path.endswith('receipt.json'):
            checks['preserved_'+Path(path).parent.name+'_receipt']=item['sha256']==old['files'][path]['sha256']
        if path.endswith('/diag/scaling_p1_identity_20261116_v1/report.json'):
            checks['diagnostic_pin']=item['sha256']=='c2e32e9cfab36a76c79b05913e350b9ec7f15484f73e2a271552c06f6e980908'
    (dest/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2));print(value['commands']['squeue']['stdout']);print(value['commands']['quota']['stdout'])
    if not all(checks.values()):raise SystemExit('preflight rejected')

if __name__=='__main__':main()
