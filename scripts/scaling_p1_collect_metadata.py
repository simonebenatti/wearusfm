"""Bounded read-only collection for the approved P1 identity diagnostic (Mac).

Only metadata, small receipts/logical submission records and frozen source hashes;
large payload hashes are verified on the diagnostic compute node, not the login.
No deployment, submission, repair, or score inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'results/infra/scaling_300m_20261104/p1_recovery_20261116'
REMOTE = r'''
import datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path('/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/T100-G-300M')
code=Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_300m_evaluation19618_20261106_v1')
control=Path('/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/tokenizer_evaluation_20261103/19618/T100-G/p1')
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'commands':{},'sources':{}}
def cmd(key,args):
 p=subprocess.run(args,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
 result['commands'][key]={'argv':args,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 if p.returncode: raise RuntimeError(key+': '+p.stderr)
for key,args in {
 'sacct':['sacct','-n','-P','-j','64571999,64572000,64572001,64596915','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,Start,End,Timelimit'],
 'squeue':['squeue','-u','sbenatti','-h','-o','%i|%j|%T|%q|%R'],
 'config':['scontrol','show','config'],
 'code_head':['git','-C',str(code),'rev-parse','HEAD'],
 'code_status':['git','-C',str(code),'status','--porcelain','--untracked-files=all'],
 'deploy_hooks':['git','-C','/leonardo_work/IscrB_WearUsFM/wearusfm','config','--list'],
 'work_quota':['lfs','quota','-hp','20147512','/leonardo_work/IscrB_WearUsFM'],
}.items(): cmd(key,args)
paths=[root/'p1/ninapro_db6.npz',root/'p1-r1/ninapro_db3.npz',root/'p1-r1/ninapro_db2.npz',root/'pose/val.npz',root/'p2/epn612.npz',control/'ninapro_db3.npz']
small=[root/'pose/receipt.json',root/'p2/receipt.json',control/'receipt.json']
for p in paths:
 q=Path(str(p)+'.provenance.json')
 if q.exists(): small.append(q)
for p in paths+small:
 s=p.stat(); row={'path':str(p),'canonical':str(p.resolve()),'bytes':s.st_size,'inode':s.st_ino,'dev':s.st_dev,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'symlink':p.is_symlink()}
 if p in small:
  if s.st_size>16000000: raise ValueError('unexpected large metadata')
  data=p.read_bytes(); row.update(sha256=hashlib.sha256(data).hexdigest(),text=data.decode())
 result['files'][str(p)]=row
for folder in ('scripts','src'):
 for p in sorted((code/folder).rglob('*.py')):
  result['sources'][str(p.relative_to(code))]=hashlib.sha256(p.read_bytes()).hexdigest()
result['hooks']={}
for name in ('post-receive','post-update','update','pre-receive'):
 p=Path('/leonardo_work/IscrB_WearUsFM/wearusfm/.git/hooks')/name
 if p.exists(): result['hooks'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps(result,sort_keys=True))
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    if out.exists(): raise SystemExit('new output required')
    result=subprocess.run(['ssh','leonardo','-o','ConnectTimeout=15','-o','ProxyCommand=none',
                           '-o','ControlMaster=no','-o','ControlPath=none','python3 -c '+shlex.quote(REMOTE)],
                          capture_output=True,text=True,timeout=180)
    if result.returncode: raise SystemExit(result.stderr)
    evidence=json.loads(result.stdout)
    out.mkdir(parents=True)
    (out/'remote_metadata.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (out/'ssh.stderr.txt').write_text(result.stderr)
    for key in ('sacct','squeue','config','work_quota'):
        (out/(key+'.txt')).write_text(evidence['commands'][key]['stdout'])
    for path,row in evidence['files'].items():
        if 'text' in row:
            short=hashlib.sha256(path.encode()).hexdigest()[:12]+'_'+Path(path).name
            (out/short).write_text(row['text'])
    print(json.dumps({'out':str(out),'files':len(evidence['files']),'sources':len(evidence['sources'])}))

if __name__=='__main__':main()
