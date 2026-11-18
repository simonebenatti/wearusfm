"""Read-only acquisition of published receipts and checkpoint byte hashes on the Mac."""
from pathlib import Path
import hashlib,json,subprocess,datetime,sys,shlex
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent/'input_evidence'
SSH=['ssh','leonardo','-o','ProxyCommand=none','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=15']
CANDIDATE='/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/scaling_300m_20261104/T100-G-300M-19618-from5000-v3'
CONTROL='/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_20261104/continuation/checkpoints/checkpoint_postdecay.pt'
PINS=[(CANDIDATE+'/checkpoint_step0019618.pt','d7ebe9d87a08882ae0c1c836b62988ea21c7f1246998a5f82613878a0499387e',4673783740),(CONTROL,'bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490',454381728)]
def acquire():
 OUT.mkdir(parents=True,exist_ok=False)
 result={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hashing_location':'Mac','checkpoints':[]}
 for path,expected,size in PINS:
  digest=hashlib.sha256();n=0
  with subprocess.Popen(SSH+['cat -- '+shlex.quote(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) as child:
   while True:
    block=child.stdout.read(8<<20)
    if not block:break
    digest.update(block);n+=len(block)
   error=child.stderr.read().decode();code=child.wait()
  assert code==0,(path,error)
  assert (digest.hexdigest(),n)==(expected,size),path
  result['checkpoints'].append({'path':path,'sha256':expected,'bytes':n,'verified':True})
 request={'candidate':CANDIDATE,'root':'/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105','files':[]}
 manifest=json.loads((OUT.parent/'control_line2_evidence/cache_manifest.json').read_text())
 for v in manifest['classification']['49046'].values():request['files'] += [v['receipt']['path']]
 for v in manifest['pose']['49046'].values():request['files'] += [v['receipt']['path']]
 request['files']+=['/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105/fits/pose/49046/report.json']
 remote="""import json,pathlib,hashlib,subprocess,sys
q=json.load(sys.stdin);out={'files':{},'candidate_files':{},'commands':{}}
for p in sorted(set(q['files'])):
 x=pathlib.Path(p);b=x.read_bytes();assert len(b)<2000000
 out['files'][p]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'json':json.loads(b)}
for p in pathlib.Path(q['candidate']).iterdir():
 s=p.stat();out['candidate_files'][p.name]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
 if p.suffix in ('.json','.jsonl') and s.st_size<2000000:
  b=p.read_bytes();out['candidate_files'][p.name].update(sha256=hashlib.sha256(b).hexdigest(),text=b.decode())
for k,a in {'quota':['lfs','quota','-hp','IscrB_WearUsFM','/leonardo_work/IscrB_WearUsFM'],'partition':['scontrol','show','partition','boost_usr_prod','-o'],'qos':['sacctmgr','-n','-P','show','qos','where','name=normal','format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU'],'scheduler':['scontrol','show','config']}.items():
 d=subprocess.run(a,capture_output=True,text=True);out['commands'][k]={'exit':d.returncode,'stdout':d.stdout,'stderr':d.stderr}
print(json.dumps(out))"""
 command='python3 -c '+shlex.quote(remote)
 d=subprocess.run(SSH+[command],input=json.dumps(request),capture_output=True,text=True,timeout=180)
 assert d.returncode==0,d.stderr
 result['metadata']=json.loads(d.stdout)
 (OUT/'snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'checkpoints':result['checkpoints'],'output':str(OUT/'snapshot.json')},indent=2))
if __name__=='__main__':acquire()
