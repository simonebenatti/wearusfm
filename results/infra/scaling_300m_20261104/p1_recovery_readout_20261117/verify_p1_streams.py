"""Stream each P1 cache read-only to the Mac and SHA256 locally; retain no large copy.

The remote side performs only ordinary file transfer and descriptor/path stat,
not numerical computation or hashing on the login node. No writes to remote files.
"""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess
import tempfile
import time

BASE=Path(__file__).resolve().parent
EVIDENCE=BASE/'evidence'
REMOTE=r'''
import os,sys,json,stat
path=PATH
fields=('dev','ino','size','mode','uid','gid','nlink','atime_ns','mtime_ns','ctime_ns')
def record(s):return {k:int(getattr(s,'st_'+k)) for k in fields}
def emit(k,v):sys.stderr.write('P1_TRANSFER '+json.dumps(dict(event=k,value=v))+'\n');sys.stderr.flush()
assert os.path.realpath(path)==path
before=record(os.lstat(path));assert stat.S_ISREG(before['mode'])
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
try:
 first=record(os.fstat(fd));emit('before',dict(path=before,descriptor=first))
 count=0
 while True:
  b=os.read(fd,1024*1024)
  if not b:break
  sys.stdout.buffer.write(b);count+=len(b)
 sys.stdout.buffer.flush()
 emit('after',dict(path=record(os.lstat(path)),descriptor=record(os.fstat(fd)),bytes=count))
finally:os.close(fd)
'''

def main():
    target=EVIDENCE/'payload_stream_verification.json'
    if target.exists():raise SystemExit('verification already recorded; no silent overwrite')
    snapshot=json.loads((EVIDENCE/'snapshot.json').read_text())
    receipt=json.loads((EVIDENCE/'p1/receipt.json').read_text())
    result=dict(protocol='scaling-p1-full-payload-readback-20261117-v1',hash_computed_on='Mac',
                remote_operation='read-only file transfer plus descriptor/path stat',cache_copies_retained=False,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=False,records=[])
    def write():target.write_text(json.dumps(result,indent=2)+'\n')
    for ds,out in receipt['outputs'].items():
        start=time.monotonic();h=hashlib.sha256();count=0
        args=['ssh','leonardo','-o','ProxyCommand=none','-o','ControlMaster=no','-o','ControlPath=none',
              '-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
              'python3 -c '+shlex.quote(REMOTE.replace('PATH',repr(out['path'])))]
        print('Reading '+ds+' ('+str(out['bytes'])+' bytes)',flush=True)
        with tempfile.TemporaryFile() as stderr:
            p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=stderr)
            while True:
                block=p.stdout.read(1024*1024)
                if not block:break
                h.update(block);count+=len(block)
            code=p.wait();stderr.seek(0);error=stderr.read().decode()
        observations=[json.loads(x[len('P1_TRANSFER '):]) for x in error.splitlines() if x.startswith('P1_TRANSFER ')]
        rec=dict(dataset=ds,path=out['path'],bytes=count,sha256=h.hexdigest(),exit=code,
                 elapsed_seconds=time.monotonic()-start,observations=observations,stderr=error,accepted=False)
        result['records'].append(rec)
        try:
            assert code==0,error
            assert [v['event'] for v in observations]==['before','after']
            assert count==out['bytes']==observations[1]['value']['bytes']
            assert h.hexdigest()==out['sha256']
            states=[observations[0]['value'][k] for k in ('path','descriptor')]+[observations[1]['value'][k] for k in ('path','descriptor')]
            historical=out['output_identity_audit']['observations'][-1]['stat_after']
            current=snapshot['cache_states'][out['path']]
            for s in states:
                for k in historical:
                    if k!='atime_ns':assert s[k]==historical[k]==current[k],(ds,k,s[k],historical[k],current[k])
            rec['accepted']=True
        except Exception as exc:
            rec['error']=str(exc);write();raise
        write();print(ds+' SHA256 and unchanged path/descriptor: PASS',flush=True)
    result['complete']=True
    result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['total_bytes']=sum(v['bytes'] for v in result['records'])
    write();print('All three P1 payloads verified.',flush=True)

if __name__=='__main__':main()
