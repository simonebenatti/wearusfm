"""Copy three verified native checkpoints into a new persistent archive; never edit sources."""
from __future__ import annotations
import torch  # Preserve the training import order even in this read-only verifier.
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_continue49046 as L
P,M,require=L.P,L.M,L.require


def copy_verified(source,destination,expected,expected_bytes):
    before=P.file_identity(source)
    require(before['sha256']==expected and before['bytes']==expected_bytes,'archive source pin differs')
    with Path(source).open('rb') as incoming,Path(destination).open('xb') as outgoing:
        shutil.copyfileobj(incoming,outgoing,8<<20)
        outgoing.flush();os.fsync(outgoing.fileno())
    actual=P.file_identity(destination)
    require(actual['sha256']==expected and actual['bytes']==expected_bytes,'archive copy differs')
    require(P.file_identity(source)==before,'archive source changed')
    return {k:actual[k] for k in ('path','sha256','bytes')}


def execute(plan_path,plan_sha,code,out):
    P.code_identity(code)
    job=os.environ.get('SLURM_JOB_ID','')
    require(job.isdigit(),'real archive job required')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
    fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
    require(all(fields.get(k)==v for k,v in {'JobId':job,'JobState':'RUNNING','Partition':'lrd_all_serial',
            'NumCPUs':'2','NumNodes':'1','NumTasks':'1','TimeLimit':'01:00:00'}.items()),'archive allocation differs')
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    require(tres.get('mem') in ('30G','30720M') and not any('gpu' in k and v!='0' for k,v in tres.items()),'archive RAM/GPU differs')
    pin=M.resolve_pin({'path':str(plan_path),'sha256':plan_sha})
    plan=json.loads(Path(pin['path']).read_text())
    require(plan['protocol']=='scaling49046-preservation-20261118-v1','wrong archive plan')
    out=Path(out)
    require(str(out)==plan['output'] and out.is_absolute() and not out.exists() and not out.is_symlink(),'new archive required')
    command=subprocess.check_output(['lfs','quota','-p','20147512','/leonardo_work/IscrB_WearUsFM'],text=True)
    words=command.split();i=words.index('/leonardo_work/IscrB_WearUsFM')
    used,limit=map(lambda x:int(x.rstrip('*')),words[i+1:i+3])
    require((limit-used)*1024>=plan['minimum_work_free_bytes']==30*2**30,'archive quota reserve missing')
    require(shutil.disk_usage(out.parent if out.parent.exists() else out.parent.parent).free>=20*2**30,'archive filesystem full')
    inputs={};rows=[]
    for item in plan['checkpoints']:
        if 'receipt' in item:
            identity=P.file_identity(item['receipt']);receipt=json.loads(Path(identity['path']).read_text());inputs[item['receipt']]=identity
            require(receipt.get('complete') is True and receipt.get('protocol')==L.PROTOCOL and
                    receipt.get('checkpoint_step')==item['step'] and receipt.get('checkpoint')==item['source'] and
                    receipt.get('previous_run_unchanged') is True and receipt.get('code_sha')==code and
                    receipt.get('plan_sha256')==item['plan_sha256'],'incomplete native checkpoint receipt')
            native=receipt['run_inventory'][Path(item['source']).name]
            sha=receipt['checkpoint_sha256'];size=native['bytes']
            require(native['sha256']==sha,'receipt native digest differs')
        else:sha=item['sha256'];size=4673783740
        require(type(size) is int and 0<size<=6*2**30,'archive checkpoint exceeds storage bound')
        rows.append((item,sha,size))
    require([x[0]['step'] for x in rows]==[19618,39237,49046],'all three checkpoints required')
    out.mkdir(parents=True,exist_ok=False)
    require(sum(size for _,_,size in rows)<=20*2**30,'archive exceeds reserved space')
    outputs=[]
    for item,sha,size in rows:
        outputs.append({'step':item['step'],**copy_verified(item['source'],out/item['name'],sha,size)})
    P.check_unchanged(inputs);P.check_unchanged({'plan':pin});P.code_identity(code)
    result={'protocol':plan['protocol'],'complete':True,'code_sha':code,'plan_sha256':plan_sha,
            'slurm_job_id':os.environ['SLURM_JOB_ID'],'quota_raw':command,'outputs':outputs,
            'persistent':True,'retention':'WORK; no automatic scratch purge','sources_unchanged':True}
    P.write_json(out/'report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--code',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();execute(a.plan,a.plan_sha,a.code,a.out)
if __name__=='__main__':main()
