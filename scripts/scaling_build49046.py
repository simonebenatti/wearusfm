"""Build registered segment plans from the collected accepted source inventory."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_continue49046 as L
BASE='results/infra/scaling_49046_20261118/launcher'
SCRATCH='/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs'
WORK='/leonardo_work/IscrB_WearUsFM/wearusfm_runs'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pin(relative):
    p=ROOT/relative
    return {'path':relative,'sha256':sha(p),'bytes':p.stat().st_size}

def dump(value):return json.dumps(value,indent=2,allow_nan=False).encode()+b'\n'

def build():
    metadata=json.loads((ROOT/BASE/'metadata_snapshot.json').read_text())
    files=metadata['candidate_files']
    inventory={name:{'bytes':v['bytes'],'sha256':v['sha256']} for name,v in files.items() if 'sha256' in v}
    inventory['metrics.jsonl']={'bytes':files['metrics.jsonl']['bytes'],'sha256':'78f2682e9d9be132d35b08fc2ad94187e4251c66a04ba5f093e96878bcca4c2e'}
    for name in ('checkpoint.pt','checkpoint_step0019618.pt'):
        inventory[name]={'bytes':4673783740,'sha256':L.INITIAL_SHA256}
    assert set(inventory)==set(files)
    contract=pin('results/infra/scaling_300m_20261104/training_source_contract.json')
    receipt=pin('results/infra/scaling_300m_20261104/continuation19618_20261106/evidence/training/receipt.json')
    sources={name:sha(ROOT/name) for name in sorted(L.EXECUTION_SOURCES)}
    context=L.M.check_source_contract(ROOT/contract['path'])
    previous_dir=SCRATCH+'/runs/scaling_300m_20261104/T100-G-300M-19618-from5000-v3'
    plans=[]
    for start,end,wall in L.SEGMENTS:
        p={'protocol':L.PROTOCOL,'arm':L.M.ARM,'start_step':start,'end_step':end,'additional_steps':end-start,
           'scientific_source_sha256':context['contract']['scientific_source_sha256'],
           'source_contract':contract,'execution_source_sha256':sources,
           'allocation':dict(L.C.ALLOCATION,walltime_seconds=wall),'memory':L.C.MEMORY,
           'minimum_free_bytes':20*2**30,'resource_policy':pin(BASE+'/resource_policy.json'),
           'protected_until_utc':'2026-12-19T00:00:00Z',
           'run_dir':SCRATCH+f'/runs/scaling_49046_20261118/T100-G-300M-{end}',
           'receipt_dir':SCRATCH+f'/results/scaling_49046_20261118/training/{end}',
           'control_checkpoint':{'path':WORK+'/results/consolidation_20261104/continuation/checkpoints/checkpoint_postdecay.pt',
                                 'sha256':L.CONTROL_SHA256,'bytes':454381728}}
        if start==19618:
            p['previous']={'run_dir':previous_dir,'receipt':receipt,'files':inventory}
        else:
            p['previous']={'run_dir':plans[0]['run_dir'],'receipt':{'path':plans[0]['receipt_dir']+'/receipt.json'},'files_from_receipt':True}
            p['predecessor_plan_sha256']=hashlib.sha256(dump(plans[0])).hexdigest()
        plans.append(p)
    return plans


def main():
    plans=build()
    for p in plans:
        target=ROOT/BASE/f"training_{p['end_step']}.json"
        target.write_bytes(dump(p));print(target)
        directory=ROOT/'results/infra/scaling_49046_20261118/evaluation/training'
        directory.mkdir(parents=True,exist_ok=True)
        (directory/target.name).write_bytes(target.read_bytes())
    archive=json.loads((ROOT/BASE/'archive_plan.json').read_text())
    for item in archive['checkpoints']:
        if 'receipt' in item:
            item['plan_sha256']=hashlib.sha256(dump(next(p for p in plans if p['end_step']==item['step']))).hexdigest()
    (ROOT/BASE/'archive_plan.json').write_bytes(dump(archive))
    for name in ('resource_policy.json','archive_plan.json'):
        (directory/name).write_bytes((ROOT/BASE/name).read_bytes())
if __name__=='__main__':main()
