from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, subprocess, sys
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
ROOT=args.repo
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('p1mixed_fixtures',ROOT/'tests/cpu/test_scaling_plans19618.py')
f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
S=f.S
checks=[]
def check(name,value):
    assert value,name
    checks.append({'name':name,'passed':True})
check('reviewed exact commit',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip().startswith('c2ae7ee'))
extraction,(receipts,pins,provenances),_=f.new_candidates()
receipts['p1']['code_sha']='d'*40
for ds in S.TASKS['p1']:
    prov=json.loads(provenances[ds]); prov['code_sha']='d'*40
    provenances[ds]=f.body(prov)
    receipts['p1']['outputs'][ds]['provenance_sha256']=f.sha(provenances[ds])
pins['p1']=S.pin(pins['p1']['path'],f.body(receipts['p1']))
try:
    S.candidate_probe_caches(receipts,pins,provenances,extraction)
except ValueError as e:
    check('current pure builder rejects mixed extraction code',str(e)=='extraction receipt differs: code_sha')
else: raise AssertionError('mixed code unexpectedly accepted')
plan=f.new_probe_plan(lambda control,candidate:[candidate[ds].update(extraction_code_sha='d'*40) for ds in S.TASKS['p1']])
check('reader validates mixed-code cache schema', {p['extraction_code_sha'] for p in plan['caches'][S.ARM].values()}=={'c'*40,'d'*40})
check('same checkpoint retained for all candidate caches',len({p['checkpoint_sha256'] for p in plan['caches'][S.ARM].values()})==1)
contract=json.loads((ROOT/'results/infra/scaling_300m_20261104/training_source_contract.json').read_text())
check('shared helper is in frozen training contract','scripts/tokenizer_preflight.py' in contract['scientific_source_sha256'])
check('helper matches frozen pin',hashlib.sha256((ROOT/'scripts/tokenizer_preflight.py').read_bytes()).hexdigest()==contract['scientific_source_sha256']['scripts/tokenizer_preflight.py'])
probe=(ROOT/'scripts/scaling_probe19618.py').read_text()
pose=(ROOT/'scripts/scaling_pose_readout19618.py').read_text()
check('classification load validates code per cache','"code_sha": pin["extraction_code_sha"]' in probe and 'pin["checkpoint_sha256"], pin["extraction_code_sha"]' in probe)
check('pose has independent long-run stat guard','require(states == {p: Q.T.file_state(p) for p in states}, "input artifact changed during execution")' in pose)
record={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checks':checks,'n_checks':len(checks),'limitations':['Mixed schema check uses synthetic receipt/cache pin metadata derived from 5k fixtures, not new19k remote payloads.','load_cache per-cache enforcement additionally inspected in source; no actual cache payloads opened.','No SSH, jobs, deploy, collection or edits to reviewed repo.']}
args.out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
