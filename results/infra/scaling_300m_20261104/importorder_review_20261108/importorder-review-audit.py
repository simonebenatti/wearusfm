"""Local review only. No SSH, no data loading, no edits to reviewed worktrees."""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess
parser=argparse.ArgumentParser()
parser.add_argument('--diagnosis-repo',type=Path,required=True)
parser.add_argument('--fix-repo',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
DIAG=args.diagnosis_repo
FIX=args.fix_repo
EVIDENCE=Path(__file__).resolve().parent
BASE=DIAG/'results/infra/scaling_300m_20261104/continuation19618_20261106'
D=BASE/'failure_63696864_diagnosis_20261108'
checks=[]
def check(name,value):
    checks.append({'name':name,'passed':bool(value)})
    assert value,name
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)
check('diagnosis HEAD',git(DIAG,'rev-parse','HEAD').decode().strip()=='913b4ae4bdfd8de709641ef39383a9fa7e0020f0')
check('fix HEAD',git(FIX,'rev-parse','HEAD').decode().strip()=='0c5b974af2483faf767a2e171183ae1af0dcdb19')
check('reviewed worktrees clean',not git(DIAG,'status','--porcelain') and not git(FIX,'status','--porcelain'))
changed=git(FIX,'diff','--name-only','0c5b974^','0c5b974').decode().splitlines()
check('fix changes only wrapper and test',changed==['scripts/scaling_continue.py','tests/cpu/test_scaling_continue_launcher.py'])
check('scientific src and trainer unchanged',not git(FIX,'diff','0c5b974^','0c5b974','--','src/','scripts/train_jepa.py'))
old=git(FIX,'show','0c5b974^:scripts/scaling_continue.py').decode()
new=(FIX/'scripts/scaling_continue.py').read_text()
a,b=ast.parse(old),ast.parse(new)
b.body=[n for n in b.body if not(isinstance(n,ast.Import) and any(x.name=='torch' for x in n.names))]
check('only semantic wrapper change is import torch',ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
reports={m:load(D/'diag4_63864244'/f'{m}_report.json') for m in ['numpy_first','run_first','torch_first']}
table=load(D/'diag4_63864244/summary.json')['table']
for name,r in reports.items():
    for panel in ['fixed','panel']:
        n=8 if panel=='fixed' else 64
        for ref in ['A','B']:
            c=r[f'{panel}_vs_{ref}']
            check(f'{name} {panel} {ref} recount',len(c['hash_equal'])==n and sum(c['hash_equal'])==c['n_hash_equal']==table[name][f'{panel}_{ref}'] and c['n']==n)
        check(f'{name} {panel} same windows',r[f'{panel}_vs_A']['windows_equal'])
    check(f'{name} no skips and thread1',not r['skips'] and r['torch_threads']==1 and r['omp']=='1' and r['ftz']=={'ftz':False,'daz':False})
check('torch first matches original identities',reports['torch_first']['fixed_vs_A']['identical'] and reports['torch_first']['panel_vs_A']['identical'])
check('run first and numpy first reproduce failed contents',all(reports[m]['fixed_vs_B']['identical'] and reports[m]['panel_vs_B']['identical'] for m in ['numpy_first','run_first']))
check('run first and numpy first per-component hashes same',reports['run_first']['fixed_components']==reports['numpy_first']['fixed_components'])
changed_components=sorted({k for x,y in zip(reports['torch_first']['fixed_components'],reports['numpy_first']['fixed_components']) for k in x if x[k]!=y[k]})
check('only signals and anchor targets differ',changed_components==['anchor_targets','signals'])
plan=load(BASE/'training_plan.json')
identities={}
for name in ['fixed_eval_set.json','pose_panel_set.json']:
    p=D/'diag4_63864244/plain_cli_run'/name
    got={'sha256':digest(p),'bytes':p.stat().st_size}
    check('CLI '+name+' exact historical pin',got==plan['previous']['files'][name])
    identities[name]=got
arrays=load(D/'diag4_63864244/summary.json')['array_diffs_vs_numpy_first']
check('run-first array differences none',all(v=='uguale' for v in arrays['run_first'].values()))
different={k:v for k,v in arrays['torch_first'].items() if v!='uguale'}
signals={k:v for k,v in different.items() if '.signals[' in k}
other={k:v for k,v in different.items() if not '.signals[' in k}
check('array differences only signals and anchors',all('.anchor_targets' in k for k in other))
compare3=load(D/'diag3_63861825/comparison.json')
check('diagnostic3 variants same contents',all(r['identical'] for r in compare3['component_pairs'].values()))
check('diagnostic3 GPU and CPU same new identity',all(r['equals_login08'] and not r['equals_seed'] and r['windows_equal_seed'] for r in compare3['summary'].values()))
log=(D/'failed_job/training.log').read_text()
check('failure at fixed identity and start5000',"ripresa dal passo 5000" in log and 'run.py\", line 1431' in log and "_write_or_check" in log)
source=(DIAG/'src/wearusfm/training/run.py').read_text()
check('fixed set guard before data stream and train loop',source.index('_write_or_check(out_dir / "fixed_eval_set.json"') < source.index('data = torch.utils.data.DataLoader'))
oldsha=plan['execution_source_sha256']['scripts/scaling_continue.py']; newsha=digest(FIX/'scripts/scaling_continue.py')
check('v1 pins old wrapper exactly',oldsha==hashlib.sha256(old.encode()).hexdigest() and oldsha!=newsha)
check('40 launcher tests green','40 passed in 45.62s' in (EVIDENCE/'importorder-review-launcher-tests.txt').read_text())
check('seven expected v1 pin failures reproduced','7 failed, 110 deselected in 3.32s' in (EVIDENCE/'importorder-review-v1-pin-tests.txt').read_text())
result={'diagnosis_commit':'913b4ae4bdfd8de709641ef39383a9fa7e0020f0','fix_commit':'0c5b974af2483faf767a2e171183ae1af0dcdb19','checks':checks,'n_checks':len(checks),'historical_cli_identity_files':identities,'changed_components':changed_components,'array_differences':{'arrays':len(arrays['torch_first']),'different':len(different),'signals':signals,'max_other_abs':max(v['max_abs_diff'] for v in other.values())},'old_wrapper_sha256':oldsha,'new_wrapper_sha256':newsha,'diagnostic_fingerprints':load(D/'diag4_63864244/fingerprints.json')['summary'],'tests':{'launcher_passed':40,'launcher_seconds':45.62,'expected_old_plan_failures':7,'targeted_pin_seconds':3.32},'assessment':'Minimal code correction is technically approved for the diagnosed path; operational recovery package is not yet released because v1 pins, output directory, IDs and retry accounting require a new immutable plan. Historical identity equality is demonstrated; GPU trajectory bitwise equality is not newly proven.','limits':['Diagnostic array magnitudes cover only dumped first two fixed/panel batches, not all training examples.','Current412 fingerprints are a future baseline; not a full historical hash proof of75760posefiles.','dladdr/CDLL global symbols and lazy MKL loading corroborate library mechanism, while causal import-order identity reproduction is stronger than threadpool inventory alone.']}
args.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'different_arrays':len(different),'wrapper_sha256':newsha},indent=2))
