import sys,json,subprocess,hashlib,ast
from pathlib import Path
sys.dont_write_bytecode=True
R=Path('/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-scaling-exec')
A='f6dc8ec0fb33ae1bcf6d34018194c2e9c71f7cb5';B='40797b1ccb640b8b986f30539f05a73ac0a8f485'
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
import scaling_plans19618 as S,scaling_derive19618 as D
checks=[]
def check(n,c):
 checks.append({'check':n,'pass':bool(c)})
 assert c,n
def git(*a):return subprocess.check_output(['git','-C',str(R),*a])
def blob(sha,p):return git('show',sha+':'+p)
changed=git('diff','--name-only',A,B).decode().splitlines()
allowed={'scripts/scaling_plans19618.py','tests/cpu/test_scaling_gates19618.py','tests/cpu/test_scaling_plans19618.py','docs/decisioni.md',S.PACKAGE+'/evaluation_chain_plan.md'}
check('exact5scopefiles',set(changed)==allowed)
check('no src changes',git('diff',A,B,'--','src/')==b'')
check('derivations verify15',len(D.DERIVATIONS)==15 and D.check()==[])
for p,(source,*_) in D.DERIVATIONS.items():
 check('derived unchanged '+p,blob(A,p)==blob(B,p))
 check('frozen source unchanged '+source,blob(A,source)==blob(B,source))
for p in [S.FREEZE,S.PACKAGE+'/training_plan.json',S.PACKAGE+'/endpoint19618_reading.md',S.PACKAGE+'/scientific_reading_policy_19618.json','scripts/scaling_continue.py','scripts/slurm/scaling_continue.sbatch']:
 check('pin unchanged '+p,blob(A,p)==blob(B,p))
# Semantic builders/job tables stay unchanged, only queue admission and its audit contract is amended.
a=ast.parse(blob(A,'scripts/scaling_plans19618.py'));b=ast.parse(blob(B,'scripts/scaling_plans19618.py'))
afunc={n.name:ast.dump(n,include_attributes=False) for n in a.body if isinstance(n,ast.FunctionDef)}
bfunc={n.name:ast.dump(n,include_attributes=False) for n in b.body if isinstance(n,ast.FunctionDef)}
for n in ['extraction_plan','candidate_pose_caches','pose_plan','candidate_probe_caches','control_probe_caches','probe_plan','accounting','pose_budget','probe_resources','fit_state','comparison_inputs','jobs','checkout_job','stage_table','stage_lines','checkout_lines','build_budgets','build_comparison_inputs','collect','evidence','accounting_pin','release_check']:
 check('unchanged numerical/resource function '+n,afunc[n]==bfunc[n])
table=S.budget_table();check('budget maximum345.166667',abs(table['core_hours_with_scheduler_margin']-345.1666666666667)<1e-10)
check('budget flags',table['within_total_cap'] and table['within_stage_caps'])
check('fits default15shards246coreh',table['stages']['probe-fit']['jobs']==15 and table['stages']['probe-fit']['core_hours_with_scheduler_margin']==246)
check('checkout max0.8coreh',table['stages']['checkout']['core_hours_with_scheduler_margin']==0.8)
check('eight-thread CPUstage unchanged',blob(A,'scripts/slurm/scaling_readers19618.sbatch')==blob(B,'scripts/slurm/scaling_readers19618.sbatch'))
report={'old':A,'new':B,'read_only_local':True,'checks':checks,'checks_passed':len(checks),'changed_files':changed,'budget_table':table,
'line_limit_assessment':'Only changes temporal tranche admission; gate remains bounded by global10slots and unchanged fixed job table. Does not authorize extra jobs or cost. Default4 until Simone changes coordination.',
'checkout_evidence_assessment':'New _collected_checkout compares collected COMPLETED0:0 accounting jobID to commonledger. Existing collect verified rawreceipt and rejects missing/failingjob. Gate now requires evidence for everyusedcheckout.'}
Path('/private/tmp/eval19618-b1-resource.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'cpu_worst':table['core_hours_with_scheduler_margin'],'new_resource_blocker':False}))
