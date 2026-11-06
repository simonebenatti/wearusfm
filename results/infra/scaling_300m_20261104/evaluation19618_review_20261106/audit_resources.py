import sys, json, ast, hashlib
from pathlib import Path
from fractions import Fraction
sys.dont_write_bytecode = True
root=Path('/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-scaling-exec')
sys.path[:0]=[str(root/'src'),str(root/'scripts')]
import scaling_plans19618 as S
freeze=json.loads((root/S.FREEZE).read_text())
readers=freeze['readers']; stages=readers['stages']; checks=[]
def check(name, okay):
    checks.append({'name':name,'pass':bool(okay)})
    assert okay, name
# Independent costs computed directly from numerical freeze and proposed allocations.
checkout_req=Fraction((2+4)*300,3600)
checkout_margin=Fraction((2+4)*(300+180),3600)
bench_times=list(v for group in stages['probe_benchmark']['seconds_by_cell_and_dataset'].values() for v in group.values())
independent={
'checkout':(checkout_req,checkout_margin),
'pose-validate':(Fraction(2*1800,3600),Fraction(2*1980,3600)),
'pose-benchmark':(Fraction(3*8*600,3600),Fraction(3*8*780,3600)),
'probe-benchmark':(Fraction(8*sum(bench_times),3600),Fraction(8*(sum(bench_times)+15*180),3600)),
'pose-ridge':(Fraction(8*3600,3600),Fraction(8*3780,3600)),
'probe-fit':(Fraction(240),Fraction(240)+Fraction(15*8*180,3600)),
'probe-aggregate':(Fraction(6*2*600,3600),Fraction(6*2*780,3600)),
'compare':(Fraction(8*1800,3600),Fraction(8*1980,3600))}
computed=S.budget_table([{'cpus':8,'seconds':7200} for _ in range(15)])
for name,(req,margin) in independent.items():
    check(name+' requested exactly reproduced',abs(computed['stages'][name]['requested_core_hours']-float(req))<1e-10)
    check(name+' margin exactly reproduced',abs(computed['stages'][name]['core_hours_with_scheduler_margin']-float(margin))<1e-10)
    check(name+' within stage cap',float(margin)<=computed['stages'][name]['cap_with_scheduler_margin']+1e-10)
worst=sum(v[1] for v in independent.values())
check('worst total345+1/6',worst==Fraction(2071,6))
check('worst matches freeze',abs(float(worst)-readers['budgeted_core_hours_with_scheduler_margin'])<1e-10)
check('reserve29/6',Fraction(350)-worst==Fraction(29,6))
check('computed total agrees',computed['core_hours_with_scheduler_margin']==float(worst))
check('computed flags true',computed['within_stage_caps'] and computed['within_total_cap'])
extract=S.jobs('extractions')
check('three extractions',len(extract)==3)
check('only300M',all(j['id'].startswith('SCALING-300M-eval19618-') for j in extract))
check('extract exact resource tuples',[(j['gpus'],j['cpus'],j['memory'],j['seconds']) for j in extract]==[(1,8,'100G',9000),(1,8,'100G',7200),(1,8,'100G',9000)])
check('extract GPU7',sum(Fraction(j['gpus']*j['seconds'],3600) for j in extract)==7)
check('extract CPU56',sum(Fraction(j['cpus']*j['seconds'],3600) for j in extract)==56)
check('extract marginGPU0.15',sum(Fraction(j['gpus']*180,3600) for j in extract)==Fraction(15,100))
check('extract marginCPU1.2',sum(Fraction(j['cpus']*180,3600) for j in extract)==Fraction(12,10))
# Actual launcher table must equal independently frozen benchmark allocations.
launcher=(root/'scripts/slurm/scaling_readers19618.sbatch').read_text()
python=launcher.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
node=ast.parse(python)
constants={n.targets[0].id:ast.literal_eval(n.value) for n in node.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='BENCHMARK_SECONDS'}
check('dispatcher benchmark table matchesfreeze',constants['BENCHMARK_SECONDS']==stages['probe_benchmark']['seconds_by_cell_and_dataset'])
check('dispatcher exact login08',"'NodeList': 'login08'" in launcher and "#SBATCH --nodelist=login08" in launcher)
check('dispatcher fit15cap',"1 <= len(shards) <= 15" in launcher)
check('dispatcher walltime3h',"0 < s['walltime_seconds'] <= 10800" in launcher)
check('dispatcher fitCPU240',"sum(s['cpus']*s['walltime_seconds'] for s in shards) <= 240*3600" in launcher)
check('dispatcher30GiB',"'mem') in ('30G', '30720M')" in launcher)
check('dispatcher exact time',"actual_seconds == seconds" in launcher)
# Raw successful checkout receipt independently recomputed.
p=root/S.PACKAGE/'evidence/checkout-training/training'
account=json.loads((p/'accounting.json').read_text()); raw=(p/'sacct.txt').read_bytes()
check('rawaccount SHA',hashlib.sha256(raw).hexdigest()==account['raw_sacct']['sha256'])
rows=raw.decode().strip().splitlines(); header=rows[0].split('|'); rows=[dict(zip(header,r.split('|'))) for r in rows[1:]]
parent=[r for r in rows if r['JobIDRaw']=='63695634'][0]
check('checkout success',parent['State']=='COMPLETED' and parent['ExitCode']=='0:0')
check('checkout allocation',parent['AllocCPUS']=='2' and parent['Timelimit']=='00:05:00')
check('checkout measuredCPU0.038333',int(parent['ElapsedRaw'])==69 and float(Fraction(69*2,3600))==account['core_hours'])
report={'commit':'f6dc8ec0fb33ae1bcf6d34018194c2e9c71f7cb5','read_only_local':True,'checks':checks,'count':len(checks),'independent_costs':{k:{'requested':float(v[0]),'with_scheduler_margin':float(v[1])} for k,v in independent.items()},'worst_cpu_only_core_hours':float(worst),'reserve_cpu_only_core_hours':float(Fraction(350)-worst),'default_budget_table_total':S.budget_table()['core_hours_with_scheduler_margin'],'default_is10shard_scenario_not_max15shard':True}
Path('/private/tmp/eval19618-resources-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks_passed':len(checks),'cpu_max':float(worst),'reserve':float(Fraction(350)-worst)}))
