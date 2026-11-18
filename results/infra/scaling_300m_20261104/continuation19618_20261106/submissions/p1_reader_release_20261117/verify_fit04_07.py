from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_plans19618 as S
selected={f"SCALING-19618-probe-fit-{i:02d}-20261106" for i in range(4,8)}
table=[j for j in S.stage_table('probe-fit',ROOT) if j['id'] in selected]
ids={j['id']:S.ledger_job(j['id']) for j in table}
reading=S.remote_read([p for j in table for p in j['files'].values()]+[S.log_path(j,ids[j['id']]) for j in table],ids.values(),hashed=[p for j in table for p in j['pinned'].values()],listings=[j['folder'] for j in table])
out=ROOT/S.PACKAGE/'submissions/p1_reader_release_20261117/fit04_07_checked'
S.collect('probe-fit',table,ids,reading,out)
resources=S._load(S.PACKAGE+'/readers/probe_resources.json',ROOT)
summary=[]
for j in table:
 r=json.loads((out/j['local']/'shard.json').read_text())
 assert r['complete'] is True and r['step']==19618 and r['code_sha']=='df2de35c05381e7a5a69edde409171b210a3b5d8'
 assert r['plan_sha256']==S.document('probe_plan',ROOT)['sha256'] and r['numeric_sha256']==resources['numeric_sha256']
 a=r['audit_summary'];assert a['converged_by_criterion'] is True and not a.get('unresolved_fits')
 for name,key in [('shard.oof.npz','oof'),('audit.jsonl','audit')]:
  pin=next(x for x in reading['files'] if x['path']==j['folder']+'/'+name)
  assert r[key+'_sha256']==pin['sha256'] and r[key]==pin['path']
 summary.append({'id':j['id'],'job_slurm':ids[j['id']],'audit_summary':a})
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
