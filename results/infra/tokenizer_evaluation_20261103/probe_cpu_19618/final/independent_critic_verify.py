import sys,json,hashlib
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]); r=json.loads(p.read_text())
assert r['kind']=='comparison' and r['complete'] is True and r['step']==19618
assert r['code_sha']=='b315dadfedc09c522deeb56af829d34ebc04f4b4'
assert r['plan_sha256']=='dac83e0d3206da9f0c9c9cd3290cc1e7430fa40d496ba73bb01915cc0b573010'
assert r['n_bootstrap']==2000 and r['seed']==20261102
arms=['T25-L','T100-L','T100-G']; rng=np.random.default_rng(20261102)
points={}; boots={}; errors=[]; out={}
contrasts={'T100-L_minus_T25-L':(1,0),'T100-G_minus_T100-L':(2,1),'T100-G_minus_T25-L':(2,0)}
def equal(a,b,label):
 d=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
 assert d<1e-11,(label,d)
 errors.append(d)
def verify(block,point,bs):
 for k,a in enumerate(arms):
  equal(block['cells'][a]['balanced_accuracy'],point[k],a+' point')
  equal(block['cells'][a]['ci95_percent'],100*np.percentile(bs[:,k],[2.5,97.5]),a+' CI')
 summary={a:float(point[k]*100) for k,a in enumerate(arms)}
 for name,(candidate,control) in contrasts.items():
  difference=100*(bs[:,candidate]-bs[:,control]); ci=np.percentile(difference,[2.5,97.5]); delta=100*(point[candidate]-point[control])
  equal(block['contrasts'][name]['delta_pp'],delta,name+' delta')
  equal(block['contrasts'][name]['ci95_pp'],ci,name+' CI')
  ni='supported' if ci[0]>-1 else 'loss_beyond_margin' if ci[1]<-1 else 'unresolved'
  if 'noninferiority_1pp' in block['contrasts'][name]:
   assert block['contrasts'][name]['noninferiority_1pp']==ni
  summary[name]={'delta_pp':float(delta),'ci95_pp':ci.tolist()}
  if 'noninferiority_1pp' in block['contrasts'][name]:
   summary[name]['NI_1pp']=ni
 return summary
for ds in sorted(r['datasets']):
 block=r['datasets'][ds]; pcs=block['paired_counts']; base=pcs[arms[0]]
 users=sorted(base['subjects']); classes=base['classes']; assert users==block['subjects'] and classes==block['classes']
 denominator=np.asarray([base['subjects'][s]['n'] for s in users],dtype=float)
 hits=[]
 for a in arms:
  assert sorted(pcs[a]['subjects'])==users and pcs[a]['classes']==classes
  counts=np.asarray([pcs[a]['subjects'][s]['n'] for s in users]); h=np.asarray([pcs[a]['subjects'][s]['correct'] for s in users],dtype=float)
  assert np.array_equal(counts,denominator) and np.all(counts>=0) and np.all((h>=0)&(h<=counts))
  hits.append(h)
 draw=rng.integers(len(users),size=(2000,len(users)))
 digest=hashlib.sha256(); digest.update(json.dumps([draw.dtype.str,draw.shape]).encode()); digest.update(draw.tobytes()); digest.update(b'\0')
 assert digest.hexdigest()==block['draws_sha256']
 weights=np.array([np.bincount(row,minlength=len(users)) for row in draw])
 den=weights@denominator; present=den>0; nclass=present.sum(axis=1)
 samples=[]
 for h in hits:
  numerator=weights@h
  rates=np.zeros_like(numerator); rates[present]=numerator[present]/den[present]
  samples.append(rates.sum(axis=1)/nclass)
 bs=np.array(samples).T
 totals=denominator.sum(axis=0); included=totals>0
 point=np.asarray([(h.sum(axis=0)[included]/totals[included]).mean() for h in hits])
 assert int((nclass<len(classes)).sum())==block['replicates_missing_classes']
 assert int((nclass==1).sum())==block['single_class_replicates']
 assert block['role']==('primary_classification' if ds=='epn612' else 'secondary_dataset')
 points[ds]=point; boots[ds]=bs; out[ds]=verify(block,point,bs)
 out[ds].update(subjects=len(users),windows=int(denominator.sum()),classes=len(classes),replicates_missing_classes=int((nclass<len(classes)).sum()))
for task,datasets,weights in [('p1',['ninapro_db2','ninapro_db3','ninapro_db6'],[8/13,3/13,2/13]),('p2',['epn612','uci_emg'],[.5,.5])]:
 point=sum(w*points[ds] for ds,w in zip(datasets,weights)); bs=sum(w*boots[ds] for ds,w in zip(datasets,weights))
 out[task]=verify(r['tasks'][task],point,bs)
audit=r['fit_audit']; assert len(audit)==6 and sum(x['fits'] for x in audit.values())==420
assert all(x['converged_by_criterion'] and x['convergence_warnings']==0 and x['reached_limit']==0 and not x['unresolved_fits'] for x in audit.values())
assert r['decision_valid'] is True and r['reading_status']=='solver_converged_by_criterion'
shard_root=p.parent.parent/'completed_shards'
records=[]; audit_files=sorted(shard_root.glob('*/*/audit.jsonl'))
assert len(audit_files)==15
for audit_path in audit_files:
 report=json.loads(audit_path.with_name('shard.json').read_text())
 assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==report['audit_sha256']
 records.extend(json.loads(line) for line in audit_path.read_text().splitlines())
assert len(records)==420
keys=set()
for record in records:
 key=tuple(record[k] for k in ('arm','task','dataset','fold','fit_phase','C')); assert key not in keys; keys.add(key)
 assert record['converged_by_criterion'] and not record['reached_limit'] and record['convergence_warnings']==0 and record['exception'] is None
 assert record['max_iter']==12000 and record['n_iter']<12000 and record['dtype']=='float32' and record['n_features']==768
 assert record['code_sha']==r['code_sha'] and record['plan_sha256']==r['plan_sha256']
for arm in arms:
 for ds in r['datasets']:
  for fold in range(4):
   subset=[v for v in records if v['arm']==arm and v['dataset']==ds and v['fold']==fold]
   assert len(subset)==7
   candidates=[v for v in subset if v['fit_phase']=='candidate']; refit=[v for v in subset if v['fit_phase']=='refit']
   assert sorted(v['C'] for v in candidates)==[.001,.01,.1,1.,10.,100.] and len(refit)==1
   chosen=max(candidates,key=lambda v:(v['val_bacc'],-v['C']))['C']
   assert refit[0]['C']==chosen and all(v['selected']==(v['C']==chosen) for v in candidates)
raw_audit={'files':len(audit_files),'fits':len(records),'max_iterations':max(v['n_iter'] for v in records),'solver_warning_count':sum(v['convergence_warnings'] for v in records),'scoring_warning_records':sum(bool(v.get('scoring_warnings', [])) for v in records),'exceptions':sum(v['exception'] is not None for v in records),'all_candidate_and_refit_coverage_valid':True}
result={'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'independent_recalculation':'subject count multiplicity matrices; no project numerical functions imported','all_checks_passed':True,'max_numeric_difference':max(errors),'fit_audit':audit,'raw_audit_independent':raw_audit,'results':out}
output=Path(sys.argv[2]) if len(sys.argv)>2 else p.with_name('independent_critic_verification.json'); output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
