from pathlib import Path
import json,sys,math
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_plans19618 as S
import scaling_p1_reader_bridge as B
r=S._load(S.PACKAGE+'/evidence/compare/report.json',ROOT)
i=S._load(S.PACKAGE+'/readers/comparison_inputs.json',ROOT)
assert r['complete'] is True and r['checkpoint_step']==19618 and r['inputs']==i and r['inputs_content_sha256']==S.C.jhash(i)
assert all(r['quality'].values()) and r['reading']==S.M.S8.read_comparison(r)
checks=[]
for name,v in r['secondary']['classification']['datasets'].items():
 c=v['cells'];a=c['T100-G-30M-natural']['balanced_accuracy'];b=c['T100-G-300M-natural']['balanced_accuracy'];delta=100*(b-a)
 assert math.isclose(delta,v['contrasts']['natural_300m_minus_30m']['delta_pp'],abs_tol=1e-10)
 # Independent class-wise pooling of integer correct/count records.
 for cell,counts in v['paired_counts'].items():
  rows=list(counts['subjects'].values());totals=[sum(x['n'][j] for x in rows) for j in range(len(counts['classes']))];hits=[sum(x['correct'][j] for x in rows) for j in range(len(totals))]
  point=sum(h/n for h,n in zip(hits,totals))/len(totals)
  assert math.isclose(point,c[cell]['balanced_accuracy'],abs_tol=1e-10)
 checks.append(name)
for task,v in r['secondary']['classification']['tasks'].items():
 delta=100*(v['cells']['T100-G-300M-natural']['balanced_accuracy']-v['cells']['T100-G-30M-natural']['balanced_accuracy'])
 assert math.isclose(delta,r[task]['delta_pp'],abs_tol=1e-10)
pose=r['pose'];assert math.isclose(pose['candidate_E_dyn']-pose['control_E_dyn'],pose['delta_E_dyn'],abs_tol=1e-12)
budget=B.budget(S,ROOT,S.LEDGER)
out={'complete':True,'independent_integer_count_points_verified':checks,'reading_reproduced':True,'manifest_verified':True,'quality':r['quality'],'budget':budget,'limits':'Point estimates recomputed locally from recorded integer counts. Full arrays, bootstrap and numerical consistency were checked by the frozen comparison job; no independent second execution of that full job.'}
(ROOT/S.PACKAGE/'submissions/p1_reader_release_20261117/final_report_check.json').write_bytes(S.dumps(out))
print(json.dumps({'checked_datasets':checks,'budget':budget['total'],'reading':r['reading']['primary_reading']},indent=2))
