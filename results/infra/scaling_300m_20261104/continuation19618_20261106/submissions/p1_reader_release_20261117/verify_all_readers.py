from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_plans19618 as S
import scaling_p1_reader_bridge as B
code=B.CODE
fits=S.build_fit_state(code,ROOT)
assert fits['all_converged_by_criterion'] and fits['unresolved_fits']==0 and fits['reached_limit']==0 and fits['convergence_warnings']==0
pose=S._load(S.PACKAGE+'/readers/pose_plan.json',ROOT)
report=S._load(S.PACKAGE+'/evidence/pose-ridge/report.json',ROOT)
assert report['complete'] and report['status']=='completed' and report['step']==19618
assert report['plan_sha256']==S.document('pose_plan',ROOT)['sha256']
for k,v in [('input_pins',pose['caches']),('reference_pins',pose['references']),('projection_pin',pose['projection']),('training_source_contract',pose['training_source_contract'])]: assert report[k]==v,k
bench=S.evidence('pose-benchmark',ROOT)[0]
assert set(report['cells'])==set(S.CELLS)
pins=[]
for cell,d in report['cells'].items():
 S.M.validate_pose_selection(d['fit']['selection'],pose['fold_users'])
 assert not d['fit']['selection']['boundary_unresolved']
 assert all(report['runtime'][k]==json.loads(bench[cell+'/report.json'][0])['runtime'][k] for k in S.RUNTIME_KEYS)
 pins.extend(d[k] for k in ('coefficients','predictions'))
reading=S.remote_read(hashed=[p['path'] for p in pins])
for pin in pins:
 f=next(f for f in reading['files'] if f['path']==pin['path']);assert f.get('canonical') is True and not f.get('missing') and f['sha256']==pin['sha256']
budget=B.budget(S,ROOT,S.LEDGER)
assert budget['within_total_cap'] and budget['within_stage_caps']
result={'fits':fits,'pose_complete':True,'pose_selection_and_runtime_verified':True,'pose_output_pins_verified':pins,'pose_output_reading':reading,'budget':budget,'ready_for_aggregates':True,'comparison_requires':'six accepted aggregates, published comparison inputs and fresh operational gate; full numerical comparison validates pose arrays and paired statistics'}
out=ROOT/S.PACKAGE/'submissions/p1_reader_release_20261117/all_readers_verification.json'
out.write_bytes(S.dumps(result))
print(json.dumps({'fits':fits,'pose_output_files':len(pins),'budget_core_hours':budget['total']['core_hours'],'ready_for_aggregates':True},indent=2))
