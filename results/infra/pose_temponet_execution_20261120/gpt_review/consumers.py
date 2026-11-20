"""Independent consumer and inverse-mapping checks; real IDs, synthetic arrays, no cluster I/O."""
import copy, importlib.util, itertools, json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(root/'src'), str(root/'scripts')]
from pose_temponet import provenance as P, benchmark as B, workflow as W, controls as C

plan = json.loads((root/'results/infra/pose_temponet_execution_20261120/plan.json').read_text())
P.validate_plan(plan, source_root=root)
spec=importlib.util.spec_from_file_location('real_layout',root/'tests/cpu/test_pose_temponet_preflight_real_layout.py')
layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
data = layout._cache('train',plan['producers']['train']['users'],2,2719)
folds, population = B._validate_train(data,plan['producers']['train']['users'],{'cache_sha256':plan['inputs']['cache_train']['sha256']})
bundle = W.prepare_real_folds(data,allowed_train_users=plan['producers']['train']['users'],source_identity={'cache_sha256':plan['inputs']['cache_train']['sha256']})
assert [x.tolist() for x in folds] == [x.tolist() for x in bundle.fitting]
partition = C.validate_outer_partition(data['meta'],folds[0],folds[1],plan['producers']['train']['users'])
old = plan['producers']['train']['storage_directories']
failed=[]
for name,op in [('benchmark',lambda:B._validate_train(data,old,{'cache_sha256':'x'})),('workflow',lambda:W.prepare_real_folds(data,allowed_train_users=old,source_identity={'cache_sha256':'x'})),('controls',lambda:C.validate_outer_partition(data['meta'],folds[0],folds[1],old))]:
 try:op()
 except ValueError as exc:failed.append([name,str(exc)])
 else:raise AssertionError(name+' unexpectedly allowed old IDs')
assert len(failed)==3

# Enumerate safe/unsafe IDs: when the inverse accepts a directory, it must recover the exact input;
# distinct inputs colliding on a sanitized directory must all be rejected.
collisions={}; recovered=0; rejected=0
alphabet='u0aZ._@-/ é'
for n in range(1,5):
 for tup in itertools.product(alphabet,repeat=n):
  user=''.join(tup); directory=P.storage_directory(user)
  collisions.setdefault(directory,set()).add(user)
  try: decoded=P.cache_user(directory)
  except ValueError:rejected+=1
  else:
   assert decoded==user,(decoded,user,directory)
   recovered+=1
ambiguous=[d for d,ids in collisions.items() if len(ids)>1]
for d in ambiguous:
 try:P.cache_user(d)
 except ValueError:pass
 else:raise AssertionError('ambiguous directory accepted: '+d)
train=plan['producers']['train']['users'];val=plan['producers']['val']['users']
assert len(train)==40 and len(val)==15 and not set(train)&set(val)
out={'plan_validated':True,'real_train_ids':len(train),'real_val_ids':len(val),'train_val_disjoint':True,
     'benchmark_and_workflow_same_folds':True,'synthetic_windows_real_geometry':len(data['meta']),
     'users_per_fold':[len(partition['fitting_users']),len(partition['held_users'])],
     'old_names_rejected_in_consumers':failed,'inverse_cases':recovered+rejected,'unambiguous_accepted':recovered,
     'ambiguous_directories_refused':len(ambiguous),'array_values':'synthetic','cluster_operations':False}
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out,indent=2))
