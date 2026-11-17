"""Independent local review of the accepted P1-r2 -> frozen CPU reader bridge.
Run with wearusfm-tok Python; no remote calls or repository writes.
"""
from pathlib import Path
import hashlib, json, subprocess, sys
ROOT = Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-p1-recovery')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import scaling_plans19618 as S
import scaling_p1_reader_bridge as B
sha = lambda data: hashlib.sha256(data).hexdigest()
base=ROOT/B.BASE
contract=json.loads((base/'evidence/release/training_source_contract.json').read_bytes())['scientific_source_sha256']
source_results={path:sha((ROOT/path).read_bytes()) == expected for path,expected in contract.items()}
frozen_paths=['scripts/scaling_evaluation19618.py','scripts/scaling_probe19618.py','scripts/scaling_pose_readout19618.py','scripts/scaling_compare19618.py','scripts/slurm/scaling_readers19618.sbatch','scripts/submit_job.py']
frozen={path:subprocess.check_output(['git','-C',str(ROOT),'show',B.CODE+':'+path]) == (ROOT/path).read_bytes() for path in frozen_paths}
pose,probe,lineage=B.plans(S,ROOT)
budget=B.budget(S,ROOT,S.LEDGER)
dependencies=B.dependencies(S,'pose-validate',ROOT,S.LEDGER)
files=[*B.MAC_SOURCES,'tests/cpu/test_scaling_p1_reader_bridge.py','tests/cpu/test_scaling_p1_reader_budget.py',S.PACKAGE+'/submissions/p1_reader_release_20261117/plan.md']
assert all(source_results.values()) and len(source_results)==103
assert all(frozen.values())
assert set(dependencies.values()) == {'64571999','64620434','64572001'}
assert budget['within_total_cap'] is True and budget['within_stage_caps'] is True
result={
 'protocol':'independent-p1-reader-controller-review-20261117-v1',
 'verdict':'APPROVATO', 'blocking_findings':[],
 'scope':'Mac-only explicit P1-r2 reader profile, bridge, cumulative CPU budget and unchanged frozen remote numerical files. No remote inspection, deploy, submission, cancellation or repository mutation.',
 'test_run_observed':{'command':'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider tests/cpu/test_scaling_p1_reader_bridge.py tests/cpu/test_scaling_p1_reader_budget.py tests/cpu/test_scaling_gates19618.py tests/cpu/test_scaling_plans19618.py','passed':277,'failed':0,'elapsed_seconds':33.09,'run_separately_before_this_audit':True},
 'reviewed_files_sha256':{path:sha((ROOT/path).read_bytes()) for path in files},
 'scientific_source_contract':{'entries':len(source_results),'all_unchanged':all(source_results.values())},
 'exact_frozen_comparisons':frozen,
 'real_evidence':{'acceptance_sha256':B.ACCEPTANCE_SHA256,'plans_reconstructed_exactly':True,'lineage_sha256':sha(S.dumps(lineage)),'predecessors':dependencies},
 'budget':{'total':budget['total'],'cap_core_hours':budget['cap_core_hours'],'fixed_prior_reserve':budget['fixed_prior_reserve'],'remaining_chain':budget['remaining_chain'],'fit_plan_ready':budget['fit_plan_ready'],'actual_jobs':budget['actual_jobs'],'three_remaining_checkouts_already_included':True},
 'manual_review':[
  'Only accepted P1-r2 64620434 may replace failed P1 attempts; pose and P2 keep frozen producer df2de35.',
  'Validation report and accepted accounting are required before either benchmark stage; all benchmark collections precede fit/ridge release.',
  'The active tranche follows queue room and line limit, chains afterok, and next tranche waits for previously submitted stage jobs COMPLETED 0:0.',
  'lines requires a recent passed explicit-profile gate, identical evidence/controller source attestation, documents and all checkout commits; ledger changes invalidate the gate.',
  'Metadata checkouts descend from df2de35; release_check and rereview_check have no broad exception. Numerical code, dispatcher and submitter are byte-identical.',
  'Unknown/ambiguous logical CPU IDs stop; absent accounting keeps a full reservation; verified actual cost replaces reservation once.',
  'Foreground collect/gate/lines/submitter calls suffice; no new autonomous sender or persistent automation is needed.'
 ],
 'limits':[
  'A passed gate is a bounded-time snapshot, not an atomic scheduler reservation; a concurrent QoS change may cause a submission refusal, which must stop the chain.',
  'The legacy ledger reader now recognizes GPT for the registered remaining CPU IDs even without the new flag; default numerical operations and historical commands remain unchanged.',
  'Review approves the operator code and frozen scope, not future benchmark outcomes. Runtime, memory, budget and convergence gates must still pass.',
  'The new identity guard was not causally exercised by a timestamp transition in P1-r2; no Lustre root cause is inferred.'
 ]
}
Path('/private/tmp/p1-reader-controller-review-20261117.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'verdict':result['verdict'],'scientific_files':len(source_results),'frozen_files':len(frozen),'budget':budget['total'],'output':'/private/tmp/p1-reader-controller-review-20261117.json'},indent=2))
