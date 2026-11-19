import copy
import hashlib
import json
from pathlib import Path
import pytest
import contrastive_targets_plan as P
from contrastive_benchmark_readout import trajectory_drift


def test_amended_matrix_has_four_reloads_after_their_own_full_checkpoint():
    science=P.science_plan(json.loads(P.PLAN.read_text())); plan=P.preflight(science)
    reloads=[c for c in plan['cells'] if c['mode']=='resume']
    assert len(reloads)==4
    assert {(c['arm'],c['repeat']) for c in reloads}=={('CN','a'),('CN','b'),('CR','a'),('CR','b')}
    assert all(c['depends']==['full_'+c['arm']] for c in reloads)
    assert len(plan['cells'])+plan['checkouts']['count']==17
    assert plan['checkouts']['stages']==['audit','calibration','cuda','full','reloads','readout']
    assert plan['checkouts']['mandatory_sbatch_overrides']==['--cpus-per-task=4','--mem=2G','--time=00:04:00']
    resources=plan['resources']
    assert resources['gpu_hours_cap']==18
    assert resources['core_hours_cap']==pytest.approx(169.6)
    assert resources['historical_actual']['gpu_hours']==pytest.approx(6.086666666666667)
    assert resources['historical_actual']['core_hours']==pytest.approx(57.82777777777778)
    assert resources['historical_unspent_not_new_authorization']['gpu_hours']==pytest.approx(7.913333333333333)
    assert resources['slurm_extra_allowance']['extra_gpu_hours']==pytest.approx(3.6)
    assert resources['slurm_extra_allowance']['extra_core_hours']==pytest.approx(32.8)
    assert plan['science_plan']['sha256']==hashlib.sha256(P.encoded(science)).hexdigest()
    assert not plan['executable'] and not plan['submission_authorized']


def test_decisive_rule_is_byte_parity_and_old_trajectory_tolerances_are_descriptive():
    p=P.science_plan(json.loads(P.PLAN.read_text()))
    assert P.science_plan(p)==p
    n=p['numerical_amendment']['neutrality']
    assert n['updates_before']==[128,256] and n['global_batch']==32 and n['backwards_per_arm']==2
    assert n['trajectory_1_256']=={'relative_tolerance':1e-4,'absolute_tolerance':1e-6,'role':'descriptive_only'}
    assert p['numerical_amendment']['resume']['fresh_reloads_per_arm']==2
    assert p['evaluation']==json.loads(P.PLAN.read_text())['evaluation']
    assert p['common']['numeric_profile']['sdpa']=='math_only'


def test_longitudinal_drift_keeps_every_scalar_even_when_all_historical_checks_fail():
    a=[{'step':i,'total':1.,'grad_norm':2.} for i in range(1,257)]
    b=[dict(r,total=1.1,grad_norm=2.1) for r in a]
    report=trajectory_drift(a,b)
    assert report['descriptive_only'] and 'passed' not in report
    assert report['fields']['total']['outside_count']==256
    assert len(report['fields']['grad_norm']['updates'])==256
    b[0]['total']=float('nan')
    with pytest.raises(ValueError,match='finite'):trajectory_drift(a,b)
    with pytest.raises(ValueError,match='exactly'):trajectory_drift(a[:-1],a[:-1])
