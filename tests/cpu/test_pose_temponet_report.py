"""Paired-user bootstrap, three-head-seed estimand and directional decisions."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import metrics as M
from pose_temponet import report as R
from test_pose_temponet_data import fixture


@pytest.fixture(scope="module")
def matrix():
    data,_ = fixture(users=tuple(f"u{i:02d}" for i in range(15)),windows=2)
    data.pop("lat"); data.pop("pre")
    data["pose_var"][1::2] = .1
    targets = data["target"].astype(float)
    n = len(targets)
    heterogeneity = np.repeat(np.linspace(-.1,.1,15),2)[:,None,None]
    time = np.sin(np.arange(160)[None,:,None]/7)
    joint = np.cos(np.arange(20)[None,None,:])
    cells,predictions = {},{}
    for head,factor in {"L0":.15,"L13":.5,"MLP":.55,"TCN_post":.65,"TCN_pre":.1}.items():
        cells[head],predictions[head] = {},{}
        for seed in R.SEEDS:
            # Seed variation must be averaged as loss, never ensembled away.
            pred = targets*(factor+.12*(seed-1)+heterogeneity)
            pred += .07*(seed+1)*time*joint
            pred += 1 if head == "TCN_post" else .02  # explicit level trade-off
            predictions[head][seed] = pred
            cells[head][seed] = {"complete":True,"analysis":M.evaluate(pred,data,per_joint=True)}
    references = {h:{"complete":True,"analysis":M.evaluate(targets*f,data)}
                  for h,f in zip(R.REFERENCES,(.2,.22))}
    assessments = {h:{"optimizer_alarms":[],"optimization_unresolved":False,
                      "objective_tradeoff":{"flag":False}} for h in R.HEADS}
    gates = {g:True for g in R.COMMON_GATES}
    result = R.aggregate(cells,references,assessments=assessments,common_gates=gates)
    assert result["complete"], result
    return cells,references,assessments,gates,result,data,predictions


def naive_error(cell, users):
    rows = cell["analysis"]["groups"]["dynamic"]["centered"]["per_user"]
    return np.array([rows[u]["sums"]["centered"] for u in users])


def test_all_five_paired_intervals_match_independent_loop_and_fixed_draws(matrix):
    cells,references,_,_,result,_,_ = matrix
    users = result["users"]
    draws = np.random.default_rng(20261102).integers(0,15,(2000,15))
    expected_hash = hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()
    assert result["bootstrap"]["draws_sha256"] == expected_hash
    assert result["bootstrap"]["individual_confidence_percent"] == 99
    assert result["bootstrap"]["family_confidence_percent"] == 95
    assert not result["bootstrap"]["hierarchical_seed_bootstrap"]
    candidate = np.mean([naive_error(cells["TCN_post"][s],users) for s in R.SEEDS],axis=0)
    for i,head in enumerate(R.COMPARATORS,1):
        control = (np.mean([naive_error(cells[head][s],users) for s in R.SEEDS],axis=0)
                   if head in cells else naive_error(references[head],users))
        expected = [100*(1-sum(candidate[j] for j in draw)/sum(control[j] for j in draw)) for draw in draws]
        effect = result["contrasts"][str(i)]["relative_E_dyn"]
        assert effect["point"] == pytest.approx(100*(1-candidate.sum()/control.sum()),abs=1e-12)
        np.testing.assert_allclose(effect["ci"],np.percentile(expected,[.5,99.5]),atol=1e-12,rtol=0)
        assert effect["replicates_discarded"] == effect["undefined_replicates"] == 0
    assert set(result["contrasts"]) == set("12345")
    assert result["contrasts"]["1"]["primary"]
    assert not result["contrasts"]["2"]["primary"]
    published = result["paired_user_accumulators"]["TCN_post"]["dynamic"]
    np.testing.assert_allclose([published[u]["centered"] for u in users],candidate,rtol=0,atol=0)


def test_estimand_averages_losses_not_predictions_and_reports_seed_sd_separately(matrix):
    cells,_,_,_,result,data,predictions = matrix
    head = result["heads"]["TCN_post"]["dynamic"]
    values = [cells["TCN_post"][s]["analysis"]["groups"]["dynamic"]["centered"]["centered_mse_ratio"]
              for s in R.SEEDS]
    assert head["mean_accumulators_metrics"]["E_dyn"] == pytest.approx(np.mean(values))
    assert head["sample_sd_seed_metrics"]["E_dyn"] == pytest.approx(np.std(values,ddof=1))
    ensemble = np.mean(list(predictions["TCN_post"].values()),axis=0)
    ensemble_ratio = M.evaluate(ensemble,data)["groups"]["dynamic"]["centered"]["centered_mse_ratio"]
    assert ensemble_ratio < head["mean_accumulators_metrics"]["E_dyn"]-.005
    corrs = [head["per_seed"][str(s)]["correlation"] for s in R.SEEDS]
    assert head["arithmetic_mean_seed_metrics"]["correlation"] == pytest.approx(np.mean(corrs))
    assert abs(head["mean_accumulators_metrics"]["correlation"]-np.mean(corrs)) > 1e-5
    effect = result["contrasts"]["1"]
    assert len(effect["cross_seed_points_descriptive_only"]) == 9
    gains = [effect["per_seed_same_seed_contrasts"][str(s)]["point"] for s in R.SEEDS]
    assert effect["sample_sd_seed_gains"] == pytest.approx(np.std(gains,ddof=1))


def test_descriptives_include_all_groups_and_joints_without_extra_candidate_gate(matrix):
    _,_,_,_,result,_,_ = matrix
    assert result["candidacy"]["candidate"]
    contrast = result["contrasts"]["1"]
    assert set(contrast["secondary_descriptive"]) == {"all","dynamic","static"}
    level = contrast["secondary_descriptive"]["dynamic"]["level_delta_ci95"]
    assert level["confidence_percent"] == 95 and level["resolved_worsening"]
    assert level["positive_means"] == "candidate worse"
    assert set(result["descriptive"]) == {"L0_vs_historical_post_ridge","L0_vs_historical_trained_head",
                                          "MLP_vs_L0","L13_vs_L0","TCN_post_vs_TCN_pre"}
    for head in R.HEADS:
        for group in R.GROUPS:
            assert set(result["per_joint"][head][group]) == set(M.JOINTS)
    anchor = result["historical_pre_ridge"]
    assert anchor["E_dyn"] == 1.0321280032044744 and anchor["paired_CI"] is None and anchor["summary_anchor_only"]
    pre = result["heads"]["TCN_pre"]["dynamic"]["mean_accumulators_metrics"]["E_dyn"]
    assert anchor["TCN_pre_relative_gain_percent"] == pytest.approx(100*(1-pre/anchor["E_dyn"]))
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("point,lo,hi,expected",[
    (2.,.1,1.9,R.BENEFIT),  # priority: benefit before unsupported
    (-2.,-3.,-1.,R.HARM),
    (1.,.1,1.9,R.UNSUPPORTED),
    (-1.,-1.5,-.5,R.UNSUPPORTED),
    (2.,0.,3.,R.INCONCLUSIVE),
    (1.9,.1,2.,R.INCONCLUSIVE),
])
def test_first_true_labels_at_practical_and_interval_boundaries(point,lo,hi,expected):
    assert R.outcome(point,lo,hi) == expected


def test_mc_border_is_annotation_not_relabel():
    assert R.monte_carlo_border([-.09,3.])
    assert R.monte_carlo_border([.5,1.91])
    assert not R.monte_carlo_border([.11,2.11])
    assert not R.monte_carlo_border(None)
    assert R.outcome(3.,.09,4.) == R.BENEFIT
    with pytest.raises(ValueError,match="invalid"):
        R.outcome(np.nan,0,3)


def decisions():
    contrasts = {}
    for i in range(1,6):
        contrasts[str(i)] = {"statistical_label":R.BENEFIT,
                             "relative_E_dyn":{"evaluable":True,"point":3.,"ci":[1.,5.]}}
    assessments = {h:{"optimization_unresolved":False,"objective_tradeoff":False} for h in R.HEADS}
    return contrasts,assessments,{g:True for g in R.COMMON_GATES}


def test_candidate_requires_1_4_5_and_admitted_comparators_but_tcn_alarm_keeps_benefit():
    contrasts,assessments,gates = decisions()
    assessments["TCN_post"]["optimization_unresolved"] = True
    assert R.candidacy(contrasts,assessments,gates,candidate_e_dyn=.7)["candidate"]
    for i in (1,4,5):
        change = copy.deepcopy(contrasts)
        change[str(i)]["statistical_label"] = R.INCONCLUSIVE
        assert not R.candidacy(change,assessments,gates,candidate_e_dyn=.7)["candidate"]
    for comparator in ("L0","MLP","L13"):
        change = copy.deepcopy(assessments)
        change[comparator]["optimization_unresolved"] = True
        assert not R.candidacy(contrasts,change,gates,candidate_e_dyn=.7)["candidate"]
    for i in (2,3):
        change = copy.deepcopy(contrasts)
        change[str(i)]["statistical_label"] = R.HARM
        assert not R.candidacy(change,assessments,gates,candidate_e_dyn=.7)["candidate"]


def test_mixed_candidate_and_s2_are_named_without_claiming_noninferiority():
    contrasts,assessments,gates = decisions()
    contrasts["3"] = {"statistical_label":R.UNSUPPORTED,
                      "relative_E_dyn":{"evaluable":True,"point":-1.,"ci":[-1.5,-.5]}}
    assessments["L13"]["objective_tradeoff"] = True
    assessments["TCN_post"]["optimization_unresolved"] = True
    result = R.candidacy(contrasts,assessments,gates,candidate_e_dyn=.7)
    assert result["candidate"] and result["mixed_better_comparators"] == ["L13"]
    assert result["name"] == "candidata, con L13 risolti migliori ma stima sotto il 2%"
    assert any("non inferiorità" in q for q in result["qualifiers"])
    assert any("selezione MSE comune" in q for q in result["qualifiers"])


def test_directional_optimization_alarms_preserve_positive_not_false_negative():
    assert R.directional_reading(R.BENEFIT,True,False)["scientific_label"] == R.BENEFIT
    for label in (R.HARM,R.UNSUPPORTED,R.INCONCLUSIVE):
        assert R.directional_reading(label,True,False)["interpretation_suspended"]
    blocked = R.directional_reading(R.BENEFIT,False,True)
    assert blocked["scientific_label"] is None and not blocked["may_use_absence_of_practical_veto"]
    assert R.directional_reading(R.HARM,False,True)["scientific_label"] == R.HARM


@pytest.mark.parametrize("mutation,reason",[
    ("seed","three head seeds"),("complete","incomplete"),("support","support hash"),
    ("target","paired target"),("nan","nonfinite"),("user","users differ"),
    ("joint","missing all-joint"),("joint_target","paired target"),("threshold","threshold changed"),
])
def test_missing_or_unpaired_matrix_is_technical_failure_without_scientific_contrasts(matrix,mutation,reason):
    original,references,assessments,gates,_,_,_ = matrix
    cells = copy.deepcopy(original)
    analysis = cells["TCN_post"][2]["analysis"]
    dynamic = analysis["groups"]["dynamic"]
    if mutation == "seed":
        cells["TCN_post"].pop(2)
    elif mutation == "complete":
        cells["TCN_post"][2]["complete"] = False
    elif mutation == "support":
        analysis["support_sha256"] = "different"
    elif mutation == "target":
        dynamic["centered"]["per_user"]["u00"]["sums"]["vary"] += 1
    elif mutation == "nan":
        dynamic["centered"]["per_user"]["u00"]["sums"]["centered"] = float("nan")
    elif mutation == "user":
        dynamic["centered"]["per_user"].pop("u00")
    elif mutation == "joint":
        dynamic["by_joint"].pop(M.JOINTS[0])
    elif mutation == "joint_target":
        dynamic["by_joint"][M.JOINTS[0]]["centered"]["per_user"]["u00"]["sums"]["vary"] += 1
    elif mutation == "threshold":
        analysis["dynamic_threshold"] = .25
    result = R.aggregate(cells,references,assessments=assessments,common_gates=gates)
    assert result["status"] == "technical_failure" and reason in result["reason"]
    assert not result["complete"] and not result["contrasts"] and not result["candidacy"]["candidate"]


def test_all_quality_gates_required_before_any_scientific_result(matrix):
    cells,references,assessments,gates,_,_,_ = matrix
    for gate in R.COMMON_GATES:
        result = R.aggregate(cells,references,assessments=assessments,common_gates={**gates,gate:False})
        assert result["status"] == "quality_gate_not_passed"
        assert not result["contrasts"] and result["scientific_verdict"] is None


def test_undefined_bootstrap_replicates_are_not_discarded():
    result = R._interval(1.,np.array([0.,np.nan,2.]),99)
    assert not result["evaluable"] and result["ci"] is None
    assert result["undefined_replicates"] == 1 and result["replicates_discarded"] == 0


@pytest.mark.parametrize("value,passed", [
    (.7630564046643697, True),
    (float(np.nextafter(.7630564046643697, np.inf)), False),
    (.77, False), (None, False), (float("nan"), False), (float("inf"), False),
])
def test_absolute_candidate_threshold_is_explicit_even_when_all_contrasts_pass(value, passed):
    contrasts, assessments, gates = decisions()
    result = R.candidacy(contrasts, assessments, gates, candidate_e_dyn=value)
    assert result["candidate"] is passed
    assert result["absolute_E_dyn"]["passed"] is passed
    assert result["absolute_E_dyn"]["maximum"] == .7630564046643697
    assert result["absolute_E_dyn"]["value"] == (value if value is not None and np.isfinite(value) else None)
