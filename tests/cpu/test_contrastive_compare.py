"""Synthetic checks for the preregistered three-arm scientific comparison."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_compare as C


def inputs():
    classification = {}
    for arm in C.ARMS:
        classification[arm] = {}
        for ds in C.DATASETS:
            y = np.tile(np.array(["a", "b", "a", "b"]), 4)
            pred = y.copy()
            pred[[0, 4, 8, 12]] = "b"
            classification[arm][ds] = {"row": np.arange(16), "y": y, "pred": pred,
                                       "s": np.repeat(np.array(["u0", "u1", "u2", "u3"]), 4),
                                       "fold": np.repeat(np.arange(4), 4)}
    pose = {}
    for arm in C.ARMS:
        rows = {f"u{i}": {"n_windows": 2, "n_window_joints": 40, "n_valid_elements": 400,
                          "sums": {"vary": 10., "centered": 8.}} for i in range(4)}
        pose[arm] = {"complete": True, "arm": arm, "checkpoint_step": 19618, "which": "teacher",
                     "checkpoint_sha256": hashlib.sha256(arm.encode()).hexdigest(),
                     "fit": {"selection": {"boundary_unresolved": False}},
                     "analysis": {"support_sha256": "s"*64, "targets_sha256": "t"*64, "dynamic_threshold": .05,
                                  "groups": {"dynamic": {"n_selected_windows": 8, "centered": {"per_user": rows}}}}}
    return classification, pose


def audits():
    records = []
    for arm in C.ARMS:
        for task, datasets in C.TASKS.items():
            for ds in datasets:
                for fold in range(4):
                    for phase, grid in (("candidate", C.C_GRID), ("refit", [1.])):
                        for c in grid:
                            records.append({"arm": arm, "task": task, "dataset": ds, "fold": fold,
                                            "checkpoint_sha256": hashlib.sha256(arm.encode()).hexdigest(),
                                            "fit_phase": phase, "C": c, "n_iter": 5, "max_iter": 12000,
                                            "exception": None, "convergence_warnings": 0, "converged_by_criterion": True})
    return records


def test_identical_arms_zero_all_contrasts_and_preserve_pooled_metric():
    cl, po = inputs()
    r = C.compare(cl, po, n_boot=100, audit_records=audits())
    assert r["datasets"]["epn612"]["balanced_accuracy"] == dict.fromkeys(C.ARMS, .75)
    assert r["pose"]["E_dyn"] == dict.fromkeys(C.ARMS, .8)
    for pair in r["contrasts"].values():
        for endpoint in C.MARGINS:
            assert pair[endpoint]["benefit"] == 0
            assert pair[endpoint]["ci95"] == [0., 0.]
            assert pair[endpoint]["ci_simultaneous"] == [0., 0.]
    assert r["reading"] == "interpretation_withheld"  # reduced bootstrap is a test, not signed analysis


def test_preregistered_inference_and_head_to_head_do_not_force_a_winner():
    cl, po = inputs()
    # CN improves pose by10%, CR improves classification by25pp. A tradeoff.
    for row in po["CN"]["analysis"]["groups"]["dynamic"]["centered"]["per_user"].values():
        row["sums"]["centered"] *= .9
    for ds in C.DATASETS:
        cl["CR"][ds]["pred"] = cl["CR"][ds]["y"].copy()
    r = C.compare(cl, po, audit_records=audits())
    assert r["decision_valid"] and r["reading"] == "tradeoff_with_demonstrated_material_harm"
    assert r["contrasts"]["CN_minus_C0"]["E_dyn"]["useful_superiority"]
    assert r["contrasts"]["CR_minus_C0"]["p1"]["useful_superiority"]
    assert r["contrasts"]["CN_minus_CR"]["E_dyn"]["benefit"] == pytest.approx(10.)
    assert r["reverse_head_to_head"]["E_dyn"]["benefit"] == pytest.approx(-100/9)


def test_clear_pareto_improvement_supports_confirmation_only():
    cl, po = inputs()
    for row in po["CN"]["analysis"]["groups"]["dynamic"]["centered"]["per_user"].values():
        row["sums"]["centered"] *= .95
    r = C.compare(cl, po, audit_records=audits())
    assert r["reading"] == "CN_supported_for_confirmation"
    assert "no automatic" in r["scope"]
    assert r["bootstrap"]["family_size"] == 9
    assert r["bootstrap"]["nominal_simultaneous_confidence"] == pytest.approx(1-.05/9)


@pytest.mark.parametrize("field", ["row", "y", "s", "fold", "pred_nonfinite", "missing_dataset"])
def test_mismatched_oof_is_rejected(field):
    cl, po = inputs()
    if field == "missing_dataset":
        del cl["CN"]["epn612"]
    elif field == "pred_nonfinite":
        cl["CN"]["epn612"]["pred"] = np.full(16, np.nan)
    elif field == "row":
        cl["CN"]["epn612"][field][0] = 99
    elif field == "fold":
        cl["CN"]["epn612"][field][0] = 1
    else:
        cl["CN"]["epn612"][field][0] = "different"
    with pytest.raises(ValueError):
        C.compare(cl, po, n_boot=100)


@pytest.mark.parametrize("mutation", ["step", "teacher", "support", "target", "threshold", "counts", "denominator", "user", "negative_sse"])
def test_pose_pairing_guards(mutation):
    cl, po = inputs()
    cell = po["CR"]
    row = cell["analysis"]["groups"]["dynamic"]["centered"]["per_user"]["u0"]
    if mutation == "step": cell["checkpoint_step"] = 5000
    elif mutation == "teacher": cell["which"] = "student"
    elif mutation in ("support", "target"):
        cell["analysis"]["support_sha256" if mutation == "support" else "targets_sha256"] = "other"
    elif mutation == "threshold": cell["analysis"]["dynamic_threshold"] = .1
    elif mutation == "counts": row["n_windows"] += 1
    elif mutation == "denominator": row["sums"]["vary"] += 1
    elif mutation == "negative_sse": row["sums"]["centered"] = -1
    else: del cell["analysis"]["groups"]["dynamic"]["centered"]["per_user"]["u0"]
    with pytest.raises(ValueError):
        C.compare(cl, po, n_boot=100)


def test_unresolved_solver_withholds_reading_without_discarding_estimates():
    cl, po = inputs()
    a = audits()
    a[0]["n_iter"] = 12000
    a[0]["converged_by_criterion"] = False
    r = C.compare(cl, po, audit_records=a, n_boot=100)
    assert r["fit_audit"]["fits"] == 420 and not r["decision_valid"]
    assert r["contrasts"]["CN_minus_C0"]["p1"]["benefit"] == 0
    assert r["contrasts"]["CN_minus_C0"]["p1"]["useful_superiority"] is None
    a[0]["converged_by_criterion"] = True
    with pytest.raises(ValueError, match="flag"):
        C.verify_audits(a)


def test_audit_grid_cannot_hide_a_duplicate_or_missing_fit():
    a = audits()
    with pytest.raises(ValueError): C.verify_audits(a[:-1])
    a[-1] = copy.deepcopy(a[-2])
    with pytest.raises(ValueError): C.verify_audits(a)


def test_pose_and_classification_must_use_the_same_checkpoint():
    cl, po = inputs()
    a = audits()
    a[0]["checkpoint_sha256"] = "e"*64
    with pytest.raises(ValueError, match="checkpoint identity"):
        C.compare(cl, po, audit_records=a, n_boot=100)


def test_pooled_macro_recall_is_not_unweighted_subject_mean():
    cl, _ = inputs()
    # One person has many class-a mistakes, so an average of per-person BAcc
    # would give a different result than the historical pooled class counts.
    a = {"row": np.arange(8), "y": np.array([0,0,0,0,0,1,0,1]),
         "pred": np.array([1,1,1,1,1,1,0,1]), "s": np.array(["a"]*6+["b"]*2),
         "fold": np.array([0]*6+[1]*2)}
    for arm in C.ARMS:
        cl[arm] = {ds: copy.deepcopy(a) for ds in C.DATASETS}
    point, _, _ = C.classification_statistics(cl, 100, C.SEED)
    assert point["p1"][0] == pytest.approx((1/6+1)/2)
    assert point["p1"][0] != .75


def test_missing_classes_are_retained_in_draws():
    cl, _ = inputs()
    a = {"row": np.arange(4), "y": np.array([0,0,1,1]), "pred": np.array([0,0,1,1]),
         "s": np.array(["a","a","b","b"]), "fold": np.array([0,0,1,1])}
    for arm in C.ARMS:
        cl[arm] = {ds: copy.deepcopy(a) for ds in C.DATASETS}
    _, sample, report = C.classification_statistics(cl, 100, C.SEED)
    assert sample["p2"].shape == (100, 3) and np.all(sample["p2"] == 1)
    assert report["epn612"]["single_class_replicates"] > 0


def test_input_pin_hash_is_enforced(tmp_path):
    p = tmp_path/"a.json"
    p.write_text("{}")
    pin = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    assert C.read_pin(pin, tmp_path)[1] == b"{}"
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash"):
        C.read_pin(pin, tmp_path)


def test_complete_pinned_manifest_roundtrip_and_corrupted_oof(tmp_path):
    cl, po = inputs()
    all_records = audits()
    manifest = {"protocol": C.PROTOCOL, "step": 19618, "which": "teacher", "pose": {}, "classification": {}}

    def pin(p):
        return {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    for arm in C.ARMS:
        p = tmp_path/f"{arm}.pose.json"
        p.write_text(json.dumps(po[arm]))
        manifest["pose"][arm] = pin(p)
        manifest["classification"][arm] = {}
        for task, datasets in C.TASKS.items():
            p = tmp_path/f"{arm}.{task}.npz"
            np.savez(p, **{f"{ds}_{k}": v for ds in datasets for k, v in cl[arm][ds].items()})
            a = tmp_path/f"{arm}.{task}.audit.jsonl"
            a.write_text("\n".join(json.dumps(r) for r in all_records if r["arm"] == arm and r["task"] == task)+"\n")
            manifest["classification"][arm][task] = {"oof": pin(p), "audit": pin(a)}
    mp = tmp_path/"manifest.json"
    mp.write_text(json.dumps(manifest))
    result = C.from_manifest(mp)
    assert result["decision_valid"] and result["reading"] == "practically_equivalent_on_decision_endpoints"
    assert result["fit_audit"]["fits"] == 420
    assert result["input_manifest_sha256"] == hashlib.sha256(mp.read_bytes()).hexdigest()
    (tmp_path/"CN.p2.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        C.from_manifest(mp)


def synthetic_endpoints(n=1000):
    points = {e: np.array([.5,.5,.5]) for e in (*C.PRIMARY_ENDPOINTS, *C.DESCRIPTIVE_ENDPOINTS)}
    samples = {e: np.tile(v, (n,1)) for e,v in points.items()}
    return points, samples


def test_bidirectional_useful_contrasts_do_not_prefer_cn_by_order():
    points, samples = synthetic_endpoints()
    # Both recipes improve one classification endpoint by .6pp. The direct
    # loss on the other endpoint is .6pp, inside its 1pp NI margin.
    points["p1"][1] += .006
    points["epn612"][2] += .006
    samples = {e: np.tile(v, (1000,1)) for e,v in points.items()}
    contrasts = {a+"_minus_"+b: C.summarize_pair(a,b,points,samples) for a,b in C.PAIRS}
    reverse = C.summarize_pair("CR","CN",points,samples)
    assert contrasts["CN_minus_CR"]["useful_without_material_loss"]
    assert reverse["useful_without_material_loss"]
    assert C.scientific_reading(contrasts,reverse) == "bidirectional_benefit_tradeoff_no_winner"


def test_numeric_label_dtypes_compatible_but_string_numeric_mix_rejected():
    cl,_ = inputs()
    a = cl["C0"]["epn612"]
    a["y"] = np.arange(16,dtype=np.int64)%2
    a["pred"] = a["y"].astype(np.float32)
    assert np.array_equal(C.validate_oof(a)["y"], C.validate_oof(a)["pred"])
    a["pred"] = a["y"].astype(str)
    with pytest.raises(ValueError,match="label families"):
        C.validate_oof(a)


def test_pose_bootstrap_is_relative_per_replica_and_reverse_is_nonlinear():
    points,samples = synthetic_endpoints()
    points["E_dyn"] = np.array([.8,.4,.6])
    samples["E_dyn"][:,0] = np.linspace(.4,1.2,1000)
    samples["E_dyn"][:,1] = np.linspace(.2,.8,1000)
    forward = C.summarize_pair("CN","C0",points,samples)["E_dyn"]
    expected = 100*(1-samples["E_dyn"][:,1]/samples["E_dyn"][:,0])
    assert forward["benefit"] == 50
    assert forward["ci95"] == pytest.approx(np.quantile(expected,[.025,.975]))
    assert forward["ci_simultaneous"] == pytest.approx(np.quantile(expected,[.05/18,1-.05/18]))
    reverse = C.summarize_pair("C0","CN",points,samples)["E_dyn"]
    assert reverse["benefit"] == -100
    assert reverse["ci95"] == pytest.approx(np.quantile(100*(1-samples["E_dyn"][:,0]/samples["E_dyn"][:,1]),[.025,.975]))
    assert reverse["ci95"] != [-forward["ci95"][1],-forward["ci95"][0]]


def test_monte_carlo_boundary_withholds_only_affected_flag():
    points,samples = synthetic_endpoints(100000)
    draw = np.linspace(-2,2,100000)
    draw += -.9999-np.quantile(draw,.05/18)
    samples["p1"][:,1] += draw/100
    pair = C.summarize_pair("CN","C0",points,samples)
    p = pair["p1"]
    assert p["ci_simultaneous"][0] > -1
    assert p["monte_carlo"]["lower"]["mcse"] > 0
    assert p["boundary"]["noninferiority"]
    assert not p["noninferiority"]
    assert pair["epn612"]["noninferiority"]
    # Estimates and errors are deterministic without an extra random draw.
    assert C.summarize_pair("CN","C0",points,samples) == pair


def test_epn_decision_does_not_rename_or_silently_redefine_historical_p2():
    cl,po = inputs()
    cl["CN"]["uci_emg"]["pred"] = cl["CN"]["uci_emg"]["y"].copy()
    result = C.compare(cl,po,audit_records=audits())
    assert result["decision_endpoints"] == ["E_dyn","p1","epn612"]
    assert result["historical_P2"]["definition"] == "0.5 EPN612 + 0.5 UCI"
    pair = result["contrasts"]["CN_minus_C0"]
    assert pair["epn612"]["benefit"] == 0
    assert pair["descriptive"]["p2"]["benefit"] == 12.5
    assert pair["descriptive"]["uci_emg"]["benefit"] == 25
    assert not pair["useful_without_material_loss"]
    assert result["bootstrap"]["family_size"] == 9


def test_explicit_profiles_distinguish_uncertain_ni_from_material_harm():
    points,samples = synthetic_endpoints()
    samples["p1"][:,1] += np.linspace(-.04,.04,1000)
    uncertain = C.summarize_pair("CN","C0",points,samples)
    assert "p1" in uncertain["profile"]["unresolved_noninferiority_endpoints"]
    assert not uncertain["profile"]["material_harm_endpoints"]
    assert not uncertain["profile"]["no_practical_advantage_supported"]
    points["p1"][1] -= .02
    samples["p1"][:,1] = points["p1"][1]
    harmful = C.summarize_pair("CN","C0",points,samples)
    assert harmful["profile"]["material_harm_endpoints"] == ["p1"]
    assert harmful["profile"]["label"] == "demonstrated_material_harm"


def test_common_benefit_without_direct_difference_has_explicit_profile():
    points,samples = synthetic_endpoints()
    points["E_dyn"][1:] *= .95
    samples = {e: np.tile(v,(1000,1)) for e,v in points.items()}
    contrasts = {a+"_minus_"+b: C.summarize_pair(a,b,points,samples) for a,b in C.PAIRS}
    reverse = C.summarize_pair("CR","CN",points,samples)
    assert C.scientific_reading(contrasts,reverse) == "common_benefit_without_resolved_recipe_difference"
