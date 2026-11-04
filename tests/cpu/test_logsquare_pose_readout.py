"""Oracles for candidate-only ridge, provenance and four-cell paired bootstrap."""
import copy
import json
from pathlib import Path
import subprocess
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import logsquare_pose_readout as L
from test_tokenizer_pose_readout import data, cell


def four_cells(equal=False):
    d = data()
    factors = (0.9, 0.8, 0.6, 0.9) if not equal else (0.8,)*4
    result = {}
    for arm, factor in zip(L.ARMS, factors):
        result[arm] = cell(arm, d, factor)
        result[arm]["fit"] = {"selection": {"boundary_unresolved": False}}
    return result


def plan(step=5000):
    return L.Q.read_json(ROOT/f"results/infra/logsquare_evaluation_20261104/pose_readout_plan_{step}.json")


def test_four_cell_bootstrap_exactly_matches_historical_draws_and_arithmetic():
    cells = four_cells()
    old = L.Q.paired_comparison({a: cells[a] for a in L.Q.ARMS}, 5000)
    actual = L.paired_comparison(cells, 5000)
    for group in ("dynamic", "all", "static"):
        assert actual[group]["draws_sha256"] == old[group]["draws_sha256"]
        for arm in L.Q.ARMS:
            assert actual[group]["cell_ratio_ci95"][arm] == old[group]["cell_ratio_ci95"][arm]
        for control in ("T100-L", "T25-L"):
            expected = old[group]["contrasts"]["T100-G_minus_"+control]
            got = actual[group]["contrasts"][L.ARM+"_minus_"+control]
            for key in ("delta_ratio", "ci95", "relative_error_reduction", "signal_2percent_and_ci_below_zero", "per_user"):
                assert got[key] == expected[key]
        assert actual[group]["contrasts"][L.ARM+"_minus_T100-L"]["delta_ratio"] < 0
        assert actual[group]["contrasts"][L.ARM+"_minus_T100-G"]["delta_ratio"] == 0


def test_identical_predictions_zero_contrast_and_no_signal():
    for group in L.paired_comparison(four_cells(equal=True), 5000).values():
        for contrast in group["contrasts"].values():
            assert contrast["delta_ratio"] == 0
            assert contrast["ci95"] == [0, 0]
            assert not contrast["signal_2percent_and_ci_below_zero"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "incomplete", "dose", "which", "support", "targets", "users", "count", "denominator", "selected", "unresolved"])
def test_cell_grid_and_support_refuse_silent_intersection(mutation):
    cells = four_cells()
    c = cells[L.ARM]
    row = c["analysis"]["groups"]["dynamic"]["centered"]["per_user"]
    if mutation == "missing": del cells["T100-G"]
    elif mutation == "extra": cells["unknown"] = copy.deepcopy(c)
    elif mutation == "incomplete": c["complete"] = False
    elif mutation == "dose": c["checkpoint_step"] = 19618
    elif mutation == "which": c["which"] = "student"
    elif mutation == "support": c["analysis"]["support_sha256"] = "different"
    elif mutation == "targets": c["analysis"]["targets_sha256"] = {"different": True}
    elif mutation == "users": row.pop("a")
    elif mutation == "count": row["a"]["n_valid_elements"] += 1
    elif mutation == "denominator": row["a"]["sums"]["vary"] += 1
    elif mutation == "selected": c["analysis"]["groups"]["dynamic"]["n_selected_windows"] += 1
    elif mutation == "unresolved": c["fit"]["selection"]["boundary_unresolved"] = True
    with pytest.raises(ValueError): L.paired_comparison(cells, 5000)


def test_candidate_only_fit_is_bit_identical_to_historical_numerics(tmp_path, monkeypatch):
    train, val = data(), data(seed=91, users=("v", "w"))
    expected_w, expected_b, expected_fit = L.Q.fit_reader(train)
    expected_prediction = L.Q.predict(val["lat"], val["lengths"], expected_w, expected_b)
    calls = []
    original = L.Q.fit_reader
    def fit(d, folds):
        calls.append("fit")
        return original(d, folds)
    def read(pin):
        calls.append(pin["path"])
        if pin["path"] == "val": assert "fit" in calls
        return copy.deepcopy(train if pin["path"] == "train" else val)
    monkeypatch.setattr(L.Q, "read_data", read)
    monkeypatch.setattr(L.Q, "fit_reader", fit)
    p = {"step":5000,"fold_users":[["a","c"],["b","d"]],"caches":{"train":{"path":"train","checkpoint_sha256":"1"*64},"val":{"path":"val"}}}
    report = L.run_candidate(p, tmp_path)
    assert calls == ["train", "fit", "val"] and set(report["cells"]) == {L.ARM}
    got = report["cells"][L.ARM]
    with np.load(got["coefficients"]["path"]) as z:
        np.testing.assert_array_equal(z["weights"], expected_w)
        np.testing.assert_array_equal(z["bias"], expected_b)
    np.testing.assert_array_equal(np.load(got["predictions"]["path"]), expected_prediction)
    assert got["fit"]["selection"] == expected_fit["selection"]
    assert got["analysis"] == L.Q.summarize(val, expected_prediction)


def test_benchmark_validation_never_opens_val(monkeypatch):
    called = []
    p = {"fold_users":[["a"],["b"]],"references":{"train":{"split":"train"},"val":{"split":"val"}},"caches":{"train":{},"val":{}}}
    def reference(pin):
        called.append(pin["split"])
        assert pin["split"] == "train"
        return {"users":["a","b"],"states":{}}
    def artifact(pin, split, plan, ref):
        called.append("cache/"+split)
        assert split == "train"
        return {"states":{}}
    monkeypatch.setattr(L.Q,"validate_reference",reference)
    monkeypatch.setattr(L,"validate_artifact",artifact)
    L.validate_inputs(p, train_only=True)
    assert called == ["train", "cache/train"]


@pytest.mark.parametrize("step", [5000,19618])
def test_preparatory_plans_preserve_history_and_fail_closed(step):
    p = plan(step)
    old = L.validate_plan(p)
    assert p["references"] == old["references"] and p["fold_users"] == old["fold_users"]
    assert all(v is None for pin in p["caches"].values() for v in pin.values())
    assert p["fit_release"] is None
    with pytest.raises(ValueError): L.validate_plan(p, runtime=True)
    report = L.load_historical(p,{})
    assert report["runtime"]["code_sha"] == p["historical"]["code_sha"]
    assert report["runtime"]["sources"] == L.Q.source_hashes()


@pytest.mark.parametrize("mutation", ["source","recipe","fold","historypin","historycode","context"])
def test_historical_or_source_mismatch_fails(mutation):
    p = plan()
    if mutation == "source": p["sources"]["scripts/pose_e3_b.py"] = "0"*64
    elif mutation == "recipe": p["recipe"]["offset"] = 5
    elif mutation == "fold": p["fold_users"][0][0] = "new"
    elif mutation == "historypin": p["historical"]["plan"]["sha256"] = "0"*64
    elif mutation == "historycode": p["historical"]["code_sha"] = "0"*40
    else: p["historical"]["numeric_context"]["versions"]["numpy"] = "new"
    with pytest.raises(ValueError):
        L.validate_plan(p)
        L.load_historical(p,{})


def test_plan_hash_excludes_only_late_fit_receipts_and_release_state():
    p = plan()
    first = L.evaluation_identity(p)
    p["status"] = "APPROVED_FOR_EVALUATION"
    p["fit_release"] = {"receipt":"future"}
    assert L.evaluation_identity(p) == first
    p["caches"]["train"]["sha256"] = "0"*64
    assert L.evaluation_identity(p) != first


def test_timing_proxy_one_candidate_and_gate_no_receipt():
    assert L.required_ridge_seconds(10,3,4) == 300
    with pytest.raises(ValueError): L.required_ridge_seconds(1,float("nan"),1)
    with pytest.raises(ValueError,match="not read/released"):
        L.fit_gate(plan(),{}, {})


def make_gate(tmp_path):
    p = plan()
    context = copy.deepcopy(p["historical"]["numeric_context"])
    identity = {**context,"code_sha":"c"*40,"sources":L.source_hashes()}
    release = {"status":"BENCHMARK_READ_AND_ACCEPTED","required_seconds":300}
    for stage in ("validate","benchmark"):
        report={"protocol":L.PROTOCOL,"complete":True,"stage":stage,"step":5000,"recipe":L.Q.recipe(),
            "status":"validated" if stage=="validate" else "train_only_timing",
            "evaluation_sha256":L.evaluation_identity(p),"runtime":identity,"slurm_job_id":"123"+stage,
            "input_pins":p["caches"],"reference_pins":p["references"],
            "timings":{"statistics_seconds":3.,"three_eigensystems_seconds":4.}}
        path=tmp_path/(stage+".json");path.write_text(json.dumps(report))
        release[stage]={"report":{"path":str(path),"sha256":L.Q.D.file_sha256(path)},
            "accounting":{"State":"COMPLETED","ExitCode":"0:0","JobIDRaw":"123"+stage,"ElapsedRaw":10,"AllocCPUS":L.CAPS[stage]["cpus"]}}
    p["fit_release"]=release
    return p,identity


def test_fit_gate_checks_receipts_context_cache_and_fixed_cap(tmp_path):
    p, identity = make_gate(tmp_path)
    assert L.fit_gate(p,{},identity)["required_seconds"] == 300
    p["fit_release"]["benchmark"]["accounting"]["ExitCode"]="1:0"
    with pytest.raises(ValueError,match="not completed"): L.fit_gate(p,{},identity)
    p["fit_release"]["benchmark"]["accounting"]["ExitCode"]="0:0"
    reportpin=p["fit_release"]["benchmark"]["report"]
    path=Path(reportpin["path"]);r=json.loads(path.read_text());r["input_pins"]["train"]["sha256"]="a"*64
    path.write_text(json.dumps(r));reportpin["sha256"]=L.Q.D.file_sha256(path)
    with pytest.raises(ValueError,match="input/recipe"): L.fit_gate(p,{},identity)
    r["input_pins"]=p["caches"];r["timings"]["statistics_seconds"]=1000
    path.write_text(json.dumps(r));reportpin["sha256"]=L.Q.D.file_sha256(path)
    p["fit_release"]["required_seconds"]=L.required_ridge_seconds(10,1000,4)
    with pytest.raises(ValueError,match="exceeds timing"): L.fit_gate(p,{},identity)


def test_launcher_syntax_and_no_submission():
    path=ROOT/"scripts/slurm/logsquare_pose_readout.sbatch"
    subprocess.run(["bash","-n",str(path)],check=True)
    text=path.read_text()
    assert not re.search(r"(?m)^\s*(?:sbatch|srun)\s", text)
    assert '--expected-plan-sha256 "$PLAN_SHA"' in text
    assert '"$SECS" -ne "$SECONDS_CAP"' in text
    assert 'CUDA_VISIBLE_DEVICES=""' in text


def lq_artifact(tmp_path):
    from test_tokenizer_pose_readout import artifact_fixture
    d, pin, p, ref, prov, receipt = artifact_fixture(tmp_path)
    prov["arm"] = receipt["arm"] = L.ARM
    receipt["protocol"] = prov["protocol"] = L.EXTRACTION_PROTOCOL
    prov["model_config"].update(tokenizer_activation="identity", tokenizer_logsquare=True, tokenizer_logsquare_eps=.001)
    def write():
        receipt["model_config"] = copy.deepcopy(prov["model_config"])
        Path(pin["provenance_path"]).write_text(json.dumps(prov))
        pin["provenance_sha256"] = L.Q.D.file_sha256(pin["provenance_path"])
        receipt["outputs"]["train"]["provenance_sha256"] = pin["provenance_sha256"]
        Path(pin["receipt_path"]).write_text(json.dumps(receipt))
        pin["receipt_sha256"] = L.Q.D.file_sha256(pin["receipt_path"])
    write()
    return pin,p,ref,prov,receipt,write


@pytest.mark.parametrize("mutation", [None,"flag","epsilon","protocol","complete","target"])
def test_candidate_artifact_requires_real_lq_identity_and_original_support(tmp_path,mutation):
    pin,p,ref,prov,receipt,write = lq_artifact(tmp_path)
    if mutation == "flag": prov["model_config"]["tokenizer_logsquare"] = False
    elif mutation == "epsilon": prov["model_config"]["tokenizer_logsquare_eps"] = .01
    elif mutation == "protocol": receipt["protocol"] = L.Q.PROTOCOL
    elif mutation == "complete": receipt["complete"] = False
    elif mutation == "target": ref["targets"]["central_mask"] = "0"*64
    write()
    if mutation is None:
        assert L.validate_artifact(pin,"train",p,ref)["arm"] == L.ARM
    else:
        with pytest.raises(ValueError): L.validate_artifact(pin,"train",p,ref)


def test_predicted_values_must_match_saved_sufficient_statistics_not_only_file_hash(tmp_path):
    d = data(users=("a","b"),windows=2,patches=24,dim=384,joints=20)
    prediction = d["target"][:,6:-6].astype(np.float64)*.8
    coeffpath=tmp_path/"weights.npz";predpath=tmp_path/"predictions.npy"
    weights,bias=np.zeros((384,20)),np.zeros(20)
    np.savez_compressed(coeffpath,weights=weights,bias=bias);np.save(predpath,prediction)
    c={"coefficients":{"path":str(coeffpath),"sha256":L.Q.D.file_sha256(coeffpath)},
       "predictions":{"path":str(predpath),"sha256":L.Q.D.file_sha256(predpath)},
       "fit":{"coefficients_sha256":L.Q.arrays_hash(weights,bias)},"analysis":L.Q.summarize(d,prediction)}
    L.verify_scientific_artifacts(c,d,{})
    prediction[0,0,0] += .2
    np.save(predpath,prediction);c["predictions"]["sha256"]=L.Q.D.file_sha256(predpath)
    with pytest.raises(ValueError,match="statistics differ"):
        L.verify_scientific_artifacts(c,d,{})


def diagnostic_fixture(tmp_path, step):
    import torch
    import logsquare_evaluation as E
    values = E.log_weight_diagnostics({k: {E.NEW_WEIGHT: torch.arange(384*64,dtype=torch.float32).reshape(384,64)*(1e-5 if k=="teacher" else 2e-5)} for k in ("student","teacher")})
    p={"step":step,"caches":{}}
    for split in ("train","val"):
        path=tmp_path/(split+".json")
        path.write_text(json.dumps({"protocol":L.EXTRACTION_PROTOCOL,"complete":True,"arm":L.ARM,"which":"teacher",
            "checkpoint_step":step,"checkpoint_sha256":"1"*64,"log_weight_diagnostics":values}))
        p["caches"][split]={"receipt_path":str(path),"receipt_sha256":L.Q.D.file_sha256(path),"checkpoint_sha256":"1"*64}
    return p,values


@pytest.mark.parametrize("step",[5000,19618])
def test_r1b_preserves_all_checkpoint_columns_and_families_at_each_endpoint(tmp_path,step):
    p,expected=diagnostic_fixture(tmp_path,step)
    got=L.load_log_weight_diagnostics(p,{})
    assert got["checkpoint_step"]==step and got["descriptive_only"] is True
    for kind in ("student","teacher"):
        assert got[kind]==expected[kind]
        assert len(got[kind]["column_l2_norms"])==64
        assert got[kind]["families"]["mlp"]["n"]==16
    assert set(got["source_receipts"])=={"train","val"}


@pytest.mark.parametrize("bad",["step","checkpoint","family","missing","different"])
def test_r1b_rejects_mixed_or_malformed_checkpoint_diagnostics(tmp_path,bad):
    p,_=diagnostic_fixture(tmp_path,5000)
    pin=p["caches"]["val"];path=Path(pin["receipt_path"]);r=json.loads(path.read_text())
    if bad=="step":r["checkpoint_step"]=19618
    elif bad=="checkpoint":r["checkpoint_sha256"]="2"*64
    elif bad=="family":r["log_weight_diagnostics"]["teacher"]["families"]["mlp"]["mean"]+=1
    elif bad=="missing":r.pop("log_weight_diagnostics")
    else:r["log_weight_diagnostics"]["student"]=copy.deepcopy(r["log_weight_diagnostics"]["teacher"])
    path.write_text(json.dumps(r));pin["receipt_sha256"]=L.Q.D.file_sha256(path)
    with pytest.raises(ValueError):L.load_log_weight_diagnostics(p,{})
