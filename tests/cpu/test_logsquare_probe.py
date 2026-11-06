"""Synthetic CPU guards for the candidate-only LQ adapter; no scientific fits/jobs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from frozen_campaign_support import run_frozen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import logsquare_probe as M
spec = importlib.util.spec_from_file_location("lq_legacy_probe_test_helpers", ROOT / "tests/cpu/test_tokenizer_probe.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def test_fit_function_is_shared_and_predictions_are_numerically_identical():
    assert M.fit_fold is M.T.fit_fold and M.AuditLog is M.T.AuditLog
    x, y, s = L.sample(strings=True)
    roles = M.CP.fold_roles(sorted(set(s)))
    got = M.fit_fold(x, y, s, roles, 0, M.AuditLog())
    expected = L.M.fit_fold(x, y, s, roles, 0, L.M.AuditLog())
    assert got["report"] == expected["report"]
    assert all(np.array_equal(got[k], expected[k]) for k in ("row", "y", "pred", "s", "fold"))


def cells():
    c = L.comparison_cells()
    c[M.CANDIDATE] = copy.deepcopy(c["T100-L"])
    return c


def test_four_cell_pairing_zeros_signs_and_shared_historical_draws():
    result = M.compare_arrays(cells())
    old = M.T.compare_arrays(L.comparison_cells())
    for ds in M.DATASETS:
        block = result["datasets"][ds]
        assert block["draws_sha256"] == old["datasets"][ds]["draws_sha256"]
        assert block["contrasts"]["T100-LQ_minus_T100-L"]["delta_pp"] == 0
        assert block["contrasts"]["T100-LQ_minus_T100-G"]["ci95_pp"] == [0, 0]
        assert block["contrasts"]["T100-LQ_minus_T25-L"]["delta_pp"] > 0
        for arm in M.HISTORICAL_ARMS:
            assert block["cells"][arm] == old["datasets"][ds]["cells"][arm]
    assert result["datasets"]["epn612"]["role"] == "primary_classification"
    assert result["n_bootstrap"] == 2000 and result["seed"] == 20261102


@pytest.mark.parametrize("failure", ["missing", "row", "y", "s", "fold", "pred_nonfinite"])
def test_pairing_refuses_missing_or_mismatched_oof(failure):
    c = cells()
    a = c[M.CANDIDATE]["epn612"]
    if failure == "missing":
        del c[M.CANDIDATE]["epn612"]
    elif failure == "row":
        a["row"][0] = a["row"][1]
    elif failure == "pred_nonfinite":
        a["pred"] = np.full(len(a["pred"]), np.nan)
    elif failure in ("y", "s"):
        a[failure][0] = "x"
    else:
        a["fold"][0] = 3
    with pytest.raises(ValueError):
        M.compare_arrays(c, n_boot=20)


def test_unresolved_solver_retains_estimates_but_removes_decision():
    result = M.compare_arrays(cells(), n_boot=30)
    result["fit_audit"] = {a+"/"+t: {"converged_by_criterion": True} for a in M.ARMS for t in M.TASKS}
    result["fit_audit"][M.CANDIDATE+"/p2"]["converged_by_criterion"] = False
    result = M.apply_solver_reading(result)
    assert result["reading_status"] == "solver_unresolved" and result["decision_valid"] is False
    assert result["tasks"]["p2"]["contrasts"]["T100-LQ_minus_T100-L"]["noninferiority_1pp"] == "unresolved_solver"
    assert result["datasets"]["epn612"]["contrasts"]["T100-LQ_minus_T100-L"]["delta_pp"] == 0


def test_preparatory_plan_is_closed_and_real_sources_pinned(tmp_path):
    relative='results/infra/logsquare_evaluation_20261104/probe_plan_19618.json'
    plan=json.loads((ROOT/relative).read_text())
    assert plan['caches'][M.CANDIDATE] is None and plan['release'] is None
    with pytest.raises(ValueError,match='not released'):M.validate_plan(plan)
    plan['status']='APPROVED_FOR_PROBES'
    # A historical plan must still reject the different merged source tree.
    with pytest.raises(ValueError,match='recipe/source'):M.validate_plan(plan)
    observed=run_frozen(tmp_path,'63eba9a80e729a2ec989b2b63b07e21cd24a5152',
        ['src','scripts','results/infra','splits'],
        """import json
import logsquare_probe as M
plan=json.loads((root/sys.argv[2]).read_text())
assert plan['sources']==M.source_hashes()
try:M.validate_plan(plan)
except ValueError as e:assert 'not released' in str(e)
else:raise AssertionError('unreviewed plan was executable')
plan['status']='APPROVED_FOR_PROBES'
try:M.validate_plan(plan)
except ValueError as e:assert 'missing candidate' in str(e)
else:raise AssertionError('missing candidate was accepted')
print(json.dumps({'closed':True,'historical_sources_verified':True}))
""",relative)
    assert observed=={'closed':True,'historical_sources_verified':True}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lq-probe")
    mp = pytest.MonkeyPatch()
    old = L.plan.__wrapped__(tmp, mp)
    mp.setattr(M.T, "SPLITS_PATH", L.M.SPLITS_PATH)
    oldpath = tmp / "original-plan.json"
    oldpath.write_text(json.dumps(old))
    identity = {"code_sha": "e"*40, "plan_sha256": M.sha(oldpath), "plan_path": str(oldpath)}
    # Completion identity's git checks are separately guarded by the production CLI.
    # Synthetic reports use no plan_path until after fits; provenance remains exact.
    fitid = {k:v for k,v in identity.items() if k != "plan_path"}
    refs = {}
    for arm in M.HISTORICAL_ARMS:
        for task in M.TASKS:
            directory = tmp / (arm+"-"+task+"-shard")
            M.T.run_shard(old, arm, task, list(M.TASKS[task]), list(range(4)), directory, fitid)
            path = tmp / (arm+"-"+task+".json")
            r = M.T.aggregate(old, arm, task, [directory/"shard.json"], path, fitid)
            r["plan_path"] = str(oldpath)
            path.write_text(json.dumps(r))
            refs[arm+"/"+task] = {"path": str(path), "sha256": M.sha(path), **{k:r[k] for k in ("code_sha", "plan_sha256", "plan_content_sha256", "numeric_sha256", "plan_path")}}
    candidate = {}
    for ds in M.DATASETS:
        oldpin = old["caches"]["T100-L"][ds]
        folder = tmp / "candidate" / ds
        folder.mkdir(parents=True)
        cp, pp, rp = folder/"features.npz", folder/"provenance.json", folder/"receipt.json"
        with np.load(oldpin["path"], allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}
        arrays["checkpoint_sha256"] = np.asarray("9"*64)
        np.savez(cp, **arrays)
        common = {"arm":M.CANDIDATE, "which":"teacher", "checkpoint_step":19618, "checkpoint_sha256":"9"*64, "postload_recipe_passed":True, "protocol":"tokenizer-logsquare-evaluation-20261104-v1", "model_config":{"tokenizer_logsquare":True,"tokenizer_logsquare_eps":.001,"window_ms":100.,"patch_ms":25.,"tokenizer_activation":"identity","dim":384,"k_latents":64}}
        prov = {**common, "cache_sha256":M.sha(cp), **{k:oldpin[k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")}}
        pp.write_text(json.dumps(prov))
        rec = {**common, "complete":True, "checkpoint_unchanged":True, "code_sha":"c"*40, "plan_sha256":"d"*64, "outputs":{ds:{"path":str(cp),"sha256":M.sha(cp),"provenance_sha256":M.sha(pp)}}}
        rp.write_text(json.dumps(rec))
        candidate[ds] = {**oldpin, "path":str(cp), "sha256":M.sha(cp), "provenance_path":str(pp), "provenance_sha256":M.sha(pp), "receipt_path":str(rp), "receipt_sha256":M.sha(rp), "checkpoint_sha256":"9"*64}
    rp = ROOT / "results/infra/logsquare_evaluation_20261104/resources_probe.json"
    policy = json.loads(rp.read_text())
    policy["status"] = "APPROVED_FOR_PROBES"
    policy["numeric_sha256"] = M.jhash(M.numeric_context())
    rp = tmp / "resources.json"
    rp.write_text(json.dumps(policy))
    plan = {"protocol":M.PROTOCOL,"status":"APPROVED_FOR_PROBES","step":19618,"recipe":M.recipe(),"sources":M.source_hashes(),"datasets":old["datasets"],"caches":{M.CANDIDATE:candidate},"historical_plan":{"path":str(oldpath),"sha256":M.sha(oldpath)},"historical_reports":refs,"expected_numeric_sha256":M.jhash(M.numeric_context()),"resources":{"path":str(rp),"sha256":M.sha(rp)}}
    M.validate_plan(plan)
    yield tmp, plan, fitid
    mp.undo()


def test_full_candidate_pipeline_imports_original_provenance_and_exact_oof(experiment):
    tmp, plan, historical_identity = experiment
    identity = {"code_sha":"f"*40,"plan_sha256":"a"*64}
    shards = {}
    for key, (task, datasets, folds) in M.SHARDS.items():
        path = tmp / ("candidate-shard-"+key)
        r = M.run_shard(plan, M.CANDIDATE, task, datasets, folds, path, identity)
        assert r["arm"] == M.CANDIDATE and r["audit_summary"]["converged_by_criterion"]
        shards[key] = path/"shard.json"
    reports = []
    for task in M.TASKS:
        path = tmp / ("candidate-aggregate-"+task+".json")
        subset = [shards[k] for k,v in M.SHARDS.items() if v[0] == task]
        r = M.aggregate(plan, M.CANDIDATE, task, subset, path, identity)
        r["plan_path"] = str(tmp/"candidate-plan.json")
        path.write_text(json.dumps(r))
        reports.append(path)
        assert r["audit_summary"]["fits"] == 28*len(M.TASKS[task])
    result = M.compare(plan, reports, tmp/"comparison.json", identity)
    assert result["decision_valid"] is True
    assert len(result["original_provenance"]) == 8
    assert result["original_provenance"]["T100-L/p2"]["code_sha"] == historical_identity["code_sha"]
    assert result["original_provenance"][M.CANDIDATE+"/p2"]["code_sha"] == identity["code_sha"]
    assert result["datasets"]["epn612"]["contrasts"]["T100-LQ_minus_T100-L"]["ci95_pp"] == [0, 0]
    with pytest.raises(ValueError, match="two distinct"):
        M.compare(plan, reports[:1], tmp/"missing.json", identity)
    with pytest.raises(ValueError, match="source identity"):
        M.compare(plan, reports, tmp/"sourcebad.json", {**identity,"code_sha":"0"*40})


@pytest.mark.parametrize("failure", ["recipe", "sources", "numeric", "cache", "history", "historical_source", "cpu"])
def test_plan_guards(experiment, failure):
    _, original, _ = experiment
    plan = copy.deepcopy(original)
    if failure == "recipe": plan["recipe"]["pca"] = True
    elif failure == "sources": plan["sources"]["scripts/tokenizer_probe.py"] = "0"*64
    elif failure == "numeric": plan["expected_numeric_sha256"] = "0"*64
    elif failure == "cache": plan["caches"][M.CANDIDATE]["epn612"] = None
    elif failure == "history": del plan["historical_reports"]["T25-L/p2"]
    elif failure == "historical_source": plan["historical_reports"]["T25-L/p2"]["plan_content_sha256"] = "0"*64
    else: plan["resources"]["sha256"] = "0"*64
    with pytest.raises(ValueError): M.validate_plan(plan)


@pytest.mark.parametrize("field", ["sha256", "code_sha", "numeric_sha256", "plan_path", "plan_sha256"])
def test_historical_import_rejects_bad_provenance(experiment, field):
    _, original, _ = experiment
    plan = copy.deepcopy(original)
    plan["historical_reports"]["T25-L/p2"][field] = "0"*(40 if field == "code_sha" else 64)
    with pytest.raises(ValueError): M.read_historical(plan, "T25-L/p2")


def test_candidate_numeric_mismatch_fails_before_output(experiment, monkeypatch):
    tmp, plan, identity = experiment
    monkeypatch.setattr(M, "numeric_context", lambda:{"different":"runtime"})
    with pytest.raises(ValueError, match="numeric contexts"):
        M.run_shard(plan, M.CANDIDATE, *M.SHARDS["p2"], tmp/"forbidden-fit", identity)
    assert not (tmp/"forbidden-fit").exists()


def test_launcher_has_single_candidate_no_submission_and_explicit_resource_caps():
    path = ROOT / "scripts/slurm/logsquare_probe.sbatch"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    text = path.read_text()
    assert 'T25-L|T100-L|T100-G' not in text and '--arm' not in text
    assert 'SECONDS_CAP=3600' in text and 'SECONDS_CAP=1800' in text and 'SECONDS_CAP=1200' in text
    assert 'logsquare_probe.py' in text and 'code_dir_setup' in text
    assert not any(line.strip().startswith(('sbatch ', 'srun ', 'ssh ')) for line in text.splitlines())

@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("lq_launcher_helpers", ROOT/"tests/cpu/test_tokenizer_probe_launcher.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(helper, "LAUNCHER", ROOT/"scripts/slurm/logsquare_probe.sbatch")
    return helper.launch.__wrapped__(tmp_path)


@pytest.mark.parametrize("key,minutes", [("db2-01", 60), ("db2-23", 60), ("db3", 30), ("db6", 30), ("p2", 60)])
def test_launcher_registered_dispatch(launcher, key, minutes):
    run, _, _ = launcher
    r = run(args=(key,), minutes=minutes)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["argv"][-2:] == ["--shard", key]
    assert payload["threads"] == "8" and payload["cuda"] == ""


@pytest.mark.parametrize("overrides", [
    {"args":("db3",), "minutes":60}, {"cpus":16},
    {"env_overrides":{"SLURM_MEM_PER_NODE":"32768"}},
    {"env_overrides":{"SLURM_JOB_PARTITION":"boost_usr_prod"}},
    {"args":("unregistered",)}, {"stage":"benchmark"},
    {"digest":"b"*64}, {"plan":"../../../absent.json"},
])
def test_launcher_rejects_before_science(launcher, overrides):
    run, _, _ = launcher
    r = run(**{"args":("p2",), **overrides})
    assert r.returncode != 0 and '"argv"' not in r.stdout


@pytest.mark.parametrize("stage,task,count,minutes", [("aggregate","p1",4,20), ("aggregate","p2",1,20), ("compare",None,2,30)])
def test_launcher_reports_dispatch(launcher, stage, task, count, minutes):
    run, tmp, _ = launcher
    paths = [tmp/f"report{i}.json" for i in range(count)]
    for path in paths: path.write_text("{}")
    args = [task,*paths] if task else paths
    r = run(stage=stage,args=args,cpus=2,minutes=minutes)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["argv"][-count:] == list(map(str, paths)) and payload["threads"] == "2"

@pytest.mark.parametrize("field,value", [("tokenizer_logsquare",False), ("tokenizer_logsquare_eps",.01), ("window_ms",25), ("tokenizer_activation","gelu"), ("protocol","historical")])
def test_candidate_requires_real_lq_loaded_config(experiment, field, value):
    _, original, _ = experiment
    plan = copy.deepcopy(original)
    pin = plan["caches"][M.CANDIDATE]["epn612"]
    path = Path(pin["receipt_path"])
    saved = path.read_bytes()
    obj = json.loads(saved)
    if field == "protocol": obj[field] = value
    else: obj["model_config"][field] = value
    try:
        path.write_text(json.dumps(obj))
        pin["receipt_sha256"] = M.sha(path)
        with pytest.raises(ValueError, match="LQ"):
            M.load_cache(plan, M.CANDIDATE, "epn612")
    finally:
        path.write_bytes(saved)
