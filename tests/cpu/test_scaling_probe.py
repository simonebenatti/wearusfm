"""CPU regression tests for dimension-aware scaling probes; no cluster actions."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_probe as M


def pin(path):
    return {"path": str(path), "sha256": M.sha(path)}


def save_json(path, value):
    Path(path).write_text(json.dumps(value))
    return pin(path)


def sample(width=768, subjects=None, strings=False):
    subjects = subjects or [f"s{i:02}" for i in range(8)]
    s = np.repeat(subjects, 6)
    y = np.tile([0, 1, 0, 1, 0, 1], len(subjects))
    x = np.zeros((len(y), width), np.float32)
    x[:, :6] = np.random.default_rng(9).normal(size=(len(y), 6))
    x[:, 0] += 2 * y
    return x, np.where(y == 0, "noGesture", "fist") if strings else y, s


@pytest.fixture(scope="module")
def projection(tmp_path_factory):
    return M.SP.save(tmp_path_factory.mktemp("projection") / "fixed")


@pytest.fixture
def plan(tmp_path, projection):
    contract = pin(ROOT / "results/infra/scaling_300m_20261104/training_source_contract.json")
    context = M.ST.check_source_contract(contract["path"])
    p1_subjects = [f"s{i:02}" for i in range(8)]
    all_p2 = [f"p{i:02}" for i in range(24)]
    p2_subjects = M.T.CP.p2_internal_subjects(all_p2)
    splits = save_json(tmp_path / "splits.json", {"datasets": {ds: {"pretraining": p1_subjects, "test": ["never-open"]} for ds in M.TASKS["p1"]}})
    caches, datasets, references = {a: {} for a in M.ARMS}, {}, {}
    for ds in M.DATASETS:
        task = next(t for t in M.TASKS if ds in M.TASKS[t])
        subjects = p1_subjects if task == "p1" else p2_subjects
        datasets[ds] = {"task": task, "subjects": subjects, "roles_sha256": M.jhash(M.T.CP.fold_roles(subjects)),
                        "input_sha256": M.jhash({"dataset": ds}), "splits_sha256": splits["sha256"] if task == "p1" else M.T.CP.P2_SPLITS_SHA256,
                        "extraction_recipe": M.T.CR.recipe(task)}
        x, y, s = sample(768, subjects, ds == "epn612")
        data = {"x_pool": x, "y": y, "s": s, "subjects": np.asarray(subjects), "all_subjects": np.asarray(all_p2),
                "checkpoint": np.asarray("/checkpoint.pt"), "checkpoint_sha256": np.asarray("9"*64),
                "input_sha256": np.asarray(datasets[ds]["input_sha256"]), "splits_sha256": np.asarray(datasets[ds]["splits_sha256"]),
                "extraction_recipe": np.asarray(json.dumps(M.E.C.recipe(task))), "encoder_dim": np.asarray(384), "channels": np.asarray(8),
                "protocol": np.asarray(M.E.C.PROTOCOL), "calibration_protocol": np.asarray(M.T.CP.PROTOCOL), "task": np.asarray(task),
                "feature_set": np.asarray(M.E.C.CR.FEATURE_SET), "code_sha": np.asarray("c"*40),
                "montage_signature": np.asarray(json.dumps([[0, i] for i in range(8)]))}
        ref = tmp_path / (ds + "-reference.npz")
        np.savez(ref, **data)
        references[ds] = pin(ref)
        for arm in M.ARMS:
            folder = tmp_path / arm / ds
            folder.mkdir(parents=True)
            cache, pp, rp = folder/"cache.npz", folder/"cache.provenance.json", folder/"receipt.json"
            cp = context["contract"]["reference_checkpoint_sha256"] if arm == M.ARMS[0] else "2"*64
            data["x_pool"] = sample(2 * M.E.dimension(arm), subjects, ds == "epn612")[0]
            data["checkpoint_sha256"], data["encoder_dim"] = np.asarray(cp), np.asarray(M.E.dimension(arm))
            np.savez(cache, **data)
            details = M.E.verify_classification_cache(cache, references[ds], ds, "/checkpoint.pt", cp, "c"*40, splits, arm)
            cfg = M.ST.R.config_to_dict(context["config_30m" if arm == M.ARMS[0] else "config_300m"])["model"]
            common = {"protocol": M.E.PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": 5000,
                      "checkpoint_sha256": cp, "postload_recipe_passed": True, "code_sha": "c"*40,
                      "plan_sha256": "d"*64, "training_source_contract_sha256": contract["sha256"], "model_config": cfg}
            save_json(pp, {**common, **details, "cache": str(cache), "cache_sha256": M.sha(cache)})
            save_json(rp, {**common, "complete": True, "stage": "extract-classification", "checkpoint_unchanged": True,
                          "outputs": {ds: {"path": str(cache), "sha256": M.sha(cache), "provenance_path": str(pp), "provenance_sha256": M.sha(pp)}}})
            caches[arm][ds] = {"path": str(cache), "sha256": M.sha(cache), "provenance_path": str(pp), "provenance_sha256": M.sha(pp),
                              "receipt_path": str(rp), "receipt_sha256": M.sha(rp), "checkpoint_path": "/checkpoint.pt", "checkpoint_sha256": cp,
                              "extraction_code_sha": "c"*40, "extraction_plan_sha256": "d"*64,
                              **{k: datasets[ds][k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")}}
    return M.make_plan(caches, datasets, references, splits, contract, projection)


@pytest.mark.parametrize("cell", M.CELLS)
def test_real_cache_dimensions_projection_and_text_labels(plan, cell):
    x, y, subjects = M.cell_features(plan, cell, "epn612")
    assert x.shape == (len(y), M.width(cell)) and x.dtype == np.float32
    assert y.dtype.kind == "U" and set(y) == {"noGesture", "fist"}
    assert sorted(set(subjects)) == plan["datasets"]["epn612"]["subjects"]
    if cell.endswith("matched"):
        original, _, _ = M.load_cache(plan, M.ARMS[1], "epn612")
        p = M.SP.load(plan["projection"])
        assert np.array_equal(x, np.concatenate([original[:, :896] @ p.backbone, original[:, 896:] @ p.local], axis=1))


@pytest.mark.parametrize("width,strings", [(768, False), (768, True), (1792, True)])
def test_fit_is_historical_primitive_with_same_grid_and_train_only_scaler(width, strings):
    assert M.fit_fold is M.T.fit_fold and M.labels is M.T.labels
    x, y, subjects = sample(width, strings=strings)
    roles = M.T.CP.fold_roles(sorted(set(subjects)))
    result = M.fit_fold(x, y, subjects, roles, 0, M.AuditLog())
    historical = M.T.fit_fold(x, y, subjects, roles, 0, M.T.AuditLog())
    assert np.array_equal(result["pred"], historical["pred"])
    assert result["report"] == historical["report"]
    held = np.isin(subjects, roles[0]["held"])
    x[held] += 1000
    changed = M.fit_fold(x, y, subjects, roles, 0, M.AuditLog())
    assert changed["report"]["c"] == result["report"]["c"]
    assert changed["report"]["scaler_sha256"] == result["report"]["scaler_sha256"]
    assert M.C_GRID == (1e-3, 1e-2, 1e-1, 1., 10., 100.) and M.MAX_ITER == 12000


@pytest.mark.parametrize("mutation", ["dose", "arm", "roles", "sources", "cp30", "split", "projection"])
def test_plan_fails_closed(plan, mutation):
    bad = copy.deepcopy(plan)
    if mutation == "dose": bad["step"] = 19618
    if mutation == "arm": del bad["caches"][M.ARMS[0]]
    if mutation == "roles": bad["datasets"]["epn612"]["roles_sha256"] = "a"*64
    if mutation == "sources": bad["sources"]["scripts/scaling_probe.py"] = "a"*64
    if mutation == "cp30":
        for pin in bad["caches"][M.ARMS[0]].values(): pin["checkpoint_sha256"] = "a"*64
    if mutation == "split": bad["datasets"]["ninapro_db2"]["subjects"] += ["never-open"]
    if mutation == "projection": bad["projection"]["sha256"] = "a"*64
    with pytest.raises(ValueError): M.validate_plan(bad)


@pytest.mark.parametrize("where,key,value", [("receipt", "checkpoint_step", 19618), ("receipt", "complete", False),
    ("receipt", "which", "student"), ("receipt", "stage", "validate"), ("provenance", "postload_recipe_passed", False),
    ("provenance", "training_source_contract_sha256", "a"*64), ("provenance", "pool_shape", [1, 1792])])
def test_real_provenance_guard_rejects_rehashed_false_claims(plan, where, key, value):
    p = plan["caches"][M.ARMS[1]]["epn612"]
    file = Path(p[where+"_path"])
    record = json.loads(file.read_text())
    record[key] = value
    save_json(file, record)
    p[where+"_sha256"] = M.sha(file)
    if where == "provenance":
        receipt = json.loads(Path(p["receipt_path"]).read_text())
        receipt["outputs"]["epn612"]["provenance_sha256"] = p["provenance_sha256"]
        save_json(p["receipt_path"], receipt)
        p["receipt_sha256"] = M.sha(p["receipt_path"])
    with pytest.raises(ValueError): M.load_cache(plan, M.ARMS[1], "epn612")


@pytest.fixture
def identity():
    return {"code_sha": "e"*40, "plan_sha256": "f"*64}


def test_benchmark_observes_only_train_and_no_prediction(plan, tmp_path, identity, monkeypatch):
    cell, ds = "T100-G-300M-matched", "epn612"
    x, y, subjects = M.cell_features(plan, cell, ds)
    roles = M.T.CP.fold_roles(plan["datasets"][ds]["subjects"])
    mask = np.isin(subjects, roles[0]["train"])
    expected_x = M.T.StandardScaler().fit_transform(x[mask])
    fits = []
    original = M.T.LogisticRegression.fit
    def observed(self, a, b):
        fits.append((a.copy(), b.copy()))
        return original(self, a, b)
    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark must not predict validation or held-out")
    monkeypatch.setattr(M.T.LogisticRegression, "fit", observed)
    monkeypatch.setattr(M.T.LogisticRegression, "predict", forbidden)
    result = M.benchmark(plan, cell, ds, tmp_path/"benchmark", identity)
    assert len(fits) == 6 and all(np.array_equal(a, expected_x) and np.array_equal(b, y[mask]) for a, b in fits)
    assert result["train_only"] and not result["selection_performed"] and result["audit_summary"]["fits"] == 6
    assert all(r["selected"] is None and "val_bacc" not in r for r in result["fits"])


def make_budget(plan, tmp_path, identity, cell, datasets, folds):
    benchmarks = {}
    for ds in datasets:
        target = tmp_path / (ds + "-benchmark")
        M.benchmark(plan, cell, ds, target, identity)
        benchmarks[ds] = pin(target/"benchmark.json")
    allocation = {"cell": cell, "task": "p2", "datasets": datasets, "folds": folds,
                  "cpus": 1, "walltime_seconds": 10000, "memory_gib": 50}
    budget = {"protocol": M.RESOURCE_PROTOCOL, "complete": True, "plan_content_sha256": M.jhash(plan),
              "code_sha": identity["code_sha"], "caps": {"cpus": 4, "walltime_seconds": 11000, "memory_gib": 60},
              "benchmarks": {cell: benchmarks}, "shards": [allocation]}
    return save_json(tmp_path/"resources.json", budget)


def test_budget_needs_matching_train_benchmark_and_published_caps(plan, tmp_path, identity):
    cell, datasets, folds = "T100-G-30M-natural", ["epn612"], [0]
    p = make_budget(plan, tmp_path, identity, cell, datasets, folds)
    b = M.validate_resources(plan, cell, "p2", datasets, folds, p, identity)
    assert b["estimated_seconds_with_margin"] > 120
    budget = json.loads(Path(p["path"]).read_text())
    budget["caps"]["cpus"] = 0
    bad = save_json(tmp_path/"bad.json", budget)
    with pytest.raises(ValueError, match="positive published"): M.validate_resources(plan, cell, "p2", datasets, folds, bad, identity)
    budget["caps"]["cpus"] = 4
    budget["shards"][0]["walltime_seconds"] = 1
    bad = save_json(tmp_path/"bad.json", budget)
    with pytest.raises(ValueError, match="measured benchmark"): M.validate_resources(plan, cell, "p2", datasets, folds, bad, identity)


@pytest.fixture
def completed_shard(plan, tmp_path, identity):
    cell, datasets, folds = "T100-G-30M-natural", list(M.TASKS["p2"]), [0, 1, 2, 3]
    budget = make_budget(plan, tmp_path, identity, cell, datasets, folds)
    target = tmp_path / "shard"
    M.run_shard(plan, cell, "p2", datasets, folds, target, identity, budget)
    return target / "shard.json"


def test_shard_aggregate_matches_historical_fits(plan, completed_shard, tmp_path, identity):
    cell = "T100-G-30M-natural"
    report = M.aggregate(plan, cell, "p2", [completed_shard], tmp_path/"aggregate.json", identity)
    assert report["audit_summary"]["fits"] == 56
    assert report["aliases"] == M.ALIASES
    with np.load(report["oof"], allow_pickle=False) as z:
        for ds in M.TASKS["p2"]:
            x, y, s = M.load_cache(plan, M.ARMS[0], ds)
            roles = M.T.CP.fold_roles(plan["datasets"][ds]["subjects"])
            ref = [M.T.fit_fold(x, y, s, roles, f, M.T.AuditLog()) for f in range(4)]
            assert np.array_equal(z[ds+"_pred"], np.concatenate([r["pred"] for r in ref]))
    assert report["balanced_accuracy"] == .5 * sum(v["balanced_accuracy"] for v in report["datasets"].values())


@pytest.mark.parametrize("mutation", ["truncated_audit", "missing_fold", "wrong_label", "wrong_cell", "wrong_width", "wrong_projection", "truncated_oof", "wrong_convergence"])
def test_aggregate_refuses_corrupt_or_partial_outputs(plan, completed_shard, tmp_path, identity, mutation):
    report = json.loads(completed_shard.read_text())
    if mutation == "wrong_cell": report["cell"] = "T100-G-300M-natural"
    if mutation == "wrong_projection": report["projection_sha256"] = "a"*64
    if mutation in ("truncated_audit", "wrong_width", "wrong_convergence"):
        rows = [json.loads(s) for s in Path(report["audit"]).read_text().splitlines()]
        if mutation == "truncated_audit": rows.pop()
        if mutation == "wrong_width": rows[0]["n_features"] = 1792
        if mutation == "wrong_convergence": rows[0]["n_iter"] = 12000
        Path(report["audit"]).write_text("".join(json.dumps(r)+"\n" for r in rows))
        report["audit_sha256"] = M.sha(report["audit"])
        report["audit_summary"] = M.audit_summary(rows)
    if mutation in ("missing_fold", "wrong_label", "truncated_oof"):
        with np.load(report["oof"], allow_pickle=False) as z: values = {k: z[k] for k in z.files}
        if mutation == "missing_fold":
            selected = values["epn612_fold"] != 3
            values = {k: v[selected] if k.startswith("epn612_") else v for k, v in values.items()}
            report["datasets"]["epn612"]["folds"] = report["datasets"]["epn612"]["folds"][:3]
        if mutation == "wrong_label": values["epn612_y"][0] = "invalid"
        if mutation == "truncated_oof": del values["epn612_pred"]
        np.savez(report["oof"], **values)
        report["oof_sha256"] = M.sha(report["oof"])
    save_json(completed_shard, report)
    with pytest.raises(ValueError): M.aggregate(plan, "T100-G-30M-natural", "p2", [completed_shard], tmp_path/"bad.json", identity)


def test_matched_baseline_alias_cannot_be_fitted():
    with pytest.raises(ValueError, match="alias"): M.width("T100-G-30M-matched")


def test_bootstrap_exact_historical_draws_and_metrics_without_runtime_arm_remapping():
    cells = {c: {} for c in M.CELLS}
    historical = {a: {} for a in M.T.ARMS}
    for ds in M.DATASETS:
        _, y, s = sample(strings=ds == "epn612")
        for index, (cell, arm) in enumerate(zip(M.CELLS, M.T.ARMS)):
            pred = y.copy()
            pred[::(index+2)] = y[1::1][:len(pred[::(index+2)])]
            a = {"row": np.arange(len(y)), "y": y, "pred": pred, "s": s, "fold": np.arange(len(y)) % 4}
            cells[cell][ds] = a
            historical[arm][ds] = a
    actual = M.compare_arrays(cells)
    old = M.T.compare_arrays(historical)
    for ds in M.DATASETS:
        assert actual["datasets"][ds]["draws_sha256"] == old["datasets"][ds]["draws_sha256"]
        for cell, arm in zip(M.CELLS, M.T.ARMS):
            assert actual["datasets"][ds]["cells"][cell] == old["datasets"][ds]["cells"][arm]
    for task in M.TASKS:
        assert actual["tasks"][task]["contrasts"]["natural_300m_minus_30m"]["ci95_pp"] == old["tasks"][task]["contrasts"]["T100-L_minus_T25-L"]["ci95_pp"]
    bad = copy.deepcopy(cells)
    bad["T100-G-300M-matched"]["epn612"]["s"] = bad["T100-G-300M-matched"]["epn612"]["s"].copy()
    bad["T100-G-300M-matched"]["epn612"]["s"][0] = "wrong"
    with pytest.raises(ValueError, match="pairing"): M.compare_arrays(bad)


@pytest.mark.parametrize("cell", ["T100-G-300M-natural", "T100-G-300M-matched"])
def test_300m_end_to_end_shards_keep_natural_and_matched_identity(plan, tmp_path, identity, cell):
    datasets, folds = list(M.TASKS["p2"]), [0, 1, 2, 3]
    resource = make_budget(plan, tmp_path, identity, cell, datasets, folds)
    M.run_shard(plan, cell, "p2", datasets, folds, tmp_path/"shard", identity, resource)
    r = M.aggregate(plan, cell, "p2", [tmp_path/"shard/shard.json"], tmp_path/"aggregate.json", identity)
    assert r["cell"] == cell and r["arm"] == M.ARMS[1] and r["audit_summary"]["fits"] == 56
    _, _, audit = M.read_completed(tmp_path/"aggregate.json", "aggregate", plan)
    assert all(f["n_features"] == M.width(cell) for f in audit)


def test_failed_fit_keeps_progress_and_never_publishes_complete(plan, tmp_path, identity, monkeypatch):
    def broken(self, x, y):
        raise RuntimeError("injected optimizer failure")
    monkeypatch.setattr(M.T.LogisticRegression, "fit", broken)
    with pytest.raises(RuntimeError, match="injected"):
        M.benchmark(plan, "T100-G-30M-natural", "epn612", tmp_path/"failure", identity)
    assert not (tmp_path/"failure/benchmark.json").exists()
    assert json.loads((tmp_path/"failure/failed.json").read_text())["complete"] is False
    records = [json.loads(r) for r in (tmp_path/"failure/audit_progress.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["exception"].endswith("injected optimizer failure")
