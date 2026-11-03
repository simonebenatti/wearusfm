"""Synthetic tests only: no scientific caches, remote actions, or training jobs."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("tokenizer_probe_test", ROOT / "scripts/tokenizer_probe.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def sample(strings=False, width=6, subjects=None):
    subs = subjects or [f"s{i:02}" for i in range(8)]
    s = np.repeat(np.asarray(subs), 6)
    y = np.tile(np.asarray([0, 1, 0, 1, 0, 1]), len(subs))
    x = np.zeros((len(y), width), dtype=np.float32)
    x[:, :min(width, 6)] = np.random.default_rng(19).normal(size=(len(y), min(width, 6)))
    x[:, 0] += 1.5*y
    if strings:
        y = np.where(y == 0, "relaxed", "gesture")
    return x, y, s


@pytest.mark.parametrize("strings", [False, True])
def test_fit_matches_legacy_numerics_at_12000_and_train_only_scaler(strings):
    x, y, s = sample(strings)
    roles, audit = M.CP.fold_roles(sorted(set(s))), M.AuditLog()
    result = M.fit_fold(x, y, s, roles, 0, audit)
    mt, mv, mh = (np.isin(s, roles[0][k]) for k in ("train", "val", "held"))
    scaler = StandardScaler().fit(x[mt])
    xt, xv, xh = (scaler.transform(x[m]) for m in (mt, mv, mh))
    fits = {c: LogisticRegression(C=c, max_iter=12000, class_weight="balanced", random_state=0).fit(xt, y[mt]) for c in M.C_GRID}
    scores = {c: balanced_accuracy_score(y[mv], f.predict(xv)) for c, f in fits.items()}
    best = max(M.C_GRID, key=lambda c: (scores[c], -c))
    refit = LogisticRegression(C=best, max_iter=12000, class_weight="balanced", random_state=0).fit(xt, y[mt])
    assert result["report"]["c"] == best
    assert np.array_equal(result["pred"], refit.predict(xh))
    assert result["report"]["scaler_sha256"] == M.DA.fingerprint(scaler.mean_, scaler.scale_)
    assert len(audit) == 7 and all(r["max_iter"] == 12000 and r["dtype"] == "float32" for r in audit)
    # Alter held-out features: the C, training scaler and selected fit remain unchanged.
    altered = x.copy()
    altered[mh] += 1000
    again = M.fit_fold(altered, y, s, roles, 0, M.AuditLog())
    assert again["report"]["c"] == best
    assert again["report"]["scaler_sha256"] == result["report"]["scaler_sha256"]


def test_audit_failure_is_persisted_before_reraise(tmp_path, monkeypatch):
    x, y, s = sample()
    def broken(self, x, y):
        raise RuntimeError("injected fit failure")
    monkeypatch.setattr(M.LogisticRegression, "fit", broken)
    path = tmp_path / "progress.jsonl"
    with pytest.raises(RuntimeError, match="injected"):
        M.fit_fold(x, y, s, M.CP.fold_roles(sorted(set(s))), 0, M.AuditLog(path))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1 and records[0]["exception"] == "RuntimeError: injected fit failure"
    assert records[0]["fit_phase"] == "candidate"


def test_convergence_warning_attributed_to_fit_without_dropping_candidate(monkeypatch):
    original = M.LogisticRegression.fit
    def warn_fit(self, x, y):
        import warnings
        value = original(self, x, y)
        if self.C == 100:
            warnings.warn("injected convergence warning", ConvergenceWarning)
        return value
    monkeypatch.setattr(M.LogisticRegression, "fit", warn_fit)
    x, y, s = sample()
    audit = M.AuditLog()
    with pytest.warns(ConvergenceWarning):
        r = M.fit_fold(x, y, s, M.CP.fold_roles(sorted(set(s))), 0, audit)
    assert len(r["report"]["val_scores"]) == 6
    records = [r for r in audit if r["convergence_warnings"]]
    assert records and all(r["C"] == 100 and r["anomalous_stop"] for r in records)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    pre = [f"s{i:02}" for i in range(8)]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"datasets": {d: {"pretraining": pre, "test": ["never-open"]} for d in M.TASKS["p1"]}}))
    monkeypatch.setattr(M, "SPLITS_PATH", splits)
    all_p2 = [f"p{i:02}" for i in range(24)]
    p2 = M.CP.p2_internal_subjects(all_p2)
    assert len(p2) >= 4
    datasets, caches = {}, {arm: {} for arm in M.ARMS}
    for ds in M.DATASETS:
        task = next(t for t in M.TASKS if ds in M.TASKS[t])
        subjects = pre if task == "p1" else p2
        datasets[ds] = {"task": task, "subjects": subjects, "roles_sha256": M.jhash(M.CP.fold_roles(subjects)),
                        "input_sha256": M.jhash({"dataset": ds}), "splits_sha256": M.sha(splits) if task == "p1" else M.CP.P2_SPLITS_SHA256,
                        "extraction_recipe": M.CR.recipe(task)}
        for i, arm in enumerate(M.ARMS):
            folder = tmp_path / arm / ds
            folder.mkdir(parents=True)
            cache, pp, rp = folder/"features.npz", folder/"features.npz.provenance.json", folder/"receipt.json"
            x, y, s = sample(ds == "epn612", 768, subjects)
            x[:, 1] *= 1+i*.05
            checkpoint = str(i+1)*64
            np.savez(cache, x_pool=x, y=y, s=s, subjects=np.asarray(subjects), all_subjects=np.asarray(all_p2),
                     protocol=np.asarray(M.CR.PROTOCOL), feature_set=np.asarray("channel_readout"), task=np.asarray(task),
                     encoder_dim=np.asarray(384), checkpoint_sha256=np.asarray(checkpoint), code_sha=np.asarray("c"*40),
                     input_sha256=np.asarray(datasets[ds]["input_sha256"]), splits_sha256=np.asarray(datasets[ds]["splits_sha256"]),
                     extraction_recipe=np.asarray(json.dumps(datasets[ds]["extraction_recipe"])))
            common = {"arm": arm, "which": "teacher", "checkpoint_step": 19618, "checkpoint_sha256": checkpoint, "postload_recipe_passed": True}
            prov = {**common, "cache_sha256": M.sha(cache), **{k: datasets[ds][k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")}}
            pp.write_text(json.dumps(prov))
            rp.write_text(json.dumps({**common, "complete": True, "checkpoint_unchanged": True, "code_sha": "c"*40, "plan_sha256": "d"*64,
                                     "outputs": {ds: {"path": str(cache), "sha256": M.sha(cache), "provenance_sha256": M.sha(pp)}}}))
            caches[arm][ds] = {"path": str(cache), "sha256": M.sha(cache), "provenance_path": str(pp), "provenance_sha256": M.sha(pp),
                               "receipt_path": str(rp), "receipt_sha256": M.sha(rp), "checkpoint_sha256": checkpoint,
                               "extraction_code_sha": "c"*40, "extraction_plan_sha256": "d"*64,
                               **{k: datasets[ds][k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")}}
    return M.make_plan(caches, datasets)


def test_cache_pins_and_teacher_guard_before_fit(plan):
    x, y, s = M.load_cache(plan, "T100-G", "epn612")
    assert x.dtype == np.float32 and x.shape[1] == 768 and y.dtype.kind == "U"
    bad = copy.deepcopy(plan)
    bad["caches"]["T100-G"]["epn612"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="hash mismatch"):
        M.load_cache(bad, "T100-G", "epn612")
    pin = plan["caches"]["T100-G"]["epn612"]
    receipt = json.loads(Path(pin["receipt_path"]).read_text())
    receipt["postload_recipe_passed"] = False
    Path(pin["receipt_path"]).write_text(json.dumps(receipt))
    pin["receipt_sha256"] = M.sha(pin["receipt_path"])
    with pytest.raises(ValueError, match="loaded model"):
        M.load_cache(plan, "T100-G", "epn612")


def test_plan_refuses_missing_arm_and_wrong_roles(plan):
    bad = copy.deepcopy(plan)
    del bad["caches"]["T25-L"]
    with pytest.raises(ValueError, match="all three"):
        M.validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["datasets"]["epn612"]["roles_sha256"] = "0"*64
    with pytest.raises(ValueError, match="roles"):
        M.validate_plan(bad)


def test_shards_aggregate_exactly_and_reject_missing_or_duplicate_folds(plan, tmp_path):
    identity = {"code_sha": "e"*40, "plan_sha256": "f"*64}
    paths = []
    for name, folds in (("first", [0, 1]), ("second", [2, 3])):
        path = tmp_path / name
        M.run_shard(plan, "T25-L", "p2", list(M.TASKS["p2"]), folds, path, identity)
        paths.append(path/"shard.json")
    report = M.aggregate(plan, "T25-L", "p2", paths[::-1], tmp_path/"aggregate.json", identity)
    assert report["audit_summary"]["fits"] == 56
    assert report["balanced_accuracy"] == .5*sum(v["balanced_accuracy"] for v in report["datasets"].values())
    with np.load(report["oof"], allow_pickle=False) as z:
        for ds in M.TASKS["p2"]:
            x, y, s = M.load_cache(plan, "T25-L", ds)
            roles = M.CP.fold_roles(plan["datasets"][ds]["subjects"])
            ref = [M.fit_fold(x, y, s, roles, f, M.AuditLog()) for f in range(4)]
            assert np.array_equal(z[ds+"_pred"], np.concatenate([r["pred"] for r in ref]))
    with pytest.raises(ValueError, match="missing dataset/fold"):
        M.aggregate(plan, "T25-L", "p2", paths[:1], tmp_path/"missing.json", identity)
    with pytest.raises(ValueError, match="invalid aggregation"):
        M.aggregate(plan, "T25-L", "p2", [paths[0], paths[0]], tmp_path/"duplicate.json", identity)
    duplicate = tmp_path/"duplicate-shard.json"
    duplicate.write_text(paths[0].read_text())
    with pytest.raises(ValueError, match="duplicate fold"):
        M.aggregate(plan, "T25-L", "p2", [paths[0], duplicate, paths[1]], tmp_path/"duplicate-fold.json", identity)
    assert not (tmp_path/"missing.json").exists()


def test_failed_shard_never_has_complete_manifest(plan, tmp_path, monkeypatch):
    def broken(self, x, y):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(M.LogisticRegression, "fit", broken)
    out = tmp_path/"failed-shard"
    with pytest.raises(RuntimeError):
        M.run_shard(plan, "T25-L", "p2", ["epn612"], [0], out, {"code_sha": "e"*40, "plan_sha256": "f"*64})
    assert not (out/"shard.json").exists()
    assert json.loads((out/"failed.json").read_text())["complete"] is False
    assert len((out/"audit_progress.jsonl").read_text().splitlines()) == 1


def comparison_cells():
    cells = {a: {} for a in M.ARMS}
    for ds in M.DATASETS:
        _, y, s = sample(ds == "epn612")
        for i, arm in enumerate(M.ARMS):
            pred = y.copy()
            if i == 0:
                pred[::3] = np.roll(y, 1)[::3]
            order = np.arange(len(y))[::-1] if i == 2 else np.arange(len(y))
            cells[arm][ds] = {"row": np.arange(len(y))[order], "y": y[order], "pred": pred[order], "s": s[order],
                               "fold": (np.arange(len(y))//6 % 4)[order]}
    return cells


def test_paired_bootstrap_row_permutation_exact_zero_and_frozen_weights():
    result = M.compare_arrays(comparison_cells(), n_boot=200)
    for ds in M.DATASETS:
        zero = result["datasets"][ds]["contrasts"]["T100-G_minus_T100-L"]
        assert zero["delta_pp"] == 0 and zero["ci95_pp"] == [0, 0]
        assert result["datasets"][ds]["cells"]["T100-G"]["balanced_accuracy"] == 1
    for task in M.TASKS:
        for arm in M.ARMS:
            expected = sum(w*result["datasets"][ds]["cells"][arm]["balanced_accuracy"] for ds, w in zip(M.TASKS[task], M.WEIGHTS[task]))
            assert result["tasks"][task]["cells"][arm]["balanced_accuracy"] == pytest.approx(expected)


def test_same_imperfect_predictions_have_zero_delta_with_nonzero_marginal_uncertainty():
    cells = {arm: {} for arm in M.ARMS}
    for arm in M.ARMS:
        for ds in M.DATASETS:
            cells[arm][ds] = {"row": np.arange(4), "y": np.asarray([0, 1, 0, 1]),
                               "pred": np.asarray([0, 1, 1, 0]), "s": np.asarray(["u0", "u0", "u1", "u1"]),
                               "fold": np.asarray([0, 0, 1, 1])}
    result = M.compare_arrays(cells, n_boot=200)
    for ds in M.DATASETS:
        block = result["datasets"][ds]
        assert block["cells"]["T25-L"]["ci95_percent"] == [0, 100]
        for value in block["contrasts"].values():
            assert value["delta_pp"] == 0 and value["ci95_pp"] == [0, 0]
    for task in M.TASKS:
        assert all(value["ci95_pp"] == [0, 0] for value in result["tasks"][task]["contrasts"].values())


@pytest.mark.parametrize("failure", ["row", "target", "subject", "fold", "missing"])
def test_compare_refuses_misalignment(failure):
    cells = comparison_cells()
    ds = cells["T100-L"]["epn612"]
    if failure == "row":
        ds["row"][0] = ds["row"][1]
    elif failure == "target":
        ds["y"][0] = "another-label"
    elif failure == "subject":
        ds["s"][0] = "s99"
    elif failure == "fold":
        ds["fold"][0] = 3
    else:
        del cells["T100-L"]["epn612"]
    with pytest.raises(ValueError):
        M.compare_arrays(cells, n_boot=20)


def test_predictions_absent_from_oof_targets_are_errors_not_invalid_labels():
    cells = comparison_cells()
    for arm in M.ARMS:
        a = cells[arm]["epn612"]
        a["pred"] = np.full(len(a["y"]), "training-only-class")
    result = M.compare_arrays(cells, n_boot=50)
    assert all(v["balanced_accuracy"] == 0 for v in result["datasets"]["epn612"]["cells"].values())


def test_single_class_bootstrap_is_retained_and_counted():
    cells = comparison_cells()
    for arm in M.ARMS:
        for ds, a in cells[arm].items():
            order = np.argsort(a["row"])
            for k in a:
                a[k] = a[k][order][:4]
            a["s"] = np.asarray(["s0", "s0", "s1", "s1"])
            a["y"] = np.asarray(["a", "a", "b", "b"])
            a["pred"] = a["y"].copy()
    result = M.compare_arrays(cells, n_boot=200)
    for ds in M.DATASETS:
        assert result["datasets"][ds]["single_class_replicates"] > 0
        assert result["datasets"][ds]["contrasts"]["T100-G_minus_T25-L"]["ci95_pp"] == [0, 0]


def test_nonfinite_numeric_targets_and_predictions_rejected():
    cells = comparison_cells()
    a = cells["T25-L"]["ninapro_db2"]
    a["pred"] = a["pred"].astype(float)
    a["pred"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        M.compare_arrays(cells, n_boot=20)


def test_solver_unresolved_preserves_numbers_and_withholds_all_ni_decisions():
    result = M.compare_arrays(comparison_cells(), n_boot=50)
    before = copy.deepcopy(result)
    result["fit_audit"] = {"T25-L/p1": {"converged_by_criterion": True}, "T100-G/p2": {"converged_by_criterion": False}}
    M.apply_solver_reading(result)
    assert result["reading_status"] == "solver_unresolved" and result["decision_valid"] is False
    for group in ("datasets", "tasks"):
        for name, block in result[group].items():
            for key, contrast in block["contrasts"].items():
                if group == "tasks":
                    assert contrast["noninferiority_1pp"] == "unresolved_solver"
                else:
                    assert "noninferiority_1pp" not in contrast
                assert contrast["delta_pp"] == before[group][name]["contrasts"][key]["delta_pp"]
                assert contrast["ci95_pp"] == before[group][name]["contrasts"][key]["ci95_pp"]


def test_completion_guard_precedes_complete_shard_publication(plan, tmp_path, monkeypatch):
    def changed(plan, identity):
        raise ValueError("source changed before completion")
    monkeypatch.setattr(M, "completion_identity", changed)
    out = tmp_path/"source-changed"
    with pytest.raises(ValueError, match="source changed"):
        M.run_shard(plan, "T25-L", "p2", ["epn612"], [0], out, {"code_sha": "e"*40, "plan_sha256": "f"*64})
    assert not (out/"shard.json").exists()
    assert json.loads((out/"failed.json").read_text())["complete"] is False
    assert len((out/"audit_progress.jsonl").read_text().splitlines()) == 7
