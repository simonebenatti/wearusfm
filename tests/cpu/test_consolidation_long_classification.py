"""Small CPU checks for long-matrix classification; no remote data or jobs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import consolidation_long_classification as M


def sample(width=8, strings=False):
    s = np.repeat([f"s{i:02}" for i in range(8)], 6)
    y = np.tile([0, 1, 0, 1, 0, 1], 8)
    x = np.random.default_rng(8).normal(size=(len(y), width)).astype(np.float32)
    x[:, 0] += 1.2*y
    return x, np.where(y, "gesture", "rest") if strings else y, s


def cache(tmp_path, dim=8, step=39237):
    x, y, s = sample(dim)
    subjects = sorted(set(s.tolist()))
    arrays = {"x_pre": x+np.float32(.3), "x_post": x,
              "x_pool": np.concatenate([x, -x], 1), "y": y, "s": s, "subjects": np.asarray(subjects)}
    meta = {"dataset": "ninapro_db2", "task": "p1", "protocol": M.PROTOCOL,
            "calibration_protocol": M.CP.PROTOCOL, "feature_set": "backbone_depth", "encoder_dim": dim,
            "encoder": "teacher", "input_sha256": "a"*64, "splits_sha256": "b"*64,
            "extraction_recipe": M.extraction_recipe("p1"), "historical_extraction_recipe": M.CR.recipe("p1"),
            "montage_signature": [[0, 0]], "channels": 1,
            "roles_sha256": M.jhash(M.CP.fold_roles(subjects)),
            "ordered_rows_sha256": M.DA.fingerprint(np.arange(len(y), dtype=np.int64), y, s)}
    old = tmp_path / f"historical{dim}-{step}.npz"
    np.savez(old, x_pool=arrays["x_pool"], y=y, s=s, subjects=np.asarray(subjects), input_sha256=np.asarray("a"*64),
             splits_sha256=np.asarray("b"*64), task=np.asarray("p1"), channels=np.asarray(1),
             extraction_recipe=np.asarray(json.dumps(M.CR.recipe("p1"))), montage_signature=np.asarray("[[0, 0]]"),
             checkpoint_sha256=np.asarray("c"*64))
    path = tmp_path / f"features{dim}-{step}.npz"
    identity = {"step": step, "checkpoint": "/external/teacher.pt", "checkpoint_sha256": "c"*64,
                "code_sha": "d"*40, "plan_sha256": "e"*64}
    pin = M.save_cache(path, {"arrays": arrays, "metadata": meta}, identity, {"path": str(old), "sha256": M.sha(old)})
    return {**pin, "subjects": subjects}, arrays, meta, old


@pytest.mark.parametrize("dim", [384, 896])
def test_cache_supports_general_d_and_pool_parity(tmp_path, dim):
    pin, arrays, _, _ = cache(tmp_path, dim=dim, step=19618)
    assert pin["historical_pool_parity"]
    for view in M.VIEWS:
        x, y, s, meta = M.load_cache(pin["path"], view, expected=pin)
        assert x.shape == (len(y), dim*(2 if view == "pool" else 1))
        assert np.array_equal(x, arrays["x_"+view])
        assert M.probe_recipe(view, dim)["width"] == x.shape[1]
        assert meta["roles_sha256"] == M.jhash(M.CP.fold_roles(pin["subjects"]))


def test_cache_rejects_input_row_nonfinite_and_view_mismatch(tmp_path):
    pin, arrays, meta, old = cache(tmp_path)
    ref = {"path": str(old), "sha256": M.sha(old)}
    bad = copy.deepcopy(arrays)
    bad["y"] = bad["y"][::-1]
    with pytest.raises(ValueError, match="ordered rows"):
        M.verify_historical_inputs({"arrays": bad, "metadata": meta}, ref)
    bad = copy.deepcopy(arrays)
    bad["x_post"][0, 0] += 1
    with pytest.raises(ValueError, match="first half"):
        M._check_arrays(bad, 8)
    bad["x_pre"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        M._check_arrays(bad, 8)
    with pytest.raises(ValueError, match="hash"):
        M.load_cache(pin["path"], "pre", expected={**pin, "sha256": "f"*64})
    with pytest.raises(ValueError, match="official split"):
        M.load_cache(pin["path"], "pre", expected={**pin, "subjects": ["wrong"]})


def test_p1_preserves_session_order_scale_keys_and_historical_fingerprint(monkeypatch):
    subjects = [f"s{i}" for i in range(8)]
    splits = {"datasets": {"ninapro_db2": {"pretraining": subjects, "test": ["never"]}}}
    calls, digest = [], __import__("hashlib").sha256()
    window = np.ones((2, 20, 1), dtype=np.float32)
    labels = np.array([0, 1])
    montage = {"groups": [{"channels": [{"sensor_coords": {"channel_index": 0}}]}]}
    monkeypatch.setattr(M.CR, "_root_of", lambda *args: Path("/fake"))
    monkeypatch.setattr(M.P1, "sessions_of", lambda root, ds, subject: [Path("a"), Path("b")])
    monkeypatch.setattr(M.P1, "session_windows", lambda session, ds, window_s:
                        SimpleNamespace(windows=window, labels=labels, session=session.name, fs=200., montage=montage))
    def fake(model, windows, keys, fs, montage, **kwargs):
        calls.append(keys.copy())
        post = np.ones((len(windows), 5), dtype=np.float32)
        return {"pre": -post, "post": post, "pool": np.concatenate([post, post], 1)}
    monkeypatch.setattr(M.BD, "extract_windows", fake)
    model = SimpleNamespace(cfg=SimpleNamespace(dim=5), eval=lambda: None)
    result = M.extract_dataset(model, "ninapro_db2", roots=["/fake"], splits=splits, splits_sha256="a"*64)
    for subject in subjects:
        for session in ("a", "b"):
            key = np.full(2, subject+"/"+session)
            digest.update(M.DA.fingerprint("pretraining", subject, session, window, labels, 200., montage, key).encode())
    assert result["metadata"]["input_sha256"] == digest.hexdigest()
    assert len(calls) == 16 and calls[0].tolist() == ["s0/a"]*2
    assert result["arrays"]["s"].tolist() == np.repeat(subjects, 4).tolist()


def test_p2_excludes_union_of_official_test_subjects_before_forward(monkeypatch):
    from wearusfm.harness import benchmarks as B, fm_features as FF
    subjects = np.repeat([f"p{i:02}" for i in range(24)], 2)
    w = np.ones((len(subjects), 60, 8), dtype=np.float32)
    y = np.tile([0, 1], 24)
    monkeypatch.setattr(B, "load_epn612", lambda roots: (w, y, subjects, 200.))
    calls = []
    def fake(model, windows, keys, fs, montage, **kwargs):
        calls.append(keys.copy())
        x = np.zeros((len(windows), 7), dtype=np.float32)
        return {"pre": x, "post": x, "pool": np.concatenate([x, x], 1)}
    monkeypatch.setattr(M.BD, "extract_windows", fake)
    model = SimpleNamespace(cfg=SimpleNamespace(dim=7), eval=lambda: None)
    result = M.extract_dataset(model, "epn612", epn_roots=["/fake"])
    internal = M.CP.p2_internal_subjects(sorted(set(subjects)))
    keep = np.isin(subjects, internal)
    assert len(calls) == 1 and sorted(set(calls[0])) == internal
    assert result["metadata"]["input_sha256"] == M.DA.fingerprint(w[keep], y[keep], subjects[keep], 200., FF.myo8_montage("epn612", 200.))


@pytest.mark.parametrize("strings", [False, True])
def test_fit_matches_convergent_historical_probe_and_has_no_held_selection(strings):
    x, y, s = sample(strings=strings)
    roles = M.CP.fold_roles(sorted(set(s)))
    audit = M.TP.AuditLog()
    actual = M.fit_fold(x, y, s, roles, 0, audit)
    old = M.TP.fit_fold(x, y, s, roles, 0, M.TP.AuditLog())
    assert actual["report"] == old["report"]
    for key in M.OOF_KEYS:
        assert np.array_equal(actual[key], old[key])
    changed = x.copy()
    changed[np.isin(s, roles[0]["held"])] += 1000
    other = M.fit_fold(changed, y, s, roles, 0)
    assert other["report"]["c"] == actual["report"]["c"]
    assert other["report"]["scaler_sha256"] == actual["report"]["scaler_sha256"]
    assert len(audit) == 7 and all(r["max_iter"] == 12000 for r in audit)


def test_roles_and_absent_class_support_are_enforced():
    x, y, s = sample()
    roles = M.CP.fold_roles(sorted(set(s)))
    bad = copy.deepcopy(roles)
    bad[0]["train"].append(bad[0]["held"][0])
    with pytest.raises(ValueError, match="overlap"):
        M.fit_fold(x, y, s, bad, 0)
    held_row = np.flatnonzero(np.isin(s, roles[0]["held"]))[0]
    y[held_row] = 9
    result = M.fit_fold(x, y, s, roles, 0)
    assert result["report"]["dropped_held_windows"] == 1
    assert held_row not in result["row"]
    M._verify_support(result, y, s, roles, [0])
    wrong = {k: result[k].copy() for k in M.OOF_KEYS}
    wrong["row"][0] = held_row
    with pytest.raises(ValueError, match="support"):
        M._verify_support(wrong, y, s, roles, [0])


def test_nonconvergence_retains_candidates_but_invalidates_decision(tmp_path, monkeypatch):
    pin, _, _, _ = cache(tmp_path)
    old_fit = M.LogisticRegression.fit
    def warning_fit(self, x, y):
        result = old_fit(self, x, y)
        if self.C == 100:
            warnings.warn("injected solver warning", ConvergenceWarning)
        return result
    monkeypatch.setattr(M.LogisticRegression, "fit", warning_fit)
    out = tmp_path / "probe.json"
    with pytest.warns(ConvergenceWarning):
        result = M.fit_cache(pin["path"], "pre", out, expected=pin,
                             identity={"code_sha": "1"*40, "plan_sha256": "2"*64}, folds=[0])
    assert result["complete"] and not result["decision_valid"]
    assert len(result["fit_audit"]) == 7 and len(result["folds"][0]["val_scores"]) == 6
    assert result["audit_summary"]["convergence_warnings"] >= 1
    report, arrays = M._read_pinned_report({"path": str(out), "sha256": M.sha(out)})
    assert report["oof_sha256"] == M.sha(out.with_suffix(".oof.npz"))
    assert len(arrays["row"]) == result["folds"][0]["n_held"]


def test_nonfinite_fit_is_persisted_without_complete_result(tmp_path, monkeypatch):
    pin, _, _, _ = cache(tmp_path)
    old_fit = M.LogisticRegression.fit
    def nonfinite_fit(self, x, y):
        value = old_fit(self, x, y)
        self.coef_[0, 0] = np.nan
        return value
    monkeypatch.setattr(M.LogisticRegression, "fit", nonfinite_fit)
    out = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="nonfinite fitted"):
        M.fit_cache(pin["path"], "post", out, expected=pin,
                    identity={"code_sha": "1"*40, "plan_sha256": "2"*64}, folds=[0])
    assert not out.exists() and not out.with_suffix(".oof.npz").exists()
    audit = json.loads(out.with_suffix(".audit.jsonl").read_text())
    assert "nonfinite" in audit["exception"]


def matrix():
    cells = {}
    for step in M.DOSES:
        for view in M.VIEWS:
            cells[M.cell_name(step, view)] = {}
            for ds in M.DATASETS:
                row = np.arange(16)
                y = np.tile([0, 0, 1, 1], 4)
                pred = y.copy()
                pred[::4] = 1 if step == 19618 else 0
                cells[M.cell_name(step, view)][ds] = {"row": row, "y": y, "pred": pred,
                    "s": np.repeat([f"s{i}" for i in range(8)], 2), "fold": np.repeat(range(4), 4)}
    convergence = {name: {ds: True for ds in M.DATASETS} for name in cells}
    return cells, convergence


def test_paired_bootstrap_all_dose_depth_contrasts_and_vr1_reference():
    cells, convergence = matrix()
    result = M.compare_arrays(cells, convergence=convergence, n_boot=100)
    assert len(result["tasks"]["p1"]["contrasts"]) == 12
    assert result["tasks"]["p1"]["contrasts"]["pool:49046-19618"]["role"] == "primary_dose"
    assert result["reading"]["classification_reference_candidate"] == 39237
    assert result["datasets"]["epn612"]["replicates_missing_classes"] > 0
    again = M.compare_arrays(cells, convergence=convergence, n_boot=100)
    assert result == again
    unresolved = M.compare_arrays(cells, n_boot=100)
    assert unresolved["reading"]["classification_reference_candidate"] is None
    assert unresolved["tasks"]["p1"]["contrasts"]["pool:49046-19618"]["reading"] == "unresolved_solver"


@pytest.mark.parametrize("field", ["row", "y", "s", "fold"])
def test_pairing_rejects_changed_oof_support(field):
    cells, convergence = matrix()
    a = cells["49046/post"]["epn612"]
    if field == "row":
        a[field][-1] = 99
    elif field == "s":
        a[field][-1] = "zz"
    else:
        a[field][-1] = 0
    with pytest.raises(ValueError, match="pairing"):
        M.compare_arrays(cells, convergence=convergence, n_boot=20)


@pytest.mark.parametrize("mismatch", ["new_vs_historical", "historical_tasks", "new_cells"])
def test_report_aggregation_rejects_numeric_mismatch_across_entire_matrix(monkeypatch, mismatch):
    hashes = {"old-p1": "a"*64, "old-p2": "a"*64}
    new = [{"protocol": M.PROTOCOL, "kind": "probe", "numeric_sha256": "a"*64,
            "step": step, "view": "post", "dataset": "epn612", "task": "p2"} for step in (19618, 49046)]
    old = ["old-p1", "old-p2"]
    if mismatch == "new_vs_historical":
        new[0]["numeric_sha256"] = "b"*64
    elif mismatch == "historical_tasks":
        hashes["old-p2"] = "b"*64
    else:
        old = []
        new[-1]["numeric_sha256"] = "b"*64
    monkeypatch.setattr(M, "read_historical_pool", lambda pin, *args, **kwargs: {"numeric_sha256": hashes[pin]})
    monkeypatch.setattr(M, "_read_pinned_report", lambda pin: (new[pin], {}))
    with pytest.raises(ValueError, match="numeric environment"):
        M.compare_reports([0, 1], cache_pins={}, historical_report_pins=old)


def test_vr1_tradeoff_is_not_a_win():
    cells, convergence = matrix()
    result = M.compare_arrays(cells, convergence=convergence, n_boot=50)
    task = result["tasks"]
    task["p2"]["contrasts"]["pool:49046-19618"].update(delta_pp=-2., ci95_pp=[-3., -1.1])
    reading = M.classification_reading(task)
    assert reading["contrasts"]["pool:49046-19618"]["status"] == "tradeoff"
    assert not reading["contrasts"]["pool:49046-19618"]["practical_benefit"]


def test_benchmark_never_fits_validation_or_held_and_projects_full_budget(tmp_path, monkeypatch):
    pin, arrays, _, _ = cache(tmp_path)
    rows_seen = []
    old_fit = M.LogisticRegression.fit
    def capture(self, x, y):
        rows_seen.append(len(y))
        return old_fit(self, x, y)
    monkeypatch.setattr(M.LogisticRegression, "fit", capture)
    result = M.benchmark_cache(pin["path"], "post", expected=pin, max_rows=12)
    assert rows_seen == [12]*6
    assert set(result["subjects"]) <= set(result["fold0_train_subjects"])
    assert result["validation_or_held_scored"] is False
    projection = M.project_full_matrix([result], [{"dataset": "ninapro_db2", "view": "post",
        "width": 8, "train_rows": [30]*4, "classes": [2]*4}])
    assert projection["projected_seconds"] > sum(f["elapsed_s"] for f in result["fits"])
    assert projection["max_iter_charged_per_fit"] == 12000
    assert projection["fits_per_fold"] == 7


def test_real_teacher_same_forward_crops_and_historical_pool():
    from wearusfm.harness import fm_features as FF
    from wearusfm.model.fm import FMConfig, WearUsFM
    torch.manual_seed(14)
    model = WearUsFM(FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, muscle_dropout=.4,
                             backbone_layers=1, decoder_layers=1, window_ms=100., tokenizer_activation="gelu"))
    windows = np.random.default_rng(17).normal(size=(2, 900, 8)).astype(np.float32)
    subjects = np.array(["a", "b"])
    montage = FF.myo8_montage("epn612", 200.)
    calls = []
    handle = model.pool.register_forward_hook(lambda *args: calls.append(1))
    actual = M.BD.extract_windows(model, windows, subjects, 200., montage, batch=2)
    handle.remove()
    assert len(calls) == 2  # one real encode per crop, shared by all three views
    historical = FF.extract_features(model, windows, 200., montage, subjects, batch=2)
    np.testing.assert_allclose(actual["pool"], historical, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(actual["post"], historical[:, :16], atol=1e-6, rtol=1e-5)


def test_iteration_limit_without_warning_is_unresolved_and_audit_cannot_be_rewritten(tmp_path, monkeypatch):
    pin, _, _, _ = cache(tmp_path)
    original = M.LogisticRegression.fit
    def limit(self, x, y):
        model = original(self, x, y)
        self.n_iter_ = np.asarray([12000])
        return model
    monkeypatch.setattr(M.LogisticRegression, "fit", limit)
    out = tmp_path/"limited.json"
    result = M.fit_cache(pin["path"], "post", out, expected=pin,
        identity={"code_sha": "1"*40, "plan_sha256": "2"*64}, folds=[0])
    assert not result["decision_valid"] and result["audit_summary"]["reached_limit"] == 7
    assert result["audit_summary"]["convergence_warnings"] == 0
    M._read_pinned_report({"path": str(out), "sha256": M.sha(out)})
    result["fit_audit"][0]["elapsed_s"] += 1
    out.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="persisted/embedded"):
        M._read_pinned_report({"path": str(out), "sha256": M.sha(out)})


@pytest.mark.parametrize("sharded", [False, True])
def test_historical_pool_reuses_complete_aggregate_or_shards_and_checks_selection(tmp_path, sharded):
    pins, cells, folds, records = {}, {}, {}, []
    for ds in M.TASKS["p2"]:
        x, y, s = sample(width=768)
        subjects = sorted(set(s.tolist()))
        path = tmp_path/(ds+"-old.npz")
        np.savez(path, y=y, s=s, subjects=np.asarray(subjects), encoder_dim=np.asarray(384),
            checkpoint_sha256=np.asarray("a"*64), input_sha256=np.asarray("b"*64), splits_sha256=np.asarray("c"*64))
        pins[ds] = {"path": str(path), "sha256": M.sha(path), "subjects": subjects, "input_sha256": "b"*64}
        audit = M.TP.AuditLog()
        roles = M.CP.fold_roles(subjects)
        results = [M.TP.fit_fold(x, y, s, roles, f, audit, {"arm": "T100-G", "dataset": ds,
            "task": "p2", "cache_sha256": pins[ds]["sha256"], "checkpoint_sha256": "a"*64,
            "code_sha": "d"*40, "plan_sha256": "e"*64}) for f in range(4)]
        cells[ds] = {k: np.concatenate([v[k] for v in results]) for k in M.OOF_KEYS}
        folds[ds] = {"folds": [r["report"] for r in results],
                     "balanced_accuracy": M.TP.scored(cells[ds]["y"], cells[ds]["pred"])[0]}
        records.extend(audit)
    reports = []
    groups = [[ds] for ds in M.TASKS["p2"]] if sharded else [list(M.TASKS["p2"])]
    expected_plan = {"science": "frozen"}
    for i, group in enumerate(groups):
        path, oof, audit = (tmp_path/f"historical-{i}{suffix}" for suffix in (".json", ".oof.npz", ".audit.jsonl"))
        fit_records = [r for r in records if r["dataset"] in group]
        np.savez(oof, **{f"{ds}_{k}": cells[ds][k] for ds in group for k in M.OOF_KEYS})
        audit.write_text("".join(json.dumps(r)+"\n" for r in fit_records))
        numeric = M.TP.numeric_context()
        value = {"protocol": M.TP.PROTOCOL, "kind": "shard" if sharded else "aggregate", "complete": True,
            "arm": "T100-G", "task": "p2", "step": 19618, "recipe": M.TP.recipe(),
            "weights": dict(zip(M.TASKS["p2"], M.WEIGHTS["p2"])), "plan_content_sha256": M.jhash(expected_plan),
            "code_sha": "d"*40, "plan_sha256": "e"*64,
            "datasets": {ds: folds[ds] for ds in group}, "numeric_context": numeric, "numeric_sha256": M.jhash(numeric),
            "audit": str(audit), "audit_sha256": M.sha(audit), "audit_summary": M.TP.audit_summary(fit_records),
            "oof": str(oof), "oof_sha256": M.sha(oof)}
        path.write_text(json.dumps(value))
        reports.append({"path": str(path), "sha256": M.sha(path)})
    result = M.read_historical_pool(reports, pins, expected_plan=expected_plan)
    assert set(result["cells"]) == set(M.TASKS["p2"])
    assert all(result["convergence"].values()) and len(result["records"]) == 56
    path = Path(reports[0]["path"])
    bad = json.loads(path.read_text())
    ds = next(iter(bad["datasets"]))
    bad["datasets"][ds]["folds"][0]["c"] = 999
    path.write_text(json.dumps(bad))
    reports[0]["sha256"] = M.sha(path)
    with pytest.raises(ValueError, match="selection"):
        M.read_historical_pool(reports, pins, expected_plan=expected_plan)
