"""Temporal indexing, leakage and paired-statistic oracles; no remote data."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("pose_temporal_readout", SCRIPTS / "pose_temporal_readout.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def data(seed=20, users=("a", "b", "c", "d"), windows=5, patches=16, dim=4, joints=2):
    rng = np.random.default_rng(seed)
    n = len(users)*windows
    lat = rng.normal(size=(n, patches, dim)).astype(np.float16)
    lengths = np.full(n, patches, dtype=np.int64)
    x = T.normalized_inputs(lat, lengths)
    weights = rng.normal(size=(dim, joints))*.4
    y = (x@weights + rng.normal(size=(n, patches, joints))).astype(np.float32)
    mask = rng.random(y.shape) > .15
    meta = [{"user": u, "n_patches": patches, "recording": i, "start": 0}
            for i, u in enumerate(np.repeat(users, windows))]
    return {"lat": lat, "lengths": lengths, "target": y, "mask": mask,
            "central_mask": T.central_mask(mask, lengths), "meta": meta,
            "std": np.full((n, joints), .7, np.float32),
            "pose_var": np.tile([.1, .2, .5, .8, .9], len(users)).astype(np.float32)}


def test_lag_oracle_exact_order_and_no_window_crossing():
    x = np.arange(2*13*3).reshape(2, 13, 3).astype(np.float32)
    actual = T.lag_features(x, "temporal")
    expected = np.asarray([[np.concatenate([x[w, t-4], x[w, t], x[w, t+4]])
                            for t in range(4, 9)] for w in range(2)])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(T.lag_features(x, "instantaneous"), actual[..., 3:6])
    assert actual.shape == (2, 5, 9)


def test_pose_missing_at_context_does_not_change_inputs_and_padding_is_excluded():
    d = data()
    d["lengths"][0] = 11
    d["lat"][0, 11:] = np.nan
    d["mask"][0, 11:] = False
    first = T.normalized_inputs(d["lat"], d["lengths"])
    d["mask"][:, 0:4] = False  # left contexts lack pose, but still have EMG.
    second = T.normalized_inputs(d["lat"], d["lengths"])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0, :11], torch.nn.functional.layer_norm(
        torch.from_numpy(d["lat"][0, :11].astype(np.float32)), (4,), eps=1e-5).numpy())
    support = T.central_mask(d["mask"], d["lengths"])
    assert not support[0, 3:].any()  # centers >=7 would need context >=11.
    assert support[0, :3].any()
    d["lat"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="real input context"):
        T.normalized_inputs(d["lat"], d["lengths"])


def test_instantaneous_and_temporal_fit_use_identical_central_rows():
    d = data()
    for reader in T.READERS:
        parts, width, joints = T.grouped_stats(d, reader, batch=3)
        assert width == (4 if reader == "instantaneous" else 12)
        for user, stats in parts.items():
            pick = np.array([m["user"] == user for m in d["meta"]])
            for group in stats:
                for joint in group["joints"]:
                    assert group["n"] == int(d["central_mask"][pick, :, joint].sum())


def test_fit_matches_direct_masked_ridge_and_grouped_cv():
    d = data(seed=61, windows=5)
    w, b, audit = T.fit_reader(d, "instantaneous", [["a", "c"], ["b", "d"]])
    cv = audit["selection"]
    assert cv["fold_users"] == [["a", "c"], ["b", "d"]]
    assert not set(cv["fold_users"][0]) & set(cv["fold_users"][1])
    x = T.lag_features(T.normalized_inputs(d["lat"], d["lengths"]), "instantaneous").reshape(-1, 4).astype(float)
    y, mask = d["target"][:, 4:-4].reshape(-1, 2), d["central_mask"].reshape(-1, 2)
    for j in range(2):
        xx, yy = x[mask[:, j]], y[mask[:, j], j].astype(float)
        mx, my = xx.mean(0), yy.mean()
        direct = np.linalg.solve((xx-mx).T@(xx-mx) + cv["alpha"]*np.eye(4), (xx-mx).T@(yy-my))
        np.testing.assert_allclose(w[:, j], direct, atol=1e-10)
        assert b[j] == pytest.approx(my-mx@direct, abs=1e-10)
    with pytest.raises(ValueError, match="fold users"):
        T.fit_reader(d, "instantaneous", [["a", "b"], ["c", "d"]])


def test_masked_targets_and_evaluation_targets_cannot_leak_into_fit_or_prediction():
    train = data(seed=61)
    w, b, first = T.fit_reader(train, "instantaneous")
    changed = copy.deepcopy(train)
    changed["target"][~changed["mask"]] = np.nan
    w2, b2, second = T.fit_reader(changed, "instantaneous")
    np.testing.assert_array_equal(w, w2)
    np.testing.assert_array_equal(b, b2)
    assert first["selection"] == second["selection"]
    val = data(seed=72, users=("v", "w"))
    prediction = T.predict(val["lat"], val["lengths"], "instantaneous", w, b)
    val["target"][:] = 1e8
    val["mask"][:] = False
    val["pose_var"][:] = 1e8
    np.testing.assert_array_equal(prediction, T.predict(val["lat"], val["lengths"], "instantaneous", w, b))


def test_unresolved_alpha_stops_before_scientific_prediction(monkeypatch):
    monkeypatch.setattr(T.E, "cross_validate", lambda *a: {"boundary_unresolved": True})
    with pytest.raises(ValueError, match="alpha boundary unresolved"):
        T.fit_reader(data(), "instantaneous")


def test_dynamic_membership_uses_frozen_full_window_variance_not_cropped_targets():
    d = data()
    result = T.summarize(d, d["target"][:, 4:-4].astype(float))
    assert result["groups"]["dynamic"]["n_selected_windows"] == 12
    assert result["groups"]["dynamic"]["centered"]["centered_mse_ratio"] == pytest.approx(0)
    # Extremely different target variance must not silently relabel windows.
    d["target"] *= 1e4
    second = T.summarize(d, d["target"][:, 4:-4].astype(float))
    assert second["groups"]["dynamic"]["n_selected_windows"] == 12


def cells():
    d = data()
    return {f"{model}/{reader}": {"analysis": T.summarize(d, d["target"][:, 4:-4]*factor)}
            for model, reader, factor in (("B", "instantaneous", .6), ("B", "temporal", .9),
                                          ("R3c", "instantaneous", .5), ("R3c", "temporal", .7))}


def test_paired_bootstrap_oracle_same_draws_and_difference_of_differences():
    c = cells()
    got = T.paired_comparison(c)
    primary = got["dynamic"]
    assert primary["n_bootstrap"] == 2000 and primary["seed"] == 20261102
    assert primary["contrasts"]["B_temporal_minus_instantaneous"]["delta_ratio"] == pytest.approx(.01-.16, abs=1e-7)
    assert primary["contrasts"]["R3c_temporal_minus_instantaneous"]["delta_ratio"] == pytest.approx(.09-.25, abs=1e-7)
    item = primary["contrasts"]["difference_of_differences"]
    assert item["delta_ratio"] == pytest.approx(.01, abs=1e-7)
    users = primary["users"]
    draws = np.random.default_rng(20261102).integers(0, len(users), (2000, len(users)))
    num = np.array([item["per_user"][u]["delta_centered_sse"] for u in users])
    den = np.array([item["per_user"][u]["target_variance_sum"] for u in users])
    np.testing.assert_array_equal(item["ci95"], np.percentile(num[draws].sum(1)/den[draws].sum(1), [2.5, 97.5]))
    c["R3c/temporal"]["analysis"]["support_sha256"] = "wrong"
    with pytest.raises(ValueError, match="support differs"):
        T.paired_comparison(c)


def test_plan_pins_expected_inputs_and_code():
    plan = T.make_plan()
    T.validate_plan(plan)
    assert plan["caches"]["B"]["train"]["n_windows"] == 1907
    assert sum(p["bytes"] for m in plan["caches"].values() for p in m.values()) == 5238999342
    assert len(plan["fold_users"][0]) == len(plan["fold_users"][1]) == 20
    changed = copy.deepcopy(plan)
    changed["dynamic_threshold"] += .001
    with pytest.raises(ValueError, match="dynamic_threshold"):
        T.validate_plan(changed)
    changed = copy.deepcopy(plan)
    changed["sources"]["scripts/pose_temporal_readout.py"] = "0"*64
    with pytest.raises(ValueError, match="sources"):
        T.validate_plan(changed)
    changed = copy.deepcopy(plan)
    changed["caches"]["B"]["train"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="cache pin"):
        T.validate_plan(changed)


def test_headers_do_not_materialize_feature_payloads(tmp_path, monkeypatch):
    path = tmp_path / "tiny.npz"
    np.savez_compressed(path, lat=np.ones((2, 16, 4), np.float16), target=np.zeros((2, 16, 2), np.float32))
    monkeypatch.setattr(np.lib.format, "read_array", lambda *a, **kw: pytest.fail("payload read"))
    result = T.headers(path)
    assert result["lat"] == {"shape": [2, 16, 4], "dtype": "<f2"}


def test_benchmark_dispatch_never_opens_val_cache(tmp_path, monkeypatch):
    plan = T.make_plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    touched = []
    identity = {"sources": plan["sources"]}
    monkeypatch.setattr(T, "runtime_identity", lambda *a: identity)
    def inspect(pin, model, split):
        touched.append((model, split))
        assert split == "train"
        return {"users": sorted(sum(plan["fold_users"], [])), "states": {}}
    monkeypatch.setattr(T, "validate_artifact", inspect)
    monkeypatch.setattr(T, "read_data", lambda pin: touched.append(pin["path"]) or {})
    monkeypatch.setattr(T, "benchmark_train", lambda train: {"timing": 1})
    out = tmp_path / "benchmark.json"
    assert T.main(["benchmark", "--plan", str(path), "--expected-code-sha", "x", "--out", str(out)]) == 0
    assert touched[0] == ("B", "train") and len(touched) == 2
    assert "train_subset.npz" in touched[1]
    assert json.loads(out.read_text())["status"] == "train_only_timing"


def test_source_mutation_before_output_is_rejected(tmp_path, monkeypatch):
    plan = T.make_plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    states = iter([{"sources": plan["sources"]}, {"sources": {}}])
    monkeypatch.setattr(T, "runtime_identity", lambda *a: next(states))
    monkeypatch.setattr(T, "validate_artifact", lambda p, m, s: {"users": sorted(sum(plan["fold_users"], [])), "states": {}})
    monkeypatch.setattr(T, "read_data", lambda *a: {})
    monkeypatch.setattr(T, "benchmark_train", lambda *a: {})
    out = tmp_path / "out.json"
    with pytest.raises(ValueError, match="changed during execution"):
        T.main(["benchmark", "--plan", str(path), "--expected-code-sha", "x", "--out", str(out)])
    assert not out.exists()


def test_read_data_rejects_padding_target_or_nonfinite_supported_arrays(tmp_path):
    d = data()
    path = tmp_path / "cache.npz"
    def save(current):
        np.savez(path, **{k: current[k] for k in ("lat", "target", "mask", "std", "pose_var")},
                 meta_json=np.asarray(json.dumps(current["meta"])))
    save(d)
    loaded = T.read_data({"path": str(path)})
    np.testing.assert_array_equal(loaded["central_mask"], d["central_mask"])
    d["meta"][0]["n_patches"] = 10
    d["mask"][0, 10:] = True
    save(d)
    with pytest.raises(ValueError, match="mask includes padding"):
        T.read_data({"path": str(path)})
    d["mask"][0, 10:] = False
    d["lat"][0, 10:] = np.nan
    save(d)
    T.read_data({"path": str(path)})
    d["mask"][0, 0, 0] = True
    d["target"][0, 0, 0] = np.nan
    save(d)
    with pytest.raises(ValueError, match="supported target"):
        T.read_data({"path": str(path)})


def test_full_four_cell_run_refits_same_support_and_rejects_target_mismatch(monkeypatch):
    train, val = data(seed=61), data(seed=85, users=("v", "w"))
    val["pose_var"] = np.tile([.1, .2, T.THRESHOLD, .8, .9], 2).astype(np.float32)
    datasets = {"B/train": train, "R3c/train": copy.deepcopy(train),
                "B/val": val, "R3c/val": copy.deepcopy(val)}
    plan = {"fold_users": [["a", "c"], ["b", "d"]], "caches": {
        model: {split: {"fixture": f"{model}/{split}"} for split in ("train", "val")} for model in T.MODELS}}
    monkeypatch.setattr(T, "read_data", lambda pin: copy.deepcopy(datasets[pin["fixture"]]))
    result, coefficients = T.run(plan)
    assert result["status"] == "completed" and len(result["cells"]) == 4
    assert len(coefficients) == 8
    assert result["paired"]["dynamic"]["contrasts"]["difference_of_differences"]["delta_ratio"] == 0
    assert len({r["fit"]["fit_support_sha256"] for r in result["cells"].values()}) == 1
    for reader in T.READERS:
        np.testing.assert_array_equal(coefficients[f"B_{reader}_weights"], coefficients[f"R3c_{reader}_weights"])
    # B and R3c must use identical targets, including unobserved placeholders.
    datasets["R3c/train"]["target"][0, 4, 0] += 1
    with pytest.raises(ValueError, match="target/support mismatch"):
        T.run(plan)


def test_validation_hashes_identity_and_reads_only_metadata(tmp_path, monkeypatch):
    path = tmp_path / "train_subset.npz"
    cfg = {"which": "student", "split": "train", "code_sha": T.E.CODE_SHA,
           "pose_stats_sha256": T.E.STATS_SHA, "checkpoint_pose_stats_sha256": T.E.STATS_SHA,
           "window_s": 4., "seed": 20261026}
    meta = [{"user": "a", "n_patches": 160}, {"user": "b", "n_patches": 160}]
    np.savez_compressed(path, lat=np.ones((2, 160, 384), np.float16), target=np.ones((2, 160, 20), np.float32),
        mask=np.ones((2, 160, 20), bool), pred=np.zeros((2, 160, 20), np.float32),
        std=np.ones((2, 20), np.float32), pose_var=np.ones(2, np.float32),
        config_json=np.asarray(json.dumps(cfg)), meta_json=np.asarray(json.dumps(meta)),
        dim=np.asarray(384), n_recordings=np.asarray(2), has_head=np.asarray(True))
    prov = {"model": "B", "which": "student", "split": "train", "checkpoint_step": 19618,
            "checkpoint_sha256": "a"*64, "stats_sha256": T.E.STATS_SHA, "code_sha": T.E.CODE_SHA,
            "cache": str(path), "cache_sha256": T.D.file_sha256(path), "cache_bytes": path.stat().st_size,
            "same_metadata_target_mask_std_pose_var_as_B5000": True, "config": cfg}
    pp = Path(str(path) + ".provenance.json")
    pp.write_text(json.dumps(prov))
    receipt = {k: prov[k] for k in ("model", "checkpoint_sha256", "checkpoint_step", "code_sha")}
    receipt.update(complete=True, checkpoint_unchanged=True,
                   outputs={"train": {"sha256": prov["cache_sha256"], "provenance_sha256": T.D.file_sha256(pp)}})
    rp = tmp_path / "extraction_receipt.json"
    rp.write_text(json.dumps(receipt))
    pin = {"path": str(path), "original_path": str(path), "sha256": prov["cache_sha256"], "bytes": prov["cache_bytes"],
           "provenance_sha256": T.D.file_sha256(pp), "receipt_sha256": T.D.file_sha256(rp),
           "checkpoint_sha256": "a"*64, "n_windows": 2, "n_users": 2, "n_recordings": 2}
    touched = []
    original = np.lib.npyio.NpzFile.__getitem__
    def selected_only(self, key):
        touched.append(key)
        assert key in {"dim", "n_recordings", "has_head", "config_json", "meta_json"}
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected_only)
    result = T.validate_artifact(pin, "B", "train")
    assert result["users"] == ["a", "b"]
    assert set(touched) == {"dim", "n_recordings", "has_head", "config_json", "meta_json"}
    touched.clear()
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA/size"):
        T.validate_artifact(pin, "B", "train")
    assert not touched
