"""Crop, train-only selection, three-arm pairing and artifact gate oracles."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import tokenizer_pose_readout as R


def data(seed=61, users=("a", "b", "c", "d"), windows=8, patches=24, dim=4, joints=2):
    rng = np.random.default_rng(seed)
    n = len(users)*windows
    lat = rng.normal(size=(n, patches, dim)).astype(np.float16)
    lengths = np.full(n, patches, dtype=np.int64)
    x = R.T.normalized_inputs(lat, lengths)
    y = (x@rng.normal(size=(dim, joints))*.4+rng.normal(size=(n, patches, joints))).astype(np.float32)
    mask = rng.random(y.shape) > .15
    return {"lat": lat, "target": y, "mask": mask, "lengths": lengths,
            "central_mask": R.T.central_mask(mask, lengths, offset=6),
            "meta": [{"user": u, "recording": i, "start": 0, "n_patches": patches}
                     for i, u in enumerate(np.repeat(users, windows))],
            "std": np.full((n, joints), .7, np.float32), "n_recordings": n,
            "pose_var": np.resize(np.array([.1, .5], np.float32), n)}


def cell(arm, d, factor, step=5000):
    return {"complete": True, "arm": arm, "which": "teacher", "checkpoint_step": step,
            "analysis": R.summarize(d, d["target"][:, 6:-6]*factor)}


def cells():
    d = data()
    return {a: cell(a, d, f) for a, f in zip(R.ARMS, (.6, .8, .9))}


def fake_plan(tmp_path):
    refs, caches = {}, {}
    for split, expected in R.SELECTION.items():
        refs[split] = {"path": str(tmp_path/("ref_"+split+".npz")), "sha256": "1"*64, "bytes": 1, **expected}
    for i, arm in enumerate(R.ARMS):
        caches[arm] = {}
        for split in ("train", "val"):
            prefix = str(tmp_path/(arm+"_"+split))
            caches[arm][split] = {"path": prefix+".npz", "sha256": "2"*64, "bytes": 1,
                                  "provenance_path": prefix+".provenance.json", "provenance_sha256": "3"*64,
                                  "receipt_path": prefix+".receipt.json", "receipt_sha256": "4"*64,
                                  "checkpoint_sha256": str(i+5)*64, "extraction_code_sha": "a"*40}
    users = [f"u{i:02}" for i in range(40)]
    return R.make_plan(caches, refs, [users[::2], users[1::2]], 5000)


def test_crop6_stats_and_prediction_match_direct_ridge():
    d = data()
    weights, bias, fit = R.fit_reader(d, [["a", "c"], ["b", "d"]])
    alpha = fit["selection"]["alpha"]
    x = R.T.normalized_inputs(d["lat"], d["lengths"])[:, 6:-6].reshape(-1, 4).astype(np.float64)
    y = d["target"][:, 6:-6].reshape(-1, 2).astype(np.float64)
    mask = d["central_mask"].reshape(-1, 2)
    for j in range(2):
        xx, yy = x[mask[:, j]], y[mask[:, j], j]
        mx, my = xx.mean(0), yy.mean()
        w = np.linalg.solve((xx-mx).T@(xx-mx)+alpha*np.eye(4), (xx-mx).T@(yy-my))
        np.testing.assert_allclose(weights[:, j], w, atol=1e-11)
        assert bias[j] == pytest.approx(my-mx@w, abs=1e-11)
    prediction = R.predict(d["lat"], d["lengths"], weights, bias)
    np.testing.assert_allclose(prediction, (x@weights+bias).reshape(len(d["lat"]), 12, 2), atol=1e-12)
    parts, _, _ = R.grouped_stats(d, batch=3)
    for user, stats in parts.items():
        pick = np.asarray([m["user"] == user for m in d["meta"]])
        for group in stats:
            for j in group["joints"]:
                assert group["n"] == d["central_mask"][pick, :, j].sum()


def test_boundary_targets_and_masked_targets_do_not_change_fit():
    d = data()
    first = R.fit_reader(d)
    changed = copy.deepcopy(d)
    changed["target"][:, :6] = 1e12
    changed["target"][:, -6:] = -1e12
    changed["target"][~changed["mask"]] = np.nan
    second = R.fit_reader(changed)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2]["selection"] == second[2]["selection"]
    with pytest.raises(ValueError, match="fold users"):
        R.fit_reader(d, [["a", "b"], ["c", "d"]])


def test_prediction_independent_of_val_targets_and_missing_neighbor_pose():
    train, val = data(), data(seed=91, users=("v", "w"))
    w, b, _ = R.fit_reader(train)
    prediction = R.predict(val["lat"], val["lengths"], w, b)
    val["target"][:] = np.nan
    val["mask"][:] = False
    val["pose_var"][:] = 1e9
    np.testing.assert_array_equal(prediction, R.predict(val["lat"], val["lengths"], w, b))
    # Padding is excluded by real patch availability, not neighboring pose.
    val["lengths"][0] = 19
    val["lat"][0, 19:] = np.nan
    mask = np.ones_like(val["mask"])
    mask[0, 19:] = False
    central = R.T.central_mask(mask, val["lengths"], offset=6)
    assert central[0, :7].all() and not central[0, 7:].any()
    assert np.isfinite(R.predict(val["lat"], val["lengths"], w, b)).all()


def test_original_pose_variance_controls_dynamic_membership():
    d = data()
    got = R.summarize(d, d["target"][:, 6:-6].astype(float))
    assert got["groups"]["dynamic"]["n_selected_windows"] == len(d["lat"])/2
    assert got["groups"]["dynamic"]["centered"]["centered_mse_ratio"] == pytest.approx(0)
    d["target"] *= 1e4
    changed = R.summarize(d, d["target"][:, 6:-6].astype(float))
    assert changed["groups"]["dynamic"]["n_selected_windows"] == got["groups"]["dynamic"]["n_selected_windows"]


def test_three_contrasts_share_bootstrap_draws_and_exact_oracle():
    got = R.paired_comparison(cells(), 5000)["dynamic"]
    assert got["n_bootstrap"] == 2000 and got["seed"] == 20261102
    expected = {"T100-L_minus_T25-L": -.12, "T100-G_minus_T100-L": -.03, "T100-G_minus_T25-L": -.15}
    users = got["users"]
    draws = np.random.default_rng(20261102).integers(0, len(users), (2000, len(users)))
    assert got["draws_sha256"] == R.arrays_hash(draws)
    for name, delta in expected.items():
        contrast = got["contrasts"][name]
        assert contrast["delta_ratio"] == pytest.approx(delta, abs=1e-7)
        num = np.asarray([contrast["per_user"][u]["delta_centered_sse"] for u in users])
        den = np.asarray([contrast["per_user"][u]["target_variance_sum"] for u in users])
        np.testing.assert_array_equal(contrast["ci95"], np.percentile(num[draws].sum(1)/den[draws].sum(1), [2.5, 97.5]))


@pytest.mark.parametrize("mutation", ["missing", "incomplete", "dose", "support", "denominator"])
def test_incomplete_or_unpaired_cells_fail(mutation):
    c = cells()
    if mutation == "missing":
        del c["T100-G"]
    elif mutation == "incomplete":
        c["T100-G"]["complete"] = False
    elif mutation == "dose":
        c["T100-G"]["checkpoint_step"] = 19618
    elif mutation == "support":
        c["T100-G"]["analysis"]["support_sha256"] = "different"
    else:
        c["T100-G"]["analysis"]["groups"]["dynamic"]["centered"]["per_user"]["a"]["sums"]["vary"] += 1
    with pytest.raises(ValueError):
        R.paired_comparison(c, 5000)


def test_zero_bootstrap_denominator_is_not_silently_dropped():
    c = cells()
    for cell_value in c.values():
        for row in cell_value["analysis"]["groups"]["dynamic"]["centered"]["per_user"].values():
            row["sums"]["vary"] = 0
    with pytest.raises(ValueError, match="no replicas discarded"):
        R.paired_comparison(c, 5000)


def test_plan_requires_full_selection_recipe_and_three_distinct_cells(tmp_path):
    plan = fake_plan(tmp_path)
    R.validate_plan(plan)
    for mutate in (lambda p: p["recipe"].update(offset=4),
                   lambda p: p["references"]["train"].update(n_users=39),
                   lambda p: p["caches"].pop("T100-G"),
                   lambda p: p["caches"]["T100-G"]["val"].update(checkpoint_sha256="0"*64),
                   lambda p: p.update(step=49046)):
        changed = copy.deepcopy(plan)
        mutate(changed)
        with pytest.raises(ValueError):
            R.validate_plan(changed)


def artifact_fixture(tmp_path):
    d = data(users=("a", "b"), windows=2, patches=160, dim=384, joints=20)
    cfg = {"which": "teacher", "split": "train", "code_sha": "a"*40, "pose_stats_sha256": R.E.STATS_SHA,
           "checkpoint_pose_stats_sha256": R.E.STATS_SHA, "n_windows": 10, "window_s": 4., "seed": 20261026, "per_side": False}
    path = tmp_path/"cache.npz"
    def save():
        np.savez(path, **{k: d[k] for k in ("lat", "target", "mask", "std", "pose_var")},
                 meta_json=np.asarray(json.dumps(d["meta"])), config_json=np.asarray(json.dumps(cfg)),
                 n_recordings=d["n_recordings"], dim=384, has_head=True)
    save()
    pp, rp = tmp_path/"provenance.json", tmp_path/"receipt.json"
    common = {"arm": "T100-G", "which": "teacher", "checkpoint_step": 5000, "checkpoint_sha256": "b"*64, "code_sha": "a"*40}
    counts = {"n_windows": 4, "n_users": 2, "n_recordings": 4}
    prov = {**common, **counts, "split": "train", "stats_sha256": R.E.STATS_SHA, "cache": str(path),
            "cache_sha256": R.D.file_sha256(path), "cache_bytes": path.stat().st_size, "config": cfg,
            "model_config": {"dim": 384, "k_latents": 64, "patch_ms": 25., "window_ms": 100., "tokenizer_activation": "gelu", "pose_head": True}}
    pp.write_text(json.dumps(prov))
    receipt = {**common, "complete": True, "checkpoint_unchanged": True, "postload_recipe_passed": True,
               "outputs": {"train": {"sha256": prov["cache_sha256"], "provenance_sha256": R.D.file_sha256(pp)}}}
    rp.write_text(json.dumps(receipt))
    pin = {"path": str(path), "sha256": prov["cache_sha256"], "bytes": prov["cache_bytes"],
           "provenance_path": str(pp), "provenance_sha256": R.D.file_sha256(pp), "receipt_path": str(rp),
           "receipt_sha256": R.D.file_sha256(rp), "checkpoint_sha256": "b"*64, "extraction_code_sha": "a"*40}
    plan = {"step": 5000, "stats_sha256": R.E.STATS_SHA, "references": {"train": counts}}
    ref = {"users": ["a", "b"], "targets": R.target_signature(d)}
    return d, pin, plan, ref, prov, receipt


def test_artifact_identity_gate_and_tampered_complete_receipt(tmp_path):
    _, pin, plan, ref, prov, receipt = artifact_fixture(tmp_path)
    assert R.validate_artifact(pin, "T100-G", "train", plan, ref)["users"] == ["a", "b"]
    receipt["complete"] = False
    Path(pin["receipt_path"]).write_text(json.dumps(receipt))
    pin["receipt_sha256"] = R.D.file_sha256(pin["receipt_path"])
    with pytest.raises(ValueError, match="incomplete extraction"):
        R.validate_artifact(pin, "T100-G", "train", plan, ref)


def test_artifact_gelu_and_reference_identity_not_just_rehashed_receipts(tmp_path):
    _, pin, plan, ref, prov, receipt = artifact_fixture(tmp_path)
    prov["model_config"]["tokenizer_activation"] = "identity"
    Path(pin["provenance_path"]).write_text(json.dumps(prov))
    pin["provenance_sha256"] = R.D.file_sha256(pin["provenance_path"])
    receipt["outputs"]["train"]["provenance_sha256"] = pin["provenance_sha256"]
    Path(pin["receipt_path"]).write_text(json.dumps(receipt))
    pin["receipt_sha256"] = R.D.file_sha256(pin["receipt_path"])
    with pytest.raises(ValueError, match="tokenizer model"):
        R.validate_artifact(pin, "T100-G", "train", plan, ref)


def test_target_hash_change_is_rejected_against_historical_selection(tmp_path):
    _, pin, plan, ref, _, _ = artifact_fixture(tmp_path)
    ref["targets"]["arrays"] = "0"*64
    with pytest.raises(ValueError, match="historical metadata/target"):
        R.validate_artifact(pin, "T100-G", "train", plan, ref)


def test_reference_reader_never_loads_latents(tmp_path, monkeypatch):
    d, pin, _, _, _, _ = artifact_fixture(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__
    def guarded(self, key):
        assert key not in ("lat", "pred")
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
    assert R.target_signature(R.read_targets(pin["path"])) == R.target_signature(d)


def test_benchmark_dispatch_never_reads_val_and_preserves_plan(tmp_path, monkeypatch):
    plan = fake_plan(tmp_path)
    plan_path = tmp_path/"plan.json"
    plan_path.write_text(json.dumps(plan))
    identity = {"sources": plan["sources"]}
    monkeypatch.setattr(R, "runtime_identity", lambda _: identity)
    users = sorted(sum(plan["fold_users"], []))
    touched = []
    def reference(pin):
        assert "ref_train" in pin["path"]
        touched.append("reference_train")
        return {"users": users, "targets": {}, "states": {}}
    def artifact(pin, arm, split, *args):
        assert split == "train"
        touched.append(arm+"/"+split)
        return {"users": users, "states": {}}
    monkeypatch.setattr(R, "validate_reference", reference)
    monkeypatch.setattr(R, "validate_artifact", artifact)
    monkeypatch.setattr(R, "read_data", lambda pin: touched.append(pin["path"]) or {})
    monkeypatch.setattr(R, "benchmark_train", lambda _: {"seconds": 1})
    out = tmp_path/"benchmark"
    assert R.main(["benchmark", "--plan", str(plan_path), "--expected-code-sha", "a"*40,
                   "--arm", "T100-G", "--out", str(out)]) == 0
    assert len(touched) == 3 and touched[1] == "T100-G/train" and "T100-G_train" in touched[2]
    assert json.loads((out/"report.json").read_text())["status"] == "train_only_timing"


def test_unresolved_alpha_and_source_changes_have_no_success_report(tmp_path, monkeypatch):
    monkeypatch.setattr(R.E, "cross_validate", lambda *args: {"boundary_unresolved": True})
    with pytest.raises(ValueError, match="alpha boundary unresolved"):
        R.fit_reader(data())
    p = tmp_path/"duplicate.json"
    p.write_text('{"complete": true, "complete": false}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        R.read_json(p)


def test_three_arm_runner_saves_coefficients_predictions_and_complete_comparison(tmp_path):
    train, val = data(), data(seed=72, users=("v", "w"))
    plan = {"step": 19618, "fold_users": [["a", "c"], ["b", "d"]], "caches": {}}
    for i, arm in enumerate(R.ARMS):
        plan["caches"][arm] = {}
        for split, original in (("train", train), ("val", val)):
            path = tmp_path/(arm+"_"+split+".npz")
            np.savez(path, **{k: original[k] for k in ("lat", "target", "mask", "std", "pose_var")},
                     meta_json=np.asarray(json.dumps(original["meta"])), n_recordings=original["n_recordings"])
            plan["caches"][arm][split] = {"path": str(path), "checkpoint_sha256": str(i+1)*64}
    out = tmp_path/"ridge"
    out.mkdir()
    got = R.run(plan, out)
    assert got["status"] == "completed" and got["complete"] is True
    assert set(got["cells"]) == set(R.ARMS)
    for arm, result in got["cells"].items():
        assert result["fit"]["selection"]["fold_users"] == plan["fold_users"]
        for key in ("coefficients", "predictions"):
            artifact = result[key]
            assert R.D.file_sha256(artifact["path"]) == artifact["sha256"]
        pred = np.load(result["predictions"]["path"], allow_pickle=False)
        assert pred.shape == (16, 12, 2) and pred.dtype == np.float64
        assert R.arrays_hash(pred) == result["analysis"]["prediction_sha256"]
    for contrast in got["paired"]["dynamic"]["contrasts"].values():
        assert contrast["delta_ratio"] == 0
        assert contrast["ci95"] == [0., 0.]
