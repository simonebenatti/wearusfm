"""Synthetic identities and provenance for the standalone cache-only dynamics analysis."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/pose_dynamics.py"
spec = importlib.util.spec_from_file_location("pose_dynamics", SCRIPT)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def data_for(pred, target, mask=None, pose_var=None, users=None):
    pred, target = np.asarray(pred, dtype=np.float32), np.asarray(target, dtype=np.float32)
    n, _, joints = target.shape
    mask = np.ones_like(target, dtype=bool) if mask is None else mask
    return {"pred": pred, "target": target, "mask": mask, "std": np.ones((n, joints), dtype=np.float32),
            "pose_var": np.ones(n, dtype=np.float32) if pose_var is None else np.asarray(pose_var, dtype=np.float32),
            "meta": [{"user": u, "recording": i} for i, u in enumerate(users or [f"u{i}" for i in range(n)])],
            "config": {}, "has_head": True, "n_recordings": n}


@pytest.mark.parametrize("kind,ratio,corr", [("perfect", 0., 1.), ("window_mean", 1., None), ("antiphase", 4., -1.),
                                          ("bias", 0., 1.)])
def test_expected_temporal_decomposition(kind, ratio, corr):
    target = np.array([[[1., 3.], [2., 4.], [3., 5.]], [[2., 5.], [4., 7.], [6., 9.]]])
    means = target.mean(axis=1, keepdims=True)
    pred = {"perfect": target, "window_mean": np.broadcast_to(means, target.shape),
            "antiphase": 2 * means - target, "bias": target + 2}[kind]
    result = D.analyze(data_for(pred, target))
    centered = result["groups"]["dynamic"]["centered_support_n_ge_2"]
    assert centered["centered_mse_ratio"] == pytest.approx(ratio)
    assert centered["pooled_centered_correlation"] == pytest.approx(corr) if corr is not None else centered["pooled_centered_correlation"] is None
    assert result["groups"]["dynamic"]["bootstrap_centered"]["evaluable"]
    m = centered["means"]
    assert m["total"] == pytest.approx(m["mean_error"] + m["centered"])
    assert m["baseline_total"] == pytest.approx(m["mean_target_squared"] + m["vary"])
    assert m["total_gain"] == pytest.approx(m["gain_mean"] + m["gain_centered"])
    if kind == "bias":
        assert m["mean_error"] == pytest.approx(4.) and m["total"] == pytest.approx(4.)
    if kind == "window_mean":
        assert centered["variance_ratio"] == 0 and m["gain_centered"] == 0
        assert "pooled_centered_correlation" in centered["undefined_metrics"]


def test_irregular_masks_and_one_or_zero_observation_use_identical_support():
    target = np.array([[[1., 4., 2.], [3., 99., 99.], [9., 99., 99.]], [[2., 8., 8.], [4., 10., 8.], [8., 99., 8.]]])
    mask = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [1, 0, 0]]], dtype=bool)
    pred = target + 1
    pred[~mask] = np.nan  # no contamination from invalid observations
    result = D.analyze(data_for(pred, target, mask))
    full = result["groups"]["all"]["total_support_n_ge_1"]
    centered = result["groups"]["all"]["centered_support_n_ge_2"]
    assert full["n_valid_elements"] == 8 and full["n_window_joints"] == 4
    assert centered["n_valid_elements"] == 7 and centered["n_window_joints"] == 3
    assert full["means"]["total"] == 1 and centered["means"]["total"] == 1
    assert centered["means"]["centered"] == pytest.approx(0, abs=1e-14)
    moments = D.window_joint_moments(pred, target, mask)
    assert moments["n"][0].tolist() == [2, 1, 0]
    assert moments["vary"][0, 1] == 0 and moments["total"][0, 2] == 0


def test_element_pooling_is_not_equal_window_or_equal_user_ratio():
    y = np.array([[[-1.], [1.], [0.], [0.]], [[-3.], [3.], [-3.], [3.]]])
    mask = np.array([[[1], [1], [0], [0]], [[1], [1], [1], [1]]], dtype=bool)
    p = y.copy(); p[1] = 0
    result = D.analyze(data_for(p, y, mask), batch_windows=1)
    centered = result["groups"]["dynamic"]["centered_support_n_ge_2"]
    assert centered["centered_mse_ratio"] == pytest.approx(36 / 38)
    assert centered["centered_mse_ratio"] != pytest.approx(.5)
    assert centered["means"]["centered"] == pytest.approx(6.)
    assert centered["means"]["vary"] == pytest.approx(38 / 6)
    assert centered["user_centered_gain_signs"] == {"positive": 1, "zero": 1, "negative": 0}


def test_bootstrap_carries_all_user_sums_and_element_counts_in_sorted_order():
    per = {"z": {"n_valid_elements": 6, "sums": {"vary": 15., "centered": 10.}},
           "a": {"n_valid_elements": 2, "sums": {"vary": 2., "centered": 0.}},
           "b": {"n_valid_elements": 8, "sums": {"vary": 32., "centered": 40.}}}
    got = D.bootstrap_centered(per, n_boot=1000)
    rng = np.random.default_rng(20261009)
    ratios, gains = [], []
    for _ in range(1000):
        draw = rng.choice(sorted(per), size=3, replace=True)
        n = sum(per[u]["n_valid_elements"] for u in draw)
        v = sum(per[u]["sums"]["vary"] for u in draw)
        e = sum(per[u]["sums"]["centered"] for u in draw)
        ratios.append(e / v); gains.append((v - e) / n)
    assert got["users"] == ["a", "b", "z"] and got["evaluable"]
    np.testing.assert_allclose(got["ratio_ci95"], np.percentile(ratios, [2.5, 97.5]))
    np.testing.assert_allclose(got["gain_ci95"], np.percentile(gains, [2.5, 97.5]))
    assert D.bootstrap_centered(dict(reversed(list(per.items())))) == got


def test_undefined_bootstrap_never_drops_zero_denominator_replicates():
    per = {"a": {"n_valid_elements": 4, "sums": {"vary": 4., "centered": 1.}},
           "b": {"n_valid_elements": 4, "sums": {"vary": 0., "centered": 0.}}}
    got = D.bootstrap_centered(per)
    assert not got["evaluable"] and got["invalid_replicates"] > 0
    assert not D.bootstrap_centered({"a": per["a"]})["evaluable"]
    data = data_for(np.ones((2, 2, 1)), np.ones((2, 2, 1)))
    centered = D.analyze(data)["groups"]["dynamic"]["centered_support_n_ge_2"]
    assert centered["centered_mse_ratio"] is None and centered["pooled_centered_correlation"] is None
    assert "centered_mse_ratio" in centered["undefined_metrics"]


def test_dynamic_split_uses_cached_global_median_and_nan_only_in_all():
    data = data_for(np.zeros((4, 2, 1)), np.ones((4, 2, 1)), pose_var=[np.nan, 1., 2., 3.])
    result = D.analyze(data)
    assert result["median_pose_var_cached"] == 2 and result["n_dynamic"] == 2
    assert result["groups"]["all"]["total_support_n_ge_1"]["n_windows"] == 4
    assert result["groups"]["static"]["total_support_n_ge_1"]["n_windows"] == 1
    assert result["n_windows_nonfinite_pose_var"] == 1


@pytest.fixture
def cache(tmp_path):
    y = np.broadcast_to(np.array([1., 2., 3., 4.])[None, :, None], (3, 4, 20)).copy()
    data = data_for(y * .5, y, pose_var=[1., 2., 3.])
    data["config"] = {"which": "student", "split": "val", "checkpoint": "/trusted/B/checkpoint.pt", "code_sha": "a" * 40,
                      "pose_stats_sha256": "b" * 64, "checkpoint_pose_stats_sha256": "b" * 64}
    path = tmp_path / "val.npz"
    np.savez_compressed(path, **{k: data[k] for k in ("pred", "target", "mask", "std", "pose_var")},
                        meta_json=json.dumps(data["meta"]), config_json=json.dumps(data["config"]),
                        has_head=np.asarray(True), n_recordings=np.asarray(3), lat=np.zeros((1, 1, 1)))
    digest = D.file_sha256(path)
    prov = {"mode": "report", "cache": str(path), "cache_sha256": digest, "cache_bytes": path.stat().st_size,
            "checkpoint_step": 5000, "checkpoint_sha256": "c" * 64, "checkpoint": data["config"]["checkpoint"],
            "which": "student", "split": "val", "execution_code_sha": "a" * 40, "pose_stats_sha256": "b" * 64,
            "n_recordings_requested": 3, "n_recordings_with_windows": 3, "n_windows": 3,
            "n_windows_with_valid_pose": 3, "n_users_with_valid_pose": 3}
    provenance = tmp_path / "report_provenance.json"
    provenance.write_text(json.dumps(prov))
    return path, provenance, digest, data


def test_cli_selectively_reads_npz_without_lat_and_preserves_inputs(cache, monkeypatch, tmp_path):
    path, provenance, digest, data = cache
    cls = type(np.load(path, allow_pickle=False))
    original, requested = cls.__getitem__, []
    def read(self, key):
        requested.append(key)
        assert key != "lat", "latent array must never be decompressed"
        return original(self, key)
    monkeypatch.setattr(cls, "__getitem__", read)
    out = tmp_path / "result.json"
    args = ["--cache", str(path), "--out", str(out), "--expected-cache-sha256", digest,
            "--report-provenance", str(provenance), "--analysis-source-sha256", D.file_sha256(SCRIPT), "--wrapper-sha256", "d" * 64]
    assert D.main(args) == 0
    result = json.loads(out.read_text())
    assert set(requested) == set(D.CACHE_FIELDS) and D.file_sha256(path) == digest
    assert result["provenance"]["cache_sha256"] == digest
    assert result["groups"]["dynamic"]["centered_support_n_ge_2"]["centered_mse_ratio"] == pytest.approx(.25)
    with pytest.raises(ValueError, match="already exists"):
        D.main(args)


def test_invalid_cache_hash_fails_before_reading_npz(cache, monkeypatch, tmp_path):
    path, provenance, _, _ = cache
    monkeypatch.setattr(D, "load_cache", lambda _: pytest.fail("must not read unauthenticated cache"))
    with pytest.raises(ValueError, match="cache SHA mismatch"):
        D.main(["--cache", str(path), "--out", str(tmp_path / "out.json"), "--expected-cache-sha256", "0" * 64,
                "--report-provenance", str(provenance)])


@pytest.mark.parametrize("key,value", [("checkpoint_step", 4999), ("which", "teacher"), ("pose_stats_sha256", "f" * 64),
                                       ("n_windows_with_valid_pose", 2), ("cache_sha256", "f" * 64)])
def test_provenance_disagreements_fail_closed(cache, tmp_path, key, value):
    path, provenance, digest, _ = cache
    prov = json.loads(provenance.read_text()); prov[key] = value
    provenance.write_text(json.dumps(prov))
    with pytest.raises(ValueError):
        D.main(["--cache", str(path), "--out", str(tmp_path / "out.json"), "--expected-cache-sha256", digest,
                "--report-provenance", str(provenance)])


def test_e1_reference_reproduction_float32_pipeline(cache):
    _, _, _, data = cache
    result = D.analyze(data)
    mask = data["mask"]
    errs = (np.abs(data["pred"] - data["target"]) * data["std"][:, None, :] * mask).sum(axis=(1, 2))
    const = (np.abs(data["target"]) * data["std"][:, None, :] * mask).sum(axis=(1, 2))
    counts = mask.sum(axis=(1, 2)); sel = data["pose_var"] >= np.median(data["pose_var"])
    per = {m["user"]: {"n_windows": 1, "mae": float(errs[i] / counts[i]), "mae_const": float(const[i] / counts[i]),
                       "ratio": float(errs[i] / const[i])} for i, m in enumerate(data["meta"]) if sel[i]}
    block = {"n_users": 2, "n_windows": 2, "mae": float(errs[sel].sum() / counts[sel].sum()),
             "mae_const": float(const[sel].sum() / counts[sel].sum()), "ratio": .5,
             "ratio_ci95_users": [.5, .5], "median_user_ratio": .5, "per_user": per}
    reference = {"n_windows": 3, "n_dynamic": 2, "E1": {"dynamic": block}}
    assert D.check_reference(result, reference)["consistent"]
    block["ratio"] = .6
    with pytest.raises(ValueError, match="reference value mismatch"):
        D.check_reference(result, reference)


def test_valid_nan_rejected_and_empty_support_declared():
    data = data_for(np.ones((2, 2, 1)), np.ones((2, 2, 1)))
    data["pred"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite prediction"):
        D.analyze(data)
    data["mask"][:] = False
    result = D.analyze(data)
    assert not result["e1_dynamic_control"]["evaluable"]
    assert result["groups"]["all"]["centered_support_n_ge_2"]["n_valid_elements"] == 0
    assert not result["groups"]["all"]["bootstrap_centered"]["evaluable"]
