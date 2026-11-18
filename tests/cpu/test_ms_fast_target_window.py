"""Fast spectral targets 125 ms (18/11/2026; 100 ms retired): independent Fourier oracle, identity and historical resume."""

import importlib.util
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from wearusfm.data import pretraining_loader as L
from wearusfm.model import anchor_targets as A
from wearusfm.model.spectral_targets import LEGACY_TARGET_VERSION, TARGET_VERSION
from wearusfm.training import run as R
from wearusfm.training.spectral_calibration import _loader_signature, calibration_signature


@pytest.mark.parametrize("fs", [200, 1000, 2000, 2048])
def test_fast125_matches_raw_index_and_fourier_oracle(fs):
    x = np.random.default_rng(62).normal(size=(2, 2 * fs)).astype(np.float32)
    limit = np.array([min(450, fs / 2), min(70, fs / 2)])
    got = A.anchor_targets(x, fs, limit, multiscale=True)
    n = int(np.floor(fs * 0.125 + 0.5))
    starts = np.array([(fs * (2 * p + 1) - 40 * n + 40) // 80 for p in range(80)])
    valid = (starts >= 0) & (starts + n <= x.shape[1])
    edges = np.geomspace(20, 450, 7)
    freq = np.arange(n // 2 + 1) * fs / n
    bins = [(freq >= lo) & (freq < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    available = np.array([[hi <= lim and mask.any() for hi, mask in zip(edges[1:], bins)] for lim in limit])
    assert got.ms_fast.shape == (2, 80, 6)
    np.testing.assert_array_equal(got.ms_fast_available, available)
    np.testing.assert_array_equal(got.ms_fast_valid, valid[None, :] & available.any(1)[:, None])
    # Independent numpy FFT in float64; no calls to production windows/PSD/band helpers.
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    for p in np.flatnonzero(valid):
        segment = x[:, starts[p]:starts[p] + n].astype(np.float64)
        fft = np.fft.rfft(segment * hann, axis=-1)
        density = np.abs(fft) ** 2 / (fs * np.sum(hann ** 2))
        density[:, 1:-1 if n % 2 == 0 else None] *= 2
        expected = np.log(np.stack([density[:, b].sum(1) * fs / n for b in bins], axis=1) + 1e-8)
        np.testing.assert_allclose(got.ms_fast[:, p][available], expected[available], rtol=0, atol=3e-5)
    if fs == 200:
        assert n == 25 and np.all(np.diff(starts) == 5) and np.all(n - np.diff(starts) == 20)
        assert available[0].tolist() == [True, True, True, False, False, False]
    old = A.anchor_targets(x, fs, limit, multiscale=True, ms_fast_window_ms=50)
    for f in fields(got):
        if not f.name.startswith("ms_fast"):
            np.testing.assert_array_equal(getattr(got, f.name), getattr(old, f.name))
    assert not np.allclose(got.ms_fast[:, valid], old.ms_fast[:, valid])


def test_real_loader_wires_fast_support_without_changing_inputs(tmp_path):
    from test_pretraining_loader import _tree

    root, manifest = _tree(tmp_path)
    index = L.ManifestIndex.load(manifest, [root])
    cfg = replace(L.signed_config(filter_band_hz=None), multiscale_anchor=True)
    new = L.PretrainLoader(index, cfg).batch(3, np.random.default_rng(19))
    old = L.PretrainLoader(index, replace(cfg, ms_fast_window_ms=50)).batch(3, np.random.default_rng(19))
    for i, x in enumerate(new.signals):
        np.testing.assert_array_equal(x, old.signals[i])
        expected = A.anchor_targets(x, new.fs[i], min(new.fs[i] / 2, 450), multiscale=True)
        np.testing.assert_array_equal(new.anchor_targets[i].ms_fast, expected.ms_fast)
        np.testing.assert_array_equal(new.anchor_targets[i].ms_slow, old.anchor_targets[i].ms_slow)
        assert not np.allclose(new.anchor_targets[i].ms_fast, old.anchor_targets[i].ms_fast)
    np.testing.assert_array_equal(new.visible, old.visible)


def test_legacy_recipe_is_pinned_and_new_support_cannot_resume_it(tmp_path):
    cfg = R.sanity_config()
    old = replace(cfg, loader=replace(cfg.loader, rms_window_ms=25, ms_fast_window_ms=50), target_version=LEGACY_TARGET_VERSION)
    historical = R.config_to_dict(old)
    del historical["loader"]["rms_window_ms"]
    del historical["loader"]["ms_fast_window_ms"]
    R._check_resume_recipe(old, {"config": historical})
    with pytest.raises(ValueError, match="loader"):
        R._check_resume_recipe(cfg, {"config": historical})
    assert "ms_fast_window_ms" not in _loader_signature(old.loader)
    assert _loader_signature(cfg.loader)["ms_fast_window_ms"] == 125
    manifest = tmp_path / "manifest"
    manifest.write_text("test")
    a, b = [calibration_signature(c, manifest, {}, None) for c in (old, cfg)]
    assert a["target_version"] == LEGACY_TARGET_VERSION and b["target_version"] == TARGET_VERSION
    with pytest.raises(ValueError, match="target_version"):
        calibration_signature(replace(old, target_version=TARGET_VERSION), manifest, {}, None)
    with pytest.raises(ValueError, match="target_version"):
        R.train(replace(old, target_version=TARGET_VERSION), manifest, [], tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


@pytest.mark.parametrize("bad", [0, -50, 75, 100, float("nan"), float("inf")])  # 100: ritirato il 18/11
def test_unregistered_fast_window_rejected(bad):
    with pytest.raises(ValueError, match="ms_fast_window_ms"):
        replace(L.signed_config(), ms_fast_window_ms=bad)
    with pytest.raises(ValueError, match="ms_fast_window_ms"):
        A.anchor_targets(np.ones((1, 200)), 200, 100, ms_fast_window_ms=bad)


def test_cli_fast125_default_and_explicit_legacy_recipe():
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_jepa.py"
    spec = importlib.util.spec_from_file_location("train_ms_fast_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = ["--preset", "small", "--manifest", "unused", "--root", "unused", "--out-dir", "unused"]
    cfg = cli.build_config(cli.build_parser().parse_args(args))
    assert cfg.loader.rms_window_ms == 100 and cfg.loader.ms_fast_window_ms == 125
    assert cfg.target_version == TARGET_VERSION
    old = cli.build_config(cli.build_parser().parse_args(args + ["--rms-window-ms", "25", "--ms-fast-window-ms", "50"]))
    assert old.loader.ms_fast_window_ms == 50 and old.loader.rms_window_ms == 25 and old.target_version == LEGACY_TARGET_VERSION


def test_historical_stats_baseline_remains_50ms():
    from wearusfm.harness.fm_features import spectral_stats
    from wearusfm.model.spectral_step1 import global_spectral_targets

    x = np.random.default_rng(31).normal(size=(2, 4000)).astype(np.float32)
    limit, good = np.full(2, 450.0), np.ones(2, bool)
    got = spectral_stats(x, 2000, limit, good, 200)
    old = A.anchor_targets(x, 2000, limit, multiscale=True, guard=(200, 200), ms_fast_window_ms=50)
    expected = global_spectral_targets([old], [2], good)
    np.testing.assert_array_equal(got[0], expected[0][0])
    np.testing.assert_array_equal(got[1], expected[1][0])
