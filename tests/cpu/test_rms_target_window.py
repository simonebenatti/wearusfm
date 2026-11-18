"""RMS100: raw-index oracle, unchanged companion targets, loader wiring and legacy resume."""

import importlib.util
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from wearusfm.data import pretraining_loader as L
from wearusfm.model import anchor_targets as A
from wearusfm.training import run as R
from wearusfm.training.spectral_calibration import _loader_signature


@pytest.mark.parametrize("fs", [200, 1000, 2000, 2048])
def test_default_rms100_matches_independent_raw_index_oracle(fs):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(2, 2 * fs)) * np.linspace(0.1, 3, 2 * fs)
    got = A.anchor_targets(x, fs, min(450, fs / 2), multiscale=True)
    n = round(fs / 10)  # 100 ms; independent rational arithmetic for the patch centers
    starts = np.array([(fs * (2 * p + 1) - 40 * n + 40) // 80 for p in range(80)])
    valid = (starts >= 0) & (starts + n <= x.shape[1])
    assert got.log_rms.shape == (2, 80)  # still one target every 25 ms
    assert np.array_equal(got.rms_valid, np.broadcast_to(valid, (2, 80)))
    for p in np.flatnonzero(valid):
        raw = x[:, starts[p]:starts[p] + n]
        expected = np.log(np.sqrt(np.mean(raw * raw, axis=1)) + 1e-8)
        np.testing.assert_allclose(got.log_rms[:, p], expected, rtol=0, atol=1e-11)
    if fs == 200:
        assert n == 20 and np.all(np.diff(starts) == 5)
        assert np.all(n - np.diff(starts) == 15)  # 75 ms overlap, no alternating 4/6-sample hop

    legacy = A.anchor_targets(x, fs, min(450, fs / 2), multiscale=True, rms_window_ms=25)
    assert not np.allclose(got.log_rms[:, valid], legacy.log_rms[:, valid])
    for f in fields(got):
        if f.name not in ("log_rms", "rms_valid"):
            np.testing.assert_array_equal(getattr(got, f.name), getattr(legacy, f.name))


def test_loader_uses_configured_rms_support_and_preserves_input_stream(tmp_path):
    from test_pretraining_loader import _tree

    root, manifest = _tree(tmp_path)
    index = L.ManifestIndex.load(manifest, [root])
    cfg = L.signed_config(filter_band_hz=None)
    assert cfg.rms_window_ms == 100 and cfg.patch_ms == 25
    new = L.PretrainLoader(index, cfg).batch(3, np.random.default_rng(19))
    old = L.PretrainLoader(index, replace(cfg, rms_window_ms=25)).batch(3, np.random.default_rng(19))
    for i, (x, previous) in enumerate(zip(new.signals, old.signals)):
        np.testing.assert_array_equal(x, previous)
        expected = A.anchor_targets(x, new.fs[i], min(new.fs[i] / 2, 450), rms_window_ms=100)
        np.testing.assert_array_equal(new.anchor_targets[i].log_rms, expected.log_rms)
        assert not np.allclose(new.anchor_targets[i].log_rms, old.anchor_targets[i].log_rms)
    np.testing.assert_array_equal(new.visible, old.visible)


@pytest.mark.parametrize("bad", [0, -25, float("nan"), float("inf")])
def test_invalid_rms_support_is_rejected(bad):
    with pytest.raises(ValueError, match="rms_window_ms"):
        A.anchor_targets(np.ones((1, 200)), 200, 100, rms_window_ms=bad)
    with pytest.raises(ValueError, match="rms_window_ms"):
        replace(L.signed_config(), rms_window_ms=bad)


def test_checkpoint_without_rms_field_retains_25ms_and_cannot_resume_at_100ms():
    legacy_cfg = R.sanity_config()
    legacy_cfg = replace(legacy_cfg, loader=replace(legacy_cfg.loader, rms_window_ms=25))
    legacy_json = R.config_to_dict(legacy_cfg)
    del legacy_json["loader"]["rms_window_ms"]
    state = {"config": legacy_json}
    R._check_resume_recipe(legacy_cfg, state)
    assert "rms_window_ms" not in legacy_json["loader"]  # normalisation does not mutate source evidence
    with pytest.raises(ValueError, match="loader"):
        R._check_resume_recipe(R.sanity_config(), state)
    assert "rms_window_ms" not in _loader_signature(legacy_cfg.loader)
    assert _loader_signature(R.sanity_config().loader)["rms_window_ms"] == 100
    R._check_resume_recipe(R.sanity_config(), {"config": R.config_to_dict(R.sanity_config())})


def test_training_cli_defaults_to_100_and_explicit_legacy_option_is_independent_of_tokenizer():
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_jepa.py"
    spec = importlib.util.spec_from_file_location("train_jepa_rms_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = ["--preset", "small", "--manifest", "unused", "--root", "unused", "--out-dir", "unused"]
    cfg = cli.build_config(cli.build_parser().parse_args(args))
    assert R.config_to_dict(cfg)["loader"]["rms_window_ms"] == 100
    old = cli.build_config(cli.build_parser().parse_args(args + ["--rms-window-ms", "25", "--window-ms", "100"]))
    assert old.loader.rms_window_ms == 25 and old.model.window_ms == 100
    with pytest.raises(ValueError, match="rms-window-ms"):
        cli.build_config(cli.build_parser().parse_args(args + ["--rms-window-ms", "nan"]))
