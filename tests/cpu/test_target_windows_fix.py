"""Review di Claude del 18/11/2026 sui target RMS100/ms_fast: spettro rapido a 125 ms (100 ms ritirato) e ricetta storica esplicita
negli strumenti che ricostruiscono run nati prima del 18/11."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wearusfm.model import anchor_targets as A
from wearusfm.model.spectral_targets import (LEGACY_TARGET_VERSION, MS_FAST_WINDOW_MS, RETIRED_TARGET_VERSIONS, TARGET_VERSION,
                                            target_version_for_fast_window)
from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[2]
RATES = (1000, 1024, 1200, 1926, 2000, 2048, 2148, 4000, 4096, 5120, 10240)


def _bins_per_band(fs, window_ms):
    """Bin della FFT con centro in [lo, hi) per le sei bande log-spaziate 20-450 Hz; supporto arrotondato come in
    `anchor_targets._windows` (`round`, meta' al pari: 268 campioni a 2148 Hz)."""
    n = max(1, int(round(window_ms / 1000.0 * fs)))
    freq = np.arange(n // 2 + 1) * fs / n
    edges = np.geomspace(20, 450, 7)
    return [int(((freq >= lo) & (freq < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]


def test_fast125_has_the_same_fft_bins_per_band_at_every_sampling_rate():
    counts = {fs: tuple(_bins_per_band(fs, MS_FAST_WINDOW_MS)) for fs in RATES}
    assert set(counts.values()) == {(2, 3, 4, 8, 14, 23)}, counts
    # Il motivo del ritiro di 100 ms: la banda 20-33,6 Hz aveva 2 bin a 1000/2000 Hz e 1 solo a 1926/2048 Hz.
    old = {fs: _bins_per_band(fs, 100.0) for fs in RATES}
    assert old[1000][0] == old[2000][0] == 2 and old[1926][0] == old[2048][0] == 1


def test_production_fast125_band_one_is_consistent_across_rates():
    """Stesso processo fisico (toni nella banda 1 e nelle altre), target rapido della banda 1 entro 0,05 in log fra 2000 Hz e
    1926/2048 Hz; con il codice di `b4b0963` (100 ms) lo scarto su questo segnale era di -0,61 (review di Claude del 18/11)."""
    def band_one(fs):
        t = np.arange(4 * fs) / fs
        rng = np.random.default_rng(7)
        tones = rng.uniform(21, 33, size=12)
        x = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(tones, rng.uniform(0, 2 * np.pi, size=12)))
        x = x + 0.5 * sum(np.sin(2 * np.pi * f * t) for f in (45.0, 80.0, 130.0, 210.0, 350.0))
        x = (x / x.std()).astype(np.float32)[None, :]
        got = A.anchor_targets(x, fs, np.array([450.0]), multiscale=True)
        return float(np.mean(got.ms_fast[0, got.ms_fast_valid[0], 0]))
    ref = band_one(2000)
    for fs in (1926, 2048):
        assert abs(band_one(fs) - ref) < 0.05


def test_fast100_is_retired_and_125_is_the_new_registered_recipe():
    assert MS_FAST_WINDOW_MS == 125.0 and TARGET_VERSION == "step1_psd_power_shape_energy_fast125_v3"
    assert "step1_psd_power_shape_energy_fast100_v2" in RETIRED_TARGET_VERSIONS
    assert target_version_for_fast_window(125.0) == TARGET_VERSION
    assert target_version_for_fast_window(50.0) == LEGACY_TARGET_VERSION
    with pytest.raises(ValueError, match="ritirato"):
        target_version_for_fast_window(100.0)


def _cli():
    spec = importlib.util.spec_from_file_location("train_jepa_target_fix", ROOT / "scripts" / "train_jepa.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_historical_args_make_the_legacy_recipe_explicit_without_overriding():
    cli = _cli()
    base = ["--preset", "small", "--manifest", "unused", "--root", "unused", "--out-dir", "unused"]
    assert cli.historical_args(base) == base + ["--rms-window-ms", "25", "--ms-fast-window-ms", "50"]
    assert cli.historical_args(base + ["--rms-window-ms", "100"]) == base + ["--rms-window-ms", "100", "--ms-fast-window-ms", "50"]
    assert cli.historical_args(cli.historical_args(base)) == cli.historical_args(base)
    new = cli.build_config(cli.build_parser().parse_args(base))
    old = cli.build_config(cli.build_parser().parse_args(cli.historical_args(base)))
    assert (new.loader.rms_window_ms, new.loader.ms_fast_window_ms, new.target_version) == (100.0, 125.0, TARGET_VERSION)
    assert (old.loader.rms_window_ms, old.loader.ms_fast_window_ms, old.target_version) == (25.0, 50.0, LEGACY_TARGET_VERSION)
    # Un run storico senza i campi nel config e' esattamente la ricetta esplicita.
    historical = R.config_to_dict(old)
    for key in ("rms_window_ms", "ms_fast_window_ms"):
        del historical["loader"][key]
    assert R._config_with_defaults(historical) == R.config_to_dict(old)
    R._check_resume_recipe(old, {"config": historical})
    with pytest.raises(ValueError, match="loader"):
        R._check_resume_recipe(new, {"config": historical})


def test_historical_args_resolves_abbreviations_like_argparse():
    """Un valore gia' scritto con un'abbreviazione accettata da argparse o con `=` non viene sovrascritto (verifica del 18/11)."""
    cli = _cli()
    base = ["--preset", "small", "--manifest", "unused", "--root", "unused", "--out-dir", "unused"]
    def recipe(extra):
        cfg = cli.build_config(cli.build_parser().parse_args(cli.historical_args(base + extra)))
        return cfg.loader.rms_window_ms, cfg.loader.ms_fast_window_ms
    for extra in (["--ms-fast", "125"], ["--ms-fast-window", "125"], ["--ms-fast-window-ms=125"]):
        assert recipe(extra) == (25.0, 125.0), extra
        assert cli.historical_args(base + extra) == base + extra + ["--rms-window-ms", "25"]
    for extra in (["--rms", "100"], ["--rms-window", "100"], ["--rms-window-ms=100"]):
        assert recipe(extra) == (100.0, 50.0), extra
    assert recipe([]) == (25.0, 50.0)


def test_legacy_target_windows_fill_only_the_two_new_fields():
    """I confronti esatti degli strumenti storici completano solo i due campi del 18/11: un altro campo mancante nel config
    salvato resta un errore, mentre `_config_with_defaults` (ripresa) lo completerebbe (review di Claude del 18/11)."""
    cli = _cli()
    base = ["--preset", "small", "--manifest", "unused", "--root", "unused", "--out-dir", "unused"]
    old = R.config_to_dict(cli.build_config(cli.build_parser().parse_args(cli.historical_args(base))))
    stored = copy.deepcopy(old)
    for key in ("rms_window_ms", "ms_fast_window_ms"):
        del stored["loader"][key]
    assert R.with_legacy_target_windows(stored) == old
    missing = copy.deepcopy(stored)
    del missing["model"]["tokenizer_activation"]
    assert "tokenizer_activation" not in R.with_legacy_target_windows(missing)["model"]
    assert R.with_legacy_target_windows(missing) != old
    assert "tokenizer_activation" in R._config_with_defaults(missing)["model"]
    new = R.config_to_dict(cli.build_config(cli.build_parser().parse_args(base)))
    assert R.with_legacy_target_windows(new) == new  # campi scritti: mai sovrascritti
    assert R.with_legacy_target_windows({"model": {}}) == {"model": {}}


def test_retired_fast100_version_is_refused_explicitly(tmp_path):
    from dataclasses import replace
    from wearusfm.training.spectral_calibration import calibration_signature
    retired = replace(R.sanity_config(), target_version="step1_psd_power_shape_energy_fast100_v2")
    manifest = tmp_path / "manifest"
    manifest.write_text("test")
    with pytest.raises(ValueError, match="ritirata"):
        calibration_signature(retired, manifest, {}, None)
    with pytest.raises(ValueError, match="ritirata"):
        R.train(retired, manifest, [], tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


def test_with_historical_targets_pins_the_library_builders():
    cfg = R.with_historical_targets(R.sanity_config())
    assert (cfg.loader.rms_window_ms, cfg.loader.ms_fast_window_ms, cfg.target_version) == (25.0, 50.0, LEGACY_TARGET_VERSION)
    assert R.config_to_dict(R.with_historical_targets(cfg)) == R.config_to_dict(cfg)
    other = R.sanity_config()
    assert (other.loader.rms_window_ms, other.loader.ms_fast_window_ms) == (100.0, 125.0)
