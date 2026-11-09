"""Catena della replica di TEMPONet su dati finti, su CPU: cache dai .mat, training dei tre modelli, sintesi e verdetto."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import temponet_db8  # noqa: E402
import temponet_db8_cache  # noqa: E402
import temponet_db8_summary  # noqa: E402


def _raw_root(tmp_path, subjects=(1,), n=3000):
    root = tmp_path / "raw"
    (root / "DB8").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for s in subjects:
        for a in (1, 2, 3):
            t = np.arange(n) / 2000.0
            glove = (45 + 30 * np.sin(2 * np.pi * 0.5 * t)[:, None] * np.ones((1, 18))).astype(np.float32)
            emg = (rng.normal(0, 5e-5, size=(n, 16)) * (1 + np.abs(np.sin(2 * np.pi * 0.5 * t)))[:, None]).astype(np.float32)
            lab = np.zeros((n, 1), dtype=np.int8)
            sio.savemat(root / "DB8" / f"S{s}_E1_A{a}.mat", {"emg": emg, "glove": glove, "restimulus": lab, "rerepetition": lab})
    return root


@pytest.fixture()
def cache(tmp_path):
    out = tmp_path / "cache"
    assert temponet_db8_cache.main(["--raw-root", str(_raw_root(tmp_path)), "--out", str(out), "--workers", "1", "--subjects", "1"]) == 0
    return out


def test_cache_manifest(cache):
    m = json.loads((cache / "manifest.json").read_text())
    assert m["kind"] == "temponet_db8_cache_v1" and len(m["entries"]) == 3
    e = m["entries"][0]
    assert e["n_samples"] == 3000 and e["outputs"]["emg"]["shape"] == [3000, 16] and e["outputs"]["doa"]["dtype"] == "float32"
    assert len(e["source_sha256"]) == 64
    # una cartella non vuota non si riusa
    assert temponet_db8_cache.main(["--raw-root", str(cache), "--out", str(cache), "--workers", "1"]) == 2


def test_training_and_summary_end_to_end(cache, tmp_path):
    out = tmp_path / "run"
    args = ["--cache", str(cache), "--out", str(out), "--subjects", "1", "--seeds", "0", "--epochs", "2", "--batch", "16",
            "--workers", "1", "--device", "cpu"]
    assert temponet_db8.main(args) == 0
    unit = json.loads((out / "units" / "S1_seed0.json").read_text())
    assert set(unit["models"]) == {"fold1", "fold2", "full"}
    f1, full = unit["models"]["fold1"], unit["models"]["full"]
    assert f1["train_acquisitions"] == [1] and f1["val_acquisition"] == 2 and full["train_acquisitions"] == [1, 2]
    assert len(f1["history"]) == 2 and "val_mae_raw_deg" in f1["history"][0]
    assert full["n_train_windows"] == 2 * f1["n_train_windows"]
    for m in unit["models"].values():
        assert np.isfinite(m["test_ema"]["mae_deg"]) and len(m["alpha"]) == 5
    assert (out / "models" / "S1_seed0_full.pt").is_file()
    assert np.load(out / "predictions" / "S1_seed0_fold2_test_raw.npy").shape == (np.load(out / "predictions" / "S1_test_target.npy").shape)
    run = json.loads((out / "run.json").read_text())
    assert run["model"]["parameters"] == 70_917 and run["status"] == {"S1_seed0": "ok"}
    assert temponet_db8_summary.main([str(out), "--seeds", "0"]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["verdict"] == "incompleta" and not s["complete"]
    expected = np.mean([unit["models"][f]["test_ema"]["mae_deg"] for f in ("fold1", "fold2")])
    assert s["primary_mae_deg"] == pytest.approx(expected)
    # la cartella delle uscite deve essere nuova
    assert temponet_db8.main(args) == 2


def test_cache_tampering_is_detected(cache, tmp_path):
    arr = np.load(cache / "S1_A1_emg.npy")
    arr[0, 0] += 1.0
    np.save(cache / "S1_A1_emg.npy", arr)
    out = tmp_path / "run"
    assert temponet_db8.main(["--cache", str(cache), "--out", str(out), "--subjects", "1", "--seeds", "0", "--epochs", "1",
                              "--batch", "16", "--workers", "1", "--device", "cpu"]) == 1
    assert "sha256" in (out / "units" / "S1_seed0.error").read_text()


def _unit(subject, seed, fold_mae, full_mae=7.0):
    def model(mae):
        return {"test_ema": {"mae_deg": mae, "mae_per_doa_deg": [mae] * 5, "acc_10deg": 0.8, "acc_15deg": 0.9},
                "test_raw": {"mae_deg": mae + 1, "mae_per_doa_deg": [mae + 1] * 5}, "alpha": [0.86] * 5}
    return {"subject": subject, "seed": seed, "models": {"fold1": model(fold_mae), "fold2": model(fold_mae), "full": model(full_mae)}}


@pytest.mark.parametrize("mae,verdict", [(6.89, "replicata"), (7.19, "replicata"), (7.25, "non replicata"),
                                         (6.50, "non coerente (migliore del pubblicato)")])
def test_summary_verdicts(mae, verdict):
    units = [_unit(s, k, mae) for s in range(1, 13) for k in (0, 1, 2)]
    s = temponet_db8_summary.summarize(units, [0, 1, 2])
    assert s["complete"] and s["verdict"] == verdict
    assert s["primary_mae_deg"] == pytest.approx(mae) and s["full_mae_deg"] == pytest.approx(7.0)
    assert s["alpha_mean"] == pytest.approx(0.86) and s["seed_sd_deg"] == pytest.approx(0.0)


def test_summary_incomplete_when_a_seed_is_missing():
    units = [_unit(s, k, 6.9) for s in range(1, 13) for k in (0, 1)]
    assert temponet_db8_summary.summarize(units, [0, 1, 2])["verdict"] == "incompleta"
