"""Replica di NeuroRVQ (passo 5): lo script gira da capo a fondo su dati sintetici con un FM inizializzato a caso (i pesi veri sono su Leonardo).
Richiede torch, einops e yaml: si salta se mancano."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("yaml")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "replicate_neurorvq.py"


def _mod():
    spec = importlib.util.spec_from_file_location("replicate_neurorvq", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_preprocessing_and_schedule():
    m = _mod()
    x = np.random.default_rng(0).normal(size=(3, 1000, 8))
    xn = m.preprocess_native(x, 200.0)
    assert xn.shape == (3, 8, 1000) and xn.dtype == np.float32
    assert m.resample_1k(xn, 200.0).shape == (3, 8, 5000) and m.resample_1k(xn[..., :999], 1000.0).shape == (3, 8, 800)
    lrs = [m.lr_at(s, 10, 100, 1e-3) for s in (0, 25, 50, 51, 500, 999, 1000)]
    assert lrs[0] == pytest.approx(1e-5) and lrs[2] == pytest.approx(1e-3) and lrs[3] < 1e-3 and lrs[-1] == pytest.approx(1e-6, abs=1e-9)
    w = m.class_weights(np.array([0, 0, 0, 1]), 3)
    assert w[0] < w[1] and w[2] == 0.0 and w.sum() == pytest.approx(2.0)


def test_end_to_end_on_synthetic_epn(tmp_path):
    m = _mod()
    rng = np.random.default_rng(0)
    for folder in ("trainingJSON", "testingJSON"):
        for u in range(5):
            samples = {}
            for i in range(4):
                g = ("fist", "open")[i % 2]
                amp = 1.0 if g == "fist" else 3.0
                samples[f"idx_{i}"] = {"gestureName": g, "emg": {f"ch{c}": (rng.normal(size=1000) * amp).tolist() for c in range(1, 9)}}
            d = tmp_path / folder / f"user{u}"
            d.mkdir(parents=True)
            (d / f"user{u}.json").write_text(json.dumps({"generalInfo": {"samplingFrequencyInHertz": 200}, "trainingSamples": samples}))
    ckpt = tmp_path / "fm.pt"
    torch.save({}, ckpt)  # FM a caso: le chiavi mancanti finiscono nel report
    out = tmp_path / "rep.json"
    assert m.main(["--dataset", "epn612", "--root", str(tmp_path / "trainingJSON"), "--root", str(tmp_path / "testingJSON"), "--checkpoint",
                   str(ckpt), "--seeds", "0", "--epochs", "1", "--batch", "8", "--device", "cpu", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    r = rep["runs"][0]
    assert rep["classes"] == ["fist", "open"] and r["subjects"] == {"train": 7, "val": 1, "test": 2} and len(r["curve"]) == 1
    assert 0.0 <= rep["mean_test_acc"] <= 100.0 and r["missing_keys"] and rep["tolerance"] == [92.65, 96.65]
