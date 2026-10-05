"""`scripts/probe_p2.py` (metrica P2 di D12) da capo a fondo su EPN-612 e UCI-EMG sintetici con un FM piccolo: stessi split della replica,
aggregazione sui 3 split, fasi separate uguali alla fase unica. Richiede torch (si salta se non c'e')."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

try:  # torch prima di sklearn: sul Mac l'ordine opposto manda in segmentation fault le operazioni di torch
    import torch
except ImportError:
    torch = None

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_p2.py"
sys.path.insert(0, str(SCRIPT.parent))


def _epn(root: Path, n_users: int):
    rng = np.random.default_rng(0)
    for u in range(1, n_users + 1):
        samples = {}
        for i, g in enumerate(("fist", "open", "waveIn") * 2):
            amp = 1.0 + 3.0 * (np.arange(8) == ("fist", "open", "waveIn").index(g))
            samples[f"idx_{i}"] = {"gestureName": g, "emg": {f"ch{c + 1}": (rng.normal(size=1000) * amp[c] * 5).round().tolist() for c in range(8)}}
        d = root / f"user{u}"
        d.mkdir(parents=True)
        (d / f"user{u}.json").write_text(json.dumps({"generalInfo": {"samplingFrequencyInHertz": 200}, "trainingSamples": samples}))


def _uci(root: Path, n_subjects: int):
    rng = np.random.default_rng(1)
    for s in range(1, n_subjects + 1):
        d = root / f"{s:02d}"
        d.mkdir(parents=True)
        lab = np.repeat([0, 1, 0, 2, 0, 3], 2000)
        rows = ["time\tch1\tch2\tch3\tch4\tch5\tch6\tch7\tch8\tclass"]
        for i, c in enumerate(lab):
            amp = 1.0 + 3.0 * (np.arange(8) == c)
            codes = np.round(rng.normal(size=8) * amp * 5)
            rows.append("\t".join([str(i)] + [f"{v * 1e-5:.5f}" for v in codes] + [str(int(c))]))
        (d / "1_raw_data.txt").write_text("\n".join(rows) + "\n")


def test_probe_p2_end_to_end(tmp_path):
    if torch is None:
        pytest.skip("torch non installato")
    from wearusfm.model.fm import FMConfig, WearUsFM

    _epn(tmp_path / "epn" / "trainingJSON", 10)
    _uci(tmp_path / "uci", 10)
    cfg = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4)
    torch.manual_seed(0)
    model = WearUsFM(cfg)
    torch.save({"config": {"model": asdict(cfg)}, "teacher": model.state_dict(), "student": model.state_dict()}, tmp_path / "ckpt.pt")
    base = [sys.executable, str(SCRIPT), "--checkpoint", str(tmp_path / "ckpt.pt"), "--epn-root", str(tmp_path / "epn" / "trainingJSON"),
            "--uci-root", str(tmp_path / "uci"), "--device", "cpu", "--batch", "8"]
    r = subprocess.run(base + ["--out", str(tmp_path / "all.json")], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((tmp_path / "all.json").read_text())
    for ds, n_win in (("epn612", 60), ("uci_emg", 60)):
        d = rep["datasets"][ds]
        assert len(d["splits"]) == 3 and [s["split_seed"] for s in d["splits"]] == [0, 1, 2]
        assert all(s["n_train"] + s["n_val"] + s["n_test"] == n_win for s in d["splits"])
        assert d["test_bacc"] == pytest.approx(np.mean([s["test_bacc"] for s in d["splits"]]))
    assert rep["datasets"]["uci_emg"]["classes"] == ["1", "2", "3"]
    assert rep["p2"] == pytest.approx((rep["datasets"]["epn612"]["test_bacc"] + rep["datasets"]["uci_emg"]["test_bacc"]) / 2)
    feat = tmp_path / "f"
    r1 = subprocess.run(base + ["--stage", "extract", "--features", str(feat)], capture_output=True, text=True, timeout=900)
    assert r1.returncode == 0, r1.stderr[-3000:]
    r2 = subprocess.run(base + ["--stage", "probe", "--features", str(feat), "--out", str(tmp_path / "two.json")], capture_output=True, text=True,
                        timeout=900)
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert json.loads((tmp_path / "two.json").read_text())["p2"] == pytest.approx(rep["p2"])


def test_split_aggregation_does_not_shrink_the_error():
    pytest.importorskip("sklearn")
    import probe_p2

    rng = np.random.default_rng(0)
    subjects = np.repeat([f"u{i}" for i in range(20)], 12)
    labels = np.tile(np.repeat(["a", "b", "c"], 4), 20)
    x = rng.normal(size=(240, 5)) + 2.0 * (labels[:, None] == np.array(["a", "b", "c", "a", "b"])[None, :])
    d = probe_p2.probe_dataset(x, labels, subjects, (0, 1, 2))
    se = np.array([s["test_bacc_se"] for s in d["splits"]])
    assert d["test_bacc_se"] == pytest.approx(float(np.sqrt((se ** 2).mean())))
    assert d["test_bacc_se"] >= se.min()
