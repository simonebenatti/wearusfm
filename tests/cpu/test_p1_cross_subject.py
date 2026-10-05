"""Metrica P1 (D12): finestre etichettate dai dati processati di NinaPro, ruoli dei soggetti, errore standard della media fra dataset; e lo script
`probe_p1.py` da capo a fondo con un FM piccolo (quella parte richiede torch: si salta se non c'e')."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

try:  # torch prima di sklearn (lo carica il test del bootstrap): sul Mac l'ordine opposto manda in segmentation fault le operazioni di torch
    import torch
except ImportError:
    torch = None

from wearusfm.harness import p1_cross_subject as P1
from wearusfm.ingest import ninapro_std
from wearusfm.ingest.common import montage_to_dict

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_p1.py"
FS = 2000.0


def _session(root: Path, subject: str, seed: int, n_rep: int = 3, gain: float = 1.0):
    """Due esercizi; movimenti 1-3 nel primo, 4-5 nel secondo (numerazione globale come DB2), 2 s ciascuno, 1 s di riposo fra uno e l'altro;
    l'ampiezza del canale k cresce col movimento: classi separabili. Riempimento -1 in coda al secondo esercizio."""
    rng = np.random.default_rng(seed)
    segs, labs, trials, off = [], [], [], 0
    for moves in ((1, 2, 3), (4, 5)):
        x_ex, y_ex = [], []
        for _ in range(n_rep):
            for mv in moves:
                for lab, dur in ((mv, 2.0), (0, 1.0)):
                    n = int(dur * FS)
                    amp = np.ones(12) if lab == 0 else 1.0 + 3.0 * (np.arange(12) == (lab % 12))
                    x_ex.append(rng.normal(size=(n, 12)) * amp * gain)
                    y_ex.append(np.full(n, lab))
        y = np.concatenate(y_ex)
        if moves[0] == 4:
            y[-500:] = -1
        segs.append(np.concatenate(x_ex))
        labs.append(y)
        trials.append({"offset": off, "n_samples": len(y), "exercise": len(trials) + 1})
        off += len(y)
    x, y = np.concatenate(segs), np.concatenate(labs)
    d = root / "ninapro_db2" / subject / "session1"
    d.mkdir(parents=True)
    np.save(d / "data_int16.npy", np.round(x * 1000).astype(np.int16))
    np.savez(d / "labels.npz", restimulus=y.astype(np.int16), stimulus=y.astype(np.int16))
    m = montage_to_dict(ninapro_std.build_montage_metadata(ninapro_std.DB2, int(subject[1:]), "right"), [True] * 12)
    (d / "metadata.json").write_text(json.dumps({"montage": m, "native_fs_hz": FS, "shape": list(x.shape), "int16_scale": 1000.0,
                                                 "trials": trials}))
    return d


def test_session_windows_stay_inside_one_label_and_one_exercise(tmp_path):
    d = _session(tmp_path, "s01", 0)
    sw = P1.session_windows(d, "ninapro_db2", window_s=1.0)
    assert sw.windows.shape[1:] == (2000, 12) and sw.fs == FS
    assert set(sw.labels.tolist()) == {1, 2, 3, 4, 5}  # niente riposo, niente -1
    assert np.bincount(sw.labels)[1:].tolist() == [6, 6, 6, 6, 6]  # 2 finestre da 1 s per ripetizione da 2 s, 3 ripetizioni
    assert np.allclose(sw.windows[0, :3], np.load(d / "data_int16.npy")[:3] / 1000.0, atol=1e-6)  # unita' fisiche
    loud = sw.windows.std(axis=1)  # il canale del movimento e' il piu' ampio
    assert all(int(np.argmax(s)) == lab % 12 for s, lab in zip(loud, sw.labels))


def test_subject_roles_from_the_signed_splits():
    splits = {"datasets": {"ninapro_db2": {"pretraining": [f"s{i:02d}" for i in range(1, 33)], "test": ["s40", "s39"]},
                           "ninapro_db6": {"pretraining": ["s01", "s02"], "test": ["s03"]}, "x": {"pretraining": ["a"], "test": []}}}
    r = P1.subject_roles(splits, "ninapro_db2")
    assert len(r["val"]) == 4 and len(r["train"]) == 28 and r["test"] == ["s39", "s40"]
    assert not set(r["val"]) & set(r["train"]) and r == P1.subject_roles(splits, "ninapro_db2")  # deterministico
    assert P1.subject_roles(splits, "ninapro_db6")["val"] in (["s01"], ["s02"])
    with pytest.raises(ValueError):
        P1.subject_roles(splits, "x")


def test_classes_and_bootstrap_of_the_mean_over_datasets():
    mv, mt = P1.keep_train_classes(np.array([1, 2, 2]), np.array([1, 3]), np.array([2, 2, 4]))
    assert mv.tolist() == [True, False] and mt.tolist() == [True, True, False]
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    s = np.repeat(np.arange(8), 50)
    assert P1.mean_bacc_bootstrap_se([(y, y, s), (y, y, s)]) == 0.0  # perfetto ovunque: nessuna varianza
    noisy = np.where(rng.random(400) < 0.3 + 0.08 * s, rng.integers(0, 4, 400), y)  # errori diversi per soggetto
    se = P1.mean_bacc_bootstrap_se([(y, noisy, s), (y, y, s)])
    assert 0.0 < se < 0.2
    w = P1.dataset_weights([8, 3, 2], "subjects")
    assert w.tolist() == pytest.approx([8 / 13, 3 / 13, 2 / 13]) and P1.dataset_weights([8, 3], "datasets").tolist() == [0.5, 0.5]
    assert P1.mean_bacc_bootstrap_se([(y, noisy, s), (y, y, s)], weights=np.array([0.0, 1.0])) == 0.0  # solo il dataset perfetto


def test_probe_p1_script_end_to_end(tmp_path):
    if torch is None:
        pytest.skip("torch non installato")
    from wearusfm.model.fm import FMConfig, WearUsFM

    root = tmp_path / "processed"
    subjects = [f"s{i:02d}" for i in range(1, 8)]
    for i, s in enumerate(subjects):
        _session(root, s, i, n_rep=2, gain=1.0 + 0.2 * i)
    splits = {"datasets": {"ninapro_db2": {"pretraining": subjects[:5], "test": subjects[5:]}}}
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    cfg = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4)
    torch.manual_seed(0)
    model = WearUsFM(cfg)
    torch.save({"config": {"model": asdict(cfg)}, "teacher": model.state_dict(), "student": model.state_dict()}, tmp_path / "ckpt.pt")
    base = [sys.executable, str(SCRIPT), "--checkpoint", str(tmp_path / "ckpt.pt"), "--root", str(root), "--splits", str(tmp_path / "splits.json"),
            "--datasets", "ninapro_db2", "--device", "cpu", "--batch", "8"]
    r = subprocess.run(base + ["--out", str(tmp_path / "all.json")], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((tmp_path / "all.json").read_text())
    d = rep["datasets"]["ninapro_db2"]
    assert d["n_train"] + d["n_val"] == 5 * 20 and d["n_test"] == 2 * 20 and d["n_classes"] == 5
    assert d["subjects"]["test"] == ["s06", "s07"] and len(d["subjects"]["val"]) == 1
    assert rep["p1"] == pytest.approx(d["test_bacc"]) and rep["p1_se"] >= 0.0
    r1 = subprocess.run(base + ["--stage", "extract", "--features", str(tmp_path / "f")], capture_output=True, text=True, timeout=900)
    assert r1.returncode == 0, r1.stderr[-3000:]
    r2 = subprocess.run(base + ["--stage", "probe", "--features", str(tmp_path / "f"), "--out", str(tmp_path / "two.json")],
                        capture_output=True, text=True, timeout=900)
    assert r2.returncode == 0, r2.stderr[-3000:]
    two = json.loads((tmp_path / "two.json").read_text())
    assert two["p1"] == pytest.approx(rep["p1"]) and two["datasets"]["ninapro_db2"]["c"] == d["c"]  # stesse feature, stesso risultato
