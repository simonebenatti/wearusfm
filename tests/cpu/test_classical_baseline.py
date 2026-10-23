"""Baseline classica (23/10/2026): feature di Hudgins del progetto per canale (`fm_features.td_features`), sonde ufficiali P1/P2 senza modello
(`probe_p1`/`probe_p2 --feature-set td --checkpoint none`), validazione interna (`scripts/classical_baseline.py`, la stessa sonda di
`calibration_probe`) su NinaPro, EPN-612 e UCI-EMG sintetici, launcher; e il checkpoint della rete casuale (`scripts/random_checkpoint.py`)."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import probe_p1  # noqa: E402
import probe_p2  # noqa: E402

from test_campaign_deploy import work  # noqa: E402,F401 -- $WORK finto col repo canonico
from test_p1_cross_subject import _session  # noqa: E402
from test_probe_launchers import _env as _launcher_env  # noqa: E402
from test_probe_p2 import _epn, _uci  # noqa: E402
from wearusfm.harness import features_hudgins as FH  # noqa: E402
from wearusfm.harness import fm_features as FF  # noqa: E402
from wearusfm.harness.feature_sets import cache_record, check_cache  # noqa: E402

spec = importlib.util.spec_from_file_location("classical_baseline", REPO / "scripts" / "classical_baseline.py")
CB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CB)
FS = 2000.0


def _sines():
    t = np.arange(int(2 * FS)) / FS
    rng = np.random.default_rng(0)
    x = np.stack([np.sin(2 * np.pi * 100 * t), 0.2 * rng.standard_normal(len(t)), 3 * np.sin(2 * np.pi * 200 * t)], 1)  # (T, C)
    return np.stack([x, x * 2])


def test_td_features_are_the_project_hudgins_features_on_the_model_signal():
    windows = _sines()
    f = FF.td_features(windows, FS, np.array(["a", "a"]))
    c = 3
    assert f.shape == (2, 4 * c) and np.isfinite(f).all() and FF.TD_FEATURES == ("mav", "zc", "ssc", "wl")
    # stesso calcolo a mano: filtro del pretraining, scala del soggetto, zone di bordo, `features_hudgins` con la soglia in unita' della scala
    scale = FF.subject_scales(windows, np.array(["a", "a"]), FS, (20.0, 450.0))["a"]
    g = int(round(0.1 * FS))
    x = FF._filtered(windows[0], FS, (20.0, 450.0)).astype(np.float64) / scale
    ref = FH.extract_hudgins_features(x[:, g:-g].T, zc_threshold=FF.TD_DEADZONE, ssc_threshold=FF.TD_DEADZONE)
    np.testing.assert_allclose(f[0], ref, rtol=1e-5)
    zc = f[0][c:2 * c]  # ordine [MAV di tutti i canali, ZC, SSC, WL]
    seconds = (len(windows[0]) - 2 * g) / FS
    assert abs(zc[0] / seconds - 200) < 5 and abs(zc[2] / seconds - 400) < 10  # due attraversamenti dello zero per periodo
    np.testing.assert_allclose(f[1][:c], 2 * f[0][:c], rtol=1e-4)  # stessa scala del soggetto: il doppio dell'ampiezza raddoppia MAV
    assert np.array_equal(FF.extract_features(None, windows, FS, {}, np.array(["a", "a"]), feature_set="td"), f)  # nessun modello


def test_channels_discarded_by_the_qc_have_zero_features():
    """Come il modello, che maschera i canali scartati dal QC: qui le loro feature valgono 0 (niente valori di un canale piatto)."""
    windows = _sines()
    montage = FF.myo8_montage("epn612", FS)
    windows8 = np.concatenate([windows, windows, windows[:, :, :2]], 2)  # 8 canali
    full = FF.td_features(windows8, FS, np.array(["a", "a"]), montage)
    montage["groups"][0]["channels"][5]["qc_valid"] = False
    masked = FF.td_features(windows8, FS, np.array(["a", "a"]), montage)
    cols = np.arange(4) * 8 + 5
    assert np.all(masked[:, cols] == 0) and np.any(full[:, cols] != 0)
    keep = np.setdiff1d(np.arange(32), cols)
    np.testing.assert_array_equal(masked[:, keep], full[:, keep])


class _Z(dict):
    files = property(lambda self: list(self))


def test_td_cache_records_its_channels():
    rec = cache_record("td", 32, 8)
    assert int(rec["td_channels"]) == 8 and int(rec["model_dim"]) == 32
    with pytest.raises(ValueError, match="4 per canale"):
        cache_record("td", 30, 8)
    check_cache(_Z(rec), "td", 32)
    with pytest.raises(ValueError, match="4 per canale"):
        check_cache(_Z({k: v for k, v in rec.items() if k != "td_channels"}), "td", 32)


def _p1(tmp_path):
    root = tmp_path / "processed"
    subjects = [f"s{i:02d}" for i in range(1, 11)]
    for i, s in enumerate(subjects):
        _session(root, s, i, n_rep=2, gain=1.0 + 0.2 * i)
    splits = {"datasets": {"ninapro_db2": {"pretraining": subjects[:8], "test": subjects[8:]}}}
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits))
    return root, path, subjects


def test_official_p1_probe_runs_without_a_model(tmp_path):
    root, splits, _ = _p1(tmp_path)
    base = ["--checkpoint", "none", "--root", str(root), "--splits", str(splits), "--datasets", "ninapro_db2", "--feature-set", "td",
            "--device", "cpu", "--window-s", "1", "--val-frac", "0.1", "--weighting", "subjects"]
    assert probe_p1.main(base + ["--stage", "extract", "--features", str(tmp_path / "f"), "--audit-inputs"]) == 0
    with np.load(tmp_path / "f" / "ninapro_db2.npz") as z:
        assert str(z["feature_set"]) == "td" and int(z["td_channels"]) == 12 and z["train_x"].shape[1] == 48
        check_cache(z, "td", z["train_x"].shape[1])
    assert probe_p1.main(base + ["--stage", "probe", "--features", str(tmp_path / "f"), "--out", str(tmp_path / "p1.json")]) == 0
    rep = json.loads((tmp_path / "p1.json").read_text())
    assert rep["feature_set"] == "td" and rep["checkpoint"] == "none" and rep["p1"] > 0.5  # classi separabili per ampiezza


def test_internal_p1_baseline_equals_the_calibration_probe_on_the_same_features(tmp_path, monkeypatch):
    """`classical_baseline --task p1` sui soggetti di pretraining (`CP.pretraining_subjects`) e con la sonda di `calibration_probe.probe`:
    scritte le stesse feature td nel formato della cache della calibrazione, `CP.probe` da' la stessa S, gli stessi fold e le stesse righe."""
    root, splits, subjects = _p1(tmp_path)
    CP = CB._script("calibration_probe")
    monkeypatch.setattr(CB.P1, "P1_DATASETS", ("ninapro_db2",))
    out = tmp_path / "td_p1.json"
    assert CB.main(["--task", "p1", "--root", str(root), "--splits", str(splits), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    d = rep["datasets"]["ninapro_db2"]
    assert d["subjects"] == CP.pretraining_subjects(json.loads(splits.read_text()), "ninapro_db2") == subjects[:8]
    assert rep["differences_from_model"]["window"].startswith("intera") and Path(rep["oof"]).exists()
    x, y, s, input_sha = CB.features_p1([root], "ninapro_db2", d["subjects"], lambda m: None)
    feats = tmp_path / "cp"
    feats.mkdir()
    np.savez(feats / "ninapro_db2.npz", x=x, y=y, s=s, subjects=np.asarray(d["subjects"]), input_sha256=np.asarray(input_sha),
             checkpoint=np.asarray("none"), checkpoint_sha256=np.asarray("0" * 64), encoder_dim=np.asarray(0), protocol=np.asarray(CP.PROTOCOL),
             splits_sha256=np.asarray(CB._sha256(splits)), extraction_recipe=np.asarray(json.dumps(CP.EXTRACTION_RECIPE, sort_keys=True)),
             **cache_record("model", x.shape[1]))
    cp = CP.probe(argparse.Namespace(task="p1", features=feats, datasets=["ninapro_db2"], seed=0, splits=splits, out=tmp_path / "cp.json"),
                  json.loads(splits.read_text()), lambda m: None)
    assert cp["S"] == pytest.approx(rep["S"]) and cp["datasets"]["ninapro_db2"]["oof_rows_sha256"] == d["oof_rows_sha256"]
    assert [f["c"] for f in cp["datasets"]["ninapro_db2"]["folds"]] == [f["c"] for f in d["folds"]]
    assert d["input_sha256"] == cp["datasets"]["ninapro_db2"]["input_sha256"]


def _data(tmp_path, users=24, subjects=20):
    _epn(tmp_path / "epn" / "trainingJSON", users)
    _uci(tmp_path / "uci", subjects)
    return tmp_path / "epn" / "trainingJSON", tmp_path / "uci"


def test_official_p2_probe_runs_without_a_model(tmp_path):
    epn, uci = _data(tmp_path, users=10, subjects=10)
    feats = tmp_path / "feats"
    argv = ["--checkpoint", "none", "--epn-root", str(epn), "--uci-root", str(uci), "--feature-set", "td", "--device", "cpu"]
    assert probe_p2.main(argv + ["--stage", "extract", "--features", str(feats), "--audit-inputs"]) == 0
    for ds in ("epn612", "uci_emg"):
        with np.load(feats / f"{ds}.npz") as z:
            assert str(z["feature_set"]) == "td" and int(z["td_channels"]) == 8 and z["x"].shape[1] == 32
            check_cache(z, "td", z["x"].shape[1])
    out = tmp_path / "p2_td.json"
    assert probe_p2.main(argv + ["--stage", "probe", "--features", str(feats), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["feature_set"] == "td" and rep["checkpoint"] == "none" and 0 <= rep["p2"] <= 1


def test_internal_p2_baseline_uses_the_internal_subjects_and_the_calibration_probe(tmp_path):
    epn, uci = _data(tmp_path)
    out = tmp_path / "td_p2.json"
    assert CB.main(["--task", "p2", "--epn-root", str(epn), "--uci-root", str(uci), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    CP = CB._script("calibration_probe")
    from wearusfm.harness.splits import split_subjects

    for ds, users in (("epn612", [f"trainingJSON/user{i}" for i in range(1, 25)]), ("uci_emg", [f"{i:02d}" for i in range(1, 21)])):
        test = set().union(*(split_subjects(sorted(set(users)), (0.7, 0.1, 0.2), seed=k).test for k in (0, 1, 2)))
        assert rep["datasets"][ds]["subjects"] == CP.p2_internal_subjects(users) and not set(rep["datasets"][ds]["subjects"]) & test
        assert rep["datasets"][ds]["columns"] == 32
    assert rep["protocol"] == CB.PROTOCOL and rep["weights"] == {"epn612": 0.5, "uci_emg": 0.5} and 0 < rep["S"] <= 1
    assert rep["probe"]["k_folds"] == CP.K_FOLDS and rep["probe"]["fold_seed"] == CP.FOLD_SEED
    with np.load(rep["oof"]) as z:  # formato di calibration_probe: confronti accoppiati possibili
        assert sorted(z["datasets"].tolist()) == ["epn612", "uci_emg"] and all(f"{d}_row" in z.files for d in ("epn612", "uci_emg"))


def test_classical_baseline_launcher(work):  # noqa: F811
    root, canon, first, second = work
    env = _launcher_env(root)
    launcher = REPO / "scripts" / "slurm" / "classical_baseline.sbatch"
    out = root / "rapporti" / "td"
    r = subprocess.run(["bash", str(launcher), "p1", str(out)], env=env, capture_output=True, text=True)
    assert "VENV_PYTHON scripts/classical_baseline.py --task p1" in r.stdout, r.stdout + r.stderr
    assert "VENV_PYTHON scripts/probe_p1.py --checkpoint none --feature-set td" in r.stdout and "--weighting subjects" in r.stdout
    for bad, why in (("rel/td", "relativa"), (f"{root}/wearusfm/td", "dentro il codice")):
        r = subprocess.run(["bash", str(launcher), "p2", bad], env=env, capture_output=True, text=True)
        assert r.returncode == 2 and why in r.stdout and "VENV_PYTHON" not in r.stdout
    (out / "vecchio.json").write_text("{}")
    r = subprocess.run(["bash", str(launcher), "p2", str(out)], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "non vuota" in r.stdout and "VENV_PYTHON" not in r.stdout


def test_random_checkpoint_has_the_run_architecture_and_no_trained_weights(tmp_path):
    """`scripts/random_checkpoint.py`: architettura del `config.json` di A, pesi dal seme (riproducibili, diversi fra semi), caricabile come
    un checkpoint allenato (`FF.load_model`), dichiarato come non il passo 0 del run; niente sovrascritture."""
    import random_checkpoint as RC

    cfg = json.loads((REPO / "results/infra/w1arm_20261021/runs/W1ARM-A-30M-s0/config.json").read_text())
    cfg = {**cfg, "model": {**cfg["model"], "dim": 32, "n_heads": 4, "k_latents": 4, "backbone_layers": 1, "local_layers": 1}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "s1" / "checkpoint.pt"
    assert RC.main(["--config", str(path), "--seed", "1", "--out", str(out)]) == 0
    a = torch.load(out, weights_only=False)
    b, c = RC.random_state(cfg, 1), RC.random_state(cfg, 0)
    assert a["step"] == 0 and a["config"] == {**cfg, "seed": 1} and a["random_init"]["config_seed"] == cfg["seed"]
    assert all(torch.equal(a["teacher"][k], b["teacher"][k]) for k in a["teacher"])
    assert any(not torch.equal(a["teacher"][k], c["teacher"][k]) for k in a["teacher"] if a["teacher"][k].is_floating_point())
    prov = json.loads((out.parent / "random_provenance.json").read_text())
    assert prov["seed"] == 1 and prov["is_run_step0"] is False and FF.load_model(str(out), "cpu", "teacher").cfg.dim == 32
    with pytest.raises(ValueError, match="gia' presenti"):
        RC.main(["--config", str(path), "--seed", "1", "--out", str(out)])
