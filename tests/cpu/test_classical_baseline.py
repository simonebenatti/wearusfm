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
    other = json.loads(str(rec["td_recipe"]))
    other["deadzone"] = 0.05  # una cache fatta con un'altra soglia (o bordi, banda, QC) non passa
    with pytest.raises(ValueError, match="ricetta"):
        check_cache(_Z({**rec, "td_recipe": np.asarray(json.dumps(other))}), "td", 32)


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
    root, splits, subjects = _p1(tmp_path)
    meta_path = root / "ninapro_db2" / "s03" / "session1" / "metadata.json"  # un canale scartato dal QC in una sessione vera
    meta = json.loads(meta_path.read_text())
    meta["montage"]["groups"][0]["channels"][4]["qc_valid"] = False
    meta_path.write_text(json.dumps(meta))
    base = ["--checkpoint", "none", "--root", str(root), "--splits", str(splits), "--datasets", "ninapro_db2", "--feature-set", "td",
            "--device", "cpu", "--window-s", "1", "--val-frac", "0.1", "--weighting", "subjects"]
    assert probe_p1.main(base + ["--stage", "extract", "--features", str(tmp_path / "f"), "--audit-inputs"]) == 0
    with np.load(tmp_path / "f" / "ninapro_db2.npz") as z:
        assert str(z["feature_set"]) == "td" and int(z["td_channels"]) == 12 and z["train_x"].shape[1] == 48
        check_cache(z, "td", z["train_x"].shape[1])
        x = np.concatenate([z[f"{r}_x"] for r in ("train", "val", "test")])
        s = np.concatenate([z[f"{r}_s"] for r in ("train", "val", "test")])
        cols = np.arange(4) * 12 + 4
        assert np.all(x[s == "s03"][:, cols] == 0) and np.all(np.any(x[s != "s03"][:, cols] != 0, 1))  # solo la sessione col canale scartato
    with pytest.raises(SystemExit):  # td senza modello: il checkpoint dev'essere «none», e «none» vale solo con td
        probe_p1.main([a if a != "none" else str(tmp_path / "ckpt.pt") for a in base] + ["--stage", "extract", "--features", str(tmp_path / "g")])
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
    assert cp["S"] == pytest.approx(rep["S"]) and cp["S_se"] == pytest.approx(rep["S_se"]) and cp["weights"] == rep["weights"]
    assert cp["datasets"]["ninapro_db2"]["oof_rows_sha256"] == d["oof_rows_sha256"]
    assert [f["c"] for f in cp["datasets"]["ninapro_db2"]["folds"]] == [f["c"] for f in d["folds"]]
    # l'impronta degli ingressi e' quella dell'estrazione della calibrazione (che non dipende dal modello): stesse finestre di A
    import random_checkpoint as RC
    from wearusfm.model.fm import FMConfig

    tiny = {"model": {"dim": 16, "n_heads": 4, "k_latents": 4, "local_layers": 1, "backbone_layers": 1, "decoder_layers": 1,
                      "muscle_dropout": 0.4}, "seed": 0}
    ckpt = tmp_path / "tiny.pt"
    torch.save(RC.random_state(tiny, 0), ckpt)
    assert FMConfig(**tiny["model"]).dim == 16
    cal = tmp_path / "cal"
    CP.extract(argparse.Namespace(checkpoint=str(ckpt), device="cpu", batch=8, root=[root], datasets=["ninapro_db2"], features=cal,
                                  splits=splits), json.loads(splits.read_text()), lambda m: None)
    with np.load(cal / "ninapro_db2.npz") as z:
        assert str(z["input_sha256"]) == d["input_sha256"]


def _fake_p1(monkeypatch, counts, root):
    """Sessioni P1 finte in memoria per piu' dataset di larghezza diversa (DB2 12 colonne, DB6 16 con le colonne vuote 8 e 9 scartate dal QC,
    come `ingest.ninapro_db6.EMPTY_COLUMNS`), al posto
    di `P1.sessions_of` / `P1.session_windows` (il formato dei file di DB6 qui non serve)."""
    from types import SimpleNamespace

    from wearusfm.harness import p1_cross_subject as P1
    from wearusfm.ingest import ninapro_db6, ninapro_std
    from wearusfm.ingest.common import montage_to_dict

    montages = {"ninapro_db2": montage_to_dict(ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"), [True] * 12),
                "ninapro_db6": montage_to_dict(ninapro_db6.build_montage_metadata(1, 1, 1), [True] * 8 + [False] * 2 + [True] * 6)}
    sessions = {}
    for k, (ds, n) in enumerate(counts.items()):
        c = 12 if ds == "ninapro_db2" else 16
        for i in range(n):
            rng = np.random.default_rng(100 * k + i)
            y = np.repeat(np.arange(1, 5), 6)
            amp = 1.0 + 3.0 * (np.arange(c)[None, None, :] == y[:, None, None])
            w = (rng.normal(size=(len(y), int(FS), c)) * amp).astype(np.float32)
            if c == 16:
                w[:, :, 8:10] = 0  # colonne vuote del DB6 vero
            sessions[ds, f"s{i + 1:02d}"] = SimpleNamespace(windows=w, labels=y, fs=FS, montage=montages[ds], session="session1")
            (root / ds / f"s{i + 1:02d}").mkdir(parents=True, exist_ok=True)  # `probe_p1._root_of` cerca la cartella del soggetto
    monkeypatch.setattr(P1, "sessions_of", lambda root, ds, s: [SimpleNamespace(name="session1", ds=ds, s=s)])
    monkeypatch.setattr(P1, "session_windows", lambda sess, ds, window_s=1.0: sessions[ds, sess.s])
    return {ds: [f"s{i + 1:02d}" for i in range(n)] for ds, n in counts.items()}


def test_td_channels_are_kept_per_dataset_in_the_official_p1_probe(tmp_path, monkeypatch):
    """Terza review del critico (23/10): i canali della cache td valgono per dataset; DB6 (16 colonne) dopo DB2 (12) non e' un errore."""
    subjects = _fake_p1(monkeypatch, {"ninapro_db2": 3, "ninapro_db6": 3}, tmp_path)
    args = argparse.Namespace(window_s=1.0, audit_inputs=True, batch=8, device="cpu", feature_set="td")
    out2 = probe_p1.dataset_features(None, [tmp_path], "ninapro_db2", {"train": subjects["ninapro_db2"]}, args, lambda m: None)
    assert args.td_channels == 12 and out2["train"][0].shape[1] == 48
    out6 = probe_p1.dataset_features(None, [tmp_path], "ninapro_db6", {"train": subjects["ninapro_db6"]}, args, lambda m: None)
    assert args.td_channels == 16 and out6["train"][0].shape[1] == 64
    empty = np.concatenate([np.arange(4) * 16 + 8, np.arange(4) * 16 + 9])
    assert np.all(out6["train"][0][:, empty] == 0)  # colonne vuote 8 e 9 di DB6, scartate dal QC
    assert int(cache_record("td", 64, args.td_channels)["td_channels"]) == 16


def test_td_channel_check_refuses_a_width_different_from_the_montage_or_from_the_other_sessions(tmp_path, monkeypatch):
    """Il ramo d'errore del controllo dei canali di `probe_p1.dataset_features` (quarta review del critico): 16 colonne su un montaggio da 12,
    oppure due sessioni dello stesso dataset da 12 e 16 colonne (montaggi giusti), si fermano prima di estrarre."""
    from types import SimpleNamespace

    from wearusfm.harness import p1_cross_subject as P1

    subjects = _fake_p1(monkeypatch, {"ninapro_db2": 2, "ninapro_db6": 1}, tmp_path)
    real = P1.session_windows
    db6 = real(SimpleNamespace(name="session1", s="s01"), "ninapro_db6")
    args = argparse.Namespace(window_s=1.0, audit_inputs=False, batch=8, device="cpu", feature_set="td")
    for case, sw2 in (("montaggio", lambda sw: SimpleNamespace(**{**vars(sw), "windows": np.concatenate([sw.windows, sw.windows[:, :, :4]], 2)})),
                      ("sessioni", lambda sw: db6)):
        monkeypatch.setattr(P1, "session_windows", lambda sess, ds, window_s=1.0, f=sw2: f(real(sess, ds)) if sess.s == "s02" else real(sess, ds))
        with pytest.raises(ValueError, match="diversi dal montaggio o dalle altre sessioni"):
            probe_p1.dataset_features(None, [tmp_path], "ninapro_db2", {"train": subjects["ninapro_db2"]}, args, lambda m: None)


def test_internal_p1_weights_equal_the_calibration_probe_with_two_datasets(tmp_path, monkeypatch):
    """Due dataset P1 con soggetti di test diversi: S, S_se e pesi di `classical_baseline` uguali a quelli di `calibration_probe.probe`."""
    subjects = _fake_p1(monkeypatch, {"ninapro_db2": 9, "ninapro_db6": 7}, tmp_path)
    splits = {"datasets": {"ninapro_db2": {"pretraining": subjects["ninapro_db2"][:6], "test": subjects["ninapro_db2"][6:]},
                           "ninapro_db6": {"pretraining": subjects["ninapro_db6"][:6], "test": subjects["ninapro_db6"][6:]}}}
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits))
    monkeypatch.setattr(CB.P1, "P1_DATASETS", ("ninapro_db2", "ninapro_db6"))
    out = tmp_path / "td.json"
    assert CB.main(["--task", "p1", "--root", str(tmp_path), "--splits", str(path), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    CP = CB._script("calibration_probe")
    feats = tmp_path / "cp"
    feats.mkdir()
    for ds in ("ninapro_db2", "ninapro_db6"):
        pre = CP.pretraining_subjects(splits, ds)
        x, y, s, input_sha = CB.features_p1([tmp_path], ds, pre, lambda m: None)
        np.savez(feats / f"{ds}.npz", x=x, y=y, s=s, subjects=np.asarray(pre), input_sha256=np.asarray(input_sha), checkpoint=np.asarray("none"),
                 checkpoint_sha256=np.asarray("0" * 64), encoder_dim=np.asarray(0), protocol=np.asarray(CP.PROTOCOL),
                 splits_sha256=np.asarray(CB._sha256(path)), extraction_recipe=np.asarray(json.dumps(CP.EXTRACTION_RECIPE, sort_keys=True)),
                 **cache_record("model", x.shape[1]))
    cp = CP.probe(argparse.Namespace(task="p1", features=feats, datasets=["ninapro_db2", "ninapro_db6"], seed=0, splits=path,
                                     out=tmp_path / "cp.json"), splits, lambda m: None)
    assert rep["weights"] == cp["weights"] and rep["weights"]["ninapro_db2"] == pytest.approx(3 / 4)  # 3 contro 1 soggetti di test
    assert cp["S"] == pytest.approx(rep["S"]) and cp["S_se"] == pytest.approx(rep["S_se"])


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
    # stessa S della sonda della calibrazione sulle stesse feature, scritte nel suo formato (P2: soggetti interni, split dei tre semi)
    feats = tmp_path / "cp2"
    feats.mkdir()
    args = argparse.Namespace(epn_root=[epn], uci_root=uci)
    for ds in ("epn612", "uci_emg"):
        x, y, s, input_sha, internal = CB.features_p2(args, ds)
        all_subjects = (sorted(f"trainingJSON/user{i}" for i in range(1, 25)) if ds == "epn612" else [f"{i:02d}" for i in range(1, 21)])
        np.savez(feats / f"{ds}.npz", x=x, y=y, s=s, subjects=np.asarray(internal), all_subjects=np.asarray(all_subjects),
                 input_sha256=np.asarray(input_sha), checkpoint=np.asarray("none"), checkpoint_sha256=np.asarray("0" * 64),
                 encoder_dim=np.asarray(0), protocol=np.asarray(CP.PROTOCOL), task=np.asarray("p2"), splits_sha256=np.asarray(CP.P2_SPLITS_SHA256),
                 extraction_recipe=np.asarray(json.dumps(CP.EXTRACTION_RECIPE_P2, sort_keys=True)), **cache_record("model", x.shape[1]))
    cp = CP.probe(argparse.Namespace(task="p2", features=feats, datasets=["epn612", "uci_emg"], seed=0, splits=None, out=tmp_path / "cp2.json"),
                  None, lambda m: None)
    assert cp["S"] == pytest.approx(rep["S"]) and cp["S_se"] == pytest.approx(rep["S_se"]) and cp["weights"] == rep["weights"]
    # impronta degli ingressi uguale a quella dell'estrazione P2 della calibrazione (che non dipende dal modello): stesse finestre di A
    import random_checkpoint as RC

    tiny = {"model": {"dim": 16, "n_heads": 4, "k_latents": 4, "local_layers": 1, "backbone_layers": 1, "decoder_layers": 1,
                      "muscle_dropout": 0.4}, "seed": 0}
    torch.save(RC.random_state(tiny, 0), tmp_path / "tiny.pt")
    cal = tmp_path / "cal2"
    CP.extract_p2(argparse.Namespace(checkpoint=str(tmp_path / "tiny.pt"), device="cpu", batch=8, datasets=["epn612", "uci_emg"],
                                     epn_root=[epn], uci_root=uci, features=cal), lambda m: None)
    for ds in ("epn612", "uci_emg"):
        with np.load(cal / f"{ds}.npz") as z:
            assert str(z["input_sha256"]) == rep["datasets"][ds]["input_sha256"]


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
