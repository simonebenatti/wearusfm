"""Readout per canale (23/10/2026; `harness.channel_readout`, `scripts/channel_readout.py`, launcher): la stessa passata in avanti di
`encode`, il readout medio uguale a quello delle sonde, canali del QC a 0, il decoder per canale, la sonda uguale a `calibration_probe` a
larghezza intera e la riduzione a larghezza pari, il confronto accoppiato con la regola, la P2 interna e la regola dei checkout nei launcher."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import random_checkpoint as RC  # noqa: E402

from test_campaign_deploy import work  # noqa: E402,F401 -- $WORK finto col repo canonico
from test_classical_baseline import _p1  # noqa: E402
from test_probe_launchers import _env as _launcher_env  # noqa: E402
from test_probe_p2 import _epn, _uci  # noqa: E402
from wearusfm.harness import channel_readout as CR  # noqa: E402
from wearusfm.harness import fm_features as FF  # noqa: E402
from wearusfm.harness.probe import linear_probe  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model.channel_identity import codes_to_tensors  # noqa: E402
from wearusfm.model.fm import FMConfig, ModelInputs, WearUsFM  # noqa: E402
from wearusfm.model.local_encoder import sets_to_tensors  # noqa: E402

spec = importlib.util.spec_from_file_location("channel_readout_script", REPO / "scripts" / "channel_readout.py")
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)
CP = SC.CP
SLURM = REPO / "scripts" / "slurm"
CFG = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4)
TINY = {"model": {"dim": 16, "n_heads": 4, "k_latents": 4, "local_layers": 1, "backbone_layers": 1, "decoder_layers": 1, "muscle_dropout": 0.4},
        "seed": 0}


def _model():
    torch.manual_seed(0)
    return WearUsFM(CFG).eval()


def _windows(n=6, t=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, t, 8)) * np.linspace(0.5, 2.0, 8)[None, None, :], np.array(["a", "a", "a", "b", "b", "b"][:n])


def _inputs(model, windows, fs, montage, n):
    layout, code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
    sets = CC.attention_sets(layout, 8)
    sig = [torch.from_numpy(np.ascontiguousarray(w.T).astype(np.float32)) for w in windows[:n]]
    return ModelInputs(sig, [fs] * n, codes_to_tensors(CC.pack_codes([code] * n)), sets_to_tensors(CC.pack_attention_sets([sets] * n)),
                       [layout.n_channels] * n, torch.as_tensor(layout.qc_valid).repeat(n))


def test_forward_layers_is_the_unmasked_encode_and_dec_is_the_decoder_per_channel():
    model = _model()
    m = FF.myo8_montage("epn612", 200.0)
    w, _ = _windows()
    inp = _inputs(model, w, 200.0, m, 2)
    with torch.no_grad():
        enc_ref = model.encode(inp, None)
        enc, dec = CR.forward_layers(model, inp)
    assert torch.allclose(enc.local, enc_ref.local, atol=1e-6) and torch.allclose(enc.z, enc_ref.z, atol=1e-6)
    assert torch.equal(enc.visible, enc_ref.visible) and torch.equal(enc.key_time_valid, enc_ref.key_time_valid)
    assert enc.local.shape == dec.shape == (16, w.shape[1] // 5, 16)  # 2 finestre x 8 canali, patch da 25 ms a 200 Hz
    assert CR.LAYERS == ("pool", "local", "dec") and CR.CHANNEL_LAYERS == ("local", "dec")
    # dec: il decoder interrogato una (canale, patch) alla volta da' lo stesso valore
    with torch.no_grad():
        for c, t in ((0, 3), (9, 7), (15, 0)):
            s = 0 if c < 8 else 1
            one = model.decode(enc, torch.tensor([c]), torch.tensor([t]), torch.tensor([0] * (s + 1) + [1] * (2 - s)))
            assert torch.allclose(one[0], dec[c, t], atol=1e-5)


def test_extraction_pool_equals_the_probe_readout_and_channel_layers_have_channel_times_dim_columns():
    model = _model()
    m = FF.myo8_montage("epn612", 200.0)
    w, s = _windows()
    out = CR.extract_channel_readouts(model, w, 200.0, m, s, batch=4)
    ref = FF.extract_features(model, w, 200.0, m, s, batch=4)
    assert out["channels"] == 8 and out["pool"].shape == (6, 32) and np.allclose(out["pool"], ref, atol=1e-5)  # il readout di sempre
    for k in CR.CHANNEL_LAYERS:
        assert out[k].shape == (6, 8 * 16) and out[k].dtype == np.float16 and np.isfinite(out[k]).all()
        assert not np.allclose(out[k][0], out[k][1])  # finestre diverse, feature diverse
    again = CR.extract_channel_readouts(model, w, 200.0, m, s, batch=2)
    for k in CR.LAYERS:  # non dipende dal batch (float16: un ulp di differenza ammesso)
        np.testing.assert_allclose(out[k].astype(np.float32), again[k].astype(np.float32), rtol=3e-3, atol=3e-3)
    # la media per canale dell'encoder locale, calcolata a mano su una finestra sola (un ritaglio: 5 s a 200 Hz > 4 s, due ritagli mediati)
    assert FF._crops(1000, 200.0, 4.0) == [(0, 800), (200, 1000)]
    scales = FF.subject_scales(w, s, 200.0, (20.0, 450.0))
    parts = []
    for a, b in ((0, 800), (200, 1000)):
        x = FF._filtered(w[0, a:b], 200.0, (20.0, 450.0)) / np.float32(scales["a"])
        inp = _inputs(model, x.T[None], 200.0, m, 1)
        with torch.no_grad():
            enc = model.encode(inp, None)
        parts.append(enc.local.mean(dim=1).reshape(-1).numpy())
    np.testing.assert_allclose(out["local"][0].astype(np.float32), np.mean(parts, axis=0), rtol=3e-3, atol=3e-3)


def test_channels_discarded_by_the_qc_are_zero_in_every_channel_layer():
    model = _model()
    w, s = _windows()
    full = CR.extract_channel_readouts(model, w, 200.0, FF.myo8_montage("epn612", 200.0), s)
    m = FF.myo8_montage("epn612", 200.0)
    m["groups"][0]["channels"][5]["qc_valid"] = False
    masked = CR.extract_channel_readouts(model, w, 200.0, m, s)
    cols = np.arange(5 * 16, 6 * 16)
    for k in CR.CHANNEL_LAYERS:
        assert np.all(masked[k][:, cols] == 0) and np.any(full[k][:, cols] != 0)
    with pytest.raises(ValueError, match="canali diversi"):
        CR.channel_means(torch.zeros(5, 3, 2), torch.ones(5, 3, dtype=torch.bool), [2, 3])


def test_linear_probe_without_a_new_scale_equals_the_probe_on_prescaled_inputs_and_the_default_is_unchanged():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 6)) * 2
    y = np.tile(np.arange(3), 60)
    x = centers[y] + rng.normal(size=(180, 6)) * np.array([1, 5, 0.2, 1, 1, 3])
    s = np.repeat([f"u{i}" for i in range(9)], 20)
    tr, va, te = np.isin(s, [f"u{i}" for i in range(6)]), np.isin(s, ["u6"]), np.isin(s, ["u7", "u8"])
    a = linear_probe(x[tr], y[tr], x[va], y[va], x[te], y[te], s[te], n_boot=50)  # il percorso di default (pilot, diagnostica, calibrazione)
    sc = StandardScaler().fit(x[tr])
    b = linear_probe(sc.transform(x[tr]), y[tr], sc.transform(x[va]), y[va], sc.transform(x[te]), y[te], s[te], n_boot=50, scale=False)
    assert np.array_equal(a.test_pred, b.test_pred) and a.c == b.c and a.test_bacc == b.test_bacc
    import inspect

    assert inspect.signature(linear_probe).parameters["scale"].default is True  # default invariato: chi non lo passa ha la sonda di prima
    scaler, pca = CR.fit_reduction(x[tr], 6)
    assert pca is None  # k = larghezza: nessuna riduzione
    scaler, pca = CR.fit_reduction(x[tr], 2)
    z = CR.apply_reduction(scaler, pca, x[te])
    assert pca is not None and z.shape == (len(x[te]), 2)
    np.testing.assert_allclose(scaler.mean_, x[tr].mean(0))
    np.testing.assert_allclose(CR.apply_reduction(scaler, pca, x[tr]).mean(0), 0, atol=1e-9)
    # k oltre il rango del train: tetto a righe - 1 (UCI-EMG: ~220-240 finestre di train per fold contro k = 768), dichiarato con `k_capped`
    assert CR.effective_k(768, 221, 3072) == 220 and CR.effective_k(768, 50000, 3072) == 768 and CR.effective_k(32, 18, 128) == 17
    assert CR.effective_k(768, 100, 768) is None and CR.effective_k(0, 10, 100) is None and CR.effective_k(None, 10, 100) is None
    _, pca4 = CR.fit_reduction(x[tr][:4], 5)
    assert pca4.n_components_ == 3
    with pytest.raises(ValueError, match="nessuna riduzione"):
        CR.effective_k(5, 1, 6)


def _extract_p1(tmp_path, monkeypatch, seeds=(0,)):
    """Sessioni P1 finte, una rete casuale minuscola per seme, estrazione del readout per canale su CPU."""
    root, splits, subjects = _p1(tmp_path)
    monkeypatch.setattr(SC.P1, "P1_DATASETS", ("ninapro_db2",))
    ckpts, feats = [], []
    for seed in seeds:
        ckpt, f = tmp_path / f"tiny{seed}.pt", tmp_path / f"cr{seed}"
        torch.save(RC.random_state(TINY, seed), ckpt)
        assert SC.main(["--stage", "extract", "--checkpoint", str(ckpt), "--root", str(root), "--splits", str(splits), "--features", str(f),
                        "--device", "cpu", "--batch", "8"]) == 0
        ckpts.append(ckpt)
        feats.append(f)
    return root, splits, subjects, ckpts, feats


def _probe(feats, splits, layer, out, extra=()):
    assert SC.main(["--stage", "probe", "--features", str(feats), "--splits", str(splits), "--layer", layer, *extra, "--out", str(out)]) == 0
    return json.loads(Path(out).read_text())


def test_p1_extraction_matches_the_calibration_cache_and_the_pool_probe_equals_calibration_probe(tmp_path, monkeypatch):
    root, splits, subjects, (ckpt,), (feats,) = _extract_p1(tmp_path, monkeypatch)
    sp = json.loads(splits.read_text())
    cal = tmp_path / "cal"
    CP.extract(argparse.Namespace(checkpoint=str(ckpt), device="cpu", batch=8, root=[root], datasets=["ninapro_db2"], features=cal, splits=splits),
               sp, lambda m: None)
    with np.load(feats / "ninapro_db2.npz") as z, np.load(cal / "ninapro_db2.npz") as zc:
        assert str(z["input_sha256"]) == str(zc["input_sha256"])  # stesse finestre, stesso ordine: righe accoppiabili con le cache dei bracci
        assert str(z["feature_set"]) == "channel_readout" and int(z["channels"]) == 12 and str(z["task"]) == "p1"
        np.testing.assert_allclose(z["x_pool"], zc["x"], atol=1e-5)  # il readout medio e' quello delle sonde
        assert z["x_local"].shape == (len(zc["y"]), 12 * 16) and np.array_equal(z["y"], zc["y"]) and np.array_equal(z["s"], zc["s"])
        assert z["subjects"].tolist() == CP.pretraining_subjects(sp, "ninapro_db2") == subjects[:8]
        # l'ordine delle colonne, registrato: DB2 = anello di 8 (gruppo 0) + 4 mirati (gruppo 1)
        assert json.loads(str(z["montage_signature"])) == [[0, i] for i in range(8)] + [[1, i] for i in range(8, 12)]
        from wearusfm.harness.feature_sets import check_cache

        with pytest.raises(ValueError, match="channel_readout"):  # le sonde di sempre rifiutano questa cache
            check_cache(z, "model", z["x_pool"].shape[1])
    out = tmp_path / "pool.json"
    assert SC.main(["--stage", "probe", "--features", str(feats), "--splits", str(splits), "--layer", "pool", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    cp = CP.probe(argparse.Namespace(task="p1", features=cal, datasets=["ninapro_db2"], seed=0, splits=splits, out=tmp_path / "cp.json"), sp,
                  lambda m: None)
    d, dc = rep["datasets"]["ninapro_db2"], cp["datasets"]["ninapro_db2"]
    assert rep["S"] == pytest.approx(cp["S"], abs=1e-9) and rep["S_se"] == pytest.approx(cp["S_se"], abs=1e-9) and rep["weights"] == cp["weights"]
    assert d["oof_rows_sha256"] == dc["oof_rows_sha256"] and [f["c"] for f in d["folds"]] == [f["c"] for f in dc["folds"]]
    assert d["readout"] == {"layer": "pool", "width": 32, "channels": 12, "k": 32, "reduced": False, "k_capped_folds": 0,
                            "k_by_fold": [32] * 4}  # k = 2d = larghezza: nessuna PCA
    assert all(f["explained_variance"] == 1.0 for f in d["folds"]) and d["features"]["finite"]
    assert rep["protocol"] == "readout_per_canale_v1" and rep["calibration_protocol"] == CP.PROTOCOL and Path(rep["oof"]).exists()
    with np.load(rep["oof"]) as z, np.load(cp["oof"]) as zc:
        assert np.array_equal(z["ninapro_db2_pred"], zc["ninapro_db2_pred"]) and np.array_equal(z["ninapro_db2_row"], zc["ninapro_db2_row"])


def test_channel_layer_is_reduced_on_the_train_fold_and_compare_applies_the_arms_rule(tmp_path, monkeypatch):
    root, splits, subjects, ckpts, feats = _extract_p1(tmp_path, monkeypatch, seeds=(0, 1))
    rep = {}
    for i, f in enumerate(feats):
        rep["pool", i] = _probe(f, splits, "pool", tmp_path / f"pool{i}.json")
        rep["local", i] = _probe(f, splits, "local", tmp_path / f"local{i}.json")  # k = 2d = 32 su 192 colonne
        rep["dec", i] = _probe(f, splits, "dec", tmp_path / f"dec{i}.json", ["--k", "0"])  # larghezza intera
    pool, local, dec = rep["pool", 0], rep["local", 0], rep["dec", 0]
    dl = local["datasets"]["ninapro_db2"]
    assert dl["readout"] == {"layer": "local", "width": 192, "channels": 12, "k": 32, "reduced": True, "k_capped_folds": 0, "k_by_fold": [32] * 4}
    assert all(f["k"] == 32 and not f["k_capped"] and 0 < f["explained_variance"] <= 1 for f in dl["folds"]) and 0 <= local["S"] <= 1
    assert dec["datasets"]["ninapro_db2"]["readout"] == {"layer": "dec", "width": 192, "channels": 12, "k": 192, "reduced": False,
                                                        "k_capped_folds": 0, "k_by_fold": [192] * 4}
    assert 0 <= dec["S"] <= 1 and np.isfinite(local["S_se"]) and np.isfinite(dec["S_se"])  # rete casuale minuscola: nessun livello atteso
    assert dl["input_sha256"] == pool["datasets"]["ninapro_db2"]["input_sha256"] and dl["oof_rows_sha256"] == pool["datasets"]["ninapro_db2"]["oof_rows_sha256"]
    assert rep["pool", 1]["checkpoint_sha256"] != pool["checkpoint_sha256"] == local["checkpoint_sha256"]  # due semi, stesso modello per seme
    # il confronto dei bracci: due semi, soggetti ricampionati insieme, regola firmata
    paths = {k: tmp_path / f"{k[0]}{k[1]}.json" for k in rep}
    out = tmp_path / "confronto.json"
    assert SC.main(["--stage", "compare", "--reference", str(paths["pool", 0]), str(paths["pool", 1]),
                    "--candidate", str(paths["local", 0]), str(paths["local", 1]), "--out", str(out)]) == 0
    c = json.loads(out.read_text())
    row = c["comparison"]
    assert row["A"] == pytest.approx((pool["S"] + rep["pool", 1]["S"]) / 2, abs=1e-9) and row["B"] == pytest.approx((local["S"] + rep["local", 1]["S"]) / 2, abs=1e-9)
    assert row["delta"] == pytest.approx(row["B"] - row["A"], abs=1e-9) and len(row["delta_by_model_seed"]) == 2 and row["ci95"][0] <= row["ci95"][1]
    assert c["bootstrap"] == {"n_boot": 1000, "seed": 20261009, "method": c["bootstrap"]["method"]} and c["task"] == "p1"
    assert c["reading"]["verdict"] in ("migliora", "peggiora", "regola non superata") and c["reading"]["min_delta"] == 0.01
    assert c["reading"]["beats"] == (row["delta"] >= 0.01 and row["ci95"][0] > 0 and all(d > 0 for d in row["delta_by_model_seed"]))
    # la regola su righe scritte a mano
    assert SC.reading({"delta": 0.02, "ci95": [0.005, 0.03], "delta_by_model_seed": [0.025, 0.015]})["verdict"] == "migliora"
    assert SC.reading({"delta": 0.02, "ci95": [0.005, 0.03], "delta_by_model_seed": [0.045, -0.005]})["verdict"] == "regola non superata"
    assert SC.reading({"delta": 0.02, "ci95": [-0.001, 0.04], "delta_by_model_seed": [0.025, 0.015]})["verdict"] == "regola non superata"
    assert SC.reading({"delta": -0.02, "ci95": [-0.03, -0.005], "delta_by_model_seed": [-0.025, -0.015]})["verdict"] == "peggiora"
    # identita' dei modelli: stesso candidato due volte, semi scambiati, riferimenti uguali con un modello, due file dello stesso checkpoint
    copy0 = tmp_path / "local0_copia.json"
    copy0.write_text(json.dumps(local))
    for cand, msg in (((paths["local", 0], paths["local", 0]), "candidati diversi"), ((paths["local", 1], paths["local", 0]), "checkpoint diversi"),
                      ((paths["local", 0], copy0), "stesso checkpoint")):
        with pytest.raises(ValueError, match=msg):
            SC.main(["--stage", "compare", "--reference", str(paths["pool", 0]), str(paths["pool", 1]), "--candidate", str(cand[0]), str(cand[1]),
                     "--out", str(tmp_path / "no.json")])
    with pytest.raises(ValueError, match="checkpoint diversi"):
        SC.main(["--stage", "compare", "--reference", str(paths["pool", 0]), str(paths["pool", 0]), "--candidate", str(paths["local", 0]),
                 str(paths["local", 1]), "--out", str(tmp_path / "no.json")])
    # i due semi di un lato devono essere lo stesso tipo di rapporto (la normalizzazione per `s_comparison` non deve nascondere il contrario):
    # strati mescolati, k mescolati, riferimenti uno con modello e uno senza
    _probe(feats[1], splits, "local", tmp_path / "local1_full.json", ["--k", "0"])
    for cand, msg in (((paths["local", 0], paths["dec", 1]), "tipo diverso"),  # insieme di feature diverso (channel_readout/local vs /dec)
                      ((paths["local", 0], tmp_path / "local1_full.json"), "strato, k o larghezza")):  # stesso strato, k diverso
        with pytest.raises(ValueError, match=msg):
            SC.main(["--stage", "compare", "--reference", str(paths["pool", 0]), str(paths["pool", 1]), "--candidate", str(cand[0]), str(cand[1]),
                     "--out", str(tmp_path / "no.json")])
    nomodel = {k: v for k, v in rep["pool", 1].items() if k not in ("checkpoint", "checkpoint_sha256")}
    (tmp_path / "pool1_senza_modello.json").write_text(json.dumps(nomodel))
    with pytest.raises(ValueError, match="uno con modello e uno senza"):
        SC.main(["--stage", "compare", "--reference", str(paths["pool", 0]), str(tmp_path / "pool1_senza_modello.json"), "--candidate",
                 str(paths["local", 0]), str(paths["local", 1]), "--out", str(tmp_path / "no.json")])
    # controllo d'uguaglianza (protocollo, punto 4): lo stesso rapporto copiato = uguale; un altro strato dello stesso checkpoint = diverso
    assert SC.main(["--stage", "equality", "--reference", str(paths["pool", 0]), "--candidate", str(copy0.with_name("pool0_copia.json")) if False else str(paths["pool", 0]),
                    "--out", str(tmp_path / "eq.json")]) == 0
    eq = json.loads((tmp_path / "eq.json").read_text())
    assert eq["equal"] and eq["delta_S"] == 0 and all(v["predictions_differing"] == 0 for v in eq["datasets"].values()) and eq["tolerance"] == 1e-3
    assert SC.main(["--stage", "equality", "--reference", str(paths["pool", 0]), "--candidate", str(paths["local", 0]), "--out", str(tmp_path / "eq2.json")]) == 1
    assert not json.loads((tmp_path / "eq2.json").read_text())["equal"]
    with pytest.raises(ValueError, match="stesso checkpoint"):
        SC.main(["--stage", "equality", "--reference", str(paths["pool", 1]), "--candidate", str(paths["pool", 0]), "--out", str(tmp_path / "no.json")])
    # rapporti non accoppiati: altre finestre
    other = {**local, "datasets": {"ninapro_db2": {**dl, "input_sha256": "0" * 64}}}
    (tmp_path / "altro.json").write_text(json.dumps(other))
    with pytest.raises(ValueError, match="accoppiato"):
        SC.main(["--stage", "compare", "--reference", str(paths["pool", 0]), str(paths["pool", 1]), "--candidate", str(tmp_path / "altro.json"),
                 str(paths["local", 1]), "--out", str(tmp_path / "no.json")])
    # predizioni fuori fold copiate altrove (i rapporti del cluster letti sul Mac): rimappatura del prefisso, SHA-256 sempre verificato
    moved = tmp_path / "copia"
    moved.mkdir()
    for k in (("pool", 0), ("pool", 1)):
        r = rep[k]
        (moved / Path(r["oof"]).name).write_bytes(Path(r["oof"]).read_bytes())
        (tmp_path / f"remoto_{k[1]}.json").write_text(json.dumps({**r, "oof": "/leonardo_finto/risultati/" + Path(r["oof"]).name}))
    with pytest.raises(FileNotFoundError):
        SC.main(["--stage", "compare", "--reference", str(tmp_path / "remoto_0.json"), str(tmp_path / "remoto_1.json"),
                 "--candidate", str(paths["local", 0]), str(paths["local", 1]), "--out", str(tmp_path / "no.json")])
    assert SC.main(["--stage", "compare", "--reference", str(tmp_path / "remoto_0.json"), str(tmp_path / "remoto_1.json"),
                    "--candidate", str(paths["local", 0]), str(paths["local", 1]), "--oof-map", f"/leonardo_finto/risultati={moved}",
                    "--out", str(tmp_path / "rimappato.json")]) == 0
    assert json.loads((tmp_path / "rimappato.json").read_text())["comparison"] == row
    (moved / Path(pool["oof"]).name).write_bytes(b"x")  # un file diverso da quello del rapporto: rifiutato
    with pytest.raises(ValueError, match="SHA-256"):
        SC.main(["--stage", "compare", "--reference", str(tmp_path / "remoto_0.json"), str(tmp_path / "remoto_1.json"),
                 "--candidate", str(paths["local", 0]), str(paths["local", 1]), "--oof-map", f"/leonardo_finto/risultati={moved}",
                 "--out", str(tmp_path / "no.json")])
    # un rapporto con i parametri della sonda sotto "probe" (baseline classica) si accoppia allo stesso modo; una chiave assente e' un errore
    td = {**pool, "probe": {k: pool[k] for k in SC.PROBE_KEYS}}
    for k in SC.PROBE_KEYS:
        td.pop(k)
    SC.check_readout_pairing(td, local)
    with pytest.raises(ValueError, match="sonda non accoppiata"):
        SC.check_readout_pairing({**td, "probe": {**td["probe"], "fold_seed": 1}}, local)
    with pytest.raises(ValueError, match="mancanti"):
        SC.check_readout_pairing({**td, "probe": {k: v for k, v in td["probe"].items() if k != "window_s"}}, local)


def test_compare_between_two_models_with_the_same_readout_needs_the_models_flag(tmp_path, monkeypatch):
    """Opzione ancore-only del protocollo: riferimento ancore-only s0/s1, candidato A s0/s1, stesso strato. Quattro reti casuali minuscole:
    (0, 1) fanno da candidato, (2, 3) da riferimento."""
    root, splits, subjects, ckpts, feats = _extract_p1(tmp_path, monkeypatch, seeds=(0, 1, 2, 3))
    rep = {i: _probe(f, splits, "local", tmp_path / f"local{i}.json") for i, f in enumerate(feats)}
    assert len({r["checkpoint_sha256"] for r in rep.values()}) == 4
    p = {i: str(tmp_path / f"local{i}.json") for i in rep}
    base = ["--stage", "compare", "--reference", p[2], p[3], "--candidate", p[0], p[1]]
    with pytest.raises(ValueError, match="checkpoint diversi"):  # il confronto 1-2 vuole lo stesso modello per seme
        SC.main(base + ["--out", str(tmp_path / "no.json")])
    out = tmp_path / "modelli.json"
    assert SC.main(base + ["--models", "--out", str(out)]) == 0
    c = json.loads(out.read_text())
    row = c["comparison"]
    assert c["mode"].startswith("due modelli") and c["reading"]["min_delta"] == 0.01 and c["bootstrap"]["seed"] == 20261009
    assert row["A"] == pytest.approx((rep[2]["S"] + rep[3]["S"]) / 2, abs=1e-9) and row["B"] == pytest.approx((rep[0]["S"] + rep[1]["S"]) / 2, abs=1e-9)
    assert row["delta"] == pytest.approx(row["B"] - row["A"], abs=1e-9) and len(row["delta_by_model_seed"]) == 2
    # rifiuti: riferimenti dello stesso checkpoint, candidato uguale al riferimento del seme, strati diversi fra i lati, riferimento senza modello
    _probe(feats[2], splits, "dec", tmp_path / "dec2.json", ["--k", "0"])
    _probe(feats[3], splits, "dec", tmp_path / "dec3.json", ["--k", "0"])
    nomodel = {k: v for k, v in rep[2].items() if k not in ("checkpoint", "checkpoint_sha256")}
    (tmp_path / "local2_senza_modello.json").write_text(json.dumps(nomodel))
    (tmp_path / "local3_senza_modello.json").write_text(json.dumps({k: v for k, v in rep[3].items() if k not in ("checkpoint", "checkpoint_sha256")}))
    for ref, cand, msg in (((p[2], p[2]), (p[0], p[1]), "due riferimenti sono lo stesso checkpoint"),
                           ((p[0], p[3]), (p[0], p[1]), "stesso seme sono lo stesso checkpoint"),
                           ((str(tmp_path / "dec2.json"), str(tmp_path / "dec3.json")), (p[0], p[1]), "tipo diverso"),
                           ((str(tmp_path / "local2_senza_modello.json"), str(tmp_path / "local3_senza_modello.json")), (p[0], p[1]), "senza modello")):
        with pytest.raises(ValueError, match=msg):
            SC.main(["--stage", "compare", "--models", "--reference", *ref, "--candidate", *cand, "--out", str(tmp_path / "no.json")])
    with pytest.raises(SystemExit):  # solo con compare
        SC.main(["--stage", "equality", "--models", "--reference", p[0], "--candidate", p[0], "--out", str(tmp_path / "no.json")])


def test_scaler_and_pca_see_only_the_train_rows_of_each_fold(monkeypatch):
    rng = np.random.default_rng(0)
    subjects = [f"S{i:02d}" for i in range(8)]
    s = np.repeat(subjects, 30)
    y = np.tile(np.arange(3), len(s) // 3)
    x = (rng.normal(size=(3, 12))[y] + rng.normal(size=(len(s), 12))).astype(np.float32)
    roles = CP.fold_roles(subjects)
    seen, real = [], CR.fit_reduction

    def spy(x_train, k, seed=0):
        seen.append((len(x_train), k, seed))
        return real(x_train, k, seed)

    monkeypatch.setattr(CR, "fit_reduction", spy)
    res = SC.oof_predictions(x, y, s, roles, 7, 4)
    assert seen == [(int(np.isin(s, r["train"]).sum()), 4, 7) for r in roles]  # solo le righe di train del fold, k e seme della sonda
    assert len(res["row"]) == len(s) and all(f["k"] == 4 for f in res["folds"])


def test_sessions_with_a_different_channel_order_are_refused(tmp_path, monkeypatch):
    root, splits, subjects = _p1(tmp_path)
    monkeypatch.setattr(SC.P1, "P1_DATASETS", ("ninapro_db2",))
    meta_path = root / "ninapro_db2" / "s02" / "session1" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    chans = meta["montage"]["groups"][0]["channels"]
    chans[0]["sensor_coords"]["channel_index"], chans[1]["sensor_coords"]["channel_index"] = 1, 0  # due canali scambiati in una sessione
    meta_path.write_text(json.dumps(meta))
    ckpt = tmp_path / "tiny.pt"
    torch.save(RC.random_state(TINY, 0), ckpt)
    with pytest.raises(ValueError, match="ordine dei canali"):
        SC.main(["--stage", "extract", "--checkpoint", str(ckpt), "--root", str(root), "--splits", str(splits), "--features", str(tmp_path / "f"),
                 "--device", "cpu"])
    assert SC.montage_signature(FF.myo8_montage("epn612", 200.0)) == [[0, i] for i in range(8)]


def test_p2_extraction_and_probe_use_the_internal_subjects_and_pair_with_calibration_and_classical_reports(tmp_path):
    from test_classical_baseline import CB
    from wearusfm.harness.feature_sets import cache_record

    epn, uci = tmp_path / "epn" / "trainingJSON", tmp_path / "uci"
    _epn(epn, 12)
    _uci(uci, 10)
    p2 = ["--task", "p2", "--epn-root", str(epn), "--uci-root", str(uci)]
    feats, reports = [], []
    for seed in (0, 1):
        ckpt, f = tmp_path / f"tiny{seed}.pt", tmp_path / f"cr2_{seed}"
        torch.save(RC.random_state(TINY, seed), ckpt)
        assert SC.main(["--stage", "extract", *p2, "--checkpoint", str(ckpt), "--features", str(f), "--device", "cpu", "--batch", "8"]) == 0
        feats.append(f)
    with np.load(feats[0] / "epn612.npz") as z:
        users = z["all_subjects"].tolist()
        assert z["subjects"].tolist() == CP.p2_internal_subjects(users) and sorted(set(z["s"].tolist())) == CP.p2_internal_subjects(users)
        assert int(z["channels"]) == 8 and z["x_dec"].shape[1] == 8 * 16 and str(z["task"]) == "p2"
        x_pool, y, s = {"epn612": z["x_pool"]}, {"epn612": z["y"]}, {"epn612": z["s"]}
        sha, all_subjects = {"epn612": str(z["input_sha256"])}, {"epn612": users}
    with np.load(feats[0] / "uci_emg.npz") as z:
        x_pool["uci_emg"], y["uci_emg"], s["uci_emg"], sha["uci_emg"], all_subjects["uci_emg"] = z["x_pool"], z["y"], z["s"], str(z["input_sha256"]), z["all_subjects"].tolist()
    # stessa impronta degli ingressi e stesso readout medio dell'estrazione della calibrazione (`CP.extract_p2`)
    cal = tmp_path / "cal2"
    CP.extract_p2(argparse.Namespace(checkpoint=str(tmp_path / "tiny0.pt"), device="cpu", batch=8, datasets=["epn612", "uci_emg"], features=cal,
                                     epn_root=[epn], uci_root=uci), lambda m: None)
    for ds in ("epn612", "uci_emg"):
        with np.load(cal / f"{ds}.npz") as zc:
            assert str(zc["input_sha256"]) == sha[ds] and np.allclose(zc["x"], x_pool[ds], atol=1e-5)
    out = tmp_path / "p2_local_cap.json"  # k = 2d = 32 > 18 finestre di train nei dati finti: tetto a righe - 1 per fold, dichiarato
    assert SC.main(["--stage", "probe", "--task", "p2", "--features", str(feats[0]), "--layer", "local", "--out", str(out)]) == 0
    cap = json.loads(out.read_text())["datasets"]["epn612"]
    assert cap["readout"]["k_capped_folds"] == 4 and cap["readout"]["k_by_fold"] == [f["n_train"] - 1 for f in cap["folds"]]
    assert all(f["k_capped"] and f["k"] == f["n_train"] - 1 < 32 for f in cap["folds"])
    for i, f in enumerate(feats):
        assert SC.main(["--stage", "probe", "--task", "p2", "--features", str(f), "--layer", "local", "--k", "8", "--out", str(tmp_path / f"p2_local{i}.json")]) == 0
        reports.append(json.loads((tmp_path / f"p2_local{i}.json").read_text()))
    rep = reports[0]
    assert rep["task"] == "p2" and rep["weights"] == {"epn612": 0.5, "uci_emg": 0.5} and rep["splits_sha256"] == CP.P2_SPLITS_SHA256
    assert rep["datasets"]["epn612"]["readout"]["reduced"] and rep["datasets"]["epn612"]["readout"]["k"] == 8 and rep["readout"]["k_requested"] == 8
    assert rep["datasets"]["epn612"]["readout"]["k_capped_folds"] == 0
    # si accoppia con un rapporto di S della P2 di `calibration_probe` (il riferimento storico) ...
    ref = tmp_path / "ref_p2"
    ref.mkdir()
    for ds in ("epn612", "uci_emg"):
        np.savez(ref / f"{ds}.npz", x=x_pool[ds], y=y[ds], s=s[ds], subjects=np.asarray(CP.p2_internal_subjects(all_subjects[ds])),
                 all_subjects=np.asarray(all_subjects[ds]), input_sha256=np.asarray(sha[ds]), checkpoint=np.asarray(str(tmp_path / "tiny0.pt")),
                 checkpoint_sha256=np.asarray(rep["checkpoint_sha256"]), encoder_dim=np.asarray(16), protocol=np.asarray(CP.PROTOCOL),
                 task=np.asarray("p2"), splits_sha256=np.asarray(CP.P2_SPLITS_SHA256),
                 extraction_recipe=np.asarray(json.dumps(CP.EXTRACTION_RECIPE_P2, sort_keys=True)), **cache_record("model", 32))
    cp = CP.probe(argparse.Namespace(task="p2", features=ref, datasets=["epn612", "uci_emg"], seed=0, splits=None, out=tmp_path / "cp2.json"), None,
                  lambda m: None)
    (tmp_path / "cp2.json").write_text(json.dumps(cp))
    SC.check_readout_pairing(cp, rep)  # la P2 non ha `window_s` (None nei due rapporti): accoppiata lo stesso
    # ... e con un rapporto vero della baseline classica (stesse finestre, parametri della sonda sotto "probe", nessun modello)
    td = tmp_path / "td_p2.json"
    assert CB.main(["--task", "p2", "--epn-root", str(epn), "--uci-root", str(uci), "--out", str(td)]) == 0
    td_rep = json.loads(td.read_text())
    SC.check_readout_pairing(td_rep, rep)
    out = tmp_path / "vs_td.json"
    assert SC.main(["--stage", "compare", "--reference", str(td), str(td), "--candidate", str(tmp_path / "p2_local0.json"),
                    str(tmp_path / "p2_local1.json"), "--out", str(out)]) == 0
    c = json.loads(out.read_text())
    assert c["task"] == "p2" and c["comparison"]["A"] == pytest.approx(td_rep["S"], abs=1e-9) and c["reading"]["verdict"]
    assert c["comparison"]["B"] == pytest.approx((reports[0]["S"] + reports[1]["S"]) / 2, abs=1e-9)
    fake = {"datasets": {"epn612": {"pretraining": CP.p2_internal_subjects(users), "test": ["nessuno"]}}}
    feats = feats[0]
    (tmp_path / "fake_splits.json").write_text(json.dumps(fake))
    with pytest.raises(ValueError, match="altro compito"):  # cache della P2 letta come P1
        SC.probe(argparse.Namespace(task="p1", features=feats, datasets=["epn612"], seed=0, layer="local", k=None,
                                    splits=tmp_path / "fake_splits.json", out=tmp_path / "x.json"), fake, lambda m: None)


@pytest.mark.parametrize("name,args,expected", [
    ("channel_readout_extract", ["/C/ckpt.pt", "{root}/out/f1"], "VENV_PYTHON scripts/channel_readout.py --stage extract --checkpoint /C/ckpt.pt"),
    ("channel_readout_probe", ["{root}/out/f1", "{root}/out/r.json", "--layer", "local"], "VENV_PYTHON scripts/channel_readout.py --stage probe"),
])
def test_launchers_use_the_code_dir_rule(work, name, args, expected):  # noqa: F811
    root, canon, first, second = work
    env = _launcher_env(root)
    launcher = SLURM / f"{name}.sbatch"
    args = [a.format(root=root) for a in args]
    r = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True, text=True)
    assert expected in r.stdout, r.stdout + r.stderr  # canonico pulito: accettato
    if name.endswith("probe"):
        assert "--layer local" in r.stdout  # gli argomenti dopo cartella e rapporto passano allo script
    bad = list(args)
    bad[1] = "rel/x"
    r = subprocess.run(["bash", str(launcher), *bad], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "relativ" in r.stdout and "VENV_PYTHON" not in r.stdout
    bad[1] = f"{root}/wearusfm/x"  # dentro il codice
    r = subprocess.run(["bash", str(launcher), *bad], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "dentro il codice" in r.stdout
    r = subprocess.run(["bash", str(launcher), *args], env={**env, "CODE_DIR": str(root / "wearusfm_campaigns" / "x")}, capture_output=True, text=True)
    assert r.returncode == 2 and "VENV_PYTHON" not in r.stdout  # checkout di campagna senza SHA atteso: rifiutato
