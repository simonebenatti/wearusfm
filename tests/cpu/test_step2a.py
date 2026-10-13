"""Test 2A dello step 2 (`scripts/step2a.py`, protocollo `docs/step2a_stats_probe.md`): statistiche sulla CPU allineate riga per riga alle cache
degli embedding gia' estratte, varianti concatenate, sonde P2 ufficiali da capo a fondo su dati sintetici, rifiuto di cache cambiate o non
allineate, confronto accoppiato e criteri congelati."""

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch  # prima di sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import step2a as S2  # noqa: E402

from test_downstream_ab import _reports  # noqa: E402
from test_probe_p2 import _epn, _uci  # noqa: E402


def _setup(tmp_path, monkeypatch, epn_users=10):
    from wearusfm.model.fm import FMConfig, WearUsFM

    _epn(tmp_path / "epn" / "trainingJSON", epn_users)
    _uci(tmp_path / "uci", 10)
    cfg = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4)
    for seed, arm in enumerate(S2.ARMS):
        torch.manual_seed(seed)
        ckpt = tmp_path / "runs" / arm / "checkpoint.pt"
        ckpt.parent.mkdir(parents=True)
        m = WearUsFM(cfg)
        torch.save({"config": {"model": asdict(cfg)}, "teacher": m.state_dict(), "student": m.state_dict()}, ckpt)
        monkeypatch.setitem(S2.SD.CHECKPOINTS, arm, S2.sha256_file(ckpt))
    monkeypatch.setattr(S2.SD, "check_source", lambda _: None)
    monkeypatch.setattr(S2, "DOWN_DIM", 32)
    return SimpleNamespace(expected_commit="test", task="p2", runs_root=tmp_path / "runs", downstream_root=tmp_path / "down",
                           eval_root=tmp_path / "eval", root=[], epn_root=[tmp_path / "epn" / "trainingJSON"], uci_root=tmp_path / "uci",
                           variant=None, arm=None, compare_after=False)


def _downstream(args, tmp_path, epn_root=None):
    """Le cache degli embedding come le scrive la tranche downstream (estrazione con audit + provenance.json)."""
    for arm in S2.ARMS:
        folder = args.downstream_root / "p2" / arm
        argv = ["--checkpoint", str(args.runs_root / arm / "checkpoint.pt"), "--features", str(folder), "--stage", "extract",
                "--epn-root", str(epn_root or args.epn_root[0]), "--uci-root", str(args.uci_root), "--audit-inputs", "--device", "cpu",
                "--batch", "8"]
        assert S2.probe_p2.main(argv) == 0
        identity = {"protocol": S2.SD.PROTOCOL, "source_commit": "downstream", "training_commit": S2.SD.TRAIN_COMMIT,
                    "checkpoint_sha256": S2.SD.CHECKPOINTS[arm], "arm": arm, "task": "p2", "which": "teacher", "probe_seed": 0,
                    "split_sha256": S2.SD.SPLIT_SHA}
        (folder / "provenance.json").write_text(json.dumps({"identity": identity, "cache": S2.inventory(folder, "p2", 32)}))


def test_p2_stats_variants_and_probes_end_to_end(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    _downstream(args, tmp_path)
    S2.stats(args)
    for ds in S2.SD.DATASETS["p2"]:
        with np.load(args.downstream_root / "p2" / "B_seed1" / f"{ds}.npz") as m, np.load(args.eval_root / "p2" / "S" / f"{ds}.npz") as s, \
                np.load(args.eval_root / "p2" / "B_seed1" / "MS" / f"{ds}.npz") as ms, np.load(args.eval_root / "p2" / "K" / f"{ds}.npz") as k:
            assert s["x"].shape[1] == 64 and str(s["input_sha256"]) == str(m["input_sha256"])
            np.testing.assert_array_equal(ms["x"], np.concatenate([m["x"], s["x"]], 1))
            np.testing.assert_array_equal(k["x"], s["x"][:, 32:])
            np.testing.assert_array_equal(ms["y"], m["y"])
    patterns = json.loads((args.eval_root / "p2" / "S" / "provenance.json").read_text())["mask_patterns"]
    assert patterns == {"epn612": 1, "uci_emg": 1}  # dentro un dataset stessa fs e stessa banda: indicatori costanti
    with pytest.raises(ValueError, match="NEW"):
        S2.stats(args)  # niente sovrascritture
    for variant in S2.VARIANTS:
        for arm in (S2.ARMS if variant in S2.PER_ARM else (None,)):
            S2.probe(SimpleNamespace(**{**vars(args), "variant": variant, "arm": arm}))
            report = json.loads(S2._report_path(args, "p2", variant, arm).read_text())
            assert report["feature_set"] == S2.VARIANTS[variant] and report["provenance"]["identity"]["variant"] == variant
            assert 0 <= report["p2"] <= 1
    with pytest.raises(ValueError, match="braccio"):
        S2.probe(SimpleNamespace(**{**vars(args), "variant": "S", "arm": "B_seed0"}))
    path = args.downstream_root / "p2" / "B_seed0" / "uci_emg.npz"
    path.write_bytes(path.read_bytes() + b"x")  # la cache d'origine cambia dopo: ogni variante che ne deriva si ferma
    with pytest.raises(ValueError, match="cambiat"):
        S2._variant_meta(args, "p2", "MS", "B_seed0")


def test_probe_checks_the_feature_set_of_the_cache(tmp_path, monkeypatch):
    """Review GPT del 13/10: una cache di 64 statistiche dichiarata «mask» (32) veniva accettata. Ora la cache registra insieme e larghezza
    dell'embedding e la sonda li verifica; una cache senza registrazione (tranche downstream) vale solo come embedding."""
    args = _setup(tmp_path, monkeypatch)
    feat = tmp_path / "f"
    base = ["--checkpoint", str(args.runs_root / "B_seed0" / "checkpoint.pt"), "--epn-root", str(args.epn_root[0]), "--uci-root",
            str(args.uci_root), "--device", "cpu", "--batch", "8", "--features", str(feat)]
    assert S2.probe_p2.main(base + ["--stage", "extract", "--feature-set", "stats"]) == 0
    with np.load(feat / "epn612.npz") as z:
        assert str(z["feature_set"]) == "stats" and int(z["model_dim"]) == 0 and z["x"].shape[1] == 64
    with pytest.raises(ValueError, match="estratta come 'stats'"):
        S2.probe_p2.main(base + ["--stage", "probe", "--feature-set", "mask", "--out", str(tmp_path / "bad.json")])
    assert S2.probe_p2.main(base + ["--stage", "probe", "--feature-set", "stats", "--out", str(tmp_path / "ok.json")]) == 0
    for ds in S2.SD.DATASETS["p2"]:  # come una cache scritta prima della registrazione
        with np.load(feat / f"{ds}.npz") as z:
            np.savez(feat / f"{ds}.npz", **{k: z[k] for k in ("x", "y", "s")})
    with pytest.raises(ValueError, match="estratta come 'model'"):
        S2.probe_p2.main(base + ["--stage", "probe", "--feature-set", "stats", "--out", str(tmp_path / "old.json")])
    with np.load(feat / "epn612.npz") as z:  # e una larghezza che non torna con quella registrata
        np.savez(feat / "epn612.npz", x=z["x"][:, :40], y=z["y"], s=z["s"], feature_set=np.asarray("stats"), model_dim=np.asarray(0))
    with pytest.raises(ValueError, match="40 colonne"):
        S2.probe_p2.main(base + ["--stage", "probe", "--feature-set", "stats", "--out", str(tmp_path / "w.json")])


def test_stats_refuse_caches_of_other_windows(tmp_path, monkeypatch):
    """Embedding estratti su finestre diverse (un utente EPN in meno): l'hash degli ingressi non coincide e niente viene concatenato."""
    args = _setup(tmp_path, monkeypatch)
    _epn(tmp_path / "epn_small" / "trainingJSON", 9)
    _downstream(args, tmp_path, epn_root=tmp_path / "epn_small" / "trainingJSON")
    with pytest.raises(ValueError, match="stesse finestre"):
        S2.stats(args)
    assert not (args.eval_root / "p2" / "B_seed0" / "MS").exists()


def test_downstream_identity_is_checked(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    _downstream(args, tmp_path)
    meta_path = args.downstream_root / "p2" / "B_seed1" / "provenance.json"
    meta = json.loads(meta_path.read_text())
    meta["identity"]["checkpoint_sha256"] = "altro"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="provenienza"):
        S2.stats(args)


def _result(delta, ci_low, seeds):
    row = lambda d, lo, sd: {"A": .7, "B": .7 + d, "delta": d, "ci95": [lo, d + .01], "delta_by_model_seed": sd}  # noqa: E731
    return {"tasks": {t: row(delta[t], ci_low[t], seeds[t]) for t in ("p1", "p2")}}


@pytest.mark.parametrize("case,positive", [
    (dict(delta={"p1": .02, "p2": 0.}, ci_low={"p1": .005, "p2": -.01}, seeds={"p1": [.01, .03], "p2": [0., 0.]}), True),
    (dict(delta={"p1": .02, "p2": -.015}, ci_low={"p1": .005, "p2": -.03}, seeds={"p1": [.01, .03], "p2": [-.01, -.02]}), False),  # dannoso in P2
    (dict(delta={"p1": .02, "p2": 0.}, ci_low={"p1": -.001, "p2": -.01}, seeds={"p1": [.01, .03], "p2": [0., 0.]}), False),  # CI tocca lo 0
    (dict(delta={"p1": .02, "p2": 0.}, ci_low={"p1": .005, "p2": -.01}, seeds={"p1": [-.001, .041], "p2": [0., 0.]}), False),  # un teacher giu'
    (dict(delta={"p1": .009, "p2": .009}, ci_low={"p1": .001, "p2": .001}, seeds={"p1": [.009, .009], "p2": [.009, .009]}), False),  # < 1 punto
    (dict(delta={"p1": 0., "p2": .03}, ci_low={"p1": -.01, "p2": .01}, seeds={"p1": [0., 0.], "p2": [.02, .04]}), True),  # utile in P2
])
def test_frozen_decision_rule(case, positive):
    d = S2.decide(_result(**case))
    assert d["positive"] is positive
    assert d["thresholds"] == {"useful_delta": .01, "harm_delta": -.01}


def test_compare_pairs_m_with_ms(tmp_path, monkeypatch):
    monkeypatch.setattr(S2.SD, "check_source", lambda _: None)
    inv = {"data": {"sha256": "x", "input_sha256": "i", "rows": {"": {"n": 16}}}}
    base = _reports()  # A: 0,75, B: 1,0 per task e dataset (test_downstream_ab)
    args = SimpleNamespace(expected_commit="test", eval_root=tmp_path / "eval", downstream_root=tmp_path / "down")
    metas = {}
    for task in ("p1", "p2"):
        for arm in S2.ARMS:  # i report della tranche downstream, che la sonda rifatta su M deve riprodurre
            (args.downstream_root / task).mkdir(parents=True, exist_ok=True)
            (args.downstream_root / task / f"{arm}.json").write_text(json.dumps(base[task]["A_seed0"]))
        for variant in S2.VARIANTS:
            for arm in (S2.ARMS if variant in S2.PER_ARM else (None,)):
                src = base[task]["A_seed0" if variant in ("M", "K") else "B_seed0"]
                meta = {"identity": S2._identity("test", task, variant, arm), "cache": copy.deepcopy(inv)}
                if variant == "S":
                    meta["mask_patterns"] = {"data": 1}
                metas[(task, variant, arm)] = meta
                path = S2._report_path(args, task, variant, arm)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({**copy.deepcopy(src), "feature_set": S2.VARIANTS[variant], "provenance": meta}))
    monkeypatch.setattr(S2, "_variant_meta", lambda a, t, v, arm: (None, copy.deepcopy(metas[(t, v, arm)])))
    S2.compare(args)
    out = json.loads((args.eval_root / "comparison.json").read_text())
    assert out["decision"]["positive"] and out["decision"]["p1"]["delta_stats"] == pytest.approx(.25)
    assert all(out["descriptive"]["M_reproduces_downstream_report"].values())
    assert out["descriptive"]["mask_patterns_per_dataset"]["p2"] == {"data": 1}
    assert "both_practical_guards_pass" not in out["paired_M_vs_MS"]
    (args.eval_root / "comparison.json").unlink()
    old = args.downstream_root / "p1" / "B_seed1.json"
    old.write_text(json.dumps({**base["p1"]["A_seed0"], "p1": .74}))
    S2.compare(args)
    assert not json.loads((args.eval_root / "comparison.json").read_text())["descriptive"]["M_reproduces_downstream_report"]["p1/B_seed1"]
    (args.eval_root / "comparison.json").unlink()
    path = S2._report_path(args, "p2", "MS", "B_seed1")
    r = json.loads(path.read_text())
    r["feature_set"] = "model"
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="insieme di feature"):
        S2.compare(args)
