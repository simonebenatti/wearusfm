"""Pannello della posa del collaudo lungo (B) di R3 (`RunConfig.pose_panel_*`, 28/10/2026): insieme fisso di finestre di emg2pose con la posa
(utenti x sessioni x finestre) materializzato dal loader, errore per finestra in `pose_panel.jsonl`, identita' in `pose_panel_set.json`; il
training non cambia (pesi e metriche del passo bit per bit); ripresa senza ripetere il passo 0 e con l'insieme controllato; guardie del config
e del launcher; `PretrainLoader.materialize` riproduce `batch` dalle stesse estrazioni."""

import gzip
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_pose_targets import _stats  # noqa: E402
from test_pose_training import _r3_config  # noqa: E402
from test_pretraining_loader import BM, SIGMA, _write  # noqa: E402
from test_pretraining_loader import _cfg as _loader_cfg  # noqa: E402
from test_stability_diagnostics import _records, _same_training  # noqa: E402
from wearusfm.data import pose_targets as PT  # noqa: E402
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.ingest import emg2pose  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.pairing import batch_sha256  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N = 24000  # 12 s a 2 kHz
USERS = ("u1", "u2", "u3")
SESSIONS = ("sessA", "sessB")


def _multi_tree(tmp_path: Path):
    """Tre utenti di emg2pose con due sessioni continue ciascuno, manifest dal costruttore vero, posa lineare nel tempo per ogni sessione."""
    rng = np.random.default_rng(0)
    root, pose_root = tmp_path / "processed", tmp_path / "pose"
    nb = N // PT.BLOCK
    centers = np.arange(nb)[:, None] * PT.BLOCK + (PT.BLOCK - 1) / 2.0
    for u in USERS:
        for s in SESSIONS:
            m = emg2pose.build_montage_metadata(u, s, "left")
            x = rng.normal(scale=SIGMA, size=(N, 16))
            _write(root / "emg2pose" / u / s, np.round(x * 1000).astype(np.int16),
                   {"montage": montage_to_dict(m, [True] * 16), "native_fs_hz": 2000.0, "shape": [N, 16], "int16_scale": 1000.0})
            d = pose_root / u / s
            d.mkdir(parents=True)
            np.save(d / PT.POSE_FILE, (0.001 * centers + np.arange(20)[None, :]).astype(np.float16))
            np.save(d / PT.VALID_FILE, np.ones(nb, dtype=bool))
    _stats(pose_root, mean=[0.5] * 20, std=[0.25] * 20)
    splits = {"datasets": {"emg2pose": {"pretraining": list(USERS), "test": []}}}
    rows, _ = BM.build([root], splits, {"A": 0.3, "B": 0.5, "C": 0.2}, 0.5, 1e9, 25.0)
    doc = {"params": {"quota": {"A": 0.3, "B": 0.5, "C": 0.2}}, "columns": list(BM.asdict(rows[0]).keys()),
           "rows": [list(BM.asdict(r).values()) for r in rows]}
    mpath = tmp_path / "manifest.json.gz"
    mpath.write_bytes(gzip.compress(json.dumps(doc).encode()))
    return root, mpath, pose_root


def _base(pose_root):
    """La configurazione di R3 dei test (6 passi) con l'allarme di collasso spento: sulle 6 sessioni di puro rumore di `_multi_tree` il
    rapporto di collasso scende sotto 0,05 e fermerebbe il run; qui si misura il pannello, non l'allarme (provato altrove)."""
    return replace(_r3_config(pose_root), alarm=R.AlarmRule(collapse_ratio_min=0.0, erank_fraction_min=0.0))


def test_panel_logs_per_window_errors_keeps_training_unchanged_and_resumes(tmp_path):
    root, mpath, pose_root = _multi_tree(tmp_path)
    base = _base(pose_root)  # 6 passi, pannello fisso ogni 3
    cfg = replace(base, pose_panel_every=3, pose_panel_users=2, pose_panel_recordings=2, pose_panel_windows=2)
    run = lambda c, out: R.train(c, mpath, [root], tmp_path / out, log=lambda m: None)  # noqa: E731
    run(base, "plain")
    s = run(cfg, "panel")
    assert s["stopped"] == "max_steps" and s["steps"] == 6
    assert s["pose_panel_timing"]["prepare_s"] >= 0 and len(s["pose_panel_timing"]["evaluations_s"]) == 3
    _same_training(tmp_path / "plain", tmp_path / "panel")  # il pannello non tocca pesi, RNG ne' metriche del passo
    assert not (tmp_path / "plain" / "pose_panel.jsonl").exists() and not (tmp_path / "plain" / "pose_panel_set.json").exists()
    ident = json.loads((tmp_path / "panel" / "pose_panel_set.json").read_text())
    assert ident["seed"] == [cfg.seed, R.POSE_PANEL_SEED] and len(ident["users"]) == 2 and set(ident["users"]) < set(USERS)
    assert len(ident["windows"]) == 8 and len(ident["batch_sha256"]) == 4  # 2 utenti x 2 sessioni (un batch l'una) x 2 finestre
    assert [w[0].split("/")[1] for w in ident["windows"]] == [u for u in ident["users"] for _ in range(4)]
    assert all(w[2] < w[3] <= N for w in ident["windows"])
    assert all(w[3] - w[2] == 8000 for w in ident["windows"])
    for a, b in zip(ident["windows"], ident["windows"][1:]):
        assert a[0] != b[0] or a[3] <= b[2]
    ident_sha = hashlib.sha256(json.dumps(ident, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    panel = _records(tmp_path / "panel" / "pose_panel.jsonl")
    assert [p["step"] for p in panel] == [0, 3, 6]
    for p in panel:
        assert p["panel_sha256"] == ident_sha
        assert [w["n_valid"] for w in p["per_window"]] == ident["n_valid"]
        assert p["n_windows"] == 8 and p["n_windows_valid"] == 8 and p["n_users"] == 2 and len(p["per_window"]) == 8
        assert [(w["session"], *w["window"]) for w in p["per_window"]] == [tuple(w) for w in ident["windows"]]  # stesse finestre a ogni passo
        for w in p["per_window"]:
            assert w["n_valid"] > 0 and w["mse"] >= 0 and w["mae"] >= 0 and w["mae"] <= np.sqrt(w["mse"]) + 1e-9 and w["pred_var"] >= 0
            assert w["mae_rad"] == pytest.approx(0.25 * w["mae"], rel=1e-6)
            assert sum(w["n_valid_per_joint"]) == w["n_valid"] and w["n_joints_temporal"] == 20
            assert w["user"] in ident["users"] and w["session"].startswith(f"emg2pose/{w['user']}/")
        assert p["mse"] == pytest.approx(np.mean([w["mse"] for w in p["per_window"]]))
    # al passo 0 la testa e' a zero: predice la costante (0 = media del train standardizzata), varianza delle predizioni nulla
    p0 = panel[0]
    assert p0["pred_var"] == 0.0 and all(w["mse"] == pytest.approx(w["mse_const"], rel=1e-6) for w in p0["per_window"])
    assert all(w["mse_const"] > 0 for w in p0["per_window"]) and panel[1]["pred_var"] > 0  # dopo tre passi la testa non e' piu' costante
    # ripresa: interrotto al passo 4, ripreso fino al 6: stesso insieme, nessuna ripetizione del passo 0, stesse valutazioni
    run(replace(cfg, max_steps=4), "resume")
    assert [p["step"] for p in _records(tmp_path / "resume" / "pose_panel.jsonl")] == [0, 3]
    run(cfg, "resume")
    assert [p["step"] for p in _records(tmp_path / "resume" / "pose_panel.jsonl")] == [0, 3, 6]
    assert json.loads((tmp_path / "resume" / "pose_panel_set.json").read_text()) == ident
    assert _records(tmp_path / "resume" / "pose_panel.jsonl")[-1]["mse"] == pytest.approx(panel[-1]["mse"], rel=1e-5)
    # insieme cambiato sotto il run: la ripresa si rifiuta
    run(replace(cfg, max_steps=4), "tampered")
    bad = json.loads((tmp_path / "tampered" / "pose_panel_set.json").read_text())
    bad["windows"][0][2] += 1
    (tmp_path / "tampered" / "pose_panel_set.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="pannello della posa"):
        run(cfg, "tampered")


def test_panel_skips_ineligible_support_and_never_uses_inactive_or_duplicate_recordings(tmp_path):
    root, mpath, pose_root = _multi_tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    cfg = replace(_base(pose_root), pose_panel_users=2, pose_panel_recordings=2, pose_panel_windows=2)
    # u1 ha una registrazione senza posa valida: va sostituito con un utente che riempie le quote, non con finestre vuote.
    np.save(pose_root / "u1" / "sessA" / PT.VALID_FILE, np.zeros(N // PT.BLOCK, dtype=bool))
    _, ident = R._pose_panel_set(idx, cfg, {}, None)
    assert ident["users"] == ["u2", "u3"] and all(n > 0 for n in ident["n_valid"])
    assert all(len(n) == 20 for n in ident["n_valid_per_joint"])
    # Un duplicato della sola registrazione attiva non soddisfa il requisito di due registrazioni.
    row = next(r for r in idx.rows if r["subject"] == "u2")
    duplicate = replace(idx, rows=[row, row], weights=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="utenti di emg2pose"):
        R._pose_panel_set(duplicate, replace(cfg, pose_panel_users=1), {}, None)
    # La probabilita' effettiva include le esclusioni window_seconds, non solo i pesi grezzi.
    seconds = {R._session_key(r): (0.0 if r["subject"] == "u3" else 4.0) for r in idx.rows}
    timed = replace(cfg, loader=replace(cfg.loader, time_weighted=True))
    with pytest.raises(ValueError, match="supporto valido sufficiente"):
        R._pose_panel_set(idx, timed, {}, seconds)
    zero_weight = replace(idx, weights=np.array([0.0 if r["subject"] == "u3" else 0.25 for r in idx.rows]))
    with pytest.raises(ValueError, match="supporto valido sufficiente"):
        R._pose_panel_set(zero_weight, cfg, {}, None)


def test_temporal_prediction_variance_does_not_confuse_joint_bias_with_movement():
    pred = torch.tensor([[[1., 9.], [1., 9.], [1., 9.]]])
    valid = torch.ones_like(pred, dtype=torch.bool)
    assert R._pose_temporal_variance(pred, valid).item() == 0.0
    pred[0, :, 0] = torch.tensor([0., 2., 4.])
    pred[0, 2, 1] = 1000.  # valore mascherato: non conta
    valid[0, 2, 1] = False
    assert R._pose_temporal_variance(pred, valid).item() == pytest.approx((8 / 3) / 2)


def test_panel_guards(tmp_path):
    root, mpath, pose_root = _multi_tree(tmp_path)
    base = _base(pose_root)
    run = lambda c, out: R.train(c, mpath, [root], tmp_path / out, log=lambda m: None)  # noqa: E731
    with pytest.raises(ValueError, match="pose_panel_every >= 0"):
        run(replace(base, pose_panel_every=-1), "bad1")
    with pytest.raises(ValueError, match="pose_panel_every >= 0"):
        run(replace(base, pose_panel_every=3, pose_panel_windows=0), "bad2")
    with pytest.raises(ValueError, match="solo col termine di posa"):
        run(replace(base, pose_panel_every=3, jepa=replace(base.jepa, pose_weight=0.0, jepa_weight=1.0), model=replace(base.model, pose_head=False),
                    loader=replace(base.loader, pose_root=None)), "bad3")
    with pytest.raises(ValueError, match="utenti di emg2pose"):  # 3 utenti nel manifest, chiesti 4
        run(replace(base, pose_panel_every=3, pose_panel_users=4, pose_panel_recordings=2, pose_panel_windows=1), "bad4")
    with pytest.raises(ValueError, match="utenti di emg2pose"):  # nessun utente con 3 sessioni
        run(replace(base, pose_panel_every=3, pose_panel_users=1, pose_panel_recordings=3, pose_panel_windows=1), "bad5")
    # i campi nuovi stanno nel config (identita' del run) e un config vecchio li riceve coi default
    d = R.config_to_dict(replace(base, pose_panel_every=500))
    assert d["pose_panel_every"] == 500 and (d["pose_panel_users"], d["pose_panel_recordings"], d["pose_panel_windows"]) == (32, 2, 4)
    assert {k: R._LATER_DEFAULTS[k] for k in ("pose_panel_every", "pose_panel_users", "pose_panel_recordings", "pose_panel_windows")} == \
        {"pose_panel_every": 0, "pose_panel_users": 32, "pose_panel_recordings": 2, "pose_panel_windows": 4}


def test_materialize_reproduces_batch_from_the_same_draws(tmp_path):
    root, mpath, _ = _multi_tree(tmp_path)
    idx = L.ManifestIndex.load(mpath, [root])
    a = L.PretrainLoader(idx, _loader_cfg(block_s=None)).batch(3, np.random.default_rng(7))
    loader = L.PretrainLoader(idx, _loader_cfg(block_s=None))
    rng = np.random.default_rng(7)
    picked = [loader.sample(rng) for _ in range(3)]
    b = loader.materialize(picked, rng)
    assert batch_sha256(a) == batch_sha256(b) and a.rows == b.rows and [w.start for w in a.windows] == [w.start for w in b.windows]
    assert loader.materialize(picked, np.random.default_rng(7), dry=True) is None


def test_launcher_flags_for_the_panel():
    sp = importlib.util.spec_from_file_location("train_jepa_panel", REPO / "scripts" / "train_jepa.py")
    TJ = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(TJ)
    base = ["--preset", "sanity", "--manifest", "-", "--root", "-", "--out-dir", "-"]
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--pose-root", "/x", "--jepa-weight", "0"]))
    assert cfg.pose_panel_every == 0  # spento di default: la ricetta di R3 lo accende esplicitamente
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--pose-root", "/x", "--jepa-weight", "0", "--pose-panel-every", "500"]))
    assert cfg.pose_panel_every == 500 and (cfg.pose_panel_users, cfg.pose_panel_recordings, cfg.pose_panel_windows) == (32, 2, 4)
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--pose-root", "/x", "--jepa-weight", "0", "--pose-panel-every", "3",
                                                                "--pose-panel-users", "2", "--pose-panel-recordings", "1", "--pose-panel-windows", "5"]))
    assert (cfg.pose_panel_users, cfg.pose_panel_recordings, cfg.pose_panel_windows) == (2, 1, 5) and R.config_to_dict(cfg)["pose_panel_every"] == 3
    with pytest.raises(ValueError, match="--pose-panel-every senza --pose-root"):
        TJ.build_config(TJ.build_parser().parse_args(base + ["--pose-panel-every", "500"]))
