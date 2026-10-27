"""Valutazione della posa fuori campione (emendamento 1, E1-E3; `harness/pose_eval.py`, `scripts/pose_eval.py`): registrazioni dal CSV e
dall'albero processato, finestre fisse non sovrapposte sulla griglia del loader, estrazione col modello vero (latenti per istante, testa, target e
maschere), regole congelate di E1/E2/E3 su dati costruiti, appaiamento del controllo disallineato, ridge, CLI da un capo all'altro, launcher."""

import csv
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_pose_targets import _stats  # noqa: E402
from wearusfm.data import pose_targets as PT  # noqa: E402
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.harness import pose_eval as PE  # noqa: E402
from wearusfm.ingest import emg2pose  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model.fm import FMConfig, WearUsFM  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "pose_eval.py"
CFG = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4, pose_head=True)
FS, N = 2000.0, 24000  # 12 s


def _recording(root: Path, pose_root: Path, user: str, stem: str, side: str, seed: int, phase: float):
    rng = np.random.default_rng(seed)
    m = emg2pose.build_montage_metadata(user, stem, side)
    x = rng.normal(scale=3.0, size=(N, 16))
    d = root / f"u{user}" / stem
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "data_int16.npy", np.round(x * 1000).astype(np.int16))
    (d / "metadata.json").write_text(json.dumps({"montage": montage_to_dict(m, [True] * 16), "native_fs_hz": FS, "shape": [N, 16],
                                                 "int16_scale": 1000.0, "csv_row": {"side": side, "moving_hand": "both"}}))
    t = np.arange(N // PT.BLOCK) * PT.BLOCK / FS
    blocks = (0.5 * np.sin(2 * np.pi * (0.3 + 0.05 * np.arange(20))[None, :] * t[:, None] + phase) + 1.0).astype(np.float16)
    pd = pose_root / f"u{user}" / stem
    pd.mkdir(parents=True, exist_ok=True)
    np.save(pd / PT.POSE_FILE, blocks)
    np.save(pd / PT.VALID_FILE, np.ones(blocks.shape[0], dtype=bool))


def _setup(tmp_path):
    root, pose_root = tmp_path / "processed" / "emg2pose", tmp_path / "pose"
    rows = []
    specs = [("1", "val", "A", "left", 0.0), ("1", "val", "B", "left", 0.3), ("2", "val", "A", "right", 1.0), ("2", "val", "B", "right", 1.3),
             ("3", "train", "A", "left", 2.0), ("3", "train", "B", "left", 2.3), ("4", "train", "A", "right", 0.7)]
    for i, (user, split, stage, side, phase) in enumerate(specs):
        stem = f"2021-01-0{i + 1}-1-h{i}-cv-emg-pose-{stage}@2-recording-0_{side}"
        _recording(root, pose_root, user, stem, side, seed=i, phase=phase)
        rows.append({"session": f"s{user}", "user": user, "stage": stage, "start": 0, "end": 12, "side": side, "filename": stem + ".hdf5",
                     "moving_hand": "both", "held_out_user": split == "val", "held_out_stage": False, "split": split, "generalization": "user"})
    csvp = tmp_path / "meta.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    _stats(pose_root, mean=[1.0] * 20, std=[0.35] * 20)
    return root, pose_root, csvp


def test_recordings_and_fixed_windows(tmp_path):
    root, pose_root, csvp = _setup(tmp_path)
    val = PE.recordings_for(csvp, root, "val")
    assert len(val) == 4 and {r["user"] for r in val} == {"1", "2"} and all(Path(r["path"]).exists() for r in val)
    sub = PE.recordings_for(csvp, root, "train", n_users=1, n_recordings=1)
    assert len(sub) == 1 and sub[0]["user"] in ("3", "4")
    cfg = L.signed_config()
    view = L.SessionView.open(Path(val[0]["path"]), True)
    wins = PE.fixed_windows(view, cfg, 10, 4.0, PE.SEED_WINDOWS)
    assert 1 <= len(wins) <= 3 and all(w.n_patches == 160 and w.stop - w.start == 8000 for w in wins)  # 12 s: al piu' 3 finestre da 4 s
    starts = sorted(w.start for w in wins)
    assert all(b - a >= 8000 for a, b in zip(starts, starts[1:])) and all(s % 400 == 0 for s in starts)  # griglia a 200 ms
    assert [w.start for w in PE.fixed_windows(view, cfg, 10, 4.0, PE.SEED_WINDOWS)] == [w.start for w in wins]  # ripetibili


def test_extract_matches_the_store_and_the_head(tmp_path):
    root, pose_root, csvp = _setup(tmp_path)
    torch.manual_seed(0)
    model = WearUsFM(CFG).eval()
    with torch.no_grad():
        model.pose_head.out.weight.normal_(std=0.1)
    store = PT.PoseStore(pose_root)
    recs = PE.recordings_for(csvp, root, "val")
    ev = PE.EvalConfig(n_windows=2, loader=L.signed_config())
    data = PE.extract(model, recs, store, ev, log=lambda m: None)
    n = data["lat"].shape[0]
    assert data["has_head"] and data["dim"] == 16 and 4 <= n <= 8 and data["lat"].shape == (n, 160, 16) and data["target"].shape == (n, 160, 20)
    assert data["mask"].all() and np.allclose(data["std"], 0.35) and np.isfinite(data["pose_var"]).all() and len(data["meta"]) == n
    # i target sono quelli di PoseStore sulla stessa finestra; la predizione e' la testa sui latenti estratti
    m0 = data["meta"][0]
    view = L.SessionView.open(Path(recs[m0["recording"]]["path"]), True)
    t, mk, sd = store.window({"dataset": "emg2pose"}, view, L.Window(None, m0["start"], m0["start"] + 8000, (0, N), 160), 25.0)
    assert np.allclose(t, data["target"][0]) and np.array_equal(mk, data["mask"][0])
    with torch.no_grad():
        pred0 = model.pose_head(torch.from_numpy(data["lat"][0].astype(np.float32))[None]).numpy()[0]
    assert np.allclose(pred0, data["pred"][0], atol=1e-2)  # latenti in float16
    PE.save_npz(tmp_path / "v.npz", data)
    back = PE.load_npz(tmp_path / "v.npz")
    assert np.array_equal(back["lat"], data["lat"]) and back["meta"] == data["meta"] and back["config"]["seed"] == PE.SEED_WINDOWS


def _fake(n_users=6, per_user=8, P=20, seed=0, head="half"):
    """Dati costruiti: utenti x finestre, due fasi, target sinusoidali; `head`: half = meta' dell'errore del costante, zero = costante,
    perfect = esatta."""
    rng = np.random.default_rng(seed)
    n = n_users * per_user
    t = np.linspace(0, 1, P)
    target = np.stack([np.sin(2 * np.pi * (1 + rng.random()) * t + rng.random() * 6)[:, None] * np.ones((1, 20)) * (0.5 + rng.random())
                       for _ in range(n)]).astype(np.float32)
    target += rng.normal(scale=0.05, size=target.shape).astype(np.float32)
    mask = np.ones_like(target, dtype=bool)
    pred = {"half": target * 0.5, "zero": np.zeros_like(target), "perfect": target}[head].astype(np.float32)
    meta = [{"user": f"u{i // per_user}", "session": f"s{i // per_user}", "stage": "A" if (i % per_user) < per_user // 2 else "B", "side": "left",
             "stem": f"r{i}", "recording": i // 2, "start": (i % 2) * 20000, "fs": 2000.0, "n_patches": P} for i in range(n)]
    a = np.random.default_rng(99).normal(size=(20, 16)).astype(np.float32)  # la stessa mappa lineare posa -> latenti per train e val
    lat = (target @ a + rng.normal(scale=0.1, size=(n, P, 16))).astype(np.float16)
    return {"lat": lat, "pred": pred, "target": target, "mask": mask, "std": np.full((n, 20), 0.5, dtype=np.float32),
            "pose_var": target.var(axis=1).mean(axis=1).astype(np.float32), "has_head": True, "dim": 16, "meta": meta, "n_recordings": n // 2,
            "config": {"seed": 1}}


def test_rules_e1_e2_e3_on_constructed_data():
    val = _fake(head="half")
    rep = PE.report(val, _fake(seed=3, head="zero"))
    d = rep["E1"]["dynamic"]
    assert rep["n_dynamic"] == val["lat"].shape[0] // 2 and abs(d["ratio"] - 0.5) < 0.02 and d["ratio_ci95_users"][1] < 0.9 and rep["E1"]["learned"]
    assert d["n_users"] == 6 and all(abs(v["ratio"] - 0.5) < 0.05 for v in d["per_user"].values()) and len(rep["E1"]["per_joint_mae"]) == 20
    assert rep["E1"]["all"]["mae_const"] > 0 and rep["E1"]["static"]["n_windows"] + d["n_windows"] == val["lat"].shape[0]
    # E2: coppie stesso utente e stessa fase; con la testa «a meta'» il disallineato NON batte il costante
    e2 = rep["E2"]
    assert e2["pairing"]["n_with_partner"] == val["lat"].shape[0] and e2["pairing"]["n_kept"] >= 0.8 * val["lat"].shape[0]
    assert e2["ratio"] > 0.95 and e2["misaligned_ok"] and all(np.isfinite(g) for g in e2["gap_mae_per_user"].values())
    partner, _ = PE.misaligned_pairs(val["meta"], val["target"], val["mask"])
    for i, j in enumerate(partner):
        if j >= 0:
            mi, mj = val["meta"][i], val["meta"][j]
            assert mi["user"] == mj["user"] and mi["stage"] == mj["stage"] and mi["side"] == mj["side"] and i != j
            if mi["recording"] == mj["recording"]:
                assert abs(mi["start"] - mj["start"]) >= 8 * 2000
            else:
                assert mi["session"] != mj["session"]
    # il gap e' sullo stesso supporto: MAE disallineata del blocco = MAE disallineata del supporto appaiato (stesse finestre, stessi pesi)
    assert abs(e2["paired_support"]["mae_misaligned"] - e2["mae"]) < 1e-9 and e2["paired_support"]["n_windows"] == e2["n_windows"]
    # E3: latenti che codificano linearmente la posa -> la ridge batte il costante
    assert rep["E3"]["dynamic"]["ratio"] < 0.5 and rep["E3"]["accessible"] and rep["E3"]["alpha"] in PE.RULE["ridge_alphas"]
    # testa a zero = costante: ratio 1, non appresa; testa perfetta: ratio 0
    rep0 = PE.report(_fake(head="zero"))
    assert abs(rep0["E1"]["dynamic"]["ratio"] - 1.0) < 1e-6 and not rep0["E1"]["learned"] and "E3" not in rep0
    rep1 = PE.report(_fake(head="perfect"))
    assert rep1["E1"]["dynamic"]["ratio"] == 0.0 and rep1["E1"]["learned"] and rep1["E1"]["learned_substantially"]


def _three_windows(shape=(3, 4, 20)):
    """Il controesempio della review di GPT del 27/10/2026 (`docs/reviews/repro_pose_eval_20261027.py`): tre finestre dinamiche di un utente;
    le prime due nella stessa registrazione a 8 s di distanza (errore allineato 0, disallineato 1), la terza in un'altra fase senza partner
    (errore allineato 100)."""
    wave = np.asarray([-1, 1, -1, 1], dtype=np.float32)[None, :, None]
    target = np.broadcast_to(wave + np.asarray([1, 2, 100])[:, None, None], shape).astype(np.float32).copy()
    pred = target.copy()
    pred[2] += 100
    meta = [dict(user="u", session="s", stage="A" if i < 2 else "B", side="left", recording=0 if i < 2 else 1,
                 start=i * 16000 if i < 2 else 0, fs=2000.0, n_patches=4) for i in range(3)]
    return dict(meta=meta, lat=np.zeros((3, 4, 2), dtype=np.float16), target=target, pred=pred, mask=np.ones(shape, dtype=bool),
                std=np.ones((3, 20), dtype=np.float32), pose_var=target.var(axis=1).mean(axis=1), has_head=True, dim=2, n_recordings=2, config={})


def test_e2_gap_is_computed_on_the_same_support_as_the_misaligned_error():
    # controesempio della review: prima della correzione il gap riportato era -32,33 (MAE allineata su TUTTE le finestre dinamiche, 100/3);
    # sul supporto delle coppie e' +1
    data = _three_windows()
    rep = PE.report(data)
    e2 = rep["E2"]
    assert abs(e2["gap_mae_per_user"]["u"] - 1.0) < 1e-9 and abs(e2["paired_support"]["gap_mae"] - 1.0) < 1e-9
    ps = e2["paired_support"]
    assert ps["n_windows"] == 2 and ps["n_users"] == 1 and ps["mae_aligned"] == 0.0 and abs(ps["mae_misaligned"] - 1.0) < 1e-9
    assert ps["per_user"]["u"] == {"mae_aligned": 0.0, "mae_misaligned": ps["per_user"]["u"]["mae_misaligned"], "gap": ps["per_user"]["u"]["gap"],
                                   "n_windows": 2, "n_valid_elements": 2 * 4 * 20, "n_same_recording": 2, "n_other_session": 0}
    assert e2["pairing"]["n_no_candidate"] == 1 and e2["pairing"]["n_same_recording"] == 2 and e2["pairing"]["n_kept"] == 2
    # la E1 non cambia: la MAE allineata su tutte le finestre dinamiche resta 100 / 3
    assert abs(rep["E1"]["dynamic"]["per_user"]["u"]["mae"] - 100 / 3) < 1e-9 and rep["E1"]["dynamic"]["n_windows"] == 3
    # maschere parziali diverse: il supporto e' l'intersezione, identico per i due errori; un errore allineato fuori dall'intersezione non conta
    d2 = _three_windows()
    d2["mask"][0, 2:, :] = False  # finestra 0 valida solo nelle patch 0-1
    d2["mask"][1, :1, :] = False  # finestra 1 valida solo nelle patch 1-3 -> intersezione: patch 1
    d2["pred"][0, 2:, :] += 5.0  # errore allineato fuori dall'intersezione (patch 2-3): invisibile al gap, visibile alla E1? no: la E1 usa la
    #                               maschera della finestra, che li' e' falsa; allora errore nella patch 0, valida per la 0 ma fuori dall'intersezione
    d2["pred"][0, 0, :] += 5.0
    rep2 = PE.report(d2)
    ps2 = rep2["E2"]["paired_support"]
    assert ps2["n_valid_elements"] == 2 * 1 * 20 and ps2["per_user"]["u"]["n_valid_elements"] == 40
    assert ps2["mae_aligned"] == 0.0 and abs(ps2["mae_misaligned"] - 1.0) < 1e-9 and abs(ps2["gap_mae"] - 1.0) < 1e-9
    assert rep2["E1"]["dynamic"]["per_user"]["u"]["mae"] > 100 / 3  # la E1 vede l'errore nella patch 0 della finestra 0
    assert abs(rep2["E2"]["mae"] - ps2["mae_misaligned"]) < 1e-9  # il gate usa lo stesso supporto del gap
    # riproducibilita': due letture identiche
    assert json.dumps(PE.report(data)["E2"], sort_keys=True) == json.dumps(rep["E2"], sort_keys=True)


def test_e2_pairs_require_same_side_and_fall_back_to_another_session_only():
    base = _three_windows()
    meta = [dict(m) for m in base["meta"]]
    target, mask = base["target"], base["mask"]
    # mani opposte della stessa sessione allo stesso istante (registrazioni diverse): prima della correzione erano partner [1, 0]; ora nessuno
    meta[1].update(side="right", recording=1, start=0)
    partner, info = PE.misaligned_pairs(meta[:2], target[:2], mask[:2])
    assert partner.tolist() == [-1, -1] and info["n_no_candidate"] == 2 and info["n_with_partner"] == 0 and info["n_other_session"] == 0
    data = dict(base, meta=meta[:2], target=target[:2], pred=base["pred"][:2], mask=mask[:2], std=base["std"][:2], pose_var=base["pose_var"][:2],
                lat=base["lat"][:2])
    rep = PE.report(data)
    assert rep["E2"]["misaligned_ok"] is None and rep["E2"]["note"] == "nessuna coppia" and rep["E2"]["pairing"]["n_no_candidate"] == 2
    # fallback: stessa sessione e registrazione diversa NON basta (B); un'altra sessione dello stesso utente, fase e lato si' (C); la stessa
    # registrazione a >= 8 s ha la precedenza (D per A)
    meta4 = [dict(user="u", session="s1", stage="A", side="left", recording=0, start=0, fs=2000.0, n_patches=4),      # A
             dict(user="u", session="s1", stage="A", side="left", recording=1, start=0, fs=2000.0, n_patches=4),      # B: stessa sessione
             dict(user="u", session="s2", stage="A", side="left", recording=2, start=0, fs=2000.0, n_patches=4),      # C: altra sessione
             dict(user="u", session="s1", stage="A", side="left", recording=0, start=16000, fs=2000.0, n_patches=4)]  # D: come A, 8 s dopo
    rng = np.random.default_rng(0)
    t4 = rng.normal(size=(4, 4, 20)).astype(np.float32)
    m4 = np.ones((4, 4, 20), dtype=bool)
    p4, i4 = PE.misaligned_pairs(meta4, t4, m4, drop_quantile=0.0)
    assert p4[0] == 3 and p4[3] == 0  # A <-> D: stessa registrazione a 8 s
    assert p4[1] == 2 and p4[2] in (0, 1, 3)  # B: solo C (altra sessione), mai A o D (stessa sessione); C: una qualunque dell'altra sessione
    assert i4["n_same_recording"] == 2 and i4["n_other_session"] == 2 and i4["n_no_candidate"] == 0 and i4["n_kept"] == 4
    # i conteggi del supporto appaiato distinguono stessa registrazione e altra sessione
    d4 = dict(meta=meta4, lat=np.zeros((4, 4, 2), dtype=np.float16), target=t4, pred=t4.copy(), mask=m4, std=np.ones((4, 20), dtype=np.float32),
              pose_var=np.ones(4, dtype=np.float32), has_head=True, dim=2, n_recordings=3, config={})  # tutte dinamiche
    ps = PE.report(d4, rule={**PE.RULE, "misaligned_drop_quantile": 0.0})["E2"]["paired_support"]
    assert ps["n_same_recording"] + ps["n_other_session"] == ps["n_windows"] and ps["n_other_session"] >= 1
    # senza candidati i vincoli non si rilassano: utente solo con fasi diverse, o lati diversi, o una sola sessione e finestre vicine
    lone = [dict(user="u", session="s1", stage="A", side="left", recording=0, start=0, fs=2000.0, n_patches=4),
            dict(user="u", session="s1", stage="A", side="left", recording=0, start=4000, fs=2000.0, n_patches=4),  # 2 s: troppo vicina
            dict(user="u", session="s1", stage="B", side="left", recording=1, start=0, fs=2000.0, n_patches=4),
            dict(user="u", session="s2", stage="A", side="right", recording=2, start=0, fs=2000.0, n_patches=4)]
    pl, il = PE.misaligned_pairs(lone, t4, m4)
    assert pl.tolist() == [-1, -1, -1, -1] and il["n_no_candidate"] == 4 and il["n_with_partner"] == 0
    # determinismo: stessa scelta a ogni chiamata (seme fisso, ordine delle finestre)
    assert PE.misaligned_pairs(meta4, t4, m4, drop_quantile=0.0)[0].tolist() == p4.tolist()
    # coordinate incompatibili (std diverse nella coppia) si escludono e si contano, anche se l'appaiamento le ammette
    d5 = dict(d4)
    d5["std"] = np.ones((4, 20), dtype=np.float32)
    d5["std"][2] = 2.0  # C con statistiche diverse: le coppie con C cadono
    rep5 = PE.report(d5, rule={**PE.RULE, "misaligned_drop_quantile": 0.0})
    assert rep5["E2"]["pairing"]["n_excluded_std_mismatch"] >= 1 and rep5["E2"]["paired_support"]["n_windows"] + rep5["E2"]["pairing"]["n_excluded_std_mismatch"] == 4


def test_report_without_head_only_runs_e3():
    val = _fake(head="zero")
    val["has_head"] = False
    rep = PE.report(val, _fake(seed=5))
    assert "E1" not in rep and "E2" not in rep and "E3" in rep and rep["has_head"] is False


def test_cli_extract_and_report_end_to_end(tmp_path):
    root, pose_root, csvp = _setup(tmp_path)
    torch.manual_seed(0)
    model = WearUsFM(CFG)
    ck = tmp_path / "checkpoint_postdecay.pt"
    torch.save({"config": {"model": asdict(CFG)}, "teacher": model.state_dict(), "student": model.state_dict(), "pose_stats_sha256": "ab" * 32}, ck)
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}
    common = [sys.executable, str(SCRIPT), "--stage", "extract", "--checkpoint", str(ck), "--csv", str(csvp), "--processed-root", str(root),
              "--pose-root", str(pose_root), "--n-windows", "2"]
    r = subprocess.run(common + ["--split", "val", "--out", str(tmp_path / "val.npz")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(common + ["--split", "train", "--n-users", "2", "--n-recordings", "3", "--out", str(tmp_path / "train.npz")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(SCRIPT), "--stage", "report", "--val", str(tmp_path / "val.npz"), "--train", str(tmp_path / "train.npz"),
                        "--out", str(tmp_path / "rep.json")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["has_head"] and "E1" in rep and "E2" in rep and "E3" in rep and rep["n_users"] == 2 and rep["config"]["split"] == "val"
    assert abs(rep["E1"]["dynamic"]["ratio"] - 1.0) < 1e-5 and rep["E1"]["learned"] is False  # testa a zero = costante
    assert rep["config"]["checkpoint_pose_stats_sha256"] == "ab" * 32
    summary = json.loads(r.stdout)
    assert summary["E1"]["verdict"] is False


def test_launchers_use_the_frozen_checkout_and_write_outside_the_repo():
    for name in ("pose_eval_extract.sbatch", "pose_eval_report.sbatch"):
        text = (REPO / "scripts" / "slurm" / name).read_text()
        assert "lib_code_dir.sh" in text and "code_dir_setup" in text and "dentro il codice: rifiutat" in text
    assert "--gres=gpu:1" in (REPO / "scripts" / "slurm" / "pose_eval_extract.sbatch").read_text()
    assert "lrd_all_serial" in (REPO / "scripts" / "slurm" / "pose_eval_report.sbatch").read_text()
