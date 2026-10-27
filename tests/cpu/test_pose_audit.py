"""Audit della catena dei bersagli di posa su dati costruiti con geometria reale (`scripts/pose_audit.py`; correzioni R1-R4 della review GPT
`docs/reviews/pose_audit_e198a87_review_20261027.md`): UNA fixture HDF5 attraversa l'ingest dell'EMG e quello della posa; stati
verificato / fallito / non verificato con copertura; EMG sostituito o spostato fallisce anche con sidecar plausibili; senza tar l'audit e' parziale;
oracolo con NaN / Inf; movimento su coppie adiacenti valide; ridge con split interno; disallineato come in E2; scansione del ritardo esplorativa
che recupera segno e grandezza di uno spostamento noto (200 ms = 8 patch) senza un verdetto «minimo a zero»."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_ingest_emg2pose_pose import _tar  # noqa: E402
from test_pose_targets import _stats  # noqa: E402
from wearusfm.data import pose_targets as PT  # noqa: E402
from wearusfm.ingest import emg2pose as E2P  # noqa: E402
from wearusfm.ingest import emg2pose_pose as PP  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "pose_audit.py"
sys.path.insert(0, str(REPO / "scripts"))
import pose_audit as PA  # noqa: E402

FS, N, M = 2000.0, 120000, 280  # 60 s a 2 kHz: ~11 finestre da 4 s per registrazione (coppie a >= 8 s fra le finestre di test)
SPECS = [("1", "sA", "A", "left", 1), ("1", "sB", "A", "left", 2), ("2", "sC", "A", "right", 3), ("2", "sD", "B", "right", 4)]


def _hdf5(angles, emg, rate=2000.0):
    n = angles.shape[0]
    dt = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)), ("emg", "<f4", (16,))])
    arr = np.zeros(n, dtype=dt)
    arr["time"], arr["joint_angles"], arr["emg"] = np.arange(n) / rate, angles, emg
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        g = f.create_group("emg2pose")
        g.create_dataset("timeseries", data=arr)
        g.attrs["sample_rate"] = rate
    return buf.getvalue()


SPECS_SIM = SPECS + [("3", "sE", "A", "left", 5), ("3", "sE", "A", "right", 6)]  # due lati simultanei della stessa sessione


def _pose_and_emg(n, seed, shift_blocks=0, nonfinite=False):
    """Posa lenta per giunto; EMG col canale c modulato in ampiezza dal giunto c all'ISTANTE STESSO (log-RMS lineare nella posa, nessun
    ritardo). `shift_blocks` sposta la posa scritta nel file rispetto all'EMG (posa disallineata di proposito); `nonfinite` mette NaN e Inf
    in qualche riga (righe non valide da sole, senza dilatazione)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    pose = (0.6 * np.sin(2 * np.pi * (0.25 + 0.05 * np.arange(20))[None, :] * t[:, None] + rng.normal(size=20)[None, :]) + 1.0).astype(np.float32)
    amp = 0.3 + 0.7 * (pose[:, :16] - 0.4) / 1.2
    emg = (rng.normal(size=(n, 16)) * amp * 3.0).astype(np.float32)
    written = np.roll(pose, shift_blocks * PT.BLOCK, axis=0) if shift_blocks else pose.copy()
    written[5000:5300] = 0.0  # un blocco di zeri (fallimento della cinematica inversa)
    if nonfinite:
        written[20000:20003, 4] = np.nan
        written[40000, 7] = np.inf
    return written, emg


def _setup(tmp_path, shift_blocks=0, mutate=None, *, specs=SPECS, per_side=False, nonfinite=False, csv_start=True, mutate_pose=None):
    """Alberi EMG e posa dall'UNICA fixture HDF5 per registrazione, con i due ingest veri; CSV con inizi assoluti (stessa sessione = stesso
    inizio; `csv_start=False` li toglie dal CSV); tar con gli stessi byte. `mutate`: funzione (stem, dir_emg) applicata DOPO l'ingest (EMG
    sostituito o spostato); `mutate_pose`: lo stesso sull'albero della posa (sidecar alterato)."""
    root, pose_root = tmp_path / "processed" / "emg2pose", tmp_path / "pose"
    rows, recs, starts = [], {}, {}
    for user, sess, stage, side, seed in specs:
        stem = f"2021-01-0{seed}-1-h{seed}-cv-emg-pose-{stage}@2-recording-0_{side}"
        pose, emg = _pose_and_emg(N, seed, shift_blocks, nonfinite)
        data = _hdf5(pose, emg)
        start = starts.setdefault(sess, 1_600_000_000 + 100 * len(starts))
        row = {"session": sess, "user": user, "stage": stage, "start": start, "end": start + 30, "side": side,
               "filename": stem + ".hdf5", "moving_hand": "both", "held_out_user": False, "held_out_stage": False, "split": "train", "generalization": "user"}
        ref = E2P.RecordingRef(member=f"emg2pose_data/{stem}.hdf5", stem=stem)
        E2P.ingest_recording(io.BytesIO(data), ref, root, row)
        PP.ingest_pose_recording(io.BytesIO(data), ref, pose_root, row, zero_margin_samples=M, measure=False)
        recs[stem] = data
        rows.append(dict(row) if csv_start else {**row, "start": "", "end": ""})
        if mutate:
            mutate(stem, root / f"u{user}" / stem)
        if mutate_pose:
            mutate_pose(stem, pose_root / f"u{user}" / stem)
    if per_side:
        _stats(pose_root, mean=[1.0] * 20, std=[0.42] * 20, per_side=True,
               by_side={"left": {"mean": [1.0] * 20, "std": [0.42] * 20}, "right": {"mean": [1.1] * 20, "std": [0.5] * 20}})
    else:
        _stats(pose_root, mean=[1.0] * 20, std=[0.42] * 20)
    csvp = tmp_path / "meta.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root, pose_root, csvp, _tar(tmp_path / "emg2pose_dataset.tar", recs)


def _run(root, pose_root, csvp, tar, out, extra=(), n_users=2, n_recordings=4):
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    cmd = [sys.executable, str(SCRIPT), "--csv", str(csvp), "--processed-root", str(root), "--pose-root", str(pose_root),
           "--n-users", str(n_users), "--n-recordings", str(n_recordings), "--n-windows", "24", "--out", str(out),
           *([] if tar is None else ["--tar", str(tar)]), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(out.read_text())


def test_single_fixture_through_both_ingests_is_verified_and_the_ridge_beats_the_constant(tmp_path):
    root, pose_root, csvp, tar = _setup(tmp_path)
    rep = _run(root, pose_root, csvp, tar, tmp_path / "audit.json", extra=["--context", "0", "--lags", "12"])
    s = rep["sintesi"]
    assert rep["tar"] == {"found": 4, "missing": 0, "index_seconds": rep["tar"]["index_seconds"]} and s["stato_catena"] == "verificato"
    assert all(c["verificato"] == 4 and c["fallito"] == 0 and c["non_verificato"] == 0 for c in s["copertura"].values())
    for r in rep["per_recording"]:
        assert all(r["sidecar"].values()) and r["emg"]["errore_massimo"] <= r["emg"]["tolleranza"] + 1e-5 and r["emg"]["errore_massimo_blocco"] < 1e-4
        assert r["frac_valid_blocks"] < 1.0 and r["movimento_adiacenti_validi"] is not None
    assert sum(s["blocchi_validi_per_patch_istogramma_0_5"]) > 0 and s["blocchi_validi_per_patch_istogramma_0_5"][5] > 0
    for u, v in rep["ridge"].items():
        assert v["n_fit"] >= 2 and v["n_gap"] >= 1 and v["n_test"] >= 2 and v["n_fit"] + v["n_gap"] + v["n_test"] == v["n_windows"]
        assert v["cronologia"].startswith("tempo assoluto") and v["simultanee_fit_test"] == 0 and v["alpha"]["n_fit_interno"] >= 2
        assert v["ratio_all"] < 0.8 and v["mae"] < v["mae_const"]
        d = v["disallineato"]
        assert d["stato"] == "verificato" and d["n_coppie"] >= 1 and d["mae_disallineata"] > d["mae_allineata"] and d["gap"] > 0
        sc = v["scansione_ritardo"]
        assert sc["esplorativa"] and sc["per_lag"]["0"]["n"] == sc["per_lag"]["8"]["n"] == sc["n_righe_test"]  # supporto comune
        assert sc["mae_test_a_0"] <= sc["per_lag"]["8"]["mae"] and sc["mae_test_a_0"] <= sc["per_lag"]["-8"]["mae"]
        assert abs(sc["lag_operativo_dal_fit"]) <= 1  # allineato: nessuna pretesa che il minimo sia esattamente 0
    assert "lag_zero_per_tutti" not in s


def test_shifted_pose_is_recovered_in_sign_and_magnitude_by_the_lag_scan_only(tmp_path):
    root, pose_root, csvp, tar = _setup(tmp_path, shift_blocks=40)  # 40 blocchi = 400 campioni = 200 ms = 8 patch da 25 ms
    rep = _run(root, pose_root, csvp, tar, tmp_path / "audit_shift.json", extra=["--context", "0", "--lags", "12"])
    assert rep["sintesi"]["stato_catena"] == "verificato"  # la catena e' coerente con l'albero: il problema sta nei dati
    for u, v in rep["ridge"].items():
        sc = v["scansione_ritardo"]
        assert abs(sc["lag_operativo_dal_fit"] - 8) <= 1 and sc["mae_test_al_lag_operativo"] < 0.8 * sc["mae_test_a_0"]


def test_substituted_or_shifted_emg_fails_even_with_plausible_sidecars(tmp_path):
    def substitute(stem, d):
        other = np.load(d / "data_int16.npy")[::-1].copy()  # stesso file, al contrario: stessa scala e stessa forma
        np.save(d / "data_int16.npy", other)

    root, pose_root, csvp, tar = _setup(tmp_path / "sub", mutate=substitute)
    rep = _run(root, pose_root, csvp, tar, tmp_path / "sub.json", extra=["--lags", "0"])
    assert rep["sintesi"]["stato_catena"] == "fallito" and rep["sintesi"]["copertura"]["emg_grezzo"]["fallito"] == 4
    assert rep["sintesi"]["copertura"]["sidecar"]["verificato"] == 4 and rep["sintesi"]["copertura"]["posa_ricostruzione"]["verificato"] == 4

    def shift(stem, d):
        np.save(d / "data_int16.npy", np.roll(np.load(d / "data_int16.npy"), 100, axis=0))

    root, pose_root, csvp, tar = _setup(tmp_path / "shift", mutate=shift)
    rep = _run(root, pose_root, csvp, tar, tmp_path / "shift.json", extra=["--lags", "0"])
    assert rep["sintesi"]["stato_catena"] == "fallito" and rep["sintesi"]["copertura"]["emg_grezzo"]["fallito"] == 4


def test_without_raw_the_audit_is_partial_never_ok(tmp_path):
    root, pose_root, csvp, tar = _setup(tmp_path)
    rep = _run(root, pose_root, csvp, None, tmp_path / "partial.json", extra=["--lags", "0"])
    s = rep["sintesi"]
    assert s["stato_catena"] == "parziale" and rep["tar"] is None
    assert s["copertura"]["emg_grezzo"]["non_verificato"] == 4 and s["copertura"]["posa_ricostruzione"]["non_verificato"] == 4
    assert s["copertura"]["oracolo"]["non_verificato"] == 4 and s["copertura"]["sidecar"]["verificato"] == 4 and s["copertura"]["lettura_blocco"]["verificato"] == 4
    rep2 = _run(root, pose_root, csvp, tar, tmp_path / "budget0.json", extra=["--lags", "0", "--tar-index-budget-s", "0"])
    assert rep2["tar"]["found"] == 0 and rep2["sintesi"]["stato_catena"] == "parziale"


def test_oracle_follows_the_ingest_contract_for_nonfinite_rows_and_side_stats():
    angles = np.ones((2000, 20), np.float32)
    angles[0, 0] = np.nan
    angles[700, 3] = np.inf
    angles[1000:1010] = 0.0  # un blocco di zeri: dilatato di M; i non finiti no
    for margin in (0, 50, 280):
        t, m, n_v = PA.oracle_patch_targets(angles, 0, 40, 50, margin, np.zeros(20), np.ones(20))
        bad = PP.invalid_rows(angles, margin)
        blocks, valid = PP.block_reduce(angles, ~bad)
        assert not valid[0] and not valid[70] and np.isfinite(t).all()
        assert n_v[0] == 4 and n_v[14] == valid[70:75].sum() <= 4, (margin, n_v[:16])  # la prima patch perde il blocco col NaN; la 15a quello
        #                                                                                 con l'Inf (e, con M = 280, anche quelli nel margine degli zeri)
        exp = np.array([valid[5 * p:5 * p + 5].sum() for p in range(40)])
        assert np.array_equal(n_v, exp) and np.array_equal(m, exp > 0)
    # statistiche per lato: l'oracolo riceve quelle del lato della finestra (lo stesso che PoseStore.norm applica)
    t_l, _, _ = PA.oracle_patch_targets(angles, 0, 4, 50, 0, np.full(20, 0.5), np.full(20, 2.0))
    t_r, _, _ = PA.oracle_patch_targets(angles, 0, 4, 50, 0, np.full(20, 1.0), np.full(20, 0.5))
    assert np.allclose(t_l[1], 0.25) and np.allclose(t_r[1], 0.0)


def test_movement_never_jumps_across_a_gap_and_dynamic_patches_need_a_valid_neighbour():
    blocks = np.concatenate([np.zeros((10, 20)), np.full((5, 20), 9.0), np.ones((10, 20))])  # plateau 0, buco (valori 9), plateau 1
    valid = np.array([True] * 10 + [False] * 5 + [True] * 10)
    mv, ma, n = PA.movement_adjacent(blocks, valid)
    assert mv == 0.0 and n == 18 and ma > 0  # nessun salto attraverso il buco fra le coppie valide
    target = np.concatenate([np.zeros((5, 20)), np.ones((1, 20)), np.full((4, 20), 1.5)])
    v = np.array([True, True, True, False, False, True, True, True, True, True])
    dyn = PA.dynamic_patches(target, v)
    assert not dyn[5] and dyn[6]  # la patch 5 ha la precedente non valida: mai dinamica per il salto attraverso il buco


def test_per_side_stats_nonfinite_rows_and_simultaneous_sides_go_through_the_whole_audit(tmp_path):
    root, pose_root, csvp, tar = _setup(tmp_path, specs=SPECS_SIM, per_side=True, nonfinite=True)
    rep = _run(root, pose_root, csvp, tar, tmp_path / "sides.json", extra=["--context", "0", "--lags", "4"], n_users=3, n_recordings=6)
    s = rep["sintesi"]
    assert s["stato_catena"] == "verificato" and s["registrazioni"] == 6 and all(c["verificato"] == 6 for c in s["copertura"].values())
    assert all(r["posa_ricostruzione"]["frac_nonfinite_rows"] > 0 for r in rep["per_recording"])  # NaN/Inf attraversano ingest, albero e oracolo
    # due lati simultanei con le stesse finestre: la separazione del 10% (2 finestre = un istante per lato) copre la coppia a cavallo, quindi 0
    assert rep["ridge"]["3"]["simultanee_fit_test"] == 0 and rep["ridge"]["3"]["n_gap"] >= 2 and rep["ridge"]["3"]["disallineato"]["stato"] == "verificato"
    w = lambda t, side: {"session": "s", "stage": "A", "side": side, "t_abs": t}  # noqa: E731
    assert PA.simultaneous_pairs([w(10.0, "left")], [w(12.0, "right"), w(15.0, "right"), w(12.0, "left")], 4.0) == 1  # caso positivo del contatore


def test_pose_sidecar_with_wrong_stem_or_user_fails_the_sidecar_check_without_stopping_the_audit(tmp_path):
    def once(fn):
        done = []

        def wrapped(stem, d):
            if not done:
                done.append(stem)
                fn(stem, d)
        return wrapped

    def wrong_stem(stem, d):  # stessa cartella dopo la sanificazione ('#' -> '_'): la guardia di PoseStore.entry passa, il controllo `stem` no
        meta = json.loads((d / PT.META_FILE).read_text())
        meta["stem"] = "#".join(stem.rsplit("_", 1))
        (d / PT.META_FILE).write_text(json.dumps(meta))

    root, pose_root, csvp, tar = _setup(tmp_path / "stem", mutate_pose=once(wrong_stem))
    rep = _run(root, pose_root, csvp, tar, tmp_path / "stem.json", extra=["--lags", "0"])
    cov = rep["sintesi"]["copertura"]
    assert rep["sintesi"]["stato_catena"] == "fallito" and cov["sidecar"] == {"verificato": 3, "fallito": 1, "non_verificato": 0}
    bad = [r for r in rep["per_recording"] if r["stati"]["sidecar"] == "fallito"]
    assert len(bad) == 1 and bad[0]["sidecar"]["stem"] is False and bad[0]["sidecar"]["source_member"] and bad[0]["stati"]["emg_grezzo"] == "verificato"

    def wrong_user(stem, d):  # la guardia di PoseStore.entry rifiuta l'albero: registrazione fallita, l'audit arriva in fondo
        meta = json.loads((d / PT.META_FILE).read_text())
        meta["user"] = "99"
        (d / PT.META_FILE).write_text(json.dumps(meta))

    root, pose_root, csvp, tar = _setup(tmp_path / "user", mutate_pose=once(wrong_user))
    rep = _run(root, pose_root, csvp, tar, tmp_path / "user.json", extra=["--lags", "0"])
    cov = rep["sintesi"]["copertura"]
    assert rep["sintesi"]["stato_catena"] == "fallito" and cov["sidecar"] == {"verificato": 3, "fallito": 1, "non_verificato": 0}
    assert cov["emg_grezzo"] == {"verificato": 3, "fallito": 0, "non_verificato": 1} and rep["sintesi"]["registrazioni"] == 4
    bad = [r for r in rep["per_recording"] if r["stati"]["sidecar"] == "fallito"]
    assert len(bad) == 1 and "guardia d'identita'" in bad[0]["nota"] and len(rep["ridge"]) == 2


def test_without_absolute_start_in_the_csv_the_chronology_falls_back_to_session_stem_start(tmp_path):
    root, pose_root, csvp, tar = _setup(tmp_path, csv_start=False)
    rep = _run(root, pose_root, csvp, None, tmp_path / "nostart.json", extra=["--lags", "0"])
    for v in rep["ridge"].values():
        assert v["cronologia"].startswith("ordine lessicografico") and v["n_gap"] >= 1
        assert tuple(v["bordo_split"]["ultima_fit"]) < tuple(v["bordo_split"]["prima_test"])  # ordine dichiarato (sessione, stem, inizio)
