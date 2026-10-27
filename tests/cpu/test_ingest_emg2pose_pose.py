"""Ingest della posa di emg2pose (seconda ondata del pilot, `docs/pilot_posa_v1.md` §2 e §6): lettura dei soli angoli, righe a zero e margine,
blocchi a 200 Hz allineati per indice, profilo dei bordi e minimi simultanei, selezione del collaudo, statistiche per lato e insieme, driver."""

import csv
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from wearusfm.ingest import emg2pose_pose as PP

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ingest_emg2pose_pose.py"


def _angles(n=3000, seed=0, zero_runs=(), nan_rows=(), v_dip_at=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 2000.0
    a = 0.8 * np.sin(2 * np.pi * (0.5 + np.arange(20) * 0.05)[None, :] * t[:, None] + rng.normal(size=20)[None, :]) + 1.0
    a = a.astype(np.float32)
    for s, e in zero_runs:
        a[s:e] = 0.0
    for r in nan_rows:
        a[r, 3] = np.nan
    if v_dip_at is not None:  # una «V» verso lo zero larga un frame, senza righe esattamente nulle
        half = 17
        for k in range(-half, half + 1):
            a[v_dip_at + k] *= abs(k) / half if k else 1e-3
    return a


def _hdf5_bytes(angles, *, rate=2000.0, attrs=None):
    n = angles.shape[0]
    dt = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)), ("emg", "<f4", (16,))])
    arr = np.zeros(n, dtype=dt)
    arr["time"] = np.arange(n) / rate
    arr["joint_angles"] = angles
    arr["emg"] = np.random.default_rng(1).normal(size=(n, 16)).astype(np.float32)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        g = f.create_group("emg2pose")
        g.create_dataset("timeseries", data=arr)
        g.attrs["sample_rate"] = rate
        for k, v in (attrs or {}).items():
            g.attrs[k] = v
    return buf.getvalue()


def _tar(path, recs):
    """recs: {stem: bytes}; i nomi dei membri come nel tar vero."""
    with tarfile.open(path, "w") as tf:
        for stem, data in recs.items():
            info = tarfile.TarInfo(f"emg2pose_data/{stem}.hdf5")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _csv(path, rows):
    cols = ["session", "user", "stage", "start", "end", "side", "filename", "moving_hand", "held_out_user", "held_out_stage", "split", "generalization"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


# --- validita' e blocchi ----------------------------------------------------------------------------------------------------------------

def test_read_pose_reads_only_the_angles_and_checks_the_layout():
    a = _angles(500)
    rec = PP.read_pose(io.BytesIO(_hdf5_bytes(a, attrs={"side": b"left"})))
    assert rec["angles"].shape == (500, 20) and rec["angles"].dtype == np.float32 and rec["attrs"]["side"] == "left"
    assert np.allclose(rec["angles"], a)
    with pytest.raises(ValueError, match="sample_rate"):
        PP.read_pose(io.BytesIO(_hdf5_bytes(a, rate=1000.0)))


def test_zero_rows_margin_and_nonfinite_rows():
    a = _angles(1000, zero_runs=[(100, 140), (600, 601)], nan_rows=[800])
    z = PP.zero_rows(a)
    assert z.sum() == 41 and PP.runs_of(z) == [(100, 140), (600, 601)]
    assert PP.nonfinite_rows(a).sum() == 1
    bad0 = PP.invalid_rows(a, 0)
    assert bad0.sum() == 42 and bad0[800]
    bad = PP.invalid_rows(a, 33)
    # il margine copre 33 campioni per lato di ogni tratto a zero (e la riga non finita)
    assert bad[100 - 33] and not bad[100 - 34] and bad[139 + 33] and not bad[139 + 34] and bad[600 + 33] and bad.sum() == 41 + 66 + 66 + 1
    d = PP.distance_to(z)
    assert d[100] == 0 and d[99] == 1 and d[140] == 1 and d[370] == 230 and d[0] == 100


def test_blocks_at_200hz_are_aligned_by_sample_index_and_valid_only_if_whole():
    a = _angles(1005, zero_runs=[(23, 25)])  # 100 blocchi interi, 5 campioni di resto scartati
    blocks, valid = PP.block_reduce(a, ~PP.invalid_rows(a, 0))
    assert blocks.shape == (100, 20) and blocks.dtype == np.float16 and valid.shape == (100,)
    assert not valid[2] and valid[1] and valid[3] and valid.sum() == 99
    # blocco j = campioni [10 j, 10 j + 10): media esatta sui campioni validi
    assert np.allclose(blocks[5].astype(np.float64), a[50:60].astype(np.float64).mean(axis=0), atol=2e-3)
    # il blocco non valido tiene la media dei suoi 8 campioni validi (mai usata: la maschera lo dice)
    assert np.allclose(blocks[2].astype(np.float64), a[np.r_[20:23, 25:30]].astype(np.float64).mean(axis=0), atol=2e-3)
    # nessun campione valido nel blocco -> 0
    a2 = _angles(100, zero_runs=[(10, 20)])
    b2, v2 = PP.block_reduce(a2, ~PP.invalid_rows(a2, 0))
    assert not v2[1] and float(np.abs(b2[1].astype(np.float64)).max()) == 0.0


# --- misure del collaudo ----------------------------------------------------------------------------------------------------------------

def test_edge_profile_sees_a_ramp_next_to_a_zero_block_and_dips_count_the_v_shapes():
    a = _angles(4000, zero_runs=[(1000, 1100)])
    # rampa lineare di 33 campioni verso lo zero ai due bordi (l'ipotesi dell'interpolazione dopo gli zeri)
    for k in range(1, 34):
        a[1000 - k] *= k / 33
        a[1099 + k] *= k / 33
    prof = PP.edge_profile(a, PP.zero_rows(a))
    assert prof["n_zero_rows"] == 100 and prof["n_zero_runs"] == 1 and len(prof["bins"]) == 40 and prof["bins"][0]["lag_from"] == 1
    near = [b["mean_abs_diff"] for b in prof["bins"][:3]]
    far = prof["far"]["mean"]
    assert all(v is not None and v > 3 * far for v in near)  # dentro il frame la differenza prima e' molto piu' grande che lontano
    assert prof["bins"][10]["mean_abs_diff"] < 1.5 * far  # a 100 campioni il profilo e' tornato al valore lontano
    flat = PP.edge_profile(_angles(1000), np.zeros(1000, dtype=bool))
    assert flat["bins"] == [] and flat["far"]["n"] > 0
    v = _angles(4000, zero_runs=[(1000, 1100)], v_dip_at=3000)
    dips = PP.simultaneous_dips(v, PP.zero_rows(v))
    assert dips["n_dip_runs"] == 1 and 1 <= dips["n_dip_rows"] <= 5
    none = PP.simultaneous_dips(a, PP.zero_rows(a))
    assert none["n_dip_runs"] == 0  # la rampa sta entro due frame dagli zeri: non conta come «V»
    # profilo per blocco di zeri (R5): un blocco, altezza del salto misurata a 301-400 dal bordo, conteggi per bin uguali a quelli del profilo
    assert len(prof["runs"]) == 1 and prof["runs"][0]["len"] == 100 and prof["runs"][0]["h"] is not None and prof["runs"][0]["n_h"] > 0
    assert prof["runs"][0]["n_per_bin"] == [b["n"] for b in prof["bins"]]
    assert abs(sum(prof["runs"][0]["sum_per_bin"]) - sum(b["n"] * b["mean_abs_diff"] for b in prof["bins"] if b["mean_abs_diff"] is not None)) < 1e-9
    two = _angles(6000, zero_runs=[(1000, 1100), (1300, 1350)])  # due blocchi vicini: ogni campione va al piu' vicino
    p2 = PP.edge_profile(two, PP.zero_rows(two))
    assert len(p2["runs"]) == 2 and sum(sum(r["n_per_bin"]) for r in p2["runs"]) == sum(b["n"] for b in p2["bins"])
    # fra i due blocchi (1100-1300) la meta' di sinistra va al blocco 0 e quella di destra al blocco 1: a forza bruta, bin per bin
    zr = PP.zero_rows(two)
    dist = PP.distance_to(zr)
    left = [i for i in range(1100, 1300) if (i - 1099) <= (1300 - i)]  # piu' vicini al blocco 0 (a pari distanza: il precedente)
    right = [i for i in range(1100, 1300) if (i - 1099) > (1300 - i)]
    good = lambda i: 1 <= dist[i] <= 400 and not zr[i - 1]  # la differenza prima esiste solo se anche la riga precedente non e' a zero
    n0 = sum(1 for i in list(range(600, 1000)) + left if good(i))
    n1 = sum(1 for i in right + list(range(1350, 1750)) if good(i))
    assert sum(p2["runs"][0]["n_per_bin"]) == n0 == 499 and sum(p2["runs"][1]["n_per_bin"]) == n1 == 499


def test_collaudo_picker_balances_sides_and_moving_hand():
    pk = PP.CollaudoPicker(6, both_sides=True)
    rows = [{"side": s, "moving_hand": m} for s in ("left", "right") for m in ("left", "right", "both")]
    assert all(pk.accept(r) for r in rows) and pk.full
    assert not pk.accept({"side": "left", "moving_hand": "left"})
    pk2 = PP.CollaudoPicker(4, both_sides=True)
    assert [pk2.accept({"side": "left", "moving_hand": "both"}) for _ in range(3)] == [True, False, False]  # un solo «both» per lato
    assert pk2.accept({"side": "left", "moving_hand": "left"}) and not pk2.accept({"side": "left", "moving_hand": "right"})  # lato pieno (2)
    pk3 = PP.CollaudoPicker(2, both_sides=False)
    assert pk3.accept(None) and pk3.accept(None) and not pk3.accept(None)
    pk4 = PP.CollaudoPicker(6, both_sides=False, max_per_user=2)  # quota per utente (27/10): la terza dello stesso utente e' rifiutata
    assert [pk4.accept({"user": "a"}) for _ in range(3)] == [True, True, False] and pk4.accept({"user": "b"}) and pk4.per_user == {"a": 2, "b": 1}
    pk5 = PP.CollaudoPicker(4, both_sides=False, exclude_users={"a"})  # gli utenti del primo collaudo non rientrano nel campione nuovo
    assert not pk5.accept({"user": "a"}) and pk5.accept({"user": "b"}) and pk5.total == 1


# --- una registrazione e le statistiche -----------------------------------------------------------------------------------------------------

def test_ingest_recording_writes_blocks_mask_and_meta(tmp_path):
    a = _angles(2000, zero_runs=[(200, 240)], nan_rows=[900])
    ref = PP.RecordingRef("emg2pose_data/2021-01-01-1-abc-cv-emg-pose-x@2-recording-0_left.hdf5",
                          "2021-01-01-1-abc-cv-emg-pose-x@2-recording-0_left")
    row = {"user": "u1", "side": "left", "moving_hand": "left", "split": "train"}
    s = PP.ingest_pose_recording(io.BytesIO(_hdf5_bytes(a)), ref, tmp_path, row, zero_margin_samples=10)
    d = PP.out_dir_for(tmp_path, "u1", ref.stem)
    blocks, valid, meta = np.load(d / PP.POSE_FILE), np.load(d / PP.VALID_FILE), json.loads((d / PP.META_FILE).read_text())
    assert blocks.shape == (200, 20) and valid.shape == (200,) and s["n_blocks"] == 200
    # zeri 200-239 piu' 10 di margine per lato: blocchi 19..24 non validi; la riga non finita 900 invalida il blocco 90
    assert not valid[19:25].any() and valid[18] and valid[25] and not valid[90] and valid.sum() == 200 - 6 - 1 == s["n_valid_blocks"]
    assert meta["n_zero_rows"] == 40 and meta["n_nonfinite_rows"] == 1 and meta["zero_margin_samples"] == 10 and meta["split"] == "train"
    assert meta["joint_stats_valid_blocks"]["n"] == 193 and len(meta["max_abs_per_joint"]) == 20 and "edge_profile" in meta
    assert meta["joints"][0] == "THUMB_CMC_FE" and meta["joints"][-1] == "PINKY_DIP_FE" and len(meta["joints"]) == 20


def test_aggregate_stats_uses_only_the_pretraining_split_and_applies_the_side_criterion(tmp_path):
    def rec(stem, user, side, split, shift=0.0, scale=1.0, seed=0):
        a = (_angles(1500, seed=seed) - 1.0) * scale + 1.0 + shift
        PP.ingest_pose_recording(io.BytesIO(_hdf5_bytes(a)), PP.RecordingRef(f"x/{stem}.hdf5", stem), tmp_path,
                                 {"user": user, "side": side, "moving_hand": "both", "split": split}, measure=False)
    # stesso contenuto (seme 0) a destra e a sinistra: il criterio per lato non deve scattare
    rec("r1_left", "a", "left", "train")
    rec("r2_left", "b", "left", "train")
    rec("r3_right", "a", "right", "train")
    rec("r4_right", "c", "right", "test", shift=5.0)  # fuori dal pretraining: non entra
    st = PP.aggregate_stats(tmp_path)
    assert st["n_recordings"] == 3 and st["n_recordings_per_side"] == {"left": 2, "right": 1} and st["n_recordings_other_splits_skipped"] == 1
    expected = _angles(1500)[:1500].astype(np.float64).reshape(150, 10, 20).mean(axis=1).mean(axis=0)
    assert st["all"]["n"] == 3 * 150 and len(st["all"]["mean"]) == 20 and np.allclose(st["all"]["mean"], expected, atol=2e-3)
    assert st["side_criterion"]["per_side"] is False and st["side_criterion"]["max_mean_gap_sd"] < 1e-6 and st["side_criterion"]["max_sd_ratio"] < 1.01
    assert (tmp_path / PP.STATS_FILE).exists() and json.loads((tmp_path / PP.STATS_FILE).read_text())["side_criterion"]["per_side"] is False
    # un lato spostato di una deviazione intera fa scattare il criterio
    rec("r5_right", "d", "right", "train", shift=1.0, seed=5)
    rec("r6_right", "e", "right", "train", shift=1.0, seed=6)
    st2 = PP.aggregate_stats(tmp_path)
    assert st2["side_criterion"]["per_side"] is True and st2["side_criterion"]["max_mean_gap_sd"] > 0.5


# --- driver ------------------------------------------------------------------------------------------------------------------------------

def test_driver_collaudo_then_full_ingest_with_skip_and_stats(tmp_path):
    stems = [f"2021-01-0{i}-1-h{i}-cv-emg-pose-s@2-recording-0_{side}" for i, side in enumerate(("left", "right", "left", "right", "left", "right"), 1)]
    recs = {s: _hdf5_bytes(_angles(1200, seed=i, zero_runs=[(300, 340)] if i % 2 else [])) for i, s in enumerate(stems)}
    recs["._junk"] = b"not hdf5"
    tarp = _tar(tmp_path / "pose.tar", recs)
    rows = [{"user": f"u{i % 3}", "side": s.rsplit("_", 1)[1], "moving_hand": ("left", "right", "both")[i % 3], "split": "train" if i < 5 else "test",
             "filename": s + ".hdf5", "start": 0, "end": 0.6} for i, s in enumerate(stems)]
    csvp = _csv(tmp_path / "meta.csv", rows)
    out, rep = tmp_path / "pose_root", tmp_path / "rep.json"
    env = {"PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, str(SCRIPT), "--tar", str(tarp), "--csv", str(csvp), "--out-root", str(out), "--report", str(rep),
                        "--limit", "4", "--both-sides", "--profile-max-lag", "600", "--max-per-user", "2", "--exclude-users", "u2"],
                       capture_output=True, text=True, env={**env, "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads(rep.read_text())
    assert report["n_recordings"] == 4 and report["stopped"] == "limit" and report["collaudo"]["per_side"] == {"left": 2, "right": 2}
    assert report["params"]["profile_max_lag"] == 600 and len(report["collaudo"]["edge_profile"]["bins"]) == 60  # profilo lungo a richiesta
    assert report["params"]["max_per_user"] == 2 and max(report["collaudo"]["per_user"].values()) <= 2 and len(report["collaudo"]["per_user"]) >= 2
    assert report["params"]["exclude_users"] == ["u2"] and "u2" not in report["collaudo"]["per_user"]  # utenti esclusi dal collaudo
    ep = report["collaudo"]["edge_profile"]
    assert ep["by_jump_tercile"] is None and ep["n_runs_with_h"] < PP.JUMP_MIN_RUNS  # pochi blocchi: niente terzili
    assert set(ep["by_user"]) <= set(report["collaudo"]["per_user"]) and all(len(u["bins"]) == 60 for u in ep["by_user"].values())
    assert report["collaudo"]["fields_in_first_file"]["timeseries_fields"] == ["time", "joint_angles", "emg"]
    assert report["collaudo"]["edge_profile"]["bins"][0]["n"] > 0 and report["collaudo"]["frac_zero_rows"] > 0
    assert len(list(out.rglob(PP.META_FILE))) == 4
    # la regola del margine legge il rapporto VERO (stessi nomi dei campi): qui 2 per lato e 1,7% di zeri, quindi si ferma SOLO su quelle due
    # precondizioni (lati, frazione di zeri), e il resto della lettura (profilo, perdita, «V») viene comunque calcolato
    lettura = PP.margin_from_collaudo(report)
    problemi = lettura["precondizioni"]["problemi"]
    assert len(problemi) == 2 and {p.split(" ")[0] for p in problemi} == {"lati", "frazione"}
    assert lettura["margine"]["stato"] in ("determinato", "nessuna_contaminazione", "non_determinato") and lettura["perdita"]["rows"] == 4 * 1200
    assert lettura["V"]["frac_dip_rows_over_nonzero"] is not None and lettura["proposta"].startswith("FERMARSI")
    # ingest intero con margine 33, ripresa (--skip-existing) e statistiche: le 4 del collaudo (margine 0) si RIFANNO con la regola nuova, le
    # altre 2 si fanno; nessuna saltata (saltare vale solo a regola uguale)
    r2 = subprocess.run([sys.executable, str(SCRIPT), "--tar", str(tarp), "--csv", str(csvp), "--out-root", str(out), "--report", str(rep),
                         "--skip-existing", "--zero-margin-samples", "33", "--no-measure", "--stats"], capture_output=True, text=True,
                        env={**env, "PATH": "/usr/bin:/bin"})
    assert r2.returncode == 0, r2.stdout + r2.stderr
    report2 = json.loads(rep.read_text())
    assert report2["n_recordings"] == 6 and report2["n_skipped_existing"] == 0 and report2["params"]["zero_margin_samples"] == 33
    assert all(json.loads(p.read_text())["zero_margin_samples"] == 33 for p in out.rglob(PP.META_FILE))
    assert report2["stats"]["n_recordings"] == 5 and (out / PP.STATS_FILE).exists()  # 5 di train su 6
    assert set(report2["collaudo"]["per_side_stats"]) == {"left", "right"} and report2["collaudo"]["per_side_stats"]["left"]["n_recordings"] == 3
    # una terza passata con la stessa regola salta tutte e 6
    r2b = subprocess.run([sys.executable, str(SCRIPT), "--tar", str(tarp), "--csv", str(csvp), "--out-root", str(out), "--report", str(rep),
                          "--skip-existing", "--zero-margin-samples", "33", "--no-measure"], capture_output=True, text=True,
                         env={**env, "PATH": "/usr/bin:/bin"})
    assert r2b.returncode == 0 and json.loads(rep.read_text())["n_skipped_existing"] == 6
    assert json.loads(next(out.rglob(PP.META_FILE)).read_text())["fields_in_file"]["timeseries_fields"] == ["time", "joint_angles", "emg"]
    r3 = subprocess.run([sys.executable, str(SCRIPT), "--csv", str(csvp), "--out-root", str(out), "--report", str(rep), "--stats-only"],
                        capture_output=True, text=True, env={**env, "PATH": "/usr/bin:/bin"})
    assert r3.returncode == 0 and json.loads(r3.stdout)["n_recordings"] == 5


def test_launcher_uses_the_frozen_checkout_and_writes_outside_the_repo():
    text = (REPO / "scripts" / "slurm" / "ingest_emg2pose_pose.sbatch").read_text()
    assert "lib_code_dir.sh" in text and "code_dir_setup" in text and 'cd "$WORK/wearusfm"' not in text
    assert "emg2pose_pose_ingest_report_${SLURM_JOB_ID}.json" in text and "$SCRATCH/data/processed/emg2pose_pose" in text
    assert "--time-budget-s 13200" in text and "lrd_all_serial" in text and "--cpus-per-task=2" in text


# --- regola per il margine M dal rapporto del collaudo: v1 (26/10, congelata prima del rapporto) + riserva R1-R5 (27/10, post hoc) --------

def _report(ratios, *, far=0.002, n=1000, sides=(20, 20), frac_zero=0.127, max_abs=3.3, dips_rows=0, failed=None, n_samples=600000, n_runs=30,
            terciles=None):
    bins = [{"lag_from": 1 + 10 * i, "lag_to": 10 + 10 * i, "n": n, "mean_abs_diff": far * r} for i, r in enumerate(ratios)]
    recs = [{"user": f"u{i % 4}", "n_samples": n_samples, "n_zero_rows": int(frac_zero * n_samples), "n_zero_runs": n_runs,
             "n_invalid_rows": int(frac_zero * n_samples)} for i in range(sum(sides))]
    prof = {"bins": bins, "far_mean_abs_diff": far, "far_n": 10 ** 6}
    if terciles is not None:  # {nome: lista di rapporti al proprio plateau}: bins assoluti con plateau 0,01
        prof["by_jump_tercile"] = {name: {"n_runs": 100, "h_min": 0.1, "h_max": 1.0, "bins": [{"lag_from": 1 + 10 * i, "lag_to": 10 + 10 * i,
                                                                                               "n": n, "mean_abs_diff": 0.01 * r}
                                                                                              for i, r in enumerate(rs)]}
                                   for name, rs in terciles.items()}
    return {"collaudo": {"per_side": {"left": sides[0], "right": sides[1]}, "frac_zero_rows": frac_zero, "max_abs_overall": max_abs,
                         "n_nonfinite_rows": 0, "edge_profile": prof,
                         "simultaneous_dips": {"n_dip_rows_total": dips_rows, "n_dip_runs_total": dips_rows // 3,
                                               "recordings_with_dips": 1 if dips_rows else 0}},
            "per_recording": recs, "failed": failed or {}}


# il profilo VERO del collaudo 62364572 (rapporti al valore lontano, 40 bin), arrotondato
PROF_VERO = [25.9, 25.5, 24.1, 10.2, 3.4, 3.3, 3.1, 3.0, 3.0, 2.9, 2.5, 2.46, 2.42, 2.28, 2.09, 2.1, 2.08, 2.09, 2.13, 2.07,
             1.92, 1.95, 1.93, 1.88, 1.85, 1.78, 1.80, 1.74, 1.70, 1.75, 1.71, 1.78, 1.80, 1.77, 1.82, 1.80, 1.76, 1.79, 1.84, 1.78]


def test_margin_rule_v1_is_frozen_and_mechanical(tmp_path):
    ratios = [6.0, 2.5, 1.3, 1.1, 0.95] + [1.0] * 35
    r = PP.margin_from_collaudo(_report(ratios))
    assert r["precondizioni"]["ok"] and r["margine"]["M"] == 30 and r["margine"]["stato"] == "determinato" and not r["v2"]["calcolata"]
    assert r["proposta"].startswith("M = 30 campioni (15 ms")
    assert abs(r["perdita"]["frac_rows_margine_tetto"] - 2 * 30 * 30 / 600000) < 1e-12  # tetto 2 M n_blocchi / righe
    # comanda l'ULTIMO bin fuori, anche se in mezzo il profilo rientra
    assert PP.margin_from_collaudo(_report([3.0, 1.0, 1.0, 1.0, 1.4] + [1.0] * 35))["margine"]["M"] == 50
    # profilo piatto: nessuna contaminazione, M = 0
    r3 = PP.margin_from_collaudo(_report([1.0] * 40))
    assert r3["margine"]["stato"] == "nessuna_contaminazione" and r3["margine"]["M"] == 0 and r3["perdita"]["frac_rows_margine_tetto"] == 0.0
    # un bin troppo magro non conta
    thin = _report(ratios)
    thin["collaudo"]["edge_profile"]["bins"][2]["n"] = 50
    assert PP.margin_from_collaudo(thin)["margine"]["M"] == 20 and PP.margin_from_collaudo(thin)["profilo"]["n_bins_non_contati"] == 1
    # «V» oltre lo 0,1% delle righe non nulle: M c'e', ma la proposta chiede la regola in piu'
    r5 = PP.margin_from_collaudo(_report(ratios, dips_rows=30000))
    assert r5["V"]["regola_aggiuntiva_necessaria"] and "regola in piu'" in r5["proposta"] and r5["margine"]["M"] == 30
    assert not PP.margin_from_collaudo(_report(ratios, dips_rows=2000))["V"]["regola_aggiuntiva_necessaria"]
    # precondizioni: non radianti, un lato solo, file falliti, zeri implausibili -> FERMARSI
    for bad in (dict(max_abs=12.0), dict(sides=(40, 0)), dict(failed={"x": "err"}), dict(frac_zero=0.6)):
        rb = PP.margin_from_collaudo(_report(ratios, **bad))
        assert not rb["precondizioni"]["ok"] and rb["proposta"].startswith("FERMARSI")
    # da riga di comando, sul file del rapporto (come dal Mac dopo il rsync)
    path = tmp_path / "rapporto.json"
    path.write_text(json.dumps(_report(ratios)))
    out = subprocess.run([sys.executable, "-m", "wearusfm.ingest.emg2pose_pose", str(path)], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})  # il modulo del worktree, non quello installato
    assert json.loads(out.stdout)["margine"]["M"] == 30


def test_margin_reserve_rule_r1_r5_needs_the_long_profile_and_two_sided_persistence():
    # profilo corto (400 campioni) col profilo vero: v1 non determinata; riserva solo PROVVISORIA (M3 = 200 con tolleranza 0,10), stato fermo
    r = PP.margin_from_collaudo(_report(PROF_VERO))
    v2 = r["v2"]
    assert r["v1"]["stato"] == "non_determinato" and v2["calcolata"] and v2["R2_profilo_corto"] and not v2["applicabile"]
    assert abs(v2["R1_plateau_level_ratio_to_far"] - sum(PROF_VERO[30:40]) / 10) < 1e-9 and v2["R3_M3"] == 200 and v2["M"] == 200
    assert v2["sensibilita_R3"] == {"0.25": 140, "0.2": 140, "0.15": 200, "0.1": 200, "0.05": 240}
    assert r["margine"]["stato"] == "non_determinato" and r["margine"]["M"] is None and "PROVVISORIO" in r["margine"]["motivo"]
    assert r["proposta"].startswith("FERMARSI") and "profilo lungo" in r["proposta"]
    # piatto a 2x ovunque, corto: riserva provvisoria con M = 0, stato fermo (non un M a vuoto)
    r4 = PP.margin_from_collaudo(_report([2.0] * 40))
    assert r4["margine"]["M"] is None and r4["v2"]["calcolata"] and r4["v2"]["R2_profilo_corto"] and r4["v2"]["M"] == 0
    # profilo lungo (2 s = 400 bin) che DECADE LENTAMENTE oltre il plateau restando sopra il 25% dal lontano: v1 non determinata, R2 regge -> M = M3 = 200
    slow = PROF_VERO + [1.7] * 60 + [1.6] * 100 + [1.5] * 100 + [1.4] * 100
    ok = PP.margin_from_collaudo(_report(slow))
    # se invece rientra nel 25% dal lontano entro i 2 s (qui a 1,5 s), la v1 e' determinata (M = 3.000): la riserva e' solo informativa, ci si ferma
    back = PP.margin_from_collaudo(_report(PROF_VERO + [1.7] * 60 + [1.5] * 100 + [1.3] * 100 + [1.2] * 100))
    assert back["v1"]["M"] == 3000 and back["v2"]["calcolata"] and not back["v2"]["applicabile"] and back["margine"]["M"] is None
    assert back["proposta"].startswith("FERMARSI") and "M = 3000" in back["proposta"] and "R3 = 200" in back["proposta"]
    assert ok["v2"]["applicabile"] and not ok["v2"]["R2_bins_sotto_banda_401_1000"] and ok["margine"]["stato"] == "determinato_riserva"
    assert ok["margine"]["M"] == 200 and ok["v2"]["R2_max_lag_counted"] == 4000 and ok["proposta"].startswith("M = 200 campioni (100 ms")
    assert abs(ok["perdita"]["frac_rows_margine_tetto"] - 2 * 200 * 30 / 600000) < 1e-12
    # il profilo torna a 1 subito dopo il plateau (casi B, C, D del critico: artefatto che decade): R2 cade, niente riserva; v1 decide o ci si ferma
    for tail in ([1.78] * 1 + [1.0] * 359, [1.78] * 20 + [1.0] * 340, [1.78 - 0.78 * k / 59 for k in range(60)] + [1.0] * 300):
        bad = PP.margin_from_collaudo(_report(PROF_VERO + tail))
        assert bad["v2"]["calcolata"] and not bad["v2"]["applicabile"] and bad["v2"]["R2_bins_sotto_banda_401_1000"]
        assert bad["margine"]["stato"] in ("non_determinato", "determinato")
    # caso A: torna a 1 dal bin 401: la v1 e' determinata con M = 400 (<= 400) e la riserva non scatta nemmeno
    a = PP.margin_from_collaudo(_report(PROF_VERO + [1.0] * 360))
    assert a["margine"] == {"M": 400, "stato": "determinato", "motivo": a["margine"]["motivo"]} and not a["v2"]["calcolata"]
    # rialzo oltre il plateau sopra 1,25 x: R2 cade
    up = PP.margin_from_collaudo(_report(PROF_VERO + [1.7] * 50 + [2.5] * 10 + [1.6] * 300))
    assert not up["v2"]["applicabile"] and up["v2"]["R2_bins_sopra_tetto_oltre_400"] and up["margine"]["stato"] == "non_determinato"
    # v1 «determinata» con M > 400: la contaminazione FINISCE entro il misurato, la riserva e' solo informativa -> fermarsi, proposta con la v1 e la
    # perdita (seconda review del critico: anche piatta e lunga oltre 500 ms, anche con R2 che reggerebbe)
    for prof_long, m1 in (([3.0] * 45 + [1.0] * 355, 450), ([3.0] * 100 + [1.0] * 300, 1000), (PROF_VERO[:4] + [3.0] * 96 + [1.0] * 300, 1000),
                          (PROF_VERO[:4] + [3.0] * 146 + [1.0] * 250, 1500), (PROF_VERO + [1.78] * 60 + [1.0] * 300, 1000)):
        big = PP.margin_from_collaudo(_report(prof_long))
        assert big["v2"]["calcolata"] and big["v2"]["v1_M"] == m1 and not big["v2"]["applicabile"] and big["margine"]["M"] is None
        assert big["proposta"].startswith("FERMARSI") and f"M = {m1}" in big["proposta"] and "decisione di Simone" in big["proposta"]
    # contaminazione additiva che scala col salto e FINISCE a 1.000 campioni in ogni terzile: R5 (normalizzato al proprio plateau, dentro la
    # contaminazione) non la vede (M5 = 0), la v1 si' (M = 1.000) -> fermarsi, con R5 informativo nella proposta
    terc = {name: [1 + h * (20 if i < 100 else 0) for i in range(400)] for name, h in (("low", 1), ("mid", 2), ("high", 3))}
    hid = PP.margin_from_collaudo(_report([3.0] * 100 + [1.0] * 300, terciles=terc))
    assert hid["v2"]["R5_calcolabile"] and hid["v2"]["R5"]["M5"] == 0 and hid["v1"]["M"] == 1000 and hid["margine"]["stato"] == "non_determinato"
    assert "R5 = 0" in hid["proposta"] and "R3 = " in hid["proposta"]
    # code NON monotone (terza review): rientra nel 25% e poi riesce; oscilla attorno a 1,25; ultimi bin magri. L'innesco e' «nessun bin contato rientra»
    for tail in ([1.78] * 60 + [1.0] * 299 + [1.3], [1.78] * 60 + [1.2] * 100 + [1.5] * 200, [1.78] * 60 + [1.2] * 100 + [0.6] * 200):
        nm = PP.margin_from_collaudo(_report(PROF_VERO + tail))
        assert nm["v2"]["calcolata"] and nm["v2"]["profilo_rientra"] and not nm["v2"]["applicabile"] and nm["margine"]["M"] is None
        assert nm["proposta"].startswith("FERMARSI") and "decisione di Simone" in nm["proposta"] and "R3 = " in nm["proposta"]
    osc = _report(PROF_VERO + [1.6] * 160 + [1.26 if i % 2 else 1.24 for i in range(200)])
    o = PP.margin_from_collaudo(osc)
    assert o["v2"]["profilo_rientra"] and not o["v2"]["applicabile"] and o["margine"]["M"] is None
    thin_tail = _report(PROF_VERO + [1.6] * 360)  # non rientra mai, ma gli ultimi 5 bin non sono contati: la v1 dice «determinata» (M = 3.950)
    for b in thin_tail["collaudo"]["edge_profile"]["bins"][-5:]:
        b["n"] = 10
    tt = PP.margin_from_collaudo(thin_tail)
    assert tt["v1"]["stato"] == "determinato" and tt["v1"]["M"] == 3950 and not tt["v2"]["profilo_rientra"] and tt["v2"]["applicabile"]
    assert tt["margine"]["stato"] == "determinato_riserva" and tt["margine"]["M"] == 200
    # precondizione del collaudo lungo: almeno 10 utenti (qui 4)
    few = PP.margin_from_collaudo(_report(slow), min_users=10)
    assert not few["precondizioni"]["ok"] and any(p.startswith("utenti 4 < 10") for p in few["precondizioni"]["problemi"])
    # invarianza al valore lontano: stesso rapporto e stesso M se tutto il profilo e il lontano sono scalati di k
    k = 7.3
    sc = _report(slow, far=0.002 * k)
    r_sc = PP.margin_from_collaudo(sc)
    assert r_sc["margine"]["M"] == 200 and abs(r_sc["v2"]["R1_plateau_level_ratio_to_far"] - ok["v2"]["R1_plateau_level_ratio_to_far"]) < 1e-9
    # plateau non misurabile (bin 301-400 magri): riserva non applicabile, fermi
    thin = _report(slow)
    for b in thin["collaudo"]["edge_profile"]["bins"][30:37]:
        b["n"] = 10
    th = PP.margin_from_collaudo(thin)
    assert not th["v2"]["calcolata"] and "plateau" in th["v2"]["motivo"] and th["margine"]["stato"] == "non_determinato"
    # R5: i terzili dell'altezza del salto; alto / basso > 1,25 fino a 250 campioni -> M5 = 250 > M3 = 200 -> M = 250
    low = [2.0] * 25 + [1.0] * 375
    high = [3.0] * 25 + [1.0] * 375  # rapporto 1,5 nei primi 25 bin, poi 1
    r5 = PP.margin_from_collaudo(_report(slow, terciles={"low": low, "mid": low, "high": high}))
    assert r5["v2"]["R5"]["applicabile"] and r5["v2"]["R5"]["M5"] == 250 and r5["v2"]["R3_M3"] == 200 and r5["margine"]["M"] == 250
    same = PP.margin_from_collaudo(_report(slow, terciles={"low": low, "mid": low, "high": low}))
    assert same["v2"]["R5"]["M5"] == 0 and same["margine"]["M"] == 200
    assert not ok["v2"]["R5_calcolabile"] and "R5 NON calcolabile" in ok["proposta"] and r5["v2"]["R5_calcolabile"]  # senza terzili: M = M3, detto
    # R5 da solo: profili dei terzili in unita' diverse (plateau proprio): il rapporto e' lo stesso
    r5b = PP.jump_profile_margin({"low": {"bins": _report(low, far=1.0)["collaudo"]["edge_profile"]["bins"]},
                                  "high": {"bins": _report([x * 5 for x in high], far=1.0)["collaudo"]["edge_profile"]["bins"]}})
    assert r5b["M5"] == 250 and abs(r5b["bins"][0]["high_over_low"] - 1.5) < 1e-9



def test_driver_aggregates_jump_terciles_and_per_user_profiles():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ingest_pose_driver", SCRIPT)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    metas, done = [], []
    for i in range(4):  # 4 registrazioni x 3 blocchi = 12 blocchi con altezza misurabile (>= JUMP_MIN_RUNS)
        a = _angles(8000, seed=i, zero_runs=[(1000, 1100), (3500, 3550), (6000, 6100)]) * (0.5 + 0.5 * i)  # ampiezze diverse -> h diverse
        prof = PP.edge_profile(a, PP.zero_rows(a))
        metas.append({"user": f"u{i % 2}", "side": "left", "edge_profile": prof, "simultaneous_dips": PP.simultaneous_dips(a, PP.zero_rows(a)),
                      "joint_stats_valid_blocks": PP.joint_stats(*PP.block_reduce(a, ~PP.invalid_rows(a))), "per_joint_var_valid_blocks": [0.1] * 20,
                      "n_samples": 8000})
        done.append({"user": f"u{i % 2}", "side": "left", "moving_hand": "left", "n_samples": 8000, "n_zero_rows": 250, "n_nonfinite_rows": 0,
                     "max_abs_per_joint": [1.0] * 20, "frac_zero_rows": 250 / 8000})
    out = drv._aggregate_profiles(done, metas)
    ep = out["edge_profile"]
    assert ep["n_runs_with_h"] == 12 and set(ep["by_jump_tercile"]) == {"low", "mid", "high"}
    assert sum(ep["by_jump_tercile"][k]["n_runs"] for k in ("low", "mid", "high")) == 12 and ep["by_jump_tercile"]["low"]["h_max"] <= ep["by_jump_tercile"]["high"]["h_min"]
    assert all(len(ep["by_jump_tercile"][k]["bins"]) == 40 for k in ("low", "mid", "high"))
    assert set(ep["by_user"]) == {"u0", "u1"} and ep["by_user"]["u0"]["n_runs"] == 6
    # i bin dei terzili sommati danno i bin complessivi (stessi campioni, stesse somme)
    tot_n = [sum(ep["by_jump_tercile"][k]["bins"][i]["n"] for k in ("low", "mid", "high")) for i in range(40)]
    assert tot_n == [b["n"] for b in ep["bins"]]
    assert out["per_user"] == {"u0": 2, "u1": 2}
