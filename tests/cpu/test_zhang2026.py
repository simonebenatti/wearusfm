"""Ingest di Zhang 2026 su CSV sintetici nella forma vista sui file reali il 01/10/2026: 7 righe d'intestazione, separatore `,`, frequenze con la
virgola tra virgolette, colonne contigue dall'inizio (4000 Hz su tutte le righe, 2000 Hz sulla prima meta'), IMU piu' corte."""

import json

import numpy as np
import pytest

from wearusfm.data.processed import load_session, validate_session
from wearusfm.ingest import zhang2026 as Z

# colonne nell'ordine dei file reali: 4 EMG del Quattro, 1 IMU del polso, poi per ogni Avanti EMG + 1 IMU (ridotto: 1 colonna IMU invece di 6)
LAYOUT = [("S4", "EMG", "FDS", "2000 Hz", "blue", "70786", "mV"), ("S2", "EMG", "PTE", "2000 Hz", "yellow", "70786", "mV"),
          ("S6", "EMG", "SUP", "2000 Hz", "green", "70786", "mV"), ("S8", "EMG", "EDC", "2000 Hz", "black", "70786", "mV"),
          ("NaN", "ACC X", "Wrist", "148,1481 Hz", "NaN", "70786", "G"),
          ("S5", "EMG", "ECR", "4000 Hz", "1", "73242", "mV"), ("S5", "ACC X", "ECR", "74,0741 Hz", "1", "73242", "G"),
          ("S3", "EMG", "ECU", "4000 Hz", "2", "72970", "mV"), ("S1", "EMG", "FCR", "4000 Hz", "3", "72978", "mV"),
          ("S7", "EMG", "FCU", "4000 Hz", "4", "72908", "mV")]


def _signals(n2, seed):
    rng = np.random.default_rng(seed)
    t2, t4 = np.arange(n2) / 2000.0, np.arange(2 * n2) / 4000.0
    out = {}
    for k, (s, kind, _m, fs, *_rest) in enumerate(LAYOUT):
        if kind != "EMG":
            out[k] = rng.normal(size=n2 // 27)
        elif fs == "2000 Hz":
            out[k] = 0.01 * np.sin(2 * np.pi * (50 + 10 * k) * t2) + 1e-4 * rng.normal(size=n2)
        else:  # 4000 Hz: una componente a 120 Hz (da conservare) e una a 1500 Hz (sopra la Nyquist di 2 kHz: da togliere)
            out[k] = 0.02 * np.sin(2 * np.pi * 120 * t4) + 0.02 * np.sin(2 * np.pi * 1500 * t4) + 1e-4 * rng.normal(size=2 * n2)
    return out


def _write_seq(d, n2=4000, seed=0, random_mode=False, gap=False):
    d.mkdir(parents=True, exist_ok=True)
    sig = _signals(n2, seed)
    header = [list(r) for r in zip(*LAYOUT)]
    if random_mode:
        header[2] = ["NaN"] * len(header[2])  # nel modo random la riga del muscolo e' NaN
    lines = [",".join(f'"{c}"' if "," in c else c for c in row) for row in header]
    for i in range(2 * n2):
        cells = []
        for k in range(len(LAYOUT)):
            x = sig[k]
            cells.append("" if i >= len(x) or (gap and k == 0 and i == 10) else f"{x[i]:.7f}")
        lines.append(",".join(cells))
    (d / "sensor_data.csv").write_text("\n".join(lines) + "\n")
    (d / "label.csv").write_text("ID,sequence,timestamp_start,timestamp_stop,gesture_code\nHG_X,1,0.5,2.0,3\n")
    return sig


def _raw(root, subject="HG_A468E29", seqs=(1, 8), modes=("anatomical", "random")):
    for m in modes:
        for k, s in enumerate(seqs):
            _write_seq(root / subject / m / f"sequence_{s:02d}", seed=k, random_mode=(m == "random"))
    (root / "participants.csv").write_text("﻿ID;Gender;Age;dominant hand;Random sequence;Anatomical sequence ;\n"
                                           f"{subject};male;19;right;5;5;\nHG_M7873Q;male;19;left;5;5;\n", encoding="utf-8")
    return root


def test_header_columns_sorted_by_sensor_and_frequencies(tmp_path):
    _write_seq(tmp_path / "s")
    h = Z.read_header(tmp_path / "s" / "sensor_data.csv")
    assert len(h) == 7 and h[3][4] == "148,1481 Hz"
    cols = Z.emg_columns(h)
    assert [c["sensor"] for c in cols] == list(range(1, 9))
    assert [(c["sensor"], c["muscle"], c["raw_fs_hz"]) for c in cols][:2] == [(1, "FCR", 4000.0), (2, "PTE", 2000.0)]


def test_read_sequence_resamples_4k_channels_keeping_band_and_removing_alias(tmp_path):
    sig = _write_seq(tmp_path / "s", n2=4000)
    r = Z.read_sequence(tmp_path / "s" / "sensor_data.csv")
    assert r["emg"].shape == (4000, 8) and r["n_raw"] == {"2000": 4000, "4000": 8000} and r["trimmed"] == 0
    assert np.allclose(r["emg"][:, 1], sig[1], atol=1e-6)  # S2 (2000 Hz) passa com'e', nella colonna dell'indice 1
    s1 = r["emg"][200:-200, 0]  # S1 (4000 Hz) ricampionato: resta la sinusoide a 120 Hz, sparisce quella a 1500 Hz
    spec = np.abs(np.fft.rfft(s1)) ** 2
    f = np.fft.rfftfreq(len(s1), 1 / 2000.0)
    assert spec[np.abs(f - 120) < 3].sum() / spec.sum() > 0.97
    assert spec[np.abs(f - 500) < 3].sum() / spec.sum() < 1e-3  # 1500 Hz ripiegato a 500 Hz se non filtrato


def test_read_sequence_refuses_non_contiguous_column(tmp_path):
    _write_seq(tmp_path / "s", gap=True)
    with pytest.raises(ValueError, match="contigui"):
        Z.read_sequence(tmp_path / "s" / "sensor_data.csv")


def test_participant_lookup_and_laterality():
    hands = {"HG_A468E29": "right", "HG_M7873Q": "left"}
    assert Z.lookup_participant("HG_A468E29", hands) == ("HG_A468E29", "right")
    assert Z.lookup_participant("HG_M7873Q0", hands) == ("HG_M7873Q", "left")
    assert Z.lookup_participant("HG_Z", hands) == (None, "")
    assert Z.chirality_from_dominant("right").value == "left" and Z.chirality_from_dominant("").value == "unknown"


def test_ingest_both_modes_sessions_labels_and_montage(tmp_path):
    raw = _raw(tmp_path / "raw")
    found = Z.scan_raw(raw)
    assert [n for n, _ in found["HG_A468E29"]["anatomical"]] == [1, 8]
    hands = Z.load_dominant_hands(raw / "participants.csv")
    for mode in ("anatomical", "random"):
        r = Z.ingest_session("HG_A468E29", mode, found["HG_A468E29"][mode], tmp_path / "out", hands["HG_A468E29"], "HG_A468E29")
        assert r["n_samples"] == 8000 and r["laterality"] == "l" and r["labels_copied"] == 2 and not r["discarded_channels"]
    a = tmp_path / "out" / "HG_A468E29" / "anatomical"
    meta = json.loads((a / "metadata.json").read_text())
    assert [(t["sequence"], t["offset"], t["n_samples"]) for t in meta["trials"]] == [(1, 0, 4000), (8, 4000, 4000)]
    assert meta["labels_aligned_to_emg"] is False and (a / "labels_video_time" / "sequence_08.csv").is_file()
    assert [c["resampled_from_hz"] for c in meta["channels"]] == [4000.0, None, 4000.0, None, 4000.0, None, 4000.0, None]
    chans = meta["montage"]["groups"][0]["channels"]
    ids = [c["anatomical_identity"] for c in chans]
    assert [i["muscle"] for i in ids] == ["FCR", "PT", "ECU", "FDS", "ECRL", None, "FCU", "EDC"]
    assert ids[5]["region"] == "forearm_proximal" and ids[5]["precision"] == "region"  # supinatore: non in tassonomia
    assert chans[0]["electrode_type"] == "Delsys_Trigno_Avanti" and chans[1]["electrode_type"] == "Delsys_Trigno_Quattro"
    rnd = json.loads((tmp_path / "out" / "HG_A468E29" / "random" / "metadata.json").read_text())
    g = rnd["montage"]["groups"][0]
    assert g["topology"] == "sparse" and all(c["anatomical_identity"]["precision"] == "unknown" for c in g["channels"])
    assert all(c["sensor_coords"].get("ring_angle_deg") is None for c in g["channels"])
    s = load_session(a, "zhang2026", "HG_A468E29", "anatomical")
    assert [seg.shape for seg in s.segments] == [(4000, 8), (4000, 8)] and s.fs == 2000.0
    assert not validate_session(a)


def test_script_collaudo_and_resume(tmp_path):
    import importlib.util
    from pathlib import Path

    raw = _raw(tmp_path / "raw", seqs=(2,))
    spec = importlib.util.spec_from_file_location("ingest_zhang2026", Path(__file__).resolve().parents[2] / "scripts" / "ingest_zhang2026.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = ["--raw-root", str(raw), "--out-root", str(tmp_path / "out"), "--report", str(tmp_path / "r.json")]
    assert mod.main(args + ["--max-sessions", "1"]) == 0
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["n_sessions"] == 1 and r["stopped"] == "max-sessions"
    assert mod.main(args + ["--skip-existing"]) == 0
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["n_sessions"] == 1 and r["n_skipped_existing"] == 1 and r["stopped"] is None
