"""Ingest di NinaPro DB6 (passo 2): 10 soggetti intatti, 14 Delsys Trigno in 16 colonne, 2 kHz,
5 giorni consecutivi x 2 sessioni al giorno (fatto n. 21).

Formato verificato su file reali il 30/09/2026 (DB6_s1_a.zip e DB6_s1_b.zip, via leonardo-ops): DUE zip
per soggetto, `DB6_s<N>_a.zip` (giorni 1-3, sei file) e `DB6_s<N>_b.zip` (giorni 4-5, quattro file), con
cartella `DB6_s<N>_a/` e talvolta `.DS_Store`. File `S<N>_D<d>_T<t>.mat`: d = giorno 1-5, t = 1 mattina /
2 pomeriggio (fatto n. 21). Variabili: `emg` (T, 16) single in volt, `acc` (T, 48), `stimulus`,
`restimulus`, `repetition`, `rerepetition`, `repetition_object`, `object`, `reobject` (T, 1) int8, e
scalari `subj`, `daytesting` (= d), `time` (= t).

**Le colonne 9 e 10 (1-indicizzate; indici 8 e 9) valgono zero in tutti i file**: coerente con i 14 elettrodi
dichiarati in 16 colonne. Il QC le scarta (deviazione standard nulla). Il montaggio assume, in ORDINE di
colonna: indici 0-7 = anello di 8 al livello radio-omerale ("8 electrodes at the radio-humeral joint
level, 6 positioned below", fatto n. 21), indici 8-9 = colonne vuote, indici 10-15 = i 6 elettrodi piu'
distali. **L'assegnazione delle colonne e' un'assunzione**, dedotta dall'ordine del testo e dalle colonne a
zero: da verificare (fatto n. 21). Anatomia dei 6 elettrodi distali ignota (identita' UNKNOWN).

Ogni file (giorno, ora) e' UNA SESSIONE: `<out_root>/s<NN>/D<d>_T<t>/`, con `day_index = d`
nel montaggio, perche' DB6 esiste per lo studio della stabilita' fra giorni. Etichette in `labels.npz`; una
etichetta piu' corta di `emg` si riempie con -1 e una piu' lunga si taglia (stessa regola di
`ninapro_std`, MAX 1%): DB6 non ha mostrato l'anomalia sui file letti, ma la regola e' uguale per tutti i DB.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

from wearusfm.ingest.common import reconcile_labels, relative_min_std
from wearusfm.metadata.schema import (
    AnatomicalIdentity,
    AnatomicalPrecision,
    ChannelGroup,
    ChannelMetadata,
    MontageMetadata,
    RawOrEnvelope,
    SensorCoordinates,
    Topology,
)

NATIVE_FS_HZ = 2000.0
N_COLUMNS = 16
RING_SIZE = 8
EMPTY_COLUMNS = (8, 9)  # indici 0-based, a zero nei dati reali
DISTAL_COLUMNS = tuple(range(10, 16))
DAYS = (1, 2, 3, 4, 5)
TIMES = (1, 2)
LABEL_FIELDS = ("stimulus", "restimulus", "repetition", "rerepetition", "repetition_object", "object", "reobject")
# solo queste variabili: `acc` (48 colonne) non serve e allunga di molto il caricamento di file da 200 MB
LOAD_VARS = ["emg", *LABEL_FIELDS, "subj", "daytesting", "time"]
_ZIP_RE = re.compile(r"^DB6_s(\d+)_([ab])\.zip$")
_MEMBER_RE = re.compile(r"(?:^|/)S(\d+)_D(\d+)_T(\d+)\.mat$")


@dataclass
class SessionFile:
    day: int
    time: int
    emg: np.ndarray  # (T, 16) float64
    labels: dict[str, np.ndarray]
    day_field_in_file: int | None
    time_field_in_file: int | None
    subject_field_in_file: int | None
    label_length_adjustments: dict[str, int] = field(default_factory=dict)


def scan_db6(raw_root: Path) -> dict[int, list[Path]]:
    """{soggetto: [zip _a, zip _b]} per `<raw_root>/DB6/DB6_s<N>_<a|b>.zip`."""
    out: dict[int, list[Path]] = {}
    for p in sorted((Path(raw_root) / "DB6").glob("*.zip")):
        m = _ZIP_RE.match(p.name)
        if m:
            out.setdefault(int(m.group(1)), []).append(p)
    return out


def _scalar(mat: dict, key: str):
    return None if key not in mat else np.asarray(mat[key]).ravel()[0]


def parse_session(mat: dict, day: int, time: int) -> SessionFile:
    emg = np.asarray(mat["emg"], dtype=np.float64)
    if emg.ndim != 2 or emg.shape[1] != N_COLUMNS:
        raise ValueError(f"forma emg inattesa {emg.shape}, attese {N_COLUMNS} colonne")
    labels, adjustments = reconcile_labels(mat, emg.shape[0], LABEL_FIELDS)
    f = lambda k: None if _scalar(mat, k) is None else int(_scalar(mat, k))  # noqa: E731
    return SessionFile(day, time, emg, labels, f("daytesting"), f("time"), f("subj"), adjustments)


def load_subject_sessions(zip_paths: list[Path], subject: int) -> list[SessionFile]:
    """Tutte le sessioni (giorno, ora) di un soggetto dai suoi zip _a e _b, in ordine (giorno, ora)."""
    found: dict[tuple[int, int], SessionFile] = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                m = _MEMBER_RE.search(name)
                if not m or "__MACOSX" in name:
                    continue
                if int(m.group(1)) != subject:
                    raise ValueError(f"{name}: soggetto {m.group(1)} in {zp.name}, atteso {subject}")
                key = (int(m.group(2)), int(m.group(3)))
                if key in found:
                    raise ValueError(f"sessione {key} duplicata in {zp.name}")
                found[key] = parse_session(sio.loadmat(io.BytesIO(zf.read(name)), variable_names=LOAD_VARS), *key)
    expected = {(d, t) for d in DAYS for t in TIMES}
    if set(found) != expected:
        raise ValueError(f"soggetto {subject}: sessioni {sorted(found)}, attese {sorted(expected)}")
    return [found[k] for k in sorted(found)]


def build_montage_metadata(subject: int, day: int, time: int) -> MontageMetadata:
    """Tre gruppi in ordine di colonna: anello da 8 (D_8), 2 colonne vuote, 6 elettrodi distali (sparso).
    Vedi il docstring del modulo: assegnazione delle colonne = assunzione da verificare."""
    common = dict(
        native_fs_hz=NATIVE_FS_HZ, effective_band_hz=(0.0, NATIVE_FS_HZ / 2),  # da verificare
        mains_frequency_hz=50,  # da verificare: paese di acquisizione non confermato (fatto n. 21)
        raw_or_envelope=RawOrEnvelope.RAW,
    )
    unknown = AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN)

    def ch(idx, etype, angle=None, region="forearm"):
        return ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=idx, ring_angle_deg=angle),
            electrode_type=etype, anatomical_identity=unknown, body_region=region, **common,
        )

    ring = [ch(i, "Delsys_Trigno_double_differential", 360.0 * i / RING_SIZE) for i in range(RING_SIZE)]
    empty = [ch(i, "empty_column_no_electrode") for i in EMPTY_COLUMNS]
    distal = [ch(i, "Delsys_Trigno_double_differential") for i in DISTAL_COLUMNS]
    groups = [
        ChannelGroup(group_id="ring8_radiohumeral", topology=Topology.RING, symmetry="D_8", channels=ring),
        ChannelGroup(group_id="empty_columns", topology=Topology.SPARSE, symmetry="none", channels=empty),
        ChannelGroup(group_id="distal6", topology=Topology.SPARSE, symmetry="none", channels=distal),
    ]
    return MontageMetadata(
        dataset_name="ninapro_db6", subject_id=f"ninapro_db6_s{subject:02d}",
        session_id=f"D{day}_T{time}", groups=groups, day_index=day,
    )


def ingest_subject(zip_paths: list[Path], out_root: Path, subject: int) -> dict:
    """Un soggetto -> 10 sessioni `<out_root>/s<NN>/D<d>_T<t>/{data_int16.npy, labels.npz, metadata.json}`."""
    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    sessions = load_subject_sessions(zip_paths, subject)
    per_session = []
    for s in sessions:
        # soglia di canale piatto relativa (vedi ninapro_std): le colonne vuote hanno deviazione standard 0
        min_std = relative_min_std(s.emg)
        valid = qc_channel_validity(s.emg, min_std=min_std)
        quantized, scale = to_int16(s.emg)
        out_dir = out_root / f"s{subject:02d}" / f"D{s.day}_T{s.time}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "data_int16.npy", quantized)
        np.savez_compressed(out_dir / "labels.npz", **s.labels)
        montage = build_montage_metadata(subject, s.day, s.time)
        montage_dict = montage_to_dict(montage, valid)
        validate_montage_dict(montage_dict)
        sidecar = {
            "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape),
            "native_fs_hz": NATIVE_FS_HZ, "label_fields": sorted(s.labels),
            "n_channels_discarded_by_qc": int((~valid).sum()), "day": s.day, "time_of_day": s.time,
            "day_field_in_file": s.day_field_in_file, "time_field_in_file": s.time_field_in_file,
            "subject_field_in_file": s.subject_field_in_file,
            "label_length_adjustments": s.label_length_adjustments,
        }
        (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))
        per_session.append({
            "day": s.day, "time": s.time, "n_samples": int(quantized.shape[0]),
            "discarded_channels": [int(i) for i in np.flatnonzero(~valid)],
            "fields_match_filename": (s.day_field_in_file, s.time_field_in_file, s.subject_field_in_file) == (s.day, s.time, subject),
            "label_length_adjustments": s.label_length_adjustments, "emg_max_abs": float(np.abs(s.emg).max()),
        })
    return {
        "subject": subject, "n_sessions": len(per_session), "n_samples": sum(x["n_samples"] for x in per_session),
        "hours": sum(x["n_samples"] for x in per_session) / NATIVE_FS_HZ / 3600, "sessions": per_session,
        "discarded_channels_union": sorted({c for x in per_session for c in x["discarded_channels"]}),
        "all_fields_match_filename": all(x["fields_match_filename"] for x in per_session),
    }
