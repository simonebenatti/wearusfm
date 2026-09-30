"""Ingest dei DB NinaPro "standard" a 12 elettrodi (DB4, DB3, DB2; DB7 con la sua configurazione).

Formato verificato su file reali il 30/09/2026 (DB4, s1.zip via leonardo-ops): uno zip per
soggetto con `s<N>/S<N>_E<m>_A1.mat`, m = 1, 2, 3. Variabili: `emg` (T, 12) single, `stimulus`,
`restimulus`, `repetition`, `rerepetition` (T, 1), scalari `subject`, `exercise`, `frequency`,
`laterality`, `sensor`, piu' dati anagrafici che NON si portano nell'ingest.

**Anomalia come in DB5:** lo scalare `exercise` dentro i file e' ruotato rispetto al nome
(DB4 s1: E1 -> 2, E2 -> 1, E3 -> 3) mentre il numero di movimenti segue il nome (12, 17, 23 = 52,
"52 different movements"). L'ingest si fida del NOME e registra lo scalare nel sidecar; lo stesso
per `subject`.

Montaggio a 12 elettrodi (fatto n. 21, raccolto e in attesa di firma; v10 §2.1/§3.4: montaggio
MISTO, non sparso puro): colonne 1-8 = anello di 8 equispaziati attorno all'avambraccio all'altezza
dell'articolazione radio-omerale (gruppo ad anello, D_8, muscolo ignoto); colonne 9-12 = elettrodi
mirati (gruppo sparso): flessore delle dita, estensore delle dita, bicipite, tricipite. **L'ORDINE
delle colonne 9-12 e' desunto dall'ordine del testo ufficiale, non dichiarato esplicitamente**
(fatto n. 21): da verificare. I muscoli si mappano a FDS, EDC, BB, TB della tassonomia (D7b).
Orientamento dell'anello sul braccio ignoto; banda hardware e frequenza di rete (50 Hz) non
verificate.

Un soggetto = una sessione: gli esercizi si concatenano; etichette in `labels.npz` (solo quelle
presenti nei file); il significato di 0 nelle etichette non e' documentato (fatto n. 19).
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

from wearusfm.metadata.schema import (
    AnatomicalIdentity,
    AnatomicalPrecision,
    Chirality,
    ChannelGroup,
    ChannelMetadata,
    MontageMetadata,
    RawOrEnvelope,
    SensorCoordinates,
    Topology,
)
from wearusfm.metadata.taxonomy import identity

_MEMBER_RE = re.compile(r"(?:^|/)S(\d+)_E(\d+)_A1\.mat$")
# si caricano SOLO queste variabili: i .mat di DB2/DB3/DB7 portano anche acc, gyro, mag, glove, force
# (fino a 1 GB per file in DB7) che qui non servono
LOAD_VARS = ["emg", "stimulus", "restimulus", "repetition", "rerepetition", "subject", "exercise",
             "frequency", "laterality", "sensor"]
LABEL_FIELDS = ("stimulus", "restimulus", "repetition", "rerepetition")
# **Anomalie verificate su DB2 (30/09/2026):** in alcuni file le etichette hanno lunghezza diversa da `emg`:
# s1 E3 `restimulus` 877.072 righe contro 877.073 di emg; s12 `restimulus` e `rerepetition` 875.435 contro
# 875.707 (272 campioni = 136 ms) mentre `stimulus`/`repetition` hanno la lunghezza di emg. L'EMG e' il
# riferimento e NON si tocca: un'etichetta piu' corta si riempie in coda con -1 ("senza etichetta", mai 0,
# che potrebbe essere un'etichetta vera), una piu' lunga si taglia; l'allineamento e' "dall'inizio" (non
# si sa se sia giusto). Tolleranza: max(5 campioni, 1% della lunghezza di emg); oltre si solleva. Ogni
# correzione si registra nel sidecar (`label_length_adjustments`, con segno: negativo = riempita).
MAX_LEN_DIFF = 5
MAX_LEN_DIFF_FRACTION = 0.01
LABEL_PAD = -1
TARGETED = (("FDS", "forearm"), ("EDC", "forearm"), ("BB", "upper_arm"), ("TB", "upper_arm"))
RING_SIZE = 8


@dataclass(frozen=True)
class StdDB:
    key: str  # "db4"
    dataset_name: str  # "ninapro_db4"
    n_channels: int
    fs_hz: float
    electrode_type: str
    zip_re: str  # cattura il numero di soggetto nel nome dello zip
    exercises: tuple[int, ...]
    sensor_expected: str | None = None
    amputee_subjects: frozenset[int] = field(default_factory=frozenset)


DB4 = StdDB(
    key="db4", dataset_name="ninapro_db4", n_channels=12, fs_hz=2000.0, electrode_type="Cometa_active_wireless",
    zip_re=r"^s(\d+)\.zip$", exercises=(1, 2, 3), sensor_expected="Cometa",
)

# DB3: 11 amputati transradiali (fatto n. 21) -> anatomia NOMINALE per tutti. Nei .mat mancano
# `frequency`, `sensor` e `laterality` (solo `subject` ed `exercise`): la frequenza (2 kHz) viene dalla
# documentazione ufficiale, non dai dati; la lateralita' e' ignota. Zip `s<N>_0.zip` con cartella
# `DB3_s<N>/` (e talvolta `.DS_Store`); i movimenti per esercizio variano per soggetto (amputati).
DB3 = StdDB(
    key="db3", dataset_name="ninapro_db3", n_channels=12, fs_hz=2000.0, electrode_type="Delsys_Trigno_double_differential",
    zip_re=r"^s(\d+)_0\.zip$", exercises=(1, 2, 3), amputee_subjects=frozenset(range(1, 12)),
)

# DB2: 40 soggetti intatti, 12 Delsys Trigno (fatto n. 21), 2 kHz; 49 movimenti (17 + 23 + 9, verificato
# sui file reali). Zip `DB2_s<N>.zip` con cartella `DB2_s<N>/`. Nei .mat solo `subject` ed `exercise`
# (come DB3): frequenza dalla documentazione, lateralita' ignota. Emg in volt.
DB2 = StdDB(
    key="db2", dataset_name="ninapro_db2", n_channels=12, fs_hz=2000.0, electrode_type="Delsys_Trigno_double_differential",
    zip_re=r"^DB2_s(\d+)\.zip$", exercises=(1, 2, 3),
)

# DB7: 20 intatti + 2 amputati (Subject_21 e Subject_22: mano destra, 50% dell'avambraccio residuo; fonte
# ufficiale, fatto n. 21) -> anatomia NOMINALE per 21 e 22. Solo 2 esercizi (17 + 23 = 40 movimenti; il
# soggetto 21 non ha fatto gli ultimi due movimenti funzionali dell'esercizio 2: 38 classi). 12 Delsys Trigno
# IM in volt; zip `Subject_<N>.zip` con i .mat alla radice, fino a ~1 GB ciascuno (con acc, gyro, mag).
# L'ORDINE dei 12 canali non e' dichiarato dalla fonte: si assume quello di DB2 (8 anello + FDS, EDC, BB, TB),
# da verificare. Lo scalare `subject` nei .mat di Subject_22 vale 2 (registrato, non usato).
DB7 = StdDB(
    key="db7", dataset_name="ninapro_db7", n_channels=12, fs_hz=2000.0, electrode_type="Delsys_Trigno_IM_double_differential",
    zip_re=r"^Subject_(\d+)\.zip$", exercises=(1, 2), amputee_subjects=frozenset({21, 22}),
)


@dataclass
class ExerciseData:
    exercise: int  # dal NOME del file
    emg: np.ndarray  # (T, n_channels) float64
    labels: dict[str, np.ndarray]  # solo quelle presenti, (T,) int16
    laterality: str
    exercise_field_in_file: int | None
    subject_field_in_file: int | None
    n_movements: int | None  # valori distinti > 0 di `restimulus` (il riempimento -1 non conta), se presente
    label_length_adjustments: dict[str, int] = field(default_factory=dict)  # nome -> len(etichetta) - len(emg)


def scan_zips(raw_root: Path, cfg: StdDB) -> dict[int, Path]:
    """{soggetto: percorso dello zip} per `<raw_root>/<DBn>/<zip>` secondo `cfg.zip_re`."""
    out = {}
    rx = re.compile(cfg.zip_re)
    for p in sorted((Path(raw_root) / cfg.key.upper()).glob("*.zip")):
        m = rx.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _scalar(mat: dict, key: str):
    if key not in mat:
        return None
    return np.asarray(mat[key]).ravel()[0]


def parse_exercise(mat: dict, exercise: int, cfg: StdDB) -> ExerciseData:
    freq = _scalar(mat, "frequency")
    if freq is not None and float(freq) != cfg.fs_hz:
        raise ValueError(f"frequenza {freq} diversa da {cfg.fs_hz}")
    sensor = _scalar(mat, "sensor")
    if cfg.sensor_expected and sensor is not None and str(sensor).strip() != cfg.sensor_expected:
        raise ValueError(f"sensore {str(sensor)!r}, atteso {cfg.sensor_expected!r}")
    emg = np.asarray(mat["emg"], dtype=np.float64)
    if emg.ndim != 2 or emg.shape[1] != cfg.n_channels:
        raise ValueError(f"forma emg inattesa {emg.shape}, attese {cfg.n_channels} colonne")
    n = emg.shape[0]
    tol = max(MAX_LEN_DIFF, int(MAX_LEN_DIFF_FRACTION * n))
    labels, adjustments = {}, {}
    for name in LABEL_FIELDS:
        if name not in mat:
            continue
        arr = np.asarray(mat[name]).ravel().astype(np.int16)
        d = arr.shape[0] - n
        if abs(d) > tol:
            raise ValueError(f"lunghezze incompatibili: emg {n}, {name} {arr.shape[0]} (tolleranza {tol})")
        if d < 0:
            arr = np.concatenate([arr, np.full(-d, LABEL_PAD, dtype=np.int16)])
        elif d > 0:
            arr = arr[:n]
        if d:
            adjustments[name] = int(d)
        labels[name] = arr
    n_mov = int((np.unique(labels["restimulus"]) > 0).sum()) if "restimulus" in labels else None
    lat = _scalar(mat, "laterality")
    ex_f, su_f = _scalar(mat, "exercise"), _scalar(mat, "subject")
    return ExerciseData(
        exercise, emg, labels, str(lat).strip().lower() if lat is not None else "",
        int(ex_f) if ex_f is not None else None, int(su_f) if su_f is not None else None, n_mov,
        label_length_adjustments=adjustments,
    )


def load_subject(zip_path: Path, subject: int, cfg: StdDB) -> list[ExerciseData]:
    """Gli esercizi di un soggetto, in ordine. Solleva se ne manca uno o se la lateralita' cambia."""
    found: dict[int, ExerciseData] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = _MEMBER_RE.search(name)
            if not m or "__MACOSX" in name:
                continue
            if int(m.group(1)) != subject:
                raise ValueError(f"{name}: soggetto {m.group(1)} in {zip_path.name}, atteso {subject}")
            ex = int(m.group(2))
            found[ex] = parse_exercise(sio.loadmat(io.BytesIO(zf.read(name)), variable_names=LOAD_VARS), ex, cfg)
    if set(found) != set(cfg.exercises):
        raise ValueError(f"{zip_path}: esercizi {sorted(found)}, attesi {list(cfg.exercises)}")
    if len({d.laterality for d in found.values()}) != 1:
        raise ValueError(f"{zip_path}: lateralita' diverse fra esercizi")
    return [found[e] for e in cfg.exercises]


def build_montage_metadata(cfg: StdDB, subject: int, laterality: str) -> MontageMetadata:
    """Anello da 8 (D_8) + 4 elettrodi mirati (sparso): vedi il docstring del modulo."""
    nominal = subject in cfg.amputee_subjects
    chir = {"r": Chirality.RIGHT, "l": Chirality.LEFT}.get(laterality.lower(), Chirality.UNKNOWN)
    common = dict(
        electrode_type=cfg.electrode_type, native_fs_hz=cfg.fs_hz,
        effective_band_hz=(0.0, cfg.fs_hz / 2),  # da verificare: banda hardware non documentata qui
        mains_frequency_hz=50,  # da verificare: paese di acquisizione non confermato (fatto n. 21)
        raw_or_envelope=RawOrEnvelope.RAW, chirality=chir,
    )
    ring = [
        ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=i, ring_angle_deg=360.0 * i / RING_SIZE),
            anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN, nominal=nominal),
            body_region="forearm", **common,
        )
        for i in range(RING_SIZE)
    ]
    targeted = [
        ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=RING_SIZE + j),
            anatomical_identity=identity(key, nominal=nominal), body_region=region, **common,
        )
        for j, (key, region) in enumerate(TARGETED)
    ]
    groups = [
        ChannelGroup(group_id="ring8_radiohumeral", topology=Topology.RING, symmetry="D_8", channels=ring),
        ChannelGroup(group_id="targeted", topology=Topology.SPARSE, symmetry="none", channels=targeted),
    ]
    return MontageMetadata(dataset_name=cfg.dataset_name, subject_id=f"{cfg.dataset_name}_s{subject:02d}",
                           session_id="s1", groups=groups)


def ingest_subject(zip_path: Path, out_root: Path, subject: int, cfg: StdDB) -> dict:
    """Un soggetto -> `<out_root>/s<NN>/session1/{data_int16.npy, labels.npz, metadata.json}`.
    Scala int16 dal MASSIMO assoluto (mai percentili); QC di canale sul soggetto intero."""
    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    exercises = load_subject(zip_path, subject, cfg)
    emg = np.concatenate([e.emg for e in exercises], axis=0)
    label_names = sorted(set.intersection(*(set(e.labels) for e in exercises)))
    labels = {k: np.concatenate([e.labels[k] for e in exercises]) for k in label_names}
    trials, offset = [], 0
    for e in exercises:
        trials.append({
            "exercise": e.exercise, "offset": offset, "n_samples": int(e.emg.shape[0]),
            "exercise_field_in_file": e.exercise_field_in_file, "subject_field_in_file": e.subject_field_in_file,
            "n_movements": e.n_movements, "label_length_adjustments": e.label_length_adjustments,
        })
        offset += e.emg.shape[0]

    # soglia di canale piatto RELATIVA (1e-3 x la mediana delle deviazioni standard): i dati sono in
    # unita' diverse (DB4: conteggi ~1e3; DB2/DB3/DB7: volt ~1e-5) e la soglia assoluta di 1e-6 del
    # QC condiviso sarebbe di scala arbitraria; un canale a zero resta scartato (std = 0)
    min_std = max(1e-12, 1e-3 * float(np.median(emg.std(axis=0))))
    channel_valid = qc_channel_validity(emg, min_std=min_std)
    n_discarded = int((~channel_valid).sum())
    quantized, scale = to_int16(emg)
    out_dir = out_root / f"s{subject:02d}" / "session1"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_int16.npy", quantized)
    if labels:
        np.savez_compressed(out_dir / "labels.npz", **labels)

    montage = build_montage_metadata(cfg, subject, exercises[0].laterality)
    montage_dict = montage_to_dict(montage, channel_valid)
    validate_montage_dict(montage_dict)
    sidecar = {
        "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape),
        "native_fs_hz": cfg.fs_hz, "trials": trials, "label_fields": label_names,
        "n_channels_discarded_by_qc": n_discarded, "laterality": exercises[0].laterality,
        "nominal_anatomy": subject in cfg.amputee_subjects,
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))
    return {
        "subject": subject, "n_samples": int(quantized.shape[0]), "hours": quantized.shape[0] / cfg.fs_hz / 3600,
        "n_channels_discarded": n_discarded, "discarded_channels": [int(i) for i in np.flatnonzero(~channel_valid)],
        "movements_per_exercise": [e.n_movements for e in exercises],
        "label_length_adjustments": {str(e.exercise): e.label_length_adjustments for e in exercises if e.label_length_adjustments},
        "exercise_field_matches_filename": all(e.exercise_field_in_file == e.exercise for e in exercises),
        "subject_field_matches_filename": all(e.subject_field_in_file == subject for e in exercises),
        "int16_scale": scale, "emg_max_abs": float(np.abs(emg).max()), "memmap_path": str(out_dir / "data_int16.npy"),
    }
