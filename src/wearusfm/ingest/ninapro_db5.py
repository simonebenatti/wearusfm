"""Ingest di NinaPro DB5 (passo 2, piano_operativo_v10.md): due bracciali Myo, 16 canali, 200 Hz.

Formato verificato su un file reale il 30/09/2026 (s1.zip, via leonardo-ops): uno zip per
soggetto (`s<N>.zip`) con `s<N>/S<N>_E<m>_A1.mat`, m = 1, 2, 3 (tre esercizi). Variabili: `emg`
(T, 16) single con valori interi nel range [-128, 127] (l'ADC del Myo e' a 8 bit), `acc`,
`glove`, `stimulus`, `restimulus`, `repetition`, `rerepetition` (T, 1) int8, e scalari
`subject`, `exercise`, `frequency` (200), `laterality` ('r'), `sensor` ('Double Myo') piu' dati
anagrafici (eta', sesso, altezza, peso, circonferenza) che NON si portano nell'ingest.

Un soggetto = una sessione: i tre esercizi si concatenano lungo il tempo, con offset per
esercizio nel sidecar; le etichette (`stimulus`, `restimulus`, `repetition`, `rerepetition`)
vanno in `labels.npz` accanto ai dati e servono all'harness, non al pretraining.

DB5 e' l'unico dataset a 200 Hz del corpus: non entra nella vista canonica del tokenizer
(sotto 1 kHz) ma solo come ancora mascherata (docs/decisioni.md, "Bivio di v10 §6.3").
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
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

NATIVE_FS_HZ = 200.0
N_CHANNELS = 16
CHANNELS_PER_MYO = 8
EXERCISES = (1, 2, 3)
SENSOR_NAME = "Double Myo"
_MEMBER_RE = re.compile(r"(?:^|/)S(\d+)_E(\d+)_A1\.mat$")
_ZIP_RE = re.compile(r"^s(\d+)\.zip$")
LABEL_FIELDS = ("stimulus", "restimulus", "repetition", "rerepetition")


@dataclass
class ExerciseData:
    exercise: int
    emg: np.ndarray  # (T, 16) float64
    labels: dict[str, np.ndarray]  # ciascuna (T,) int16
    laterality: str


def scan_db5(raw_root: Path) -> dict[int, Path]:
    """{soggetto: percorso dello zip} per `<raw_root>/DB5/s<N>.zip`."""
    out = {}
    for p in sorted((Path(raw_root) / "DB5").glob("s*.zip")):
        m = _ZIP_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _scalar_str(x) -> str:
    return str(np.asarray(x).ravel()[0]).strip()


def _scalar_num(x) -> float:
    return float(np.asarray(x).ravel()[0])


def parse_exercise(mat: dict, subject: int, exercise: int) -> ExerciseData:
    """Controlla che il contenuto sia quello dichiarato dal nome del file prima di fidarsene."""
    if int(_scalar_num(mat["subject"])) != subject:
        raise ValueError(f"soggetto nel file {_scalar_num(mat['subject'])}, atteso {subject}")
    if int(_scalar_num(mat["exercise"])) != exercise:
        raise ValueError(f"esercizio nel file {_scalar_num(mat['exercise'])}, atteso {exercise}")
    if _scalar_num(mat["frequency"]) != NATIVE_FS_HZ:
        raise ValueError(f"frequenza {mat['frequency']} diversa da {NATIVE_FS_HZ}")
    if _scalar_str(mat["sensor"]) != SENSOR_NAME:
        raise ValueError(f"sensore {_scalar_str(mat['sensor'])!r}, atteso {SENSOR_NAME!r}")
    emg = np.asarray(mat["emg"], dtype=np.float64)
    if emg.ndim != 2 or emg.shape[1] != N_CHANNELS:
        raise ValueError(f"forma emg inattesa {emg.shape}")
    labels = {}
    for name in LABEL_FIELDS:
        arr = np.asarray(mat[name]).ravel().astype(np.int16)
        if arr.shape[0] != emg.shape[0]:
            raise ValueError(f"{name}: {arr.shape[0]} righe contro {emg.shape[0]} di emg")
        labels[name] = arr
    return ExerciseData(exercise, emg, labels, _scalar_str(mat["laterality"]).lower())


def load_subject(zip_path: Path, subject: int) -> list[ExerciseData]:
    """I tre esercizi di un soggetto, in ordine E1, E2, E3. Solleva se ne manca uno o se la
    lateralita' cambia fra esercizi."""
    found: dict[int, ExerciseData] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = _MEMBER_RE.search(name)
            if not m or name.startswith("__MACOSX"):
                continue
            if int(m.group(1)) != subject:
                raise ValueError(f"{name}: soggetto {m.group(1)} in {zip_path.name}, atteso {subject}")
            ex = int(m.group(2))
            mat = sio.loadmat(io.BytesIO(zf.read(name)))
            found[ex] = parse_exercise(mat, subject, ex)
    if set(found) != set(EXERCISES):
        raise ValueError(f"{zip_path}: esercizi {sorted(found)}, attesi {list(EXERCISES)}")
    lat = {d.laterality for d in found.values()}
    if len(lat) != 1:
        raise ValueError(f"{zip_path}: lateralita' diverse fra esercizi {lat}")
    return [found[e] for e in EXERCISES]


def build_montage_metadata(subject: int, laterality: str) -> MontageMetadata:
    """Due bracciali Myo da 8 canali, ciascuno un anello (v10 §3.4: la topologia e' del gruppo).

    NON verificato da fonte ufficiale in questa sessione (docs/fatti_da_verificare.md, fatto
    n. 19): quale bracciale e' `emg[:, :8]` e quale `emg[:, 8:]`, dove stanno sull'avambraccio
    e con quale rotazione relativa; banda hardware del Myo; frequenza di rete del paese di
    acquisizione. Qui: gruppi 'myo_1' (colonne 0-7) e 'myo_2' (colonne 8-15), identita'
    anatomica UNKNOWN, orientamento della fascia ignoto (None)."""
    chir = {"r": Chirality.RIGHT, "l": Chirality.LEFT}.get(laterality.lower(), Chirality.UNKNOWN)
    groups = []
    for g in range(N_CHANNELS // CHANNELS_PER_MYO):
        channels = [
            ChannelMetadata(
                sensor_coords=SensorCoordinates(
                    channel_index=CHANNELS_PER_MYO * g + i, ring_angle_deg=360.0 * i / CHANNELS_PER_MYO,
                ),
                electrode_type="Myo_dry_8bit",
                native_fs_hz=NATIVE_FS_HZ,
                effective_band_hz=(0.0, NATIVE_FS_HZ / 2),  # da verificare: banda hardware del Myo non documentata qui
                mains_frequency_hz=50,  # da verificare: paese di acquisizione non confermato in questa sessione
                raw_or_envelope=RawOrEnvelope.RAW,
                anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN),
                chirality=chir,
                body_region="forearm",
            )
            for i in range(CHANNELS_PER_MYO)
        ]
        groups.append(ChannelGroup(group_id=f"myo_{g + 1}", topology=Topology.RING, symmetry="D_8", channels=channels))
    return MontageMetadata(
        dataset_name="ninapro_db5", subject_id=f"ninapro_db5_s{subject:02d}", session_id="s1", groups=groups,
    )


def ingest_subject(zip_path: Path, out_root: Path, subject: int) -> dict:
    """Un soggetto -> `<out_root>/s<NN>/session1/{data_int16.npy, labels.npz, metadata.json}`.
    I tre esercizi si concatenano (offset per esercizio nel sidecar); il QC di canale e' sul
    soggetto intero; la scala int16 dal MASSIMO assoluto (mai percentili)."""
    import json

    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    exercises = load_subject(zip_path, subject)
    emg = np.concatenate([e.emg for e in exercises], axis=0)
    labels = {k: np.concatenate([e.labels[k] for e in exercises]) for k in LABEL_FIELDS}
    trials, offset = [], 0
    for e in exercises:
        trials.append({"exercise": e.exercise, "offset": offset, "n_samples": int(e.emg.shape[0])})
        offset += e.emg.shape[0]

    channel_valid = qc_channel_validity(emg)
    n_discarded = int((~channel_valid).sum())
    quantized, scale = to_int16(emg)

    out_dir = out_root / f"s{subject:02d}" / "session1"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_int16.npy", quantized)
    np.savez_compressed(out_dir / "labels.npz", **labels)

    montage = build_montage_metadata(subject, exercises[0].laterality)
    montage_dict = montage_to_dict(montage, channel_valid)
    validate_montage_dict(montage_dict)
    sidecar = {
        "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape),
        "native_fs_hz": NATIVE_FS_HZ, "trials": trials, "label_fields": list(LABEL_FIELDS),
        "n_channels_discarded_by_qc": n_discarded, "laterality": exercises[0].laterality,
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))
    rest = labels["restimulus"] == 0
    return {
        "subject": subject, "n_samples": int(quantized.shape[0]), "hours": quantized.shape[0] / NATIVE_FS_HZ / 3600,
        "n_channels_discarded": n_discarded, "rest_fraction_restimulus": float(rest.mean()),
        "n_stimulus_values": int(len(np.unique(labels["restimulus"]))),
        "int16_scale": scale, "emg_max_abs": float(np.abs(emg).max()), "memmap_path": str(out_dir / "data_int16.npy"),
    }
