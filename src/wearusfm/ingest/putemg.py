"""Ingest di putEMG (solo emg_gestures/HDF5, passo 2, piano_operativo_v10.md).

File HDF5 in formato pandas/PyTables (`pd.to_hdf`), NON HDF5 puro: le colonne
categoriali sono codificate come int8 con una tabella di lookup separata
(`/data/meta/<colonna>/meta/table`). Nessuna libreria `tables`/pytables nel venv
(verificato in sessione) - parser manuale con h5py puro, niente pip install
(CLAUDE.md: richiederebbe conferma, non necessaria qui).

24 canali EMG in 3 fasce da 8 (fatto n. 12, v10 §2.1, gia' firmato D4): a 45°, primo
elettrodo di ogni fascia sull'ulna, numerazione oraria, 5120 Hz, monopolare. La
frequenza NON e' nel file HDF5 stesso: viene dalla documentazione ufficiale gia'
verificata, non da un valore letto qui.

A differenza di CapgMyo/GRABMyo (trial discreti pre-segmentati), un file putEMG e'
una SESSIONE CONTINUA con un'etichetta di gesto per campione (colonna TRAJ_GT) - la
segmentazione in trial e' compito di chi consuma i dati (harness), non dell'ingest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

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

NATIVE_FS_HZ = 5120.0  # v10 §2.1, fatto n. 12 (docs/fatti_da_verificare.md) - non dal file
N_EMG_CHANNELS = 24
N_BANDS = 3
CHANNELS_PER_BAND = 8

# values_block_3, 29 colonne nell'ordine confermato via h5py (ispezione diretta, non
# dal metadato "kind" - quello e' un pickle Python 2 non banale da riparsare)
VALUES_BLOCK_3_COLUMNS: tuple[str, ...] = (
    *(f"EMG_{i}" for i in range(1, N_EMG_CHANNELS + 1)),
    "TRAJ_1", "subject", "TRAJ_GT_NO_FILTER", "VIDEO_STAMP", "TRAJ_GT",
)

# Codice sorgente reale, biolab-put/putemg_examples, shallow_learn.py righe 142-151
# (fatto n. 17, docs/fatti_da_verificare.md). -1 = pausa/transizione (filter_transitions
# in biolab_utilities.py), non un gesto - va gestito separatamente, non e' nella mappa.
GESTURE_MAP: dict[int, str] = {
    0: "Idle", 1: "Fist", 2: "Flexion", 3: "Extension",
    6: "Pinch index", 7: "Pinch middle", 8: "Pinch ring", 9: "Pinch small",
}
PAUSE_LABEL = -1

_FILENAME_RE = re.compile(
    r"^emg_gestures-(\d+)-(\w+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+)\.hdf5$"
)


@dataclass(frozen=True)
class RecordFile:
    path: Path
    participant: int
    trajectory: str
    timestamp: str


def scan_putemg(root: Path) -> list[RecordFile]:
    files = []
    for path in sorted(root.glob("*.hdf5")):
        m = _FILENAME_RE.match(path.name)
        if m is None:
            continue
        participant, trajectory, timestamp = m.group(1), m.group(2), m.group(3)
        files.append(RecordFile(path=path, participant=int(participant),
                                 trajectory=trajectory, timestamp=timestamp))
    return files


def _read_categories(f: h5py.File, block_name: str) -> list[str]:
    meta_table = f[f"data/meta/{block_name}/meta/table"]
    return [row["values"].decode("utf-8") if isinstance(row["values"], bytes) else str(row["values"])
            for row in meta_table[:]]


@dataclass
class LoadedRecord:
    emg: np.ndarray  # (T, 24) float64
    traj_gt: np.ndarray  # (T,) int64
    subject: int
    experiment_type: str
    trajectory_type: str


def load_record(path: Path) -> LoadedRecord:
    with h5py.File(path, "r") as f:
        table = f["data/table"]
        raw = table[:]
        values = raw["values_block_3"]  # (T, 29)
        col = {name: i for i, name in enumerate(VALUES_BLOCK_3_COLUMNS)}

        emg = values[:, [col[f"EMG_{i}"] for i in range(1, N_EMG_CHANNELS + 1)]]
        traj_gt = values[:, col["TRAJ_GT"]].astype(np.int64)
        subjects = values[:, col["subject"]]
        subject = int(subjects[0])
        if not np.all(subjects == subjects[0]):
            raise ValueError(f"{path}: colonna 'subject' non costante nel file (atteso un solo soggetto/sessione)")

        exp_categories = _read_categories(f, "values_block_0")
        traj_categories = _read_categories(f, "values_block_1")
        exp_codes = raw["values_block_0"][:, 0]
        traj_codes = raw["values_block_1"][:, 0]
        if not np.all(exp_codes == exp_codes[0]) or not np.all(traj_codes == traj_codes[0]):
            raise ValueError(f"{path}: experiment_type/trajectory_type non costanti nel file")
        experiment_type = exp_categories[exp_codes[0]]
        trajectory_type = traj_categories[traj_codes[0]]

    return LoadedRecord(emg=emg, traj_gt=traj_gt, subject=subject,
                         experiment_type=experiment_type, trajectory_type=trajectory_type)


def qc_channel_validity(data: np.ndarray, *, min_std: float = 1e-3, max_abs_frac_clipped: float = 0.01) -> np.ndarray:
    std = data.std(axis=0)
    not_flat = std >= min_std
    abs_data = np.abs(data)
    per_channel_max = abs_data.max(axis=0, keepdims=True)
    near_max = abs_data >= 0.999 * np.where(per_channel_max > 0, per_channel_max, 1.0)
    clipped_frac = near_max.mean(axis=0)
    not_clipped = clipped_frac <= max_abs_frac_clipped
    return not_flat & not_clipped


def to_int16(data: np.ndarray, *, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Come capgmyo.to_int16: scala dal MASSIMO assoluto, mai dal percentile (i picchi
    di contrazione non vanno tagliati per guadagnare risoluzione sul resto)."""
    if scale is None:
        peak = np.max(np.abs(data))
        scale = (32000.0 / peak) if peak > 0 else 1.0
    quantized = np.clip(np.round(data * scale), -32768, 32767).astype(np.int16)
    return quantized, scale


def build_montage_metadata(participant: int, trajectory: str, timestamp: str) -> MontageMetadata:
    """3 fasce da 8 elettrodi (v10 §2.1, fatto n. 12): topologia ad anello per
    fascia (v10 §3.4). Ordine EMG_N -> fascia NON verificato da fonte ufficiale
    in questa sessione (fatto n. 18, docs/fatti_da_verificare.md)."""
    groups = []
    for band in range(N_BANDS):
        channels = [
            ChannelMetadata(
                sensor_coords=SensorCoordinates(
                    channel_index=i, ring_angle_deg=360.0 * i / CHANNELS_PER_BAND,
                ),
                electrode_type="dry_electrode_monopolar",
                native_fs_hz=NATIVE_FS_HZ,
                effective_band_hz=(0.0, NATIVE_FS_HZ / 2),  # da verificare: banda reale non documentata qui
                mains_frequency_hz=50,  # da verificare: paese di acquisizione (Polonia -> 50Hz atteso, non confermato in sessione)
                raw_or_envelope=RawOrEnvelope.RAW,
                anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN),
                body_region="forearm",
            )
            for i in range(CHANNELS_PER_BAND)
        ]
        groups.append(ChannelGroup(
            group_id=f"band{band + 1}", topology=Topology.RING, symmetry="D_8", channels=channels,
        ))
    return MontageMetadata(
        dataset_name="putemg", subject_id=f"putemg_p{participant:02d}",
        session_id=f"{trajectory}_{timestamp}", groups=groups,
    )
