"""Ingest di CSL-hdemg (passo 2, piano_operativo_v10.md).

5 soggetti, 5 sessioni/soggetto (giorni diversi), 27 gesti (gest0=idle, 30 ripetizioni;
gest1-26, 10 ripetizioni ciascuno), 2048 Hz, 3s/trial. Fonte: Amma et al., "Advancing
Muscle-Computer Interfaces with High-Density Electromyography", CHI 2015.

Specifiche verificate dal README.txt e src/example.py UFFICIALI del dataset (non
riassunti, letti alla lettera via leonardo-ops, 24/09/2026): 192 canali fisici in
catena bipolare (canale i = differenza fra elettrodo i ed elettrodo i+1), un canale
ogni 8 (indici 0-based 7,15,...,191) e' puramente differenziale/senza dati
significativi e va scartato -> 168 canali reali, griglia 24 colonne x 7 righe
(reshape(168,)->(24,7) nell'esempio ufficiale, C-order: flat_idx = col*7 + row dopo
il filtro, cioe' canale grezzo = col*8 + row prima del filtro).

Il dataset resta zippato (12GB): legge direttamente dai membri dello zip via
scipy.io.loadmat su BytesIO, senza mai estrarre tutto su disco (raddoppierebbe lo
spazio senza bisogno).
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

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

NATIVE_FS_HZ = 2048.0
N_CHANNELS_RAW = 192
GRID_COLS = 24
GRID_ROWS_RAW = 8  # per colonna, prima del filtro
GRID_ROWS_EMG = 7  # per colonna, dopo aver scartato il canale differenziale
N_CHANNELS_EMG = GRID_COLS * GRID_ROWS_EMG  # 168
N_GESTURES = 27  # gest0 (idle) .. gest26
IDLE_GESTURE_ID = 0

# indici 0-based dei canali da scartare: uno ogni 8, a partire dal settimo
# (src/example.py ufficiale: np.delete(trial, np.s_[7:192:8], 0))
_DIFFERENTIAL_INDICES = np.arange(7, N_CHANNELS_RAW, 8)

_MEMBER_RE = re.compile(r"^(?:.*/)?subject(\d+)/session(\d+)/gest(\d+)\.mat$")


@dataclass(frozen=True)
class GestFile:
    member_name: str
    subject: int
    session: int
    gesture: int


def scan_csl_hdemg(zip_path) -> list[GestFile]:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    files = []
    for name in names:
        m = _MEMBER_RE.match(name)
        if m is None:
            continue
        subject, session, gesture = (int(g) for g in m.groups())
        files.append(GestFile(member_name=name, subject=subject, session=session, gesture=gesture))
    return files


def load_gest_file(zip_path, member_name: str) -> list[np.ndarray]:
    """Ritorna una lista di trial grezzi (192, L) - uno per ripetizione, PRIMA del
    filtro dei canali differenziali (fatto separatamente, cosi' resta esplicito e
    testabile in isolamento)."""
    with zipfile.ZipFile(zip_path) as z:
        raw_bytes = z.read(member_name)
    mat = sio.loadmat(io.BytesIO(raw_bytes))
    gestures = mat["gestures"]  # cell array MATLAB, shape (n_reps, 1) object array
    n_reps = gestures.shape[0]
    trials = []
    for i in range(n_reps):
        trial = gestures[i, 0]
        if trial.shape[0] != N_CHANNELS_RAW:
            raise ValueError(f"{member_name} rep {i}: {trial.shape[0]} canali, attesi {N_CHANNELS_RAW}")
        trials.append(trial)
    return trials


def filter_differential_channels(trial: np.ndarray) -> np.ndarray:
    """(192, L) -> (168, L): scarta i canali puramente differenziali (README
    ufficiale). Verificato identico a `np.delete(trial, np.s_[7:192:8], 0)` di
    src/example.py."""
    return np.delete(trial, _DIFFERENTIAL_INDICES, axis=0)


def qc_channel_validity(data: np.ndarray, *, min_std: float = 1e-6, max_abs_frac_clipped: float = 0.01) -> np.ndarray:
    """data: (C, T) - a differenza degli altri moduli ingest, qui i canali sono il
    primo asse (convenzione nativa CSL-hdemg), quindi gli assi di riduzione sono
    invertiti rispetto a capgmyo/grabmyo/putemg."""
    std = data.std(axis=1)
    not_flat = std >= min_std
    abs_data = np.abs(data)
    per_channel_max = abs_data.max(axis=1, keepdims=True)
    near_max = abs_data >= 0.999 * np.where(per_channel_max > 0, per_channel_max, 1.0)
    clipped_frac = near_max.mean(axis=1)
    not_clipped = clipped_frac <= max_abs_frac_clipped
    return not_flat & not_clipped


def to_int16(data: np.ndarray, *, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Come capgmyo/putemg: scala dal MASSIMO assoluto, mai da un percentile."""
    if scale is None:
        peak = np.max(np.abs(data))
        scale = (32000.0 / peak) if peak > 0 else 1.0
    quantized = np.clip(np.round(data * scale), -32768, 32767).astype(np.int16)
    return quantized, scale


def build_montage_metadata(subject: int, session: int) -> MontageMetadata:
    """Griglia 2D unica da 168 canali (v10 §3.4): non a fasce/anelli come
    GRABMyo/putEMG, un'unica griglia HD 24x7."""
    channels = []
    for idx in range(N_CHANNELS_EMG):
        col, row = divmod(idx, GRID_ROWS_EMG)
        channels.append(ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=idx, grid_row=row, grid_col=col),
            electrode_type="bipolar_chain",
            native_fs_hz=NATIVE_FS_HZ,
            effective_band_hz=(0.0, NATIVE_FS_HZ / 2),  # da verificare: banda reale non documentata qui
            mains_frequency_hz=50,  # da verificare: Germania -> 50Hz atteso, non confermato in sessione
            raw_or_envelope=RawOrEnvelope.RAW,
            anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN),
            body_region="forearm",
        ))
    group = ChannelGroup(
        group_id="grid", topology=Topology.GRID_2D, symmetry="translational_2d", channels=channels,
    )
    return MontageMetadata(
        dataset_name="csl_hdemg", subject_id=f"csl_s{subject:02d}",
        session_id=f"session{session}", day_index=session, groups=[group],
    )
