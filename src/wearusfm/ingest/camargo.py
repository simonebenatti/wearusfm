"""Ingest di Camargo 2021 (passo 2, piano_operativo_v10.md): biomeccanica arto inferiore,
unico dataset del corpus con vicinato metrico vuoto (v10 §2.3, §3.5, §5.4, §10.1) -
caso di controllo del percorso di identita' anatomica, mantenuto nel corpus (D7b).

22 soggetti (ID AB06-AB30 con buchi: mancano AB22, AB26, AB29; tre parti), 11 canali EMG
@ 1000 Hz (Header/tempo + 11 muscoli),
attivita': levelground, ramp, stair, treadmill. Solo il sensore `emg` serve qui.

**I .mat originali sono illeggibili da Python**: contengono tabelle MATLAB MCOS,
nessun tool Python (scipy.io.loadmat, h5py, pymatreader, mat73) le decodifica, Octave
non e' su Leonardo (verificato in sessione, 29/09/2026). I file letti da questo modulo
sono la versione CONVERTITA una tantum con MATLAB locale
(scripts/convert_camargo_mcos_tables.m): .mat v7 con due sole variabili, `colnames` e
`data` - leggibile da scipy senza problemi.

Ogni canale ha identita' anatomica a livello MUSCOLO (v10 §4.5: il test su Camargo
"richiede il livello muscolo", a livello di compartimento i canali collasserebbero e
diventerebbero scambiabili), non regione/compartimento come per gli anelli.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from wearusfm.metadata.schema import (
    ChannelGroup,
    ChannelMetadata,
    MontageMetadata,
    RawOrEnvelope,
    SensorCoordinates,
    Topology,
)
from wearusfm.metadata.taxonomy import CAMARGO_EMG_COLUMNS, identity

NATIVE_FS_HZ = 1000.0
TIME_COLUMN = "Header"  # colonna tempo in secondi, non un canale EMG
EMG_COLUMNS: tuple[str, ...] = tuple(CAMARGO_EMG_COLUMNS)  # ordine del file, fatto n. 6
N_CHANNELS = len(EMG_COLUMNS)  # 11

_SUBJECT_RE = re.compile(r"^AB(\d+)$")


@dataclass(frozen=True)
class EmgFile:
    path: Path
    subject: int
    date: str
    activity: str
    trial_name: str


def scan_camargo(root: Path) -> list[EmgFile]:
    """Struttura convertita: <root>/<partN>/<ABxx>/<data>/<activity>/emg/<trial>.mat -
    partN e' solo un artefatto dello staging, non fa parte dell'identita' del trial."""
    files = []
    for path in sorted(root.rglob("*.mat")):
        if path.parent.name != "emg":
            continue
        parts = path.relative_to(root).parts  # (partN, ABxx, date, activity, 'emg', file)
        subject_dir = next((p for p in parts if _SUBJECT_RE.match(p)), None)
        if subject_dir is None:
            continue
        idx = parts.index(subject_dir)
        if len(parts) < idx + 4:
            continue
        subject = int(_SUBJECT_RE.match(subject_dir).group(1))
        files.append(EmgFile(path=path, subject=subject, date=parts[idx + 1],
                             activity=parts[idx + 2], trial_name=path.stem))
    return files


def load_emg_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (emg (T, 11), time_s (T,)). Verifica che i nomi colonna siano quelli
    attesi (fatto n. 6, firmato D7b) prima di fidarsi dell'ordine."""
    mat = sio.loadmat(path)
    colnames = [str(c[0]) if hasattr(c, "__len__") and len(c) else str(c)
                for c in np.asarray(mat["colnames"]).flatten()]
    expected = [TIME_COLUMN, *EMG_COLUMNS]
    if colnames != expected:
        raise ValueError(f"{path}: colonne {colnames}, attese {expected}")
    data = np.asarray(mat["data"], dtype=np.float64)
    return data[:, 1:], data[:, 0]


def check_time_axis(time_s: np.ndarray, *, tolerance: float = 1e-6) -> bool:
    """La colonna Header deve avanzare a 1/fs (1 ms): verifica la frequenza dichiarata
    (README ufficiale: 1000 Hz) sui dati veri invece di fidarsi solo della documentazione."""
    if time_s.shape[0] < 2:
        return False
    return bool(np.allclose(np.diff(time_s), 1.0 / NATIVE_FS_HZ, atol=tolerance))


def qc_channel_validity(data: np.ndarray, *, min_std: float = 1e-6, max_abs_frac_clipped: float = 0.01) -> np.ndarray:
    std = data.std(axis=0)
    not_flat = std >= min_std
    abs_data = np.abs(data)
    per_channel_max = abs_data.max(axis=0, keepdims=True)
    near_max = abs_data >= 0.999 * np.where(per_channel_max > 0, per_channel_max, 1.0)
    clipped_frac = near_max.mean(axis=0)
    not_clipped = clipped_frac <= max_abs_frac_clipped
    return not_flat & not_clipped


def to_int16(data: np.ndarray, *, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Come gli altri moduli ingest: scala dal MASSIMO assoluto, mai da un percentile."""
    if scale is None:
        peak = np.max(np.abs(data))
        scale = (32000.0 / peak) if peak > 0 else 1.0
    quantized = np.clip(np.round(data * scale), -32768, 32767).astype(np.int16)
    return quantized, scale


def build_montage_metadata(subject: int, date: str) -> MontageMetadata:
    """Un gruppo SPARSE (v10 §3.4: nessuna simmetria, identita' del muscolo), un
    canale per muscolo con identita' a livello MUSCOLO dalla tassonomia firmata (D7b).
    L'obliquo esterno risulta nella regione `trunk`, non arto inferiore (D7b)."""
    channels = []
    for idx, column in enumerate(EMG_COLUMNS):
        muscle_key = CAMARGO_EMG_COLUMNS[column]
        muscle_identity = identity(muscle_key)
        channels.append(ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=idx),
            electrode_type="surface_bipolar",
            native_fs_hz=NATIVE_FS_HZ,
            effective_band_hz=(20.0, 400.0),  # README ufficiale: "bandpass filtered (20-400Hz)"
            mains_frequency_hz=60,  # da verificare: Georgia Tech, USA -> 60Hz atteso, non confermato in sessione
            raw_or_envelope=RawOrEnvelope.RAW,
            anatomical_identity=muscle_identity,
            body_region=muscle_identity.region or "",
        ))
    group = ChannelGroup(group_id="sparse", topology=Topology.SPARSE, symmetry="none", channels=channels)
    return MontageMetadata(
        dataset_name="camargo2021", subject_id=f"camargo_ab{subject:02d}",
        session_id=date, groups=[group],
    )
