"""Lettura di un file HDF5 di emg2pose (Meta) come SessionData, per il riferimento di V1-V4.

Formato verificato su un file reale il 29/09/2026 (via leonardo-ops): chiave `emg2pose`,
dataset `timeseries` con dtype [('time','<f8'), ('joint_angles','<f4',(20,)), ('emg','<f4',(16,))],
attributi `sample_rate` (2000.0) e `num_channels` (16). Ogni file e' una registrazione (una
mano): la sessione e' il file; il soggetto viene dal CSV di metadati (colonna `user`, con la
colonna `filename`). Il QC di canale non si applica al riferimento: i 16 canali sono tutti validi.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np

from wearusfm.tokenizer_checks.sessions import SessionData

N_CHANNELS = 16


def load_user_map(csv_path: Path) -> dict[str, str]:
    """stem del file -> utente, dal CSV di metadati (colonne `filename` e `user`)."""
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out[Path(row["filename"]).stem] = row["user"]
    return out


def load_emg2pose_session(path: Path, user_map: dict[str, str]) -> SessionData:
    with h5py.File(path, "r") as f:
        grp = f["emg2pose"]
        fs = float(grp.attrs["sample_rate"])
        emg = np.asarray(grp["timeseries"][:]["emg"], dtype=np.float64)
    if emg.ndim != 2 or emg.shape[1] != N_CHANNELS:
        raise ValueError(f"{path}: forma emg inattesa {emg.shape}")
    subject = user_map.get(path.stem, "sconosciuto")
    return SessionData("emg2pose", subject, path.stem, fs, [emg], np.ones(N_CHANNELS, dtype=bool))
