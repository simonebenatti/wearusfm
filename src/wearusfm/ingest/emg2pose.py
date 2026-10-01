"""Ingest di emg2pose (Meta, passo 2): un bracciale sEMG-RD da 16 canali per registrazione, 2 kHz.

Formato verificato su file reali il 29-30/09/2026 (mini e riferimento, via leonardo-ops): tar NON compresso
`emg2pose_dataset.tar` (462 GB, cartella radice `emg2pose_data/`) con un HDF5 per registrazione, nome
`<data>-<ts>-<hash>-cv-emg-pose-<stage>@2-recording-<n>_<left|right>.hdf5`. Dentro: gruppo `emg2pose`, dataset
`timeseries` con dtype `[('time','<f8'), ('joint_angles','<f4',(20,)), ('emg','<f4',(16,))]`, attributi
`sample_rate` (2000.0), `num_channels` (16), `stage`, `split`, `held_out_user`, `held_out_stage`, `moving_hand`,
`generalization`. Il CSV di metadati (25.253 righe, 193 utenti; colonne session, user, stage, start, end, side,
filename, moving_hand, held_out_user, held_out_stage, split, generalization) da' utente e lato; la CHIRALITA' del
bracciale viene da `side` del CSV (e dal suffisso del nome).

Qui si ingerisce SOLO l'EMG: gli angoli articolari (`joint_angles`, 20 valori) sono le etichette del compito di
stima della posa e servono all'harness, non al pretraining auto-supervisionato: NON si leggono. Lo scopo e' lo
stesso di Kaifosh/emg2qwerty. Anello da 16 (D_16), stesse assunzioni sul dispositivo (fatto n. 20).

Il tar e' non compresso: si legge in accesso casuale con `tarfile.extractfile` (file-like seekable, letto da h5py).
"""

from __future__ import annotations

import csv
import io
import json
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from wearusfm.ingest.kaifosh import time_axis_report
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

NATIVE_FS_HZ = 2000.0
N_CHANNELS = 16
EFFECTIVE_BAND_HZ = (20.0, 850.0)
_SAFE_RE = re.compile(r"[^0-9A-Za-z_.@-]")


@dataclass(frozen=True)
class RecordingRef:
    member: str
    stem: str
    # il membro del tar: si estrae con QUESTO, non col nome. `tf.extractfile(nome)` passa da getmember, che carica l'indice intero del tar (sul tar
    # da 462 GB di emg2pose: lettura di tutto il file prima della prima registrazione; collaudo 59093601, 01/10/2026)
    info: tarfile.TarInfo | None = field(default=None, compare=False, repr=False)


def load_metadata_csv(path: Path) -> dict[str, dict]:
    """{stem del file: riga del CSV} (`filename` con o senza estensione)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[Path(row["filename"]).stem] = row
    return out


def _is_recording(name: str) -> bool:
    """Un HDF5 di registrazione, esclusi i file di metadati di macOS (`._<nome>`, `__MACOSX/`) che un tar creato su Mac contiene: non sono
    HDF5 e verrebbero contati come registrazioni fallite (visto il 30/09/2026 con un tar di prova fatto sul Mac)."""
    p = Path(name)
    return p.suffix == ".hdf5" and not p.name.startswith("._") and "__MACOSX" not in p.parts


def iter_recordings(tf: tarfile.TarFile):
    """Le registrazioni nell'ordine del tar, una alla volta, leggendo l'indice SOLO fin dove serve. Per l'ingest: `scan_tar` (getmembers) legge
    l'intero indice prima di cominciare, e sul tar da 462 GB di emg2pose su Lustre vuol dire scorrere tutto il file (collaudo 59092330,
    01/10/2026: piu' di 30 minuti senza aver elaborato nulla)."""
    for m in tf:
        if m.isfile() and _is_recording(m.name):
            yield RecordingRef(m.name, Path(m.name).stem, m)


def scan_tar(tf: tarfile.TarFile) -> list[RecordingRef]:
    return sorted(iter_recordings(tf), key=lambda r: r.member)


def read_recording(fileobj) -> dict:
    """Ritorna {emg (T, 16) float32, time (T,), attrs}. NON legge `joint_angles`."""
    with h5py.File(fileobj, "r") as f:
        grp = f["emg2pose"]
        dset = grp["timeseries"]
        if not {"emg", "time"} <= set(dset.dtype.names or ()):
            raise ValueError(f"campi di `timeseries` inattesi: {dset.dtype.names}")
        attrs = {}
        for k, v in grp.attrs.items():
            attrs[k] = v.decode() if isinstance(v, bytes) else (v.item() if isinstance(v, np.generic) else v)
        if float(attrs.get("sample_rate", NATIVE_FS_HZ)) != NATIVE_FS_HZ:
            raise ValueError(f"sample_rate {attrs.get('sample_rate')} diversa da {NATIVE_FS_HZ}")
        emg = np.ascontiguousarray(dset.fields("emg")[:]) if hasattr(dset, "fields") else np.asarray(dset[:]["emg"])
        time_s = np.asarray(dset.fields("time")[:] if hasattr(dset, "fields") else dset[:]["time"], dtype=np.float64)
    if emg.ndim != 2 or emg.shape[1] != N_CHANNELS:
        raise ValueError(f"forma emg inattesa {emg.shape}")
    return {"emg": emg, "time": time_s, "attrs": attrs}


def _chirality(side: str | None, stem: str) -> Chirality:
    s = (side or "").lower()
    if not s:
        s = "left" if stem.endswith("_left") else "right" if stem.endswith("_right") else ""
    return {"left": Chirality.LEFT, "right": Chirality.RIGHT}.get(s, Chirality.UNKNOWN)


def build_montage_metadata(user: str, stem: str, side: str | None) -> MontageMetadata:
    channels = [
        ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=i, ring_angle_deg=360.0 * i / N_CHANNELS),
            electrode_type="sEMG-RD_bipolar", native_fs_hz=NATIVE_FS_HZ, effective_band_hz=EFFECTIVE_BAND_HZ,
            mains_frequency_hz=60,  # da verificare: paese di acquisizione non documentato nelle fonti lette
            raw_or_envelope=RawOrEnvelope.RAW,
            anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN),
            chirality=_chirality(side, stem), body_region="wrist",
        )
        for i in range(N_CHANNELS)
    ]
    group = ChannelGroup(group_id="sEMG_RD_ring", topology=Topology.RING, symmetry="D_16", channels=channels)
    return MontageMetadata(
        dataset_name="emg2pose", subject_id=f"emg2pose_u{_SAFE_RE.sub('_', user)}", session_id=stem, groups=[group],
    )


def ingest_recording(fileobj, ref: RecordingRef, out_root: Path, meta_row: dict | None) -> dict:
    """Una registrazione -> `<out_root>/u<utente>/<stem>/{data_int16.npy, metadata.json}`. Se il CSV non ha la riga,
    l'utente e' 'sconosciuto' e si segnala (non si inventa)."""
    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, relative_min_std, validate_montage_dict

    rec = read_recording(fileobj)
    emg = rec["emg"].astype(np.float64)
    user = (meta_row or {}).get("user", "sconosciuto")
    tax = time_axis_report(rec["time"])
    min_std = relative_min_std(emg)
    valid = qc_channel_validity(emg, min_std=min_std)
    quantized, scale = to_int16(emg)

    out_dir = out_root / f"u{_SAFE_RE.sub('_', user)}" / _SAFE_RE.sub("_", ref.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_int16.npy", quantized)
    montage = build_montage_metadata(user, ref.stem, (meta_row or {}).get("side"))
    montage_dict = montage_to_dict(montage, valid)
    validate_montage_dict(montage_dict)
    sidecar = {
        "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape), "native_fs_hz": NATIVE_FS_HZ,
        "time_axis": tax, "source_member": ref.member, "attrs": rec["attrs"], "csv_row": meta_row,
        "n_channels_discarded_by_qc": int((~valid).sum()), "joint_angles": "non ingeriti (etichette di posa, per l'harness)",
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2, default=str))
    return {
        "user": user, "stem": ref.stem, "n_samples": int(quantized.shape[0]), "hours": quantized.shape[0] / NATIVE_FS_HZ / 3600,
        "discarded_channels": [int(i) for i in np.flatnonzero(~valid)], "time_axis_regular": tax["regular"],
        "n_gaps": tax.get("n_gaps"), "in_csv": meta_row is not None, "split": (meta_row or {}).get("split"),
        "emg_max_abs": float(np.abs(emg).max()), "int16_scale": scale,
    }
