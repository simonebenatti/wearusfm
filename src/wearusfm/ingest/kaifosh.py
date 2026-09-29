"""Ingest di Kaifosh et al. 2025, "Discrete Gestures" (passo 2, piano_operativo_v10.md, D3a = si').

Formato verificato su un file reale il 30/09/2026 (via leonardo-ops): `full_data.tar` (33 GB) con
un HDF5 per (utente, dataset), `discrete_gestures_user_<UUU>_dataset_<DDD>.hdf5`. Dentro: `data`
(array strutturato `[('emg', '<f4', (16,)), ('time', '<f8')]`, ~5,17 M campioni, attributo
`task = discrete_gestures`), piu' i gruppi `prompts` e `stages` in formato pandas "fixed". Il CSV
`discrete_gestures_corpus.csv` assegna a ogni segmento (start, end) uno split
(train/val/test) per utente.

Qui si ingerisce SOLO l'EMG (il corpus e' per pretraining auto-supervisionato): `prompts` e
`stages` NON si leggono. Il nome dei gesti sta in un blocco di oggetti pickled di pandas, che non
si deserializza (deserializzare pickle di terzi e' un rischio, e servirebbe comunque pytables,
assente nel venv). Le etichette servono al benchmark "Discrete Gestures" dell'harness (passo 5):
si leggeranno li', dal tar.

Bracciale Meta sEMG-RD (fatto n. 20, raccolto e in attesa di firma): 16 canali differenziali in un
anello al polso, 2 kHz, filtro analogico 20-850 Hz; il repo del dataset dichiara "high pass
filtered at 40 Hz", quindi la banda effettiva qui e' 40-850 Hz. NON trovati: ordine dei 16 canali
e orientamento sul polso (identita' anatomica UNKNOWN, orientamento None); frequenza di rete
(60 Hz assunta, da verificare).

Si legge direttamente dal tar (nessun file temporaneo): `tarfile.extractfile` da' un file-like
seekable e h5py sa leggerlo.
"""

from __future__ import annotations

import csv
import json
import re
import tarfile
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

NATIVE_FS_HZ = 2000.0
N_CHANNELS = 16
EFFECTIVE_BAND_HZ = (40.0, 850.0)  # passa-alto del dataset 40 Hz, filtro analogico fino a 850 Hz (fatto n. 20)
MAX_GAPS_LISTED = 10000
_NAME_RE = re.compile(r"(?:^|/)discrete_gestures_user_(\d+)_dataset_(\d+)\.hdf5$")


@dataclass(frozen=True)
class RecordingRef:
    user: int
    dataset: int
    member: str


def scan_tar(tf: tarfile.TarFile) -> list[RecordingRef]:
    """Le registrazioni presenti nel tar, ordinate per (utente, dataset)."""
    refs = []
    for m in tf.getmembers():
        if not m.isfile():
            continue
        match = _NAME_RE.search(m.name)
        if match:
            refs.append(RecordingRef(int(match.group(1)), int(match.group(2)), m.name))
    return sorted(refs, key=lambda r: (r.user, r.dataset))


def load_split_table(csv_path: Path) -> dict[tuple[int, int], list[str]]:
    """{(utente, dataset): split distinti} dal CSV ufficiale (colonne user_number, dataset_number, split)."""
    out: dict[tuple[int, int], set[str]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault((int(row["user_number"]), int(row["dataset_number"])), set()).add(row["split"])
    return {k: sorted(v) for k, v in out.items()}


def read_recording(fileobj) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (emg (T, 16) float32, time (T,) float64). Verifica il dtype dichiarato e il task."""
    with h5py.File(fileobj, "r") as f:
        dset = f["data"]
        if set(dset.dtype.names or ()) != {"emg", "time"}:
            raise ValueError(f"campi di `data` inattesi: {dset.dtype.names}")
        task = dset.attrs.get("task")
        task = task.decode() if isinstance(task, bytes) else task
        if task != "discrete_gestures":
            raise ValueError(f"task {task!r}, atteso 'discrete_gestures'")
        arr = dset[:]
    emg = np.ascontiguousarray(arr["emg"])
    if emg.ndim != 2 or emg.shape[1] != N_CHANNELS:
        raise ValueError(f"forma emg inattesa {emg.shape}")
    return emg, np.asarray(arr["time"], dtype=np.float64)


def time_axis_report(time_s: np.ndarray, fs: float = NATIVE_FS_HZ, *, gap_factor: float = 1.5) -> dict:
    """Regolarita' dell'asse dei tempi: la frequenza dichiarata va verificata sui dati, e i buchi
    (campioni mancanti) si contano invece di nasconderli."""
    if time_s.shape[0] < 2:
        return {"n_samples": int(time_s.shape[0]), "regular": False}
    dt = np.diff(time_s)
    nominal = 1.0 / fs
    gap_idx = np.flatnonzero(dt > gap_factor * nominal)
    return {
        # posizioni dei buchi (indice del primo campione DOPO il buco, durata in s): servono al
        # dataloader per non fare finestre a cavallo di un buco (il collaudo ne ha trovati 53 in 0,76 h)
        "gaps": [{"index": int(i + 1), "dt_s": float(dt[i])} for i in gap_idx[:MAX_GAPS_LISTED]],
        "gaps_truncated": bool(len(gap_idx) > MAX_GAPS_LISTED),
        "n_samples": int(time_s.shape[0]), "duration_s": float(time_s[-1] - time_s[0]),
        "dt_median_s": float(np.median(dt)), "dt_max_s": float(dt.max()), "dt_min_s": float(dt.min()),
        "n_gaps": int((dt > gap_factor * nominal).sum()), "n_nonmonotonic": int((dt <= 0).sum()),
        "estimated_rate_hz": float(1.0 / np.median(dt)), "time_start": float(time_s[0]),
        "regular": bool(np.allclose(np.median(dt), nominal, rtol=1e-3) and (dt > 0).all()),
    }


def build_montage_metadata(user: int, dataset: int) -> MontageMetadata:
    channels = [
        ChannelMetadata(
            sensor_coords=SensorCoordinates(channel_index=i, ring_angle_deg=360.0 * i / N_CHANNELS),
            electrode_type="sEMG-RD_bipolar", native_fs_hz=NATIVE_FS_HZ, effective_band_hz=EFFECTIVE_BAND_HZ,
            mains_frequency_hz=60,  # da verificare: paese di acquisizione non documentato nelle fonti lette
            raw_or_envelope=RawOrEnvelope.RAW,
            anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN),
            body_region="wrist",
        )
        for i in range(N_CHANNELS)
    ]
    group = ChannelGroup(group_id="sEMG_RD_ring", topology=Topology.RING, symmetry="D_16", channels=channels)
    return MontageMetadata(
        dataset_name="kaifosh_discrete_gestures", subject_id=f"kaifosh_u{user:03d}",
        session_id=f"dataset{dataset:03d}", groups=[group],
    )


def ingest_recording(
    fileobj, ref: RecordingRef, out_root: Path, split_table: dict[tuple[int, int], list[str]] | None = None
) -> dict:
    """Una registrazione -> `<out_root>/u<UUU>/dataset<DDD>/{data_int16.npy, metadata.json}`.
    Scala int16 dal MASSIMO assoluto (mai percentili); QC di canale sull'intera registrazione."""
    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    emg, time_s = read_recording(fileobj)
    tax = time_axis_report(time_s)
    channel_valid = qc_channel_validity(emg.astype(np.float64))
    n_discarded = int((~channel_valid).sum())
    quantized, scale = to_int16(emg.astype(np.float64))

    out_dir = out_root / f"u{ref.user:03d}" / f"dataset{ref.dataset:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_int16.npy", quantized)
    montage = build_montage_metadata(ref.user, ref.dataset)
    montage_dict = montage_to_dict(montage, channel_valid)
    validate_montage_dict(montage_dict)
    sidecar = {
        "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape),
        "native_fs_hz": NATIVE_FS_HZ, "time_axis": tax, "source_member": ref.member,
        "official_split": (split_table or {}).get((ref.user, ref.dataset)),
        "n_channels_discarded_by_qc": n_discarded,
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))
    return {
        "user": ref.user, "dataset": ref.dataset, "n_samples": int(quantized.shape[0]),
        "hours": quantized.shape[0] / NATIVE_FS_HZ / 3600, "n_channels_discarded": n_discarded,
        "time_axis_regular": tax["regular"], "n_gaps": tax.get("n_gaps"), "emg_max_abs": float(np.abs(emg).max()),
        "int16_scale": scale, "official_split": sidecar["official_split"], "memmap_path": str(out_dir / "data_int16.npy"),
    }
