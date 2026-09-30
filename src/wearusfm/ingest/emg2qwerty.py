"""Ingest di emg2qwerty (Meta, passo 2): due bracciali sEMG-RD (sinistro e destro) da 16 canali, 2 kHz.

Formato verificato su un file reale il 30/09/2026 (via leonardo-ops): un solo archivio
`emg2qwerty-data-2021-08.tar.gz` (308 GB) con un HDF5 per sessione, nome `YYYY-MM-DD-<ts>-keystrokes[-...].hdf5`.
Dentro: gruppo `emg2qwerty`, dataset `timeseries` (array strutturato con campi `emg_right` (16), `time` (f8),
`emg_left` (16)); attributi `user`, `condition`, `daq_channels` (16), `daq_sample_rate` (2000.0),
`duration_mins`, `session_name`, `schedule`, `quality_check_tags`, e due attributi JSON: `keystrokes` e
`prompts` (etichette del compito di digitazione, copiate tal quali nel processato: servono all'harness). Il primo
file letto ha 1,25 M campioni (10,4 min) a 2 kHz con asse dei tempi perfettamente regolare.

L'archivio e' un tar.GZ: si legge SOLO in streaming sequenziale (`r|gz`), un membro alla volta, in memoria (170-500
MB); h5py legge da un `BytesIO`. Nessun file temporaneo. La colonna dell'array di uscita e' [emg_left (16) |
emg_right (16)]: due anelli, con chiralita' vera (sinistra/destra).

Bracciale Meta sEMG-RD (fatto n. 20, raccolto e in attesa di firma): 16 canali differenziali, anello al polso,
filtro analogico 20-850 Hz. NON trovati: ordine dei canali e orientamento sul polso (identita' UNKNOWN); frequenza di
rete (60 Hz assunta, da verificare). Le unita' fisiche dei valori nei file non sono documentate.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
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
EFFECTIVE_BAND_HZ = (20.0, 850.0)  # filtro analogico del dispositivo (fatto n. 20)
_NAME_RE = re.compile(r"(?:^|/)(\d{4}-\d{2}-\d{2}-\d+-keystrokes[^/]*)\.hdf5$")
_SAFE_RE = re.compile(r"[^0-9A-Za-z_.-]")


@dataclass(frozen=True)
class RecordingRef:
    member: str
    session_name: str


def iter_tar_recordings(tar_path: Path) -> Iterator[tuple[RecordingRef, bytes]]:
    """Le registrazioni dell'archivio .tar.gz IN ORDINE, una alla volta (streaming: il membro va letto prima di
    passare al successivo). Il consumatore non deve conservare i byte oltre l'iterazione."""
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            match = _NAME_RE.search(m.name)
            if match:
                yield RecordingRef(m.name, match.group(1)), tf.extractfile(m).read()


def _attr(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_recording(data: bytes) -> dict:
    """Ritorna {emg (T, 32) float32 [left|right], time (T,), attrs: dict}. Verifica dtype e parametri dichiarati."""
    with h5py.File(io.BytesIO(data), "r") as f:
        grp = f["emg2qwerty"]
        dset = grp["timeseries"]
        names = set(dset.dtype.names or ())
        if names != {"emg_left", "emg_right", "time"}:
            raise ValueError(f"campi di `timeseries` inattesi: {sorted(names)}")
        attrs = {k: _attr(v) for k, v in grp.attrs.items()}
        attrs.update({k: _attr(v) for k, v in dset.attrs.items()})
        arr = dset[:]
    if float(attrs.get("daq_sample_rate", NATIVE_FS_HZ)) != NATIVE_FS_HZ:
        raise ValueError(f"daq_sample_rate {attrs.get('daq_sample_rate')} diversa da {NATIVE_FS_HZ}")
    if int(attrs.get("daq_channels", N_CHANNELS)) != N_CHANNELS:
        raise ValueError(f"daq_channels {attrs.get('daq_channels')} diversi da {N_CHANNELS}")
    left, right = np.asarray(arr["emg_left"]), np.asarray(arr["emg_right"])
    if left.shape[1] != N_CHANNELS or right.shape[1] != N_CHANNELS:
        raise ValueError(f"forma emg inattesa: {left.shape}, {right.shape}")
    return {"emg": np.concatenate([left, right], axis=1), "time": np.asarray(arr["time"], dtype=np.float64), "attrs": attrs}


def build_montage_metadata(user: str, session_name: str) -> MontageMetadata:
    """Due anelli da 16 (D_16), sinistro e destro, in ordine di colonna."""
    groups = []
    for gi, (gid, chir) in enumerate((("left_wrist", Chirality.LEFT), ("right_wrist", Chirality.RIGHT))):
        channels = [
            ChannelMetadata(
                sensor_coords=SensorCoordinates(channel_index=gi * N_CHANNELS + i, ring_angle_deg=360.0 * i / N_CHANNELS),
                electrode_type="sEMG-RD_bipolar", native_fs_hz=NATIVE_FS_HZ, effective_band_hz=EFFECTIVE_BAND_HZ,
                mains_frequency_hz=60,  # da verificare: paese di acquisizione non documentato nelle fonti lette
                raw_or_envelope=RawOrEnvelope.RAW,
                anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN),
                chirality=chir, body_region="wrist",
            )
            for i in range(N_CHANNELS)
        ]
        groups.append(ChannelGroup(group_id=gid, topology=Topology.RING, symmetry="D_16", channels=channels))
    return MontageMetadata(
        dataset_name="emg2qwerty", subject_id=f"emg2qwerty_u{_SAFE_RE.sub('_', user)}", session_id=session_name, groups=groups,
    )


def ingest_recording(ref: RecordingRef, data: bytes, out_root: Path) -> dict:
    """Una sessione -> `<out_root>/u<utente>/<sessione>/{data_int16.npy, metadata.json, keystrokes.json, prompts.json}`.
    Scala int16 dal MASSIMO assoluto (mai percentili); QC di canale sulla sessione intera."""
    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    rec = read_recording(data)
    emg, attrs = rec["emg"].astype(np.float64), rec["attrs"]
    user = str(attrs.get("user", "sconosciuto"))
    tax = time_axis_report(rec["time"])
    min_std = max(1e-12, 1e-3 * float(np.median(emg.std(axis=0))))
    valid = qc_channel_validity(emg, min_std=min_std)
    quantized, scale = to_int16(emg)

    out_dir = out_root / f"u{_SAFE_RE.sub('_', user)}" / ref.session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_int16.npy", quantized)
    for key in ("keystrokes", "prompts"):
        if key in attrs:
            (out_dir / f"{key}.json").write_text(attrs[key] if isinstance(attrs[key], str) else json.dumps(attrs[key]))
    montage = build_montage_metadata(user, ref.session_name)
    montage_dict = montage_to_dict(montage, valid)
    validate_montage_dict(montage_dict)
    small_attrs = {k: v for k, v in attrs.items() if k not in ("keystrokes", "prompts")}
    sidecar = {
        "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape), "native_fs_hz": NATIVE_FS_HZ,
        "columns": "emg_left (0-15) | emg_right (16-31)", "time_axis": tax, "source_member": ref.member,
        "attrs": small_attrs, "n_channels_discarded_by_qc": int((~valid).sum()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2, default=str))
    return {
        "user": user, "session": ref.session_name, "n_samples": int(quantized.shape[0]),
        "hours": quantized.shape[0] / NATIVE_FS_HZ / 3600, "discarded_channels": [int(i) for i in np.flatnonzero(~valid)],
        "time_axis_regular": tax["regular"], "n_gaps": tax.get("n_gaps"), "emg_max_abs": float(np.abs(emg).max()),
        "int16_scale": scale, "condition": attrs.get("condition"),
    }
