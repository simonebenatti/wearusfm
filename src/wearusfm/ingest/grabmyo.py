"""Ingest di GRABMyo (passo 2, piano_operativo_v10.md): WFDB, PhysioNet.

43 partecipanti, 3 sessioni (giorni diversi, v10 §2.1 "multi-giorno"), 17 gesti, 7
trial/gesto, 2048 Hz, 5s/trial (10240 campioni). Formato .hea/.dat WFDB standard
(int16 little-endian, multiplexato).

**32 canali dichiarati nell'header, ma solo 28 sono EMG.** Verificato dalla pagina
ufficiale PhysioNet (23/09/2026, docs/fatti_da_verificare.md #16): U1-U4 sono
esplicitamente "unused channels", posizionati solo per marcare il confine fra i 4
anelli di elettrodi (colonne 17, 24, 25, 32) - non EMG, esclusi qui. I 28 canali EMG
reali formano 4 anelli distinti (v10 §3.4: topologia e' proprieta' del GRUPPO):
ring1=F1-F8, ring2=F9-F16 (avambraccio), ring3=W1-W6, ring4=W7-W12 (polso) - non un
anello unico da 28 come la tabella semplificata di v10 §2.1 lascerebbe intendere.

Nessuna libreria wfdb nel venv (verificato in sessione): parser manuale, il formato
WFDB con Format=16 e' documentato pubblicamente e semplice (int16 LE multiplexato,
fisico = (raw - baseline) / gain). Niente pip install per questo (CLAUDE.md: le
installazioni richiedono conferma - non necessaria qui).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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

NATIVE_FS_HZ = 2048.0
N_SAMPLES = 10240
N_SESSIONS = 3
N_PARTICIPANTS = 43
N_GESTURES = 17
N_TRIALS_PER_GESTURE = 7

# v10 §3.4/§4.5: un montaggio e' un insieme di gruppi. GRABMyo ne ha 4, non 1.
RINGS: dict[str, tuple[str, ...]] = {
    "ring1_forearm": tuple(f"F{i}" for i in range(1, 9)),
    "ring2_forearm": tuple(f"F{i}" for i in range(9, 17)),
    "ring3_wrist": tuple(f"W{i}" for i in range(1, 7)),
    "ring4_wrist": tuple(f"W{i}" for i in range(7, 13)),
}
EMG_CHANNEL_NAMES: tuple[str, ...] = tuple(name for ring in RINGS.values() for name in ring)
UNUSED_CHANNEL_NAMES = ("U1", "U2", "U3", "U4")  # PhysioNet ufficiale: non EMG, esclusi

_FILENAME_RE = re.compile(
    r"^session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)\.hea$"
)
_GAIN_RE = re.compile(r"^([-\d.eE+]+)\(([-\d]+)\)/(\S+)$")


@dataclass(frozen=True)
class TrialFile:
    hea_path: Path
    session: int
    participant: int
    gesture: int
    trial: int

    @property
    def dat_path(self) -> Path:
        return self.hea_path.with_suffix(".dat")


@dataclass(frozen=True)
class ChannelHeader:
    signal_name: str
    gain: float
    baseline: int
    units: str


@dataclass(frozen=True)
class RecordHeader:
    record_name: str
    n_channels: int
    fs_hz: float
    n_samples: int
    channels: tuple[ChannelHeader, ...]


def scan_grabmyo(root: Path) -> list[TrialFile]:
    files = []
    for path in sorted(root.rglob("*.hea")):
        m = _FILENAME_RE.match(path.name)
        if m is None:
            continue
        session, participant, gesture, trial = (int(g) for g in m.groups())
        files.append(TrialFile(hea_path=path, session=session, participant=participant,
                                gesture=gesture, trial=trial))
    return files


def parse_hea(path: Path) -> RecordHeader:
    """WFDB header di testo: prima riga = record_name n_channels fs n_samples,
    poi una riga per canale con gain(baseline)/units al terzo campo."""
    lines = path.read_text().splitlines()
    parts = lines[0].split()
    record_name, n_channels, fs_hz, n_samples = parts[0], int(parts[1]), float(parts[2]), int(parts[3])

    channels = []
    for line in lines[1 : 1 + n_channels]:
        fields = line.split()
        m = _GAIN_RE.match(fields[2])
        if m is None:
            raise ValueError(f"{path}: campo gain/baseline non nel formato atteso: {fields[2]!r}")
        gain, baseline, units = float(m.group(1)), int(m.group(2)), m.group(3)
        signal_name = fields[-1]
        channels.append(ChannelHeader(signal_name=signal_name, gain=gain, baseline=baseline, units=units))

    return RecordHeader(record_name=record_name, n_channels=n_channels, fs_hz=fs_hz,
                         n_samples=n_samples, channels=tuple(channels))


def load_dat(header: RecordHeader, dat_path: Path) -> np.ndarray:
    """.dat WFDB Format=16: int16 little-endian, multiplexato (ch0_t0, ch1_t0, ...,
    chN_t0, ch0_t1, ...). Ritorna (n_samples, n_channels) - RAW, senza conversione a
    unita' fisiche (il file e' gia' int16: nessuna quantizzazione ulteriore serve per
    "int16 memory-mapped", a differenza di CapgMyo che partiva da float64)."""
    raw = np.fromfile(dat_path, dtype="<i2")
    expected = header.n_samples * header.n_channels
    if raw.size != expected:
        raise ValueError(f"{dat_path}: {raw.size} campioni letti, attesi {expected} "
                          f"({header.n_samples} x {header.n_channels})")
    return raw.reshape(header.n_samples, header.n_channels)


def select_emg_channels(header: RecordHeader, data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Scarta U1-U4 (non EMG, docs/fatti_da_verificare.md #16). Ritorna (data_emg,
    nomi_canali_emg) nell'ordine di EMG_CHANNEL_NAMES, non nell'ordine del file."""
    name_to_idx = {c.signal_name: i for i, c in enumerate(header.channels)}
    missing = [n for n in EMG_CHANNEL_NAMES if n not in name_to_idx]
    if missing:
        raise ValueError(f"canali EMG attesi mancanti nell'header: {missing}")
    idx = [name_to_idx[n] for n in EMG_CHANNEL_NAMES]
    return data[:, idx], list(EMG_CHANNEL_NAMES)


def qc_channel_validity(data: np.ndarray, *, min_std: float = 1.0, max_abs_frac_clipped: float = 0.01) -> np.ndarray:
    """Come capgmyo.qc_channel_validity ma su int16 grezzi: soglia min_std in unita'
    ADC (non fisiche), non comparabile al min_std di CapgMyo (che era su dati in volt)."""
    data_f = data.astype(np.float64)
    std = data_f.std(axis=0)
    not_flat = std >= min_std
    abs_data = np.abs(data_f)
    per_channel_max = abs_data.max(axis=0, keepdims=True)
    near_max = abs_data >= 0.999 * np.where(per_channel_max > 0, per_channel_max, 1.0)
    clipped_frac = near_max.mean(axis=0)
    not_clipped = clipped_frac <= max_abs_frac_clipped
    return not_flat & not_clipped


def build_montage_metadata(participant: int, session: int) -> MontageMetadata:
    """4 gruppi ad anello (v10 §3.4): topologia e simmetria sono proprieta' del
    gruppo, non del dataset - GRABMyo non e' "un anello da 28", sono 4 anelli."""
    groups = []
    for ring_id, names in RINGS.items():
        n = len(names)
        channels = [
            ChannelMetadata(
                sensor_coords=SensorCoordinates(
                    channel_index=i, ring_angle_deg=360.0 * i / n,
                ),
                electrode_type="EMGUSB2+_bipolar",
                native_fs_hz=NATIVE_FS_HZ,
                effective_band_hz=(0.0, NATIVE_FS_HZ / 2),  # da verificare: banda reale non documentata qui
                mains_frequency_hz=50,  # da verificare: paese di acquisizione non confermato in questa sessione
                raw_or_envelope=RawOrEnvelope.RAW,
                anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN),
                body_region="forearm" if "forearm" in ring_id else "wrist",
            )
            for i in range(n)
        ]
        groups.append(ChannelGroup(
            group_id=ring_id, topology=Topology.RING, symmetry=f"D_{n}", channels=channels,
        ))
    return MontageMetadata(
        dataset_name="grabmyo", subject_id=f"grabmyo_p{participant:02d}",
        session_id=f"session{session}", day_index=session, groups=groups,
    )
