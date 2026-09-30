"""Ingest di Hyser (HD-sEMG, PhysioNet, ODC-By v1.0): 20 soggetti, 2 sessioni, 256 canali a 2048 Hz.

Formato verificato su file reali il 30/09/2026 (subject01_session1, via leonardo-ops) e sul readme
ufficiale: WFDB (`.hea` + `.dat`, Format 16). Cinque sotto-dataset in cartelle `<nome>_dataset/
subject<NN>_session<k>/`: `1dof`, `mvc`, `ndof`, `pr`, `random`. Per ogni registrazione tre versioni:
`*_force_*` (forza, 5 canali a 100 Hz: NON EMG), `*_preprocess_*` (EMG filtrato) e `*_raw_*` (EMG
grezzo): **si ingerisce solo `*_raw_*`** (76 GB su 143). `pr` ha `dynamic_raw_sampleN` (1 s) e
`maintenance_raw_sampleN` (4 s) e due file `label_*.txt` (etichette dei gesti, copiati tal quali).

256 canali da QUATTRO griglie 8x8 (readme ufficiale): ED = estensori, estremita' distale; EP =
estensori, prossimale; FD = flessori, distale; FP = flessori, prossimale. Nome del canale `XX-i-j`,
i = riga, j = colonna (1-8). L'ordine nel file e' ED (64), EP (64), FD (64), FP (64), con riga 8 -> 1 e
colonna 8 -> 1 dentro ciascuna. Il readme NON dice se i canali sono monopolari o differenziali.

**Anomalia verificata: guadagno e baseline WFDB sono DIVERSI in ogni file** (ogni registrazione e'
riscalata a fondo scala: 270 file del `pr` di un soggetto hanno 270 coppie (gain, baseline) distinte; nello
stesso file il guadagno varia fra canali di ~1,5x). I codici int16 di file diversi NON sono sulla stessa
scala. Quindi ogni file si porta in unita' fisiche, (codice - baseline) / gain, e le registrazioni di un
gruppo si concatenano SOLO dopo; poi si quantizza a int16 con UNA SCALA PER CANALE, dal MASSIMO
assoluto del canale nel gruppo (mai percentili). Una scala unica per gruppo non va bene: nei file
reali le ampiezze fisiche dei canali variano di ~100x (picchi da 0,03 a 16 nelle unita' dell'header:
un canale caldo darebbe ai canali quieti solo 4-8 bit); con la scala per canale ognuno usa tutta la
dinamica, come nei codici originali. Il sidecar ha `int16_scale` = lista di 256 numeri
(ricostruzione: codice / scala del canale). Si legge due volte (prima i picchi, poi la scrittura su
memmap) per non tenere in memoria un gruppo intero (ndof: ~1,6 GB in float32).

Una uscita per (soggetto, sessione, sotto-dataset): `<out_root>/s<NN>/session<k>_<gruppo>/` con
`data_int16.npy`, `metadata.json` (con `trials`: nome, offset, campioni) e, per `pr`, i txt delle
etichette. QC di canale per registrazione, poi maggioranza (>= 50%): stessa regola di CapgMyo/GRABMyo.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wearusfm.ingest.grabmyo import RecordHeader, load_dat, parse_hea
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
N_CHANNELS = 256
ARRAY_NAMES = ("ED", "EP", "FD", "FP")
# readme ufficiale: ED/FD = estremita' DISTALE, EP/FP = PROSSIMALE dei muscoli estensori/flessori
ARRAY_REGION = {"ED": "forearm_distal", "FD": "forearm_distal", "EP": "forearm_proximal", "FP": "forearm_proximal"}
_SESSION_DIR_RE = re.compile(r"^subject(\d+)_session(\d+)$")
_CHANNEL_RE = re.compile(r"^([A-Z]{2})-(\d+)-(\d+)$")
_N_FLOAT32_BYTES = 4


@dataclass(frozen=True)
class Group:
    subject: int
    session: int
    key: str  # 1dof | mvc | ndof | random | pr_dynamic | pr_maintenance
    hea_paths: tuple[Path, ...]
    label_files: tuple[Path, ...]


def _natural(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def _group_key(sub_dir: str, stem: str) -> str:
    base = sub_dir.removesuffix("_dataset")
    if base == "pr":
        return "pr_" + stem.split("_")[0]  # dynamic | maintenance
    return base


def scan_hyser(raw_root: Path) -> list[Group]:
    """I gruppi (soggetto, sessione, sotto-dataset) con i soli file `*_raw_*.hea`, in ordine naturale."""
    buckets: dict[tuple[int, int, str], list[Path]] = {}
    labels: dict[tuple[int, int, str], list[Path]] = {}
    for sub in sorted(Path(raw_root).glob("*_dataset")):
        for sd in sorted(sub.iterdir()):
            m = _SESSION_DIR_RE.match(sd.name)
            if not m or not sd.is_dir():
                continue
            subj, sess = int(m.group(1)), int(m.group(2))
            for hea in sd.glob("*_raw_*.hea"):
                buckets.setdefault((subj, sess, _group_key(sub.name, hea.stem)), []).append(hea)
            if sub.name == "pr_dataset":
                for txt in sd.glob("label_*.txt"):
                    key = "pr_" + txt.stem.removeprefix("label_")
                    labels.setdefault((subj, sess, key), []).append(txt)
    return [
        Group(k[0], k[1], k[2], tuple(sorted(v, key=_natural)), tuple(sorted(labels.get(k, []))))
        for k, v in sorted(buckets.items())
    ]


def read_physical(hea_path: Path) -> tuple[np.ndarray, RecordHeader]:
    """(T, 256) float32 in volt = (codice - baseline) / gain, per canale, dal PROPRIO header del file."""
    header = parse_hea(hea_path)
    if header.n_channels != N_CHANNELS or header.fs_hz != NATIVE_FS_HZ:
        raise ValueError(f"{hea_path}: {header.n_channels} canali a {header.fs_hz} Hz, attesi {N_CHANNELS} a {NATIVE_FS_HZ}")
    codes = load_dat(header, hea_path.with_suffix(".dat")).astype(np.float32)
    gain = np.array([c.gain for c in header.channels], dtype=np.float32)
    base = np.array([c.baseline for c in header.channels], dtype=np.float32)
    return (codes - base) / gain, header


def channel_layout(header: RecordHeader) -> list[tuple[str, int, int]]:
    """[(griglia, riga 1-8, colonna 1-8)] nell'ordine delle colonne; controlla che le quattro griglie
    siano contigue e complete (64 canali ciascuna, senza ripetizioni)."""
    out = []
    for c in header.channels:
        m = _CHANNEL_RE.match(c.signal_name)
        if not m or m.group(1) not in ARRAY_NAMES:
            raise ValueError(f"nome di canale inatteso {c.signal_name!r}")
        out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    runs = [(k, len(list(g))) for k, g in itertools.groupby(a for a, _, _ in out)]
    if sorted(k for k, _ in runs) != sorted(ARRAY_NAMES) or any(n != 64 for _, n in runs):
        raise ValueError(f"griglie non contigue o incomplete: {runs}")
    if len({(a, i, j) for a, i, j in out}) != N_CHANNELS:
        raise ValueError("canali duplicati")
    return out


def build_montage_metadata(layout: list[tuple[str, int, int]], subject: int, session: int, key: str) -> MontageMetadata:
    groups = []
    start = 0
    for name, run in itertools.groupby(a for a, _, _ in layout):
        n = len(list(run))
        channels = [
            ChannelMetadata(
                sensor_coords=SensorCoordinates(channel_index=start + k, grid_row=i - 1, grid_col=j - 1),
                electrode_type="HD_grid_8x8_polarity_unspecified", native_fs_hz=NATIVE_FS_HZ,
                effective_band_hz=(0.0, NATIVE_FS_HZ / 2),  # da verificare: banda hardware non documentata
                mains_frequency_hz=50,  # da verificare: Shanghai, non dichiarato dalla fonte
                raw_or_envelope=RawOrEnvelope.RAW,
                anatomical_identity=AnatomicalIdentity(region=ARRAY_REGION[name], precision=AnatomicalPrecision.REGION),
                body_region="forearm",
            )
            for k, (_, i, j) in enumerate(layout[start : start + n])
        ]
        groups.append(ChannelGroup(group_id=name, topology=Topology.GRID_2D, symmetry="translational_2d", channels=channels))
        start += n
    return MontageMetadata(
        dataset_name="hyser", subject_id=f"hyser_s{subject:02d}", session_id=f"session{session}_{key}",
        day_index=session, groups=groups,
    )


def ingest_group(group: Group, out_root: Path) -> dict:
    """Un gruppo -> `<out_root>/s<NN>/session<k>_<gruppo>/`. Due passate sui file (picco, poi scrittura)."""
    from wearusfm.ingest.capgmyo import qc_channel_validity
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    peak = np.zeros(N_CHANNELS)
    lengths, layout, valid_votes = [], None, []
    for hea in group.hea_paths:
        x, header = read_physical(hea)
        lay = channel_layout(header)
        if layout is None:
            layout = lay
        elif lay != layout:
            raise ValueError(f"{hea}: ordine dei canali diverso dal resto del gruppo")
        peak = np.maximum(peak, np.abs(x).max(axis=0))
        min_std = max(1e-12, 1e-3 * float(np.median(x.std(axis=0))))
        valid_votes.append(qc_channel_validity(x, min_std=min_std))
        lengths.append(x.shape[0])
    if peak.max() <= 0:
        raise ValueError(f"{group}: segnale nullo")
    scale = np.where(peak > 0, 32000.0 / np.where(peak > 0, peak, 1.0), 1.0).astype(np.float64)  # (256,)

    out_dir = out_root / f"s{group.subject:02d}" / f"session{group.session}_{group.key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(lengths)
    mm = np.lib.format.open_memmap(out_dir / "data_int16.npy", mode="w+", dtype=np.int16, shape=(total, N_CHANNELS))
    trials, offset = [], 0
    for hea, n in zip(group.hea_paths, lengths):
        x, _ = read_physical(hea)
        mm[offset : offset + n] = np.clip(np.round(x * scale), -32768, 32767).astype(np.int16)
        trials.append({"name": hea.stem, "offset": offset, "n_samples": n})
        offset += n
    mm.flush()
    del mm

    votes = np.stack(valid_votes)
    channel_valid = votes.mean(axis=0) >= 0.5
    montage = build_montage_metadata(layout, group.subject, group.session, group.key)
    montage_dict = montage_to_dict(montage, channel_valid)
    validate_montage_dict(montage_dict)
    for txt in group.label_files:
        shutil.copyfile(txt, out_dir / txt.name)
    sidecar = {
        "montage": montage_dict, "int16_scale": [float(v) for v in scale], "int16_scale_per_channel": True, "shape": [total, N_CHANNELS], "native_fs_hz": NATIVE_FS_HZ,
        "trials": trials, "unit_before_scaling": "volt = (codice WFDB - baseline) / gain, per canale e per file",
        "label_files": [t.name for t in group.label_files], "n_channels_discarded_by_qc": int((~channel_valid).sum()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))
    return {
        "subject": group.subject, "session": group.session, "group": group.key, "n_trials": len(trials),
        "n_samples": total, "hours": total / NATIVE_FS_HZ / 3600,
        "discarded_channels": [int(i) for i in np.flatnonzero(~channel_valid)],
        "peak_min_max": [float(peak.min()), float(peak.max())], "int16_scale_min_max": [float(scale.min()), float(scale.max())],
        "n_label_files": len(group.label_files),
    }
