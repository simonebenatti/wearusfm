"""Ingest di NinaPro DB10 = MeganePro, **solo MDS1** (esercizio 1, prese statiche e dinamiche), passo 2. Decisione di Simone del 01/10/2026: solo i
`.mat`; MDS2 e MDS4 contengono un inviluppo RMS a 100 Hz di 2 elettrodi (Saetta et al. 2020: «rectified via a moving root-mean-square with a window
length of 300 samples») e restano fuori dal front-end come DB1 (v10 §2.4); i file `_orig` di MDS1 non contengono EMG.

Fonti (Cognolato et al. 2020, Sci Data 7:43, PMC7010656; citazioni verificate parola per parola il 01/10/2026, `docs/fatti_da_verificare.md`):
«A total of 15 transradial amputees and 30 able-bodied subjects were recruited for this study.»; «The sEMG data are sampled at 1926 Hz»; «Twelve sEMG
electrodes were placed in two arrays around the right forearm or residual limb.»; «An array of eight electrodes was placed equidistantly around the
forearm, starting at the radio-humeral joint and moving in the direction of pronation. A second array was located approximately 45 mm more distally and
aligned with the gaps between electrodes one and two, three and four, and so on»; «columns 1-8 refer to the signals of the electrodes forming the
proximal array while columns 9-12 the signals of the electrodes forming the distal array»; filtro di rete a 50 Hz (Hampel). Eccezioni dichiarate:
S024 senza l'elettrodo 8 (11 colonne); S039 e S115 con 7 elettrodi nell'anello prossimale (11 colonne); S108 con il solo anello prossimale (8 colonne).

Formato verificato su file reali il 01/10/2026: `MDS1/S<NNN>_ex1.mat` (classico), `emg` (T, 12) single in volt, `ts` (T, 1) in secondi con **pause**
(S010: 50 salti, il piu' lungo 166 s), etichette `grasp`, `object`, ... e le versioni riallineate `re*` (int8). Frequenza dai `ts`: 1925,98 Hz.

Scelte (dichiarate):
- un soggetto = una sessione; la registrazione si spezza alle pause dell'asse dei tempi (passo > 1,5 volte il mediano): un segmento per tratto continuo,
  come prova (`trials` con offset, lunghezza, istante d'inizio), cosi' nessuna finestra attraversa una pausa;
- frequenza nativa 1926 Hz (dichiarata), controllata contro i `ts`; nessun ricampionamento;
- normodotati con 12 colonne: anello prossimale D_8 (angoli 0, 45, ... 315) e anello distale D_4 (22,5, 112,5, ...: «aligned with the gaps between
  electrodes one and two, three and four»); S024: anello prossimale con le posizioni 1-7 (la 8 manca) piu' il distale; S039 (7 elettrodi prossimali,
  spaziatura non dichiarata): gruppi sparsi senza angoli;
- amputati (S101-S115): gruppi sparsi senza angoli e anatomia nominale (stessa scelta di DB8, Simone 01/10/2026: «gli amputati hanno fasce muscolari
  irregolari, quindi va bene sparso»); lato ignoto (la fonte dice «right forearm or residual limb»; il lato del moncone e' solo nella Tabella 1, non
  verificata cella per cella); normodotati: braccio destro;
- rete 50 Hz (filtro a 50 Hz dichiarato; acquisizioni in Svizzera e Italia).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

from wearusfm.ingest.common import reconcile_labels, relative_min_std
from wearusfm.metadata.schema import (
    AnatomicalIdentity,
    AnatomicalPrecision,
    ChannelGroup,
    ChannelMetadata,
    Chirality,
    MontageMetadata,
    RawOrEnvelope,
    SensorCoordinates,
    Topology,
)

NATIVE_FS_HZ = 1926.0
MAX_FS_DEVIATION = 0.005  # frequenza dai ts entro lo 0,5% di quella dichiarata
GAP_FACTOR = 1.5
ELECTRODE_TYPE = "Delsys_Trigno_double_differential"
LABEL_FIELDS = ("grasp", "grasprepetition", "object", "objectpart", "objectrepetition", "position", "dynamic",
                "regrasp", "regrasprepetition", "reobject", "reobjectpart", "reobjectrepetition", "reposition", "redynamic")
LOAD_VARS = ["ts", "emg", *LABEL_FIELDS]
SUSPECT_STD_RATIO = 0.05  # nel report: canali con deviazione standard < 5% della mediana (sospetti, non scartati)
# colonne attese per soggetto (Cognolato et al. 2020, «Notes on the subjects»): (prossimali, distali)
EXCEPTIONS = {24: (7, 4), 39: (7, 4), 115: (7, 4), 108: (8, 0)}
_FILE_RE = re.compile(r"^S(\d{3})_ex1\.mat$")


def is_amputee(subject: int) -> bool:
    return subject >= 101


def scan_db10(raw_root: Path) -> dict[int, Path]:
    """{soggetto: file standard} per `<raw_root>/DB10/MDS1/S<NNN>_ex1.mat` (esclusi i `_orig`, che non hanno EMG)."""
    out = {}
    for p in sorted((Path(raw_root) / "DB10" / "MDS1").glob("S*_ex1.mat")):
        m = _FILE_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def segments_from_ts(ts: np.ndarray) -> tuple[list[tuple[int, int, float]], float]:
    """Tratti continui [(inizio, lunghezza, istante d'inizio in s)] e frequenza dai `ts` (passo mediano). Pausa = passo > GAP_FACTOR x mediano."""
    d = np.diff(ts)
    if (d <= 0).any():
        raise ValueError(f"asse dei tempi non crescente ({int((d <= 0).sum())} passi <= 0)")
    med = float(np.median(d))
    cuts = np.flatnonzero(d > GAP_FACTOR * med) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [ts.size]))
    return [(int(s), int(e - s), float(ts[s])) for s, e in zip(starts, ends)], 1.0 / med


def expected_layout(subject: int) -> tuple[int, int]:
    return EXCEPTIONS.get(subject, (8, 4))


def build_montage_metadata(subject: int, n_cols: int) -> MontageMetadata:
    prox, dist = expected_layout(subject)
    if prox + dist != n_cols:
        raise ValueError(f"S{subject:03d}: {n_cols} colonne EMG, attese {prox} + {dist} (Cognolato et al. 2020)")
    amputee = is_amputee(subject)
    common = dict(
        electrode_type=ELECTRODE_TYPE, native_fs_hz=NATIVE_FS_HZ, effective_band_hz=(0.0, NATIVE_FS_HZ / 2), mains_frequency_hz=50,
        raw_or_envelope=RawOrEnvelope.RAW, chirality=Chirality.UNKNOWN if amputee else Chirality.RIGHT, body_region="forearm",
        anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN, nominal=amputee),
    )

    def ch(i, angle=None):
        return ChannelMetadata(sensor_coords=SensorCoordinates(channel_index=i, ring_angle_deg=angle), **common)

    groups = []
    rings = not amputee and subject != 39  # S039: 7 elettrodi prossimali a spaziatura non dichiarata
    if rings:
        positions = range(prox) if subject != 24 else range(7)  # S024: posizioni 1-7, manca la 8
        groups.append(ChannelGroup(group_id="ring8_proximal", topology=Topology.RING, symmetry="D_8",
                                   channels=[ch(k, 45.0 * p) for k, p in enumerate(positions)]))
        groups.append(ChannelGroup(group_id="ring4_distal", topology=Topology.RING, symmetry="D_4",
                                   channels=[ch(prox + j, 22.5 + 90.0 * j) for j in range(dist)]))
    else:
        groups.append(ChannelGroup(group_id="proximal", topology=Topology.SPARSE, symmetry="none", channels=[ch(k) for k in range(prox)]))
        if dist:
            groups.append(ChannelGroup(group_id="distal", topology=Topology.SPARSE, symmetry="none", channels=[ch(prox + j) for j in range(dist)]))
    return MontageMetadata(dataset_name="ninapro_db10", subject_id=f"ninapro_db10_s{subject:03d}", session_id="ex1", groups=groups)


def ingest_subject(path: Path, out_root: Path, subject: int) -> dict:
    """Un soggetto -> `<out_root>/s<NNN>/ex1/{data_int16.npy, labels.npz, metadata.json}`."""
    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    mat = sio.loadmat(str(path), variable_names=LOAD_VARS)
    ts = np.asarray(mat["ts"], dtype=np.float64).ravel()
    emg = np.asarray(mat["emg"], dtype=np.float64)
    if emg.ndim != 2 or emg.shape[0] != ts.size:
        raise ValueError(f"emg {emg.shape} contro ts {ts.shape}")
    segs, fs_ts = segments_from_ts(ts)
    if abs(fs_ts / NATIVE_FS_HZ - 1) > MAX_FS_DEVIATION:
        raise ValueError(f"frequenza dai ts {fs_ts:.2f} Hz, dichiarata {NATIVE_FS_HZ}")
    montage = build_montage_metadata(subject, emg.shape[1])
    labels, adjustments = reconcile_labels(mat, emg.shape[0], LABEL_FIELDS)
    std = emg.std(axis=0)
    valid = qc_channel_validity(emg, min_std=relative_min_std(emg))
    quantized, scale = to_int16(emg)
    out_dir = Path(out_root) / f"s{subject:03d}" / "ex1"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_int16.npy", quantized)
    np.savez_compressed(out_dir / "labels.npz", **labels)
    montage_dict = montage_to_dict(montage, valid)
    validate_montage_dict(montage_dict)
    trials = [{"segment": k, "offset": s, "n_samples": n, "t_start_s": t0} for k, (s, n, t0) in enumerate(segs)]
    sidecar = {
        "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape), "native_fs_hz": NATIVE_FS_HZ, "fs_from_ts_hz": fs_ts,
        "trials": trials, "label_fields": sorted(labels), "label_length_adjustments": adjustments,
        "n_channels_discarded_by_qc": int((~valid).sum()), "amputee": is_amputee(subject), "nominal_anatomy": is_amputee(subject),
        "laterality": "" if is_amputee(subject) else "r", "source_file": path.name,
        "note": "segmenti = tratti continui dell'asse dei tempi (pause dove il passo supera 1,5 volte il mediano)",
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))
    med = float(np.median(std))
    return {
        "subject": subject, "amputee": is_amputee(subject), "n_channels": int(emg.shape[1]), "n_samples": int(emg.shape[0]),
        "hours": emg.shape[0] / NATIVE_FS_HZ / 3600, "fs_from_ts_hz": fs_ts, "n_segments": len(segs),
        "longest_gap_s": float(max((segs[k + 1][2] - (segs[k][2] + segs[k][1] / fs_ts) for k in range(len(segs) - 1)), default=0.0)),
        "discarded_channels": [int(i) for i in np.flatnonzero(~valid)],
        "suspect_low_channels": [int(i) for i in np.flatnonzero(std < SUSPECT_STD_RATIO * med)],
        "std_ratio_to_median": [round(float(v / med), 4) for v in std],
        "groups": [g["group_id"] for g in montage_dict["groups"]], "label_length_adjustments": adjustments,
        "n_grasps": int((np.unique(labels["grasp"]) > 0).sum()) if "grasp" in labels else None,
        "int16_scale": scale, "emg_max_abs": float(np.abs(emg).max()), "memmap_path": str(out_dir / "data_int16.npy"),
    }
