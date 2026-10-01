"""Ingest di NinaPro DB8 (passo 2): 12 soggetti, 16 Delsys Trigno IM in due righe da 8 sull'avambraccio destro, EMG acquisito a 1111 Hz e fornito
sovracampionato a 2 kHz. **Opzione A** (Simone, 01/10/2026, `docs/decisioni.md`): si tiene la griglia a 2 kHz come fornita, con la banda effettiva
dichiarata fino a 555,5 Hz (Nyquist di 1111 Hz); nessun ricampionamento.

Fonti (citazioni verificate parola per parola il 01/10/2026 con `scripts/check_quotes.py`, `docs/fatti_da_verificare.md`, fatto n. 1):
- pagina ufficiale https://ninapro.hevs.ch/instructions/DB8.html: «Ten able-bodied (Subjects 1-10) and two right-hand transradial amputee
  participants (Subjects 11-12) are included in the dataset.»; «16 active double-differential wireless sensors from a Delsys Trigno IM Wireless EMG
  system»; «two rows of eight units around the participants right forearm in correspondence to the radiohumeral joint»; «No specific muscles were
  targeted.»; «The sEMG signals were sampled at a rate of 1111 Hz [...] All signals were upsampled to 2 kHz and post-synchronized.»; tre acquisizioni,
  le prime due con 10 ripetizioni per movimento, la terza con due, raccomandata come test;
- paper (Krasoulis et al. 2019, PMC6747011): per i normodotati «two rows of eight equally spaced sensors, without targeting specific muscles. The two
  rows were placed 3 and 5.5 cm, respectively, below the elbow.»; per gli amputati «13 and 12 sensors were used, respectively, for the two amputee
  participants, due to limited space availability on their remnant limb (right limb in both subjects).»

Formato verificato su file reali il 01/10/2026 (S1_E1_A1, S1_E1_A3, S11_E1_A1 via leonardo-ops, `scripts/inspect_mat.py`): un `.mat` classico per
acquisizione, `<raw_root>/DB8/S<N>_E1_A<k>.mat`, k = 1, 2, 3. Variabili: `emg` (T, 16) single in volt, `acc`/`gyro`/`mag` (T, 48), `glove` (T, 18),
`stimulus`, `restimulus`, `repetition`, `rerepetition` (T, 1) int8, scalari `subject` ed `exercise` (= 1). Mancano `frequency`, `laterality`, `sensor`.
Lo scalare `subject` non segue il nome per gli amputati (S11 -> 101): ci si fida del NOME e lo scalare si registra. S11 ha le colonne 14-16 (indici
13-15) a zero: coerente coi 13 sensori del paper; il QC le scarta (deviazione standard nulla). La quota di potenza sopra 555,5 Hz misurata sui primi
60 s e' fra 0,3% e 3% per canale, non zero (ipotesi non verificata: residuo del sovracampionamento): il report la riporta per soggetto.

Scelte di AG (dichiarate, rivedibili):
- un soggetto = una sessione: le tre acquisizioni, registrate una dopo l'altra, si concatenano come prove (`trials` con `acquisition`, offset e
  lunghezza), come gli esercizi in `ninapro_std`;
- **ordine delle colonne non dichiarato**: si assume indici 0-7 = una riga, 8-15 = l'altra (ordine del testo, come DB6); quale riga sia a 3 cm e quale
  a 5,5 cm dal gomito non e' dichiarato. Da verificare;
- **normodotati**: due gruppi ad anello D_8 (l'equispaziatura e' dichiarata dal paper per loro); **amputati (11, 12)**: anatomia nominale e due gruppi
  SPARSI senza angoli, perche' la fonte dice solo «a similar configuration» con 13 e 12 sensori e non dichiara ne' equispaziatura ne' quali posizioni
  manchino: dichiarare una simmetria D_8 sarebbe un'affermazione non sostenuta;
- frequenza di rete 50 Hz: **deduzione** (affiliazioni del paper nel Regno Unito), non dichiarata dalla fonte; lo schema richiede 50 o 60.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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

GRID_FS_HZ = 2000.0  # griglia dei file (sovracampionata dal fornitore)
ACQUISITION_FS_HZ = 1111.0  # frequenza di acquisizione dichiarata
EFFECTIVE_BAND_HZ = (0.0, ACQUISITION_FS_HZ / 2)
N_COLUMNS = 16
RING_SIZE = 8
ACQUISITIONS = (1, 2, 3)
AMPUTEE_SUBJECTS = frozenset({11, 12})
ELECTRODE_TYPE = "Delsys_Trigno_IM_double_differential"
LABEL_FIELDS = ("stimulus", "restimulus", "repetition", "rerepetition")
# solo queste variabili: acc, gyro, mag (48 colonne ciascuna) e glove (18) sono quasi tutto il file e qui non servono
LOAD_VARS = ["emg", *LABEL_FIELDS, "subject", "exercise"]
SPECTRUM_SECONDS = 60.0  # per la quota di potenza sopra la banda effettiva nel report
_FILE_RE = re.compile(r"^S(\d+)_E1_A(\d+)\.mat$")


@dataclass
class Acquisition:
    acquisition: int  # dal NOME del file
    emg: np.ndarray  # (T, 16) float64
    labels: dict[str, np.ndarray]
    subject_field_in_file: int | None
    exercise_field_in_file: int | None
    label_length_adjustments: dict[str, int] = field(default_factory=dict)


def scan_db8(raw_root: Path) -> dict[int, dict[int, Path]]:
    """{soggetto: {acquisizione: percorso}} per `<raw_root>/DB8/S<N>_E1_A<k>.mat`."""
    out: dict[int, dict[int, Path]] = {}
    for p in sorted((Path(raw_root) / "DB8").glob("*.mat")):
        m = _FILE_RE.match(p.name)
        if m:
            out.setdefault(int(m.group(1)), {})[int(m.group(2))] = p
    return out


def _scalar(mat: dict, key: str):
    return None if key not in mat else np.asarray(mat[key]).ravel()[0]


def parse_acquisition(mat: dict, acquisition: int) -> Acquisition:
    emg = np.asarray(mat["emg"], dtype=np.float64)
    if emg.ndim != 2 or emg.shape[1] != N_COLUMNS:
        raise ValueError(f"forma emg inattesa {emg.shape}, attese {N_COLUMNS} colonne")
    ex = _scalar(mat, "exercise")
    if ex is not None and int(ex) != 1:
        raise ValueError(f"exercise = {ex}, la pagina ufficiale dice 1 in tutti i file")
    labels, adjustments = reconcile_labels(mat, emg.shape[0], LABEL_FIELDS)
    su = _scalar(mat, "subject")
    return Acquisition(acquisition, emg, labels, None if su is None else int(su), None if ex is None else int(ex), adjustments)


def load_subject(paths: dict[int, Path], subject: int) -> list[Acquisition]:
    """Le tre acquisizioni di un soggetto, in ordine. Solleva se ne manca una."""
    if set(paths) != set(ACQUISITIONS):
        raise ValueError(f"soggetto {subject}: acquisizioni {sorted(paths)}, attese {list(ACQUISITIONS)}")
    return [parse_acquisition(sio.loadmat(str(paths[k]), variable_names=LOAD_VARS), k) for k in ACQUISITIONS]


def build_montage_metadata(subject: int) -> MontageMetadata:
    """Due righe da 8 in ordine di colonna (0-7, 8-15): anelli D_8 per i normodotati, gruppi sparsi per gli amputati (vedi il docstring)."""
    amputee = subject in AMPUTEE_SUBJECTS
    common = dict(
        electrode_type=ELECTRODE_TYPE, native_fs_hz=GRID_FS_HZ, effective_band_hz=EFFECTIVE_BAND_HZ,
        mains_frequency_hz=50,  # deduzione (Regno Unito), non dichiarata
        raw_or_envelope=RawOrEnvelope.RAW, chirality=Chirality.RIGHT,
        anatomical_identity=AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN, nominal=amputee),
        body_region="forearm",
    )
    groups = []
    for row in (0, 1):
        idx = range(row * RING_SIZE, (row + 1) * RING_SIZE)
        if amputee:
            channels = [ChannelMetadata(sensor_coords=SensorCoordinates(channel_index=i), **common) for i in idx]
            groups.append(ChannelGroup(group_id=f"row{row + 1}_amputee", topology=Topology.SPARSE, symmetry="none", channels=channels))
        else:
            channels = [ChannelMetadata(sensor_coords=SensorCoordinates(channel_index=i, ring_angle_deg=360.0 * k / RING_SIZE), **common)
                        for k, i in enumerate(idx)]
            groups.append(ChannelGroup(group_id=f"ring8_row{row + 1}", topology=Topology.RING, symmetry="D_8", channels=channels))
    return MontageMetadata(dataset_name="ninapro_db8", subject_id=f"ninapro_db8_s{subject:02d}", session_id="session1", groups=groups)


def power_fraction_above(emg: np.ndarray, fs: float, above_hz: float) -> list[float | None]:
    """Per canale: potenza sopra `above_hz` / potenza totale (media tolta); None per un canale a zero."""
    x = emg - emg.mean(axis=0, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    tot = p.sum(axis=0)
    return [None if t == 0 else float(a / t) for a, t in zip(p[f > above_hz].sum(axis=0), tot)]


def ingest_subject(paths: dict[int, Path], out_root: Path, subject: int) -> dict:
    """Un soggetto -> `<out_root>/s<NN>/session1/{data_int16.npy, labels.npz, metadata.json}`. Scala int16 dal massimo assoluto; QC di canale
    sul soggetto intero (soglia relativa, come gli altri NinaPro: le colonne senza sensore degli amputati hanno deviazione standard nulla)."""
    from wearusfm.ingest.capgmyo import qc_channel_validity, to_int16
    from wearusfm.ingest.common import montage_to_dict, validate_montage_dict

    acq = load_subject(paths, subject)
    emg = np.concatenate([a.emg for a in acq], axis=0)
    labels = {k: np.concatenate([a.labels[k] for a in acq]) for k in sorted(set.intersection(*(set(a.labels) for a in acq)))}
    trials, offset = [], 0
    for a in acq:
        trials.append({"acquisition": a.acquisition, "offset": offset, "n_samples": int(a.emg.shape[0]),
                       "subject_field_in_file": a.subject_field_in_file, "exercise_field_in_file": a.exercise_field_in_file,
                       "label_length_adjustments": a.label_length_adjustments})
        offset += a.emg.shape[0]

    valid = qc_channel_validity(emg, min_std=relative_min_std(emg))
    quantized, scale = to_int16(emg)
    out_dir = out_root / f"s{subject:02d}" / "session1"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data_int16.npy", quantized)
    if labels:
        np.savez_compressed(out_dir / "labels.npz", **labels)
    montage_dict = montage_to_dict(build_montage_metadata(subject), valid)
    validate_montage_dict(montage_dict)
    sidecar = {
        "montage": montage_dict, "int16_scale": scale, "shape": list(quantized.shape), "native_fs_hz": GRID_FS_HZ,
        "acquisition_fs_hz": ACQUISITION_FS_HZ, "effective_band_hz": list(EFFECTIVE_BAND_HZ),
        "note_fs": "griglia a 2 kHz fornita dal dataset (EMG acquisito a 1111 Hz e sovracampionato); opzione A, decisioni.md 01/10/2026",
        "trials": trials, "label_fields": sorted(labels), "n_channels_discarded_by_qc": int((~valid).sum()),
        "laterality": "r", "nominal_anatomy": subject in AMPUTEE_SUBJECTS,
        "provider_recommended_test_acquisition": 3,
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))
    first = acq[0].emg[: int(SPECTRUM_SECONDS * GRID_FS_HZ)]
    return {
        "subject": subject, "n_samples": int(quantized.shape[0]), "hours": quantized.shape[0] / GRID_FS_HZ / 3600,
        "samples_per_acquisition": [int(a.emg.shape[0]) for a in acq],
        "discarded_channels": [int(i) for i in np.flatnonzero(~valid)],
        "subject_field_in_file": sorted({a.subject_field_in_file for a in acq if a.subject_field_in_file is not None}),
        "movements_per_acquisition": [int((np.unique(a.labels["restimulus"]) > 0).sum()) if "restimulus" in a.labels else None for a in acq],
        "label_length_adjustments": {str(a.acquisition): a.label_length_adjustments for a in acq if a.label_length_adjustments},
        "power_fraction_above_effective_band_first_60s_acq1": power_fraction_above(first, GRID_FS_HZ, EFFECTIVE_BAND_HZ[1]),
        "int16_scale": scale, "emg_max_abs": float(np.abs(emg).max()), "memmap_path": str(out_dir / "data_int16.npy"),
    }
