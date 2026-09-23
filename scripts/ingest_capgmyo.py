#!/usr/bin/env python3
"""Ingest di CapgMyo-DBa (passo 2, piano_operativo_v10.md §11): converte i trial .mat
grezzi in int16 memory-mapped sotto $WORK/data/processed/, con QC offline e un report
per dataset (piano: "ore, soggetti, canali, percentuale di canali scartati dal QC").

Per soggetto: una scala int16 CONDIVISA fra tutti gli 80 trial (dal massimo assoluto su
tutti i trial del soggetto), non una scala per trial - altrimenti trial diversi dello
stesso soggetto non sarebbero confrontabili in ampiezza dopo la quantizzazione.

Uso:
    python scripts/ingest_capgmyo.py --raw-root $WORK/data/raw/capgmyo \\
        --out-root $WORK/data/processed/capgmyo --subjects 1 \\
        --report $WORK/wearusfm_runs/results/passo2/capgmyo_ingest_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.capgmyo import (  # noqa: E402
    N_CHANNELS,
    NATIVE_FS_HZ,
    build_montage_metadata,
    check_completeness,
    qc_channel_validity,
    scan_capgmyo,
    to_int16,
    verify_mat_consistency,
)
from wearusfm.metadata.schema import export_json_schema  # noqa: E402


def _validate_against_schema(montage_dict: dict) -> None:
    """Validazione leggera senza dipendenza da jsonschema (non nell'ambiente): controlla
    solo i campi required di primo livello e dei gruppi - basta a intercettare un
    errore grossolano nella costruzione dei metadati, non e' una validazione completa."""
    schema = export_json_schema()
    for field in schema["required"]:
        if field not in montage_dict:
            raise ValueError(f"metadati non conformi allo schema: manca {field!r}")


def ingest_subject(raw_root: Path, out_root: Path, subject: int) -> dict:
    files = [f for f in scan_capgmyo(raw_root) if f.subject == subject]
    if not files:
        raise FileNotFoundError(f"nessun file trovato per il soggetto {subject} sotto {raw_root}")
    files.sort(key=lambda f: (f.gesture, f.trial))

    trials_data = []
    trial_meta = []
    for tf in files:
        mat = sio.loadmat(tf.path)
        verify_mat_consistency(mat, tf)
        data = np.asarray(mat["data"], dtype=np.float64)
        if data.shape != (1000, N_CHANNELS):
            raise ValueError(f"{tf.path}: shape inattesa {data.shape}, attesa (1000, {N_CHANNELS})")
        trials_data.append(data)
        trial_meta.append({"gesture": tf.gesture, "trial": tf.trial})

    stacked = np.stack(trials_data)  # (n_trials, 1000, 128)

    # QC: un canale e' valido per il soggetto solo se valido nella maggioranza dei trial
    per_trial_valid = np.stack([qc_channel_validity(d) for d in trials_data])  # (n_trials, 128)
    channel_valid = per_trial_valid.mean(axis=0) >= 0.5
    n_channels_discarded = int((~channel_valid).sum())

    # scala condivisa fra tutti i trial del soggetto (v10 §11, "int16 memory-mapped")
    quantized, scale = to_int16(stacked)

    out_dir = out_root / f"s{subject:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = out_dir / "data_int16.npy"
    np.save(memmap_path, quantized)

    montage = build_montage_metadata(subject=subject)
    montage_dict = {
        "dataset_name": montage.dataset_name,
        "subject_id": montage.subject_id,
        "session_id": montage.session_id,
        "day_index": montage.day_index,
        "groups": [{
            "group_id": g.group_id, "topology": g.topology.value, "symmetry": g.symmetry,
            "band_orientation_deg": g.band_orientation_deg,
            "channels": [{
                "sensor_coords": {
                    "channel_index": c.sensor_coords.channel_index,
                    "grid_row": c.sensor_coords.grid_row, "grid_col": c.sensor_coords.grid_col,
                    "ring_angle_deg": c.sensor_coords.ring_angle_deg,
                },
                "electrode_type": c.electrode_type, "native_fs_hz": c.native_fs_hz,
                "effective_band_hz": list(c.effective_band_hz),
                "mains_frequency_hz": c.mains_frequency_hz,
                "raw_or_envelope": c.raw_or_envelope.value,
                "anatomical_identity": {
                    "region": c.anatomical_identity.region, "compartment": c.anatomical_identity.compartment,
                    "sector": c.anatomical_identity.sector, "muscle": c.anatomical_identity.muscle,
                    "precision": c.anatomical_identity.precision.value,
                    "muscle_ontology_id": c.anatomical_identity.muscle_ontology_id,
                    "soft_compartment_weights": c.anatomical_identity.soft_compartment_weights,
                    "nominal": c.anatomical_identity.nominal,
                },
                "chirality": c.chirality.value, "calibration_status": c.calibration_status.value,
                "qc_valid": bool(channel_valid[c.sensor_coords.channel_index]),
                "body_region": c.body_region,
            } for c in g.channels],
        } for g in montage.groups],
    }
    _validate_against_schema(montage_dict)

    sidecar = {
        "montage": montage_dict,
        "int16_scale": scale,
        "shape": list(quantized.shape),  # (n_trials, 1000, 128)
        "native_fs_hz": NATIVE_FS_HZ,
        "trials": trial_meta,
        "n_channels_discarded_by_qc": n_channels_discarded,
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))

    return {
        "subject": subject,
        "n_trials": len(files),
        "n_channels_discarded": n_channels_discarded,
        "hours": len(files) * 1.0 / 3600,  # 1000 campioni @ 1kHz = 1s/trial
        "memmap_path": str(memmap_path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    all_files = scan_capgmyo(args.raw_root)
    completeness_warnings = check_completeness(all_files)

    t0 = time.time()
    per_subject = []
    for s in args.subjects:
        print(f"soggetto {s}: ingest in corso...")
        per_subject.append(ingest_subject(args.raw_root, args.out_root, s))
        print(f"  fatto: {per_subject[-1]}")
    elapsed = time.time() - t0

    report = {
        "dataset": "capgmyo_dba",
        "subjects_processed": args.subjects,
        "n_subjects_processed": len(args.subjects),
        "n_trials_total": sum(r["n_trials"] for r in per_subject),
        "hours_total": sum(r["hours"] for r in per_subject),
        "n_channels": N_CHANNELS,
        "per_subject": per_subject,
        "completeness_warnings": completeness_warnings,
        "elapsed_s": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
