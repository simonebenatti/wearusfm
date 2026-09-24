#!/usr/bin/env python3
"""Ingest di CSL-hdemg (passo 2, piano_operativo_v10.md §11).

Per soggetto+sessione: tutti i 27 file gest (idle + 26 gesti) in un unico array
int16 - una scala condivisa fra TUTTI i trial della sessione (come CapgMyo).

Legge direttamente dallo zip sorgente, senza mai estrarlo su disco.

Uso:
    python scripts/ingest_csl_hdemg.py --zip-path $WORK/data/raw/csl_hdemg/csl_hdemg.zip \\
        --out-root $WORK/data/processed/csl_hdemg --subjects 5 --sessions 3 \\
        --report $WORK/wearusfm_runs/results/passo2/csl_hdemg_ingest_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.csl_hdemg import (  # noqa: E402
    N_CHANNELS_EMG,
    N_GESTURES,
    NATIVE_FS_HZ,
    build_montage_metadata,
    filter_differential_channels,
    load_gest_file,
    qc_channel_validity,
    scan_csl_hdemg,
    to_int16,
)
from wearusfm.metadata.schema import export_json_schema  # noqa: E402


def _validate_against_schema(montage_dict: dict) -> None:
    schema = export_json_schema()
    for field in schema["required"]:
        if field not in montage_dict:
            raise ValueError(f"metadati non conformi allo schema: manca {field!r}")


def ingest_subject_session(zip_path: Path, out_root: Path, subject: int, session: int) -> dict:
    files = [f for f in scan_csl_hdemg(zip_path) if f.subject == subject and f.session == session]
    if not files:
        raise FileNotFoundError(f"nessun file per soggetto {subject} sessione {session}")
    files.sort(key=lambda f: f.gesture)
    completeness_warning = None
    if len(files) != N_GESTURES:
        completeness_warning = f"attesi {N_GESTURES} file gesto, trovati {len(files)}"

    all_trials: list[np.ndarray] = []  # ognuno (168, L), L variabile
    trial_meta = []
    for gf in files:
        for rep_idx, raw_trial in enumerate(load_gest_file(zip_path, gf.member_name)):
            filtered = filter_differential_channels(raw_trial)  # (168, L)
            all_trials.append(filtered.T)  # -> (L, 168), convenzione comune agli altri moduli ingest
            trial_meta.append({"gesture": gf.gesture, "repetition": rep_idx, "n_samples": int(filtered.shape[1])})

    # lunghezza variabile di pochi ms (README ufficiale): tronca tutti alla piu' corta
    # per poter impilare in un unico array, invece di tenere una lista ragged su disco
    min_len = min(t.shape[0] for t in all_trials)
    stacked = np.stack([t[:min_len] for t in all_trials])  # (n_trials, min_len, 168)

    per_trial_valid = np.stack([qc_channel_validity(t[:min_len].T) for t in all_trials])
    channel_valid = per_trial_valid.mean(axis=0) >= 0.5
    n_channels_discarded = int((~channel_valid).sum())

    quantized, scale = to_int16(stacked)

    out_dir = out_root / f"s{subject:02d}" / f"session{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = out_dir / "data_int16.npy"
    np.save(memmap_path, quantized)

    montage = build_montage_metadata(subject=subject, session=session)
    montage_dict = {
        "dataset_name": montage.dataset_name, "subject_id": montage.subject_id,
        "session_id": montage.session_id, "day_index": montage.day_index,
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
        "montage": montage_dict, "shape": list(quantized.shape), "native_fs_hz": NATIVE_FS_HZ,
        "int16_scale": scale, "trials": trial_meta, "truncated_to_samples": min_len,
        "n_channels_discarded_by_qc": n_channels_discarded,
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))

    return {
        "subject": subject, "session": session, "n_trials": len(all_trials),
        "n_channels_discarded": n_channels_discarded,
        "hours": len(all_trials) * min_len / NATIVE_FS_HZ / 3600,
        "memmap_path": str(memmap_path), "completeness_warning": completeness_warning,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--zip-path", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--sessions", type=int, nargs="+", required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    per_unit = []
    for subject in args.subjects:
        for session in args.sessions:
            print(f"soggetto {subject}, sessione {session}: ingest in corso...")
            per_unit.append(ingest_subject_session(args.zip_path, args.out_root, subject, session))
            print(f"  fatto: {per_unit[-1]}")
    elapsed = time.time() - t0

    report = {
        "dataset": "csl_hdemg",
        "subjects_processed": args.subjects, "sessions_processed": args.sessions,
        "n_units_processed": len(per_unit),
        "n_trials_total": sum(u["n_trials"] for u in per_unit),
        "hours_total": sum(u["hours"] for u in per_unit),
        "n_channels": N_CHANNELS_EMG, "per_unit": per_unit, "elapsed_s": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
