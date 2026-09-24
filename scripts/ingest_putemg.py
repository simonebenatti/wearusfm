#!/usr/bin/env python3
"""Ingest di putEMG (solo emg_gestures/HDF5, passo 2, piano_operativo_v10.md §11).

Per record (una sessione continua): un array int16 (T, 24) + array delle etichette
TRAJ_GT per campione (T,) - niente segmentazione in trial qui, e' compito
dell'harness (a differenza di CapgMyo/GRABMyo, gia' pre-segmentati alla fonte).

Uso:
    python scripts/ingest_putemg.py --raw-root $WORK/data/raw/putemg/data-hdf5 \\
        --out-root $WORK/data/processed/putemg --limit 1 \\
        --report $WORK/wearusfm_runs/results/passo2/putemg_ingest_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.putemg import (  # noqa: E402
    NATIVE_FS_HZ,
    build_montage_metadata,
    load_record,
    qc_channel_validity,
    scan_putemg,
    to_int16,
)
from wearusfm.metadata.schema import export_json_schema  # noqa: E402


def _validate_against_schema(montage_dict: dict) -> None:
    schema = export_json_schema()
    for field in schema["required"]:
        if field not in montage_dict:
            raise ValueError(f"metadati non conformi allo schema: manca {field!r}")


def ingest_record(raw_root: Path, out_root: Path, rf) -> dict:
    record = load_record(rf.path)
    if record.experiment_type != "emg_gestures":
        raise ValueError(f"{rf.path}: experiment_type={record.experiment_type!r}, atteso 'emg_gestures'")

    channel_valid = qc_channel_validity(record.emg)
    n_channels_discarded = int((~channel_valid).sum())

    quantized, scale = to_int16(record.emg)

    out_dir = out_root / f"p{rf.participant:02d}" / f"{rf.trajectory}_{rf.timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = out_dir / "data_int16.npy"
    np.save(memmap_path, quantized)
    np.save(out_dir / "traj_gt.npy", record.traj_gt)

    montage = build_montage_metadata(participant=rf.participant, trajectory=rf.trajectory, timestamp=rf.timestamp)
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
                "qc_valid": bool(channel_valid[8 * gi + i]),
                "body_region": c.body_region,
            } for i, c in enumerate(g.channels)],
        } for gi, g in enumerate(montage.groups)],
    }
    _validate_against_schema(montage_dict)

    label_counts = {str(k): int(v) for k, v in zip(*np.unique(record.traj_gt, return_counts=True))}
    sidecar = {
        "montage": montage_dict, "shape": list(quantized.shape), "native_fs_hz": NATIVE_FS_HZ,
        "int16_scale": scale, "trajectory_type": record.trajectory_type,
        "label_counts_traj_gt": label_counts, "n_channels_discarded_by_qc": n_channels_discarded,
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))

    return {
        "participant": rf.participant, "trajectory": rf.trajectory, "timestamp": rf.timestamp,
        "n_samples": int(quantized.shape[0]), "n_channels_discarded": n_channels_discarded,
        "hours": quantized.shape[0] / NATIVE_FS_HZ / 3600, "memmap_path": str(memmap_path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="processa solo i primi N file (collaudo)")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    files = scan_putemg(args.raw_root)
    if args.limit is not None:
        files = files[: args.limit]

    t0 = time.time()
    per_record = []
    failures = []
    for rf in files:
        print(f"{rf.path.name}: ingest in corso...")
        try:
            per_record.append(ingest_record(args.raw_root, args.out_root, rf))
            print(f"  fatto: {per_record[-1]}")
        except Exception as e:  # noqa: BLE001 - un file non buono non deve fermare gli altri
            print(f"  FALLITO: {e}", file=sys.stderr)
            failures.append({"file": rf.path.name, "error": str(e)})
    elapsed = time.time() - t0

    report = {
        "dataset": "putemg", "n_files_found": len(files), "n_files_processed": len(per_record),
        "n_files_failed": len(failures), "failures": failures,
        "n_samples_total": sum(r["n_samples"] for r in per_record),
        "hours_total": sum(r["hours"] for r in per_record),
        "n_channels": 24, "per_record": per_record, "elapsed_s": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_record"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
