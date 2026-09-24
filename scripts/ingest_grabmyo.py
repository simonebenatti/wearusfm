#!/usr/bin/env python3
"""Ingest di GRABMyo (passo 2, piano_operativo_v10.md §11).

Per soggetto+sessione: tutti i 17 gesti x 7 trial (119 record) in un unico array
int16 (119, 10240, 28) - nessuna quantizzazione necessaria, il formato WFDB
sorgente e' gia' int16 (a differenza di CapgMyo, che partiva da float64).

Uso:
    python scripts/ingest_grabmyo.py --raw-root $WORK/data/raw/grabmyo \\
        --out-root $WORK/data/processed/grabmyo --participants 1 --sessions 1 \\
        --report $WORK/wearusfm_runs/results/passo2/grabmyo_ingest_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.grabmyo import (  # noqa: E402
    N_GESTURES,
    N_TRIALS_PER_GESTURE,
    NATIVE_FS_HZ,
    build_montage_metadata,
    load_dat,
    parse_hea,
    qc_channel_validity,
    scan_grabmyo,
    select_emg_channels,
)
from wearusfm.metadata.schema import export_json_schema  # noqa: E402


def _validate_against_schema(montage_dict: dict) -> None:
    schema = export_json_schema()
    for field in schema["required"]:
        if field not in montage_dict:
            raise ValueError(f"metadati non conformi allo schema: manca {field!r}")


def ingest_participant_session(raw_root: Path, out_root: Path, participant: int, session: int) -> dict:
    files = [f for f in scan_grabmyo(raw_root) if f.participant == participant and f.session == session]
    if not files:
        raise FileNotFoundError(
            f"nessun file per partecipante {participant} sessione {session} sotto {raw_root}"
        )
    files.sort(key=lambda f: (f.gesture, f.trial))
    expected = N_GESTURES * N_TRIALS_PER_GESTURE
    completeness_warning = None
    if len(files) != expected:
        completeness_warning = f"attesi {expected} record, trovati {len(files)}"

    records = []
    trial_meta = []
    channel_names: list[str] | None = None
    for tf in files:
        header = parse_hea(tf.hea_path)
        data = load_dat(header, tf.dat_path)
        emg_data, names = select_emg_channels(header, data)
        if channel_names is None:
            channel_names = names
        elif names != channel_names:
            raise ValueError(f"{tf.hea_path}: ordine canali diverso dal resto del soggetto")
        records.append(emg_data)
        trial_meta.append({"gesture": tf.gesture, "trial": tf.trial})

    stacked = np.stack(records)  # (n_record, 10240, 28), int16 nativo

    per_record_valid = np.stack([qc_channel_validity(d) for d in records])
    channel_valid = per_record_valid.mean(axis=0) >= 0.5
    n_channels_discarded = int((~channel_valid).sum())

    out_dir = out_root / f"p{participant:02d}" / f"session{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = out_dir / "data_int16.npy"
    np.save(memmap_path, stacked)

    montage = build_montage_metadata(participant=participant, session=session)
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
                "qc_valid": bool(channel_valid[i]),
                "body_region": c.body_region,
            } for i, c in enumerate(g.channels)],
        } for g in montage.groups],
    }
    _validate_against_schema(montage_dict)

    sidecar = {
        "montage": montage_dict, "shape": list(stacked.shape), "native_fs_hz": NATIVE_FS_HZ,
        "channel_names": channel_names, "trials": trial_meta,
        "n_channels_discarded_by_qc": n_channels_discarded,
    }
    (out_dir / "metadata.json").write_text(json.dumps(sidecar, indent=2))

    return {
        "participant": participant, "session": session, "n_records": len(files),
        "n_channels_discarded": n_channels_discarded,
        "hours": len(files) * (stacked.shape[1] / NATIVE_FS_HZ) / 3600,
        "memmap_path": str(memmap_path), "completeness_warning": completeness_warning,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--participants", type=int, nargs="+", required=True)
    p.add_argument("--sessions", type=int, nargs="+", required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    per_unit = []
    for participant in args.participants:
        for session in args.sessions:
            print(f"partecipante {participant}, sessione {session}: ingest in corso...")
            per_unit.append(ingest_participant_session(args.raw_root, args.out_root, participant, session))
            print(f"  fatto: {per_unit[-1]}")
    elapsed = time.time() - t0

    report = {
        "dataset": "grabmyo",
        "participants_processed": args.participants,
        "sessions_processed": args.sessions,
        "n_units_processed": len(per_unit),
        "n_records_total": sum(u["n_records"] for u in per_unit),
        "hours_total": sum(u["hours"] for u in per_unit),
        "n_channels": 28,
        "per_unit": per_unit,
        "elapsed_s": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
