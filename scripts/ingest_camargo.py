#!/usr/bin/env python3
"""Ingest di Camargo 2021 (passo 2, piano_operativo_v10.md §11), dal formato convertito
con MATLAB (scripts/convert_camargo_mcos_tables.m).

Per soggetto+data (una sessione): tutti i trial `emg` di tutte le attivita' in un unico
array int16 concatenato (T_totale, 11) piu' un indice dei trial (attivita', nome, offset,
lunghezza) nel sidecar - i trial hanno lunghezze molto diverse (secondi..minuti), niente
troncamento a lunghezza comune come per i dataset a trial di durata fissa. Scala int16
CONDIVISA fra tutti i trial della sessione.

Uso:
    python scripts/ingest_camargo.py --raw-root $WORK/data/raw/camargo2021_emg_converted \\
        --out-root $WORK/data/processed/camargo2021 --subjects 6 \\
        --report $WORK/wearusfm_runs/results/passo2/camargo_ingest_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.ingest.camargo import (  # noqa: E402
    N_CHANNELS,
    NATIVE_FS_HZ,
    build_montage_metadata,
    check_time_axis,
    load_emg_file,
    qc_channel_validity,
    scan_camargo,
    to_int16,
)
from wearusfm.metadata.schema import export_json_schema  # noqa: E402


def _validate_against_schema(montage_dict: dict) -> None:
    schema = export_json_schema()
    for field in schema["required"]:
        if field not in montage_dict:
            raise ValueError(f"metadati non conformi allo schema: manca {field!r}")


def _montage_to_dict(montage, channel_valid: np.ndarray) -> dict:
    return {
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


def ingest_session(files: list, out_root: Path, subject: int, date: str) -> dict:
    files = sorted(files, key=lambda f: (f.activity, f.trial_name))
    trials, trial_index, offset = [], [], 0
    n_bad_time = 0
    for ef in files:
        emg, time_s = load_emg_file(ef.path)
        if not check_time_axis(time_s):
            n_bad_time += 1
        trials.append(emg)
        trial_index.append({"activity": ef.activity, "trial": ef.trial_name,
                            "offset": offset, "n_samples": int(emg.shape[0])})
        offset += emg.shape[0]

    concatenated = np.concatenate(trials, axis=0)  # (T_totale, 11)
    channel_valid = qc_channel_validity(concatenated)
    n_channels_discarded = int((~channel_valid).sum())
    quantized, scale = to_int16(concatenated)

    out_dir = out_root / f"ab{subject:02d}" / date
    out_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = out_dir / "data_int16.npy"
    np.save(memmap_path, quantized)

    montage = build_montage_metadata(subject=subject, date=date)
    montage_dict = _montage_to_dict(montage, channel_valid)
    _validate_against_schema(montage_dict)

    (out_dir / "metadata.json").write_text(json.dumps({
        "montage": montage_dict, "shape": list(quantized.shape), "native_fs_hz": NATIVE_FS_HZ,
        "int16_scale": scale, "trials": trial_index,
        "n_channels_discarded_by_qc": n_channels_discarded,
    }, indent=2))

    return {
        "subject": subject, "date": date, "n_trials": len(files),
        "n_samples": int(quantized.shape[0]), "hours": quantized.shape[0] / NATIVE_FS_HZ / 3600,
        "n_channels_discarded": n_channels_discarded, "n_trials_bad_time_axis": n_bad_time,
        "memmap_path": str(memmap_path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=None, help="default: tutti quelli presenti")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    all_files = scan_camargo(args.raw_root)
    by_session: dict[tuple[int, str], list] = defaultdict(list)
    for ef in all_files:
        if args.subjects is None or ef.subject in args.subjects:
            by_session[(ef.subject, ef.date)].append(ef)
    if not by_session:
        raise FileNotFoundError(f"nessun file emg trovato sotto {args.raw_root} per i soggetti richiesti")

    t0 = time.time()
    per_session = []
    for (subject, date), files in sorted(by_session.items()):
        print(f"AB{subject:02d} {date}: {len(files)} trial, ingest in corso...")
        per_session.append(ingest_session(files, args.out_root, subject, date))
        print(f"  fatto: {per_session[-1]}")
    elapsed = time.time() - t0

    report = {
        "dataset": "camargo2021",
        "subjects_processed": sorted({s["subject"] for s in per_session}),
        "n_sessions": len(per_session),
        "n_trials_total": sum(s["n_trials"] for s in per_session),
        "hours_total": sum(s["hours"] for s in per_session),
        "n_channels": N_CHANNELS, "per_session": per_session, "elapsed_s": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_session"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
