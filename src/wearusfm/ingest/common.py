"""Pezzi condivisi dagli ingest scritti dopo i primi cinque (NinaPro DB5, Kaifosh, ...).

I primi cinque script (capgmyo, grabmyo, putemg, csl_hdemg, camargo) hanno ciascuno la propria
copia della conversione montaggio -> dizionario JSON: non si toccano (sono in produzione e
verificati). I nuovi usano questa.
"""

from __future__ import annotations

import numpy as np

from wearusfm.metadata.schema import MontageMetadata, export_json_schema


def montage_to_dict(montage: MontageMetadata, channel_valid) -> dict:
    """Come nei sidecar esistenti: stesse chiavi, qc_valid per canale nell'ORDINE di scrittura
    (gruppi, poi canali), che e' anche l'ordine delle colonne dell'array."""
    flat = 0
    groups = []
    for g in montage.groups:
        channels = []
        for c in g.channels:
            channels.append({
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
                "qc_valid": bool(channel_valid[flat]),
                "body_region": c.body_region,
            })
            flat += 1
        groups.append({
            "group_id": g.group_id, "topology": g.topology.value, "symmetry": g.symmetry,
            "band_orientation_deg": g.band_orientation_deg, "channels": channels,
        })
    if flat != len(channel_valid):
        raise ValueError(f"{flat} canali nel montaggio, {len(channel_valid)} flag QC")
    return {
        "dataset_name": montage.dataset_name, "subject_id": montage.subject_id,
        "session_id": montage.session_id, "day_index": montage.day_index, "groups": groups,
    }


def validate_montage_dict(montage_dict: dict) -> None:
    """Validazione leggera senza jsonschema (non nell'ambiente): controlla i campi required di
    primo livello. Intercetta un errore grossolano, non e' una validazione completa."""
    schema = export_json_schema()
    for field in schema["required"]:
        if field not in montage_dict:
            raise ValueError(f"metadati non conformi allo schema: manca {field!r}")


# --- pezzi condivisi dagli ingest NinaPro DB2/3/4/6/7, Hyser, emg2qwerty, emg2pose ------------------------------------------------------

MAX_LEN_DIFF = 5
MAX_LEN_DIFF_FRACTION = 0.01
LABEL_PAD = -1  # campione di EMG senza etichetta


def relative_min_std(emg: np.ndarray) -> float:
    """Soglia di canale piatto RELATIVA: `1e-3 x` la mediana delle deviazioni standard dei canali (con un minimo assoluto di 1e-12).
    Serve quando le unita' cambiano da dataset a dataset (conteggi ~1e3 vs volt ~1e-5) e una soglia assoluta non ha senso."""
    return relative_min_std_from_std(emg.std(axis=0))


def relative_min_std_from_std(std: np.ndarray) -> float:
    """Come `relative_min_std`, ma dalle deviazioni standard dei canali gia' calcolate (serve a chi non puo' tenere i dati in memoria)."""
    return max(1e-12, 1e-3 * float(np.median(std)))


def reconcile_labels(mat: dict, n_emg: int, names) -> tuple[dict, dict]:
    """Etichette per campione (`mat[nome]`) riportate alla lunghezza dell'EMG. L'EMG non si tocca mai: un'etichetta piu' corta si
    riempie con `LABEL_PAD` in coda, una piu' lunga si taglia in coda; oltre `max(MAX_LEN_DIFF, MAX_LEN_DIFF_FRACTION x n)` campioni
    di differenza e' un errore (i due file non sono piu' allineati in modo credibile). Ritorna (etichette int16, {nome: differenza})."""
    tol = max(MAX_LEN_DIFF, int(MAX_LEN_DIFF_FRACTION * n_emg))
    labels, adjustments = {}, {}
    for name in names:
        if name not in mat:
            continue
        arr = np.asarray(mat[name]).ravel().astype(np.int16)
        d = arr.shape[0] - n_emg
        if abs(d) > tol:
            raise ValueError(f"lunghezze incompatibili: emg {n_emg}, {name} {arr.shape[0]} (tolleranza {tol})")
        if d < 0:
            arr = np.concatenate([arr, np.full(-d, LABEL_PAD, dtype=np.int16)])
        elif d > 0:
            arr = arr[:n_emg]
        if d:
            adjustments[name] = int(d)
        labels[name] = arr
    return labels, adjustments
