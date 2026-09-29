"""Pezzi condivisi dagli ingest scritti dopo i primi cinque (NinaPro DB5, Kaifosh, ...).

I primi cinque script (capgmyo, grabmyo, putemg, csl_hdemg, camargo) hanno ciascuno la propria
copia della conversione montaggio -> dizionario JSON: non si toccano (sono in produzione e
verificati). I nuovi usano questa.
"""

from __future__ import annotations

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
