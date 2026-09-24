from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from wearusfm.ingest.putemg import (
    GESTURE_MAP,
    PAUSE_LABEL,
    VALUES_BLOCK_3_COLUMNS,
    build_montage_metadata,
    load_record,
    qc_channel_validity,
    scan_putemg,
    to_int16,
)


def _write_synthetic_hdf5(
    path: Path, *, n_samples: int = 50, subject: int = 3,
    experiment_type: str = "emg_gestures", trajectory_type: str = "sequential",
) -> None:
    """Riproduce la struttura PyTables reale (verificata in sessione, 24/09/2026) con
    h5py puro: dtype strutturato, colonne categoriali con lookup table separata."""
    rng = np.random.default_rng(0)
    row_dtype = np.dtype([
        ("index", "<f8"),
        ("values_block_0", "i1", (1,)),
        ("values_block_1", "i1", (1,)),
        ("values_block_2", "<i8", (1,)),
        ("values_block_3", "<f8", (29,)),
    ])
    rows = np.zeros(n_samples, dtype=row_dtype)
    rows["index"] = np.arange(n_samples, dtype=np.float64)
    rows["values_block_0"][:, 0] = 0  # 'emg_gestures'
    rows["values_block_1"][:, 0] = 2  # 'sequential'
    rows["values_block_2"][:, 0] = 1526037050475000000 + np.arange(n_samples)

    values = np.zeros((n_samples, 29), dtype=np.float64)
    values[:, :24] = rng.standard_normal((n_samples, 24)) * 10  # EMG_1..24
    col = {name: i for i, name in enumerate(VALUES_BLOCK_3_COLUMNS)}
    values[:, col["subject"]] = subject
    # prima meta' pausa, seconda meta' gesto "Fist" (1)
    traj_gt = np.array([PAUSE_LABEL] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    values[:, col["TRAJ_GT"]] = traj_gt
    rows["values_block_3"] = values

    with h5py.File(path, "w") as f:
        f.create_dataset("data/table", data=rows)
        exp_dtype = np.dtype([("index", "i8"), ("values", "S20")])
        exp_meta = np.array([(0, b"emg_gestures"), (1, b"emg_force"), (2, b"mmg_gestures")], dtype=exp_dtype)
        f.create_dataset("data/meta/values_block_0/meta/table", data=exp_meta)
        traj_dtype = np.dtype([("index", "i8"), ("values", "S20")])
        traj_meta = np.array([
            (0, b"familiarization"), (1, b"repeats_long"), (2, b"sequential"),
            (3, b"repeats_short"), (4, b"bias"), (5, b"mvc"),
        ], dtype=traj_dtype)
        f.create_dataset("data/meta/values_block_1/meta/table", data=traj_meta)


def test_scan_putemg_parses_filenames(tmp_path: Path) -> None:
    (tmp_path / "emg_gestures-03-sequential-2018-05-11-11-10-50-475.hdf5").touch()
    (tmp_path / "emg_gestures-34-repeats_long-2018-04-20-12-14-50-393.hdf5").touch()
    (tmp_path / "not_a_match.hdf5").touch()

    files = scan_putemg(tmp_path)

    assert len(files) == 2
    assert files[0].participant == 3
    assert files[1].participant == 34 and files[1].trajectory == "repeats_long"


def test_load_record_decodes_categories_and_emg(tmp_path: Path) -> None:
    f = tmp_path / "record.hdf5"
    _write_synthetic_hdf5(f, n_samples=50, subject=3)

    record = load_record(f)

    assert record.emg.shape == (50, 24)
    assert record.subject == 3
    assert record.experiment_type == "emg_gestures"
    assert record.trajectory_type == "sequential"
    assert (record.traj_gt[:25] == PAUSE_LABEL).all()
    assert (record.traj_gt[25:] == 1).all()
    assert GESTURE_MAP[1] == "Fist"


def test_load_record_raises_on_non_constant_subject(tmp_path: Path) -> None:
    f = tmp_path / "record.hdf5"
    _write_synthetic_hdf5(f, n_samples=10, subject=3)
    # corrompi la colonna subject a meta' file: read-modify-write dell'intero dataset,
    # l'indicizzazione per campo+slice di h5py su un compound dataset ritorna una copia
    with h5py.File(f, "r+") as h5:
        col = {name: i for i, name in enumerate(VALUES_BLOCK_3_COLUMNS)}
        rows = h5["data/table"][:]
        rows["values_block_3"][5:, col["subject"]] = 99
        h5["data/table"][:] = rows

    with pytest.raises(ValueError, match="subject"):
        load_record(f)


def test_qc_channel_validity_flags_flat_channel() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 4)) * 10
    data[:, 3] = 0.0

    valid = qc_channel_validity(data)

    assert valid.tolist() == [True, True, True, False]


def test_to_int16_roundtrip_preserves_peaks() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 24)) * 100

    quantized, scale = to_int16(data)
    reconstructed = quantized.astype(np.float64) / scale

    assert quantized.dtype == np.int16
    assert np.abs(quantized).max() <= 32767
    assert np.max(np.abs(reconstructed - data)) < np.abs(data).max() / 30000  # errore di quantizzazione, non clipping


def test_build_montage_metadata_has_three_bands() -> None:
    montage = build_montage_metadata(participant=3, trajectory="sequential", timestamp="2018-05-11-11-10-50-475")

    assert montage.dataset_name == "putemg"
    assert montage.subject_id == "putemg_p03"
    assert montage.n_channels == 24
    assert [g.group_id for g in montage.groups] == ["band1", "band2", "band3"]
    assert all(g.topology.value == "ring" and g.symmetry == "D_8" for g in montage.groups)
