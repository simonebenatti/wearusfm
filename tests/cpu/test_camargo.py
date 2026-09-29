from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.ingest.camargo import (
    EMG_COLUMNS,
    N_CHANNELS,
    TIME_COLUMN,
    build_montage_metadata,
    check_time_axis,
    load_emg_file,
    qc_channel_validity,
    scan_camargo,
    to_int16,
)


def _write_converted_mat(path: Path, n_samples: int = 100, colnames: list[str] | None = None) -> None:
    """Riproduce il formato di output di scripts/convert_camargo_mcos_tables.m
    (verificato su un file reale, 29/09/2026): `colnames` (cell array di stringhe) e
    `data` (matrice double, prima colonna = Header/tempo in secondi a 1 kHz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = colnames if colnames is not None else [TIME_COLUMN, *EMG_COLUMNS]
    data = np.zeros((n_samples, len(names)))
    data[:, 0] = 14.42 + np.arange(n_samples) * 0.001
    data[:, 1:] = rng.standard_normal((n_samples, len(names) - 1)) * 0.02
    cell = np.empty((1, len(names)), dtype=object)
    for i, n in enumerate(names):
        cell[0, i] = n
    sio.savemat(path, {"colnames": cell, "data": data})


def test_scan_camargo_parses_structure(tmp_path: Path) -> None:
    _write_converted_mat(tmp_path / "part1" / "AB06" / "10_09_18" / "levelground" / "emg" / "levelground_ccw_fast_01_01.mat")
    _write_converted_mat(tmp_path / "part3" / "AB30" / "03_09_2019" / "stair" / "emg" / "stair_1_r_01_01.mat")
    _write_converted_mat(tmp_path / "part1" / "AB06" / "10_09_18" / "levelground" / "imu" / "not_emg.mat")  # ignorato

    files = scan_camargo(tmp_path)

    assert len(files) == 2
    by_subject = {f.subject: f for f in files}
    assert by_subject[6].date == "10_09_18" and by_subject[6].activity == "levelground"
    assert by_subject[30].activity == "stair" and by_subject[30].trial_name == "stair_1_r_01_01"


def test_load_emg_file_splits_time_and_channels(tmp_path: Path) -> None:
    f = tmp_path / "trial.mat"
    _write_converted_mat(f, n_samples=100)

    emg, time_s = load_emg_file(f)

    assert emg.shape == (100, N_CHANNELS)
    assert time_s.shape == (100,)
    assert time_s[0] == pytest.approx(14.42)


def test_load_emg_file_raises_on_unexpected_columns(tmp_path: Path) -> None:
    f = tmp_path / "trial.mat"
    _write_converted_mat(f, colnames=[TIME_COLUMN, "gastrocmed", "soleus"])

    with pytest.raises(ValueError, match="colonne"):
        load_emg_file(f)


def test_load_emg_file_raises_on_wrong_column_order(tmp_path: Path) -> None:
    f = tmp_path / "trial.mat"
    swapped = [TIME_COLUMN, *reversed(EMG_COLUMNS)]
    _write_converted_mat(f, colnames=swapped)

    with pytest.raises(ValueError, match="colonne"):
        load_emg_file(f)


def test_check_time_axis_accepts_1khz() -> None:
    assert check_time_axis(np.arange(100) * 0.001)


def test_check_time_axis_rejects_other_rates() -> None:
    assert not check_time_axis(np.arange(100) * 0.005)  # 200 Hz, non 1000
    assert not check_time_axis(np.array([1.0]))


def test_qc_channel_validity_flags_flat_channel() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 3)) * 0.02
    data[:, 1] = 0.0

    assert qc_channel_validity(data).tolist() == [True, False, True]


def test_to_int16_preserves_peaks() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 11)) * 0.02

    quantized, scale = to_int16(data)
    reconstructed = quantized.astype(np.float64) / scale

    assert quantized.dtype == np.int16
    assert np.max(np.abs(reconstructed - data)) < np.abs(data).max() / 30000


def test_build_montage_metadata_uses_muscle_level_taxonomy() -> None:
    montage = build_montage_metadata(subject=6, date="10_09_18")

    assert montage.dataset_name == "camargo2021"
    assert montage.subject_id == "camargo_ab06"
    assert montage.n_channels == 11
    group = montage.groups[0]
    assert group.topology.value == "sparse" and group.symmetry == "none"
    # tutti i canali a livello MUSCOLO (v10 §4.5: il test su Camargo lo richiede)
    assert all(c.anatomical_identity.precision.value == "muscle" for c in group.channels)
    by_muscle = {c.anatomical_identity.muscle: c for c in group.channels}
    assert by_muscle["EO"].anatomical_identity.region == "trunk"  # D7b: non arto inferiore
    assert by_muscle["VM"].anatomical_identity.region == "lower_limb"
    assert by_muscle["VM"].anatomical_identity.muscle_ontology_id == "UBERON:0001380"
