from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wearusfm.ingest.capgmyo import (
    build_montage_metadata,
    check_completeness,
    qc_channel_validity,
    scan_capgmyo,
    to_int16,
    verify_mat_consistency,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_scan_capgmyo_parses_filenames(tmp_path: Path) -> None:
    _touch(tmp_path / "s1" / "001-001-001.mat")
    _touch(tmp_path / "s1" / "001-008-010.mat")
    _touch(tmp_path / "s1" / "not_a_trial_file.mat")  # ignorato, non combacia il pattern

    files = scan_capgmyo(tmp_path)

    assert len(files) == 2
    assert files[0].subject == 1 and files[0].gesture == 1 and files[0].trial == 1
    assert files[1].gesture == 8 and files[1].trial == 10


def test_check_completeness_reports_missing(tmp_path: Path) -> None:
    _touch(tmp_path / "001-001-001.mat")
    files = scan_capgmyo(tmp_path)

    warnings = check_completeness(files)

    assert len(warnings) == 1
    assert "18×8×10" not in warnings[0]  # non e' un conteggio totale, solo mancanti
    assert "144" in warnings[0] or "mancanti" in warnings[0]  # 144 = 18*8*10 - 1


def test_check_completeness_empty_when_full(tmp_path: Path) -> None:
    for s in range(1, 19):
        for g in range(1, 9):
            for t in range(1, 11):
                _touch(tmp_path / f"{s:03d}-{g:03d}-{t:03d}.mat")
    files = scan_capgmyo(tmp_path)

    assert check_completeness(files) == []


def test_verify_mat_consistency_raises_on_mismatch() -> None:
    from wearusfm.ingest.capgmyo import TrialFile

    trial_file = TrialFile(path=Path("001-002-003.mat"), subject=1, gesture=2, trial=3)
    bad_dict = {"subject": np.array([[1]]), "gesture": np.array([[9]]), "trial": np.array([[3]])}

    with pytest.raises(ValueError, match="gesture"):
        verify_mat_consistency(bad_dict, trial_file)


def test_verify_mat_consistency_passes_on_match() -> None:
    from wearusfm.ingest.capgmyo import TrialFile

    trial_file = TrialFile(path=Path("001-002-003.mat"), subject=1, gesture=2, trial=3)
    good_dict = {"subject": np.array([[1]]), "gesture": np.array([[2]]), "trial": np.array([[3]])}

    verify_mat_consistency(good_dict, trial_file)  # non deve sollevare


def test_qc_channel_validity_flags_flat_channel() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 4))
    data[:, 2] = 0.0  # canale piatto, elettrodo scollegato

    valid = qc_channel_validity(data)

    assert valid.tolist() == [True, True, False, True]


def test_qc_channel_validity_flags_clipped_channel() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 2)) * 0.01
    data[:, 1] = 5.0  # saturato: quasi tutti i campioni al massimo assoluto del canale

    valid = qc_channel_validity(data)

    assert valid[0]
    assert not valid[1]


def test_to_int16_roundtrip_within_quantization_error() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 128)) * 1e-4  # scala tipica EMG raw, volt

    quantized, scale = to_int16(data)
    reconstructed = quantized.astype(np.float64) / scale

    assert quantized.dtype == np.int16
    assert np.max(np.abs(reconstructed - data)) < 1e-6


def test_to_int16_does_not_saturate_on_extreme_outlier() -> None:
    data = np.zeros((100, 2))
    data[0, 0] = 100.0  # outlier estremo isolato

    quantized, scale = to_int16(data)

    assert np.abs(quantized).max() <= 32767


def test_build_montage_metadata_shape_and_topology() -> None:
    montage = build_montage_metadata(subject=5)

    assert montage.dataset_name == "capgmyo_dba"
    assert montage.subject_id == "capgmyo_s05"
    assert montage.n_channels == 128
    group = montage.groups[0]
    assert group.topology.value == "grid_2d"
    assert group.channels[0].sensor_coords.grid_row == 0
    assert group.channels[0].sensor_coords.grid_col == 0
    assert group.channels[16].sensor_coords.grid_row == 1
    assert group.channels[16].sensor_coords.grid_col == 0
