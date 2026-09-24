from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from wearusfm.ingest.csl_hdemg import (
    GRID_COLS,
    GRID_ROWS_EMG,
    N_CHANNELS_EMG,
    N_CHANNELS_RAW,
    build_montage_metadata,
    filter_differential_channels,
    load_gest_file,
    qc_channel_validity,
    scan_csl_hdemg,
    to_int16,
)


def _make_gest_mat_bytes(n_reps: int, n_samples: int, rng: np.random.Generator) -> bytes:
    """Cell array 'gestures' (n_reps, 1), ogni cella (192, n_samples) - come nel file
    .mat reale del dataset."""
    cells = np.empty((n_reps, 1), dtype=object)
    for i in range(n_reps):
        cells[i, 0] = rng.standard_normal((N_CHANNELS_RAW, n_samples))
    buf = io.BytesIO()
    sio.savemat(buf, {"gestures": cells})
    return buf.getvalue()


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


def test_scan_csl_hdemg_parses_members(tmp_path: Path) -> None:
    zip_path = tmp_path / "csl_hdemg.zip"
    _write_zip(zip_path, {
        "subject1/session1/gest0.mat": b"x",
        "subject1/session1/gest11.mat": b"x",
        "subject5/session3/gest26.mat": b"x",
        "README.txt": b"not a gesture file",
        "src/example.py": b"not a gesture file",
    })

    files = scan_csl_hdemg(zip_path)

    assert len(files) == 3
    assert {(f.subject, f.session, f.gesture) for f in files} == {(1, 1, 0), (1, 1, 11), (5, 3, 26)}


def test_load_gest_file_reads_cell_array(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    mat_bytes = _make_gest_mat_bytes(n_reps=10, n_samples=6144, rng=rng)
    zip_path = tmp_path / "csl_hdemg.zip"
    _write_zip(zip_path, {"subject5/session3/gest11.mat": mat_bytes})

    trials = load_gest_file(zip_path, "subject5/session3/gest11.mat")

    assert len(trials) == 10
    assert all(t.shape == (192, 6144) for t in trials)


def test_load_gest_file_raises_on_wrong_channel_count(tmp_path: Path) -> None:
    cells = np.empty((1, 1), dtype=object)
    cells[0, 0] = np.zeros((100, 50))  # canali sbagliati
    buf = io.BytesIO()
    sio.savemat(buf, {"gestures": cells})
    zip_path = tmp_path / "csl_hdemg.zip"
    _write_zip(zip_path, {"subject1/session1/gest0.mat": buf.getvalue()})

    with pytest.raises(ValueError, match="canali"):
        load_gest_file(zip_path, "subject1/session1/gest0.mat")


def test_filter_differential_channels_matches_official_example() -> None:
    """Riproduce esattamente np.delete(trial, np.s_[7:192:8], 0) di src/example.py
    (letto alla lettera dal dataset ufficiale, non riassunto)."""
    trial = np.arange(192 * 3).reshape(192, 3)

    filtered = filter_differential_channels(trial)

    assert filtered.shape == (168, 3)
    expected = np.delete(trial, np.s_[7:192:8], 0)
    np.testing.assert_array_equal(filtered, expected)
    # i canali scartati sono 8, 16, 24, ... (1-based) cioe' righe 7, 15, 23 (0-based)
    assert not np.any(np.all(filtered == trial[7], axis=1))


def test_qc_channel_validity_flags_flat_channel_axis1() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1000))  # (C, T) - convenzione nativa CSL
    data[2, :] = 0.0

    valid = qc_channel_validity(data)

    assert valid.tolist() == [True, True, False, True]


def test_to_int16_preserves_peaks() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((168, 6144)) * 50

    quantized, scale = to_int16(data)
    reconstructed = quantized.astype(np.float64) / scale

    assert quantized.dtype == np.int16
    assert np.abs(quantized).max() <= 32767
    assert np.max(np.abs(reconstructed - data)) < np.abs(data).max() / 30000


def test_build_montage_metadata_single_hd_grid() -> None:
    montage = build_montage_metadata(subject=5, session=3)

    assert montage.dataset_name == "csl_hdemg"
    assert montage.subject_id == "csl_s05"
    assert montage.n_channels == N_CHANNELS_EMG
    group = montage.groups[0]
    assert group.topology.value == "grid_2d"
    assert group.channels[0].sensor_coords.grid_col == 0
    assert group.channels[0].sensor_coords.grid_row == 0
    assert group.channels[GRID_ROWS_EMG].sensor_coords.grid_col == 1
    assert group.channels[GRID_ROWS_EMG].sensor_coords.grid_row == 0
    assert len(group.channels) == GRID_COLS * GRID_ROWS_EMG
