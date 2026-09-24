"""Test della logica di caricamento/finestramento di bench/harness_smoke_uci_emg.py, con
fixture .txt sintetici che riproducono il formato reale (verificato in sessione,
23/09/2026: tab-separated, colonne [time, ch1..8, class], header)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bench"))

from harness_smoke_uci_emg import _load_subject_file, load_uci_emg, windowize  # noqa: E402


def _write_subject_file(path: Path, labels: list[int], rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["time\tchannel1\tchannel2\tchannel3\tchannel4\tchannel5\tchannel6\tchannel7\tchannel8\tclass"]
    for i, label in enumerate(labels):
        channels = rng.standard_normal(8) * 1e-5
        row = [str(i)] + [f"{c:.6f}" for c in channels] + [str(label)]
        lines.append("\t".join(row))
    path.write_text("\n".join(lines))


def test_load_subject_file_parses_columns(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    f = tmp_path / "1_raw_data.txt"
    _write_subject_file(f, [0, 0, 1, 1, 1], rng)

    samples, labels = _load_subject_file(f)

    assert samples.shape == (5, 8)
    assert labels.tolist() == [0, 0, 1, 1, 1]


def test_windowize_drops_unmarked_and_mixed_windows() -> None:
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((20, 8))
    # 0..4 unmarked, 5..14 classe 2, 15..19 classe 3 (finestra 10..19 mista 2/3)
    labels = np.array([0] * 5 + [2] * 10 + [3] * 5)

    windows, window_labels = windowize(samples, labels, window_samples=5, stride_samples=5)

    # attese: [5:10] classe 2, [10:15] resta classe 2 pura -> ok; [0:5] scartata (unmarked);
    # [15:20] classe 3 pura -> ok
    assert window_labels == [2, 2, 3]
    assert all(w.shape == (5, 8) for w in windows)


def test_windowize_empty_when_all_unmarked() -> None:
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((10, 8))
    labels = np.zeros(10, dtype=np.int64)

    windows, window_labels = windowize(samples, labels, window_samples=5, stride_samples=5)

    assert windows == []
    assert window_labels == []


def test_load_uci_emg_across_subjects_and_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for subj in ("01", "02"):
        for i in (1, 2):
            labels = [2] * 20
            _write_subject_file(tmp_path / subj / f"{i}_raw_data.txt", labels, rng)

    windows, labels, subjects = load_uci_emg(
        tmp_path, window_samples=5, stride_samples=5, n_subjects=2, seed=0,
    )

    assert set(subjects) == {"01", "02"}
    assert windows.shape[1:] == (5, 8)
    assert set(labels.tolist()) == {2}


def test_load_uci_emg_raises_when_no_subjects(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_uci_emg(tmp_path, window_samples=5, stride_samples=5, n_subjects=1, seed=0)


def test_load_uci_emg_raises_when_no_valid_windows(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    _write_subject_file(tmp_path / "01" / "1_raw_data.txt", [0] * 10, rng)

    with pytest.raises(ValueError, match="nessuna finestra"):
        load_uci_emg(tmp_path, window_samples=5, stride_samples=5, n_subjects=1, seed=0)
