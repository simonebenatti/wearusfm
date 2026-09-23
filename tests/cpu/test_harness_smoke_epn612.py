"""Test della logica di caricamento di bench/harness_smoke_epn612.py, con un fixture
JSON sintetico che riproduce la struttura reale EPN-612 (vedi ricognizione in sessione,
23/09/2026) - non serve il dataset vero per validare il parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bench"))

from harness_smoke_epn612 import _load_user, load_epn612  # noqa: E402

GESTURES = ("noGesture", "fist", "waveIn", "waveOut", "open", "pinch")


def _make_user_json(n_samples_per_class: int = 3, window_len: int = 996) -> dict:
    """Riproduce la struttura reale (verificata in sessione, 23/09/2026): solo
    `trainingSamples` ha `gestureName`, `testingSamples` e' senza etichetta (dataset
    in stile competizione) - il loader deve ignorarlo, non fallire su di esso."""
    rng = np.random.default_rng(0)
    training: dict = {}
    idx = 0
    for gesture in GESTURES:
        for _ in range(n_samples_per_class):
            emg = {f"ch{c}": rng.standard_normal(window_len).tolist() for c in range(1, 9)}
            training[f"idx_{idx}"] = {
                "startPointforGestureExecution": 100,
                "gestureName": gesture,
                "emg": emg,
            }
            idx += 1
    testing = {
        "idx_unlabeled": {
            "startPointforGestureExecution": 100,
            "emg": {f"ch{c}": rng.standard_normal(window_len).tolist() for c in range(1, 9)},
        }
    }
    return {
        "generalInfo": {"deviceModel": "Myo Armband", "samplingFrequencyInHertz": 200,
                         "recordingTimeInSeconds": 5},
        "userInfo": {"name": "synthetic"},
        "trainingSamples": training,
        "testingSamples": testing,
    }


def test_load_user_extracts_windows_and_labels(tmp_path: Path) -> None:
    user_dir = tmp_path / "user1"
    user_dir.mkdir()
    (user_dir / "user1.json").write_text(json.dumps(_make_user_json()))

    windows, labels, fs_hz = _load_user(user_dir / "user1.json")

    assert fs_hz == 200.0
    assert len(windows) == len(labels) == 6 * 3
    assert all(w.shape == (996, 8) for w in windows)
    assert set(labels) == set(GESTURES)


def test_load_epn612_across_multiple_users(tmp_path: Path) -> None:
    for i in range(1, 4):
        user_dir = tmp_path / f"user{i}"
        user_dir.mkdir()
        (user_dir / f"user{i}.json").write_text(json.dumps(_make_user_json(n_samples_per_class=2)))

    windows, labels, subjects, fs_hz = load_epn612(tmp_path, n_users=3, seed=0)

    assert windows.shape == (3 * 6 * 2, 996, 8)
    assert set(subjects) == {"user1", "user2", "user3"}
    assert fs_hz == 200.0


def test_load_epn612_truncates_mismatched_lengths(tmp_path: Path) -> None:
    user_dir = tmp_path / "user1"
    user_dir.mkdir()
    payload = _make_user_json(n_samples_per_class=1, window_len=996)
    # una finestra piu' corta, campione malformato realistico
    short_sample = next(iter(payload["trainingSamples"].values()))
    for ch in short_sample["emg"]:
        short_sample["emg"][ch] = short_sample["emg"][ch][:500]
    (user_dir / "user1.json").write_text(json.dumps(payload))

    windows, labels, subjects, fs_hz = load_epn612(tmp_path, n_users=1, seed=0)
    assert windows.shape[1] == 500  # troncato alla piu' corta, non scartato


def test_load_epn612_raises_on_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_epn612(tmp_path / "does_not_exist", n_users=1, seed=0)
