#!/usr/bin/env python3
"""Smoke test del passo 5 (harness di valutazione) su dati REALI: EPN-612.

Primo collaudo dello scheletro dell'harness (splits/normalization/features_hudgins/
baselines/protocol, gia' validati solo con dati sintetici nei test unitari) contro un
dataset vero. EPN-612 e' escluso dal pretraining (v10 §2.3): qui serve solo come uno dei
due benchmark "mai visto" assegnati al passo 5 (piano_operativo_v10.md, passo 5, "Dataset:
EPN-612 e UCI-EMG subito").

Ogni file utente e' gia' segmentato in campioni da 5s (generalInfo.recordingTimeInSeconds)
a 200 Hz (generalInfo.samplingFrequencyInHertz) - non serve finestrare, ogni campione e'
gia' una finestra (v10 §8: split per soggetto, non per finestra - qui il soggetto e' lo
user*, la finestra e' il campione).

Uso:
    python bench/harness_smoke_epn612.py --root $WORK/data/raw/epn612 \\
        --n-users 30 --seed 0 --out results/passo5/epn612_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.harness.protocol import run_hudgins_lda_protocol  # noqa: E402
from wearusfm.harness.splits import check_no_cross_dataset_subject_overlap  # noqa: E402
from wearusfm.harness.traps import warn_if_rest_looks_like_padding  # noqa: E402

EMG_CHANNEL_KEYS = tuple(f"ch{i}" for i in range(1, 9))


def _load_user(path: Path) -> tuple[list[np.ndarray], list[str], float]:
    """Un file utente EPN-612: ritorna (finestre, etichette, fs_hz).

    Solo `trainingSamples`: verificato in sessione (23/09/2026) che `testingSamples` non
    ha `gestureName` - e' un dataset in stile competizione con le etichette di test
    nascoste, non riusabile per un protocollo supervisionato senza quelle etichette."""
    with open(path) as f:
        data = json.load(f)
    fs_hz = float(data["generalInfo"]["samplingFrequencyInHertz"])
    windows: list[np.ndarray] = []
    labels: list[str] = []
    for sample in data.get("trainingSamples", {}).values():
        emg = sample["emg"]
        channels = [np.asarray(emg[k], dtype=np.float64) for k in EMG_CHANNEL_KEYS]
        lengths = {len(c) for c in channels}
        if len(lengths) != 1:
            continue  # canali di lunghezza diversa: campione malformato, si scarta
        window = np.stack(channels, axis=1)  # (T, 8)
        windows.append(window)
        labels.append(sample["gestureName"])
    return windows, labels, fs_hz


def load_epn612(root: Path, n_users: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    user_dirs = sorted(p for p in root.glob("user*") if (p / f"{p.name}.json").exists())
    if not user_dirs:
        raise FileNotFoundError(f"nessuna directory user* con json sotto {root}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(user_dirs), size=min(n_users, len(user_dirs)), replace=False)
    chosen_dirs = [user_dirs[i] for i in sorted(chosen)]

    all_windows: list[np.ndarray] = []
    all_labels: list[str] = []
    all_subjects: list[str] = []
    fs_hz_seen: set[float] = set()
    lengths_seen: set[int] = set()

    for user_dir in chosen_dirs:
        windows, labels, fs_hz = _load_user(user_dir / f"{user_dir.name}.json")
        fs_hz_seen.add(fs_hz)
        for w in windows:
            lengths_seen.add(w.shape[0])
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_subjects.extend([user_dir.name] * len(windows))

    if len(fs_hz_seen) != 1:
        raise ValueError(f"samplingFrequencyInHertz non uniforme fra utenti: {fs_hz_seen}")
    if len(lengths_seen) != 1:
        # finestre di lunghezza diversa: tronca tutte alla piu' corta invece di scartare
        min_len = min(lengths_seen)
        all_windows = [w[:min_len] for w in all_windows]

    windows_arr = np.stack(all_windows)
    labels_arr = np.array(all_labels)
    subjects_arr = np.array(all_subjects)
    return windows_arr, labels_arr, subjects_arr, fs_hz_seen.pop()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON")
    p.add_argument("--n-users", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    print(f"carico fino a {args.n_users} utenti da {args.root} ...")
    windows, labels, subjects, fs_hz = load_epn612(args.root, args.n_users, args.seed)
    print(f"caricati: {len(windows)} finestre, {len(set(subjects))} soggetti, fs={fs_hz} Hz, "
          f"shape finestra={windows.shape[1:]}")

    overlaps = check_no_cross_dataset_subject_overlap({"epn612": subjects})
    assert not overlaps, "impossibile: un solo dataset non puo' avere overlap con se stesso"

    warnings = warn_if_rest_looks_like_padding(
        labels, rest_label="noGesture",
        short_run_threshold_samples=1,  # ogni campione e' gia' una finestra indipendente
    )

    result = run_hudgins_lda_protocol(windows, labels, subjects, seed=args.seed)

    report = {
        "dataset": "epn612",
        "n_users_requested": args.n_users,
        "n_users_loaded": int(len(set(subjects))),
        "n_windows": int(len(windows)),
        "fs_hz": fs_hz,
        "window_shape": list(windows.shape[1:]),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "split": {
            "n_train_subjects": len(result.split.train),
            "n_val_subjects": len(result.split.val),
            "n_test_subjects": len(result.split.test),
        },
        "n_train_windows": result.n_train_windows,
        "n_val_windows": result.n_val_windows,
        "n_test_windows": result.n_test_windows,
        "train_balanced_accuracy": result.train_balanced_accuracy,
        "val_balanced_accuracy": result.val_balanced_accuracy,
        "test_balanced_accuracy": result.test_balanced_accuracy,
        "traps_warnings": warnings,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
