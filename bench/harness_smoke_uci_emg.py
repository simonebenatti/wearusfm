#!/usr/bin/env python3
"""Smoke test del passo 5 (harness di valutazione) su dati REALI: UCI-EMG (Lobov et al.).

Secondo dei due benchmark "mai visto" assegnati al passo 5 (piano_operativo_v10.md,
passo 5, "Dataset: EPN-612 e UCI-EMG subito"; v10 §2.3 li esclude entrambi dal
pretraining). A differenza di EPN-612 (gia' segmentato in campioni), qui i file sono
serie temporali continue con un'etichetta di classe per campione - vanno finestrati.

Formato reale (verificato in sessione, 23/09/2026): file .txt tab-separated per
soggetto/serie, colonne [time, channel1..8, class], header presente. Classi: 0=non
marcato, 1=riposo, 2=pugno, 3=flessione polso, 4=estensione polso, 5=deviazione
radiale, 6=deviazione ulnare, 7=palmo esteso (non tutti i soggetti).

Frequenza nativa: 1 kHz, verificata dal paper originale degli stessi autori (Lobov et
al., Sensors 2018, 18(4):1122, DOI 10.3390/s18041122 - docs/fatti_da_verificare.md #15).
Il finestramento resta a conteggio di CAMPIONI (non a durata in secondi): 200
campioni/passo 100 corrisponde esattamente ai "200 ms overlapping time windows at a 100
ms step" del paper originale, a questa frequenza.

Uso:
    python bench/harness_smoke_uci_emg.py --root $WORK/data/raw/uci_emg \\
        --window-samples 200 --stride-samples 100 --seed 0 \\
        --out results/passo5/uci_emg_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.harness.protocol import run_hudgins_lda_protocol  # noqa: E402
from wearusfm.harness.traps import warn_if_rest_looks_like_padding  # noqa: E402

from wearusfm.harness.benchmarks import load_uci_file as _load_subject_file  # noqa: E402
from wearusfm.harness.benchmarks import windowize  # noqa: E402


def load_uci_emg(
    root: Path, *, window_samples: int, stride_samples: int, n_subjects: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_dirs = sorted(p for p in root.rglob("*") if p.is_dir() and p.name.isdigit())
    if not subject_dirs:
        raise FileNotFoundError(f"nessuna directory soggetto (2 cifre) sotto {root}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(subject_dirs), size=min(n_subjects, len(subject_dirs)), replace=False)
    chosen_dirs = [subject_dirs[i] for i in sorted(chosen)]

    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []
    all_subjects: list[str] = []

    for subject_dir in chosen_dirs:
        for txt_file in sorted(subject_dir.glob("*.txt")):
            samples, labels = _load_subject_file(txt_file)
            windows, window_labels = windowize(
                samples, labels, window_samples=window_samples, stride_samples=stride_samples,
            )
            all_windows.extend(windows)
            all_labels.extend(window_labels)
            all_subjects.extend([subject_dir.name] * len(windows))

    if not all_windows:
        raise ValueError("nessuna finestra valida estratta (controlla window_samples/stride_samples)")

    return np.stack(all_windows), np.array(all_labels), np.array(all_subjects)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--window-samples", type=int, default=200)
    p.add_argument("--stride-samples", type=int, default=100)
    p.add_argument("--n-subjects", type=int, default=36)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    print(f"carico fino a {args.n_subjects} soggetti da {args.root} ...")
    windows, labels, subjects = load_uci_emg(
        args.root, window_samples=args.window_samples, stride_samples=args.stride_samples,
        n_subjects=args.n_subjects, seed=args.seed,
    )
    print(f"caricati: {len(windows)} finestre, {len(set(subjects))} soggetti, "
          f"shape finestra={windows.shape[1:]}")

    warnings = warn_if_rest_looks_like_padding(
        labels, rest_label=1,  # "hand at rest" (v10 §8: verificare se e' riposo genuino o padding)
        short_run_threshold_samples=3,
    )

    result = run_hudgins_lda_protocol(windows, labels, subjects, seed=args.seed)

    report = {
        "dataset": "uci_emg",
        "fs_hz": 1000.0,  # verificato, Lobov et al. Sensors 2018 (docs/fatti_da_verificare.md #15)
        "window_samples": args.window_samples,
        "stride_samples": args.stride_samples,
        "n_subjects_requested": args.n_subjects,
        "n_subjects_loaded": int(len(set(subjects))),
        "n_windows": int(len(windows)),
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
