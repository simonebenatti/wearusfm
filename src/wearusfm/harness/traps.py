"""Trappole da chiudere per iscritto (v10 §8) - diagnostiche, non blocchi assoluti: un
protocollo onesto le misura e le dichiara, non le nasconde.
"""

from __future__ import annotations

import numpy as np


def label_run_lengths(labels: np.ndarray, target_label) -> np.ndarray:
    """Lunghezze (in campioni) dei segmenti consecutivi etichettati `target_label`."""
    is_target = np.asarray(labels) == target_label
    if not is_target.any():
        return np.array([], dtype=np.int64)
    # trova i confini dei run consecutivi di True
    padded = np.concatenate([[False], is_target, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return ends - starts


def warn_if_rest_looks_like_padding(
    labels: np.ndarray,
    rest_label,
    *,
    short_run_threshold_samples: int,
    short_fraction_threshold: float = 0.7,
) -> list[str]:
    """v10 §8: "niente classe rest derivata dalle pause fra gesti" - label leakage via
    padding documentato da NeuroRVQ su NinaPro DB5. Euristica: se la maggior parte dei
    segmenti di riposo sono piu' corti di `short_run_threshold_samples`, e' probabile
    che "riposo" sia in realta' il padding automatico fra un gesto e il successivo,
    non un periodo di riposo genuino registrato come tale.

    Ritorna una lista di avvisi testuali (vuota se non c'e' nulla di sospetto). Non
    solleva eccezioni: e' una diagnostica per la revisione umana, non un gate.
    """
    runs = label_run_lengths(labels, rest_label)
    warnings: list[str] = []
    if runs.size == 0:
        return warnings
    short_fraction = float(np.mean(runs < short_run_threshold_samples))
    if short_fraction >= short_fraction_threshold:
        warnings.append(
            f"il {short_fraction:.0%} dei segmenti '{rest_label}' e' piu' corto di "
            f"{short_run_threshold_samples} campioni: sembra padding automatico fra "
            "gesti, non riposo genuino (v10 §8). Verificare prima di usarlo come classe."
        )
    return warnings
