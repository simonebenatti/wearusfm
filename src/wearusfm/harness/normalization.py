"""Normalizzazione stimata SOLO sul train (v10 §8, "Trappole da chiudere per iscritto").

Stimarla su tutto il dataset (train+val+test) e' una fuga di informazione dal test set
verso il train, anche se sottile: le statistiche di normalizzazione portano informazione
sulla distribuzione del test. Se la normalizzazione e' per sessione, si stima dai primi
N secondi di quella sessione (v10 §8), non dall'intera sessione (che includerebbe anche i
secondi usati come finestre di valutazione).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizationStats:
    median: np.ndarray  # (n_channels,)
    mad: np.ndarray  # (n_channels,) - median absolute deviation, v10 §4.3

    def apply(self, data: np.ndarray) -> np.ndarray:
        """data: (..., n_channels). Scala CONDIVISA fra canali per traccia (v10 §4.3):
        qui pero' le stats sono per-canale in ingresso - la condivisione fra canali va
        decisa a monte (mediana del MAD sui canali) da chi chiama, non imposta qui."""
        return (data - self.median) / np.where(self.mad > 0, self.mad, 1.0)


def fit_train_only(train_data: np.ndarray) -> NormalizationStats:
    """train_data: (n_samples, n_channels) o (n_samples, T, n_channels) - le statistiche
    si calcolano sull'asse 0 (e sull'asse temporale se presente), mai su val/test."""
    axes = tuple(range(train_data.ndim - 1))
    median = np.median(train_data, axis=axes)
    mad = np.median(np.abs(train_data - median), axis=axes)
    return NormalizationStats(median=median, mad=mad)


def fit_per_session_first_n_seconds(
    session_data: np.ndarray, fs_hz: float, n_seconds: float
) -> NormalizationStats:
    """v10 §8: se la normalizzazione e' per sessione, dai primi N secondi - non
    dall'intera sessione, per non includere finestre che finiranno nella valutazione.

    `session_data`: (n_samples_tempo, n_channels), una singola sessione/registrazione.
    """
    n_samples = round(fs_hz * n_seconds)
    n_samples = min(n_samples, session_data.shape[0])
    if n_samples < 1:
        raise ValueError("n_seconds troppo piccolo per la frequenza data: zero campioni")
    return fit_train_only(session_data[:n_samples])
