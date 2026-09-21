"""Feature di Hudgins (Hudgins et al. 1993): la baseline classica dell'EMG di superficie,
"non negoziabile" nel protocollo (v10 §8). Quattro feature nel dominio del tempo, per
canale: MAV, ZC, SSC, WL.
"""

from __future__ import annotations

import numpy as np


def mean_absolute_value(window: np.ndarray) -> np.ndarray:
    """window: (T, C) -> (C,)."""
    return np.mean(np.abs(window), axis=0)


def zero_crossings(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Conteggio di attraversamenti dello zero, con soglia per ignorare il rumore
    intorno allo zero (window: (T, C) -> (C,))."""
    x = window
    sign_change = (x[:-1] * x[1:]) < 0
    big_enough = np.abs(x[:-1] - x[1:]) >= threshold
    return np.sum(sign_change & big_enough, axis=0).astype(np.float64)


def slope_sign_changes(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Conteggio dei cambi di segno della pendenza (picchi/valli), con soglia
    (window: (T, C) -> (C,))."""
    if window.shape[0] < 3:
        return np.zeros(window.shape[1])
    diff_prev = window[1:-1] - window[:-2]
    diff_next = window[1:-1] - window[2:]
    is_extremum = (diff_prev * diff_next) > 0
    big_enough = (np.abs(diff_prev) >= threshold) & (np.abs(diff_next) >= threshold)
    return np.sum(is_extremum & big_enough, axis=0).astype(np.float64)


def waveform_length(window: np.ndarray) -> np.ndarray:
    """Lunghezza cumulativa della forma d'onda (window: (T, C) -> (C,))."""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def extract_hudgins_features(window: np.ndarray, zc_threshold: float = 0.0, ssc_threshold: float = 0.0) -> np.ndarray:
    """window: (T, C) -> vettore di feature (4*C,), ordine [MAV, ZC, SSC, WL] per canale
    concatenato (non interfogliato): mav_ch0..chC, zc_ch0..chC, ssc_ch0..chC, wl_ch0..chC."""
    mav = mean_absolute_value(window)
    zc = zero_crossings(window, threshold=zc_threshold)
    ssc = slope_sign_changes(window, threshold=ssc_threshold)
    wl = waveform_length(window)
    return np.concatenate([mav, zc, ssc, wl])


def extract_hudgins_features_batch(windows: np.ndarray, **kwargs) -> np.ndarray:
    """windows: (N, T, C) -> (N, 4*C)."""
    return np.stack([extract_hudgins_features(w, **kwargs) for w in windows])
