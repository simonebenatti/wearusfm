"""Physical one-sided PSD of normalized loader signals; no detrending/resampling."""

import numpy as np
from scipy import fft as sfft

LEGACY_TARGET_VERSION = "step1_psd_power_shape_energy_v1"
TARGET_VERSION = "step1_psd_power_shape_energy_fast100_v2"
MS_FAST_WINDOW_MS = 100.0
LEGACY_MS_FAST_WINDOW_MS = 50.0
EPS = 1e-8


def target_version_for_fast_window(window_ms: float) -> str:
    """Registered target recipes: 100 ms for new runs, explicit 50 ms for historical reproduction."""
    if window_ms == MS_FAST_WINDOW_MS:
        return TARGET_VERSION
    if window_ms == LEGACY_MS_FAST_WINDOW_MS:
        return LEGACY_TARGET_VERSION
    raise ValueError("ms_fast_window_ms deve essere 100 (nuovo default) o 50 (ricetta storica)")


def hann_periodogram(segments: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Density (..., F), frequencies (F,); symmetric Hann, nfft=N.

    Integrate with df=fs/N to obtain power in normalized-signal squared units.
    Degenerate Hann windows (N<3) are rejected rather than producing NaNs.
    """
    x = np.asarray(segments)
    if x.ndim < 1 or x.shape[-1] < 3 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("PSD requires N>=3 and a positive finite sampling rate")
    dtype = np.float64 if x.dtype == np.float64 else np.float32
    w = np.hanning(x.shape[-1]).astype(dtype)
    fft = sfft.rfft(x.astype(dtype, copy=False) * w, axis=-1)
    density = (fft.real ** 2 + fft.imag ** 2) / float(fs * np.sum(w ** 2, dtype=np.float64))
    # Odd N: last bin is positive, not Nyquist, and must also be doubled.
    density[..., 1:-1 if x.shape[-1] % 2 == 0 else None] *= 2
    return density, np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
