"""Vista tokenizer canonica (docs/decisioni.md, "Bivio di v10 §6.3", 29/09/2026).

Serve SOLO a calcolare l'ingresso del tokenizer NeuroRVQ congelato (i target RVQ e le
verifiche V1-V4 del passo 1-bis); il front-end del modello continua a vedere i dati nativi.

Passi, per i dataset con frequenza nativa >= 1 kHz: passabanda 20-400 Hz alla frequenza
nativa -> resampling polifase a 1000 Hz esatti -> fattore di scala per SESSIONE. I dataset
sotto 1 kHz (DB5, 200 Hz) NON sono eleggibili: entrano solo come ancora mascherata, senza
upsampling (v10 §6.3).

Il filtro ricalca `preprocessing/preprocessing_emg_example.py` del repo NeuroRVQ
(commit 926e770, righe 47-49 e 69-75): Butterworth di ordine 3, zero-phase, taglio alto
`min(400, fs/2) - 0.5` (399,5 Hz). Nel repo il filtro e' in forma (b, a); qui in forma sos,
piu' stabile ai 5120 Hz di putEMG con un taglio basso a 20 Hz.

Convenzione dei dati come nel resto del repo: (T, C), tempo sull'asse 0. Solo `patchify`
restituisce (C, A, 200), la forma che il tokenizer riceve dopo il suo `rearrange`.

NON decisi qui (D5a, definizioni operative): la scala di arrivo (`target_scale`) e come si
sceglie; il default 1,0 e' un segnaposto.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction

import numpy as np
from scipy import signal

BAND_HZ = (20.0, 400.0)
TARGET_FS = 1000.0
FILTER_ORDER = 3
PATCH_SAMPLES = 200  # 200 ms a 1000 Hz


def resample_ratio(
    fs: float, target_fs: float = TARGET_FS, *, max_denominator: int = 1024, tol: float = 1e-3
) -> tuple[int, int]:
    """(up, down) con up/down ~= target_fs / fs. Solleva ValueError se fs < target_fs
    (dataset non eleggibile) o se il rapporto non e' approssimabile entro `tol` relativa con
    un denominatore <= max_denominator (un rapporto enorme darebbe un filtro polifase
    gigantesco)."""
    if fs < target_fs:
        raise ValueError(
            f"fs={fs} Hz < {target_fs} Hz: il dataset non e' eleggibile alla vista canonica "
            "(ancora mascherata, nessun upsampling: v10 §6.3)"
        )
    exact = target_fs / fs
    frac = Fraction(exact).limit_denominator(max_denominator)
    if abs(float(frac) - exact) / exact > tol:
        raise ValueError(f"rapporto {exact!r} non approssimabile entro {tol} con den <= {max_denominator}")
    return frac.numerator, frac.denominator


def bandpass_resample(
    data: np.ndarray,
    fs: float,
    *,
    band: tuple[float, float] = BAND_HZ,
    target_fs: float = TARGET_FS,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """data: (T, C). Passabanda zero-phase alla frequenza nativa, poi resampling polifase a
    `target_fs`. Ritorna (T', C) in float64."""
    up, down = resample_ratio(fs, target_fs)
    hi = min(band[1], fs / 2.0) - 0.5
    sos = signal.butter(order, [band[0], hi], btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, np.asarray(data, dtype=np.float64), axis=0)
    if (up, down) != (1, 1):
        x = signal.resample_poly(x, up, down, axis=0)
    return x


def session_scale(chunks: Iterable[np.ndarray], channel_mask: np.ndarray | None = None) -> float:
    """Deviazione standard su TUTTI i canali e tutte le prove di una sessione, in un solo
    numero (decisione: fattore unico per registrazione, non z-score per canale).

    chunks: array (T, C) gia' in vista canonica (una per prova); si accumula in streaming
    in float64. channel_mask: (C,) bool, i canali che passano il QC; gli altri non entrano."""
    n = 0
    s = 0.0
    s2 = 0.0
    for chunk in chunks:
        x = np.asarray(chunk, dtype=np.float64)
        if channel_mask is not None:
            x = x[:, np.asarray(channel_mask, dtype=bool)]
        n += x.size
        s += float(x.sum())
        s2 += float(np.square(x).sum())
    if n == 0:
        raise ValueError("nessun campione: sessione vuota o canali tutti scartati")
    var = s2 / n - (s / n) ** 2
    std = float(np.sqrt(max(var, 0.0)))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("deviazione standard di sessione nulla o non finita")
    return std


def apply_scale(data: np.ndarray, scale: float, *, target_scale: float = 1.0) -> np.ndarray:
    """Divide per la scala di sessione e porta alla scala di arrivo (segnaposto 1,0: la
    scelta vera e' una calibrazione su emg2pose, D5a)."""
    return np.asarray(data, dtype=np.float64) * (target_scale / scale)


def patchify(data: np.ndarray, patch_samples: int = PATCH_SAMPLES) -> np.ndarray:
    """(T, C) -> (C, A, patch_samples). La griglia e' ancorata al primo campione; il resto
    finale (< una patch) si scarta."""
    n_patches = data.shape[0] // patch_samples
    if n_patches == 0:
        raise ValueError(f"segnale piu' corto di una patch ({data.shape[0]} < {patch_samples})")
    cut = data[: n_patches * patch_samples]
    return cut.reshape(n_patches, patch_samples, data.shape[1]).transpose(2, 0, 1)
