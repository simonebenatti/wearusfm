"""Target delle ancore esplicite (passo 6; v10 §6.2-6.3), calcolati dal segnale per canale e per patch. Solo numpy: si provano sul Mac.

Le ancore non possono collassare (target esterno e fisso) e tengono ampiezza e spettro nella rappresentazione (v10 §6.2). **La finestra del target
non coincide con la patch** (v10 §6.3: a 25 ms la risoluzione in frequenza e' 40 Hz, la banda 20-40 Hz non si risolve). Tutte le finestre sono
centrate sul centro della patch; un target la cui finestra esce dal segnale non c'e' (maschera).

| Target | Finestra (proposta di AG) | v10 §6.3 |
|---|---|---|
| log RMS | 25 ms = la patch | 20-50 ms; la patch stessa evita che il target guardi le patch vicine, magari visibili |
| forma spettrale: log della frazione di potenza in 5 bande | 200 ms (8 patch, la griglia RVQ) | >= 100-250 ms, risoluzione 4-10 Hz (qui 5 Hz) |
| log inviluppo (RMS su finestra lunga) | 500 ms | «finestre piu' lunghe», contesto di attivazione |

Bande: 5, **log-spaziate su 20-450 Hz** (v10 §6.3; l'alternativa «dalla PSD media del corpus» resta aperta). **Mascherate oltre la banda
disponibile** del canale (v10 §4.2): una banda il cui bordo superiore supera il limite del canale (min fra Nyquist e banda effettiva dichiarata) non
ha target, e la frazione si normalizza solo sulle bande disponibili. Esempio: DB5 a 200 Hz tiene le bande sotto 100 Hz.

**Ancora RVQ** (v10 §6.3, D5b): un target per (canale, finestra da 200 ms della griglia del tokenizer) solo dove la finestra e' **interamente
nascosta** (slab su tutti i canali, v10 §6.4) e solo sui dataset con l'ancora accesa; i codici (livello 0 del ramo 0) li calcola il tokenizer
congelato sulla vista canonica a 1 kHz, fuori da questo modulo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BAND_EDGES_HZ = tuple(float(v) for v in np.geomspace(20.0, 450.0, 6))  # 20, 37, 70, 130, 242, 450 Hz
RMS_WINDOW_MS = 25.0
SPECTRAL_WINDOW_MS = 200.0
ENVELOPE_WINDOW_MS = 500.0
RVQ_TOKEN_MS = 200.0
EPS = 1e-8


@dataclass(frozen=True)
class AnchorTargets:
    log_rms: np.ndarray  # (C, P)
    rms_valid: np.ndarray  # (C, P)
    band_shape: np.ndarray  # (C, P, B) log della frazione di potenza per banda
    spec_valid: np.ndarray  # (C, P): finestra dentro il segnale
    band_available: np.ndarray  # (C, B): banda sotto il limite del canale
    log_env: np.ndarray  # (C, P)
    env_valid: np.ndarray  # (C, P)


def _windows(x: np.ndarray, fs: float, centers_s: np.ndarray, win_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """(C, P, n) finestre centrate, (P,) valida se la finestra sta tutta nel segnale."""
    n = max(1, int(round(win_ms / 1000.0 * fs)))
    starts = np.round(centers_s * fs - n / 2.0).astype(np.int64)
    valid = (starts >= 0) & (starts + n <= x.shape[1])
    idx = np.clip(starts[:, None] + np.arange(n)[None, :], 0, x.shape[1] - 1)
    return x[:, idx], valid


def anchor_targets(x: np.ndarray, fs: float, band_limit_hz, *, patch_ms: float = 25.0, edges_hz=BAND_EDGES_HZ) -> AnchorTargets:
    """x: (C, T) segnale normalizzato alla sua fs nativa; band_limit_hz: scalare o (C,) limite superiore realmente disponibile per canale."""
    x = np.asarray(x, dtype=np.float64)
    c, t = x.shape
    patch_s = patch_ms / 1000.0
    n_patch = int(math.floor(t / fs / patch_s + 1e-9))
    if n_patch == 0:
        raise ValueError("segnale piu' corto di una patch")
    centers = (np.arange(n_patch) + 0.5) * patch_s
    limit = np.minimum(np.broadcast_to(np.asarray(band_limit_hz, dtype=np.float64), (c,)), fs / 2.0)
    edges = np.asarray(edges_hz, dtype=np.float64)
    band_available = edges[None, 1:] <= limit[:, None] + 1e-9  # (C, B)

    seg, rms_ok = _windows(x, fs, centers, RMS_WINDOW_MS)
    log_rms = np.log(np.sqrt(np.mean(seg ** 2, axis=-1)) + EPS)

    seg, spec_ok = _windows(x, fs, centers, SPECTRAL_WINDOW_MS)
    n = seg.shape[-1]
    power = np.abs(np.fft.rfft(seg * np.hanning(n), axis=-1)) ** 2  # (C, P, F)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band_power = np.stack([power[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) for lo, hi in zip(edges[:-1], edges[1:])], axis=-1)
    band_power = band_power * band_available[:, None, :]
    total = band_power.sum(axis=-1, keepdims=True)
    band_shape = np.log(band_power / (total + EPS) + EPS)

    seg, env_ok = _windows(x, fs, centers, ENVELOPE_WINDOW_MS)
    log_env = np.log(np.sqrt(np.mean(seg ** 2, axis=-1)) + EPS)

    rows = lambda v: np.broadcast_to(v[None, :], (c, n_patch)).copy()  # noqa: E731
    return AnchorTargets(log_rms, rows(rms_ok), band_shape, rows(spec_ok) & band_available.any(axis=1)[:, None], band_available, log_env, rows(env_ok))


def rvq_target_windows(visible: np.ndarray, patch_valid: np.ndarray, rvq_on: np.ndarray, *, patch_ms: float = 25.0) -> np.ndarray:
    """(C, W) True dove la finestra w del tokenizer (patch [8w, 8w+8) a 25 ms) e' tutta valida e tutta nascosta e il canale ha l'ancora accesa.
    La griglia e' quella della finestra: il dataloader fa cominciare le finestre su multipli di 200 ms dall'inizio della prova (proposta D10)."""
    per = int(round(RVQ_TOKEN_MS / patch_ms))
    if abs(per * patch_ms - RVQ_TOKEN_MS) > 1e-9:
        raise ValueError(f"la patch da {patch_ms} ms non divide i {RVQ_TOKEN_MS} ms del tokenizer")
    c, p = visible.shape
    w = p // per
    hidden = (~visible[:, : w * per] & patch_valid[:, : w * per]).reshape(c, w, per).all(axis=-1)
    return hidden & np.asarray(rvq_on, dtype=bool)[:, None]
