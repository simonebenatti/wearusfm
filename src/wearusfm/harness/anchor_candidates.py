"""Confronto dei target candidati per le ancore in frequenza (Simone, 04/10/2026: protocollo e soglie congelati in `docs/decisioni.md`, «Ancore in
frequenza: confronto dei target candidati»). Solo numpy e scikit-learn: target per blocco da 200 ms e sonde lineari. Il lavoro col modello e col
tokenizer e' in `scripts/anchor_candidates.py`.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

R2_MIN = 0.3  # soglia congelata: R² dalle rappresentazioni delle patch nascoste
ACC_GAIN_MIN = 0.10  # soglia congelata: 10 punti sopra la classe piu' frequente (64 gruppi)
EPS = 1e-12


def _spectrum(block: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """block: (C, n) -> (potenza (C, F) con finestra di Hann, frequenze (F,))."""
    n = block.shape[-1]
    spec = np.fft.rfft(block * np.hanning(n), axis=-1)
    return spec.real ** 2 + spec.imag ** 2, np.fft.rfftfreq(n, 1.0 / fs)


def band_logpower(block: np.ndarray, fs: float, n_bands: int = 16, lo: float = 20.0, hi: float = 450.0) -> np.ndarray:
    """(C, n_bands) log della potenza in bande log-spaziate fra lo e hi (il bordo alto si ferma alla Nyquist). Una banda senza frequenze: NaN."""
    power, freqs = _spectrum(block, fs)
    edges = np.geomspace(lo, min(hi, fs / 2.0), n_bands + 1)
    out = np.full((block.shape[0], n_bands), np.nan)
    for k in range(n_bands):
        sel = (freqs >= edges[k]) & (freqs < edges[k + 1])
        if sel.any():
            out[:, k] = np.log(power[:, sel].sum(axis=-1) + EPS)
    return out


def band_shape(block: np.ndarray, fs: float, n_bands: int = 5, lo: float = 20.0, hi: float = 450.0) -> np.ndarray:
    """(C, n_bands) log della frazione di potenza per banda: la forma spettrale delle ancore attuali, sul blocco."""
    lp = band_logpower(block, fs, n_bands, lo, hi)
    p = np.exp(lp)
    return np.log(p / (np.nansum(p, axis=-1, keepdims=True) + EPS) + EPS)


def spectral_moments(block: np.ndarray, fs: float, lo: float = 20.0, hi: float = 450.0) -> np.ndarray:
    """(C, 2) frequenza media e frequenza mediana dello spettro di potenza fra lo e hi."""
    power, freqs = _spectrum(block, fs)
    sel = (freqs >= lo) & (freqs <= min(hi, fs / 2.0))
    p, f = power[:, sel], freqs[sel]
    tot = p.sum(axis=-1) + EPS
    mean = (p * f).sum(axis=-1) / tot
    cum = np.cumsum(p, axis=-1) / tot[:, None]
    median = f[np.argmax(cum >= 0.5, axis=-1)]
    return np.stack([mean, median], axis=-1)


def multiscale_logpower(window: np.ndarray, fs: float, lo: int, hi: int) -> np.ndarray:
    """Opzione 3 (tempo-frequenza a piu' scale) per il blocco [lo, hi) di una finestra (C, T): log della potenza in 6 bande su ciascuno dei 4
    sottoblocchi da 50 ms (dinamica rapida) e in 24 bande su 500 ms centrati sul blocco (spettro fine), tagliati ai bordi della finestra.
    La scala da 200 ms e' gia' l'opzione 1 (16 bande). Ritorna (C, 48)."""
    n = hi - lo
    q = n // 4
    fast = np.concatenate([band_logpower(window[:, lo + i * q: lo + (i + 1) * q], fs, n_bands=6) for i in range(4)], axis=1)
    half = int(round(0.25 * fs))
    mid = (lo + hi) // 2
    slow = band_logpower(window[:, max(0, mid - half): min(window.shape[1], mid + half)], fs, n_bands=24)
    return np.concatenate([fast, slow], axis=1)


def log_rms(block: np.ndarray) -> np.ndarray:
    return np.log(np.sqrt((block ** 2).mean(axis=-1)) + EPS)[:, None]


def ridge_r2(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray) -> float:
    """R² medio sulle componenti di una regressione ridge (alpha per validazione incrociata sul train), letto sul test."""
    sx, sy = StandardScaler().fit(x_tr), StandardScaler().fit(y_tr)
    m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sx.transform(x_tr), sy.transform(y_tr))
    return float(r2_score(sy.transform(y_te), m.predict(sx.transform(x_te)), multioutput="uniform_average"))


def logistic_gain(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, seed: int = 0) -> dict:
    """Accuratezza di una regressione logistica sul test contro la classe piu' frequente del train applicata al test."""
    sx = StandardScaler().fit(x_tr)
    m = LogisticRegression(max_iter=500, C=1.0, random_state=seed).fit(sx.transform(x_tr), y_tr)
    acc = float((m.predict(sx.transform(x_te)) == y_te).mean())
    vals, counts = np.unique(y_tr, return_counts=True)
    major = float((y_te == vals[np.argmax(counts)]).mean())
    return {"accuracy": acc, "majority": major, "gain": acc - major}


def kmeans_labels(fit_on: np.ndarray, apply_to: list[np.ndarray], k: int = 64, seed: int = 0) -> list[np.ndarray]:
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096).fit(fit_on)
    return [km.predict(a) for a in apply_to]


def verdict(r2_h: float | None = None, gain_h: float | None = None) -> bool:
    """La regola congelata: prevedibile se dalle patch nascoste R² >= 0,3, oppure (classificazione) 10 punti sopra la classe piu' frequente."""
    if r2_h is not None:
        return r2_h >= R2_MIN
    return gain_h is not None and gain_h >= ACC_GAIN_MIN
