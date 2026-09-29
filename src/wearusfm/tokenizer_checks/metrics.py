"""Metriche e criteri di V1-V4, come CONGELATI in docs/decisioni.md ("D5a - Definizioni
operative di V1-V4", 29/09/2026). Solo numpy/scipy/sklearn: nessuna dipendenza da torch, si
testa in locale. Le soglie qui sotto NON si ritoccano dopo aver visto un risultato: se una si
rivela sbagliata si apre una nuova decisione.

Forme: i codici del tokenizer sono (4 rami, 16 livelli RVQ, n_token) interi in [0, 8192).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

# --- soglie congelate (D5a, 29/09/2026) ---
X_RATIO = 2.0  # V2: errore mediano su un dataset <= 2 x quello su emg2pose
Y_POINTS = 0.10  # V3: accuratezza dai codici - accuratezza dalle bande <= 10 punti
STABILITY_MIN = 0.75  # V4: >= 75% dei codici invariati, in tutti e 4 i rami, su tutti i dataset
V1_FACTOR = 1.5  # V1 (soglia nuova, D5a punto 2): errore canale-per-volta <= 1,5 x multi-canale
NOISE_FLOOR_QUANTILE = 0.10  # V4: il 10% dei token a energia minore

N_BRANCHES = 4
N_LEVELS = 16
N_CODE = 8192
N_TOKENS_PER_SAMPLE = 256  # V3: 256 token a caso della stessa sessione

# 5 bande del repo NeuroRVQ, plotting/plotting_example.py righe 86-92 (commit 926e770)
BANDS_HZ = ((20.0, 60.0), (60.0, 125.0), (125.0, 200.0), (200.0, 250.0), (250.0, 400.0))


# --- V2 (e V1) ---------------------------------------------------------------------------


def token_nmse(x: np.ndarray, x_rec: np.ndarray) -> np.ndarray:
    """Errore quadratico normalizzato per token, nel dominio standardizzato del tokenizer
    (`std_norm`, come la sua loss): sum((x - x_rec)^2) / sum(x^2) sull'ultimo asse."""
    x = np.asarray(x, dtype=np.float64)
    x_rec = np.asarray(x_rec, dtype=np.float64)
    if x.shape != x_rec.shape:
        raise ValueError(f"forme diverse: {x.shape} vs {x_rec.shape}")
    den = np.sum(x**2, axis=-1)
    if np.any(den <= 0.0):
        raise ValueError("token con energia nulla nel dominio standardizzato")
    return np.sum((x - x_rec) ** 2, axis=-1) / den


def median_ratio(dataset_nmse: np.ndarray, reference_nmse: np.ndarray) -> float:
    """mediana(dataset) / mediana(emg2pose)."""
    return float(np.median(dataset_nmse) / np.median(reference_nmse))


def v2_passes(ratio: float) -> bool:
    return ratio <= X_RATIO


def v1_passes(nmse_single_channel: np.ndarray, nmse_multi_channel: np.ndarray) -> bool:
    """V1: errore mediano canale-per-volta <= V1_FACTOR volte quello multi-canale."""
    return float(np.median(nmse_single_channel)) <= V1_FACTOR * float(np.median(nmse_multi_channel))


# --- V4 ----------------------------------------------------------------------------------


def noise_floor_rms(tokens: np.ndarray, quantile: float = NOISE_FLOOR_QUANTILE) -> float:
    """Noise floor di un dataset: RMS mediana dei token nel `quantile` a energia minore,
    calcolata nella vista canonica dopo la scala per sessione. tokens: (n, 200)."""
    rms = np.sqrt(np.mean(np.asarray(tokens, dtype=np.float64) ** 2, axis=-1))
    thr = np.quantile(rms, quantile)
    return float(np.median(rms[rms <= thr]))


def code_stability(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """Frazione di token il cui codice NON cambia, per (ramo, livello): (4, 16).
    I livelli sono confrontati in modo indipendente (non condizionati ai precedenti)."""
    if clean.shape != noisy.shape or clean.ndim != 3:
        raise ValueError(f"attese due forme uguali (rami, livelli, token): {clean.shape} {noisy.shape}")
    return np.mean(clean == noisy, axis=-1)


def stable_levels(
    fractions_by_dataset: Mapping[str, np.ndarray], threshold: float = STABILITY_MIN
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Un livello e' stabile su un dataset se in TUTTI i rami la frazione e' >= threshold; e'
    stabile se lo e' su TUTTI i dataset. Ritorna (per dataset (16,) bool, complessivo (16,))."""
    per_dataset = {name: np.all(np.asarray(f) >= threshold, axis=0) for name, f in fractions_by_dataset.items()}
    if not per_dataset:
        raise ValueError("nessun dataset")
    overall = np.all(np.stack(list(per_dataset.values())), axis=0)
    return per_dataset, overall


def v4_passes(overall_stable: np.ndarray) -> bool:
    """Nessun livello stabile = ancora scartata."""
    return bool(np.any(overall_stable))


# --- V3 ----------------------------------------------------------------------------------


def code_histogram_features(samples_codes: Sequence[np.ndarray]) -> sparse.csr_matrix:
    """Istogramma normalizzato dei codici per ogni coppia (ramo, livello), tutti i livelli,
    concatenati. samples_codes: lista di array (4, 16, n_token). Ritorna csr
    (n_campioni, 4*16*8192)."""
    rows, cols, vals = [], [], []
    for i, codes in enumerate(samples_codes):
        codes = np.asarray(codes)
        if codes.ndim != 3 or codes.shape[:2] != (N_BRANCHES, N_LEVELS):
            raise ValueError(f"forma attesa (4, 16, n_token), trovata {codes.shape}")
        if codes.min() < 0 or codes.max() >= N_CODE:
            raise ValueError("codice fuori da [0, 8192)")
        n_tok = codes.shape[-1]
        bl = (np.arange(N_BRANCHES)[:, None] * N_LEVELS + np.arange(N_LEVELS)[None, :])[..., None]
        flat = (bl * N_CODE + codes).ravel()
        rows.append(np.full(flat.shape, i))
        cols.append(flat)
        vals.append(np.full(flat.shape, 1.0 / n_tok))
    m = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(samples_codes), N_BRANCHES * N_LEVELS * N_CODE),
    )
    return m.tocsr()  # somma i duplicati


def band_power_features(
    samples_tokens: Sequence[np.ndarray], fs: float = 1000.0, bands: Sequence[tuple[float, float]] = BANDS_HZ
) -> np.ndarray:
    """Baseline di V3: log10 della potenza media dei token del campione in ciascuna banda.
    samples_tokens: lista di array (n_token, 200) nella vista canonica. Ritorna (n_campioni,
    n_bande)."""
    out = np.empty((len(samples_tokens), len(bands)))
    for i, tokens in enumerate(samples_tokens):
        tokens = np.asarray(tokens, dtype=np.float64)
        power = np.mean(np.abs(np.fft.rfft(tokens, axis=-1)) ** 2, axis=0)
        freqs = np.fft.rfftfreq(tokens.shape[-1], d=1.0 / fs)
        for j, (lo, hi) in enumerate(bands):
            sel = (freqs >= lo) & (freqs < hi)
            out[i, j] = np.log10(power[sel].sum() + 1e-12)
    return out


def split_subjects(
    subjects_by_dataset: Mapping[str, Sequence], rng: np.random.Generator, fractions=(0.6, 0.2, 0.2)
) -> dict[tuple[str, object], str]:
    """Split train/val/test con soggetti DISGIUNTI dentro ogni dataset. Ritorna
    {(dataset, soggetto): 'train'|'val'|'test'}. Serve almeno 3 soggetti per dataset."""
    out: dict[tuple[str, object], str] = {}
    for name, subjects in subjects_by_dataset.items():
        subjects = list(subjects)
        if len(subjects) < 3:
            raise ValueError(f"{name}: servono >= 3 soggetti per uno split train/val/test, ce ne sono {len(subjects)}")
        order = rng.permutation(len(subjects))
        n = len(subjects)
        n_val = max(1, int(round(fractions[1] * n)))
        n_test = max(1, int(round(fractions[2] * n)))
        for rank, idx in enumerate(order):
            part = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
            out[(name, subjects[idx])] = part
    return out


def balance_classes(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indici che pareggiano il numero di campioni per classe (sottocampionando)."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    n = counts.min()
    keep = [rng.choice(np.flatnonzero(labels == c), size=n, replace=False) for c in classes]
    return np.sort(np.concatenate(keep))


@dataclass
class ProbeResult:
    balanced_accuracy: float
    ci95: tuple[float, float]
    chance: float
    model: str
    n_test: int


def _balanced_accuracy(pred, y):
    classes = np.unique(y)
    return float(np.mean([np.mean(pred[y == c] == c) for c in classes]))


def _bootstrap_ci(pred, y, units, rng, n_boot=1000):
    """Bootstrap stratificato per classe, ricampionando i SOGGETTI (unita') dentro ogni classe."""
    classes = np.unique(y)
    per_class = {}
    for c in classes:
        us = np.unique(units[y == c])
        per_class[c] = [(np.sum((units == u) & (y == c) & (pred == c)), np.sum((units == u) & (y == c))) for u in us]
    accs = []
    for _ in range(n_boot):
        recalls = []
        for c in classes:
            pairs = per_class[c]
            pick = rng.integers(0, len(pairs), size=len(pairs))
            hit = sum(pairs[k][0] for k in pick)
            tot = sum(pairs[k][1] for k in pick)
            recalls.append(hit / tot if tot else 0.0)
        accs.append(np.mean(recalls))
    return float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))


def dataset_id_probe(
    X,
    y: np.ndarray,
    units: np.ndarray,
    part: np.ndarray,
    rng: np.random.Generator,
    *,
    top_k_columns: int = 5000,
    c_grid=(0.1, 1.0, 10.0),
) -> ProbeResult:
    """Sonda dataset-ID: regressione logistica L2 e gradient boosting; si riporta il MIGLIORE
    su validazione (una sonda debole sottostimerebbe l'identificabilita'), valutato sul test.

    X: matrice (n, d) densa o sparsa; y: id del dataset; units: id del soggetto (unico fra
    dataset); part: 'train'|'val'|'test' per campione. Le classi sono bilanciate DENTRO ogni
    parte. Il gradient boosting (che richiede una matrice densa) usa le `top_k_columns` colonne
    piu' frequenti nel train: dettaglio d'implementazione dei 524.288 bin dei codici."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y)
    part = np.asarray(part)
    units = np.asarray(units)
    idx = {}
    for p in ("train", "val", "test"):
        ii = np.flatnonzero(part == p)
        idx[p] = ii[balance_classes(y[ii], rng)]
    Xs = X.tocsr() if sparse.issparse(X) else np.asarray(X)

    def cols_for_gb():
        if not sparse.issparse(Xs):
            return None
        colsum = np.asarray(Xs[idx["train"]].sum(axis=0)).ravel()
        return np.argsort(-colsum)[:top_k_columns]

    gb_cols = cols_for_gb()

    def dense_gb(ii):
        block = Xs[ii]
        if gb_cols is not None:
            block = block[:, gb_cols]
        return block.toarray() if sparse.issparse(block) else block

    candidates = []  # (val_acc, nome, funzione di predizione sul test)
    for c in c_grid:
        lr = LogisticRegression(C=c, max_iter=2000)
        lr.fit(Xs[idx["train"]], y[idx["train"]])
        candidates.append((_balanced_accuracy(lr.predict(Xs[idx["val"]]), y[idx["val"]]), f"logreg_C{c}", lr.predict, None))
    gb = HistGradientBoostingClassifier(random_state=0)
    gb.fit(dense_gb(idx["train"]), y[idx["train"]])
    candidates.append((_balanced_accuracy(gb.predict(dense_gb(idx["val"])), y[idx["val"]]), "hist_gb", gb.predict, "dense"))

    best = max(candidates, key=lambda t: t[0])
    _, name, predict, kind = best
    te = idx["test"]
    pred = predict(dense_gb(te) if kind == "dense" else Xs[te])
    acc = _balanced_accuracy(pred, y[te])
    ci = _bootstrap_ci(pred, y[te], units[te], rng)
    return ProbeResult(acc, ci, 1.0 / len(np.unique(y)), name, int(len(te)))


def v3_passes(acc_codes: float, acc_bands: float) -> bool:
    """L'ancora si scarta se codici - bande > Y."""
    return (acc_codes - acc_bands) <= Y_POINTS
