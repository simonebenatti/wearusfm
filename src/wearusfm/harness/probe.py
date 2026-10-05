"""Sonda lineare su feature congelate (passo 5; regime «encoder congelato + probe» di v10 §8; metriche primarie di D12, firmata il 04/10/2026).

- **Split per soggetto** (`harness.splits`): la sonda si addestra sui soggetti di train, la regolarizzazione si sceglie sui soggetti di validazione,
  il numero si legge sui soggetti di test. Mai finestre dello stesso soggetto in due split.
- **Normalizzazione delle feature stimata solo sul train** (v10 §8).
- **Sonda:** regressione logistica multinomiale (sklearn, lbfgs) con pesi di classe bilanciati; C scelto su una griglia fissa dalla accuratezza
  bilanciata di validazione; poi riaddestrata sul solo train con quel C (la validazione non entra nel modello valutato).
- **Errore standard (D12):** bootstrap sui soggetti di test (ricampionati con ripetizione, 1.000 volte, seme fisso): la deviazione standard
  dell'accuratezza bilanciata sui campioni bootstrap. Con piu' seed del modello, `combine_seeds` combina i risultati.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass
class ProbeResult:
    test_bacc: float
    test_bacc_se: float  # bootstrap sui soggetti di test
    val_bacc: float
    train_bacc: float
    c: float
    n_train: int
    n_val: int
    n_test: int
    n_test_subjects: int
    test_pred: np.ndarray | None = field(default=None, repr=False)  # predizioni sul test (per gli errori standard aggregati, es. P1)


def bootstrap_bacc_se(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray, n_boot: int = 1000, seed: int = 0) -> float:
    """Deviazione standard dell'accuratezza bilanciata ricampionando i SOGGETTI di test con ripetizione (le finestre di un soggetto restano
    insieme: non sono indipendenti)."""
    rng = np.random.default_rng(seed)
    subs = np.unique(subjects)
    idx_by = {s: np.flatnonzero(subjects == s) for s in subs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def linear_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                 test_subjects: np.ndarray, *, c_grid=C_GRID, n_boot: int = 1000, seed: int = 0) -> ProbeResult:
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("servono finestre in train, validazione e test")
    scaler = StandardScaler().fit(x_train)  # solo sul train
    xt, xv, xs = scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)

    def fit(c: float) -> LogisticRegression:
        return LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed).fit(xt, y_train)

    scores = {c: balanced_accuracy_score(y_val, fit(c).predict(xv)) for c in c_grid}
    best = max(c_grid, key=lambda c: (scores[c], -c))  # a parita', la regolarizzazione piu' forte
    model = fit(best)
    pred = model.predict(xs)
    return ProbeResult(
        test_bacc=float(balanced_accuracy_score(y_test, pred)),
        test_bacc_se=bootstrap_bacc_se(np.asarray(y_test), pred, np.asarray(test_subjects), n_boot, seed),
        val_bacc=float(scores[best]),
        train_bacc=float(balanced_accuracy_score(y_train, model.predict(xt))),
        c=float(best), n_train=len(xt), n_val=len(xv), n_test=len(xs), n_test_subjects=int(len(np.unique(test_subjects))), test_pred=pred,
    )


def combine_seeds(results: list[ProbeResult]) -> tuple[float, float]:
    """Media fra seed ed errore standard combinato (D12). I seed hanno gli STESSI soggetti di test, quindi la parte dovuta al campione di soggetti
    non si riduce mediando: SE^2 = media dei SE^2 di bootstrap + varianza fra seed / n. Con un solo seed: il suo SE."""
    b = np.array([r.test_bacc for r in results], dtype=np.float64)
    se = np.array([r.test_bacc_se for r in results], dtype=np.float64)
    n = len(results)
    if n == 1:
        return float(b[0]), float(se[0])
    return float(b.mean()), float(np.sqrt((se ** 2).mean() + b.var(ddof=1) / n))
