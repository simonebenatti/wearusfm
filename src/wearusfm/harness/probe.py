"""Sonda lineare su feature congelate (passo 5; regime «encoder congelato + probe» di v10 §8; metriche primarie di D12, firmata il 04/10/2026).

- **Split per soggetto** (`harness.splits`): la sonda si addestra sui soggetti di train, la regolarizzazione si sceglie sui soggetti di validazione,
  il numero si legge sui soggetti di test. Mai finestre dello stesso soggetto in due split.
- **Normalizzazione delle feature stimata solo sul train** (v10 §8).
- **Sonda:** regressione logistica multinomiale (sklearn, lbfgs) con pesi di classe bilanciati; C scelto su una griglia fissa dalla accuratezza
  bilanciata di validazione; poi riaddestrata sul solo train con quel C (la validazione non entra nel modello valutato).
- **Errore standard (D12):** bootstrap sui soggetti di test (ricampionati con ripetizione, 1.000 volte, seme fisso): la deviazione standard
  dell'accuratezza bilanciata sui campioni bootstrap. Con piu' seed del modello, `combine_seeds` combina i risultati.
- **Audit per singolo fit (29/10/2026, controllo della convergenza delle sonde di B19618, `docs/sonde_convergenza_20261029.md`):** con
  `audit=[...]` ogni `.fit` (i sei candidati e il riaddestramento) lascia un record strutturato: `fit_phase` (candidate | refit), C,
  `get_params()`, dimensioni e classi del train, firma dello scaler, `n_iter_`, tempo, warning emessi DENTRO la chiamata (categoria,
  messaggio, posizione), raggiungimento del limite, eccezione. I warning catturati vengono riemessi subito dopo con gli stessi filtri
  dell'utente (nessuna soppressione); un'eccezione si registra e si rilancia. Se l'oggetto `audit` ha un metodo `persist(record)`, viene
  chiamato appena un record e' definitivo (candidato: dopo il punteggio di validazione; riaddestramento: dopo il fit; eccezione: prima del
  rilancio), cosi' chi chiama puo' salvarlo su disco fit per fit. Con `audit=None` (default) il percorso di calcolo e' identico a prima;
  `max_iter` resta 3000 se non si chiede altro.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MAX_ITER = 3000  # il limite di iterazioni del solver di sempre (lbfgs)
TIE_BREAK = "accuratezza bilanciata di validazione massima; a parita' il C minore (regolarizzazione piu' forte)"


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
    val_scores: dict | None = field(default=None, repr=False)  # {C: accuratezza bilanciata di validazione} di ogni candidato
    solver_params: dict | None = field(default=None, repr=False)  # get_params() del modello riaddestrato
    bootstrap_warnings: int | None = field(default=None, repr=False)  # audit: warning emessi dal bootstrap dell'errore standard (metrica, non fit)


def _json_safe(value):
    """Valori di numpy e tuple in tipi JSON (per i record dell'audit e la ricetta del solver)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def solver_recipe(max_iter: int = MAX_ITER, *, seed: int = 0, c_grid=C_GRID, scale: bool = True) -> dict:
    """La ricetta del solver come la vede sklearn nell'ambiente in cui gira: `get_params()` di un modello della griglia (C a parte, che varia
    per candidato), griglia, limite di iterazioni, regola di scelta, riaddestramento, scala e versioni delle librerie. E' la firma che i
    confronti del controllo di convergenza devono trovare uguale fra i due encoder di una stessa fase (fra le fasi cambia solo `max_iter`)."""
    import scipy
    import sklearn

    params = LogisticRegression(C=1.0, max_iter=max_iter, class_weight="balanced", random_state=seed).get_params()
    params.pop("C")
    return {"estimator": "sklearn.linear_model.LogisticRegression", "params": _json_safe(params), "c_grid": [float(c) for c in c_grid],
            "max_iter": int(max_iter), "tie_break": TIE_BREAK, "refit": "sul solo train del fold, col C scelto sulla validazione",
            "scaler": ("StandardScaler stimato sul solo train del fold" if scale
                       else "nessuna nuova scala: ingressi gia' standardizzati (e ridotti) dal chiamante sul solo train del fold"),
            "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}}


def _described(caught) -> list[dict]:
    return [{"category": w.category.__name__, "message": str(w.message), "filename": str(w.filename), "lineno": int(w.lineno)} for w in caught]


def _reemit(caught) -> None:
    """Riemette i warning catturati con i filtri correnti dell'utente: la cattura non li nasconde."""
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def _fit_audited(model: LogisticRegression, x, y, record: dict) -> LogisticRegression:
    """Un singolo `.fit` con i warning emessi dentro la sola chiamata, il tempo, `n_iter_` e il raggiungimento del limite nel record; i warning
    vengono riemessi subito dopo; un'eccezione si registra (`exception`) e si rilancia: la cattura non trasforma un errore in un successo."""
    t0 = time.perf_counter()
    caught: list = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                model.fit(x, y)
            finally:
                caught = list(w)
    except BaseException as exc:  # noqa: BLE001 -- registrato e rilanciato tale e quale
        record.update(elapsed_s=time.perf_counter() - t0, warnings=_described(caught), exception=f"{type(exc).__name__}: {exc}")
        _reemit(caught)
        raise
    n_iter = np.atleast_1d(np.asarray(getattr(model, "n_iter_", -1)))
    described = _described(caught)
    convergence = sum(d["category"] == "ConvergenceWarning" for d in described)
    reached = bool(int(n_iter.max()) >= int(model.max_iter))
    record.update(elapsed_s=time.perf_counter() - t0, n_iter=int(n_iter.max()), n_iter_all=[int(v) for v in n_iter], warnings=described,
                  convergence_warnings=int(convergence), reached_limit=reached, converged_by_criterion=bool(not reached and convergence == 0),
                  anomalous_stop=bool(convergence > 0 and not reached), exception=None)
    _reemit(caught)
    return model


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
                 test_subjects: np.ndarray, *, c_grid=C_GRID, n_boot: int = 1000, seed: int = 0, scale: bool = True,
                 max_iter: int = MAX_ITER, audit: list | None = None, context: dict | None = None) -> ProbeResult:
    """`max_iter`: limite di iterazioni di OGNI fit (candidati e riaddestramento; default quello di sempre). `audit`: lista a cui aggiungere un
    record per fit (vedi il modulo), con le chiavi di `context` (encoder, dataset, fold, ...) copiate in ciascun record; `None` = nessun audit."""
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("servono finestre in train, validazione e test")
    scaler_sha = None
    if scale:
        scaler = StandardScaler().fit(x_train)  # solo sul train
        xt, xv, xs = scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)
        if audit is not None:
            from wearusfm.harness.downstream_ab import fingerprint

            scaler_sha = fingerprint(np.asarray(scaler.mean_, dtype=np.float64), np.asarray(scaler.scale_, dtype=np.float64))
    else:  # readout per canale (Simone, 23/10/2026): il chiamante ha gia' standardizzato (e ridotto) sul solo train; nessuna nuova scala
        xt, xv, xs = np.asarray(x_train), np.asarray(x_val), np.asarray(x_test)
    if audit is not None:
        classes, counts = np.unique(np.asarray(y_train), return_counts=True)
        base = {**(context or {}), "max_iter": int(max_iter), "n_train": int(len(xt)), "n_features": int(np.asarray(xt).shape[1]),
                "dtype": str(np.asarray(xt).dtype), "classes": _json_safe(classes), "class_counts": [int(v) for v in counts],
                "scale": bool(scale), "scaler_sha256": scaler_sha}

    persist = getattr(audit, "persist", None) if audit is not None else None

    def fit(c: float, fit_phase: str) -> LogisticRegression:
        model = LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced", random_state=seed)
        if audit is None:
            return model.fit(xt, y_train)
        record = {**base, "fit_phase": fit_phase, "C": float(c), "solver_params": _json_safe(model.get_params()), "selected": None}
        audit.append(record)  # prima del fit: un'eccezione lascia il record con `exception`
        try:
            return _fit_audited(model, xt, y_train, record)
        except BaseException:
            if persist is not None:
                persist(record)  # il record del fallimento e' su disco prima del rilancio
            raise

    scores: dict = {}
    mine: list = []  # i record di questa chiamata (l'audit puo' contenere quelli di altri fold e dataset)
    for c in c_grid:
        model = fit(c, "candidate")
        if audit is None:
            scores[c] = balanced_accuracy_score(y_val, model.predict(xv))
            continue
        mine.append(audit[-1])
        with warnings.catch_warnings(record=True) as w:  # i warning della metrica (classi assenti) restano separati da quelli del fit
            warnings.simplefilter("always")
            scores[c] = balanced_accuracy_score(y_val, model.predict(xv))
        _reemit(w)
        audit[-1].update(val_bacc=float(scores[c]), scoring_warnings=_described(w))
        if persist is not None:
            persist(audit[-1])
    best = max(c_grid, key=lambda c: (scores[c], -c))  # a parita', la regolarizzazione piu' forte
    for record in mine:
        record["selected"] = bool(record["C"] == float(best))
    model = fit(best, "refit")
    if persist is not None:
        persist(audit[-1])
    pred = model.predict(xs)
    test_bacc = float(balanced_accuracy_score(y_test, pred))  # prima del bootstrap, come sempre
    if audit is None:
        se = bootstrap_bacc_se(np.asarray(y_test), pred, np.asarray(test_subjects), n_boot, seed)
        n_bw = None
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            se = bootstrap_bacc_se(np.asarray(y_test), pred, np.asarray(test_subjects), n_boot, seed)
        _reemit(w)
        n_bw = len(w)
    return ProbeResult(
        test_bacc=test_bacc,
        test_bacc_se=se,
        val_bacc=float(scores[best]),
        train_bacc=float(balanced_accuracy_score(y_train, model.predict(xt))),
        c=float(best), n_train=len(xt), n_val=len(xv), n_test=len(xs), n_test_subjects=int(len(np.unique(test_subjects))), test_pred=pred,
        val_scores={float(c): float(v) for c, v in scores.items()}, solver_params=_json_safe(model.get_params()), bootstrap_warnings=n_bw,
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
