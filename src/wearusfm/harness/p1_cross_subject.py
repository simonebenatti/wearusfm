"""Metrica P1 di D12 (firmata il 04/10/2026): **cross-soggetto sul ramo sparso**, sonda lineare sull'encoder congelato, accuratezza bilanciata sui
soggetti di test della classe A (split firmati, `splits/v1/splits_v1.json`). Protocollo proposto da AG in `docs/foglio_p1_cross_soggetto.md`, da
firmare; le scelte sono marcate li'. Solo numpy: le feature del modello sono in `harness.fm_features`, la sonda in `harness.probe`.

- **Dataset:** NinaPro DB2, DB3 (amputati) e DB6 (10 sessioni in 5 giorni): i dataset della classe A con soggetti di test e con le etichette
  allineate all'EMG (Camargo: etichette non ingerite; Zhang 2026: etichette nel tempo del video; DB4 e DB7: interi nel pretraining).
- **Etichette:** `restimulus` (DB2 e DB3 numerano i movimenti 1-49 attraverso i tre esercizi, DB6 le prese fino a 11; verificato il 05/10 sui
  dati processati); il riposo (0) e il riempimento (-1) non sono classi.
- **Finestre:** `window_s` senza sovrapposizione, ciascuna dentro un tratto con una sola etichetta e dentro un esercizio (`trials`) o una
  sessione (`benchmarks.windowize`, come UCI-EMG).
- **Soggetti:** la sonda si addestra sui soggetti di pretraining del dataset (l'encoder ne ha visto l'EMG, mai le etichette), la regolarizzazione
  si sceglie su `val_frac` di loro (almeno 1, seme fisso), il numero si legge sui soggetti di test. Le classi dei soggetti di test che non
  compaiono nel train si tolgono e si contano.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wearusfm.harness.benchmarks import windowize

P1_DATASETS = ("ninapro_db2", "ninapro_db3", "ninapro_db6")
LABEL = "restimulus"


@dataclass
class SessionWindows:
    dataset: str
    subject: str
    session: str
    windows: np.ndarray  # (N, T, C) float32 in unita' fisiche
    labels: np.ndarray  # (N,) int
    fs: float
    montage: dict


def sessions_of(root: Path, dataset: str, subject: str) -> list[Path]:
    """Le cartelle di sessione di un soggetto (DB2/DB3: `session1`; DB6: `D<d>_T<t>`), in ordine."""
    return sorted(p.parent for p in (root / dataset / subject).glob("*/metadata.json"))


def session_windows(session_dir: Path, dataset: str, *, window_s: float = 1.0, stride_s: float | None = None) -> SessionWindows:
    """Le finestre etichettate di una sessione processata (`data_int16.npy`, `labels.npz`, `metadata.json`)."""
    from wearusfm.data.processed import read_scale

    meta = json.loads((session_dir / "metadata.json").read_text())
    arr = np.load(session_dir / "data_int16.npy", mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{session_dir}: atteso (T, C), non {arr.shape}")
    with np.load(session_dir / "labels.npz") as lab:
        y = np.asarray(lab[LABEL], dtype=np.int64)
    if len(y) != arr.shape[0]:
        raise ValueError(f"{session_dir}: {len(y)} etichette per {arr.shape[0]} campioni")
    y = np.where(y < 0, 0, y)  # riempimento -1: come il riposo, mai una classe
    fs = float(meta["native_fs_hz"])
    scale = read_scale(meta)
    w_n = int(round(window_s * fs))
    s_n = int(round((stride_s or window_s) * fs))
    trials = meta.get("trials") or [{"offset": 0, "n_samples": arr.shape[0]}]
    ws, ys = [], []
    for t in trials:
        a, b = int(t["offset"]), int(t["offset"]) + int(t["n_samples"])
        x = np.asarray(arr[a:b], dtype=np.float32)
        if scale is not None:
            x = x / (np.asarray(scale, dtype=np.float32) if np.ndim(scale) else np.float32(scale))
        w, lab_w = windowize(x, y[a:b], window_samples=w_n, stride_samples=s_n)
        ws += w
        ys += lab_w
    c = arr.shape[1]
    windows = np.stack(ws).astype(np.float32) if ws else np.zeros((0, w_n, c), np.float32)
    return SessionWindows(dataset, session_dir.parent.name, session_dir.name, windows, np.asarray(ys, dtype=np.int64), fs, meta["montage"])


def subject_roles(splits: dict, dataset: str, val_frac: float = 0.1, seed: int = 0) -> dict[str, list[str]]:
    """{'train', 'val', 'test'}: test = i soggetti di test degli split firmati; val = `val_frac` dei soggetti di pretraining (per eccesso, almeno
    1), estratti con `seed`; train = gli altri soggetti di pretraining."""
    d = splits["datasets"][dataset]
    pre, test = sorted(d["pretraining"]), sorted(d["test"])
    if not test:
        raise ValueError(f"{dataset}: nessun soggetto di test negli split")
    n_val = max(1, math.ceil(val_frac * len(pre)))
    if n_val >= len(pre):
        raise ValueError(f"{dataset}: {len(pre)} soggetti di pretraining non bastano per train e validazione")
    val = sorted(np.random.default_rng(seed).choice(pre, size=n_val, replace=False).tolist())
    return {"train": [s for s in pre if s not in val], "val": val, "test": test}


def keep_train_classes(y_train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Maschere (una per array) delle finestre la cui classe compare nel train."""
    known = set(np.unique(y_train).tolist())
    return [np.isin(y, list(known)) for y in others]


def dataset_weights(n_test_subjects: list[int], weighting: str) -> np.ndarray:
    """Pesi dei dataset nella media P1: "subjects" = proporzionali ai soggetti di test (ogni soggetto conta uguale), "datasets" = uguali."""
    if weighting == "subjects":
        w = np.asarray(n_test_subjects, dtype=np.float64)
    elif weighting == "datasets":
        w = np.ones(len(n_test_subjects))
    else:
        raise ValueError(f"pesatura {weighting!r} sconosciuta")
    return w / w.sum()


def mean_bacc_bootstrap_se(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], n_boot: int = 1000, seed: int = 0,
                           weights: np.ndarray | None = None) -> float:
    """SE della media pesata fra dataset (`weights`, default uguali) dell'accuratezza bilanciata: a ogni ricampionamento, i soggetti di test di
    ciascun dataset si ricampionano con ripetizione (stratificato per dataset); parts = [(y_true, y_pred, soggetti)] per dataset."""
    from sklearn.metrics import balanced_accuracy_score

    w = np.full(len(parts), 1.0 / len(parts)) if weights is None else np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    prepared = []
    for y, p, s in parts:
        subs = np.unique(s)
        prepared.append((y, p, subs, {k: np.flatnonzero(s == k) for k in subs}))
    vals = []
    for _ in range(n_boot):
        per = []
        for y, p, subs, idx_by in prepared:
            idx = np.concatenate([idx_by[k] for k in rng.choice(subs, size=len(subs), replace=True)])
            if len(np.unique(y[idx])) < 2:
                break
            per.append(balanced_accuracy_score(y[idx], p[idx]))
        if len(per) == len(prepared):
            vals.append(float(np.dot(w, per)))
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
