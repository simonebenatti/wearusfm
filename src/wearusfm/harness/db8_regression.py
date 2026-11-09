"""Regressione delle 5 DoA di NinaPro DB8 da sEMG: dati, finestre, post-processing EMA e metriche (replica di TEMPONet, [Z21]).

Fonti e scelte: `docs/temponet_db8_replication.md`. In breve:
- dati: raw ufficiale `<raw_root>/DB8/S<s>_E1_A<a>.mat` (s = 1..12, a = 1, 2, 3), `emg` (T, 16) in volt e `glove` (T, 18), gia' a 2 kHz e
  sincronizzati (verificato il 01/10/2026, `wearusfm.ingest.ninapro_db8`); [Z21] sez. II-D2: segnali gia' filtrati passa-banda 10-500 Hz alla fonte,
  nessun altro preprocessing;
- bersaglio: 5 gradi di attuazione (DoA) = combinazioni lineari dei 18 DoF del guanto, matrice A di Krasoulis et al. 2019, materiale
  supplementare, eq. S2 (verificata il 09/11/2026 sul PDF dell'editore: la stampa ha 19 righe per 18 DoF, l'ultima tutta a zero; si usano le prime
  18, come il codice degli autori github.com/pulp-bio/event-based-semg-regression, commit 1920531b, `dataset/ninaprodb8.py`);
- finestre di 256 campioni (128 ms) con bersaglio all'ultimo campione della finestra; passo di 32 campioni (16 ms), il passo d'inferenza di [Z21]
  dichiarato da Zanghieri et al. 2023 (IWASI, sez. II-C: «every 16 ms, which is the same time step as the SoA work»);
- EMA causale sulle uscite ([Z21] eq. 2), alpha per soggetto, per DoA e per modello scelto con una griglia sui soli dati di training;
- metriche ([Z21] sez. III-A): MAE in gradi e accuratezza a tolleranza (frazione di istanti con MAE sulle 5 DoA sotto 10 o 15 gradi).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS_HZ = 2000.0
N_SUBJECTS = 12
ACQUISITIONS = (1, 2, 3)
TEST_ACQUISITION = 3
N_EMG = 16
N_DOF = 18
N_DOA = 5
DOA_NAMES = ("thumb_rotation", "thumb_flexion", "index_flexion", "middle_flexion", "ring_little_flexion")
WINDOW = 256  # 128 ms a 2 kHz, [Z21] sez. II-C2
STEP = 32  # 16 ms, passo d'inferenza di [Z21] secondo IWASI 2023
ALPHA_GRID = np.round(np.arange(0.0, 1.0, 0.01), 2)  # 0,00 ... 0,99
TOLERANCES_DEG = (10.0, 15.0)

# Krasoulis et al. 2019, Data Sheet 1, eq. S2: A^T con 18 righe (DoF del guanto) e 5 colonne (DoA nell'ordine di DOA_NAMES)
DOF_TO_DOA = np.zeros((N_DOF, N_DOA), dtype=np.float64)
for _row, _col, _val in (
    (0, 0, 0.639), (1, 0, 0.383), (2, 1, 1.0), (3, 0, -0.639), (4, 2, 0.4), (5, 2, 0.6), (6, 3, 0.4), (7, 3, 0.6),
    (9, 4, 0.1667), (10, 4, 0.3333), (12, 4, 0.1667), (13, 4, 0.3333), (16, 0, -0.19),
):
    DOF_TO_DOA[_row, _col] = _val
del _row, _col, _val


def acquisition_path(raw_root: str | Path, subject: int, acquisition: int) -> Path:
    if not 1 <= subject <= N_SUBJECTS or acquisition not in ACQUISITIONS:
        raise ValueError(f"soggetto {subject} o acquisizione {acquisition} fuori da DB8")
    return Path(raw_root) / "DB8" / f"S{subject}_E1_A{acquisition}.mat"


def glove_to_doa(glove: np.ndarray) -> np.ndarray:
    glove = np.asarray(glove, dtype=np.float64)
    if glove.ndim != 2 or glove.shape[1] != N_DOF:
        raise ValueError(f"glove atteso (T, {N_DOF}), ricevuto {glove.shape}")
    return glove @ DOF_TO_DOA


@dataclass
class Recording:
    emg: np.ndarray  # (T, 16) float32, volt
    doa: np.ndarray  # (T, 5) float32, gradi
    source: str

    @property
    def n_samples(self) -> int:
        return int(self.emg.shape[0])


RAW_VARIABLES = ["emg", "glove", "restimulus", "rerepetition"]
CACHE_ARRAYS = ("emg", "doa", "restimulus", "rerepetition")


def read_raw(path: str | Path) -> dict[str, np.ndarray]:
    """Legge da un .mat di DB8 `emg` (T, 16), `glove` (T, 18) e le etichette (T,) senza conversioni; controlla forme e lunghezze.

    La lettura con scipy e' lenta (minuti per file da 1 GB): la fa una volta il job di cache, i training leggono la cache .npy.
    """
    import scipy.io as sio

    mat = sio.loadmat(str(path), variable_names=RAW_VARIABLES)
    missing = [k for k in RAW_VARIABLES if k not in mat]
    if missing:
        raise ValueError(f"{path}: variabili mancanti {missing}")
    out = {k: np.asarray(mat[k]) for k in RAW_VARIABLES}
    if out["emg"].ndim != 2 or out["emg"].shape[1] != N_EMG:
        raise ValueError(f"{path}: emg atteso (T, {N_EMG}), ricevuto {out['emg'].shape}")
    if out["glove"].ndim != 2 or out["glove"].shape[1] != N_DOF:
        raise ValueError(f"{path}: glove atteso (T, {N_DOF}), ricevuto {out['glove'].shape}")
    n = out["emg"].shape[0]
    for k in ("glove", "restimulus", "rerepetition"):
        if out[k].shape[0] != n:
            raise ValueError(f"{path}: {k} lungo {out[k].shape[0]}, emg {n}")
    out["restimulus"] = out["restimulus"].reshape(n)
    out["rerepetition"] = out["rerepetition"].reshape(n)
    return out


MAX_TRIMMED_TAIL = 2000  # 1 s a 2 kHz


def finite_length(raw: dict[str, np.ndarray]) -> int:
    """Campioni utili: una coda finale con valori non finiti in emg o glove (al massimo MAX_TRIMMED_TAIL campioni) si scarta; ogni altro
    valore non finito e' un errore. Caso reale (09/11/2026, tutti i 36 file controllati): S5_E1_A1 ha NaN nelle 18 colonne del guanto nel solo
    ultimo campione."""
    n = raw["emg"].shape[0]
    bad = ~(np.isfinite(raw["emg"]).all(axis=1) & np.isfinite(raw["glove"]).all(axis=1))
    if not bad.any():
        return n
    first = int(np.flatnonzero(bad)[0])
    if not bad[first:].all():
        raise ValueError("valori non finiti in emg o glove non limitati alla coda della registrazione")
    if n - first > MAX_TRIMMED_TAIL:
        raise ValueError(f"coda non finita di {n - first} campioni, oltre i {MAX_TRIMMED_TAIL} ammessi")
    return first


def trim_raw(raw: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    """(raw senza la coda non finita, campioni scartati)."""
    n = finite_length(raw)
    return {k: v[:n] for k, v in raw.items()}, int(raw["emg"].shape[0] - n)


def cache_file(cache_dir: str | Path, subject: int, acquisition: int, array: str) -> Path:
    if array not in CACHE_ARRAYS:
        raise ValueError(f"array di cache sconosciuto: {array}")
    acquisition_path(".", subject, acquisition)  # controlla gli indici
    return Path(cache_dir) / f"S{subject}_A{acquisition}_{array}.npy"


def to_cache_arrays(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """emg float32 in volt (come alla fonte), DoA float32 in gradi, etichette int16; rifiuta valori non finiti."""
    if not np.isfinite(raw["emg"]).all() or not np.isfinite(raw["glove"]).all():
        raise ValueError("valori non finiti in emg o glove")
    return {
        "emg": np.ascontiguousarray(raw["emg"], dtype=np.float32),
        "doa": np.ascontiguousarray(glove_to_doa(raw["glove"]), dtype=np.float32),
        "restimulus": np.ascontiguousarray(raw["restimulus"], dtype=np.int16),
        "rerepetition": np.ascontiguousarray(raw["rerepetition"], dtype=np.int16),
    }


def load_cached(cache_dir: str | Path, subject: int, acquisition: int) -> Recording:
    emg = np.load(cache_file(cache_dir, subject, acquisition, "emg"))
    doa = np.load(cache_file(cache_dir, subject, acquisition, "doa"))
    if emg.ndim != 2 or emg.shape[1] != N_EMG or doa.shape != (emg.shape[0], N_DOA):
        raise ValueError(f"cache di S{subject} A{acquisition}: forme {emg.shape} e {doa.shape} inattese")
    if emg.dtype != np.float32 or doa.dtype != np.float32:
        raise ValueError(f"cache di S{subject} A{acquisition}: tipi {emg.dtype} e {doa.dtype}, attesi float32")
    return Recording(emg, doa, str(cache_file(cache_dir, subject, acquisition, "emg")))


def window_ends(n_samples: int, window: int = WINDOW, step: int = STEP) -> np.ndarray:
    """Indici dell'ultimo campione di ogni finestra completa: window-1, window-1+step, ..."""
    if n_samples < window:
        return np.zeros(0, dtype=np.int64)
    return np.arange(window - 1, n_samples, step, dtype=np.int64)


def ema(pred: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """EMA causale di [Z21] eq. 2, y'_t = alpha * y'_{t-1} + (1 - alpha) * y_t, con y'_0 = y_0; per colonna (alpha scalare o per colonna)."""
    from scipy.signal import lfilter

    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim != 2:
        raise ValueError("pred deve essere (T, D)")
    alphas = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (pred.shape[1],))
    if np.any(alphas < 0) or np.any(alphas >= 1):
        raise ValueError("alpha deve stare in [0, 1)")
    out = np.empty_like(pred)
    if pred.shape[0] == 0:
        return out
    for d, a in enumerate(alphas):
        if a == 0:
            out[:, d] = pred[:, d]
        else:
            out[:, d], _ = lfilter([1.0 - a], [1.0, -a], pred[:, d], zi=[a * pred[0, d]])
    return out


def tune_alpha(segments: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray = ALPHA_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Per DoA, l'alpha della griglia con il MAE minimo sui segmenti (pred, target) di validazione; l'EMA riparte a ogni segmento.

    Pareggi: vince l'alpha piu' piccolo. Ritorna (alpha per DoA, MAE di validazione per alpha e DoA, forma (len(grid), D)).
    """
    if not segments:
        raise ValueError("nessun segmento di validazione")
    d = segments[0][0].shape[1]
    abs_sum = np.zeros((len(grid), d))
    count = 0
    for pred, target in segments:
        if pred.shape != target.shape or pred.shape[1] != d:
            raise ValueError("pred e target devono avere la stessa forma (T, D)")
        for i, a in enumerate(grid):
            abs_sum[i] += np.abs(ema(pred, a) - target).sum(axis=0)
        count += pred.shape[0]
    mae = abs_sum / count
    best = np.argmin(mae, axis=0)  # argmin prende il primo minimo: l'alpha piu' piccolo
    return np.asarray(grid)[best], mae


def mae_per_doa(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64)).mean(axis=0)


def tolerance_accuracy(pred: np.ndarray, target: np.ndarray, tol_deg: float) -> float:
    """[Z21] eq. 3-4: frazione degli istanti in cui il MAE sulle DoA e' sotto la tolleranza (disuguaglianza stretta)."""
    inst = np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64)).mean(axis=1)
    return float((inst < tol_deg).mean())


def evaluate(pred: np.ndarray, target: np.ndarray) -> dict:
    per_doa = mae_per_doa(pred, target)
    out = {"mae_deg": float(per_doa.mean()), "mae_per_doa_deg": [float(v) for v in per_doa], "n_instants": int(pred.shape[0])}
    for tol in TOLERANCES_DEG:
        out[f"acc_{int(tol)}deg"] = tolerance_accuracy(pred, target, tol)
    return out
