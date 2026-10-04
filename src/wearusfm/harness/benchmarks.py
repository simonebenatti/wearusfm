"""I due benchmark «mai visti» del passo 5 (EPN-612 e UCI-EMG; v10 §2.3, piano passo 5), con le finestre del foglio della replica di NeuroRVQ
(`docs/foglio_replica_neurorvq.md`, scelte marcate li'): gli stessi dati per la replica e per la sonda sul nostro FM (D12, metrica P2).

- **EPN-612** (200 Hz, 8 canali Myo): i campioni etichettati gia' segmentati da 5 s (`trainingSamples`: in entrambe le cartelle del dataset hanno
  `gestureName`; i `testingSamples` no, verificato il 23/09). Un soggetto = un utente; gli utenti delle due cartelle hanno numeri che si ripetono,
  quindi l'identificatore e' `cartella/utente`. 6 classi.
- **UCI-EMG** (1 kHz, 8 canali Myo, fatto 15): serie continue con un'etichetta per campione; finestre da `window_s` senza sovrapposizione dentro un
  tratto con una sola etichetta, solo le classi 1-6 (scelta del foglio: la 0 non e' marcata, la 7 non e' stata eseguita da tutti).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EPN_CHANNELS = tuple(f"ch{i}" for i in range(1, 9))
EPN_FS = 200.0
UCI_FS = 1000.0
UCI_CHANNELS = 8
UCI_UNMARKED = 0


def load_epn_user(path: Path) -> tuple[list[np.ndarray], list[str], float]:
    """Un file utente EPN-612: (finestre (T, 8), etichette, fs). Solo `trainingSamples` (gli unici con l'etichetta)."""
    with open(path) as f:
        data = json.load(f)
    fs_hz = float(data["generalInfo"]["samplingFrequencyInHertz"])
    windows: list[np.ndarray] = []
    labels: list[str] = []
    for sample in data.get("trainingSamples", {}).values():
        emg = sample["emg"]
        channels = [np.asarray(emg[k], dtype=np.float64) for k in EPN_CHANNELS]
        if len({len(c) for c in channels}) != 1:
            continue  # canali di lunghezza diversa: campione malformato, si scarta
        windows.append(np.stack(channels, axis=1))
        labels.append(sample["gestureName"])
    return windows, labels, fs_hz


def fix_length(w: np.ndarray, n: int) -> np.ndarray:
    """(T, C) -> (n, C): taglia le piu' lunghe e completa con zeri le piu' corte, come `fix_length` del codice di NeuroRVQ
    (`third_party/neurorvq/preprocessing/preprocessing_emg_example.py`)."""
    return w[:n] if len(w) >= n else np.pad(w, ((0, n - len(w)), (0, 0)))


def load_epn612(roots: list[Path], n_users: int | None = None, seed: int = 0, window_s: float | None = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Tutti gli utenti delle cartelle `roots` (es. trainingJSON e testingJSON), o `n_users` estratti con `seed`. Ritorna (finestre (N, T, 8),
    etichette (N,), soggetti (N,), fs). Lunghezza fissa `window_s` (5 s, come nel paper: i campioni di EPN-612 non hanno tutti la stessa
    lunghezza), con `fix_length`; None = taglio alla piu' corta (com'era: il collaudo 59336240 ha mostrato che porta tutto a 2,4 s)."""
    user_files = [(root.name, p, p / f"{p.name}.json") for root in roots for p in sorted(root.glob("user*")) if (p / f"{p.name}.json").exists()]
    if not user_files:
        raise FileNotFoundError(f"nessun user*/user*.json sotto {roots}")
    if n_users is not None and n_users < len(user_files):
        pick = np.random.default_rng(seed).choice(len(user_files), size=n_users, replace=False)
        user_files = [user_files[i] for i in sorted(pick)]
    windows, labels, subjects, fs_seen = [], [], [], set()
    for folder, user_dir, path in user_files:
        w, lab, fs = load_epn_user(path)
        fs_seen.add(fs)
        windows += w
        labels += lab
        subjects += [f"{folder}/{user_dir.name}"] * len(w)
    if len(fs_seen) != 1:
        raise ValueError(f"frequenza non uniforme fra utenti: {fs_seen}")
    fs = fs_seen.pop()
    n = min(len(w) for w in windows) if window_s is None else int(round(window_s * fs))
    return np.stack([fix_length(w, n) for w in windows]), np.array(labels), np.array(subjects), fs


def load_uci_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Un file .txt di UCI-EMG: (campioni (T, 8), etichette (T,)). Righe col numero sbagliato di colonne scartate con un avviso (un file scaricato ha
    l'ultima riga troncata, verificato il 24/09)."""
    n_cols = 1 + UCI_CHANNELS + 1
    rows, dropped = [], 0
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != n_cols:
                dropped += 1
                continue
            rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError(f"{path}: nessuna riga valida")
    if dropped:
        print(f"  attenzione: {path.name}: scartate {dropped} righe malformate", file=sys.stderr)
    raw = np.array(rows)
    return raw[:, 1:1 + UCI_CHANNELS], raw[:, 1 + UCI_CHANNELS].astype(np.int64)


def windowize(samples: np.ndarray, labels: np.ndarray, *, window_samples: int, stride_samples: int,
              classes: tuple[int, ...] | None = None) -> tuple[list[np.ndarray], list[int]]:
    """Finestre di lunghezza fissa con un'etichetta sola (v10 §8: niente etichette ambigue, niente riposo dalle pause): una finestra a cavallo di
    un cambio di classe, non marcata (0) o di una classe fuori da `classes` si scarta."""
    out_w, out_y = [], []
    for start in range(0, len(samples) - window_samples + 1, stride_samples):
        u = np.unique(labels[start:start + window_samples])
        if len(u) != 1 or u[0] == UCI_UNMARKED or (classes is not None and int(u[0]) not in classes):
            continue
        out_w.append(samples[start:start + window_samples])
        out_y.append(int(u[0]))
    return out_w, out_y


def load_uci_emg(root: Path, *, window_s: float = 1.0, stride_s: float = 1.0, classes: tuple[int, ...] | None = (1, 2, 3, 4, 5, 6),
                 n_subjects: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ritorna (finestre (N, T, 8), etichette (N,), soggetti (N,), fs). Default: finestre da 1 s senza sovrapposizione, classi 1-6 (foglio)."""
    subject_dirs = sorted(p for p in root.rglob("*") if p.is_dir() and p.name.isdigit())
    if not subject_dirs:
        raise FileNotFoundError(f"nessuna cartella soggetto sotto {root}")
    if n_subjects is not None and n_subjects < len(subject_dirs):
        pick = np.random.default_rng(seed).choice(len(subject_dirs), size=n_subjects, replace=False)
        subject_dirs = [subject_dirs[i] for i in sorted(pick)]
    ws, ys, ss = [], [], []
    w_n, s_n = int(round(window_s * UCI_FS)), int(round(stride_s * UCI_FS))
    for d in subject_dirs:
        for f in sorted(d.glob("*.txt")):
            samples, labels = load_uci_file(f)
            w, y = windowize(samples, labels, window_samples=w_n, stride_samples=s_n, classes=classes)
            ws += w
            ys += y
            ss += [d.name] * len(w)
    if not ws:
        raise ValueError("nessuna finestra valida")
    return np.stack(ws), np.array(ys), np.array(ss), UCI_FS
