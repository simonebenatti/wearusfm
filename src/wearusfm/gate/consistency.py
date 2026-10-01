"""Gate di consistenza al ricampionamento (D8, passo 3). Definizioni FIRMATE il 01/10/2026: `docs/proposta_gate_d8.md`, `docs/decisioni.md` (D8a).

Per ogni caso (1 kHz: taglio 450 Hz, decimazione 2; 200 Hz: taglio 90 Hz, decimazione 10) e per ogni seme del front-end:
- A = registrazione a 2 kHz filtrata passa-basso (Butterworth ordine 8, fase zero) e lasciata a 2 kHz; B = A decimata (un campione ogni 2 o 10);
- feature del front-end su A e su B, sulle stesse patch in tempo fisico; si confrontano solo le patch della finestra da 4 s, calcolate con un margine
  di 0,5 s per lato (cosi' il contesto del front-end non tocca i bordi);
- (i) errore relativo RMS ||F_A - F_B|| / ||F_A|| nel totale e per famiglia: tutti <= 5%;
- (ii) sonda A-contro-B sulle finestre (media e deviazione standard nel tempo delle feature, per canale), split per soggetto 60/20/20 con le due
  versioni della stessa finestra nella stessa parte, la piu' forte fra regressione logistica e gradient boosting (scelta sulla validazione), intervallo
  al 95% con bootstrap per soggetto: accuratezza <= 0,55 e intervallo che contiene 0,50.
Il gate passa solo se passano entrambi i casi con tutti i semi.

Scelta di implementazione dichiarata (non nelle definizioni firmate): le feature della sonda si standardizzano con media e deviazione standard del solo
train, perche' la regressione logistica non sia penalizzata dalla scala; il gradient boosting non ne risente.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal

from wearusfm.tokenizer_checks import metrics as M
from wearusfm.tokenizer_checks.continuous_features import standardize_with_train

FS_A = 2000.0
CASES = {"1kHz": (450.0, 2), "200Hz": (90.0, 10)}  # (taglio Hz, fattore di decimazione)
FILTER_ORDER = 8
WINDOW_S = 4.0
MARGIN_S = 0.5
MAX_REL_ERROR = 0.05
MAX_PROBE_ACC = 0.55


@dataclass
class Window:
    subject: str
    data: np.ndarray  # (C, n) a 2 kHz, finestra piu' i margini


def make_versions(x: np.ndarray, cutoff_hz: float, decimation: int) -> tuple[np.ndarray, np.ndarray]:
    """x (C, n) a 2 kHz -> (A, B): A filtrata e lasciata a 2 kHz, B = A decimata. Si filtra prima di decimare: stesso contenuto."""
    sos = signal.butter(FILTER_ORDER, cutoff_hz, btype="low", fs=FS_A, output="sos")
    a = signal.sosfiltfilt(sos, x, axis=-1)
    return a, a[..., ::decimation]


def draw_windows(sessions: Sequence[tuple[str, np.ndarray, Sequence[tuple[int, int]]]], per_subject: int, rng: np.random.Generator,
                 align: int = 10) -> list[Window]:
    """Finestre da WINDOW_S piu' MARGIN_S per lato, a caso dentro ogni sessione (soggetto, dati (n, C) a 2 kHz, intervalli da evitare
    [(inizio, fine)] in campioni). L'inizio e' multiplo di `align` (la decimazione per 10 parte dallo stesso campione)."""
    total = int(round((WINDOW_S + 2 * MARGIN_S) * FS_A))
    out = []
    for subject, data, avoid in sessions:
        n = data.shape[0]
        if n < total:
            continue
        got, tries = 0, 0
        while got < per_subject and tries < 100 * per_subject:
            tries += 1
            s = int(rng.integers(0, (n - total) // align + 1)) * align
            if any(s < e and s + total > b for b, e in avoid):
                continue
            out.append(Window(subject, np.asarray(data[s : s + total], dtype=np.float64).T))
            got += 1
    return out


def _interior(feat, fe) -> slice:
    k = int(round(MARGIN_S / fe.patch_s))
    n = int(round(WINDOW_S / fe.patch_s))
    return slice(k, k + n)


def features_case(fe, windows: Sequence[Window], cutoff_hz: float, decimation: int) -> dict:
    """Misura (i) e vettori per la sonda, per un caso e un front-end. Usa torch e NON scikit-learn: su macOS i due insieme nello stesso processo
    vanno in segfault (visto il 30/09 e il 01/10/2026), quindi la sonda si calcola in un altro processo (`probe_case`)."""
    import torch

    sl = fe.family_slices()
    sq_diff = {f: 0.0 for f in sl} | {"total": 0.0}
    sq_ref = {f: 0.0 for f in sl} | {"total": 0.0}
    xs, ys, units = [], [], []
    for w in windows:
        a, b = make_versions(w.data, cutoff_hz, decimation)
        with torch.no_grad():
            fa = fe(torch.from_numpy(np.ascontiguousarray(a))[None], FS_A)[0].numpy()
            fb = fe(torch.from_numpy(np.ascontiguousarray(b))[None], FS_A / decimation)[0].numpy()
        it = _interior(fa, fe)
        fa, fb = fa[:, it], fb[:, it]  # (C, n_patch, d)
        for f, sli in sl.items():
            sq_diff[f] += float(((fa[..., sli] - fb[..., sli]) ** 2).sum())
            sq_ref[f] += float((fa[..., sli] ** 2).sum())
        sq_diff["total"] += float(((fa - fb) ** 2).sum())
        sq_ref["total"] += float((fa**2).sum())
        for label, feat in ((0, fa), (1, fb)):
            xs.append(np.concatenate([feat.mean(axis=1).ravel(), feat.std(axis=1).ravel()]))
            ys.append(label)
            units.append(w.subject)
    rel = {k: float(np.sqrt(sq_diff[k] / sq_ref[k])) for k in sq_diff}
    return {"rel_error": rel, "x": np.stack(xs), "y": np.array(ys), "units": np.array(units)}


def probe_case(x: np.ndarray, y: np.ndarray, units: np.ndarray, split_seed: int = 0) -> dict:
    """Misura (ii): sonda A-contro-B. Usa scikit-learn e NON torch."""
    subjects = sorted(set(units.tolist()))
    part_map = M.split_subjects({"gate": subjects}, np.random.default_rng([split_seed, 8, 0]))
    part = np.array([part_map[("gate", u)] for u in units])
    r = M.dataset_id_probe(standardize_with_train(x, part), y, units, part, np.random.default_rng([split_seed, 8, 1]))
    return {"balanced_accuracy": r.balanced_accuracy, "ci95": list(r.ci95), "chance": r.chance, "model": r.model, "n_test": r.n_test,
            "n_subjects": len(subjects)}


def verdict(rel_error: dict, probe: dict) -> dict:
    passes_err = all(v <= MAX_REL_ERROR for v in rel_error.values())
    passes_probe = bool(probe["balanced_accuracy"] <= MAX_PROBE_ACC and probe["ci95"][0] <= 0.5 <= probe["ci95"][1])
    return {"passes_error": passes_err, "passes_probe": passes_probe, "passes": bool(passes_err and passes_probe)}
