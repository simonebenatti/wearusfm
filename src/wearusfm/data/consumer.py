"""Consumatore simulato per l'asse M (spec §7-§8): un processo per rank che consuma un
batch ogni `t_passo`, con una barriera fra i rank a ogni passo.

Questo modulo contiene solo la parte di calcolo pura (testabile su CPU, senza
distribuzione reale): dati i tempi di fetch misurati per rank e per batch, calcola la
frazione di tempo in attesa e le percentili dell'intervallo fra batch. L'esecuzione
distribuita vera (torch.distributed, 4 rank su Leonardo) e' fuori da questo modulo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AsseMMetrics:
    waiting_fraction: float  # spec §8: "e' la metrica che decide"
    p50_wall_s: float
    p95_wall_s: float
    p99_wall_s: float
    windows_per_s: float  # throughput a vuoto (senza il floor della soglia per batch)
    t_passo_batch_s: float  # soglia per BATCH usata per il confronto (t_passo_s * batch_size)


def compute_asse_m_metrics(fetch_times_s: np.ndarray, t_passo_s: float, *, batch_size: int) -> AsseMMetrics:
    """`fetch_times_s`: tempo di fetch di un BATCH (non di una singola finestra), forma
    (n_rank, n_batch) o (n_batch,).

    `t_passo_s` e' il ritmo richiesto PER FINESTRA (v. `required_windows_per_s`, che
    ritorna finestre/s - qui serve il reciproco). La soglia con cui si confronta il
    tempo di fetch di un batch e' quindi `t_passo_s * batch_size`, non `t_passo_s` da
    solo: un batch contiene `batch_size` finestre, e il consumatore aspetta finche' non
    ha ricevuto l'intero batch. Confrontare il tempo-per-batch contro il tempo-per-una-
    sola-finestra sovrastima sistematicamente l'attesa (bug corretto dopo il primo run
    reale su Leonardo, dove dava waiting_fraction=0.94 con un dataloader in realta' piu'
    veloce del richiesto).

    Con la barriera fra rank, il tempo di parete per batch e' il massimo sui rank
    (spec §7): un rank lento ferma gli altri. Il tempo di parete non puo' scendere
    sotto la soglia per batch (il consumatore non consuma piu' in fretta del modello).
    """
    arr = np.asarray(fetch_times_s, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    per_batch_fetch = arr.max(axis=0)  # massimo sui rank, spec §7
    t_passo_batch_s = t_passo_s * batch_size

    wall = np.maximum(per_batch_fetch, t_passo_batch_s)
    wait = np.maximum(per_batch_fetch - t_passo_batch_s, 0.0)
    waiting_fraction = float(wait.sum() / wall.sum()) if wall.sum() > 0 else 0.0

    mean_fetch = per_batch_fetch.mean()
    windows_per_s = float(batch_size / mean_fetch) if mean_fetch > 0 else float("inf")

    return AsseMMetrics(
        waiting_fraction=waiting_fraction,
        p50_wall_s=float(np.percentile(wall, 50)),
        p95_wall_s=float(np.percentile(wall, 95)),
        p99_wall_s=float(np.percentile(wall, 99)),
        windows_per_s=windows_per_s,
        t_passo_batch_s=t_passo_batch_s,
    )


def required_windows_per_s(
    n_params: float,
    *,
    mfu: float = 0.4,
    k_latents: int = 64,
    context_s: float = 4.0,
    patch_ms: float = 25.0,
    a100_peak_flops: float = 312e12,
) -> float:
    """Ritmo richiesto per GPU (piano_operativo_v10.md, passo 0, D6):

        finestre/s ~= MFU * picco_FLOPS / (6 * N * K * contesto/patch)

    ~70 finestre/s a 30M, ~20 a 100M con K=64, contesto 4s, patch 25ms, MFU 40% (spec §9).
    """
    steps_per_window = context_s / (patch_ms / 1000.0)
    denom = 6.0 * n_params * k_latents * steps_per_window
    return mfu * a100_peak_flops / denom
