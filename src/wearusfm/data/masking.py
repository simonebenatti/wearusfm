"""Masking multiscala (v10 §6.4) applicato al campione sintetico.

Tre scale temporali + masking spaziale (~40% del budget) + quota opzionale di slab per
l'ancora RVQ (default spenta, spec §4). Le frazioni esatte del budget di masking e la
sua composizione sono decisioni ancora aperte in v10 (§12, D10): qui sono parametri
espliciti del benchmark, non valori presi da v10.
"""

from __future__ import annotations

import numpy as np

from wearusfm.data.manifest import ChannelGroup

TEMPORAL_SCALES_MS: tuple[tuple[float, float], ...] = (
    (20.0, 50.0),
    (100.0, 300.0),
    (500.0, 2000.0),
)


def temporal_mask(
    n_samples: int,
    fs_hz: float,
    rng: np.random.Generator,
    *,
    budget_fraction: float = 0.5,
    scales_ms: tuple[tuple[float, float], ...] = TEMPORAL_SCALES_MS,
) -> np.ndarray:
    """Maschera booleana (True = mascherato) sull'asse tempo, costruita da intervalli
    campionati dalle tre scale di v10 §6.4, fino a coprire `budget_fraction` del tempo."""
    mask = np.zeros(n_samples, dtype=bool)
    target = int(n_samples * budget_fraction)
    guard = 0
    while mask.sum() < target and guard < 10_000:
        guard += 1
        lo_ms, hi_ms = scales_ms[rng.integers(len(scales_ms))]
        span_samples = max(1, round(rng.uniform(lo_ms, hi_ms) * fs_hz / 1000.0))
        # tronca l'ultimo intervallo al budget residuo: senza, uno span da 2s puo'
        # sforare il target di decine di punti percentuali in un colpo solo
        remaining = target - int(mask.sum())
        span_samples = min(span_samples, n_samples, max(remaining, 1))
        start = int(rng.integers(0, n_samples - span_samples + 1))
        mask[start : start + span_samples] = True
    return mask


def spatial_mask(
    groups: tuple[ChannelGroup, ...],
    rng: np.random.Generator,
    *,
    budget_fraction: float = 0.4,
) -> np.ndarray:
    """Maschera booleana sull'asse canale (v10 §4.4, §6.4): canali singoli, gruppi
    contigui, porzioni geometriche di griglia. ~40% del budget di masking (default).
    """
    c_total = sum(g.n_channels for g in groups)
    mask = np.zeros(c_total, dtype=bool)
    target = max(1, int(c_total * budget_fraction))

    offsets = np.cumsum([0] + [g.n_channels for g in groups])
    mode = rng.integers(3)
    if mode == 0:
        # canali singoli sparsi
        idx = rng.choice(c_total, size=min(target, c_total), replace=False)
        mask[idx] = True
    elif mode == 1:
        # un gruppo contiguo intero (o piu', finche' non si copre il budget)
        remaining = target
        group_order = rng.permutation(len(groups))
        for gi in group_order:
            if remaining <= 0:
                break
            start, end = offsets[gi], offsets[gi + 1]
            mask[start:end] = True
            remaining -= end - start
    else:
        # porzione contigua di canali (geometrica su una griglia)
        start = int(rng.integers(0, max(1, c_total - target + 1)))
        mask[start : start + min(target, c_total)] = True
    return mask


def rvq_slab_mask(
    n_samples: int,
    fs_hz: float,
    rng: np.random.Generator,
    *,
    slab_ms: float = 200.0,
    n_slabs: int = 1,
) -> np.ndarray:
    """Quota di masking a slab per l'ancora RVQ (v10 §6.3, §6.4): intervalli >= 200 ms
    mascherati su TUTTI i canali insieme, allineati alla griglia del tokenizer (200 ms).
    Default spento nel benchmark (spec §4): va attivato esplicitamente.
    """
    slab_samples = max(1, round(slab_ms * fs_hz / 1000.0))
    mask = np.zeros(n_samples, dtype=bool)
    for _ in range(n_slabs):
        if slab_samples >= n_samples:
            mask[:] = True
            break
        # allineato alla griglia da 200 ms
        n_slots = n_samples // slab_samples
        if n_slots == 0:
            break
        slot = int(rng.integers(0, n_slots))
        start = slot * slab_samples
        mask[start : start + slab_samples] = True
    return mask
