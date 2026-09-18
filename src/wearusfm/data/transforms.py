"""Trasformazioni che decidono il C presentato al modello.

Fonte: docs/dataloader_bench_spec.md §3 ("Trasformazioni che decidono il C presentato") e
§4 (stadi 2-3 della catena). I parametri di default (probabilita' di dropout, range del
sottocampionamento HD) sono placeholder di lavoro per il benchmark, NON valori decisi in
docs/fm_emg_reference_v10.md: le probabilita' vere restano parametro del manifest (D9) o
del masking (D10), entrambe ancora aperte (v10 §12).
"""

from __future__ import annotations

import numpy as np

from wearusfm.data.manifest import ChannelGroup, Montage, QuotaClass


def apply_montage_dropout_with_indices(
    groups: tuple[ChannelGroup, ...],
    rng: np.random.Generator,
    group_drop_p: float = 0.2,
    channel_drop_p: float = 0.1,
) -> tuple[tuple[ChannelGroup, ...], list[int]]:
    """Come `apply_montage_dropout`, ma ritorna anche gli indici assoluti dei canali
    superstiti (posizione dentro `groups`, concatenati nell'ordine dei gruppi) - serve
    alla pipeline per selezionare le stesse colonne dall'array dei dati (pipeline.py)."""
    kept: list[ChannelGroup] = []
    keep_idx: list[int] = []
    offset = 0
    for g in groups:
        if rng.random() < group_drop_p:
            offset += g.n_channels
            continue
        n_keep = int(rng.binomial(g.n_channels, 1.0 - channel_drop_p))
        if n_keep > 0:
            kept.append(ChannelGroup(g.topology, n_keep))
            keep_idx.extend(range(offset, offset + n_keep))
        offset += g.n_channels
    if not kept:
        # non lasciare mai un campione vuoto: tiene almeno un canale del gruppo piu' piccolo
        smallest_i = min(range(len(groups)), key=lambda i: groups[i].n_channels)
        offset0 = sum(g.n_channels for g in groups[:smallest_i])
        kept = [ChannelGroup(groups[smallest_i].topology, 1)]
        keep_idx = [offset0]
    return tuple(kept), keep_idx


def apply_montage_dropout(
    groups: tuple[ChannelGroup, ...],
    rng: np.random.Generator,
    group_drop_p: float = 0.2,
    channel_drop_p: float = 0.1,
) -> tuple[ChannelGroup, ...]:
    """Montage dropout (v10 §8): per gruppo intero e per canale dentro i gruppi superstiti.

    Non spezza mai un gruppo in pezzi non contigui: riduce solo il conteggio di canali
    del gruppo (spec §2 - il campione resta il montaggio, la topologia resta dichiarata
    per gruppo).
    """
    kept, _ = apply_montage_dropout_with_indices(groups, rng, group_drop_p, channel_drop_p)
    return kept


def hd_virtual_bipolar_montage(
    groups: tuple[ChannelGroup, ...],
    rng: np.random.Generator,
    c_min: int = 8,
    c_max: int = 32,
) -> tuple[ChannelGroup, ...]:
    """Sottocampionamento di una griglia HD a montaggio bipolare virtuale (v10 §2.7, spec §3).

    C <= 32 (v10 §5). La distribuzione esatta di C dentro [8, 32] non e' specificata in v10:
    qui e' uniforme, placeholder di benchmark.
    """
    total = sum(g.n_channels for g in groups)
    c = int(rng.integers(c_min, min(c_max, total) + 1))
    return (ChannelGroup("virtual_bipolar", c),)


def presented_class_for_virtual_montage(c_presented: int, *, sparse_threshold: int = 12) -> QuotaClass:
    """Classe presentata per un montaggio virtuale ricavato da una griglia HD.

    Spec §2: "un montaggio virtuale ricavato da una griglia HD nasce in C ma si presenta
    come A o B: servono entrambe per la scelta di D9a" - la regola esatta e' D9(a), non
    ancora decisa (docs/decisioni.md). Qui una soglia placeholder: C piccolo -> presentato
    come sparso (A), altrimenti come anello/fascia (B). Va rivista quando D9(a) e' congelata.
    """
    return QuotaClass.A_RADI if c_presented <= sparse_threshold else QuotaClass.B_ANELLI


def maybe_hd_subsample(
    montage: Montage,
    groups: tuple[ChannelGroup, ...],
    rng: np.random.Generator,
    p_piena: float,
) -> tuple[tuple[ChannelGroup, ...], QuotaClass]:
    """Applica il sottocampionamento HD -> montaggio virtuale con probabilita' (1 - p_piena).

    Restituisce (gruppi presentati, classe presentata). Per montaggi non di classe C
    restituisce i gruppi invariati e la classe di origine.
    """
    if montage.quota_class != QuotaClass.C_GRIGLIE:
        return groups, montage.quota_class
    if rng.random() < p_piena:
        return groups, QuotaClass.C_GRIGLIE
    virtual = hd_virtual_bipolar_montage(groups, rng)
    return virtual, presented_class_for_virtual_montage(virtual[0].n_channels)


def add_realistic_noise(
    data: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    powerline_hz: float = 50.0,
) -> np.ndarray:
    """Rumore realistico (v10 §8): drift sotto i 20 Hz, rete, contaminazione ECG.

    `data` ha forma (n_samples, n_channels), float. Il contenuto resta comunque casuale
    (spec §5): qui si aggiunge solo la struttura di rumore, non fisiologia reale.
    """
    n_samples, n_channels = data.shape
    t = np.arange(n_samples) / fs_hz

    drift_freq = rng.uniform(0.5, 3.0, size=n_channels)
    drift_amp = rng.uniform(0.05, 0.2, size=n_channels)
    drift = drift_amp[None, :] * np.sin(2 * np.pi * drift_freq[None, :] * t[:, None])

    powerline_amp = rng.uniform(0.02, 0.1)
    powerline = powerline_amp * np.sin(2 * np.pi * powerline_hz * t)[:, None]

    out = data + drift + powerline

    # contaminazione ECG: burst periodico su un piccolo sottoinsieme di canali (prossimali)
    if rng.random() < 0.3 and n_channels > 0:
        n_ecg_channels = max(1, n_channels // 8)
        ecg_channels = rng.choice(n_channels, size=n_ecg_channels, replace=False)
        heart_rate_hz = rng.uniform(1.0, 1.7)
        burst = 0.3 * np.sin(2 * np.pi * heart_rate_hz * t) * (np.sin(2 * np.pi * heart_rate_hz * t) > 0.95)
        out[:, ecg_channels] += burst[:, None]

    return out


def time_warp(data: np.ndarray, rng: np.random.Generator, max_warp: float = 0.05) -> np.ndarray:
    """Time warping (v10 §8): interpolazione su C x campioni - probabilmente lo stadio
    piu' caro della catena (spec §4). `data` ha forma (n_samples, n_channels).
    """
    n_samples, n_channels = data.shape
    if n_samples < 4:
        return data
    # random walk regolarizzato per la mappa di warp, poi interpolazione per canale
    steps = rng.normal(scale=max_warp, size=n_samples)
    warp = np.cumsum(steps)
    warp -= np.linspace(warp[0], warp[-1], n_samples)  # ancora gli estremi
    src_t = np.arange(n_samples) + warp * n_samples / max(abs(warp).max(), 1e-6) * max_warp
    src_t = np.clip(src_t, 0, n_samples - 1)
    out = np.empty_like(data)
    idx = np.arange(n_samples)
    for c in range(n_channels):
        out[:, c] = np.interp(src_t, idx, data[:, c])
    return out
