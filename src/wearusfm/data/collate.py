"""Collate: raggruppamento per frequenza e i tre layout (asse L, spec §4-§6).

- group_by_frequency: il front-end gira una volta per gruppo di fs; il tempo non si
  padda MAI (spec §4, stadio 5).
- L1 padding + maschera (v10 §2.7): pad del canale al massimo del batch.
- L2 packing (v10 §4.6, spec §6): lista piatta di token con `cu_seqlens`. Qui si
  costruisce solo la contabilita' (indici, confini): l'attenzione vera (encoder locale
  via `gather`, cross-attention del Perceiver via `flash_attn_varlen_func`) vive nel
  proxy GPU (bench/proxy_model.py) e non e' testabile su CPU.
- L3 bucketing per micro-batch (spec §6): raggruppa per C esatto, cosi' niente padding
  per costruzione dentro il bucket. La miscela di topologie si conserva per passo di
  ottimizzazione (fra bucket), non per micro-batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearusfm.data.synthetic import SyntheticSample


def group_by_frequency(samples: list[SyntheticSample]) -> dict[float, list[SyntheticSample]]:
    groups: dict[float, list[SyntheticSample]] = {}
    for s in samples:
        groups.setdefault(s.front_end_fs_hz, []).append(s)
    return groups


@dataclass
class PaddedBatch:
    data: np.ndarray  # (B, C_max, T) int16, canali oltre c_presented sono 0
    channel_mask: np.ndarray  # (B, C_max) bool, True = canale valido
    c_presented: np.ndarray  # (B,) int

    @property
    def padding_fraction(self) -> float:
        total = self.channel_mask.size
        valid = self.channel_mask.sum()
        return 1.0 - valid / total if total else 0.0


def collate_padding(samples: list[SyntheticSample]) -> PaddedBatch:
    """L1 - padding + maschera. Tutti i campioni devono avere lo stesso T (stessa fs
    e stesso context_s): raggruppare per frequenza PRIMA di chiamare questa funzione."""
    b = len(samples)
    t = samples[0].n_samples
    c_max = max(s.c_presented for s in samples)

    data = np.zeros((b, c_max, t), dtype=np.int16)
    channel_mask = np.zeros((b, c_max), dtype=bool)
    c_presented = np.zeros(b, dtype=np.int64)

    for i, s in enumerate(samples):
        c = s.c_presented
        data[i, :c, :] = s.data.T  # sample.data e' (T, C) -> (C, T)
        channel_mask[i, :c] = True
        c_presented[i] = c

    return PaddedBatch(data=data, channel_mask=channel_mask, c_presented=c_presented)


@dataclass
class PackedBatch:
    tokens: np.ndarray  # (sum(C_i), T) int16, canali di tutti i campioni concatenati
    sample_ids: np.ndarray  # (sum(C_i),) int, indice del campione di origine per riga
    cu_seqlens: np.ndarray  # (B + 1,) int, confini cumulativi (convenzione flash_attn varlen)

    @property
    def n_tokens(self) -> int:
        return self.tokens.shape[0]


def collate_packing(samples: list[SyntheticSample]) -> PackedBatch:
    """L2 - packing. Stessa precondizione di collate_padding: T uniforme nel batch."""
    t = samples[0].n_samples
    lengths = np.array([s.c_presented for s in samples], dtype=np.int64)
    cu_seqlens = np.concatenate([[0], np.cumsum(lengths)])

    tokens = np.concatenate([s.data.T for s in samples], axis=0)  # (sum(C_i), T)
    sample_ids = np.concatenate(
        [np.full(l, i, dtype=np.int64) for i, l in enumerate(lengths)]
    )
    return PackedBatch(tokens=tokens, sample_ids=sample_ids, cu_seqlens=cu_seqlens)


def collate_bucketing(samples: list[SyntheticSample]) -> dict[int, PaddedBatch]:
    """L3 - bucketing per micro-batch, un bucket per ogni C esatto (niente padding per
    costruzione). Il chiamante accumula il gradiente sui bucket di un passo."""
    by_c: dict[int, list[SyntheticSample]] = {}
    for s in samples:
        by_c.setdefault(s.c_presented, []).append(s)
    return {c: collate_padding(group) for c, group in by_c.items()}


def padding_fraction_analytic(c_values: list[int]) -> float:
    """Stima analitica della frazione di padding (spec §8): 1 - media(C)/max(C), senza
    bisogno di GPU. Utile per il preflight prima di misurare sul proxy."""
    if not c_values:
        return 0.0
    arr = np.asarray(c_values, dtype=np.float64)
    return float(1.0 - arr.mean() / arr.max())
