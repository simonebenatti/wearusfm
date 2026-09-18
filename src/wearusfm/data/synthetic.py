"""Generatore di campioni sintetici per il benchmark del passo 0.

Fonte: docs/dataloader_bench_spec.md §2-§4. Contenuto casuale (nessun byte di EMG reale,
v10 §9), ma forma (canali, frequenza nativa, classe, gruppi) fedele alla tabella reale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearusfm.data.manifest import ChannelGroup, Montage, QuotaClass, front_end_fs_hz
from wearusfm.data.transforms import apply_montage_dropout, maybe_hd_subsample


@dataclass
class SyntheticSample:
    montage_name: str
    quota_class_origin: QuotaClass
    quota_class_presented: QuotaClass
    groups_presented: tuple[ChannelGroup, ...]
    fs_native_hz: float
    front_end_fs_hz: float
    context_s: float
    data: np.ndarray  # int16, forma (n_samples, C) - tempo x canali (spec §5)

    @property
    def c_presented(self) -> int:
        return sum(g.n_channels for g in self.groups_presented)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]


def generate_sample(
    montage: Montage,
    rng: np.random.Generator,
    *,
    context_s: float = 4.0,
    all_native_fs: bool = False,
    p_piena: float = 0.5,
    montage_dropout_group_p: float = 0.2,
    montage_dropout_channel_p: float = 0.1,
) -> SyntheticSample:
    """Genera un campione sintetico: una finestra temporale del montaggio intero.

    Ordine delle trasformazioni (spec §3-§4): sottocampionamento HD (se classe C) prima
    del montage dropout, cosi' il dropout si applica ai gruppi gia' eventualmente
    diventati un montaggio virtuale.
    """
    fe_fs = front_end_fs_hz(montage, all_native=all_native_fs)

    groups, quota_class_presented = maybe_hd_subsample(montage, montage.groups, rng, p_piena)
    groups = apply_montage_dropout(
        groups, rng, group_drop_p=montage_dropout_group_p, channel_drop_p=montage_dropout_channel_p
    )

    n_samples = max(1, round(fe_fs * context_s))
    c = sum(g.n_channels for g in groups)
    # contenuto casuale (spec §5): int16 pieno range, nessuna fisiologia reale
    data = rng.integers(-2000, 2000, size=(n_samples, c), dtype=np.int16)

    return SyntheticSample(
        montage_name=montage.name,
        quota_class_origin=montage.quota_class,
        quota_class_presented=quota_class_presented,
        groups_presented=groups,
        fs_native_hz=montage.fs_native_hz,
        front_end_fs_hz=fe_fs,
        context_s=context_s,
        data=data,
    )


def sample_montage(
    rng: np.random.Generator,
    montages: tuple[Montage, ...],
    weights: dict[str, float],
) -> Montage:
    """Estrae un montaggio secondo i pesi dati (v10 §2.8, §10.3: quote di campionamento)."""
    names = [m.name for m in montages]
    p = np.array([weights[n] for n in names], dtype=np.float64)
    p = p / p.sum()
    idx = rng.choice(len(names), p=p)
    return montages[idx]
