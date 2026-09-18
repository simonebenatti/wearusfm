"""Tabella dei montaggi sintetici per il benchmark del passo 0.

Fonte: docs/dataloader_bench_spec.md §2-3, che a sua volta cita
docs/fm_emg_reference_v10.md §2.1, §3.4. Solo i dataset che entrano nel
pretraining (esclusi EPN-612, UCI-EMG, NinaPro DB1, NinaPro DB9, Gait120).

Ogni riga della spec §3 e' un Montage: una finestra temporale ne copre
SEMPRE tutti i canali insieme (spec §2 - "il campione e' il montaggio
intero, non il gruppo di canali").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuotaClass(str, Enum):
    """Classe di quota del montaggio (v10 §2.8, spec §2). Del MONTAGGIO, non del gruppo."""

    A_RADI = "A"  # radi / anatomici - il caso di deployment
    B_ANELLI = "B"  # anelli e fasce
    C_GRIGLIE = "C"  # griglie HD


@dataclass(frozen=True)
class ChannelGroup:
    """Un gruppo di canali dentro un montaggio (v10 §3.4, §4.5): topologia dichiarata
    per gruppo, non per dataset. Non spezza mai il campione (spec §2)."""

    topology: str  # "ring" | "grid" | "targeted"
    n_channels: int


@dataclass(frozen=True)
class Montage:
    name: str
    quota_class: QuotaClass
    groups: tuple[ChannelGroup, ...]
    fs_native_hz: float
    hours: float | None  # None = non nota; usata solo per i pesi dentro la classe C
    data_verified: bool  # False se marcato "da verificare" nella spec

    @property
    def n_channels(self) -> int:
        return sum(g.n_channels for g in self.groups)


# spec §3, tabella "Distribuzione sintetica". fs_native_hz per i dataset "da verificare"
# (DB8, ~1111 Hz) resta quella indicata, marcata data_verified=False.
MONTAGES: tuple[Montage, ...] = (
    Montage(
        "emg2pose", QuotaClass.B_ANELLI,
        (ChannelGroup("ring", 16),), 2000.0, hours=370.0, data_verified=True,
    ),
    Montage(
        "emg2qwerty", QuotaClass.B_ANELLI,
        (ChannelGroup("ring", 16), ChannelGroup("ring", 16)),  # speculari
        2000.0, hours=346.0, data_verified=True,
    ),
    Montage(
        "ninapro_db5", QuotaClass.B_ANELLI,
        (ChannelGroup("ring", 8), ChannelGroup("ring", 8)),  # 2x Myo, stesso avambraccio
        200.0, hours=None, data_verified=True,
    ),
    Montage(
        "grabmyo", QuotaClass.B_ANELLI,
        (ChannelGroup("ring", 16), ChannelGroup("ring", 12)),  # avambraccio + polso
        2048.0, hours=None, data_verified=False,  # disposizione interna: da verificare
    ),
    Montage(
        "putemg", QuotaClass.B_ANELLI,
        (ChannelGroup("ring", 8), ChannelGroup("ring", 8), ChannelGroup("ring", 8)),  # 3 fasce
        5120.0, hours=None, data_verified=True,  # raccolto da fonte ufficiale (fatti_da_verificare.md #12)
    ),
    Montage(
        "ninapro_standard", QuotaClass.A_RADI,
        (ChannelGroup("ring", 8), ChannelGroup("targeted", 4)),
        2000.0, hours=90.0, data_verified=True,
    ),
    Montage(
        "ninapro_db6", QuotaClass.A_RADI,
        (ChannelGroup("ring", 8), ChannelGroup("targeted", 6)),
        2000.0, hours=None, data_verified=False,  # split ring/mirati: da verificare
    ),
    Montage(
        "ninapro_db8", QuotaClass.A_RADI,
        (ChannelGroup("ring", 8), ChannelGroup("targeted", 8)),
        1111.0, hours=None, data_verified=False,  # fs e split: da verificare
    ),
    Montage(
        "camargo2021", QuotaClass.A_RADI,
        (ChannelGroup("targeted", 11),), 1000.0, hours=None, data_verified=True,
    ),
    Montage(
        "capgmyo", QuotaClass.C_GRIGLIE,
        tuple(ChannelGroup("grid", 16) for _ in range(8)),  # 8 strisce 2x8
        1000.0, hours=None, data_verified=True,
    ),
    Montage(
        "csl_hdemg", QuotaClass.C_GRIGLIE,
        (ChannelGroup("grid", 168),), 2048.0, hours=None, data_verified=True,
    ),
    Montage(
        "hyser", QuotaClass.C_GRIGLIE,
        tuple(ChannelGroup("grid", 64) for _ in range(4)),  # 4 griglie
        2048.0, hours=None, data_verified=True,  # raccolto da PhysioNet (fatti_da_verificare.md #11)
    ),
)

MONTAGES_BY_NAME: dict[str, Montage] = {m.name: m for m in MONTAGES}
MONTAGES_BY_CLASS: dict[QuotaClass, tuple[Montage, ...]] = {
    c: tuple(m for m in MONTAGES if m.quota_class == c) for c in QuotaClass
}


def front_end_fs_hz(montage: Montage, *, all_native: bool = False) -> float:
    """Frequenza al front-end per un montaggio (v10 §4.1, spec §3).

    Default: HD nativi, Myo (200 Hz) resta nativo (banda limitata, mai upsampling),
    tutto il resto a 1 kHz. Variante di sensibilita' `all_native=True`: tutto nativo
    (caso peggiore in byte, spec §3).
    """
    if all_native:
        return montage.fs_native_hz
    if montage.quota_class == QuotaClass.C_GRIGLIE:
        return montage.fs_native_hz
    if montage.fs_native_hz <= 200.0:
        return montage.fs_native_hz
    return 1000.0


# Pesi dentro la classe (spec §3): proporzionali alle ore dove note, uniformi altrove.
# Per la classe C, due scenari che delimitano la risposta.
def class_weights(quota_class: QuotaClass, *, c_pessimistic: bool = True) -> dict[str, float]:
    montages = MONTAGES_BY_CLASS[quota_class]
    if quota_class == QuotaClass.C_GRIGLIE:
        if c_pessimistic:
            return {m.name: (1.0 if m.name == "hyser" else 0.0) for m in montages}
        return {m.name: 1.0 / len(montages) for m in montages}
    known = {m.name: m.hours for m in montages if m.hours is not None}
    if not known:
        return {m.name: 1.0 / len(montages) for m in montages}
    total_known_hours = sum(known.values())
    n_unknown = len(montages) - len(known)
    if n_unknown == 0:
        return {name: h / total_known_hours for name, h in known.items()}
    # ore note pesate proporzionalmente, il resto si spartisce uniforme lo stesso
    # budget totale (1.0): meta' alle note (proporzionale), meta' spartita fra le ignote.
    weights: dict[str, float] = {}
    for m in montages:
        if m.hours is not None:
            weights[m.name] = 0.5 * (m.hours / total_known_hours)
        else:
            weights[m.name] = 0.5 / n_unknown
    return weights


def full_montage_weights(
    quota_class_weights: dict[QuotaClass, float],
    *,
    c_pessimistic: bool = True,
) -> dict[str, float]:
    """Combina le quote per classe (D2/D9a, es. A/B/C = 40/35/25) coi pesi dentro
    ciascuna classe (`class_weights`), in un'unica distribuzione sui nomi di montaggio."""
    total = sum(quota_class_weights.values())
    out: dict[str, float] = {}
    for quota_class, w_class in quota_class_weights.items():
        inner = class_weights(quota_class, c_pessimistic=c_pessimistic)
        for name, w_inner in inner.items():
            out[name] = (w_class / total) * w_inner
    return out
