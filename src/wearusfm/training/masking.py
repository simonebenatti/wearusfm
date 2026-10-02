"""Generatore di maschere per lo studente JEPA (passo 6; v10 §6.4; schema della BOZZA D10, non firmata: `docs/proposta_d10.md` §3).

Non e' `wearusfm.data.masking`: quello genera le maschere sintetiche del benchmark del dataloader del passo 0 (test in `tests/cpu/test_masking.py`);
questo e' il generatore del training (test in `tests/cpu/test_training_masking.py`).

Per un campione (C canali, P patch valide) restituisce `visible` (C, P) e `kind` (C, P): il tipo di maschera di ogni token nascosto, perche' il
masking di canale e' due compiti diversi da registrare separatamente (v10 §6.4: «senza logging separato la loss aggregata e' dominata dal caso
facile»), e gli altri tipi pure.

Componenti e quote del budget (D10, proposta; il budget e' la frazione `ratio` dei token validi, senza default: la regola per sceglierla si congela
prima del sanity JEPA):
- temporale corta, 1-2 patch: 15%; mai da sola (se c'e', nello stesso campione c'e' almeno una maschera piu' lunga);
- temporale media, 4-12 patch: 25%, meta' come **slab** (8 patch allineate) sui campioni con l'ancora RVQ accesa;
- temporale lunga, 20-80 patch e al piu' meta' finestra: 20%, meta' come slab (multipli di 8 patch allineati);
- spaziale: 40%: canali singoli, archi contigui sugli anelli, rettangoli sulle griglie, gruppi dello stesso compartimento sui montaggi sparsi; al piu'
  meta' dei canali validi.

*Interpretazioni di AG, da confermare con D10:* una maschera temporale che non e' uno slab nasconde un intervallo su **un solo canale** (punto aperto
per la firma: su una griglia densa e' un compito facile, i vicini sono visibili; l'alternativa e' l'intervallo su un gruppo spaziale, un arco o un
rettangolo, come i «tubi» di V-JEPA); uno slab lo
nasconde su **tutti** i canali (ancora RVQ, v10 §6.3); una maschera spaziale nasconde i canali scelti **per tutta la finestra**. Le quote si
riempiono a token nuovi nascosti, con un numero massimo di tentativi; le sovrapposizioni contano una volta sola. I canali scartati dal QC non si
scelgono e non contano nel budget. Gli slab cominciano su multipli di 8 patch: la finestra comincia su un multiplo di 200 ms dall'inizio della prova
(dataloader, proposta D10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearusfm.model.channel_codes import TOPOLOGIES, ChannelLayout

VISIBLE, SHORT, MEDIUM, LONG, SLAB, SPATIAL = 0, 1, 2, 3, 4, 5
KIND_NAMES = {VISIBLE: "visibile", SHORT: "corta", MEDIUM: "media", LONG: "lunga", SLAB: "slab", SPATIAL: "spaziale"}
SLAB_PATCHES = 8  # 200 ms / 25 ms: la griglia del tokenizer
MAX_ATTEMPTS = 200  # piu' 4 volte il numero atteso di maschere del campione (vedi generate_mask)


@dataclass(frozen=True)
class MaskSpec:
    ratio: float  # frazione dei token validi da nascondere (nessun default)
    share_short: float
    share_medium: float
    share_long: float
    share_spatial: float
    short: tuple[int, int] = (1, 2)
    medium: tuple[int, int] = (4, 12)
    long: tuple[int, int] = (20, 80)
    slab_fraction: float = 0.5  # parte del medio e del lungo fatta di slab, dove l'ancora RVQ e' accesa
    max_spatial_fraction: float = 0.5  # dei canali validi

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise ValueError(f"ratio in (0, 1): {self.ratio}")
        shares = (self.share_short, self.share_medium, self.share_long, self.share_spatial)
        if abs(sum(shares) - 1.0) > 1e-9 or min(shares) < 0:
            raise ValueError(f"le quote devono sommare a 1: {shares}")

    @classmethod
    def d10_proposal(cls, ratio: float) -> "MaskSpec":
        """Le quote della tabella di `docs/proposta_d10.md` §3 (BOZZA, non firmata)."""
        return cls(ratio, 0.15, 0.25, 0.20, 0.40)


def _spatial_groups(layout: ChannelLayout, compartment: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Un gruppo di canali da nascondere, scelto secondo la topologia di un canale valido a caso."""
    c0 = int(rng.choice(np.flatnonzero(valid)))
    g = layout.group[c0]
    members = np.flatnonzero((layout.group == g) & valid)
    topo = TOPOLOGIES[layout.topology[c0]]
    if topo == "ring":
        order = members[np.argsort(layout.ring_angle[members])]
        n = int(rng.integers(1, max(1, len(order) // 4) + 1))  # arco di 1..n/4 canali
        start = int(np.flatnonzero(order == c0)[0])
        return [int(order[(start + i) % len(order)]) for i in range(n)]
    if topo == "grid_2d":
        r0, c0c = layout.grid_rc[c0]
        h, w = int(rng.integers(1, 5)), int(rng.integers(1, 5))  # rettangolo fino a 4 x 4
        rc = layout.grid_rc[members]
        inside = (rc[:, 0] >= r0) & (rc[:, 0] < r0 + h) & (rc[:, 1] >= c0c) & (rc[:, 1] < c0c + w)
        return [int(m) for m in members[inside]]
    same = members[compartment[members] == compartment[c0]] if compartment[c0] != 0 else np.array([c0])
    return [int(c0)] if rng.random() < 0.5 else [int(m) for m in same]  # canale singolo o gruppo anatomico


def _slab_lengths(lo: int, hi: int, cap: int) -> list[int]:
    lo8 = -(-lo // SLAB_PATCHES) * SLAB_PATCHES  # minimo della scala arrotondato al multiplo di 8 successivo (media: 8; lunga: 24)
    return list(range(lo8, min(hi, cap) + 1, SLAB_PATCHES))


def generate_mask(layout: ChannelLayout, compartment: np.ndarray, n_patches: int, p_max: int, rvq_on: bool, spec: MaskSpec,
                  rng: np.random.Generator, tolerance: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """visible (C, p_max) e kind (C, p_max) per un campione con `n_patches` patch valide (il resto e' padding, mai nascosto).
    `compartment` (C,): indice del compartimento piu' probabile (0 = ignoto), per i gruppi anatomici.

    Le quote sono **attese** sui token nascosti, non vincoli per campione: uno slab copre tutti i canali e da solo puo' valere piu' della sua quota
    (uno slab lungo di 24 patch e' il 15% di una finestra da 4 s). A ogni passo si sceglie il tipo di maschera con probabilita' proporzionale a
    quota / dimensione attesa sul campione, cosi' in media ogni tipo occupa la sua quota; una maschera che porterebbe il totale oltre
    budget x (1 + tolerance) si scarta; ci si ferma al budget o dopo un numero di tentativi
    proporzionale al numero atteso di maschere."""
    c = layout.n_channels
    valid = layout.qc_valid.copy()
    valid_idx = np.flatnonzero(valid)
    n_ch = len(valid_idx)
    kind = np.zeros((c, p_max), dtype=np.int8)
    if n_ch == 0 or n_patches == 0:
        raise ValueError("nessun token valido")
    budget = spec.ratio * n_ch * n_patches
    slab_frac = spec.slab_fraction if rvq_on else 0.0
    max_sp = int(np.floor(spec.max_spatial_fraction * n_ch))
    long_cap = n_patches // 2
    probe = np.random.default_rng(rng.integers(2**32))  # stima della dimensione attesa dei gruppi spaziali, senza toccare rng
    sp_size = np.mean([len(_spatial_groups(layout, compartment, valid, probe)) for _ in range(20)]) * n_patches

    # (tipo, dimensione attesa in token, quota)
    lens_med_slab = _slab_lengths(*spec.medium, n_patches)
    lens_long_slab = _slab_lengths(spec.long[0], spec.long[1], long_cap)
    comps = {
        (SHORT, False): (np.mean(spec.short), spec.share_short),
        (MEDIUM, False): (np.mean([min(spec.medium[1], n_patches), spec.medium[0]]), spec.share_medium * (1 - slab_frac)),
        (MEDIUM, True): (np.mean(lens_med_slab) * n_ch if lens_med_slab else 0.0, spec.share_medium * slab_frac),
        (LONG, False): (np.mean([spec.long[0], min(spec.long[1], long_cap)]), spec.share_long * (1 - slab_frac)),
        (LONG, True): (np.mean(lens_long_slab) * n_ch if lens_long_slab else 0.0, spec.share_long * slab_frac),
        (SPATIAL, False): (sp_size, spec.share_spatial),
    }
    feasible = {k: v for k, v in comps.items() if v[0] > 0 and v[1] > 0}
    if spec.long[0] > long_cap:
        feasible.pop((LONG, False), None)
    if spec.medium[0] > n_patches:
        feasible.pop((MEDIUM, False), None)

    def draw(kind_k, slab):
        """(righe, t0, t1) di una maschera del tipo dato, o None se non si puo'."""
        if kind_k == SPATIAL:
            group = [g for g in _spatial_groups(layout, compartment, valid, rng) if g not in hidden_ch]
            return (np.asarray(group), 0, n_patches) if group and len(hidden_ch) + len(group) <= max_sp else None
        lo, hi = {SHORT: spec.short, MEDIUM: spec.medium, LONG: spec.long}[kind_k]
        if slab:
            ln = int(rng.choice(lens_med_slab if kind_k == MEDIUM else lens_long_slab))
            t0 = int(rng.choice(np.arange(0, n_patches - ln + 1, SLAB_PATCHES)))
            return valid_idx, t0, t0 + ln
        top = min(hi, long_cap if kind_k == LONG else n_patches)
        ln = int(rng.integers(lo, top + 1))
        t0 = int(rng.integers(0, n_patches - ln + 1))
        return np.array([int(rng.choice(valid_idx))]), t0, t0 + ln

    hidden_ch: set[int] = set()
    total, tries = 0, 0
    keys = list(feasible)
    weights = np.array([feasible[k][1] / feasible[k][0] for k in keys])
    # numero atteso di maschere: su una griglia da 128 canali le maschere temporali di un canale sono piccole e ne servono ~1000
    max_attempts = MAX_ATTEMPTS + 4 * int(sum(budget * feasible[k][1] / feasible[k][0] for k in keys))
    while total < budget and tries < max_attempts and keys:
        tries += 1
        allowed = np.array([not (k[0] == SHORT and total == 0) and not (k[0] == SPATIAL and len(hidden_ch) >= max_sp) for k in keys])
        if not allowed.any():
            break
        w = weights * allowed
        k = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        got = draw(*k)
        if got is None:
            continue
        rows, t0, t1 = got
        block = kind[np.ix_(rows, np.arange(t0, t1))]
        new = int((block == VISIBLE).sum())
        if new == 0 or total + new > budget * (1 + tolerance):
            continue
        block[block == VISIBLE] = SLAB if k[1] else k[0]
        kind[np.ix_(rows, np.arange(t0, t1))] = block
        total += new
        if k[0] == SPATIAL:
            hidden_ch.update(int(r) for r in rows)
    visible = kind == VISIBLE
    visible[~valid] = True  # i canali scartati restano fuori comunque (qc_valid), qui non si contano
    return visible, kind


def pack_masks(masks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Le maschere dei campioni impacchettate sull'asse dei canali, nello stesso ordine dei token."""
    return np.concatenate([m[0] for m in masks], axis=0), np.concatenate([m[1] for m in masks], axis=0)


def kind_fractions(kind: np.ndarray, valid_tokens: int) -> dict[str, float]:
    return {KIND_NAMES[k]: float((kind == k).sum()) / max(1, valid_tokens) for k in KIND_NAMES if k != VISIBLE}
