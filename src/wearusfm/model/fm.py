"""Il modello intero (passo 6; v10 §5): token di patch -> identita' di canale -> encoder locale -> Perceiver -> backbone -> decoder a query, con le
teste delle ancore. Le scelte di ogni modulo sono nei rispettivi file e in `docs/decisioni.md` («Passo 6»).

**Cosa vede lo studente.** `encode(..., visible)` nasconde i token con `visible` False in tre punti, cosi' che nulla del contenuto nascosto arrivi
ai token visibili:
1. i **campioni grezzi** delle patch nascoste sono azzerati prima del front-end: il front-end usa 100 ms di contesto per lato (gate D8), e senza
   azzerare una patch visibile accanto a una nascosta vedrebbe fino a 100 ms del segnale nascosto. *Scelta di AG, da confermare:* ai bordi di una
   regione nascosta lo studente vede zeri, come ai bordi del segnale;
2. nell'**encoder locale** un token nascosto non e' chiave per gli altri canali;
3. nel **Perceiver** i token nascosti sono fuori dalle chiavi, e nel **backbone** gli istanti senza nessun token visibile sono fuori dalle chiavi
   temporali.
Il teacher (EMA) chiama `encode` senza maschera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from wearusfm.model.anchors import AnchorHeads, RVQHead
from wearusfm.model.backbone import TemporalBackbone
from wearusfm.model.channel_identity import ChannelIdentity
from wearusfm.model.local_encoder import LocalEncoder
from wearusfm.model.patch_tokens import PatchTokenizer
from wearusfm.model.perceiver import PerceiverPooling, offsets_from_counts
from wearusfm.model.query_decoder import QueryDecoder
from wearusfm.model.spectral_branches import BRANCH_KINDS, SpectralBranches, branch_heads
from wearusfm.model.spectral_step1 import SpectralKeepHead


@dataclass(frozen=True)
class FMConfig:
    dim: int
    n_heads: int
    k_latents: int
    local_layers: int
    backbone_layers: int
    decoder_layers: int
    muscle_dropout: float  # D15, nessun default (piano: proposta 0,4)
    patch_ms: float = 25.0  # D10, proposta
    # finestra del kernel piu' lunga del passo (R5 del pilot degli obiettivi, Simone 23/10/2026: «150ms con overlap di 125ms»): None = il banco di
    # sempre; con una finestra la griglia resta `patch_ms`, le basi scalano (`frontend.window_kwargs`) e la perdita JEPA conta solo i token
    # interamente nascosti (`training.jepa.core_hidden`)
    window_ms: float | None = None
    n_bands: int = 5
    rvq_codes: int | None = None  # None = senza testa RVQ
    grad_checkpoint: bool = False  # activation checkpointing di encoder locale, backbone e decoder (v10 §5.5)
    multiscale_anchor: bool = False  # teste dell'ancora multi-scala (Simone, 04/10/2026, dalla finestra 1)
    keep_readout: bool = False  # Step 1 experimental shared-readout supervision
    # rami spettrali continui in parallelo al raw (step 2 e 3; Simone, 13/10/2026), `model/spectral_branches.py`: "psd" e/o "wavelet", vuoto =
    # nessuno (il modello di sempre, stessi parametri). L'embedding P1/P2 diventa [readout raw, rami]. Sul readout: la keep dello Step 1 sulla testa
    # raw resta (JEPAConfig.keep_weight); il termine dei rami, (1 - branch_alpha) testa fusa + branch_alpha media delle teste dei rami, ha il suo
    # peso (JEPAConfig.branch_weight; parere GPT d5c53c1)
    spectral_branches: tuple[str, ...] = ()
    branch_dim: int = 128
    branch_hidden: int = 48
    wavelet_learnable: bool = False  # centri e larghezze del banco wavelet apprendibili (entro limiti); dopo un confronto favorevole col fisso
    branch_alpha: float = 0.5
    # obiettivo dei rami a encoder congelato (pilot del punto 2, `docs/rami_pilota_obiettivo_v1.md`): "readout" = i 32 target globali dello
    # Step 1 (il 2C, default), "masked" = predizione locale mascherata (`model/branch_masking.py`, predittori nel modello)
    branch_objective: str = "readout"
    branch_freeze_output: bool = False  # `output` di ogni scala e `combine` fissi al passo 0 (fuori da ottimizzatore ed EMA)
    # testa di posa sulla media dei latenti del backbone per istante (R3, seconda ondata del pilot degli obiettivi, 26/10/2026;
    # `model/pose_head.py`): inizializzata a zero, fuori dal percorso di encode/decode. False = il modello di sempre
    pose_head: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "spectral_branches", tuple(self.spectral_branches))  # da JSON (checkpoint) arriva una lista
        if self.spectral_branches:
            if len(set(self.spectral_branches)) != len(self.spectral_branches) or not set(self.spectral_branches) <= set(BRANCH_KINDS):
                raise ValueError(f"rami spettrali {self.spectral_branches}: distinti, fra {BRANCH_KINDS}")
            if not self.keep_readout:
                raise ValueError("i rami spettrali vogliono keep_readout (calibrazione dei target dello Step 1 e conteggi della loss)")
            if not 0.0 < self.branch_alpha <= 1.0 or self.branch_dim < 1 or self.branch_hidden < 1:
                raise ValueError("rami spettrali: branch_alpha in (0, 1], dimensioni positive")
        if self.wavelet_learnable and "wavelet" not in self.spectral_branches:
            raise ValueError("wavelet_learnable senza il ramo wavelet")
        if self.branch_objective not in ("readout", "masked"):
            raise ValueError(f"branch_objective {self.branch_objective!r}: readout o masked")
        if (self.branch_objective != "readout" or self.branch_freeze_output) and not self.spectral_branches:
            raise ValueError("branch_objective / branch_freeze_output senza rami spettrali")
        if self.branch_objective == "masked" and self.wavelet_learnable:
            raise ValueError("obiettivo mascherato: banco wavelet fisso (il raggio del supporto e' quello dei filtri fissi)")
        if self.window_ms is not None:
            from wearusfm.model.frontend import window_kwargs

            window_kwargs(self.patch_ms, self.window_ms)  # finita e maggiore della patch
            if self.spectral_branches:
                raise ValueError("finestra del kernel piu' lunga della patch e rami spettrali insieme: non previsto")

    @property
    def core_radius(self) -> int:
        """Patch per lato che un token nascosto deve avere nascoste perche' la sua finestra non contenga campioni visibili: 0 senza finestra;
        con 150 ms su 25: la finestra eccede la patch di 62,5 ms per lato, cioe' 2,5 patch -> 3."""
        if self.window_ms is None:
            return 0
        return int(math.ceil((self.window_ms - self.patch_ms) / 2.0 / self.patch_ms - 1e-9))


@dataclass
class ModelInputs:
    """Un batch impacchettato, preparato dal dataloader: segnali normalizzati (uno per campione, (C_s, T_s)) con la loro fs, codici di canale e
    insiemi di attenzione gia' impacchettati (`channel_codes`, `channel_identity.codes_to_tensors`, `local_encoder.sets_to_tensors`), canali per
    campione e validita' QC per canale."""

    signals: list[torch.Tensor]
    fs: list[float]
    codes: dict[str, torch.Tensor]
    sets: dict[str, torch.Tensor]
    counts: list[int]
    qc_valid: torch.Tensor  # (C_tot,) bool
    # solo per i rami spettrali (`spectral_branches`), che li richiedono: per campione il limite superiore della banda per canale in Hz (C_s,)
    # e i campioni (inizio, fine) nelle zone di bordo del tratto (transitori del filtro, `anchor_edge_guard_s`), senza supporto valido
    band_limit: list | None = None
    edge_guard: list | None = None


@dataclass
class Encoded:
    local: torch.Tensor  # (C_tot, P, d) uscite dell'encoder locale (target dell'opzione a di D11)
    z: torch.Tensor  # (S, P, K, d) uscite del backbone
    key_time_valid: torch.Tensor  # (S, P) istanti usati come chiavi
    ids: torch.Tensor  # (C_tot, d) identita' di canale (query del decoder)
    patch_valid: torch.Tensor  # (C_tot, P) patch che esistono (non padding)
    visible: torch.Tensor  # (C_tot, P) token visti dal modello


def zero_hidden_samples(signals: list[torch.Tensor], fs: list[float], visible: torch.Tensor, counts: list[int], patch_s: float) -> list[torch.Tensor]:
    """Azzera i campioni grezzi delle patch nascoste, canale per canale: patch p = campioni [ceil(p*patch*fs), ceil((p+1)*patch*fs)), la stessa
    griglia del front-end (ancorata al primo campione)."""
    out, a = [], 0
    for x, f, c in zip(signals, fs, counts):
        vis = visible[a:a + c]
        a += c
        n = x.shape[-1]
        # inizio delle patch 0..P (l'ultimo = fine della P-1), in float64 con la stessa espressione del front-end (`frontend.py`): in float32
        # l'1e-9 si perdeva e il confine si spostava di un campione in ~20% delle patch a 2 kHz (un campione nascosto restava visibile)
        bounds = torch.from_numpy(np.ceil(np.arange(vis.shape[1] + 1) * patch_s * f - 1e-9).astype(np.int64)).to(x.device)
        # patch di ogni campione; i campioni oltre l'ultima patch (resto finale) restano come sono
        patch_of = torch.searchsorted(bounds, torch.arange(n, device=x.device), right=True) - 1
        in_patch = patch_of < vis.shape[1]
        keep = torch.ones(c, n, dtype=torch.bool, device=x.device)
        keep[:, in_patch] = vis[:, patch_of[in_patch]]
        out.append(x * keep.to(x.dtype))
    return out


class WearUsFM(nn.Module):
    def __init__(self, cfg: FMConfig):
        super().__init__()
        self.cfg = cfg
        from wearusfm.model.frontend import window_kwargs

        self.tokenizer = PatchTokenizer(cfg.dim, patch_ms=cfg.patch_ms, **window_kwargs(cfg.patch_ms, cfg.window_ms))
        self.identity = ChannelIdentity(cfg.dim, muscle_dropout=cfg.muscle_dropout)
        self.local = LocalEncoder(cfg.dim, cfg.n_heads, cfg.local_layers)
        self.pool = PerceiverPooling(cfg.dim, cfg.n_heads, cfg.k_latents)
        self.backbone = TemporalBackbone(cfg.dim, cfg.n_heads, cfg.backbone_layers)
        self.decoder = QueryDecoder(cfg.dim, cfg.n_heads, cfg.decoder_layers)
        self.anchor_heads = AnchorHeads(cfg.dim, cfg.n_bands, multiscale=cfg.multiscale_anchor)
        self.rvq_head = RVQHead(cfg.dim, cfg.rvq_codes) if cfg.rvq_codes else None
        self.spectral_keep = SpectralKeepHead(2 * cfg.dim) if cfg.keep_readout else None
        if cfg.pose_head:  # a zero, senza estrazioni casuali: gli altri moduli nascono identici a un modello senza testa
            from wearusfm.model.pose_head import PoseHead
            self.pose_head = PoseHead(cfg.dim)
        else:
            self.pose_head = None
        self.spectral_branches = self.branch_heads = self.branch_predictors = None
        if cfg.spectral_branches:
            # dopo tutti gli altri moduli e con un generatore a parte: l'encoder raw nasce uguale e il generatore globale (dropout del muscolo)
            # resta quello del modello senza rami, cosi' i confronti con e senza rami sono accoppiati
            # solo il generatore CPU (il modello nasce sulla CPU): torch.manual_seed riseminerebbe anche CUDA, che fork_rng(devices=[]) non
            # ripristina, e il dropout su GPU non resterebbe accoppiato (review GPT del 13/10/2026)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed((torch.initial_seed() + 7919) % 2 ** 63)
                self.spectral_branches = SpectralBranches(cfg.spectral_branches, cfg.branch_dim, cfg.branch_hidden, cfg.wavelet_learnable)
                self.branch_heads = branch_heads(2 * cfg.dim, cfg.spectral_branches, cfg.branch_dim)
                if cfg.branch_objective == "masked":  # dopo rami e teste, dallo stesso generatore: i rami nascono come senza predittori
                    from wearusfm.model.branch_masking import make_predictors
                    self.branch_predictors = make_predictors(cfg.spectral_branches, cfg.branch_hidden, seed=None)
        for m in (self.local, self.backbone, self.decoder):
            m.grad_checkpoint = cfg.grad_checkpoint

    def n_patches(self, inp: ModelInputs) -> int:
        return max(int(math.floor(x.shape[-1] / f / self.tokenizer.patch_s + 1e-9)) for x, f in zip(inp.signals, inp.fs))

    def encode(self, inp: ModelInputs, visible: torch.Tensor | None = None, generator: torch.Generator | None = None) -> Encoded:
        signals = inp.signals
        if visible is not None:
            signals = zero_hidden_samples(signals, inp.fs, visible, inp.counts, self.tokenizer.patch_s)
        tokens, patch_valid, time_valid = self.tokenizer(signals, inp.fs)
        vis = patch_valid & inp.qc_valid[:, None]
        if visible is not None:
            vis = vis & visible
        ids = self.identity(inp.codes, generator)
        local = self.local(tokens + ids[:, None, :], inp.sets, vis)
        offsets = offsets_from_counts(inp.counts)
        lat = self.pool(local, offsets, vis)
        per_sample = torch.stack([vis[a:b].any(dim=0) for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist())])
        key_tv = time_valid & per_sample
        return Encoded(local, self.backbone(lat, key_tv), key_tv, ids, patch_valid, vis)

    def decode(self, enc: Encoded, q_channel: torch.Tensor, q_time: torch.Tensor, q_offsets: torch.Tensor) -> torch.Tensor:
        """Uscite del decoder alle query (canale impacchettato, istante), raggruppate per campione (`q_offsets`)."""
        return self.decoder(enc.z, enc.ids[q_channel], q_time, q_offsets, enc.key_time_valid)
