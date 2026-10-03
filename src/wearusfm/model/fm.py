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
    n_bands: int = 5
    rvq_codes: int | None = None  # None = senza testa RVQ
    grad_checkpoint: bool = False  # activation checkpointing di encoder locale, backbone e decoder (v10 §5.5)


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
        self.tokenizer = PatchTokenizer(cfg.dim, patch_ms=cfg.patch_ms)
        self.identity = ChannelIdentity(cfg.dim, muscle_dropout=cfg.muscle_dropout)
        self.local = LocalEncoder(cfg.dim, cfg.n_heads, cfg.local_layers)
        self.pool = PerceiverPooling(cfg.dim, cfg.n_heads, cfg.k_latents)
        self.backbone = TemporalBackbone(cfg.dim, cfg.n_heads, cfg.backbone_layers)
        self.decoder = QueryDecoder(cfg.dim, cfg.n_heads, cfg.decoder_layers)
        self.anchor_heads = AnchorHeads(cfg.dim, cfg.n_bands)
        self.rvq_head = RVQHead(cfg.dim, cfg.rvq_codes) if cfg.rvq_codes else None
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
