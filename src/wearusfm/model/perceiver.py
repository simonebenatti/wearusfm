"""Pooling latente Perceiver (passo 6; v10 §5, §5.1): da C token di canale a K latenti, **per patch temporale**.

Per ogni campione s e ogni patch temporale p, K latenti appresi (condivisi) fanno cross-attention sui token **visibili** dei canali di s allo
stesso istante. K e' la risoluzione spaziale del bottleneck (v10 §5.1), non un tetto informativo; i latenti non corrispondono ai canali (§5.6).
Nessun encoding posizionale sui canali: l'identita' e' gia' nei token (identita' di canale + encoder locale), quindi l'uscita e' invariante
all'ordine dei canali dentro un campione.

**Packing** (D6b): i canali di piu' campioni stanno in un'unica sequenza (C_tot, P, d), contigui per campione; `channel_offsets` (S+1,) fa da
`cu_seqlens`. Questa e' l'implementazione di riferimento (un ciclo sui campioni, provabile su CPU); in produzione il kernel e'
`flash_attn_varlen_func` con le sequenze (campione, patch) (v10 §4.6, prova funzionale fatta al passo 0): l'equivalenza fra i due va provata su
Leonardo prima di usarlo.

**Token nascosti** (masking JEPA, v10 §6.4): `visible` (C_tot, P) toglie dalle chiavi i token mascherati e i canali scartati dal QC. *Scelta di AG,
da confermare:* dove a un istante non resta nessun token visibile (slab su tutti i canali, il caso dell'ancora RVQ), il contributo della
cross-attention e' zero e i latenti restano quelli appresi (piu' l'MLP): ricostruire quell'istante tocca al backbone temporale, dal contesto.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def offsets_from_counts(counts) -> torch.Tensor:
    c = torch.as_tensor(list(counts), dtype=torch.long)
    return torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(c, 0)])


class PerceiverPooling(nn.Module):
    def __init__(self, dim: int, n_heads: int, k_latents: int, *, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} non divisibile per n_heads {n_heads}")
        self.n_heads, self.head_dim, self.k_latents = n_heads, dim // n_heads, k_latents
        self.latents = nn.Parameter(torch.randn(k_latents, dim) * 0.02)
        self.norm_q, self.norm_kv, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, channel_offsets: torch.Tensor, visible: torch.Tensor | None = None) -> torch.Tensor:
        """x: (C_tot, P, d); channel_offsets: (S+1,); visible: (C_tot, P) bool o None (tutto visibile). Ritorna (S, P, K, d)."""
        c_tot, p, d = x.shape
        off = [int(v) for v in channel_offsets]
        if off[0] != 0 or off[-1] != c_tot or any(b <= a for a, b in zip(off, off[1:])):
            raise ValueError(f"channel_offsets {off} incoerenti con {c_tot} canali (ogni campione almeno un canale, in ordine)")
        if visible is None:
            visible = torch.ones(c_tot, p, dtype=torch.bool, device=x.device)
        q = self.q(self.norm_q(self.latents)).view(self.k_latents, self.n_heads, self.head_dim)
        kv = self.kv(self.norm_kv(x)).view(c_tot, p, 2, self.n_heads, self.head_dim)
        outs = []
        for a, b in zip(off, off[1:]):
            k, v = kv[a:b].unbind(dim=2)  # (C_s, P, H, Dh)
            scores = torch.einsum("khd,cphd->phkc", q, k) / math.sqrt(self.head_dim)
            vis = visible[a:b].T[:, None, None, :]  # (P, 1, 1, C_s)
            scores = scores.masked_fill(~vis, float("-inf"))
            any_vis = vis.any(dim=-1, keepdim=True)
            attn = torch.where(any_vis, torch.softmax(scores, dim=-1), torch.zeros_like(scores))  # istante tutto nascosto: nessun contenuto
            outs.append(torch.einsum("phkc,cphd->pkhd", attn, v).reshape(p, self.k_latents, d))
        lat = self.latents + self.out(torch.stack(outs))  # (S, P, K, d)
        return lat + self.mlp(self.norm_mlp(lat))

    @staticmethod
    def flops(n_channels_total: int, n_samples: int, n_patches: int, k_latents: int, dim: int, mlp_ratio: float = 4.0) -> int:
        """Moltiplicazioni-addizioni x 2: proiezione delle chiavi e dei valori, cross-attention, uscita e MLP sui latenti."""
        kv = n_channels_total * n_patches * dim * 2 * dim
        attn = n_channels_total * n_patches * k_latents * dim * 2
        lat = n_samples * n_patches * k_latents * (dim * dim + dim * int(dim * mlp_ratio) * 2)
        return 2 * (kv + attn + lat)
