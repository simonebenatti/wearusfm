"""Backbone temporale globale (passo 6; v10 §5): attenzione fattorizzata tempo / latenti sui K latenti del Perceiver.

Un blocco (pre-norm, residui):
1. **tempo**: per ogni campione e ogni latente, attenzione lungo le P patch temporali, con **RoPE** sull'indice di patch (posizione relativa: la
   patch e' in ms fissi, D10, quindi conta solo la distanza fra istanti);
2. **latenti**: per ogni campione e ogni istante, attenzione fra i K latenti; nessuna posizione (i latenti sono slot appresi e distinti);
3. MLP.

`time_valid` (S, P) toglie dalle **chiavi** dell'attenzione temporale gli istanti che non vanno guardati: il padding delle finestre piu' corte (contesto
variabile, proposta D10) e, se il chiamante lo vuole, gli istanti tutti nascosti (slab), che dal Perceiver escono senza contenuto. *Scelta di AG, da
confermare con D11 (v10 §5.6):* gli istanti nascosti restano come query (la loro uscita e' calcolata), ma non come chiavi; se le predizioni si
leggono dal backbone o da un decoder a query che guarda solo gli istanti visibili e' la decisione D11, non questa.

Implementazione di riferimento in tensori pieni (S, P, K, d) con padding temporale: con finestre di lunghezza molto diversa il padding costa, e in
produzione l'attenzione temporale puo' passare a `flash_attn_varlen_func` (RoPE prima del kernel), da provare su Leonardo.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary embedding sull'ultima dimensione (pari) di x (..., P, H, Dh); positions (..., P) in unita' di patch."""
    dh = x.shape[-1]
    if dh % 2:
        raise ValueError(f"RoPE vuole una dimensione di testa pari, non {dh}")
    freqs = base ** (-torch.arange(0, dh, 2, dtype=torch.float32, device=x.device) / dh)  # (Dh/2,)
    ang = positions.to(torch.float32)[..., None] * freqs  # (..., P, Dh/2)
    cos, sin = ang.cos()[..., None, :], ang.sin()[..., None, :]  # (..., P, 1, Dh/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


class _Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} non divisibile per n_heads {n_heads}")
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, key_valid: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> torch.Tensor:
        """h: (B, L, d); key_valid: (B, L) bool; positions: (B, L) per RoPE o None."""
        b, length, d = h.shape
        q, k, v = self.qkv(h).view(b, length, 3, self.n_heads, self.head_dim).unbind(dim=2)  # (B, L, H, Dh)
        if positions is not None:
            q, k = rope(q, positions), rope(k, positions)
        scores = torch.einsum("blhd,bmhd->bhlm", q, k) / math.sqrt(self.head_dim)
        if key_valid is not None:
            scores = scores.masked_fill(~key_valid[:, None, None, :], float("-inf"))
            any_valid = key_valid.any(dim=-1)[:, None, None, None]
            attn = torch.where(any_valid, torch.softmax(scores, dim=-1), torch.zeros_like(scores))
        else:
            attn = torch.softmax(scores, dim=-1)
        return self.out(torch.einsum("bhlm,bmhd->blhd", attn, v).reshape(b, length, d))


class BackboneBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, *, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_t, self.norm_k, self.norm_m = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.time_attn, self.latent_attn = _Attention(dim, n_heads), _Attention(dim, n_heads)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, z: torch.Tensor, time_valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        s, p, k, d = z.shape
        zt = self.norm_t(z).permute(0, 2, 1, 3).reshape(s * k, p, d)  # sequenze lungo il tempo, una per (campione, latente)
        kv = time_valid[:, None, :].expand(s, k, p).reshape(s * k, p)
        pos = positions[:, None, :].expand(s, k, p).reshape(s * k, p)
        z = z + self.time_attn(zt, kv, pos).view(s, k, p, d).permute(0, 2, 1, 3)
        z = z + self.latent_attn(self.norm_k(z).reshape(s * p, k, d)).view(s, p, k, d)
        return z + self.mlp(self.norm_m(z))


class TemporalBackbone(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_layers: int, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([BackboneBlock(dim, n_heads, **block_kwargs) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, time_valid: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> torch.Tensor:
        """z: (S, P, K, d) dal Perceiver; time_valid: (S, P) bool o None; positions: (S, P) indici di patch o None (0..P-1)."""
        s, p = z.shape[:2]
        if time_valid is None:
            time_valid = torch.ones(s, p, dtype=torch.bool, device=z.device)
        if positions is None:
            positions = torch.arange(p, device=z.device)[None, :].expand(s, p)
        for block in self.blocks:
            z = block(z, time_valid, positions)
        return self.norm(z)

    @staticmethod
    def flops_per_block(n_samples: int, n_patches: int, k_latents: int, dim: int, mlp_ratio: float = 4.0) -> int:
        """Moltiplicazioni-addizioni x 2 di un blocco: proiezioni delle due attenzioni, punteggi e somme pesate (tempo: P^2 per latente;
        latenti: K^2 per istante), MLP."""
        tokens = n_samples * n_patches * k_latents
        proj = 2 * tokens * dim * 4 * dim
        time = n_samples * k_latents * n_patches * n_patches * dim * 2
        lat = n_samples * n_patches * k_latents * k_latents * dim * 2
        mlp = tokens * dim * int(dim * mlp_ratio) * 2
        return 2 * (proj + time + lat + mlp)
