"""Modello proxy per l'asse L (layout del batch), spec §7.

Richiede CUDA e la libreria flash_attn: non importabile ne' testabile sul Mac (nessuna
unit test in tests/cpu/ per questo modulo). Verificato solo su Leonardo.

**Semplificazione dichiarata**, rispetto all'architettura completa di v10 §5: il bottleneck
Perceiver qui attende su TUTTI i token canale-patch di un campione in un colpo solo
(niente fattorizzazione per singolo patch temporale). Cattura comunque la struttura di
costo che l'asse L deve confrontare - spreco di padding nel front-end/encoder locale,
meccanismo di attenzione del Perceiver (padded / packed / a bucket), costo del backbone -
ma non e' la topologia finale del modello. Una versione fattorizzata per patch e' un
raffinamento successivo, da fare se il confronto grezzo non basta a decidere D6b.

Front-end: dato che i campioni hanno frequenza nativa diversa (200-5120 Hz) e quindi
lunghezza in campioni diversa a parita' di context_s, ogni patch (definita in ms, v10
§4.1) viene ridotta con adaptive average pooling a una lunghezza canonica fissa PRIMA
di una singola proiezione lineare condivisa: FLOP-equivalente al banco di kernel
continui di v10 §4.2 (stesso ordine di grandezza), non identico bit per bit - lo
consente esplicitamente spec §7 ("non serve che impari: serve che costi quanto il
modello vero").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from wearusfm.model.geometry import build_neighbors  # noqa: F401 - re-esportata per comodita'

try:
    from flash_attn import flash_attn_varlen_func

    HAVE_FLASH_ATTN = True
except ImportError:  # pragma: no cover - solo su Leonardo
    HAVE_FLASH_ATTN = False

FRONT_END_CANONICAL_LEN = 64


class FrontEnd(nn.Module):
    def __init__(self, d_model: int, canonical_len: int = FRONT_END_CANONICAL_LEN):
        super().__init__()
        self.canonical_len = canonical_len
        self.proj = nn.Linear(canonical_len, d_model)

    def forward(self, x: torch.Tensor, n_patches: int) -> torch.Tensor:
        """x: (rows, T) -> (rows, n_patches, d_model)."""
        rows, t = x.shape
        patch_len = t // n_patches
        x = x[:, : patch_len * n_patches].reshape(rows * n_patches, 1, patch_len)
        x = F.adaptive_avg_pool1d(x, self.canonical_len).reshape(rows, n_patches, self.canonical_len)
        return self.proj(x)


class LocalEncoderLayer(nn.Module):
    """Attenzione sui k vicini via gather, con bias geometrico (v10 §5.3, rev. 3 §5.4:
    i kernel flash non accettano bias additivo arbitrario, quindi non si usano qui)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.bias_mlp = nn.Linear(1, n_heads)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_dist: torch.Tensor) -> torch.Tensor:
        """tokens: (C, P, d) di UN campione. neighbor_idx/dist: (C, k)."""
        c, p, d = tokens.shape
        k = neighbor_idx.shape[1]
        qkv = self.qkv(tokens).view(c, p, 3, self.n_heads, self.head_dim)
        q, kk, v = qkv.unbind(dim=2)  # ciascuno (C, P, H, Dh)

        kk_g = kk[neighbor_idx]  # (C, k, P, H, Dh)
        v_g = v[neighbor_idx]  # (C, k, P, H, Dh)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("cphd,ckphd->ckph", q, kk_g) * scale
        bias = self.bias_mlp(neighbor_dist.unsqueeze(-1))  # (C, k, H)
        scores = scores + bias.unsqueeze(2)  # broadcast su P

        attn = torch.softmax(scores, dim=1)  # softmax sui k vicini
        out = torch.einsum("ckph,ckphd->cphd", attn, v_g).reshape(c, p, d)
        return tokens + self.out(out)


class LocalEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([LocalEncoderLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, tokens: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_dist: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, neighbor_idx, neighbor_dist)
        return tokens


class PerceiverBottleneck(nn.Module):
    def __init__(self, d_model: int, n_heads: int, k_latents: int = 64):
        super().__init__()
        assert d_model % n_heads == 0
        self.k_latents = k_latents
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.latents = nn.Parameter(torch.randn(k_latents, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _q(self, b: int, device, dtype) -> torch.Tensor:
        q = self.q_proj(self.latents).to(dtype)
        return q.unsqueeze(0).expand(b, -1, -1).to(device)

    def forward_padded(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L, d) con L = C_max * P, gia' appiattito. valid_mask: (B, L) bool."""
        b, l, d = tokens.shape
        kv = self.kv_proj(tokens)
        k, v = kv.chunk(2, dim=-1)
        q = self._q(b, tokens.device, tokens.dtype)

        q = q.view(b, self.k_latents, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = torch.zeros(b, 1, 1, l, device=tokens.device, dtype=tokens.dtype)
        attn_mask.masked_fill_(~valid_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, self.k_latents, -1)
        return self.out_proj(out)

    def forward_bucket(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L, d), L uniforme nel bucket - nessuna maschera necessaria."""
        return self.forward_padded(tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device))

    def forward_packed(self, tokens_flat: torch.Tensor, cu_seqlens_k: torch.Tensor, batch_size: int) -> torch.Tensor:
        """tokens_flat: (sum(L_i), d), packed. cu_seqlens_k: (B+1,) int32 su CUDA."""
        if not HAVE_FLASH_ATTN:
            raise RuntimeError("flash_attn non disponibile: il braccio L2 richiede l'ambiente Leonardo")
        kv = self.kv_proj(tokens_flat)
        k, v = kv.chunk(2, dim=-1)
        total_k = tokens_flat.shape[0]
        k = k.view(total_k, self.n_heads, self.head_dim)
        v = v.view(total_k, self.n_heads, self.head_dim)

        q = self._q(batch_size, tokens_flat.device, tokens_flat.dtype)
        q = q.reshape(batch_size * self.k_latents, self.n_heads, self.head_dim)
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * self.k_latents, self.k_latents, dtype=torch.int32, device=tokens_flat.device
        )
        max_seqlen_q = self.k_latents
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())

        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False
        )
        out = out.reshape(batch_size, self.k_latents, -1)
        return self.out_proj(out)


class Backbone(nn.Module):
    """Sostituto del backbone temporale: prodotti di matrici con FLOP equivalenti
    (v10 §4.6, spec §7), non il backbone fattorizzato tempo/canale vero."""

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


@dataclass
class ProxyConfig:
    d_model: int = 384
    n_heads: int = 6
    n_local_layers: int = 2
    n_backbone_layers: int = 10
    k_latents: int = 64
    k_neighbors: int = 8


class ProxyModel(nn.Module):
    def __init__(self, cfg: ProxyConfig):
        super().__init__()
        self.cfg = cfg
        self.front_end = FrontEnd(cfg.d_model)
        self.local_encoder = LocalEncoder(cfg.d_model, cfg.n_heads, cfg.n_local_layers)
        self.perceiver = PerceiverBottleneck(cfg.d_model, cfg.n_heads, cfg.k_latents)
        self.backbone = Backbone(cfg.d_model, cfg.n_backbone_layers)
        self.head = nn.Linear(cfg.d_model, cfg.d_model)

    def finish(self, perceiver_out: torch.Tensor) -> torch.Tensor:
        x = self.backbone(perceiver_out)
        return self.head(x)
