"""Encoder spaziale locale (passo 6; v10 §5.3-5.4): attenzione di ogni canale sulle sue chiavi locali (`channel_codes.attention_sets`), raccolte con
`gather` (rev. 3: i kernel flash non accettano un bias additivo arbitrario, e in questa forma i canali variabili e il packing non richiedono
padding fra campioni), con il bias geometrico di §5.3.

**Bias = componente funzionale fissa + componente appresa su una base di distanze** (v10 §5.3):
- fissa: `-d`, con d in passi d'elettrodo del gruppo (decadimento con la distanza, passa-basso spaziale del volume conduttore, v10 §3.1); non
  appresa, fa da inizializzazione fisicamente sensata;
- appresa: basi radiali gaussiane sulla distanza, pesi per testa e per topologia, **inizializzati a zero** (all'inizio vale solo la componente
  fissa). Sulle griglie due basi separate per |riga| e |colonna| (v10 §3.4: «asse fibre privilegiato», isotropia rotta dalla direzione delle
  fibre; quale dei due assi sia quello delle fibre i sidecar non lo dicono, il modello puo' impararlo);
- per tipo di coppia (se stesso, anello, griglia, insieme senza geometria) uno scalare appreso per testa.

*Scelta di AG, da confermare:* il bias dipende dalla distanza e non dal verso dello spostamento angolare, quindi e' invariante alla riflessione
dell'anello; la direzione di propagazione resta nel segnale, non nel bias (v10 §3: la velocita' di conduzione non va insegnata).

Forme: token (C, P, d) — C canali (anche di piu' montaggi impacchettati, `pack_attention_sets`), P patch temporali — e chiavi (C, K). Ogni
patch temporale e' trattata indipendentemente: l'attenzione e' fra canali allo stesso istante.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from wearusfm.model.channel_codes import PAIR_GRID, PAIR_RING, PAIR_SET, AttentionSets

N_PAIR_TYPES = PAIR_SET + 1


def sets_to_tensors(sets: AttentionSets, device=None) -> dict[str, torch.Tensor]:
    return {
        "index": torch.as_tensor(sets.index, dtype=torch.long, device=device),
        "pair_type": torch.as_tensor(sets.pair_type, dtype=torch.long, device=device),
        "dist": torch.as_tensor(sets.dist, dtype=torch.float32, device=device),
        "d_row": torch.as_tensor(sets.d_row, dtype=torch.float32, device=device),
        "d_col": torch.as_tensor(sets.d_col, dtype=torch.float32, device=device),
    }


class GeometricBias(nn.Module):
    """(C, K) geometria delle coppie -> (C, K, H) bias additivo sui logit."""

    def __init__(self, n_heads: int, *, n_basis: int = 8, max_dist: float = 4.0, fixed_decay: float = 1.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, max_dist, n_basis))
        self.width = max_dist / (n_basis - 1)
        self.fixed_decay = float(fixed_decay)
        self.ring_w = nn.Parameter(torch.zeros(n_basis, n_heads))
        self.row_w = nn.Parameter(torch.zeros(n_basis, n_heads))
        self.col_w = nn.Parameter(torch.zeros(n_basis, n_heads))
        self.type_bias = nn.Parameter(torch.zeros(N_PAIR_TYPES, n_heads))

    def _rbf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(((torch.nan_to_num(x, nan=0.0)[..., None] - self.centers) / self.width) ** 2))

    def forward(self, g: dict[str, torch.Tensor]) -> torch.Tensor:
        t = g["pair_type"]
        ring, grid = (t == PAIR_RING)[..., None], (t == PAIR_GRID)[..., None]
        bias = self.type_bias[t.clamp(min=0)]
        bias = bias + torch.where(ring, self._rbf(g["dist"]) @ self.ring_w, 0.0)
        bias = bias + torch.where(grid, self._rbf(g["d_row"]) @ self.row_w + self._rbf(g["d_col"]) @ self.col_w, 0.0)
        return bias - self.fixed_decay * torch.nan_to_num(g["dist"], nan=0.0)[..., None]  # componente fissa: 0 dove la coppia non e' metrica


class LocalEncoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, *, mlp_ratio: float = 4.0, bias_kwargs: dict | None = None):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} non divisibile per n_heads {n_heads}")
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.bias = GeometricBias(n_heads, **(bias_kwargs or {}))
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, g: dict[str, torch.Tensor], visible: torch.Tensor | None = None) -> torch.Tensor:
        """visible: (C, P) bool o None. Un token nascosto non e' chiave per nessun altro canale (lo studente JEPA non deve vederlo); il canale
        stesso resta sempre ammesso (la sua uscita, se nascosto, non la usa nessuno: il Perceiver lo esclude)."""
        c, p, d = x.shape
        idx = g["index"]
        empty = idx < 0
        h = self.norm1(x)
        q = self.q(h).view(c, p, self.n_heads, self.head_dim)
        kv = self.kv(h).view(c, p, 2, self.n_heads, self.head_dim)
        kv_g = kv[idx.clamp(min=0)]  # (C, K, P, 2, H, Dh)
        k_g, v_g = kv_g.unbind(dim=3)
        scores = torch.einsum("cphd,ckphd->ckph", q, k_g) / math.sqrt(self.head_dim)
        scores = scores + self.bias(g)[:, :, None, :]
        scores = scores.masked_fill(empty[:, :, None, None], float("-inf"))
        if visible is not None:
            hidden_key = ~visible[idx.clamp(min=0)]  # (C, K, P)
            hidden_key[:, 0] = False  # se stesso sempre ammesso
            scores = scores.masked_fill(hidden_key[..., None], float("-inf"))
        attn = torch.softmax(scores, dim=1)  # la colonna 0 (il canale stesso) non e' mai vuota
        out = torch.einsum("ckph,ckphd->cphd", attn, v_g).reshape(c, p, d)
        x = x + self.out(out)
        return x + self.mlp(self.norm2(x))


class LocalEncoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_layers: int, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([LocalEncoderLayer(dim, n_heads, **layer_kwargs) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, g: dict[str, torch.Tensor], visible: torch.Tensor | None = None) -> torch.Tensor:
        if g["index"].shape[0] != x.shape[0] or (g["index"][:, 0] != torch.arange(x.shape[0], device=x.device)).any():
            raise ValueError("chiavi incoerenti con i token: la colonna 0 deve essere il canale stesso")
        for layer in self.layers:
            x = layer(x, g, visible)
        return x

    @staticmethod
    def flops_per_layer(n_channels: int, n_keys: int, n_patches: int, dim: int, mlp_ratio: float = 4.0) -> int:
        """Moltiplicazioni-addizioni x 2 di un livello (proiezioni, attenzione sulle chiavi, MLP); il bias e le norme sono trascurabili.
        Serve al conto dei FLOP per modulo del passo 6 («Chiuso quando»)."""
        tokens = n_channels * n_patches
        proj = tokens * dim * (dim + 2 * dim + dim)  # q, kv, out
        attn = tokens * n_keys * dim * 2  # q.k e somma pesata dei v
        mlp = tokens * dim * int(dim * mlp_ratio) * 2
        return 2 * (proj + attn + mlp)

