"""Decoder a query (passo 6; v10 §5, §5.6) e diagnostica del collasso da query (v10 §7.1, punto 4).

**Decoder.** I K latenti non corrispondono ai canali (§5.6): per predire il token di un canale a un istante serve una query. Query = identita' del
canale (`ChannelIdentity`, nota anche per i canali mascherati) all'istante t; la query fa cross-attention su tutti i latenti del suo campione (P x K
token del backbone), con **RoPE sul tempo**: conta la distanza fra l'istante della query e quello di ciascun latente (stessa convenzione del
backbone). Piu' livelli (cross-attention + MLP) aggiornano le query. Le chiavi escludono gli istanti non validi (`time_valid`: padding e istanti
nascosti allo studente). Lo stesso modulo serve alle due opzioni di D11: (a) e (b) differiscono per dove si leggono i target del teacher, non per
come predice lo studente; le teste delle ancore leggono dallo stesso decoder (§5.6).

**Collasso da query** (§7.1, punto 4): un'uscita che dipende dalla query e non dal contenuto ha loss bassa e rango alto (il rango effettivo non lo
vede). Diagnostica: le **stesse query di sonda** valutate sui latenti di S campioni diversi -> (S, Q, d); varianza fra campioni a query fissa contro
varianza fra query a campione fisso. Rapporto vicino a zero = collasso.

Implementazione di riferimento (un ciclo sui campioni); in produzione la cross-attention puo' passare a `flash_attn_varlen_func`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from wearusfm.model.backbone import rope


class _CrossBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} non divisibile per n_heads {n_heads}")
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.norm_q, self.norm_kv, self.norm_m = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, qx: torch.Tensor, q_pos: torch.Tensor, kv_x: torch.Tensor, kv_pos: torch.Tensor, kv_valid: torch.Tensor) -> torch.Tensor:
        """qx: (Q, d), q_pos: (Q,); kv_x: (N, d), kv_pos: (N,), kv_valid: (N,) bool."""
        nq, d = qx.shape
        q = rope(self.q(self.norm_q(qx)).view(1, nq, self.n_heads, self.head_dim), q_pos[None])[0]
        k, v = self.kv(self.norm_kv(kv_x)).view(-1, 2, self.n_heads, self.head_dim).unbind(dim=1)
        k = rope(k[None], kv_pos[None])[0]
        scores = torch.einsum("qhd,nhd->hqn", q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~kv_valid[None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1) if bool(kv_valid.any()) else torch.zeros_like(scores)
        qx = qx + self.out(torch.einsum("hqn,nhd->qhd", attn, v).reshape(nq, d))
        return qx + self.mlp(self.norm_m(qx))


class QueryDecoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_layers: int, *, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([_CrossBlock(dim, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, queries: torch.Tensor, query_time: torch.Tensor, query_offsets: torch.Tensor,
                time_valid: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> torch.Tensor:
        """z: (S, P, K, d) dal backbone; queries: (Q_tot, d) identita' dei canali da predire; query_time: (Q_tot,) indice di patch;
        query_offsets: (S+1,) query contigue per campione; time_valid: (S, P) chiavi ammesse; positions: (S, P) indici di patch (default 0..P-1)."""
        s, p, k, d = z.shape
        off = [int(v) for v in query_offsets]
        if len(off) != s + 1 or off[0] != 0 or off[-1] != queries.shape[0] or any(b < a for a, b in zip(off, off[1:])):
            raise ValueError(f"query_offsets {off} incoerenti con {s} campioni e {queries.shape[0]} query")
        if time_valid is None:
            time_valid = torch.ones(s, p, dtype=torch.bool, device=z.device)
        if positions is None:
            positions = torch.arange(p, device=z.device)[None, :].expand(s, p)
        outs = []
        for i, (a, b) in enumerate(zip(off, off[1:])):
            kv_x = z[i].reshape(p * k, d)
            kv_pos = positions[i][:, None].expand(p, k).reshape(-1)
            kv_valid = time_valid[i][:, None].expand(p, k).reshape(-1)
            q_pos = query_time[a:b] + (positions[i, 0] if p else 0)  # l'istante della query nella stessa scala delle posizioni dei latenti
            qx = queries[a:b]
            for block in self.blocks:
                qx = block(qx, q_pos, kv_x, kv_pos, kv_valid)
            outs.append(qx)
        return self.norm(torch.cat(outs, dim=0))


def query_collapse_stats(outputs: torch.Tensor) -> dict[str, float]:
    """outputs: (S, Q, d) = le stesse Q query di sonda valutate su S campioni. Varianza fra campioni a query fissa (media sulle query e sulle
    dimensioni) contro varianza fra query a campione fisso; `ratio` = la prima diviso la seconda. Vicino a zero: l'uscita dipende dalla query e non
    dal contenuto (collasso da query, v10 §7.1 punto 4)."""
    if outputs.ndim != 3 or outputs.shape[0] < 2 or outputs.shape[1] < 2:
        raise ValueError("servono almeno 2 campioni e 2 query: forma (S, Q, d)")
    x = outputs.detach().to(torch.float64)
    between_samples = float(x.var(dim=0, unbiased=False).mean())
    between_queries = float(x.var(dim=1, unbiased=False).mean())
    return {"var_between_samples": between_samples, "var_between_queries": between_queries,
            "ratio": between_samples / between_queries if between_queries > 0 else float("nan")}


def probe_query_collapse(decoder: QueryDecoder, z: torch.Tensor, probe_queries: torch.Tensor, probe_time: torch.Tensor,
                         time_valid: torch.Tensor | None = None) -> dict[str, float]:
    """Valuta le stesse query di sonda (Q, d) agli istanti `probe_time` (Q,) su tutti gli S campioni di z e ne calcola le statistiche."""
    s = z.shape[0]
    q = probe_queries.shape[0]
    with torch.no_grad():
        out = decoder(z, probe_queries.repeat(s, 1), probe_time.repeat(s), torch.arange(0, (s + 1) * q, q), time_valid)
    return query_collapse_stats(out.view(s, q, -1))
