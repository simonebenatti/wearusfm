"""Ciclo JEPA con ancore (passo 6; v10 §6.2, §5.6, §7.1).

**Perdita primaria — predizione latente mascherata** (v10 §6.2): lo studente vede il contesto (`WearUsFM.encode` con `visible`), il decoder a query
predice la rappresentazione di ogni token nascosto (canale, istante); il teacher e' una copia EMA dello studente, con stop-gradient, che vede tutto.
Dove si leggono i target del teacher e' D11 (v10 §5.6), qui un parametro senza default:
- "a": uscita per canale dell'encoder locale del teacher allo stesso (canale, istante);
- "b": uscita del decoder a query del teacher, sull'ingresso non mascherato, alle stesse query (default di lavoro del piano; apre il collasso da
  query, la cui diagnostica e' in `query_decoder.probe_query_collapse`).

**Scelte di AG, da confermare:** target normalizzati con LayerNorm senza parametri (scala comparabile fra passi, nessuna scorciatoia sulla norma);
errore quadratico medio come perdita (l'errore assoluto e' l'alternativa usata in V-JEPA); query = tutti i token nascosti e validi.

**Ancore** (v10 §6.2-6.3): RMS, forma spettrale, inviluppo e, dove c'e', RVQ, dalle stesse uscite del decoder dello studente; peso complessivo
`anchor_weight` (v10: 0,1-0,3), senza default. Momento EMA senza default (lo schedule e' della configurazione del run, D14).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from wearusfm.model.anchor_targets import RVQ_TOKEN_MS, AnchorTargets
from wearusfm.model.anchors import anchor_losses, rvq_loss
from wearusfm.model.fm import ModelInputs, WearUsFM


@dataclass(frozen=True)
class JEPAConfig:
    target: str  # "a" | "b" (D11)
    anchor_weight: float  # v10 §6.2: 0,1-0,3
    ema_momentum: float

    def __post_init__(self) -> None:
        if self.target not in ("a", "b"):
            raise ValueError(f"target {self.target!r}: 'a' o 'b' (D11)")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum in [0, 1)")


def make_teacher(student: WearUsFM) -> WearUsFM:
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, momentum: float) -> None:
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(momentum).add_(ps.detach(), alpha=1.0 - momentum)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


def hidden_queries(hidden: torch.Tensor, counts: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tutti i token nascosti (canale impacchettato, istante), in ordine di canale e quindi raggruppati per campione; offset delle query."""
    ch, t = torch.nonzero(hidden, as_tuple=True)
    bounds = torch.cumsum(torch.as_tensor([0, *counts]), 0)
    per_sample = torch.stack([((ch >= a) & (ch < b)).sum() for a, b in zip(bounds[:-1], bounds[1:])])
    return ch, t, torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(per_sample, 0)])


def gather_anchor_targets(targets: list[AnchorTargets], counts: list[int], q_ch: torch.Tensor, q_t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Dai target per campione (`anchor_targets`, calcolati dal dataloader sul segnale pulito) ai target per query."""
    offs = np.cumsum([0, *counts])
    out: dict[str, list] = {k: [] for k in ("log_rms", "rms_valid", "band_shape", "spec_valid", "log_env", "env_valid", "band_available")}
    for c, t in zip(q_ch.tolist(), q_t.tolist()):
        s = int(np.searchsorted(offs, c, side="right") - 1)
        tg, cc = targets[s], c - offs[s]
        inside = t < tg.log_rms.shape[1]
        for k in ("log_rms", "log_env"):
            out[k].append(float(getattr(tg, k)[cc, t]) if inside else 0.0)
        out["rms_valid"].append(bool(tg.rms_valid[cc, t]) if inside else False)
        out["env_valid"].append(bool(tg.env_valid[cc, t]) if inside else False)
        out["spec_valid"].append(bool(tg.spec_valid[cc, t]) if inside else False)
        out["band_shape"].append(tg.band_shape[cc, t] if inside else np.zeros(tg.band_shape.shape[2]))
        out["band_available"].append(tg.band_available[cc])
    res = {k: torch.as_tensor(np.asarray(v)) for k, v in out.items()}
    return {k: (v.float() if v.is_floating_point() else v) for k, v in res.items()}


def effective_rank(x: torch.Tensor) -> float:
    """exp dell'entropia dei valori singolari normalizzati, sulle righe centrate (v10 §7.1 punto 3)."""
    x = x.detach().to(torch.float64)
    s = torch.linalg.svdvals(x - x.mean(dim=0, keepdim=True))
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def jepa_losses(student: WearUsFM, teacher: WearUsFM, inp: ModelInputs, visible: torch.Tensor, cfg: JEPAConfig,
                anchor_targets: list[AnchorTargets] | None = None, rvq_on: torch.Tensor | None = None,
                rvq_codes: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """visible: (C_tot, P) True = visto dallo studente. rvq_on: (C_tot,) canali con l'ancora RVQ accesa; rvq_codes(canale, finestra) -> codici
    del tokenizer congelato (livello 0 del ramo 0) per quelle finestre."""
    enc = student.encode(inp, visible, generator)
    hidden = ~enc.visible & enc.patch_valid & inp.qc_valid[:, None]
    q_ch, q_t, q_off = hidden_queries(hidden, inp.counts)
    if q_ch.numel() == 0:
        raise ValueError("nessun token nascosto: la maschera non chiede nessuna predizione")
    h = student.decode(enc, q_ch, q_t, q_off)
    with torch.no_grad():
        enc_t = teacher.encode(inp, None)
        tgt = enc_t.local[q_ch, q_t] if cfg.target == "a" else teacher.decode(enc_t, q_ch, q_t, q_off)
        tgt = F.layer_norm(tgt, tgt.shape[-1:])
    losses = {"jepa": F.mse_loss(h, tgt)}
    anchor_total = h.new_zeros(())
    if anchor_targets is not None:
        for k, v in anchor_losses(student.anchor_heads(h), gather_anchor_targets(anchor_targets, inp.counts, q_ch, q_t)).items():
            losses[k] = v
            anchor_total = anchor_total + v
    if student.rvq_head is not None and rvq_on is not None and rvq_codes is not None:
        per = int(round(RVQ_TOKEN_MS / student.cfg.patch_ms))
        c_tot, p = hidden.shape
        w = p // per
        # solo slab: finestre nascoste su TUTTI i canali validi del campione, mai il masking di canale (v10 §6.3)
        win = torch.zeros(c_tot, w, dtype=torch.bool)
        bounds = np.cumsum([0, *inp.counts])
        for a, b in zip(bounds[:-1], bounds[1:]):
            ok = inp.qc_valid[a:b]
            slab = hidden[a:b][ok][:, : w * per].all(dim=0).reshape(w, per).all(dim=-1) if bool(ok.any()) else torch.zeros(w, dtype=torch.bool)
            win[a:b] = slab[None, :] & ok[:, None]
        win &= rvq_on[:, None]
        wc, ww = torch.nonzero(win, as_tuple=True)
        index = torch.full((c_tot, p), -1, dtype=torch.long)
        index[q_ch, q_t] = torch.arange(q_ch.numel())
        rows = index[wc[:, None], ww[:, None] * per + torch.arange(per)[None, :]]  # (N, 8), tutte query nascoste per costruzione
        losses["rvq"] = rvq_loss(student.rvq_head(h[rows]), rvq_codes(wc, ww)) if wc.numel() else h.sum() * 0.0
        anchor_total = anchor_total + losses["rvq"]
    losses["total"] = losses["jepa"] + cfg.anchor_weight * anchor_total
    return losses


def train_step(student: WearUsFM, teacher: WearUsFM, optimizer: torch.optim.Optimizer, inp: ModelInputs, visible: torch.Tensor,
               cfg: JEPAConfig, **kwargs) -> dict[str, float]:
    student.train()
    optimizer.zero_grad(set_to_none=True)
    losses = jepa_losses(student, teacher, inp, visible, cfg, **kwargs)
    losses["total"].backward()
    optimizer.step()
    ema_update(teacher, student, cfg.ema_momentum)
    return {k: float(v.detach()) for k, v in losses.items()}
