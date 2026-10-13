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
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._pytree import register_pytree_node

from wearusfm.model.anchor_targets import MS_SCALES, RVQ_TOKEN_MS, AnchorTargets
from wearusfm.model.anchors import anchor_losses, anchor_loss_terms, masked_mean, masked_sum_count, rvq_loss, rvq_loss_terms
from wearusfm.model.channel_codes import TOPOLOGIES
from wearusfm.model.fm import FMConfig, ModelInputs, WearUsFM
from wearusfm.model.spectral_step1 import global_spectral_targets, keep_objective, keep_objective_terms
from wearusfm.training.masking import KIND_NAMES, VISIBLE


@dataclass(frozen=True)
class JEPAConfig:
    target: str  # "a" | "b" (D11)
    anchor_weight: float  # v10 §6.2: 0,1-0,3
    ema_momentum: float
    # media della perdita prima dentro ogni finestra, poi fra le finestre (Simone, 04/10/2026, decisione 2; dalla finestra 1). False = media per
    # token, com'era nel sanity collaudato
    loss_per_window: bool = False
    # finestre dell'ancora RVQ: "slab" (nascoste su tutti i canali, v10 §6.3, l'unico uso ammesso nel training) oppure "channel", SOLO DIAGNOSTICA
    # (Simone, 04/10/2026: «sì, fai la prova con la maschera per canale»): nascoste su quel canale ma non su tutti, con dei vicini visibili. In
    # tutti e due i casi si registra anche l'altra perdita, senza gradiente, e l'accuratezza top-1
    rvq_targets: str = "slab"
    keep_weight: float = 0.0
    # rami spettrali (Claude 13/10/2026, parere GPT d5c53c1): peso del termine dei rami sul readout, separato da keep_weight (che resta la keep
    # dello Step 1 sulla sola testa raw); 0 = nessun termine dei rami
    branch_weight: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.keep_weight) or self.keep_weight < 0:
            raise ValueError("keep_weight must be finite and nonnegative")
        if not math.isfinite(self.branch_weight) or self.branch_weight < 0:
            raise ValueError("branch_weight must be finite and nonnegative")
        if self.target not in ("a", "b"):
            raise ValueError(f"target {self.target!r}: 'a' o 'b' (D11)")
        if self.rvq_targets not in ("slab", "channel"):
            raise ValueError(f"rvq_targets {self.rvq_targets!r}: 'slab' o 'channel' (diagnostica)")
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
    bounds = torch.cumsum(torch.as_tensor([0, *counts], device=hidden.device), 0)
    return ch, t, torch.searchsorted(ch, bounds)  # ch e' ordinato: l'offset del campione s e' la prima query con canale >= bounds[s]


def gather_anchor_targets(targets: list[AnchorTargets], counts: list[int], q_ch: torch.Tensor, q_t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Dai target per campione (`anchor_targets`, calcolati dal dataloader sul segnale pulito) ai target per query, sul dispositivo delle query.
    Una query oltre la fine dei target del suo campione ha valori nulli e non e' valida."""
    c_tot, n_b = sum(counts), targets[0].band_shape.shape[2]
    ch, t = q_ch.cpu().numpy(), q_t.cpu().numpy()
    p = max([tg.log_rms.shape[1] for tg in targets] + [int(t.max()) + 1 if t.size else 1])
    full = {"log_rms": np.zeros((c_tot, p), np.float32), "log_env": np.zeros((c_tot, p), np.float32),
            "rms_valid": np.zeros((c_tot, p), bool), "env_valid": np.zeros((c_tot, p), bool), "spec_valid": np.zeros((c_tot, p), bool),
            "band_shape": np.zeros((c_tot, p, n_b), np.float32)}
    per_ch = {"band_available": np.zeros((c_tot, n_b), bool)}
    for name, _, nb in MS_SCALES:  # ancora multi-scala, se i target ci sono
        if getattr(targets[0], name) is not None:
            full[name] = np.zeros((c_tot, p, nb), np.float32)
            full[name + "_valid"] = np.zeros((c_tot, p), bool)
            per_ch[name + "_available"] = np.zeros((c_tot, nb), bool)
    a = 0
    for tg, c in zip(targets, counts):
        n = tg.log_rms.shape[1]
        for k, v in full.items():
            v[a:a + c, :n] = getattr(tg, k)
        for k, v in per_ch.items():
            v[a:a + c] = getattr(tg, k)
        a += c
    out = {k: torch.from_numpy(np.ascontiguousarray(v[ch, t])) for k, v in full.items()}
    out.update({k: torch.from_numpy(np.ascontiguousarray(v[ch])) for k, v in per_ch.items()})
    return {k: v.to(q_ch.device) for k, v in out.items()}


def effective_rank(x: torch.Tensor) -> float:
    """exp dell'entropia dei valori singolari normalizzati, sulle righe centrate (v10 §7.1 punto 3)."""
    x = x.detach().to(torch.float64)
    s = torch.linalg.svdvals(x - x.mean(dim=0, keepdim=True))
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@dataclass
class LossTerms:
    """Local additive statistics. The trainer reduces counts over ALL ranks/micro-batches before any backward.

    log_num/log_den are detached mean statistics; log_sum holds additive counters (never average target/sample counts across ranks).
    No collective, optimizer update, or EMA is performed here.
    """
    num: dict[str, torch.Tensor] = field(default_factory=dict)
    den: dict[str, torch.Tensor] = field(default_factory=dict)
    weight: dict[str, float] = field(default_factory=dict)
    log_num: dict[str, torch.Tensor] = field(default_factory=dict)
    log_den: dict[str, torch.Tensor] = field(default_factory=dict)
    log_sum: dict[str, torch.Tensor] = field(default_factory=dict)

    def backward_loss(self, global_den: dict[str, torch.Tensor] | None = None, *, world_size: int = 1) -> torch.Tensor:
        """R * local numerator / global count: compensates DDP's gradient average, with no extra micro-batch divisor."""
        den = self.den if global_den is None else global_den
        if not self.num or set(den) != set(self.num) or set(self.weight) != set(self.num) or world_size < 1:
            raise ValueError("inconsistent component keys or world_size")
        return sum(self.weight[k] * n * world_size / den[k].clamp(min=1) for k, n in self.num.items())


# DDP finds tensors in dataclasses, but its unused-output sink also FLATTENS the output via pytree. Register the container so its
# differentiable numerators pass through that sink rather than treating LossTerms as an opaque, non-tensor leaf.
register_pytree_node(LossTerms,
                    lambda t: ([t.num, t.den, t.weight, t.log_num, t.log_den, t.log_sum], None),
                    lambda values, context: LossTerms(*values))


def _sample_index(q_ch: torch.Tensor, counts: list[int]) -> torch.Tensor:
    bounds = torch.as_tensor(np.cumsum([0, *counts]), device=q_ch.device)
    return torch.searchsorted(bounds, q_ch, right=True) - 1


def _rvq_selections(hidden: torch.Tensor, inp: ModelInputs, rvq_on: torch.Tensor, patch_ms: float):
    per = int(round(RVQ_TOKEN_MS / patch_ms))
    if per < 1:
        raise ValueError("RVQ token shorter than patch")
    c_tot, p = hidden.shape
    w = p // per
    win = torch.zeros(c_tot, w, dtype=torch.bool, device=hidden.device)
    bounds = np.cumsum([0, *inp.counts])
    for a, b in zip(bounds[:-1], bounds[1:]):
        ok = inp.qc_valid[a:b]
        slab = (hidden[a:b][ok][:, :w * per].all(dim=0).reshape(w, per).all(dim=-1) if bool(ok.any())
                else torch.zeros(w, dtype=torch.bool, device=hidden.device))
        win[a:b] = slab[None, :] & ok[:, None]
    win &= rvq_on[:, None]
    chan = hidden[:, :w * per].reshape(c_tot, w, per).all(dim=-1) & inp.qc_valid[:, None] & rvq_on[:, None] & ~win
    return per, win, chan


def _valid_count(per_query: torch.Tensor, sample_index: torch.Tensor | None, n_samples: int) -> torch.Tensor:
    counts = per_query.to(torch.int64)
    if sample_index is None:
        return counts.sum().detach()
    return (counts.new_zeros(n_samples).index_add(0, sample_index, counts) > 0).sum().detach()


def _keep_valid(anchor_targets, inp: ModelInputs, target_active: torch.Tensor) -> torch.Tensor:
    if target_active.shape != (32,) or target_active.dtype != torch.bool:
        raise ValueError("keep_target_active must be the calibrated boolean 32-vector")
    _, valid = global_spectral_targets(anchor_targets, inp.counts, inp.qc_valid)
    return torch.as_tensor(valid, device=inp.qc_valid.device) & target_active.to(inp.qc_valid.device)


def loss_counts(inp: ModelInputs, visible: torch.Tensor, cfg: JEPAConfig,
                anchor_targets: list[AnchorTargets] | None = None, rvq_on: torch.Tensor | None = None,
                rvq_codes: Callable | None = None, *, model_cfg: FMConfig,
                keep_target_active: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Detached int64 counts from geometry/targets, WITHOUT encode/decode or RNG consumption.

    model_cfg is mandatory: JEPAConfig has no patch duration, latent width, or optional-head geometry. For keep, supply the calibrated
    target_active buffer, not a guess based only on observed targets. Same active components on all ranks are the trainer's responsibility.
    rvq_codes is a pure lookup into frozen codes (as in the current loader); the callback must not draw randomness.
    """
    if len(inp.signals) != len(inp.counts) or len(inp.fs) != len(inp.counts) or not inp.counts:
        raise ValueError("inconsistent sample geometry")
    patch_s = model_cfg.patch_ms / 1000.0
    if not math.isfinite(patch_s) or patch_s <= 0 or any(not math.isfinite(f) or f <= 0 for f in inp.fs) or \
            any(c <= 0 or x.shape[0] != c for x, c in zip(inp.signals, inp.counts)):
        raise ValueError("invalid sample/patch geometry")
    if inp.qc_valid.shape != (sum(inp.counts),) or inp.qc_valid.dtype != torch.bool:
        raise ValueError("QC geometry/dtype does not match packed channels")
    lengths = [int(math.floor(x.shape[-1] / f / patch_s + 1e-9)) for x, f in zip(inp.signals, inp.fs)]
    if min(lengths) < 1 or visible.shape != (sum(inp.counts), max(lengths)) or visible.dtype != torch.bool:
        raise ValueError("visible shape/dtype does not match patch geometry")
    n_ch = torch.repeat_interleave(torch.as_tensor(lengths, device=visible.device), torch.as_tensor(inp.counts, device=visible.device))
    patch_valid = torch.arange(max(lengths), device=visible.device)[None, :] < n_ch[:, None]
    hidden = ~visible & patch_valid & inp.qc_valid[:, None]
    ch, qt, _ = hidden_queries(hidden, inp.counts)
    sq = _sample_index(ch, inp.counts) if cfg.loss_per_window else None
    ns = len(inp.counts)
    out = {"jepa": _valid_count(torch.ones_like(ch), sq, ns) if cfg.loss_per_window else ch.new_tensor(ch.numel() * model_cfg.dim)}
    if anchor_targets is not None:
        tg = gather_anchor_targets(anchor_targets, inp.counts, ch, qt)
        masks = {"log_rms": tg["rms_valid"], "log_env": tg["env_valid"],
                 "band_shape": (tg["spec_valid"][:, None] & tg["band_available"]).sum(-1)}
        if model_cfg.multiscale_anchor:
            masks.update({name: (tg[name + "_valid"][:, None] & tg[name + "_available"]).sum(-1)
                          for name, _, _ in MS_SCALES if name in tg})
        out.update({name: _valid_count(mask, sq, ns) for name, mask in masks.items()})
    if model_cfg.rvq_codes and rvq_on is not None and rvq_codes is not None:
        _, slab, channel = _rvq_selections(hidden, inp, rvq_on, model_cfg.patch_ms)
        wc, ww = torch.nonzero(slab if cfg.rvq_targets == "slab" else channel, as_tuple=True)
        valid = rvq_codes(wc, ww) >= 0 if wc.numel() else wc.new_zeros(0, dtype=torch.bool)
        sw = _sample_index(wc, inp.counts) if cfg.loss_per_window else None
        out["rvq"] = _valid_count(valid, sw, ns)
    if cfg.keep_weight > 0:
        if anchor_targets is None or keep_target_active is None or not model_cfg.keep_readout:
            raise ValueError("keep counts require targets, head geometry and calibrated target_active")
        out["keep"] = _keep_valid(anchor_targets, inp, keep_target_active).any(1).sum().detach()
    if cfg.branch_weight > 0:  # stessi target, maschere e calibrazione della keep: stesso conteggio, contato come componente a se'
        if anchor_targets is None or keep_target_active is None or not model_cfg.spectral_branches:
            raise ValueError("branch counts require targets, spectral branches and calibrated target_active")
        out["keep_branches"] = _keep_valid(anchor_targets, inp, keep_target_active).any(1).sum().detach()
    return out


def jepa_loss_terms(student: WearUsFM, teacher: WearUsFM, inp: ModelInputs, visible: torch.Tensor, cfg: JEPAConfig,
                    anchor_targets: list[AnchorTargets] | None = None, rvq_on: torch.Tensor | None = None,
                    rvq_codes: Callable | None = None, generator: torch.Generator | None = None,
                    kinds: torch.Tensor | None = None) -> LossTerms:
    terms = LossTerms()
    _jepa_compute(student, teacher, inp, visible, cfg, anchor_targets, rvq_on, rvq_codes, generator, kinds, terms)
    return terms


def jepa_losses(student: WearUsFM, teacher: WearUsFM, inp: ModelInputs, visible: torch.Tensor, cfg: JEPAConfig,
                anchor_targets: list[AnchorTargets] | None = None, rvq_on: torch.Tensor | None = None,
                rvq_codes: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                generator: torch.Generator | None = None, kinds: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Backward-compatible single-process API: historical means, keys and RNG ordering unchanged."""
    return _jepa_compute(student, teacher, inp, visible, cfg, anchor_targets, rvq_on, rvq_codes, generator, kinds)


def _jepa_compute(student, teacher, inp, visible, cfg, anchor_targets=None, rvq_on=None, rvq_codes=None, generator=None, kinds=None,
                  terms: LossTerms | None = None):
    """visible: (C_tot, P) True = visto dallo studente. rvq_on: (C_tot,) canali con l'ancora RVQ accesa; rvq_codes(canale, finestra) -> codici
    del tokenizer congelato (livello 0 del ramo 0) per quelle finestre. kinds: (C_tot, P) tipo di maschera (`training.masking`): se c'e', la perdita
    JEPA si registra anche per tipo di maschera e per topologia (`jepa_<tipo>`, `jepa_topo_<topologia>`; solo per il log, v10 §6.4)."""
    enc = student.encode(inp, visible, generator)
    hidden = ~enc.visible & enc.patch_valid & inp.qc_valid[:, None]
    q_ch, q_t, q_off = hidden_queries(hidden, inp.counts)
    if q_ch.numel() == 0 and terms is None:
        raise ValueError("nessun token nascosto: la maschera non chiede nessuna predizione")
    h = student.decode(enc, q_ch, q_t, q_off)
    with torch.no_grad():
        enc_t = teacher.encode(inp, None)
        tgt = enc_t.local[q_ch, q_t] if cfg.target == "a" else teacher.decode(enc_t, q_ch, q_t, q_off)
        tgt = F.layer_norm(tgt, tgt.shape[-1:])
    s_q = None
    if cfg.loss_per_window:
        s_q = torch.searchsorted(q_off, torch.arange(q_ch.numel(), device=q_ch.device), right=True) - 1  # campione di ogni query
    n_s = len(inp.counts)
    err_q = (h - tgt).pow(2).mean(dim=-1)
    losses = {"jepa": (masked_mean(err_q, torch.ones_like(err_q), s_q, n_s) if cfg.loss_per_window else F.mse_loss(h, tgt))
              if q_ch.numel() else h.sum() * 0.0}
    if terms is not None:
        terms.num["jepa"], terms.den["jepa"] = (masked_sum_count(err_q, torch.ones_like(err_q), s_q, n_s) if cfg.loss_per_window
                                              else (F.mse_loss(h, tgt, reduction="sum"), q_ch.new_tensor(h.numel())))
        terms.weight["jepa"] = 1.0
    if kinds is not None:
        with torch.no_grad():
            err = (h.float() - tgt.float()).pow(2).mean(dim=-1)  # (Q,)
            kq = kinds[q_ch, q_t]
            for k, name in KIND_NAMES.items():
                sel = kq == k
                if k != VISIBLE and bool(sel.any()):
                    losses[f"jepa_{name}"] = err[sel].mean()
            tq = inp.codes["topology"][q_ch]
            for t, name in enumerate(TOPOLOGIES):
                sel = tq == t
                if bool(sel.any()):
                    losses[f"jepa_topo_{name}"] = err[sel].mean()
            if terms is not None:
                selections = {f"jepa_{name}": kq == k for k, name in KIND_NAMES.items() if k != VISIBLE}
                selections.update({f"jepa_topo_{name}": tq == t for t, name in enumerate(TOPOLOGIES)})
                for name, sel in selections.items():
                    terms.log_num[name], terms.log_den[name] = err[sel].sum().detach(), sel.sum().detach()
    anchor_total = h.new_zeros(())
    if anchor_targets is not None:
        pred, target = student.anchor_heads(h), gather_anchor_targets(anchor_targets, inp.counts, q_ch, q_t)
        for k, v in anchor_losses(pred, target, s_q, n_s).items():
            losses[k] = v
            anchor_total = anchor_total + v
        if terms is not None:
            for k, (num, den) in anchor_loss_terms(pred, target, s_q, n_s).items():
                terms.num[k], terms.den[k], terms.weight[k] = num, den, cfg.anchor_weight
    if student.rvq_head is not None and rvq_on is not None and rvq_codes is not None:
        per, win, chan = _rvq_selections(hidden, inp, rvq_on, student.cfg.patch_ms)
        c_tot, p = hidden.shape
        w = p // per
        dev = hidden.device
        bounds = np.cumsum([0, *inp.counts])
        index = torch.full((c_tot, p), -1, dtype=torch.long, device=dev)
        index[q_ch, q_t] = torch.arange(q_ch.numel(), device=dev)
        bounds_t = torch.as_tensor(bounds, device=dev)

        def rvq_on_windows(sel: torch.Tensor, name: str):
            wc, ww = torch.nonzero(sel, as_tuple=True)
            if not wc.numel():
                if terms is not None:
                    zero, count = h.sum() * 0.0, q_ch.new_zeros(())
                    if name == "rvq":
                        terms.num[name], terms.den[name], terms.weight[name] = zero, count, cfg.anchor_weight
                    else:
                        terms.log_num[name], terms.log_den[name] = zero.detach(), count
                    terms.log_num[name + "_acc"], terms.log_den[name + "_acc"] = zero.detach(), count
                    terms.log_sum[name + "_targets"] = count
                return h.sum() * 0.0, h.new_zeros(()), h.new_zeros(())
            rows = index[wc[:, None], ww[:, None] * per + torch.arange(per, device=dev)[None, :]]  # (N, 8), tutte query nascoste
            codes = rvq_codes(wc, ww)
            logits = student.rvq_head(h[rows])
            s_w = (torch.searchsorted(bounds_t, wc, right=True) - 1) if cfg.loss_per_window else None
            ok = codes >= 0
            acc = (logits.argmax(dim=-1) == codes)[ok].float().mean() if bool(ok.any()) else h.new_zeros(())
            if terms is not None:
                num, den = rvq_loss_terms(logits, codes, s_w, n_s)
                if name == "rvq":
                    terms.num[name], terms.den[name], terms.weight[name] = num, den, cfg.anchor_weight
                else:
                    terms.log_num[name], terms.log_den[name] = num.detach(), den
                terms.log_num[name + "_acc"] = (logits.argmax(-1)[ok] == codes[ok]).sum().detach()
                terms.log_den[name + "_acc"] = ok.sum().detach()
                terms.log_sum[name + "_targets"] = ok.sum().detach()
            return rvq_loss(logits, codes, s_w, n_s), ok.sum().float(), acc.detach()

        main_sel, other_sel = (win, chan) if cfg.rvq_targets == "slab" else (chan, win)
        losses["rvq"], losses["rvq_targets"], losses["rvq_acc"] = rvq_on_windows(main_sel, "rvq")
        with torch.no_grad():  # l'altra scelta, solo per il log
            other = "rvq_channel" if cfg.rvq_targets == "slab" else "rvq_slab"
            o_loss, o_n, o_acc = rvq_on_windows(other_sel, other)
            losses[other], losses[other + "_targets"], losses[other + "_acc"] = o_loss.detach(), o_n, o_acc
        anchor_total = anchor_total + losses["rvq"]
    losses["total"] = losses["jepa"] + cfg.anchor_weight * anchor_total
    if cfg.keep_weight > 0 or cfg.branch_weight > 0:
        if anchor_targets is None:
            raise ValueError("keep loss requires multiscale targets")
        if cfg.branch_weight > 0:  # rami spettrali: keep raw dello Step 1 e termine dei rami dallo stesso encode (spectral_branches)
            from wearusfm.model.spectral_branches import readout_objectives
            parts = readout_objectives(student, inp, anchor_targets, keep=cfg.keep_weight > 0, terms=terms is not None)
        else:
            parts = {"keep": keep_objective_terms(student, inp, anchor_targets) if terms is not None
                     else keep_objective(student, inp, anchor_targets)}
        for name, weight in (("keep", cfg.keep_weight), ("keep_branches", cfg.branch_weight)):
            if name not in parts:
                continue
            if terms is not None:
                losses[name], metrics, num, den, log_num, log_den = parts[name]
                terms.num[name], terms.den[name], terms.weight[name] = num, den, weight
                terms.log_sum[name + "_valid_samples"] = den
                terms.log_num.update(log_num)
                terms.log_den.update(log_den)
            else:
                losses[name], metrics = parts[name]
            losses.update(metrics)
            losses["total"] = losses["total"] + weight * losses[name]
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
