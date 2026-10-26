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

**Bracci dell'obiettivo** (W1-OBJ, accordo del 15/10/2026; Simone 16/10): totale = `jepa_weight` x JEPA + `anchor_weight` x ancore (+ keep e
rami). `jepa_weight` = 1 e' l'obiettivo di sempre (stesso grafo, bit per bit); `anchor_weight` = 0 e' il JEPA puro; `jepa_weight` = 0 sono le
ancore attuali da sole: stesso masking, stesso decoder dello studente (che le ancore allenano), teacher EMA aggiornato come sempre. Una
perdita a peso 0 si calcola e si registra lo stesso, senza contare nel gradiente.

**Target per regione** (R2 del pilot degli obiettivi, `JEPAConfig.region_target`, 24/10/2026): il target JEPA e' definito sulla **regione
nascosta**, cioe' la maschera estratta dal generatore (tubo, slab o maschera spaziale; `PretrainBatch.region`): il target e' la LayerNorm della
media del teacher (grezzo, stesso punto di lettura di D11) sui token della regione, la predizione e' la media delle uscite del decoder dello
studente sugli stessi token; l'errore quadratico medio della regione e' attribuito a ciascuno dei suoi token e ridotto come per token (per
finestra o per elemento), cosi' le quote di D10 restano quelle di sempre e cambia solo l'unita' del target. Le ancore restano per token. Le
medie si fanno in float32 (`index_add`: su GPU l'ordine delle somme non e' deterministico). Default False: stesso grafo di sempre.
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
    # peso della perdita JEPA (16/10/2026, braccio ancore-only di W1-OBJ): 1 = l'obiettivo di sempre, 0 = sole ancore
    jepa_weight: float = 1.0
    # target JEPA per regione nascosta invece che per token (R2 del pilot degli obiettivi, 24/10/2026): servono le regioni delle maschere
    region_target: bool = False
    # peso del termine di posa (R3, seconda ondata del pilot degli obiettivi, firmato il 26/10/2026: 1,0 sul target standardizzato); 0 = nessun
    # termine (il default). Richiede `FMConfig.pose_head` e i target di posa nel batch (`PretrainBatch.pose`)
    pose_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("jepa_weight", "anchor_weight", "pose_weight"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.jepa_weight == 0 and self.anchor_weight == 0 and self.keep_weight == 0 and self.branch_weight == 0 and self.pose_weight == 0:
            raise ValueError("jepa_weight, anchor_weight, keep_weight e branch_weight tutti a zero: nessuna perdita da allenare")
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


def core_hidden(hidden: torch.Tensor, visible: torch.Tensor, radius: int) -> torch.Tensor:
    """Con una finestra del kernel piu' lunga della patch (`FMConfig.window_ms`, R5), un token nascosto al bordo di una regione contiene campioni
    visibili: predirlo e' banale. Restano «nascosti per la perdita» solo i token con nessuna patch visibile entro `radius` patch per lato
    (`FMConfig.core_radius`); le patch oltre il segnale o di padding contano come nascoste (il front-end vi legge zeri). radius 0: invariato."""
    if radius <= 0:
        return hidden
    vis = visible.to(torch.float32)[:, None, :]  # (C_tot, 1, P)
    near_visible = F.max_pool1d(vis, kernel_size=2 * radius + 1, stride=1, padding=radius)[:, 0, :] > 0
    return hidden & ~near_visible


REGION_STRIDE = 1 << 16  # le regioni sono int16 >= 0 per campione: chiave unica (campione, regione) = campione * stride + regione


def region_keys(regions: torch.Tensor, hidden_shape: tuple, q_ch: torch.Tensor, q_t: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    """Chiave (campione, regione) di ogni token nascosto: `regions` (C_tot, P) dal generatore (`generate_mask_regions`), 0 = nessuna regione."""
    if regions is None:
        raise ValueError("target per regione: servono le regioni delle maschere (PretrainBatch.region, generate_mask_regions)")
    if tuple(regions.shape) != tuple(hidden_shape):
        raise ValueError(f"regioni {tuple(regions.shape)} e maschera {tuple(hidden_shape)} di forma diversa")
    r_q = regions[q_ch, q_t].long()
    if q_ch.numel() and (bool((r_q <= 0).any()) or bool((r_q >= REGION_STRIDE).any())):
        raise ValueError("target per regione: un token nascosto senza regione (o regione oltre lo stride)")
    return sample * REGION_STRIDE + r_q


def region_means(h: torch.Tensor, raw: torch.Tensor, regions: torch.Tensor, hidden_shape: tuple, q_ch: torch.Tensor, q_t: torch.Tensor,
                 q_off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """R2: per ogni regione nascosta di ogni campione la media delle uscite dello studente `h` (Q, d) e la LayerNorm della media del teacher
    grezzo `raw` (Q, d) sui suoi token, in float32. Ritorna (h_r, tgt_r, inv, s_r): medie (R, d), regione di ogni query, campione di ogni
    regione."""
    sample = torch.searchsorted(q_off, torch.arange(q_ch.numel(), device=q_ch.device), right=True) - 1
    uniq, inv = torch.unique(region_keys(regions, hidden_shape, q_ch, q_t, sample), return_inverse=True)
    n_r, d = uniq.numel(), h.shape[-1]
    acc = torch.promote_types(h.dtype, torch.float32)  # almeno float32 (bf16 dell'autocast); float64 resta float64 (test d'equivalenza)
    cnt = torch.zeros(n_r, device=h.device, dtype=acc).index_add(0, inv, torch.ones(q_ch.numel(), device=h.device, dtype=acc))
    h_r = torch.zeros(n_r, d, device=h.device, dtype=acc).index_add(0, inv, h.to(acc)) / cnt[:, None]
    with torch.no_grad():
        t_r = torch.zeros(n_r, d, device=h.device, dtype=acc).index_add(0, inv, raw.to(acc)) / cnt[:, None]
        t_r = F.layer_norm(t_r, t_r.shape[-1:])
    return h_r, t_r, inv, torch.div(uniq, REGION_STRIDE, rounding_mode="floor")


def pose_time_valid(patch_valid: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """(S, P) patch che esistono per campione (non riempimento), dalla geometria (C_tot, P) dei canali impacchettati. Il termine di posa (R3)
    vale su TUTTE le patch esistenti con target valido, visibili, nascoste e anche senza nessun canale visibile all'istante (gli slab): il
    latente dell'istante esiste sempre (latenti appresi + contesto del backbone) e il §2 (ii) del protocollo firmato chiede tutte le patch
    (rilievo 2 del critico del 26/10: con gli slab il 12,5% delle patch di emg2pose non ha canali visibili)."""
    bounds = np.cumsum([0, *counts])
    return torch.stack([patch_valid[a:b].any(dim=0) for a, b in zip(bounds[:-1], bounds[1:])])


def pose_terms(pred: torch.Tensor, pose: tuple | None, time_valid: torch.Tensor, per_window: bool) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Termine di posa da una predizione (S, P, J) e dai target (target (S, P, J), maschera (S, P, J)[, deviazioni (S, J)]): errore
    quadratico sugli elementi validi (maschera e patch esistente). Per finestra: media dentro ogni finestra che ha almeno un elemento valido,
    poi media fra quelle finestre (come `masked_mean` con `sample_index`); per token: media sugli elementi. Ritorna (perdita, numeratore
    additivo, conteggio, maschera di validita' usata). Senza target (micro-batch senza finestre di posa) la perdita e' `pred.sum() * 0`: la
    testa resta nel grafo con gradiente nullo e il conteggio e' 0."""
    if pose is None:
        zero = pred.sum() * 0.0
        return zero, zero, pred.new_zeros((), dtype=torch.int64), torch.zeros_like(pred, dtype=torch.bool)
    target, mask = pose[0], pose[1]
    if target.shape != pred.shape or mask.shape != pred.shape:
        raise ValueError(f"posa: target {tuple(target.shape)} e maschera {tuple(mask.shape)} contro predizione {tuple(pred.shape)}")
    valid = mask & time_valid[..., None]
    err = (pred.float() - target.float()).pow(2) * valid
    if per_window:
        s_num, s_den = err.sum(dim=(1, 2)), valid.sum(dim=(1, 2))
        ok = s_den > 0
        if not bool(ok.any()):
            zero = pred.sum() * 0.0
            return zero, zero, ok.sum().detach(), valid
        means = s_num[ok] / s_den[ok]
        return means.mean(), means.sum(), ok.sum().detach(), valid
    n = valid.sum()
    if int(n) == 0:
        zero = pred.sum() * 0.0
        return zero, zero, n.detach(), valid
    return err.sum() / n, err.sum(), n.detach(), valid


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
                keep_target_active: torch.Tensor | None = None, regions: torch.Tensor | None = None,
                pose_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Detached int64 counts from geometry/targets, WITHOUT encode/decode or RNG consumption.
    Col termine di posa (`cfg.pose_weight` > 0, R3): `pose_valid` (S, P, J) e' la maschera dei target del micro-batch (None = nessuna finestra
    di posa); il conteggio e' il numero di finestre con almeno un elemento valido su un istante valido (per finestra) o degli elementi (per
    token), lo stesso di `pose_terms`.

    model_cfg is mandatory: JEPAConfig has no patch duration, latent width, or optional-head geometry. For keep, supply the calibrated
    target_active buffer, not a guess based only on observed targets. Same active components on all ranks are the trainer's responsibility.
    rvq_codes is a pure lookup into frozen codes (as in the current loader); the callback must not draw randomness.
    Con una finestra del kernel piu' lunga della patch (`model_cfg.core_radius` > 0, R5) contano solo i token interamente nascosti, come in
    `_jepa_compute` (24/10/2026: prima mancava e il contratto delle loss avrebbe fermato il run). Col target per regione (`cfg.region_target`)
    i conteggi non cambiano (ogni token porta l'errore della sua regione): `regions` e' accettato per simmetria con `jepa_loss_terms` e non
    serve.
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
    if model_cfg.core_radius:  # come in _jepa_compute: `enc.visible` e' visible & patch_valid & qc_valid
        hidden = core_hidden(hidden, visible & patch_valid & inp.qc_valid[:, None], model_cfg.core_radius)
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
    if cfg.pose_weight > 0:
        if not model_cfg.pose_head:
            raise ValueError("pose counts require the pose head (FMConfig.pose_head)")
        if pose_valid is None:
            out["pose"] = visible.new_zeros((), dtype=torch.int64)
        else:
            tv = pose_time_valid(patch_valid, inp.counts)  # patch esistenti: come `enc.patch_valid` in `_jepa_compute`
            if pose_valid.shape[:2] != tv.shape:
                raise ValueError(f"pose_valid {tuple(pose_valid.shape)} contro la geometria {tuple(tv.shape)}")
            valid = pose_valid & tv[..., None]
            out["pose"] = (valid.any(dim=(1, 2)).sum() if cfg.loss_per_window else valid.sum()).detach()
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
                    kinds: torch.Tensor | None = None, regions: torch.Tensor | None = None,
                    pose: tuple | None = None, pose_std: torch.Tensor | None = None, pose_std_log: bool = False) -> LossTerms:
    """`pose`: (target, maschera[, deviazioni per finestra]) del micro-batch (R3). `pose_std_log`: logga l'errore assoluto di posa per giunto,
    standardizzato e nell'unita' degli angoli (con le deviazioni per finestra del batch, o `pose_std` (J,) se date): solo il pannello fisso."""
    terms = LossTerms()
    _jepa_compute(student, teacher, inp, visible, cfg, anchor_targets, rvq_on, rvq_codes, generator, kinds, terms, regions, pose=pose,
                  pose_std=pose_std, pose_std_log=pose_std_log)
    return terms


def jepa_losses(student: WearUsFM, teacher: WearUsFM, inp: ModelInputs, visible: torch.Tensor, cfg: JEPAConfig,
                anchor_targets: list[AnchorTargets] | None = None, rvq_on: torch.Tensor | None = None,
                rvq_codes: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
                generator: torch.Generator | None = None, kinds: torch.Tensor | None = None,
                regions: torch.Tensor | None = None, pose: tuple | None = None) -> dict[str, torch.Tensor]:
    """Backward-compatible single-process API: historical means, keys and RNG ordering unchanged."""
    return _jepa_compute(student, teacher, inp, visible, cfg, anchor_targets, rvq_on, rvq_codes, generator, kinds, regions=regions, pose=pose)


def _jepa_compute(student, teacher, inp, visible, cfg, anchor_targets=None, rvq_on=None, rvq_codes=None, generator=None, kinds=None,
                  terms: LossTerms | None = None, regions: torch.Tensor | None = None, pose=None, pose_std=None, pose_std_log=False):
    """visible: (C_tot, P) True = visto dallo studente. rvq_on: (C_tot,) canali con l'ancora RVQ accesa; rvq_codes(canale, finestra) -> codici
    del tokenizer congelato (livello 0 del ramo 0) per quelle finestre. kinds: (C_tot, P) tipo di maschera (`training.masking`): se c'e', la perdita
    JEPA si registra anche per tipo di maschera e per topologia (`jepa_<tipo>`, `jepa_topo_<topologia>`; solo per il log, v10 §6.4).
    regions: (C_tot, P) regione di ogni token nascosto (`generate_mask_regions`), richiesta da `cfg.region_target` e ignorata altrimenti."""
    enc = student.encode(inp, visible, generator)
    hidden = ~enc.visible & enc.patch_valid & inp.qc_valid[:, None]
    radius = student.cfg.core_radius
    if radius:  # finestra piu' lunga della patch (R5): solo i token interamente nascosti, per JEPA, ancore e RVQ
        all_hidden = hidden
        hidden = core_hidden(hidden, enc.visible, radius)
    q_ch, q_t, q_off = hidden_queries(hidden, inp.counts)
    if q_ch.numel() == 0 and terms is None:
        raise ValueError("nessun token nascosto: la maschera non chiede nessuna predizione")
    h = student.decode(enc, q_ch, q_t, q_off)
    with torch.no_grad():
        enc_t = teacher.encode(inp, None)
        raw = enc_t.local[q_ch, q_t] if cfg.target == "a" else teacher.decode(enc_t, q_ch, q_t, q_off)
        tgt = F.layer_norm(raw, raw.shape[-1:])
    s_q = None
    if cfg.loss_per_window:
        s_q = torch.searchsorted(q_off, torch.arange(q_ch.numel(), device=q_ch.device), right=True) - 1  # campione di ogni query
    n_s = len(inp.counts)
    if cfg.region_target:  # R2: target e predizione sono medie per regione (float32); ogni token porta l'errore della sua regione, cosi' le
        # quote di D10 restano quelle per token (critico del 24/10: a regioni pesate alla pari il 70% della perdita finiva sulle maschere corte)
        h_r, tgt_r, inv, _ = region_means(h, raw, regions, hidden.shape, q_ch, q_t, q_off)
        err_r = (h_r - tgt_r).pow(2).mean(dim=-1)  # (R,)
        err_q = err_r[inv]  # (Q,), col gradiente: stesse riduzioni del percorso per token
        d = h.shape[-1]
        losses = {"jepa": (masked_mean(err_q, torch.ones_like(err_q), s_q, n_s) if cfg.loss_per_window else err_q.mean())
                  if q_ch.numel() else h.sum() * 0.0}
        losses["jepa_regions"] = err_r.new_tensor(float(err_r.numel()))
        losses["jepa_tokens_per_region"] = err_r.new_tensor(q_ch.numel() / max(1, err_r.numel()))
        if terms is not None:
            terms.num["jepa"], terms.den["jepa"] = (masked_sum_count(err_q, torch.ones_like(err_q), s_q, n_s) if cfg.loss_per_window
                                                  else (err_q.sum() * d, q_ch.new_tensor(q_ch.numel() * d)))
            terms.weight["jepa"] = cfg.jepa_weight
            terms.log_sum["jepa_regions"] = q_ch.new_tensor(err_r.numel())
            terms.log_sum["jepa_hidden_tokens"] = q_ch.new_tensor(q_ch.numel())  # token per regione = jepa_hidden_tokens / jepa_regions
        err_log = err_q.detach()
    else:
        err_q = (h - tgt).pow(2).mean(dim=-1)
        losses = {"jepa": (masked_mean(err_q, torch.ones_like(err_q), s_q, n_s) if cfg.loss_per_window else F.mse_loss(h, tgt))
                  if q_ch.numel() else h.sum() * 0.0}
        if terms is not None:
            terms.num["jepa"], terms.den["jepa"] = (masked_sum_count(err_q, torch.ones_like(err_q), s_q, n_s) if cfg.loss_per_window
                                                  else (F.mse_loss(h, tgt, reduction="sum"), q_ch.new_tensor(h.numel())))
            terms.weight["jepa"] = cfg.jepa_weight
        err_log = None
    if radius:  # per il log: quanti dei token nascosti sono interamente nascosti
        losses["core_hidden_fraction"] = (hidden.sum().float() / all_hidden.sum().clamp(min=1).float()).detach()
    if kinds is not None:
        with torch.no_grad():
            err = (h.float() - tgt.float()).pow(2).mean(dim=-1) if err_log is None else err_log.float()  # (Q,)
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
    if cfg.jepa_weight == 0 and anchor_targets is None and cfg.keep_weight == 0 and cfg.branch_weight == 0 and cfg.pose_weight == 0:
        raise ValueError("jepa_weight = 0 senza target delle ancore: nessuna perdita da allenare")
    # col peso 1 la JEPA entra com'era (stesso grafo, bit per bit); con un altro peso si moltiplica
    losses["total"] = (losses["jepa"] if cfg.jepa_weight == 1.0 else cfg.jepa_weight * losses["jepa"]) + cfg.anchor_weight * anchor_total
    if cfg.pose_weight > 0:  # R3: la testa di posa sulla media dei latenti del backbone per istante, su tutte le finestre del micro-batch
        if student.pose_head is None:
            raise ValueError("pose_weight > 0 senza la testa di posa (FMConfig.pose_head)")
        from wearusfm.model.pose_head import latent_mean
        pred = student.pose_head(latent_mean(enc.z))  # (S, P, 20); calcolata sempre: nessun parametro fuori dal grafo DDP
        p_loss, p_num, p_den, p_valid = pose_terms(pred, pose, pose_time_valid(enc.patch_valid, inp.counts), cfg.loss_per_window)
        losses["pose"] = p_loss
        losses["pose_windows"] = p_den.to(p_loss.dtype) if cfg.loss_per_window else p_valid.any(dim=(1, 2)).sum().to(p_loss.dtype)
        losses["total"] = losses["total"] + cfg.pose_weight * p_loss
        if terms is not None:
            terms.num["pose"], terms.den["pose"], terms.weight["pose"] = p_num, p_den, cfg.pose_weight
            terms.log_sum["pose_windows"] = p_valid.any(dim=(1, 2)).sum().detach()
            std = pose_std if pose_std is not None else (pose[2] if pose is not None and len(pose) > 2 else None)
            if pose is not None and std is not None and pose_std_log:  # pannello fisso: errore assoluto per giunto, standardizzato e in radianti
                abs_err = ((pred.float() - pose[0].float()).abs() * p_valid).detach()  # (S, P, J)
                std = std.to(abs_err.device, abs_err.dtype)
                std = std[:, None, :] if std.dim() == 2 else std.view(1, 1, -1)  # per finestra (lato della sessione) o unica
                n_j, e_j, r_j = p_valid.sum(dim=(0, 1)), abs_err.sum(dim=(0, 1)), (abs_err * std).sum(dim=(0, 1))
                terms.log_num["pose_mae_std"], terms.log_den["pose_mae_std"] = e_j.sum(), n_j.sum()
                terms.log_num["pose_mae"], terms.log_den["pose_mae"] = r_j.sum(), n_j.sum()
                for j in range(e_j.shape[0]):
                    terms.log_num[f"pose_mae_j{j:02d}"], terms.log_den[f"pose_mae_j{j:02d}"] = r_j[j], n_j[j]
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
