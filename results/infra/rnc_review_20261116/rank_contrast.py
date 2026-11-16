"""Perdita contrastiva a ranghi sulla posa (Rank-N-Contrast, RnC: Zha et al., NeurIPS 2023), sul readout per istante.

PROPOSTA NON FIRMATA (chat dell'11-12/11/2026): non sta in v10 ne' in `docs/pilot_posa_v1.md`. Spenta di default
(`JEPAConfig.rnc_weight` = 0): il modulo si carica solo col termine acceso.

**Cosa chiede.** Un'ancora i e' un istante (campione, patch) con la posa valida: h_i = media sui K latenti del
backbone (`pose_head.latent_mean`: il vettore che legge la testa di posa R3, dallo stesso encode mascherato dello
studente), y_i = angoli standardizzati. Per ogni coppia positiva (i, j):

    l_ij = -ln( exp(s_ij / tau) / sum_{k in S_ij} exp(s_ik / tau) ),   S_ij = {k != i : d(y_i, y_k) >= d(y_i, y_j)}

cioe' j deve somigliare a i piu' di ogni istante la cui posa dista da i almeno quanto quella di j.

**Supporto: quello del termine di posa** (protocollo R3, §2 (ii)): tutte le patch esistenti col target valido,
visibili e nascoste (`jepa.pose_time_valid`), una ogni `stride` (fase stride // 2).

**Contratto delle loss.** Un'ancora conta se ha almeno DUE coppie positive: allora almeno un S_ij ha due candidati
e il valore a similarita' uniformi e' > 0. La regola dipende solo da maschere e finestre, non dai target ne' dagli
embedding: `rank_contrast_count` da' a `jepa.loss_counts` il conteggio del forward, senza encode. I candidati sono
quelli del micro-batch del processo: nessuna operazione collettiva.

**Scelte da confermare:**
- s = coseno fra i readout standardizzati sulle d coordinate (LayerNorm senza parametri, come
  `branch_masking.target_tokens`). Nessun parametro, nessun numero casuale.
- d = RMS delle differenze sugli angoli validi in entrambi (deviazioni standard per giunto); `resolution` > 0
  raggruppa le distanze a passi di quella ampiezza (sotto il rumore del motion capture l'ordine non e' informazione).
- positivi: "cross" = solo fra finestre diverse del micro-batch (le stesse restano fra i candidati); "all" = tutti.
  "cross" non vuol dire sessioni diverse: `_jepa_compute` non riceve l'identita' della sessione.
- ogni ancora e' divisa per il suo valore a similarita' uniformi, sum_j ln|S_ij| (come l'entropia incrociata RVQ
  divisa per ln(8192)): vale 1 a similarita' uniformi.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from wearusfm.model.anchors import masked_sum_count

POSITIVES = ("cross", "all")
_MASKED_LOGIT = -1e4  # finito: con -inf il backward di logcumsumexp da' NaN; exp(-1e4) e' zero esatto
_CHUNK = 256  # ancore per blocco nelle distanze fra pose


def anchor_support(valid: torch.Tensor, stride: int, positives: str):
    """valid (S, P, J) bool = maschera dei target AND patch esistente (come `jepa.pose_terms`).
    Ritorna: campione e patch di ogni ancora (N,), validita' per angolo (N, J), coppie candidate e positive
    (N, N), ancore che contano (N,). Solo maschere: niente target, niente embedding, nessun numero casuale."""
    if valid.ndim != 3 or valid.dtype != torch.bool:
        raise ValueError("rank_contrast: valid booleano (S, P, J)")
    if type(stride) is not int or stride < 1:
        raise ValueError("rank_contrast: stride intero >= 1, in patch")
    if positives not in POSITIVES:
        raise ValueError(f"rank_contrast: positives {positives!r}, uno di {POSITIVES}")
    keep = torch.zeros(valid.shape[1], dtype=torch.bool, device=valid.device)
    keep[stride // 2::stride] = True
    s_idx, p_idx = (valid.any(dim=-1) & keep).nonzero(as_tuple=True)
    v = valid[s_idx, p_idx]
    with torch.autocast(device_type=valid.device.type, enabled=False):
        m = v.to(torch.float32)
        cand = (m @ m.T > 0) & ~torch.eye(s_idx.numel(), dtype=torch.bool, device=valid.device)
    pos = cand & (s_idx[:, None] != s_idx[None, :]) if positives == "cross" else cand
    return s_idx, p_idx, v, cand, pos, pos.sum(dim=1) >= 2


def rank_contrast_count(valid: torch.Tensor, *, stride: int, positives: str, per_window: bool) -> torch.Tensor:
    """Il conteggio del termine per `jepa.loss_counts` (int64, senza encode): finestre con almeno un'ancora che
    conta (per finestra) o ancore che contano (per token). Uguale al conteggio di `pose_rank_contrast`."""
    s_idx, _, _, _, _, ok = anchor_support(valid, stride, positives)
    if not per_window:
        return ok.sum().detach()
    per_sample = torch.zeros(valid.shape[0], dtype=torch.int64, device=valid.device)
    return (per_sample.index_add(0, s_idx, ok.to(torch.int64)) > 0).sum().detach()


def readout_similarity(h: torch.Tensor) -> torch.Tensor:
    """(N, d) -> (N, N): coseno fra i readout standardizzati sulle d coordinate."""
    u = F.normalize(F.layer_norm(h, h.shape[-1:]), dim=-1)
    return u @ u.T


@torch.no_grad()
def pose_distance(y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """y (N, J), valid (N, J) bool -> (N, N) float64: RMS delle differenze sugli angoli validi in entrambi (0 dove
    non ce ne sono: quelle coppie non sono candidate). Differenze dirette a blocchi: pose identiche danno
    esattamente zero e le parita' sono esatte."""
    t = torch.where(valid, y.to(torch.float64), 0.0)
    m = valid.to(torch.float64)
    d2 = torch.cat([
        ((t[a:a + _CHUNK, None, :] - t[None, :, :]).square()
         * m[a:a + _CHUNK, None, :] * m[None, :, :]).sum(dim=-1)
        for a in range(0, t.shape[0], _CHUNK)])
    return (d2 / (m @ m.T).clamp(min=1.0)).sqrt()


def rank_contrast_pairs(logits: torch.Tensor, dist: torch.Tensor, cand: torch.Tensor, pos: torch.Tensor):
    """Il nucleo, su matrici (N, N) [ancora, candidato]. Ritorna per ancora (N,): somma di l_ij sui positivi, somma
    di ln|S_ij| (il valore a similarita' uniformi), numero di positivi. Ordinamento per distanza e somma cumulata
    dalla coda (O(N^2 log N), non il ciclo O(N^3) del codice dell'articolo). Parita': S_ij comprende tutti i
    candidati alla stessa distanza di j."""
    pos = pos & cand
    d_sorted, order = torch.sort(torch.where(cand, dist, float("inf")), dim=1, stable=True)
    l_sorted = torch.where(cand, logits, _MASKED_LOGIT).gather(1, order)  # i non candidati in coda
    tail = torch.logcumsumexp(l_sorted.flip(1), dim=1).flip(1)  # ln sum_{q >= p} exp(l_sorted[i, q])
    first = torch.searchsorted(d_sorted, d_sorted)  # inizio del blocco di parita'
    pos_sorted = pos.gather(1, order)
    term = torch.where(pos_sorted, tail.gather(1, first) - l_sorted, 0.0)
    size = (cand.sum(dim=1, keepdim=True) - first).clamp(min=1)  # |S_ij|
    chance = torch.where(pos_sorted, size.to(term.dtype).log(), 0.0)
    return term.sum(dim=1), chance.sum(dim=1), pos_sorted.sum(dim=1)


def pose_rank_contrast(z_mean: torch.Tensor, pose: tuple | None, time_valid: torch.Tensor, *, temperature: float,
                       stride: int, positives: str, per_window: bool, resolution: float = 0.0):
    """Il termine per `training.jepa`, con gli ingressi di `jepa.pose_terms`: z_mean (S, P, d) =
    `latent_mean(enc.z)`; pose = (target (S, P, J), maschera (S, P, J)[, ...]) o None; time_valid (S, P) =
    `jepa.pose_time_valid`. Ritorna (perdita, numeratore additivo, conteggio int64, log). Per finestra: media
    dentro ogni finestra con almeno un'ancora che conta, poi fra quelle finestre; per token: media sulle ancore.
    Senza ancore (nessuna finestra di posa, o nessuna coppia positiva) la perdita e' `z_mean.sum() * 0`."""
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("rank_contrast: temperature finita e positiva")
    if type(resolution) not in (int, float) or not math.isfinite(resolution) or resolution < 0:
        raise ValueError("rank_contrast: resolution finita e non negativa")
    zero = z_mean.sum() * 0.0
    none = time_valid.new_zeros((), dtype=torch.int64)
    empty = (zero, zero, none, {"anchors": none, "pairs": none, "raw": zero.detach(), "chance": zero.detach()})
    if pose is None:
        return empty
    target, mask = pose[0], pose[1]
    if target.shape[:2] != z_mean.shape[:2] or mask.shape != target.shape or time_valid.shape != z_mean.shape[:2]:
        raise ValueError(f"rank_contrast: target {tuple(target.shape)}, maschera {tuple(mask.shape)} e patch "
                         f"{tuple(time_valid.shape)} contro il readout {tuple(z_mean.shape)}")
    s_idx, p_idx, v, cand, pos, ok = anchor_support(mask & time_valid[..., None], stride, positives)
    if not bool(ok.any()):
        return empty
    y = target[s_idx, p_idx]
    if not bool(torch.isfinite(y[v]).all()):
        raise ValueError("rank_contrast: target non finiti fra quelli validi")
    # float32 anche sotto autocast bf16; un modello in float64 resta in float64 (test d'equivalenza)
    dtype = torch.float64 if z_mean.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=z_mean.device.type, enabled=False):
        with torch.no_grad():
            dist = pose_distance(y, v)
            if resolution > 0:
                dist = torch.floor(dist / resolution) * resolution
        logits = readout_similarity(z_mean[s_idx, p_idx].to(dtype)) / temperature
        num, chance, n_pos = rank_contrast_pairs(logits, dist, cand, pos)
    # denominatore sicuro: 0/0 nel ramo non scelto di torch.where darebbe NaN nel backward
    value = torch.where(ok, num, 0.0) / torch.where(ok, chance, 1.0)
    r_num, count = masked_sum_count(value, ok, s_idx if per_window else None, z_mean.shape[0])
    per_pair = n_pos.clamp(min=1).to(dtype)
    log = {"anchors": ok.sum().detach(), "pairs": torch.where(ok, n_pos, 0).sum().detach(),
           "raw": torch.where(ok, num.detach() / per_pair, 0.0).sum(),  # somma sulle ancore, non normalizzata
           "chance": torch.where(ok, chance / per_pair, 0.0).sum()}  # idem, a similarita' uniformi
    return r_num / count.clamp(min=1), r_num, count, log
