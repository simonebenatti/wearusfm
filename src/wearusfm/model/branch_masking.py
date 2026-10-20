"""Obiettivo mascherato dei rami spettrali (protocollo `rami_pilota_v1`, `docs/rami_pilota_obiettivo_v1.md`; punto 2 della consegna sui rami
del 20/10/2026, scelta A di Simone): dai frame visibili di ogni ramo e scala si predicono i token del teacher (EMA) nei tratti di tempo
nascosti. Raw congelato e mai nel contesto.

Niente fughe d'informazione dal tratto nascosto:
- lo STUDENTE calcola gli spettri su una copia del segnale coi tratti nascosti AZZERATI (il banco wavelet filtra nel dominio della frequenza: la
  risposta all'impulso ha code gaussiane oltre i +-3 sigma del supporto, piccole ma non nulle; azzerando, i suoi spettri non dipendono mai dai
  campioni nascosti, esattamente);
- e usa come contesto solo i frame il cui supporto INTERO (finestra piu' raggio del filtro, per banda) non tocca nessun tratto nascosto: gli
  altri frame sono «non validi» (zeri prima e dopo ogni strato dell'encoder di scala), quindi l'azzeramento non distorce il contesto;
- una maschera per campione, uguale per tutti i canali, i rami e le scale; il predittore di un ramo vede solo i token di quel ramo e di quella
  scala, mai il raw (attenzione globale) ne' l'altro ramo.
Il TEACHER calcola i token sull'ingresso completo: target = token dei frame col centro in un tratto nascosto, normalizzati per token (LayerNorm
senza parametri), senza gradiente. Le maschere usano un generatore numpy proprio (seme del run, passo, campione): il flusso dei dati non cambia.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wearusfm.model.spectral_branches import SCALES, frame_starts

MASK_SPAN_MS = 200.0  # lunghezza di un tratto nascosto
MASK_FRACTION = 0.25  # frazione della durata utile del ritaglio coperta dai tratti nascosti (almeno uno)
PREDICTOR_DIM, PREDICTOR_LAYERS, PREDICTOR_HEADS = 64, 2, 4
# anti-collasso (congelato prima del training): diversita' dei target normalizzati e rango effettivo dei token bersaglio
COLLAPSE = {"diversity_min": 0.05, "erank_fraction_min": 0.10, "consecutive": 3, "after_updates": 100}
MASKED_OBJECTIVE_VERSION = "rami_pilota_masked_v1"


def sample_spans(t: int, fs: float, guard: tuple[int, int], rng: np.random.Generator, span_ms: float = MASK_SPAN_MS,
                 fraction: float = MASK_FRACTION) -> list[tuple[int, int]]:
    """Tratti nascosti [inizio, fine) in campioni, non sovrapposti, dentro la parte utile [guard0, t - guard1) del ritaglio. Numero: la
    frazione `fraction` della parte utile in tratti da `span_ms` (arrotondata, almeno uno, al piu' quanti ne entrano). Nessun tratto se la
    parte utile e' piu' corta di un tratto."""
    s = int(round(span_ms / 1000.0 * fs))
    lo, hi = int(guard[0]), int(t - guard[1])
    usable = hi - lo
    if s < 1 or usable < s:
        return []
    k = min(max(1, int(round(fraction * usable / s))), usable // s)
    free = usable - k * s
    offsets = np.sort(rng.integers(0, free + 1, size=k)) if free > 0 else np.zeros(k, dtype=np.int64)
    return [(lo + int(o) + i * s, lo + int(o) + (i + 1) * s) for i, o in enumerate(offsets)]


def masked_signal(x: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    """(C, T): copia del segnale coi tratti nascosti azzerati (l'ingresso dello studente)."""
    y = x.clone()
    for a, b in spans:
        y[:, a:b] = 0
    return y


def frame_roles(t: int, fs: float, win_ms: float, hop_ms: float, radius: np.ndarray, spans: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Per una scala: (contesto (P, B), bersaglio (P,)). Contesto: il supporto intero del frame [inizio - r, inizio + n + r) per la banda non
    tocca nessun tratto nascosto. Bersaglio: il centro del frame cade in un tratto nascosto. Un frame bersaglio non e' mai contesto."""
    n, starts = frame_starts(t, fs, win_ms, hop_ms)
    r = np.asarray(radius, dtype=np.int64)
    if not len(starts):
        return np.zeros((0, len(r)), dtype=bool), np.zeros(0, dtype=bool)
    lo = starts[:, None] - r[None, :]
    hi = starts[:, None] + n + r[None, :]
    centers = starts + n / 2.0
    ctx = np.ones((len(starts), len(r)), dtype=bool)
    tgt = np.zeros(len(starts), dtype=bool)
    for a, b in spans:
        ctx &= ~((lo < b) & (hi > a))
        tgt |= (centers >= a) & (centers < b)
    ctx &= ~tgt[:, None]
    return ctx, tgt


def branch_radius(branch, scale: str, fs: float, n_bands: int) -> np.ndarray:
    """Raggio del supporto per banda: 0 per la PSD (la finestra del frame e' il supporto esatto), quello della wavelet per il banco."""
    if branch.kind == "wavelet":
        return branch.frontend.banks[scale].radius(fs)
    return np.zeros(n_bands, dtype=np.int64)


def time_encoding(times_s: torch.Tensor, dim: int) -> torch.Tensor:
    """Codifica sinusoidale del tempo del centro del frame (secondi), periodi da 25 ms a ~10 s."""
    half = dim // 2
    periods = torch.logspace(math.log10(0.025), math.log10(10.0), half, device=times_s.device, dtype=torch.float32)
    ang = 2.0 * math.pi * times_s.float()[..., None] / periods
    return torch.cat([ang.sin(), ang.cos()], -1)


class FramePredictor(nn.Module):
    """Predittore di un ramo e di una scala: per ogni canale, un piccolo transformer sui frame; i frame di contesto entrano col loro token
    (proiettato), gli altri col token di maschera; l'attenzione va solo ai frame di contesto (stesso canale e campione). Uscita: un token per
    frame (si usano quelli bersaglio)."""

    def __init__(self, hidden: int, dim: int = PREDICTOR_DIM, layers: int = PREDICTOR_LAYERS, heads: int = PREDICTOR_HEADS):
        super().__init__()
        self.dim = dim
        self.inp = nn.Linear(hidden, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(dim, hidden)

    def forward(self, tokens: torch.Tensor, ctx: torch.Tensor, times_s: torch.Tensor) -> torch.Tensor:
        """tokens (N, P, H) dello studente (zeri fuori contesto), ctx (N, P) bool, times_s (N, P). Ogni riga deve avere almeno un frame di
        contesto (le righe senza si tolgono prima)."""
        if not bool(ctx.any(1).all()):
            raise ValueError("predittore: una riga senza frame di contesto")
        x = torch.where(ctx[..., None], self.inp(tokens.to(self.inp.weight.dtype)), self.mask_token.to(self.inp.weight.dtype))
        x = x + time_encoding(times_s, self.dim).to(x.dtype)
        return self.out(self.encoder(x, src_key_padding_mask=~ctx))


def make_predictors(kinds, hidden: int, seed: int) -> nn.ModuleDict:
    """Un predittore per ramo e scala, inizializzati con un generatore separato (seme del run): l'inizializzazione dei rami resta quella del
    2C (stessi pesi al passo 0)."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed) * 1000 + 7)
        return nn.ModuleDict({f"{k}_{name}": FramePredictor(hidden) for k in kinds for name, _, _, _ in SCALES})
    finally:
        torch.random.set_rng_state(state)


def target_tokens(h: torch.Tensor) -> torch.Tensor:
    """Target normalizzati per token (LayerNorm senza parametri, come data2vec)."""
    return F.layer_norm(h.float(), h.shape[-1:])


def collapse_stats(targets: torch.Tensor) -> dict:
    """Su una matrice di token bersaglio normalizzati (N, H): diversita' = media sulle coordinate della deviazione standard fra i token (circa
    1 se diversi, 0 se tutti uguali), rango effettivo (exp dell'entropia dei valori singolari normalizzati, come `jepa.effective_rank`)."""
    from wearusfm.training.jepa import effective_rank

    if targets.shape[0] < 2:
        return {"diversity": float("nan"), "erank": float("nan"), "n": int(targets.shape[0])}
    t = targets.detach().float()
    return {"diversity": float(t.std(0).mean()), "erank": float(effective_rank(t)), "n": int(t.shape[0])}


def masked_branch_objective(student, teacher, predictors: nn.ModuleDict, inp, spans: list[list[tuple[int, int]]]) -> tuple[torch.Tensor, dict]:
    """Perdita mascherata dei rami e metriche. `spans[s]` = tratti nascosti del campione s (in campioni del suo segnale). Studente sui segnali
    coi tratti azzerati e sui soli frame di contesto; teacher sull'ingresso completo, senza gradiente. Media per ramo e scala, poi media sui
    rami e scale con almeno un frame bersaglio."""
    from dataclasses import replace as dc_replace

    if len(spans) != len(inp.counts):
        raise ValueError("una lista di tratti nascosti per campione")
    masked_inp = dc_replace(inp, signals=[masked_signal(x, sp) for x, sp in zip(inp.signals, spans)])
    losses, metrics = [], {}
    for kind, sb in student.spectral_branches.branches.items():
        tb = teacher.spectral_branches.branches[kind]
        spec_s = sb.spectra(masked_inp)
        with torch.no_grad():
            spec_t = tb.spectra(inp)
        for name, win_ms, hop_ms, nb in SCALES:
            logp_s, valid_s = spec_s[name]
            logp_t, valid_t = spec_t[name]
            c_tot, p_max = valid_s.shape[:2]
            ctx = torch.zeros(c_tot, p_max, nb, dtype=torch.bool)
            tgt = torch.zeros(c_tot, p_max, dtype=torch.bool)
            times = torch.zeros(c_tot, p_max)
            a = 0
            for x, fs, c, sp in zip(inp.signals, inp.fs, inp.counts, spans):
                cx, tg = frame_roles(x.shape[-1], float(fs), win_ms, hop_ms, branch_radius(sb, name, float(fs), nb), sp)
                p = len(tg)
                ctx[a:a + c, :p] = torch.as_tensor(cx)[None]
                tgt[a:a + c, :p] = torch.as_tensor(tg)[None]
                times[a:a + c, :p] = torch.as_tensor((np.arange(p) + 0.5) * hop_ms / 1000.0, dtype=torch.float32)[None]
                a += c
            dev = valid_s.device
            ctx, tgt, times = ctx.to(dev), tgt.to(dev), times.to(dev)
            h_s, ok_s = sb.scales[name].tokens(logp_s, valid_s & ctx)  # ok_s: frame di contesto con almeno una banda valida
            with torch.no_grad():
                h_t, ok_t = tb.scales[name].tokens(logp_t, valid_t)
                y = target_tokens(h_t)
            rows = ok_s.any(1) & (tgt & ok_t).any(1)  # canali con contesto e con almeno un bersaglio
            key = f"{kind}_{name}"
            if not bool(rows.any()):
                metrics[f"masked_{key}_targets"] = 0
                continue
            pred = predictors[key](h_s[rows], ok_s[rows], times[rows])
            sel = (tgt & ok_t)[rows]
            err = (pred.float() - y[rows]).square().mean(-1)[sel]
            losses.append(err.mean())
            metrics[f"masked_{key}"] = err.mean().detach()
            metrics[f"masked_{key}_targets"] = int(sel.sum())
            metrics[f"masked_{key}_context_frames"] = int(ok_s[rows].sum())
            stats = collapse_stats(y[rows][sel])
            metrics[f"masked_{key}_diversity"], metrics[f"masked_{key}_erank"] = stats["diversity"], stats["erank"]
    if not losses:
        raise ValueError("obiettivo mascherato: nessun frame bersaglio nel batch")
    loss = torch.stack(losses).mean()
    div = [metrics[k] for k in metrics if k.endswith("_diversity") and math.isfinite(metrics[k])]
    er = [metrics[k] for k in metrics if k.endswith("_erank") and math.isfinite(metrics[k])]
    metrics["masked_diversity_min"] = min(div) if div else float("nan")
    metrics["masked_erank_min"] = min(er) if er else float("nan")
    return loss, metrics
