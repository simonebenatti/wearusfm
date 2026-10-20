"""Rami spettrali continui in parallelo all'encoder raw: step 2 (PSD multiscala) e step 3 (banco wavelet) della proposta del 07/10/2026
(`WearUsFM_Proposta_Embedding_EMG_3_Step.docx`; pacchetto di riferimento `WearUsFM_Step2_Step3_Codice_e_Test.zip`). Simone, 13/10/2026: «devi
farli entrambi come rami, non solo lo step 3 in alternativa, lo voglio in parallelo allo step 2 e magari invece di gabor valuta se usare un banco
wavelet». La guida del pacchetto voleva lo step 3 come frontend ALTERNATIVO allo step 2: qui sono due rami affiancati.

Embedding P1/P2 = [readout raw dello Step 1 (2d), z_psd, z_wavelet] (`readout`). Ogni ramo: frontend alla fs nativa -> log-potenze per (canale,
frame, banda), standardizzate con statistiche del training + disponibilita' -> piccolo encoder temporale per scala (50 ms ogni 25 ms, 500 ms ogni
100 ms) -> media sui canali validi -> `out_dim` numeri. Nessuna quantizzazione. **I rami vedono solo la vista completa** (loss sul readout e
downstream): mai la vista mascherata della JEPA, ne' le sue chiavi. Loss sul readout (`readout_objectives`): la keep dello Step 1 sulla sola testa
raw resta invariata (`keep_weight`), e il termine dei rami (testa fusa + teste dei singoli rami) si aggiunge col suo peso (`branch_weight`).

**Banco wavelet invece del Gabor del pacchetto** (valutazione nel registro condiviso, 13/10/2026): wavelet di Morlet analitiche a Q costante, una
per banda dei target (6 e 24 bande log-spaziate su 20-450 Hz, centro geometrico della banda), banda a -3 dB uguale alla banda dei target,
normalizzate a banda equivalente di rumore uguale alla larghezza della banda (per uno spettro piatto l'energia e' la potenza di banda, come la PSD;
le statistiche congelate restano valide anche se le larghezze apprendibili cambiano). Filtraggio nel dominio della frequenza (una rFFT per canale,
nessun kernel troncato), supporto di ogni filtro +-3 sigma_t col suo sigma minimo consentito: le frequenze alte hanno supporto corto. Il Gabor del
pacchetto (40-330 Hz, sigma_f = max(4 Hz, f/12), kernel comune di ~300 ms) lasciava scoperta gran parte della banda nella scala da 6 filtri.

Le energie dei filtri si sovrappongono: non sono le potenze disgiunte dei target e non si sommano in una potenza totale. I target restano quelli
PSD fissi dello Step 1 (`spectral_step1.global_spectral_targets`), mai le energie del banco corrente.

**Supporto vicino ai bordi (diverso dai target, voluto):** un frame del ramo vale solo se l'INTERA finestra (piu' il raggio del filtro) sta fuori
dalla zona di bordo del tratto (guida del pacchetto: supporto completo, niente campioni di transitorio nell'ingresso). I target dello Step 1
escludono invece solo le patch che toccano la zona di bordo, e la finestra lenta centrata su una patch valida puo' entrarci. Lontano dai bordi
coincidono (stesse finestre, provato); su una finestra da 1 s con 0,1 s di bordo i target hanno 6 frame lenti, la PSD del ramo 4 (review GPT
del 13/10/2026). Allineare i target al supporto completo sarebbe un cambio di definizione dei target (file di GPT), da decidere a parte.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from scipy import fft as sfft
from torch import nn
from torch.nn import functional as F

from wearusfm.model.spectral_targets import EPS

SCALES = (("fast", 50.0, 25.0, 6), ("slow", 500.0, 100.0, 24))  # (nome, finestra ms, passo ms, bande): come MS_SCALES e MS_HOP_MS delle ancore
BRANCH_KINDS = ("psd", "wavelet")
SUPPORT_SIGMAS = 3.0  # supporto di un filtro wavelet: +-3 sigma_t (ampiezza e^-4,5 ~ 1%)
BRANCHES_VERSION = "steps23_psd_wavelet_v1"


def band_edges(n_bands: int) -> np.ndarray:
    """I bordi delle bande dei target (`anchor_targets._band_logpower`): log-spaziati, 20 e 450 Hz esatti."""
    return np.geomspace(20.0, 450.0, n_bands + 1)


def frame_starts(t: int, fs: float, win_ms: float, hop_ms: float) -> tuple[int, np.ndarray]:
    """(n, inizi): finestre di n campioni centrate in (i + 0,5) * passo, come le finestre delle ancore; un frame per passo intero nel segnale."""
    n = int(round(win_ms / 1000.0 * fs))
    if n < 3:
        raise ValueError(f"finestra da {win_ms} ms di {n} campioni a {fs} Hz")
    n_frames = int(math.floor(t / fs / (hop_ms / 1000.0) + 1e-9))
    centers = (np.arange(n_frames) + 0.5) * (hop_ms / 1000.0)
    return n, np.round(centers * fs - n / 2.0).astype(np.int64)


def support_valid(starts: np.ndarray, n: int, radius: np.ndarray, t: int, guard: tuple[int, int]) -> np.ndarray:
    """(P, F): il supporto INTERO del frame (finestra piu' il raggio del filtro, per filtro) sta nel segnale e fuori dalle zone di bordo
    (`guard` = campioni di transitorio del filtro all'inizio e alla fine, come `anchor_edge_guard_s` del loader)."""
    r = np.asarray(radius, dtype=np.int64)[None, :]
    return (starts[:, None] - r >= guard[0]) & (starts[:, None] + n + r <= t - guard[1])


class PSDFrontend(nn.Module):
    """Step 2: la PSD monolatera dei target dello Step 1 (`spectral_targets.hann_periodogram`: Hann simmetrica, nfft = finestra, densita' / (fs
    sum w^2), bin positivi raddoppiati), integrata nelle stesse bande, sugli stessi frame. Nessun parametro."""

    kind = "psd_v1"

    def __init__(self):
        super().__init__()
        self.n_bands = {name: nb for name, _, _, nb in SCALES}

    def forward(self, x: torch.Tensor, fs: float, limit: np.ndarray, guard: tuple[int, int]) -> dict:
        c, t = x.shape
        out = {}
        for name, win_ms, hop_ms, nb in SCALES:
            n, starts = frame_starts(t, fs, win_ms, hop_ms)
            if not len(starts):
                out[name] = (x.new_zeros(c, 0, nb), torch.zeros(c, 0, nb, dtype=torch.bool, device=x.device))
                continue
            idx = torch.as_tensor(np.clip(starts[:, None] + np.arange(n)[None, :], 0, t - 1), device=x.device)
            w = torch.hann_window(n, periodic=False, dtype=torch.float32, device=x.device)
            spec = torch.fft.rfft(x[:, idx] * w, dim=-1)  # (C, P, n//2+1)
            density = (spec.real.square() + spec.imag.square()) / float(fs * np.sum(np.hanning(n) ** 2, dtype=np.float64))
            density[..., 1:-1 if n % 2 == 0 else None] *= 2  # n dispari: l'ultimo bin e' positivo, non Nyquist
            freqs = np.fft.rfftfreq(n, 1.0 / fs)
            e = band_edges(nb)
            bands = np.stack([(freqs >= lo) & (freqs < hi) for lo, hi in zip(e[:-1], e[1:])], axis=1)  # (bin, B)
            power = density @ torch.as_tensor(bands, dtype=torch.float32, device=x.device) * (fs / n)
            avail = (e[None, 1:] <= limit[:, None] + 1e-9) & bands.any(axis=0)[None, :]  # come i target
            sup = support_valid(starts, n, np.zeros(nb), t, guard)
            valid = torch.as_tensor(avail[:, None, :] & sup[None, :, :], device=x.device)
            out[name] = (torch.where(valid, power, 0.0), valid)
        return out


class WaveletBank(nn.Module):
    """Una scala del banco: un filtro di Morlet analitico per banda dei target, risposta in frequenza gaussiana
    H(f) = a * exp(-(f - c)^2 / (2 sigma^2)) per f > 0 (0 altrove), con a tale che int |H|^2 df = larghezza della banda.
    Centro e sigma apprendibili solo se `learnable`, dentro limiti fissi (come il Gabor del pacchetto: centro +-4%, sigma 0,8-1,2 del valore
    iniziale; con sigmoid). Disponibilita' e supporto dipendono solo dai limiti, non dai valori correnti: un filtro che si sposta non puo' far
    sparire una coordinata dalla loss."""

    def __init__(self, n_bands: int, learnable: bool = False):
        super().__init__()
        e = band_edges(n_bands)
        center = np.sqrt(e[:-1] * e[1:])
        width = np.diff(e)
        sigma = width / (2.0 * math.sqrt(math.log(2.0)))  # potenza a meta' (-3 dB) ai bordi: banda a -3 dB = banda del target
        f32 = lambda v: torch.as_tensor(v, dtype=torch.float32)  # noqa: E731
        self.register_buffer("center_lo", f32(center * 0.96))
        self.register_buffer("center_hi", f32(center * 1.04))
        self.register_buffer("sigma_lo", f32(sigma * 0.8))
        self.register_buffer("sigma_hi", f32(sigma * 1.2))
        self.register_buffer("width", f32(width))
        self.upper = e[1:]  # bordo superiore della banda: disponibilita' come per la PSD
        self.center_logits = nn.Parameter(torch.zeros(n_bands), requires_grad=learnable)
        self.sigma_logits = nn.Parameter(torch.zeros(n_bands), requires_grad=learnable)

    def physical(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(centro, sigma) in Hz."""
        center = self.center_lo + (self.center_hi - self.center_lo) * torch.sigmoid(self.center_logits)
        sigma = self.sigma_lo + (self.sigma_hi - self.sigma_lo) * torch.sigmoid(self.sigma_logits)
        return center, sigma

    def radius(self, fs: float) -> np.ndarray:
        """Raggio del supporto in campioni, per filtro: SUPPORT_SIGMAS * sigma_t col sigma_f minimo consentito (fisso durante il training)."""
        sigma_t = 1.0 / (2.0 * math.pi * self.sigma_lo.detach().cpu().double().numpy())
        return np.ceil(SUPPORT_SIGMAS * sigma_t * fs).astype(np.int64)

    def response(self, freqs: torch.Tensor) -> torch.Tensor:
        """(F_bins, B) ampiezza della risposta sulle frequenze di una rFFT (0 a f <= 0)."""
        center, sigma = self.physical()
        amp = torch.sqrt(self.width / (sigma * math.sqrt(math.pi)))
        h = amp * torch.exp(-0.5 * ((freqs[:, None] - center[None, :]) / sigma[None, :]).square())
        return torch.where(freqs[:, None] > 0, h, 0.0)


class WaveletFrontend(nn.Module):
    """Step 3 (variante wavelet): energia della risposta analitica di ogni filtro, |y|^2 / 2 (per una sinusoide di ampiezza 1 alla frequenza f:
    |H(f)|^2 / 2), media sulle stesse finestre dei frame della PSD. Per un rumore con PSD monolatera S: energia = int S |H|^2 df."""

    kind = "wavelet_morlet_v1"

    def __init__(self, learnable: bool = False):
        super().__init__()
        self.banks = nn.ModuleDict({name: WaveletBank(nb, learnable) for name, _, _, nb in SCALES})
        self.n_bands = {name: nb for name, _, _, nb in SCALES}

    @property
    def learnable(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def forward(self, x: torch.Tensor, fs: float, limit: np.ndarray, guard: tuple[int, int]) -> dict:
        c, t = x.shape
        radii = {name: self.banks[name].radius(fs) for name, _, _, _ in SCALES}
        pad = int(max(int(r.max()) for r in radii.values()))
        n_fft = sfft.next_fast_len(t + 2 * pad)
        spec = torch.fft.rfft(F.pad(x.float(), (pad, pad)), n=n_fft)  # (C, K)
        k = spec.shape[-1]
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / fs, device=x.device, dtype=torch.float32)
        weight = torch.full((k,), 2.0, device=x.device)  # segnale analitico: positivi x2, Nyquist x1 (DC: H = 0)
        if n_fft % 2 == 0:
            weight[-1] = 1.0
        out = {}
        for name, win_ms, hop_ms, nb in SCALES:
            bank = self.banks[name]
            n, starts = frame_starts(t, fs, win_ms, hop_ms)
            if not len(starts):
                out[name] = (x.new_zeros(c, 0, nb), torch.zeros(c, 0, nb, dtype=torch.bool, device=x.device))
                continue
            h = bank.response(freqs) * weight[:, None]  # (K, B)
            full = torch.zeros(c, nb, n_fft, dtype=spec.dtype, device=x.device)
            full[..., :k] = spec[:, None, :] * h.T[None, :, :]
            y = torch.fft.ifft(full, dim=-1)
            energy = (y.real.square() + y.imag.square()).double() / 2.0  # (C, B, n_fft); float64: differenze di somme cumulate
            cs = F.pad(energy.cumsum(-1), (1, 0))
            s = torch.as_tensor(np.clip(starts + pad, 0, n_fft - n), device=x.device)
            power = ((cs[..., s + n] - cs[..., s]) / n).clamp_min(0.0).float().transpose(1, 2)  # (C, P, B)
            avail = (bank.upper[None, :] <= limit[:, None] + 1e-9)
            sup = support_valid(starts, n, radii[name], t, guard)
            valid = torch.as_tensor(avail[:, None, :] & sup[None, :, :], device=x.device)
            out[name] = (torch.where(valid, power, 0.0), valid)
        return out


def make_frontend(kind: str, learnable: bool = False) -> nn.Module:
    if kind == "psd":
        if learnable:
            raise ValueError("la PSD non ha parametri apprendibili")
        return PSDFrontend()
    if kind == "wavelet":
        return WaveletFrontend(learnable)
    raise ValueError(f"ramo spettrale {kind!r}: uno di {BRANCH_KINDS}")


class ScaleEncoder(nn.Module):
    """Una scala: (log-potenza standardizzata, disponibilita') per frame -> proiezione -> due convoluzioni temporali residue (kernel 3) -> media e
    deviazione standard sui frame validi -> `out_dim` numeri per canale (dal pacchetto, `branch.ScaleEncoder`). I frame non validi sono zeri
    prima e dopo ogni strato: il padding fra campioni impacchettati e' equivalente al padding a zeri della convoluzione."""

    def __init__(self, n_bands: int, hidden: int, out_dim: int):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(n_bands))
        self.register_buffer("input_scale", torch.ones(n_bands))
        self.register_buffer("observed", torch.zeros(n_bands, dtype=torch.bool))
        self.project = nn.Linear(2 * n_bands, hidden)
        self.temporal = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1) for _ in range(2)])
        self.output = nn.Linear(2 * hidden, out_dim)

    def tokens(self, logp: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """I token per canale e frame prima della media sui frame (obiettivo mascherato dei rami, `branch_masking`): ((C, P, hidden), (C, P)
        frame con almeno una banda usata). Stessi valori del forward (che li usa), zeri sui frame non validi."""
        if logp.shape[1] == 0:  # nessun frame nel ritaglio (piu' corto di un passo)
            return logp.new_zeros(logp.shape[0], 0, self.project.out_features), torch.zeros(logp.shape[0], 0, dtype=torch.bool, device=logp.device)
        use = valid & self.observed  # una coordinata mai vista nella calibrazione resta fuori (stessa regola nel training e nelle sonde)
        x = torch.where(use, (logp - self.input_mean) / self.input_scale, 0.0)
        x = torch.cat((x, use.to(x.dtype)), -1)
        ok = use.any(-1)  # (C, P)
        h = torch.where(ok[..., None], F.gelu(self.project(x)), 0.0)
        for conv in self.temporal:
            v = conv(h.transpose(1, 2)).transpose(1, 2)
            h = torch.where(ok[..., None], F.gelu(h + v), 0.0)
        return h, ok

    def forward(self, logp: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """logp, valid: (C, P, B). Ritorna ((C, out_dim), (C,) canale con almeno un frame valido)."""
        if logp.shape[1] == 0:  # nessun frame nel ritaglio (piu' corto di un passo)
            return logp.new_zeros(logp.shape[0], self.output.out_features), torch.zeros(logp.shape[0], dtype=torch.bool, device=logp.device)
        h, ok = self.tokens(logp, valid)
        den = ok.sum(1).clamp(min=1)[:, None].to(h.dtype)
        mean = h.sum(1) / den
        dev = torch.where(ok[..., None], h - mean[:, None, :], 0.0)
        std = (dev.square().sum(1) / den + 1e-6).sqrt()
        return self.output(torch.cat((mean, std), -1)), ok.any(-1)


class SpectralBranch(nn.Module):
    """Un ramo: frontend -> un `ScaleEncoder` per scala -> media sui canali validi del campione -> combinazione delle scale (piu' due flag di
    presenza) in `out_dim` numeri. Invariante all'ordine dei canali: l'identita' spaziale resta nel ramo raw."""

    def __init__(self, kind: str, out_dim: int = 128, hidden: int = 48, learnable: bool = False):
        super().__init__()
        if out_dim < 1 or hidden < 1:
            raise ValueError("dimensioni del ramo positive")
        self.kind = kind
        self.frontend = make_frontend(kind, learnable)
        self.scales = nn.ModuleDict({name: ScaleEncoder(nb, hidden, 64) for name, _, _, nb in SCALES})
        self.combine = nn.Linear(64 * len(SCALES) + len(SCALES), out_dim)
        self.register_buffer("stats_ready", torch.tensor(False))

    def spectra(self, inp) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Per scala: (log-potenze (C_tot, P_max, B), validita'), campioni impacchettati sui canali, frame in coda non validi."""
        if inp.band_limit is None or inp.edge_guard is None:
            raise ValueError("i rami spettrali vogliono band_limit ed edge_guard negli ingressi (dal loader o dall'estrazione delle feature)")
        learn = getattr(self.frontend, "learnable", False) and torch.is_grad_enabled()
        per_sample = []
        with torch.autocast(device_type=inp.signals[0].device.type, enabled=False), torch.set_grad_enabled(learn):
            qc_all, a = inp.qc_valid, 0
            for x, fs, c, limit, guard in zip(inp.signals, inp.fs, inp.counts, inp.band_limit, inp.edge_guard):
                qc = qc_all[a:a + c]
                a += c
                xs = torch.where(qc[:, None], x.float(), 0.0)  # i canali scartati dal QC non entrano (neanche un NaN)
                lim = np.minimum(np.asarray(limit, dtype=np.float64).reshape(-1), fs / 2.0)
                if lim.shape != (c,):
                    raise ValueError("band_limit: un valore per canale")
                if learn:
                    out = torch.utils.checkpoint.checkpoint(self.frontend, xs, float(fs), lim, tuple(guard), use_reentrant=False)
                else:
                    out = self.frontend(xs, float(fs), lim, tuple(guard))
                per_sample.append({k: (p, v & qc[:, None, None]) for k, (p, v) in out.items()})
        packed = {}
        for name, _, _, nb in SCALES:
            p_max = max(s[name][0].shape[1] for s in per_sample)
            pw, vd = [], []
            for s in per_sample:
                p, v = s[name]
                pw.append(F.pad(p, (0, 0, 0, p_max - p.shape[1])))
                vd.append(F.pad(v, (0, 0, 0, p_max - v.shape[1])))
            power, valid = torch.cat(pw), torch.cat(vd)
            packed[name] = (torch.where(valid, power.clamp_min(0.0).add(EPS).log(), 0.0), valid)
        return packed

    @torch.no_grad()
    def accumulate_stats(self, inp, acc: dict | None = None) -> dict:
        """Somme (conteggio, somma, quadrati) delle log-potenze valide per coordinata, in float64: si chiama su batch del SOLO training."""
        acc = acc if acc is not None else {}
        for name, (logp, valid) in self.spectra(inp).items():
            x = torch.where(valid, logp.double(), 0.0)
            cur = (valid.sum((0, 1)).double(), x.sum((0, 1)), x.square().sum((0, 1)))
            acc[name] = cur if name not in acc else tuple(u + v for u, v in zip(acc[name], cur))
        return acc

    @torch.no_grad()
    def set_stats(self, acc: dict) -> None:
        """Statistiche congelate di standardizzazione (anche col frontend apprendibile). Una coordinata mai osservata resta fuori dall'ingresso."""
        for name, layer in self.scales.items():
            count, total, squares = (v.to(layer.input_mean.device) for v in acc[name])
            if not bool((count > 0).any()):
                raise ValueError(f"nessun frame valido nella calibrazione per la scala {name}")
            mean = total / count.clamp(min=1)
            std = (squares / count.clamp(min=1) - mean.square()).clamp(min=0).sqrt()
            layer.input_mean.copy_(mean.float())
            layer.input_scale.copy_(torch.where((count >= 2) & (std > 1e-5), std, 1.0).float())
            layer.observed.copy_(count > 0)
        self.stats_ready.fill_(True)

    def forward(self, inp, spectra: dict | None = None) -> torch.Tensor:
        """(S, out_dim). Un campione senza nessun frame valido in una scala ha quella scala a zero e il flag spento (niente eccezione nel
        training: crop corti o bande non acquisite restano un caso dichiarato dal flag)."""
        if not bool(self.stats_ready):
            raise RuntimeError("calibrare le statistiche d'ingresso del ramo sul training (accumulate_stats / set_stats) prima dell'uso")
        spectra = spectra if spectra is not None else self.spectra(inp)
        counts = torch.as_tensor(inp.counts, device=inp.qc_valid.device)
        owner = torch.repeat_interleave(torch.arange(len(inp.counts), device=counts.device), counts)
        dtype = self.combine.weight.dtype  # i frontend lavorano in float32; l'encoder nel tipo del modello
        parts, flags = [], []
        for name, layer in self.scales.items():
            logp, valid = spectra[name]
            z, ok = layer(logp.to(dtype), valid)  # (C_tot, 64), (C_tot,)
            z = z.to(torch.promote_types(z.dtype, torch.float32))  # media sui canali (fino a centinaia) almeno in float32, anche sotto bf16
            n_ok = torch.zeros(len(inp.counts), device=z.device, dtype=z.dtype).index_add(0, owner, ok.to(z.dtype))
            total = torch.zeros(len(inp.counts), z.shape[1], device=z.device, dtype=z.dtype).index_add(0, owner, torch.where(ok[:, None], z, 0.0))
            parts.append(total / n_ok.clamp(min=1)[:, None])
            flags.append((n_ok > 0).to(z.dtype))
        return self.combine(torch.cat((*parts, torch.stack(flags, 1)), -1))


class SpectralBranches(nn.Module):
    """I rami del modello (`FMConfig.spectral_branches`), in quell'ordine."""

    def __init__(self, kinds, out_dim: int, hidden: int, wavelet_learnable: bool = False):
        super().__init__()
        kinds = tuple(kinds)
        if not kinds or len(set(kinds)) != len(kinds) or any(k not in BRANCH_KINDS for k in kinds):
            raise ValueError(f"rami spettrali {kinds}: distinti, fra {BRANCH_KINDS}")
        self.branches = nn.ModuleDict({k: SpectralBranch(k, out_dim, hidden, learnable=(k == "wavelet" and wavelet_learnable)) for k in kinds})
        self.out_dim = out_dim

    def forward(self, inp) -> dict[str, torch.Tensor]:
        return {k: b(inp) for k, b in self.branches.items()}

    @torch.no_grad()
    def calibrate(self, batches: Iterable) -> dict:
        """Statistiche d'ingresso di ogni ramo da batch del SOLO training (ModelInputs). Ritorna il conteggio dei frame validi per scala e ramo."""
        acc = {k: {} for k in self.branches}
        n = 0
        for inp in batches:
            n += 1
            for k, b in self.branches.items():
                acc[k] = b.accumulate_stats(inp, acc[k])
        if n == 0:
            raise ValueError("calibrazione dei rami spettrali senza batch")
        for k, b in self.branches.items():
            b.set_stats(acc[k])
        return {k: {s: int(v[0].sum()) for s, v in acc[k].items()} for k in acc}


# --- readout, teste e loss sul readout con i rami ---------------------------------------------------------------------------------------

BRANCH_PREFIXES = ("spectral_branches.", "branch_heads.", "branch_predictors.")


def branch_heads(raw_dim: int, kinds, out_dim: int) -> nn.ModuleDict:
    """Teste lineari della loss sul readout (stessi 32 target dello Step 1): una sull'embedding fuso [raw, rami] e una per ramo da solo, che
    impedisce alla testa fusa di risolvere il compito col solo raw lasciando inutilizzato un ramo (pacchetto, `fusion.SpectralFusion`)."""
    from wearusfm.model.spectral_step1 import SpectralKeepHead

    kinds = tuple(kinds)
    return nn.ModuleDict({"fused": SpectralKeepHead(raw_dim + out_dim * len(kinds)), **{k: SpectralKeepHead(out_dim) for k in kinds}})


def check_head_calibration(model) -> None:
    """Le teste del readout (raw e rami) con la STESSA calibrazione dei target: statistiche stimate, coordinate attive, medie e deviazioni,
    uguali bit per bit a quelle della testa raw. Review GPT del 14/10/2026: `target_active` uguale garantisce gli stessi conteggi, non la stessa
    standardizzazione (con la media di una testa spostata la loss cambiava senza errori). Un confronto per testa (una sincronizzazione, come il
    solo `target_active`); si chiama a ogni passo e quando il trainer prepara o riprende un modello."""
    def signature(h):
        return torch.cat([h.stats_fitted.reshape(1).double(), h.target_active.double(), h.target_mean.double(), h.target_std.double()])

    ref = signature(model.spectral_keep)
    bad = [k for k, h in model.branch_heads.items() if not torch.equal(signature(h), ref)]
    if bad:
        raise ValueError(f"teste del readout {bad} con una calibrazione dei target diversa dalla testa raw (stats_fitted, target_active, "
                         "target_mean, target_std): stessa calibrazione per tutte")


def readout(model, enc, inp) -> torch.Tensor:
    """L'embedding P1/P2 (sonde e loss sul readout): il readout raw dello Step 1 (`pool_p1p2`, 2d) seguito dai rami spettrali del modello,
    nell'ordine di `FMConfig.spectral_branches`; senza rami e' `pool_p1p2`. `enc` dalla vista COMPLETA (encode senza maschera)."""
    from wearusfm.model.spectral_step1 import pool_p1p2

    z = pool_p1p2(enc, inp.counts, inp.qc_valid)
    if getattr(model, "spectral_branches", None) is None:
        return z
    return torch.cat([z.float(), *[v.float() for v in model.spectral_branches(inp).values()]], -1)


def freeze_raw(model) -> None:
    """Diagnostica dello step 2C: tutto il modello congelato tranne rami e loro teste (un ramo con filtri fissi resta fisso)."""
    for name, p in model.named_parameters():
        if not name.startswith(BRANCH_PREFIXES):
            p.requires_grad_(False)
    model.raw_frozen = True


def readout_objectives(student, inp, anchor_targets, *, keep: bool, terms: bool = False) -> dict:
    """Le loss sul readout di un modello con i rami, da UN solo encode non mascherato dello studente (proposta del 13/10/2026, parere di GPT
    `d5c53c1`: due termini additivi, nessun encode in piu' per testa):
    - "keep" (se `keep`): la loss dello Step 1 sulla sola testa raw (`spectral_keep` su `pool_p1p2`), identica a `spectral_step1.keep_objective`,
      pesata da `keep_weight`. L'encoder raw conserva cosi' il vincolo spettrale diretto del braccio B, che i rami non possono soddisfare al suo
      posto (non lo stesso aggiornamento: anche la testa fusa manda gradienti al raw);
    - "keep_branches": per campione (1 - alpha) L(testa fusa su [raw, PSD, wavelet]) + alpha * media sui rami di L(testa del ramo), pesata da
      `branch_weight`.
    Stessi target, maschere e calibrazione per tutte le teste (`check_head_calibration`): stesso denominatore. Ogni valore ha la forma di
    ritorno di `keep_objective` (o di `keep_objective_terms` con `terms`). Metriche del termine dei rami: `keep_<testa>` e
    `keep_<testa>_<famiglia>` per fused, psd, wavelet, e `keep_branches_valid_samples`. Con l'encoder congelato (`freeze_raw`) l'encode e' in
    eval e senza gradiente, e la testa raw non si usa."""
    from contextlib import nullcontext

    from wearusfm.model.spectral_step1 import global_spectral_targets, pool_p1p2

    if any(x.device.type not in ("cpu", "cuda") for x in inp.signals):
        raise ValueError("rami spettrali: solo CPU/CUDA, come lo Step 1")
    frozen = bool(getattr(student, "raw_frozen", False))
    if keep and frozen:
        raise ValueError("encoder congelato: la keep raw non allena nulla, solo il termine dei rami")
    check_head_calibration(student)
    y, valid = global_spectral_targets(anchor_targets, inp.counts, inp.qc_valid)
    devices = sorted({x.device.index for x in inp.signals if x.is_cuda})
    with torch.random.fork_rng(devices=devices):  # come lo Step 1: il passaggio in piu' non sposta il generatore della JEPA
        was = student.training
        if frozen:
            student.eval()
        try:
            with torch.no_grad() if frozen else nullcontext():
                z = pool_p1p2(student.encode(inp, None), inp.counts, inp.qc_valid)
        finally:
            student.train(was)
    out = {}
    if keep:  # come _keep_objective dello Step 1, sullo stesso z (al tipo della testa: in float32 nessuna conversione, bit per bit lo Step 1)
        prediction = student.spectral_keep(z.to(student.spectral_keep.linear.weight.dtype))
        out["keep"] = student.spectral_keep.loss_terms(prediction, y, valid) if terms else student.spectral_keep.loss(prediction, y, valid)
    zb = student.spectral_branches(inp)
    heads = student.branch_heads
    dt = heads["fused"].linear.weight.dtype  # pool_p1p2 restituisce float32; un modello in float64 (test d'equivalenza) resta in float64
    res = {"fused": heads["fused"].loss_terms(heads["fused"](torch.cat([z.to(dt), *[v.to(dt) for v in zb.values()]], -1)), y, valid)}
    res.update({k: heads[k].loss_terms(heads[k](v), y, valid) for k, v in zb.items()})
    den = res["fused"][3]
    alpha = float(student.cfg.branch_alpha)
    num = (1.0 - alpha) * res["fused"][2] + alpha * torch.stack([res[k][2] for k in zb]).mean()
    loss = num / den.clamp(min=1)
    metrics, log_num, log_den = {"keep_branches_valid_samples": res["fused"][1]["keep_valid_samples"]}, {}, {}
    for k, r in res.items():
        metrics["keep_" + k] = r[0].detach()
        log_num["keep_" + k], log_den["keep_" + k] = r[2].detach(), r[3].detach()
        metrics.update({"keep_" + k + m[len("keep"):]: v for m, v in r[1].items() if m != "keep_valid_samples"})
        log_num.update({"keep_" + k + m[len("keep"):]: v for m, v in r[4].items()})
        log_den.update({"keep_" + k + m[len("keep"):]: v for m, v in r[5].items()})
    out["keep_branches"] = (loss, metrics, num, den, log_num, log_den) if terms else (loss, metrics)
    return out
