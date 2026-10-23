"""Front-end a kernel continui (v10 §4.2; passo 3; gate D8).

Ogni patch temporale di un canale (definita in **millisecondi**, v10 §4.1) diventa un vettore di `d` numeri: la componente j e' l'integrale del prodotto
fra la patch e un kernel continuo k_j(t), approssimato con la somma di Riemann sulla griglia di campionamento del dataset:

    e_j = sum_n  k_j(t_n) x(t_n) * Δt,    Δt = 1/fs,    t_n = istanti dei campioni dentro la patch (relativi al suo inizio)

**Stessi pesi per ogni frequenza; cambia solo la griglia di valutazione.** Tre famiglie di kernel concatenate (v10 §4.2, decisa): base di Fourier
(armoniche della lunghezza della patch), spline cubiche (nodi uniformi), MLP su t con attivazioni seno (stile CKConv/SIREN).

Due requisiti di v10 §4.2, implementati per ciascuna famiglia:
- **fattore Δt**: senza, l'uscita crescerebbe con fs e il front-end consegnerebbe un identificatore di dataset gratuito;
- **anti-aliasing**: prima di campionare il kernel sulla griglia a fs, il kernel (calcolato su una griglia fine di riferimento) viene filtrato
  passa-basso alla Nyquist fs/2, con una rampa a coseno rialzato fra `taper_start * fs/2` e fs/2, e poi valutato esattamente negli istanti dei
  campioni (somma di Fourier, non interpolazione).

Scelte di AG non scritte in v10 (dichiarate qui, rivedibili): finestra di Hann sulla patch per tutte le famiglie (il kernel va a zero ai bordi della
patch, cosi' la somma di Riemann e' accurata anche con pochi campioni); griglia di riferimento a 16 kHz; rampa da 0,9 x Nyquist; numero di kernel per
famiglia. La lunghezza della patch (25 ms) e' il default di lavoro del piano (D10, non ancora deciso).

**Contesto oltre la patch.** Il kernel filtrato contro l'aliasing non e' piu' limitato alla patch: ha code ai lati (tanto piu' lunghe quanto piu' stretta
e' la rampa). Per un segnale a banda limitata il prodotto scalare e' esatto solo sommando anche i campioni vicini su cui la coda non e' trascurabile:
per questo la somma si estende di `context_ms` per lato (campioni delle patch vicine; zeri oltre i bordi del segnale, quindi le patch ai bordi sono
meno accurate). Misurato su segnali sintetici a banda limitata il 01/10/2026, prima del gate: senza contesto l'errore fra 2 kHz e 200 Hz era 13%.

Frequenze per cui la patch non contiene un numero intero di campioni (es. 2048 Hz x 25 ms = 51,2): gestite calcolando, per ogni patch, gli istanti
esatti dei campioni che cadono dentro di essa.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

FAMILIES = ("fourier", "spline", "mlp")
# finestra piu' lunga della patch (Simone, 23/10/2026: «fare una finestra un po piu lunga con overlap, in modo che lo scorrimento sia 25ms»):
# la griglia dei token resta quella della patch (il passo), il kernel vive su `window_ms` centrati sulla patch. Per conservare lo stesso
# contenuto in frequenza fisico del banco da 25 ms, le basi scalano col rapporto finestra/patch: armoniche a passo 1/finestra fino alla stessa
# frequenza massima, nodi delle spline allo stesso passo in ms, stessa frequenza al secondo per il SIREN.
WINDOW_SCALED = ("n_harmonics", "n_knots", "mlp_omega0")


def window_kwargs(patch_ms: float, window_ms: float | None, n_harmonics: int = 26, n_knots: int = 20, mlp_omega0: float = 6.0) -> dict:
    """Gli argomenti di `ContinuousKernelFrontEnd` per una finestra `window_ms` sul passo `patch_ms`: {} senza finestra (il banco di sempre)."""
    if window_ms is None:
        return {}
    if not (math.isfinite(window_ms) and window_ms > patch_ms > 0):
        raise ValueError(f"window_ms {window_ms} deve essere finita e maggiore della patch ({patch_ms} ms)")
    r = window_ms / patch_ms
    return {"window_ms": float(window_ms), "n_harmonics": int(round(n_harmonics * r)), "n_knots": int(round(n_knots * r)), "mlp_omega0": mlp_omega0 * r}


def _hann(t: torch.Tensor, period: float) -> torch.Tensor:
    return torch.sin(math.pi * t / period) ** 2


def _cubic_bspline_basis(t: torch.Tensor, period: float, n_basis: int) -> torch.Tensor:
    """(n_t, n_basis): B-spline cubiche con nodi uniformi su [0, period] (base di Cox-de Boor, nodi estesi ai lati)."""
    h = period / (n_basis - 3)
    knots = torch.arange(-3, n_basis + 1, dtype=t.dtype) * h  # n_basis + 4 nodi
    basis = ((t[:, None] >= knots[None, :-1]) & (t[:, None] < knots[None, 1:])).to(t.dtype)
    for k in range(1, 4):
        left = (t[:, None] - knots[None, : -(k + 1)]) / (knots[k:-1] - knots[: -(k + 1)])[None, :]
        right = (knots[None, k + 1 :] - t[:, None]) / (knots[k + 1 :] - knots[1:-k])[None, :]
        basis = left * basis[:, :-1] + right * basis[:, 1:]
    return basis[:, :n_basis]


class _Siren(nn.Module):
    def __init__(self, n_out: int, hidden: int, omega0: float, generator: torch.Generator):
        super().__init__()
        self.omega0 = omega0
        self.l1, self.l2, self.l3 = nn.Linear(1, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, n_out)
        with torch.no_grad():  # inizializzazione SIREN (Sitzmann et al. 2020)
            self.l1.weight.uniform_(-1.0, 1.0, generator=generator)
            for lin in (self.l2, self.l3):
                b = math.sqrt(6.0 / lin.in_features) / omega0
                lin.weight.uniform_(-b, b, generator=generator)
            for lin in (self.l1, self.l2, self.l3):
                lin.bias.uniform_(-0.1, 0.1, generator=generator)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # u in [-1, 1], (n_t, 1)
        h = torch.sin(self.omega0 * self.l1(u))
        h = torch.sin(self.omega0 * self.l2(h))
        return self.l3(h)


class ContinuousKernelFrontEnd(nn.Module):
    def __init__(
        self,
        n_fourier: int = 24,
        n_spline: int = 24,
        n_mlp: int = 16,
        patch_ms: float = 25.0,
        n_harmonics: int = 26,
        n_knots: int = 20,
        mlp_hidden: int = 32,
        mlp_omega0: float = 6.0,
        ref_fs: float = 16000.0,
        taper_start: float = 0.9,
        context_ms: float = 100.0,  # 01/10/2026, su segnali sintetici: 200 Hz contro 2 kHz 0,35% (50 ms: 2%; 0 ms: 13%)
        seed: int = 0,
        window_ms: float | None = None,  # supporto del kernel; None = la patch (il banco di sempre). Vedi `window_kwargs`
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.patch_s = patch_ms / 1000.0  # il passo: la griglia dei token, delle maschere e delle ancore
        self.window_s = self.patch_s if window_ms is None else window_ms / 1000.0  # il supporto del kernel, centrato sulla patch
        if self.window_s < self.patch_s:
            raise ValueError(f"finestra {self.window_s * 1000:.1f} ms piu' corta della patch ({patch_ms} ms)")
        self.lead_s = (self.window_s - self.patch_s) / 2.0  # quanto la finestra precede (e segue) la patch
        self.ref_fs = float(ref_fs)
        self.taper_start = float(taper_start)
        self.context_s = context_ms / 1000.0
        n_ref = int(round(self.window_s * ref_fs))
        t_ref = (torch.arange(n_ref, dtype=torch.float64) + 0.5) / ref_fs
        self.register_buffer("t_ref", t_ref, persistent=False)
        self.register_buffer("window", _hann(t_ref, self.window_s), persistent=False)
        freqs = torch.arange(n_harmonics, dtype=torch.float64) / self.window_s  # armoniche della finestra: 0, 40, 80, ... Hz a 25 ms
        self.register_buffer("fourier_basis", torch.cat([torch.cos(2 * math.pi * freqs[None] * t_ref[:, None]),
                                                         torch.sin(2 * math.pi * freqs[None] * t_ref[:, None])], dim=1), persistent=False)
        self.register_buffer("spline_basis", _cubic_bspline_basis(t_ref, self.window_s, n_knots), persistent=False)
        self.fourier_coef = nn.Parameter(torch.randn(n_fourier, 2 * n_harmonics, generator=g, dtype=torch.float64) / math.sqrt(2 * n_harmonics))
        self.spline_coef = nn.Parameter(torch.randn(n_spline, n_knots, generator=g, dtype=torch.float64) / math.sqrt(n_knots))
        self.mlp = _Siren(n_mlp, mlp_hidden, mlp_omega0, g).double()
        self.sizes = {"fourier": n_fourier, "spline": n_spline, "mlp": n_mlp}

    @property
    def d_out(self) -> int:
        return sum(self.sizes.values())

    def family_slices(self) -> dict[str, slice]:
        out, o = {}, 0
        for f in FAMILIES:
            out[f] = slice(o, o + self.sizes[f])
            o += self.sizes[f]
        return out

    def kernels_ref(self) -> torch.Tensor:
        """(d, n_ref): i kernel sulla griglia fine di riferimento (la finestra), finestrati, nell'ordine delle famiglie."""
        u = (2.0 * self.t_ref / self.window_s - 1.0)[:, None]
        k = torch.cat([self.fourier_coef @ self.fourier_basis.T, self.spline_coef @ self.spline_basis.T, self.mlp(u).T], dim=0)
        return k * self.window[None, :]

    def _lowpass_response(self, freqs: torch.Tensor, fs: float) -> torch.Tensor:
        nyq = fs / 2.0
        f0 = self.taper_start * nyq
        h = torch.ones_like(freqs)
        ramp = (freqs > f0) & (freqs < nyq)
        h[ramp] = 0.5 * (1.0 + torch.cos(math.pi * (freqs[ramp] - f0) / (nyq - f0)))
        h[freqs >= nyq] = 0.0
        return h

    def kernels_at(self, times: torch.Tensor, fs: float) -> torch.Tensor:
        """Kernel filtrati alla Nyquist di `fs` (anti-aliasing per ciascuna famiglia) e valutati negli istanti `times` (secondi, relativi
        all'inizio della FINESTRA, che coincide con l'inizio della patch quando non c'e' una finestra piu' lunga). Ritorna (d, n_times)."""
        k = self.kernels_ref()
        n_ref = k.shape[1]
        # la trasformata e' periodica: il periodo deve coprire tutti gli istanti richiesti piu' le code del kernel, altrimenti le copie si sommano
        span = float(times.max() - times.min()) if times.numel() else 0.0
        n_fft = 1 << int(math.ceil(math.log2(max(4 * n_ref, 2 * (span + self.window_s) * self.ref_fs + 1))))
        spec = torch.fft.rfft(k, n=n_fft, dim=1)  # (d, n_bins)
        dev = k.device  # tutto sul dispositivo dei parametri (su GPU un tensore creato senza `device` resta sulla CPU)
        freqs = torch.arange(spec.shape[1], dtype=torch.float64, device=dev) * self.ref_fs / n_fft
        spec = spec * self._lowpass_response(freqs, fs)[None, :]
        weights = torch.full((spec.shape[1],), 2.0, dtype=torch.float64, device=dev)
        weights[0] = 1.0
        if n_fft % 2 == 0:
            weights[-1] = 1.0
        u = times.to(device=dev, dtype=torch.float64) - 0.5 / self.ref_fs  # il campione fine i sta all'istante (i + 0,5)/ref_fs
        phase = 2 * math.pi * freqs[None, :] * u[:, None]  # (n_times, n_bins)
        out = (spec.real * weights) @ torch.cos(phase).T - (spec.imag * weights) @ torch.sin(phase).T
        return out / n_fft

    def forward(self, x: torch.Tensor, fs: float) -> torch.Tensor:
        """x: (B, C, T) campionato a `fs`, con t = 0 sul primo campione. Ritorna (B, C, n_patch, d): una riga per patch da `patch_ms`, la griglia
        delle patch ancorata a t = 0. Il resto finale (meno di una patch) si scarta."""
        b, c, n = x.shape
        n_patch = int(math.floor(n / fs / self.patch_s + 1e-9))
        if n_patch == 0:
            raise ValueError(f"segnale di {n / fs * 1000:.1f} ms, piu' corto di una patch ({self.patch_s * 1000:.1f} ms)")
        p = np.arange(n_patch)
        reach = self.context_s + self.lead_s  # oltre la patch, per lato: il contesto dell'anti-aliasing piu' la meta' eccedente della finestra
        m = int(math.ceil(reach * fs - 1e-9))  # campioni oltre la patch per lato
        starts = np.ceil(p * self.patch_s * fs - 1e-9).astype(np.int64)
        fracs = starts / fs - p * self.patch_s  # in [0, 1/fs): dove cade il primo campione dentro la patch
        n_max = int(math.ceil(self.patch_s * fs + 1e-9)) + 2 * m
        keys = np.round(fracs * 1e9).astype(np.int64)  # le patch con lo stesso sfasamento condividono i pesi
        uniq, inv = np.unique(keys, return_inverse=True)
        rel = uniq[:, None] / 1e9 + (np.arange(n_max)[None, :] - m) / fs  # (n_u, n_max) istanti relativi all'inizio della patch
        valid = (rel >= -reach - 1e-12) & (rel < self.patch_s + reach - 1e-12)
        dev = x.device
        # il kernel vive sulla finestra, che comincia `lead_s` prima della patch
        w = self.kernels_at(torch.from_numpy((rel + self.lead_s).reshape(-1)), fs).T.reshape(len(uniq), n_max, -1)  # (n_u, n_max, d)
        w = w * torch.from_numpy(valid).to(device=dev, dtype=w.dtype)[..., None]
        idx = starts[:, None] - m + np.arange(n_max)[None, :]
        inside = (idx >= 0) & (idx < n)  # fuori dal segnale: zeri
        idx = np.clip(idx, 0, n - 1)
        xg = x.to(torch.float64)[..., torch.from_numpy(idx).to(dev)]  # (B, C, n_patch, n_max)
        xg = xg * torch.from_numpy(inside).to(device=dev, dtype=xg.dtype)
        return torch.einsum("bcpj,pjd->bcpd", xg, w[torch.from_numpy(inv).to(dev)]) / fs  # fattore Δt
