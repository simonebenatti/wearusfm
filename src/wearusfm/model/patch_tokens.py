"""Token di patch per il modello (passo 6; v10 §4.1-4.2): il front-end a kernel continui (`frontend.py`, quello misurato dal gate D8) applicato a
ogni campione alla SUA frequenza nativa, poi una proiezione lineare alla dimensione del modello.

Ingresso: una lista di campioni, ciascuno (C_s, T_s) con la sua fs (gia' normalizzato dal dataloader: scala di sessione condivisa fra canali,
v10 §4.3). Uscita impacchettata sull'asse dei canali: token (C_tot, P_max, dim), con `patch_valid` (C_tot, P_max) e `time_valid` (S, P_max) per le
finestre piu' corte (contesto variabile, proposta D10): le patch oltre la fine di un campione sono padding.
Il front-end lavora in float64 (precisione della somma di Riemann, gate D8); la proiezione in float32.

**Guadagno d'ingresso** (review del 03/10): il front-end integra (fattore Δt = 1/fs), quindi su un segnale normalizzato le sue uscite hanno
deviazione ~1e-3, e all'inizializzazione il contenuto pesava ~0,1% del token contro bias e identita' di canale (perdita JEPA iniziale ~1e-10).
Un solo scalare fisso, calcolato alla costruzione su rumore a banda 20-450 Hz di varianza unitaria a 2 kHz (seme fisso, salvato nel
checkpoint), porta le uscite a deviazione ~1 prima della proiezione. Scalare e non per uscita: non amplifica i kernel che sull'EMG filtrato non
rispondono. Il front-end misurato dal gate D8 non cambia.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from wearusfm.model.frontend import ContinuousKernelFrontEnd


class PatchTokenizer(nn.Module):
    def __init__(self, dim: int, frontend: ContinuousKernelFrontEnd | None = None, **frontend_kwargs):
        super().__init__()
        self.frontend = frontend if frontend is not None else ContinuousKernelFrontEnd(**frontend_kwargs)
        self.proj = nn.Linear(self.frontend.d_out, dim)
        self.register_buffer("fe_gain", torch.tensor(self._calibrate_gain(), dtype=torch.float64))

    def _calibrate_gain(self, fs: float = 2000.0, seconds: float = 4.0, n_ch: int = 4) -> float:
        import numpy as np
        from scipy import signal as sps

        x = np.random.default_rng(0).normal(size=(n_ch, int(seconds * fs)))
        x = sps.sosfiltfilt(sps.butter(4, [20.0, 450.0], btype="bandpass", fs=fs, output="sos"), x, axis=-1)
        x = x / x.std()
        with torch.no_grad():
            f = self.frontend(torch.from_numpy(x)[None], fs)
        return float(1.0 / f.std())

    @property
    def patch_s(self) -> float:
        return self.frontend.patch_s

    def forward(self, signals: list[torch.Tensor], fs: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(signals) != len(fs) or not signals:
            raise ValueError("serve una fs per campione, e almeno un campione")
        feats = [self.frontend(x[None], float(f))[0] for x, f in zip(signals, fs)]  # (C_s, P_s, d_fe)
        p_max = max(f.shape[1] for f in feats)
        dim = self.proj.out_features
        tokens, patch_valid, time_valid = [], [], []
        for f in feats:
            c, p, _ = f.shape
            t = self.proj((f * self.fe_gain).to(self.proj.weight.dtype))
            tokens.append(torch.cat([t, t.new_zeros(c, p_max - p, dim)], dim=1))
            pv = torch.zeros(c, p_max, dtype=torch.bool, device=t.device)
            pv[:, :p] = True
            patch_valid.append(pv)
            tv = torch.zeros(p_max, dtype=torch.bool, device=t.device)
            tv[:p] = True
            time_valid.append(tv)
        return torch.cat(tokens, dim=0), torch.cat(patch_valid, dim=0), torch.stack(time_valid)

    def flops(self, n_channels: int, n_patches: int, fs: float) -> int:
        """Moltiplicazioni-addizioni x 2 per un campione: la somma pesata sul supporto di ogni patch (patch piu' contesto per lato,
        `frontend.context_s`) per ogni kernel, piu' la proiezione. Il calcolo dei kernel sulla griglia (una volta per sfasamento) e' trascurabile."""
        support = math.ceil(self.frontend.patch_s * fs + 1e-9) + 2 * math.ceil(self.frontend.context_s * fs - 1e-9)
        d_fe, dim = self.frontend.d_out, self.proj.out_features
        return 2 * n_channels * n_patches * (support * d_fe + d_fe * dim)
