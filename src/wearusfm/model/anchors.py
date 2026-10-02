"""Teste e perdite delle ancore (passo 6; v10 §6.2-6.3): leggono le uscite del decoder a query (§5.6), sono piccole (lineari: «testa piccola»),
e le perdite contano solo dove il target esiste (maschere di `anchor_targets`). Il peso complessivo delle ancore (0,1-0,3, v10 §6.2) si applica nel
ciclo di training, non qui.

- RMS, forma spettrale, inviluppo: regressione (errore quadratico) per query (canale, patch); le bande non disponibili non contano.
- RVQ (D5b: livello 0 del ramo 0, 8192 codici): classificazione per (canale, finestra da 200 ms interamente nascosta) sulla media delle uscite del
  decoder alle 8 patch della finestra (*scelta di AG, da confermare*: una query per patch, poi la media, cosi' non serve un tipo di query diverso).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

N_RVQ_CODES = 8192


class AnchorHeads(nn.Module):
    def __init__(self, dim: int, n_bands: int = 5):
        super().__init__()
        self.rms, self.bands, self.env = nn.Linear(dim, 1), nn.Linear(dim, n_bands), nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"log_rms": self.rms(h)[..., 0], "band_shape": self.bands(h), "log_env": self.env(h)[..., 0]}


def anchor_losses(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """pred/target per query: log_rms (Q,), band_shape (Q, B), log_env (Q,); maschere rms_valid, spec_valid (Q,), band_available (Q, B),
    env_valid (Q,). Media sugli elementi validi; zero (con gradiente) se non ce n'e' nessuno."""
    out = {}
    for name, mask in (("log_rms", target["rms_valid"]), ("log_env", target["env_valid"])):
        err = (pred[name] - target[name]) ** 2
        out[name] = (err * mask).sum() / mask.sum().clamp(min=1)
    m = target["spec_valid"][:, None] & target["band_available"]
    err = (pred["band_shape"] - target["band_shape"]) ** 2
    out["band_shape"] = (err * m).sum() / m.sum().clamp(min=1)
    return out


class RVQHead(nn.Module):
    def __init__(self, dim: int, n_codes: int = N_RVQ_CODES, patches_per_token: int = 8):
        super().__init__()
        self.per = patches_per_token
        self.proj = nn.Linear(dim, n_codes)

    def forward(self, h_patches: torch.Tensor) -> torch.Tensor:
        """h_patches: (N, 8, dim) le uscite del decoder alle 8 patch di ciascuna finestra con target -> logit (N, n_codes)."""
        if h_patches.shape[1] != self.per:
            raise ValueError(f"servono {self.per} patch per finestra, non {h_patches.shape[1]}")
        return self.proj(h_patches.mean(dim=1))


def rvq_loss(logits: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, codes) if logits.shape[0] else logits.sum() * 0.0
