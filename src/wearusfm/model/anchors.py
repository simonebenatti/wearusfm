"""Teste e perdite delle ancore (passo 6; v10 §6.2-6.3): leggono le uscite del decoder a query (§5.6), sono piccole (lineari: «testa piccola»),
e le perdite contano solo dove il target esiste (maschere di `anchor_targets`). Il peso complessivo delle ancore (0,1-0,3, v10 §6.2) si applica nel
ciclo di training, non qui.

- RMS, forma spettrale, inviluppo: regressione (errore quadratico) per query (canale, patch); le bande non disponibili non contano.
- RVQ (D5b: livello 0 del ramo 0, 8192 codici): classificazione per (canale, finestra da 200 ms interamente nascosta) sulla media delle uscite del
  decoder alle 8 patch della finestra (*scelta di AG, da confermare*: una query per patch, poi la media, cosi' non serve un tipo di query diverso).
  **Entropia incrociata divisa per ln(8192)** (*scelta di AG del 04/10, da confermare*): all'inizio vale ~1 come le altre ancore (~0,5-2,5), invece
  di ~9 che dominerebbe il peso complessivo delle ancore (0,2, firmato). Un codice -1 (nessun target: sessione senza codici, oltre la vista
  canonica) non conta.
"""

from __future__ import annotations

import math

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


def masked_mean(num: torch.Tensor, den: torch.Tensor, sample_index: torch.Tensor | None = None, n_samples: int | None = None) -> torch.Tensor:
    """num, den: (Q,) somma degli errori validi e numero di elementi validi per query. Senza `sample_index`: sum(num) / sum(den), cioe' la media
    per elemento (per token). Con `sample_index` (Q,) e `n_samples`: la media dentro ogni campione (finestra), poi la media fra i campioni che hanno
    almeno un elemento valido (Simone, 04/10/2026, decisione 2: la perdita rispetta le quote delle finestre, non il numero di token). Zero (con
    gradiente) se non c'e' nessun elemento valido."""
    if sample_index is None:
        return num.sum() / den.sum().clamp(min=1)
    s_num = num.new_zeros(n_samples).index_add(0, sample_index, num)
    s_den = den.new_zeros(n_samples, dtype=num.dtype).index_add(0, sample_index, den.to(num.dtype))
    ok = s_den > 0
    return (s_num[ok] / s_den[ok]).mean() if bool(ok.any()) else num.sum() * 0.0


def anchor_losses(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], sample_index: torch.Tensor | None = None,
                  n_samples: int | None = None) -> dict[str, torch.Tensor]:
    """pred/target per query: log_rms (Q,), band_shape (Q, B), log_env (Q,); maschere rms_valid, spec_valid (Q,), band_available (Q, B),
    env_valid (Q,). Media sugli elementi validi (per token, oppure per finestra con `sample_index`, vedi `masked_mean`); zero (con gradiente) se
    non ce n'e' nessuno."""
    out = {}
    for name, mask in (("log_rms", target["rms_valid"]), ("log_env", target["env_valid"])):
        err = (pred[name] - target[name]) ** 2
        out[name] = masked_mean(err * mask, mask, sample_index, n_samples)
    m = target["spec_valid"][:, None] & target["band_available"]
    err = (pred["band_shape"] - target["band_shape"]) ** 2
    out["band_shape"] = masked_mean((err * m).sum(dim=-1), m.sum(dim=-1), sample_index, n_samples)
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


def rvq_loss(logits: torch.Tensor, codes: torch.Tensor, sample_index: torch.Tensor | None = None, n_samples: int | None = None) -> torch.Tensor:
    """Entropia incrociata sui codici validi (>= 0), divisa per ln(numero di codici); media per finestra RVQ, oppure per campione con
    `sample_index` (`masked_mean`); zero (con gradiente) se non ce n'e' nessuno."""
    ok = codes >= 0
    if not logits.shape[0] or not bool(ok.any()):
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, codes.clamp(min=0), reduction="none") / math.log(logits.shape[-1])
    return masked_mean(ce * ok, ok, sample_index, n_samples)
