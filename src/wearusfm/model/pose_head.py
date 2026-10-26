"""Testa di posa (R3, seconda ondata del pilot degli obiettivi, firmata il 26/10/2026; `docs/pilot_posa_v1.md` §2).

Legge, a ogni patch temporale di un campione, la **media sui K latenti del backbone** `z[s, p]` (la parte del backbone del readout `pool_p1p2`
che il criterio legge, per istante) e predice i 20 angoli articolari standardizzati: `LayerNorm(d) -> Linear(d, 20)`, **lineare** (il criterio e'
la leggibilita' lineare del readout medio) e **inizializzata a zero** (pesi e bias): all'avvio predice la media standardizzata, quindi il termine
di posa parte esattamente da 1 per elemento (cancello del collaudo, §2 (iii)). Il generatore casuale globale non si sposta (`fork_rng`): un
modello con la testa nasce con gli stessi pesi, bit per bit, di uno senza, e lo stato del generatore dopo e' lo stesso (test di
`test_pose_objective.py` e `test_readout_reference.py`).
La testa NON sta nel percorso di `encode`/`decode`: l'estrazione del readout non cambia.
"""

from __future__ import annotations

import torch
from torch import nn

N_JOINTS = 20


class PoseHead(nn.Module):
    def __init__(self, dim: int, n_joints: int = N_JOINTS):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # `nn.Linear` estrae pesi casuali prima dell'azzeramento: sotto `fork_rng` il generatore globale resta com'era (critico del 26/10,
        # rilievo 8), cosi' il dropout del muscolo a un processo non si sposta rispetto a un modello senza testa
        with torch.random.fork_rng(devices=[]):
            self.out = nn.Linear(dim, n_joints)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, z_mean: torch.Tensor) -> torch.Tensor:
        """z_mean: (S, P, d) media sui latenti per istante -> (S, P, n_joints)."""
        return self.out(self.norm(z_mean))


def latent_mean(z: torch.Tensor) -> torch.Tensor:
    """(S, P, K, d) -> (S, P, d): la media sui K latenti, la stessa di `spectral_step1.pool_p1p2` prima della media sul tempo."""
    return z.mean(dim=2)
