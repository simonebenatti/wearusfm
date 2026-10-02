"""Identita' di canale (passo 6; v10 §3.5, §5.2): embedding anatomico gerarchico additivo, concatenato e proiettato.

    e_anat = E_regione + sum_k w_k E_compartimento[k] + E_muscolo

- **UNK appreso per livello** (indice 0 di ogni vocabolario, `channel_codes`): un livello ignoto non e' un vettore nullo.
- **Etichette soft** (funzione atlante): l'embedding di compartimento e' la somma pesata (v10 §5.2). Un compartimento noto e' un one-hot: un canale
  Myo e uno Delsys sullo stesso compartimento condividono quel vettore, e il muscolo agisce da raffinamento residuo.
- **Dropout del livello muscolo** sui canali dove il muscolo e' noto: il muscolo diventa UNK con probabilita' `muscle_dropout`, solo in
  addestramento. v10 §5.2: «p ≈ 0,3–0,5, da fissare»; qui e' un parametro OBBLIGATORIO, senza valore di default, perche' nessuno lo fissi per
  sbaglio. Il dropout e' per canale (lettura letterale di v10); la variante per campione (tutti i canali di un montaggio insieme) resta possibile.

**Concatenazione** (v10 §5.2, default): [e_anat ; e_topologia] -> proiezione lineare. *Interpretazione di AG, da confermare:* il percorso
«geometrico relativo» NON entra qui come vettore per canale: un angolo assoluto sull'anello romperebbe la simmetria ciclica (v10 §3.4: encoding
«puramente relativo»), e l'orientamento della fascia e' spesso ignoto. Per canale entra solo la classe di topologia (anello, griglia, sparso), che
dice quale struttura relativa vale; gli spostamenti relativi entrano a coppie nell'encoder locale (bias di §5.3 e vicini di §5.4, da
`channel_codes.relative_geometry` e `channel_codes.neighbors`).
"""

from __future__ import annotations

import torch
from torch import nn

from wearusfm.model.channel_codes import COMPARTMENT_KEYS, MUSCLE_KEYS, REGION_KEYS, TOPOLOGIES, AnatomyCodes


def codes_to_tensors(codes: AnatomyCodes, device=None) -> dict[str, torch.Tensor]:
    return {
        "region": torch.as_tensor(codes.region, dtype=torch.long, device=device),
        "compartment_weights": torch.as_tensor(codes.compartment_weights, dtype=torch.float32, device=device),
        "muscle": torch.as_tensor(codes.muscle, dtype=torch.long, device=device),
        "muscle_known": torch.as_tensor(codes.muscle_known, dtype=torch.bool, device=device),
        "topology": torch.as_tensor(codes.topology, dtype=torch.long, device=device),
    }


class AnatomicalEmbedding(nn.Module):
    def __init__(self, dim: int, *, muscle_dropout: float):
        super().__init__()
        if not 0.0 <= muscle_dropout < 1.0:
            raise ValueError(f"muscle_dropout in [0, 1): {muscle_dropout}")
        self.muscle_dropout = float(muscle_dropout)
        self.region = nn.Embedding(len(REGION_KEYS), dim)
        self.compartment = nn.Embedding(len(COMPARTMENT_KEYS), dim)
        self.muscle = nn.Embedding(len(MUSCLE_KEYS), dim)

    def forward(self, region: torch.Tensor, compartment_weights: torch.Tensor, muscle: torch.Tensor, muscle_known: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        if self.training and self.muscle_dropout > 0.0:
            drop = (torch.rand(muscle.shape, generator=generator, device=muscle.device) < self.muscle_dropout) & muscle_known
            muscle = torch.where(drop, torch.zeros_like(muscle), muscle)
        return self.region(region) + compartment_weights.to(self.compartment.weight.dtype) @ self.compartment.weight + self.muscle(muscle)


class ChannelIdentity(nn.Module):
    """(C,) codici -> (C, dim): concatenazione [anatomia ; topologia] proiettata (v10 §5.2)."""

    def __init__(self, dim: int, *, muscle_dropout: float):
        super().__init__()
        self.anatomy = AnatomicalEmbedding(dim, muscle_dropout=muscle_dropout)
        self.topology = nn.Embedding(len(TOPOLOGIES), dim)
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, codes: dict[str, torch.Tensor], generator: torch.Generator | None = None) -> torch.Tensor:
        e_anat = self.anatomy(codes["region"], codes["compartment_weights"], codes["muscle"], codes["muscle_known"], generator)
        return self.proj(torch.cat([e_anat, self.topology(codes["topology"])], dim=-1))
