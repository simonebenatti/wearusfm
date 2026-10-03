"""Identita' di canale (passo 6; v10 §3.5, §5.2): embedding anatomico gerarchico additivo, concatenato e proiettato.

    e_anat = E_regione + sum_k w_k E_compartimento[k] + E_muscolo

- **UNK appreso per livello** (indice 0 di ogni vocabolario, `channel_codes`): un livello ignoto non e' un vettore nullo.
- **Etichette soft** (funzione atlante): l'embedding di compartimento e' la somma pesata (v10 §5.2). Un compartimento noto e' un one-hot: un canale
  Myo e uno Delsys sullo stesso compartimento condividono quel vettore, e il muscolo agisce da raffinamento residuo.
- **Dropout del livello muscolo** sui canali dove il muscolo e' noto: il muscolo diventa UNK con probabilita' `muscle_dropout`, solo in
  addestramento. v10 §5.2: «p ≈ 0,3–0,5, da fissare»; qui e' un parametro OBBLIGATORIO, senza valore di default, perche' nessuno lo fissi per
  sbaglio. Il dropout e' per canale (lettura letterale di v10); la variante per campione (tutti i canali di un montaggio insieme) resta possibile.

**Concatenazione** (v10 §5.2, default): [e_anat ; e_topologia ; e_sensore] -> proiezione lineare.
- e_anat comprende anche il **lato** dell'arto (sinistro, destro, ignoto; v10 §3.6, coordinate anatomiche);
- e_sensore = posizione nel sistema del sensore (`channel_codes._sensor_pos`: Fourier dell'angolo sull'anello, di riga e colonna sulla griglia,
  zeri sui canali sparsi) proiettata, piu' l'ordine del gruppo nel montaggio (v10 §3.6, «sistema del sensore»).
**Correzione della review del 03/10.** La prima versione non dava nessuna posizione per canale («encoding puramente relativo», interpretazione di
AG mai confermata): i canali di un anello avevano tutti la stessa identita', e il decoder a query, che vede solo i latenti del Perceiver (senza
canali), dava la stessa predizione per tutti i canali allo stesso istante; su emg2qwerty non distingueva nemmeno la mano sinistra dalla destra.
La posizione in Fourier rende una rotazione della fascia uno sfasamento: la simmetria ciclica e' rappresentabile senza essere imposta (v10 §3.4).
Gli spostamenti relativi con segno restano nel bias dell'encoder locale.
"""

from __future__ import annotations

import torch
from torch import nn

from wearusfm.model.channel_codes import (COMPARTMENT_KEYS, MAX_GROUPS, MUSCLE_KEYS, N_SENSOR_POS, REGION_KEYS, SIDE_KEYS, TOPOLOGIES,
                                         AnatomyCodes)


def codes_to_tensors(codes: AnatomyCodes, device=None) -> dict[str, torch.Tensor]:
    return {
        "region": torch.as_tensor(codes.region, dtype=torch.long, device=device),
        "compartment_weights": torch.as_tensor(codes.compartment_weights, dtype=torch.float32, device=device),
        "muscle": torch.as_tensor(codes.muscle, dtype=torch.long, device=device),
        "muscle_known": torch.as_tensor(codes.muscle_known, dtype=torch.bool, device=device),
        "topology": torch.as_tensor(codes.topology, dtype=torch.long, device=device),
        "side": torch.as_tensor(codes.side, dtype=torch.long, device=device),
        "group": torch.as_tensor(codes.group, dtype=torch.long, device=device),
        "sensor_pos": torch.as_tensor(codes.sensor_pos, dtype=torch.float32, device=device),
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
        self.side = nn.Embedding(len(SIDE_KEYS), dim)

    def forward(self, region: torch.Tensor, compartment_weights: torch.Tensor, muscle: torch.Tensor, muscle_known: torch.Tensor,
                generator: torch.Generator | None = None, side: torch.Tensor | None = None) -> torch.Tensor:
        if self.training and self.muscle_dropout > 0.0:
            dev = generator.device if generator is not None else muscle.device  # il generatore decide il dispositivo dei numeri casuali
            drop = (torch.rand(muscle.shape, generator=generator, device=dev).to(muscle.device) < self.muscle_dropout) & muscle_known
            muscle = torch.where(drop, torch.zeros_like(muscle), muscle)
        e = self.region(region) + compartment_weights.to(self.compartment.weight.dtype) @ self.compartment.weight + self.muscle(muscle)
        return e if side is None else e + self.side(side)


class ChannelIdentity(nn.Module):
    """(C,) codici -> (C, dim): concatenazione [anatomia ; topologia ; sensore] proiettata (v10 §5.2)."""

    def __init__(self, dim: int, *, muscle_dropout: float):
        super().__init__()
        self.anatomy = AnatomicalEmbedding(dim, muscle_dropout=muscle_dropout)
        self.topology = nn.Embedding(len(TOPOLOGIES), dim)
        self.sensor_pos = nn.Linear(N_SENSOR_POS, dim, bias=False)
        self.group = nn.Embedding(MAX_GROUPS, dim)
        self.proj = nn.Linear(3 * dim, dim)

    def forward(self, codes: dict[str, torch.Tensor], generator: torch.Generator | None = None) -> torch.Tensor:
        e_anat = self.anatomy(codes["region"], codes["compartment_weights"], codes["muscle"], codes["muscle_known"], generator, codes["side"])
        e_sensor = self.sensor_pos(codes["sensor_pos"].to(self.sensor_pos.weight.dtype)) + self.group(codes["group"])
        return self.proj(torch.cat([e_anat, self.topology(codes["topology"]), e_sensor], dim=-1))
