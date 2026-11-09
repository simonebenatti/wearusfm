"""TEMPONet per la regressione della cinematica della mano da sEMG (baseline «CNN/TCN da zero», debito dell'harness v10).

Fonti (lette il 09/11/2026; dettagli e scelte in `docs/temponet_db8_replication.md`):
- [Z21] M. Zanghieri et al., «sEMG-based Regression of Hand Kinematics with Temporal Convolutional Networks on a Low-Power Edge
  Microcontroller», IEEE COINS 2021, DOI 10.1109/COINS51742.2021.9524188 (postprint https://hdl.handle.net/11585/841320), sez. II-C e Fig. 1;
- [Z20] M. Zanghieri et al., «Robust Real-Time Embedded EMG Recognition Framework Using Temporal Convolutional Networks on a Multicore IoT
  Processor», IEEE TBioCAS 14(2), 2020, sez. III-B1 e Fig. 3 (TEMPONet originale, classificazione);
- implementazione PyTorch di TEMPONet di un coautore (Burrello) per la PPG: github.com/EmbeddedML-EDAGroup/Q-PPG, commit
  ddf3866da6d5f9dda4da7d7884b4f1f3b809a6ba, `precision_search/model/TCN_variants.py`, classe `TCN_network_float` (ordine dei layer).

Architettura di [Z21] (Fig. 1 e testo): tre blocchi convoluzionali, ciascuno con due convoluzioni temporali dilatate (kernel 3, dilatazione 2, 4, 8,
«full padding») e una convoluzione con kernel 5 e stride 1, 2, 4 seguita da average pooling (kernel 2, stride 2); canali 16, 32 e 64 (Block III a 64,
la scelta migliore di Tab. I); poi tre strati densi FC I, FC II (32 unita') e FC III (5 unita', le DoA). BatchNorm e ReLU dopo ogni strato tranne
FC III; dropout 0,5 su FC I e FC II. Ingresso: 16 canali x 256 campioni (128 ms a 2 kHz), segnale grezzo.

Due punti che il testo lascia ambigui, risolti con i numeri dello stesso paper (dettagli nel documento):
- tutte e tre le convoluzioni di un blocco hanno lo stesso numero di canali (16, 32, 64);
- FC I ha 64 unita' (= canali del Block III) e riceve 4 x 64 ingressi (la lunghezza temporale dopo il terzo blocco e' 4). La frase «FC I has 4x
  units as Block III's channels» si legge come riferita all'ingresso: con 4 x 64 = 256 unita' il modello avrebbe 125.504 pesi invece dei 70,9 kB
  int8 dichiarati e, con Block III a 192, superebbe i «less than 500 k parameters» della sez. II-C2.
Con queste due letture il conteggio dei parametri addestrabili (convoluzioni e FC I/II senza bias perche' seguite da BatchNorm, FC III con bias,
BatchNorm con gamma e beta) e' 70.917, cioe' i 70,9 kB int8 di [Z21]; con Block III a 192 e' 481.541 (< 500 k). Il test CPU lo verifica.

Convoluzioni dilatate causali ([Z21] eq. 1 e testo: «2 dilated causal convolutions [...] full padding»): padding (k-1)*d solo a sinistra, la
lunghezza resta invariata. Q-PPG usa padding simmetrico: e' disponibile con `causal=False`. Convoluzione con stride: padding 2, «same» per stride 1.
Ordine nel blocco strided come Q-PPG: Conv -> AvgPool -> BatchNorm -> ReLU (in inferenza pool e BN commutano: sono lineari per canale).
"""

from __future__ import annotations

import torch
from torch import nn

# [Z21] sez. II-C2
DEFAULT_BLOCK_CHANNELS = (16, 32, 64)
DEFAULT_DILATIONS = (2, 4, 8)
DEFAULT_STRIDES = (1, 2, 4)
DEFAULT_FC_UNITS = (64, 32)
TEMPORAL_KERNEL = 3
STRIDED_KERNEL = 5
STRIDED_PADDING = 2
POOL = 2


class _TemporalConv(nn.Module):
    """Conv1d dilatata (kernel 3, senza bias) -> BatchNorm -> ReLU; lunghezza invariata."""

    def __init__(self, ch_in: int, ch_out: int, dilation: int, causal: bool):
        super().__init__()
        total = (TEMPORAL_KERNEL - 1) * dilation
        left, right = (total, 0) if causal else (total // 2, total - total // 2)
        self.pad = nn.ConstantPad1d((left, right), 0.0)
        self.conv = nn.Conv1d(ch_in, ch_out, TEMPORAL_KERNEL, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(self.pad(x))))


class _StridedConv(nn.Module):
    """Conv1d (kernel 5, stride s, padding 2, senza bias) -> AvgPool(2) -> BatchNorm -> ReLU."""

    def __init__(self, ch_in: int, ch_out: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, STRIDED_KERNEL, stride=stride, padding=STRIDED_PADDING, bias=False)
        self.pool = nn.AvgPool1d(POOL, stride=POOL)
        self.bn = nn.BatchNorm1d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pool(self.conv(x))))


class _Dense(nn.Module):
    """Linear (senza bias) -> BatchNorm -> ReLU -> Dropout."""

    def __init__(self, ft_in: int, ft_out: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(ft_in, ft_out, bias=False)
        self.bn = nn.BatchNorm1d(ft_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.relu(self.bn(self.fc(x))))


def output_length(window: int, strides=DEFAULT_STRIDES) -> int:
    """Lunghezza temporale dopo i tre blocchi (le convoluzioni dilatate non la cambiano)."""
    length = window
    for s in strides:
        length = (length + 2 * STRIDED_PADDING - STRIDED_KERNEL) // s + 1
        length //= POOL
    return length


def receptive_field(dilations=DEFAULT_DILATIONS, strides=DEFAULT_STRIDES) -> int:
    """Campi ricettivi in campioni di un'uscita del terzo blocco (378 con i valori di default; [Z20] Fig. 3 riporta 377)."""
    rf, jump = 1, 1
    for d, s in zip(dilations, strides):
        rf += 2 * (TEMPORAL_KERNEL - 1) * d * jump
        rf += (STRIDED_KERNEL - 1) * jump
        jump *= s
        rf += (POOL - 1) * jump
        jump *= POOL
    return rf


class TEMPONet(nn.Module):
    """TEMPONet per regressione ([Z21]): (batch, canali, campioni) -> (batch, uscite)."""

    def __init__(
        self,
        in_channels: int = 16,
        window: int = 256,
        out_dim: int = 5,
        block_channels=DEFAULT_BLOCK_CHANNELS,
        fc_units=DEFAULT_FC_UNITS,
        dilations=DEFAULT_DILATIONS,
        strides=DEFAULT_STRIDES,
        dropout: float = 0.5,
        causal: bool = True,
    ):
        super().__init__()
        if not (len(block_channels) == len(dilations) == len(strides)):
            raise ValueError("block_channels, dilations e strides devono avere la stessa lunghezza")
        self.in_channels, self.window, self.out_dim = int(in_channels), int(window), int(out_dim)
        layers: list[nn.Module] = []
        ch = self.in_channels
        for c, d, s in zip(block_channels, dilations, strides):
            layers += [_TemporalConv(ch, c, d, causal), _TemporalConv(c, c, d, causal), _StridedConv(c, c, s)]
            ch = c
        self.features = nn.Sequential(*layers)
        self.out_length = output_length(self.window, strides)
        if self.out_length < 1:
            raise ValueError(f"finestra di {window} campioni troppo corta per gli stride {strides}")
        dense: list[nn.Module] = []
        ft = ch * self.out_length
        for units in fc_units:
            dense.append(_Dense(ft, units, dropout))
            ft = units
        self.head = nn.Sequential(*dense)
        self.out = nn.Linear(ft, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[2] != self.window:
            raise ValueError(f"ingresso atteso (batch, {self.in_channels}, {self.window}), ricevuto {tuple(x.shape)}")
        return self.out(self.head(self.features(x).flatten(1)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: TEMPONet) -> int:
    """MAC di convoluzioni e strati densi per una finestra (stride applicato: [Z21] riporta 3,16 MMAC, conteggio non specificato)."""
    macs = 0
    x = torch.zeros(1, model.in_channels, model.window)
    hooks = []

    def conv_hook(mod, inp, out):
        nonlocal macs
        macs += out.shape[1] * out.shape[2] * (mod.in_channels // mod.groups) * mod.kernel_size[0]

    def lin_hook(mod, inp, out):
        nonlocal macs
        macs += mod.in_features * mod.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(x)
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs
