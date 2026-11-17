"""Le cinque teste del protocollo docs/pose_temponet_frozen_protocol_20261117.md (paragrafo 3) costruite alla lettera.
Scritto dal revisore per i controlli: NON e' il codice del progetto (che non esiste ancora)."""
import torch
from torch import nn

D, J, P = 384, 20, 160
EPS = 1e-5


def zero_(layer):
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()


class L0(nn.Module):
    """LN384 -> Linear(384,20)."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=EPS)
        self.out = nn.Linear(D, J)
        zero_(self.out)

    def forward(self, x):  # (B,T,D)
        return self.out(self.norm(x))


class MLP(nn.Module):
    """LN -> 384 -> 96 -> 96 -> 96 -> 64 -> 32 -> 20, ReLU sui cinque hidden, dropout 0,1 dopo i hidden da 64 e 32."""
    WIDTHS = (384, 96, 96, 96, 64, 32, 20)

    def __init__(self, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=EPS)
        layers = []
        w = self.WIDTHS
        for i in range(6):
            layers.append(nn.Linear(w[i], w[i + 1]))
            if i < 5:
                layers.append(nn.ReLU())
            if i in (3, 4):  # dopo il quarto hidden (64) e il quinto (32)
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        zero_(self.net[-1])

    def forward(self, x):
        return self.net(self.norm(x))


class TCN(nn.Module):
    """LN384 -> Conv1x1(384,64) -> ReLU -> Conv k3,d1 -> ReLU -> Conv k3,d2 -> ReLU -> Conv k3,d3 -> ReLU -> Dropout0.1
    -> Conv1x1(64,32) -> ReLU -> Dropout0.1 -> Conv1x1(32,20)."""
    def __init__(self, dropout=0.1, padding=True, padding_mode="zeros", dropout_cls=nn.Dropout):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=EPS)
        pad = (lambda d: d) if padding else (lambda d: 0)
        self.net = nn.Sequential(
            nn.Conv1d(D, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, dilation=1, padding=pad(1), padding_mode=padding_mode), nn.ReLU(),
            nn.Conv1d(64, 64, 3, dilation=2, padding=pad(2), padding_mode=padding_mode), nn.ReLU(),
            nn.Conv1d(64, 64, 3, dilation=3, padding=pad(3), padding_mode=padding_mode), nn.ReLU(),
            dropout_cls(dropout),
            nn.Conv1d(64, 32, 1), nn.ReLU(),
            dropout_cls(dropout),
            nn.Conv1d(32, J, 1))
        zero_(self.net[-1])

    def forward(self, x):  # (B,T,D) -> (B,T',J)
        return self.net(self.norm(x).transpose(1, 2)).transpose(1, 2)


class L13(nn.Module):
    """LN -> Conv(384,20,k13,pad6)."""
    def __init__(self, padding=True, padding_mode="zeros"):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=EPS)
        self.conv = nn.Conv1d(D, J, 13, padding=6 if padding else 0, padding_mode=padding_mode)
        zero_(self.conv)

    def forward(self, x):
        return self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)


def last_layer(m):
    if isinstance(m, L0):
        return m.out
    if isinstance(m, L13):
        return m.conv
    return m.net[-1]


def randomize_last(m, seed=123, scale=0.1):
    g = torch.Generator().manual_seed(seed)
    last = last_layer(m)
    with torch.no_grad():
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * scale)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g) * scale)
    return m


BUILD = {"L0": L0, "MLP": MLP, "L13": L13, "TCN_post": TCN, "TCN_pre": TCN}
