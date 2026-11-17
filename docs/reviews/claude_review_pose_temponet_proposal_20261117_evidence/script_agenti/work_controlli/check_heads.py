"""Controllo ESEGUITO: parametri, pesi, MAC e campo ricettivo delle cinque teste proposte (protocollo c15aac9, par. 3).
Nessun dato reale: solo moduli torch costruiti dalla specifica scritta nel protocollo."""
import torch
from torch import nn

torch.manual_seed(0)
D, J, P = 384, 20, 160


class Head(nn.Module):
    """LN per patch -> rete conv1d su (B, D, T)."""
    def __init__(self, body):
        super().__init__()
        self.norm = nn.LayerNorm(D, eps=1e-5)
        self.body = body

    def forward(self, x):  # x: (B, T, D)
        return self.body(self.norm(x).transpose(1, 2)).transpose(1, 2)


def tcn():
    return Head(nn.Sequential(
        nn.Conv1d(D, 64, 1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=1, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.ReLU(), nn.Dropout(0.1),
        nn.Conv1d(64, 32, 1), nn.ReLU(), nn.Dropout(0.1),
        nn.Conv1d(32, J, 1)))


def mlp():
    w = [D, 96, 96, 96, 64, 32, J]
    layers = []
    for i in range(6):
        layers.append(nn.Conv1d(w[i], w[i + 1], 1))
        if i < 5:
            layers.append(nn.ReLU())
        if i in (3, 4):  # dropout dopo gli hidden da 64 e 32
            layers.append(nn.Dropout(0.1))
    return Head(nn.Sequential(*layers))


def l0():
    return Head(nn.Conv1d(D, J, 1))


def l13():
    return Head(nn.Conv1d(D, J, 13, padding=6))


def counts(m):
    total = sum(p.numel() for p in m.parameters())
    weights = sum(p.numel() for n, p in m.named_parameters() if n.startswith("body") and p.ndim > 1)
    biases = sum(p.numel() for n, p in m.named_parameters() if n.startswith("body") and p.ndim == 1)
    ln = sum(p.numel() for n, p in m.named_parameters() if n.startswith("norm"))
    macs = sum(c.weight.numel() * P for c in m.modules() if isinstance(c, nn.Conv1d))
    return total, weights, biases, ln, macs


def rf(m):
    """Offset delle patch d'ingresso da cui dipende l'uscita al centro (gradiente non nullo), con pesi casuali non nulli."""
    m.eval()
    for p in m.parameters():
        if p.ndim > 1:
            nn.init.normal_(p, std=0.5)
        else:
            nn.init.normal_(p, std=0.1)
    x = torch.randn(1, P, D, requires_grad=True)
    y = m(x)
    t = 80
    g = torch.autograd.grad(y[0, t].sum(), x)[0][0]  # (P, D)
    nz = (g.abs().sum(dim=1) > 0).nonzero().flatten() - t
    return nz.tolist()


def crop_equiv(make):
    """I centri 6..153 non dipendono dal padding: uscita su 160 patch == uscita senza padding sugli stessi centri."""
    m = make().eval()
    for p in m.parameters():
        nn.init.normal_(p, std=0.3)
    x = torch.randn(2, P, D)
    full = m(x)[:, 6:-6]
    # stessa rete senza padding
    import copy
    m2 = copy.deepcopy(m)
    for c in m2.modules():
        if isinstance(c, nn.Conv1d):
            c.padding = (0,)
    nopad = m2(x)
    return tuple(full.shape), tuple(nopad.shape), float((full - nopad).abs().max())


for name, make in (("L0", l0), ("MLP", mlp), ("L13", l13), ("TCN", tcn)):
    m = make()
    total, weights, biases, ln, macs = counts(m)
    offs = rf(make())
    print(f"{name}: parametri {total} (pesi {weights}, bias {biases}, LN {ln}); MAC per finestra {macs}; "
          f"offset d'ingresso con gradiente non nullo: {offs[0]}..{offs[-1]} ({len(offs)} patch, contigui={offs == list(range(offs[0], offs[-1] + 1))})")
for name, make in (("L13", l13), ("TCN", tcn)):
    print(name, "crop6 contro rete senza padding (forme, scarto massimo):", crop_equiv(make))
