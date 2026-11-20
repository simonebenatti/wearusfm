"""Costruisce i moduli dalla specifica di P §3 (3e007c8) SOLO per contare parametri, pesi, bias, MAC e campo ricettivo.
Nessun training, nessun dato del progetto. Gli input usati per il campo ricettivo sono costanti (uni), non dati."""
import torch
from torch import nn

D, J, T = 384, 20, 160

class AffinePlusG(nn.Module):
    def __init__(self, g, channels_first):
        super().__init__()
        self.norm = nn.LayerNorm(D)            # affine, eps=1e-5 di default
        self.g = g
        self.channels_first = channels_first
        self.aff = nn.Linear(D, J)             # ramo affine 384->20 sulla stessa LN
    def forward(self, x):                      # x: (B,T,D)
        h = self.norm(x)
        if self.channels_first:
            gout = self.g(h.transpose(1, 2)).transpose(1, 2)
        else:
            gout = self.g(h)
        return self.aff(h) + gout

def tcn_g():
    return nn.Sequential(
        nn.Conv1d(D, 64, 1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=1, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.ReLU(),
        nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv1d(64, 32, 1), nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv1d(32, J, 1))

def mlp_g():
    return nn.Sequential(
        nn.Linear(D, 96), nn.ReLU(),
        nn.Linear(96, 96), nn.ReLU(),
        nn.Linear(96, 96), nn.ReLU(),
        nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, J))

class L0(nn.Module):
    def __init__(self):
        super().__init__(); self.norm = nn.LayerNorm(D); self.out = nn.Linear(D, J)
    def forward(self, x): return self.out(self.norm(x))

class L13(nn.Module):
    def __init__(self):
        super().__init__(); self.norm = nn.LayerNorm(D); self.out = nn.Conv1d(D, J, 13, padding=6)
    def forward(self, x): return self.out(self.norm(x).transpose(1, 2)).transpose(1, 2)

heads = {
    "L0": L0(), "L13": L13(),
    "MLP": AffinePlusG(mlp_g(), channels_first=False),
    "TCN_post": AffinePlusG(tcn_g(), channels_first=True),
    "TCN_pre": AffinePlusG(tcn_g(), channels_first=True),
}

def counts(m):
    tot = sum(p.numel() for p in m.parameters())
    ln = sum(p.numel() for n, p in m.named_parameters() if n.startswith("norm."))
    w = sum(p.numel() for n, p in m.named_parameters() if n.endswith("weight") and not n.startswith("norm."))
    b = sum(p.numel() for n, p in m.named_parameters() if n.endswith("bias") and not n.startswith("norm."))
    aff = sum(p.numel() for n, p in m.named_parameters() if n.startswith("aff."))
    return tot, ln, w, b, aff

macs = {}
def mac_hook(name):
    def hook(mod, inp, out):
        if isinstance(mod, nn.Linear):
            macs[name] = macs.get(name, 0) + mod.in_features * mod.out_features * out.numel() // (out.shape[0] * mod.out_features)
        elif isinstance(mod, nn.Conv1d):
            Tout = out.shape[-1]
            macs[name] = macs.get(name, 0) + mod.in_channels * mod.out_channels * mod.kernel_size[0] * Tout
    return hook

for name, m in heads.items():
    m.eval()
    hs = [mod.register_forward_hook(mac_hook(name)) for mod in m.modules() if isinstance(mod, (nn.Linear, nn.Conv1d))]
    with torch.no_grad():
        y = m(torch.ones(1, T, D))
    for h in hs: h.remove()
    tot, ln, w, b, aff = counts(m)
    print(f"{name:9s} tot={tot:7d} LN={ln} pesi_trasf={w:6d} bias={b:4d} ramo_aff={aff:5d} MAC/finestra160={macs[name]:9d} out={tuple(y.shape)}")

# campo ricettivo con autograd: pesi positivi (niente cancellazioni), uscita al centro t, input variabile
def rf(m, t, channels_first_out=True):
    m.eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.startswith("norm."):
                continue
            p.copy_(torch.rand_like(p) * 0.1 + 0.01)   # positivi, per contare la dipendenza, non per allenare
    x = (torch.arange(T * D, dtype=torch.float32).reshape(1, T, D) % 7).requires_grad_(True)  # input deterministico non costante per riga
    y = m(x)
    y[0, t].sum().backward()
    dep = (x.grad[0].abs().sum(dim=1) > 0).nonzero().flatten().tolist()
    return dep

for name in ["L13", "TCN_post"]:
    for t in [6, 80, 153, 0, 159]:
        dep = rf(heads[name], t)
        print(f"RF {name} centro {t}: dipende da input {dep[0]}..{dep[-1]} ({len(dep)} posizioni), offset {dep[0]-t}..{dep[-1]-t}")

# dipendenza dal padding: per i centri 6..153 nessuna posizione intermedia fuori [0,159]
def needed_positions(t):
    l3 = {t + o for o in (-3, 0, 3)}                       # input del conv d3 = uscite del conv d2
    l2 = {p + o for p in l3 for o in (-2, 0, 2)}           # input del conv d2 = uscite del conv d1
    l1 = {p + o for p in l2 for o in (-1, 0, 1)}           # input del conv d1 = uscite del conv 1x1
    return l3, l2, l1
ok = all(min(min(s) for s in needed_positions(t)) >= 0 and max(max(s) for s in needed_positions(t)) <= T - 1 for t in range(6, 154))
bad6 = [t for t in range(T) if not (min(min(s) for s in needed_positions(t)) >= 0 and max(max(s) for s in needed_positions(t)) <= T - 1)]
print("TCN: centri senza padding in nessuno strato:", ok, "; centri che toccano il padding:", len(bad6), "=", bad6[:7], "...", bad6[-7:])
print("centri crop6:", len(range(6, 154)))
