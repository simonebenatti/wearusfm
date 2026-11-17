"""Dimensione 1, punto (c): conteggi, MAC, campo ricettivo, crop6/padding, dipendenza dal solo centro."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura")
import torch
from torch import nn
from heads import BUILD, L0, MLP, TCN, L13, D, J, P, last_layer, randomize_last

torch.manual_seed(0)
print("torch", torch.__version__)

# ---------------------------------------------------------------- conteggi
CLAIM = {"L0": 8468, "MLP": 65300, "L13": 100628, "TCN_post": 65204, "TCN_pre": 65204}
print("\n== 1. parametri, pesi nelle trasformazioni, trasformazioni, ReLU, dropout ==")
for name, cls in BUILD.items():
    m = cls()
    n = sum(p.numel() for p in m.parameters())
    tr = [x for x in m.modules() if isinstance(x, (nn.Linear, nn.Conv1d))]
    w = sum(x.weight.numel() for x in tr)
    b = sum(x.bias.numel() for x in tr)
    ln = sum(p.numel() for x in m.modules() if isinstance(x, nn.LayerNorm) for p in x.parameters())
    relu = sum(isinstance(x, nn.ReLU) for x in m.modules())
    drop = [(type(x).__name__, x.p) for x in m.modules() if isinstance(x, (nn.Dropout, nn.Dropout1d))]
    print(f"{name:9s} parametri={n} (dichiarati {CLAIM[name]}, uguali={n == CLAIM[name]}) | pesi={w} bias={b} LN={ln} | trasformazioni={len(tr)} ReLU={relu} dropout={drop}")

# ---------------------------------------------------------------- MAC per finestra
print("\n== 2. MAC per finestra da 160 patch (esclusi bias, LN, ReLU) ==")
def macs(m):
    total = 0
    hooks = []
    def conv_hook(mod, inp, out):
        nonlocal total
        total += out.shape[1] * out.shape[2] * (mod.in_channels // mod.groups) * mod.kernel_size[0]
    def lin_hook(mod, inp, out):
        nonlocal total
        total += mod.in_features * mod.out_features * out.shape[1]  # (B,T,C): una volta per patch
    for x in m.modules():
        if isinstance(x, nn.Conv1d):
            hooks.append(x.register_forward_hook(conv_hook))
        elif isinstance(x, nn.Linear):
            hooks.append(x.register_forward_hook(lin_hook))
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, P, D))
    for h in hooks:
        h.remove()
    return total
for name, cls in BUILD.items():
    print(f"{name:9s} MAC = {macs(cls())}")
print("64128 * 160 =", 64128 * 160)

# ---------------------------------------------------------------- campo ricettivo
print("\n== 3. campo ricettivo: Jacobiano dell'uscita al centro t0=80 rispetto all'ingresso (eval, ultimo strato casuale non nullo) ==")
def jac_profile(m, x, t0):
    """somma su giunti e feature di |d y[t0,j] / d x[t,c]| per ogni t."""
    x = x.clone().requires_grad_(True)
    y = m(x)
    prof = torch.zeros(x.shape[1])
    for j in range(y.shape[2]):
        g, = torch.autograd.grad(y[0, t0, j], x, retain_graph=True)
        prof += g[0].abs().sum(dim=1)
    return prof

for name in ("L0", "MLP", "L13", "TCN_post"):
    offs_all = None
    mass = torch.zeros(P)
    for seed in range(5):
        torch.manual_seed(seed)
        m = randomize_last(BUILD[name](), seed=100 + seed).eval()
        x = torch.randn(1, P, D)
        prof = jac_profile(m, x, 80)
        nz = torch.nonzero(prof > 0).flatten().tolist()
        offs = [t - 80 for t in nz]
        offs_all = set(offs) if offs_all is None else (offs_all & set(offs))
        mass += prof / prof.sum()
    mass /= 5
    offs_sorted = sorted(offs_all)
    print(f"{name:9s} offset con Jacobiano non nullo in tutti e 5 i semi: {offs_sorted[0]}..{offs_sorted[-1]} ({len(offs_sorted)} patch, contigui={offs_sorted == list(range(offs_sorted[0], offs_sorted[-1] + 1))})")
    if name in ("L13", "TCN_post"):
        print("          quota media di |J| per offset -6..+6:", " ".join(f"{float(mass[80 + o]) * 100:.1f}%" for o in range(-6, 7)))
        print("          massa fuori da [-6,+6]:", float(mass.sum() - mass[74:87].sum()))

print("\n== 3b. controprova con perturbazioni: quali patch d'ingresso cambiano l'uscita al centro 80 ==")
for name in ("L0", "MLP", "L13", "TCN_post"):
    torch.manual_seed(7)
    m = randomize_last(BUILD[name](), seed=7).eval()
    with torch.no_grad():
        x0 = torch.randn(1, P, D)
        y0 = m(x0)[0, 80]
        ch = []
        for t in range(P):
            x1 = x0.clone(); x1[0, t] += torch.randn(D)
            ch.append(bool((m(x1)[0, 80] - y0).abs().max() > 0))
    idx = [t - 80 for t, c in enumerate(ch) if c]
    print(f"{name:9s} offset che cambiano l'uscita: {idx}")

print("\n== 3c. struttura esatta (pesi tutti positivi, ingresso positivo: nessuna ReLU spenta) ==")
def make_positive(m):
    with torch.no_grad():
        for x in m.modules():
            if isinstance(x, (nn.Conv1d, nn.Linear)):
                x.weight.copy_(torch.rand_like(x.weight) * 0.01 + 0.001)
                x.bias.fill_(0.01)
    return m
class NoNorm(nn.Module):
    def forward(self, x):
        return x
for name in ("TCN_post", "L13", "MLP", "L0"):
    m = make_positive(BUILD[name]()).eval()
    m.norm = NoNorm()  # la LN per patch non sposta dipendenze fra patch; tolta per tenere positivo l'ingresso
    prof = jac_profile(m, torch.rand(1, P, D) + 0.5, 80)
    nz = [t - 80 for t in torch.nonzero(prof > 0).flatten().tolist()]
    print(f"{name:9s} offset strutturali: {nz}")
# numero di cammini per offset nella TCN: a + 2b + 3c con a,b,c in {-1,0,1}
from itertools import product
from collections import Counter
paths = Counter(a + 2 * b + 3 * c for a, b, c in product((-1, 0, 1), repeat=3))
print("cammini per offset (dilatazioni 1,2,3):", dict(sorted(paths.items())), "totale", sum(paths.values()))
for dil in ((1, 2, 3), (2, 4, 8), (2, 2, 2), (1, 1, 4), (1, 2, 4)):
    reach = sorted({sum(o * d for o, d in zip(c, dil)) for c in product((-1, 0, 1), repeat=3)})
    full = list(range(-sum(dil), sum(dil) + 1))
    print(f"  tre kernel 3 con dilatazioni {dil}: RF nominale {1 + 2 * sum(dil)}, offset raggiunti {len(reach)}, senza buchi={reach == full}")

# ---------------------------------------------------------------- crop6 e padding
print("\n== 4. centri 6..153 e padding ==")
centers = [t for t in range(P) if 6 <= t and t + 6 < P]
print("centri con 6 <= t e t+6 < 160:", centers[0], "..", centers[-1], "numero", len(centers))
for name, cls in (("TCN_post", TCN), ("L13", L13)):
    torch.manual_seed(3)
    ref = randomize_last(cls(), seed=3).eval()
    x = torch.randn(4, P, D)
    with torch.no_grad():
        y_pad = ref(x)
        # stessa rete senza padding: uscita lunga 148
        nopad = cls(padding=False).eval(); nopad.load_state_dict(ref.state_dict())
        y_valid = nopad(x)
        print(f"{name:9s} uscita con padding {tuple(y_pad.shape)}, senza padding {tuple(y_valid.shape)}")
        d = (y_pad[:, 6:154] - y_valid).abs().max().item()
        print(f"          max |con padding[6:154] - senza padding| = {d:.3e}")
        for mode in ("replicate", "reflect", "circular"):
            alt = cls(padding_mode=mode).eval(); alt.load_state_dict(ref.state_dict())
            y_alt = alt(x)
            dc = (y_alt[:, 6:154] - y_pad[:, 6:154]).abs().max().item()
            de = torch.cat([(y_alt[:, :6] - y_pad[:, :6]).abs().flatten(), (y_alt[:, 154:] - y_pad[:, 154:]).abs().flatten()]).max().item()
            per_t = (y_alt - y_pad).abs().amax(dim=(0, 2))
            changed = [t for t in range(P) if per_t[t] > 0]
            print(f"          padding {mode:9s} invece di zeri: max diff sui centri 6..153 = {dc:.3e}; ai bordi = {de:.3e}; centri cambiati: {changed}")

print("\n== 4b. finestra piu' corta (N=100 patch reali, resto riempito): quali centri dipendono dal riempimento ==")
for name, cls in (("TCN_post", TCN), ("L13", L13), ("MLP", MLP), ("L0", L0)):
    torch.manual_seed(4)
    m = randomize_last(cls(), seed=4).eval()
    with torch.no_grad():
        x = torch.randn(1, P, D); N = 100
        xa = x.clone(); xa[0, N:] = 0.0
        xb = x.clone(); xb[0, N:] = torch.randn(P - N, D) * 3
        diff = (m(xa) - m(xb)).abs().amax(dim=2)[0]
        changed = [t for t in range(P) if diff[t] > 0]
        print(f"{name:9s} primo centro che dipende dal riempimento oltre N=100: {changed[0]} (centri ammessi dal crop6: t+6 < 100, cioe' t <= 93)")
