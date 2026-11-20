"""Dimensione 1, punti (d) ed (e): la specifica e' unica? L0 e' la PoseHead?"""
import sys
W = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview"
sys.path.insert(0, W + "/work_architettura")
sys.path.insert(0, W + "/gpt_c15aac9/src")  # codice dello snapshot di GPT (sola lettura)
import torch
from torch import nn
from heads import L0, MLP, TCN, L13, D, J, P, zero_

from wearusfm.model import pose_head as PH
print("pose_head importato da:", PH.__file__)

print("\n== (e) L0 contro PoseHead dello snapshot ==")
torch.manual_seed(0)
ph = PH.PoseHead(D)
l0 = L0()
print("PoseHead: norm =", ph.norm, "| out =", ph.out)
print("  eps =", ph.norm.eps, "| elementwise_affine =", ph.norm.elementwise_affine, "| gamma tutta 1:", bool((ph.norm.weight == 1).all()), "| beta tutta 0:", bool((ph.norm.bias == 0).all()))
print("  pesi out tutti zero:", bool((ph.out.weight == 0).all()), "| bias out tutti zero:", bool((ph.out.bias == 0).all()))
print("  parametri PoseHead =", sum(p.numel() for p in ph.parameters()), "| parametri L0 =", sum(p.numel() for p in l0.parameters()))
print("  stesse chiavi/forme dello state_dict:", {k: tuple(v.shape) for k, v in ph.state_dict().items()} == {k: tuple(v.shape) for k, v in l0.state_dict().items()})
# stessa funzione a pari pesi
g = torch.Generator().manual_seed(5)
with torch.no_grad():
    for p in ph.parameters():
        p.copy_(torch.randn(p.shape, generator=g))
l0.load_state_dict(ph.state_dict())
x = torch.randn(3, P, D, generator=g)
print("  max |PoseHead(x) - L0(x)| a pari pesi =", float((ph(x) - l0(x)).abs().max()))
# la PoseHead non sposta il generatore globale (fork_rng)
torch.manual_seed(42); a = torch.rand(1).item()
torch.manual_seed(42); PH.PoseHead(D); b = torch.rand(1).item()
torch.manual_seed(42); L0(); c = torch.rand(1).item()
print(f"  primo numero casuale dopo il seme 42: senza testa {a:.6f}; dopo PoseHead {b:.6f} (uguale={a == b}); dopo L0 costruita con nn.Linear normale {c:.6f} (uguale={a == c})")

print("\n== (e) normalizzazione della ridge storica: LN senza stato, eps 1e-5 ==")
import torch.nn.functional as F
xx = torch.randn(2, 5, D)
stateless = F.layer_norm(xx, (D,), eps=1e-5)
affine_init = nn.LayerNorm(D, eps=1e-5)(xx)
print("  LN affine all'inizializzazione (gamma=1, beta=0) contro LN senza stato: max diff =", float((stateless - affine_init).abs().max()))
print("  LN di una patch tutta zero: senza stato ->", float(F.layer_norm(torch.zeros(1, D), (D,), eps=1e-5).abs().max()), "; affine con beta=0.3 ->", end=" ")
ln = nn.LayerNorm(D, eps=1e-5); nn.init.constant_(ln.bias, 0.3)
print(float(ln(torch.zeros(1, D)).abs().max()))

print("\n== (d1) dropout per elemento o per canale ==")
torch.manual_seed(0)
h = torch.ones(2, 64, 160)  # (B,C,T) come nella TCN
de = nn.Dropout(0.1).train()(h)
dc = nn.Dropout1d(0.1).train()(h)
print("  nn.Dropout   su (B,C,T): elementi azzerati %.3f; canali azzerati per intero su tutte le 160 patch: %d su %d" % (float((de == 0).float().mean()), int((de == 0).all(dim=2).sum()), 2 * 64))
print("  nn.Dropout1d su (B,C,T): elementi azzerati %.3f; canali azzerati per intero su tutte le 160 patch: %d su %d" % (float((dc == 0).float().mean()), int((dc == 0).all(dim=2).sum()), 2 * 64))
hm = torch.ones(2, 160, 64)  # (B,T,C) come nella MLP con nn.Linear
dm = nn.Dropout1d(0.1).train()(hm)
print("  nn.Dropout1d su (B,T,C) (MLP con Linear): azzera intere PATCH (asse 1 trattato come canali): patch azzerate per intero %d su %d" % (int((dm == 0).all(dim=2).sum()), 2 * 160))

print("\n== (d2) ordine di creazione dei moduli a pari seme ==")
def tcn_first_weights(order):
    torch.manual_seed(0)
    if order == "ingresso->uscita":
        m = TCN()
        return m.net[0].weight.detach().clone(), m.net[2].weight.detach().clone()
    # uscita creata per prima, poi gli altri strati
    out = nn.Conv1d(32, J, 1)
    c0 = nn.Conv1d(D, 64, 1)
    c1 = nn.Conv1d(64, 64, 3, padding=1)
    return c0.weight.detach().clone(), c1.weight.detach().clone()
a0, a1 = tcn_first_weights("ingresso->uscita")
b0, b1 = tcn_first_weights("uscita per prima")
print("  seme 0, primo strato 384->64: max |diff| fra i due ordini =", float((a0 - b0).abs().max()), "| frazione di pesi diversi =", float((a0 != b0).float().mean()))
torch.manual_seed(0); lin = nn.Linear(D, 64)
torch.manual_seed(0); conv = nn.Conv1d(D, 64, 1)
print("  seme 0, nn.Linear(384,64) contro nn.Conv1d(384,64,1): pesi identici =", bool(torch.equal(lin.weight, conv.weight[:, :, 0])), "| bias identici =", bool(torch.equal(lin.bias, conv.bias)))
# LN creata prima o dopo: nessun consumo del generatore
torch.manual_seed(0); nn.LayerNorm(D); c_after_ln = nn.Conv1d(D, 64, 1)
print("  LN creata prima dello strato: pesi invariati =", bool(torch.equal(c_after_ln.weight, conv.weight)))

print("\n== (d3) che cosa cambia il seme in L0 e L13 ==")
for name, cls in (("L0", L0), ("L13", L13), ("MLP", MLP), ("TCN", TCN)):
    sd = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        sd.append(torch.cat([p.detach().flatten() for p in cls().parameters()]))
    same = bool(torch.equal(sd[0], sd[1]) and torch.equal(sd[1], sd[2]))
    print(f"  {name:4s} parametri iniziali identici nei semi 0,1,2: {same}; ha dropout: {any(isinstance(m, nn.Dropout) for m in cls().modules())}")

print("\n== (d4) LN applicata nel posto sbagliato fallisce, non cambia in silenzio ==")
try:
    nn.LayerNorm(D)(torch.zeros(2, D, P))
    print("  LayerNorm(384) su (B,384,160): accettata")
except Exception as e:
    print("  LayerNorm(384) su (B,384,160): errore ->", str(e)[:110])
gn = nn.GroupNorm(1, D)
xg = torch.randn(1, D, P)
yg = gn(xg)
print("  GroupNorm(1,384) su (B,384,160) NON e' per patch: deviazione standard per patch dopo la norma, min %.3f max %.3f (per patch sarebbe 1.000)" % (float(yg.std(dim=1).min()), float(yg.std(dim=1).max())))

print("\n== (d5) lo stato del generatore dopo la costruzione dipende da come si azzera l'ultimo strato ==")
def after(build):
    torch.manual_seed(0); build(); return torch.rand(1).item()
def tcn_fork():
    m = TCN.__new__(TCN); nn.Module.__init__(m)
    m.norm = nn.LayerNorm(D)
    layers = [nn.Conv1d(D, 64, 1), nn.Conv1d(64, 64, 3, padding=1), nn.Conv1d(64, 64, 3, dilation=2, padding=2), nn.Conv1d(64, 64, 3, dilation=3, padding=3), nn.Conv1d(64, 32, 1)]
    with torch.random.fork_rng(devices=[]):
        layers.append(nn.Conv1d(32, J, 1))
    return m
print("  primo numero casuale dopo TCN() normale: %.6f; dopo TCN con ultimo strato sotto fork_rng (come PoseHead): %.6f" % (after(TCN), after(tcn_fork)))
