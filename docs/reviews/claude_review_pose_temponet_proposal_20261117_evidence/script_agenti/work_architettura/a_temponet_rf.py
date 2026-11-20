"""Dimensione 1, punto (a): TEMPONet esistente (ramo Claude 8917e2a) con 384 canali e T=160.
Solo lettura del codice del worktree; nessuna scrittura fuori da work_architettura."""
import sys
sys.path.insert(0, "/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/src")
import torch
from torch import nn
from wearusfm.model import temponet as tn

torch.manual_seed(0)
print("file:", tn.__file__)
print("receptive_field() =", tn.receptive_field())
for T in (37, 38, 101, 102, 159, 160, 256):
    try:
        print(f"output_length({T}) =", tn.output_length(T))
    except Exception as e:
        print(T, "ERR", e)

# jump finale (distanza in passi d'ingresso fra due elementi adiacenti dell'uscita del terzo blocco)
jump = 1
for s in tn.DEFAULT_STRIDES:
    jump *= s * tn.POOL
print("jump finale =", jump, "-> unione nominale di 2 elementi =", tn.receptive_field() + jump)


def positive_structural(model):
    """Rende tutti i pesi positivi e la BN neutra: con ingresso positivo ogni ReLU e' attiva,
    quindi il gradiente e' non nullo esattamente dove c'e' dipendenza strutturale."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                m.weight.copy_(torch.rand_like(m.weight) * 0.01 + 0.001)
                if m.bias is not None:
                    m.bias.fill_(0.01)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.fill_(1.0); m.bias.fill_(0.0)
                m.running_mean.zero_(); m.running_var.fill_(1.0)


def dep_range(g):
    idx = torch.nonzero(g.abs().sum(dim=(0, 1)) > 0).flatten().tolist()
    if not idx:
        return None
    contiguous = idx == list(range(idx[0], idx[-1] + 1))
    return idx[0], idx[-1], len(idx), contiguous


for causal in (True, False):
    print(f"\n=== causal={causal}, in_channels=384, window=160, out_dim=20 ===")
    m = tn.TEMPONet(in_channels=384, window=160, out_dim=20, causal=causal)
    print("parametri:", tn.count_parameters(m), " out_length:", m.out_length)
    m.eval()
    positive_structural(m)
    x = (torch.rand(1, 384, 160) + 0.5).requires_grad_(True)
    feat = m.features(x)
    print("uscita del terzo blocco:", tuple(feat.shape))
    for j in range(feat.shape[2]):
        g, = torch.autograd.grad(feat[0, :, j].sum(), x, retain_graph=True)
        print(f"  elemento {j} del terzo blocco dipende dagli ingressi (primo, ultimo, quanti, contigui):", dep_range(g))
    y = m(x)
    print("uscita della rete:", tuple(y.shape))
    g, = torch.autograd.grad(y.sum(), x)
    print("  uscita finale (somma dei 20 angoli) dipende dagli ingressi:", dep_range(g))
    # controllo con perturbazione, pesi casuali di default (non resi positivi)
    torch.manual_seed(1)
    m2 = tn.TEMPONet(in_channels=384, window=160, out_dim=20, causal=causal).eval()
    with torch.no_grad():
        x0 = torch.randn(1, 384, 160)
        y0 = m2(x0)
        changed = []
        for t in range(160):
            x1 = x0.clone(); x1[0, :, t] += 5.0 * torch.randn(384)
            changed.append(bool((m2(x1) - y0).abs().max() > 0))
    idx = [t for t, c in enumerate(changed) if c]
    print("  perturbazione (pesi di default, eval): passi che cambiano l'uscita:", len(idx), "su 160; primo", idx[0], "ultimo", idx[-1])

# finestra lunga per isolare il campo ricettivo di un solo elemento, lontano dai bordi
print("\n=== campo ricettivo di un elemento lontano dai bordi (T=2048, 16 canali) ===")
for causal in (True, False):
    m = tn.TEMPONet(in_channels=16, window=2048, out_dim=5, causal=causal).eval()
    positive_structural(m)
    x = (torch.rand(1, 16, 2048) + 0.5).requires_grad_(True)
    feat = m.features(x)
    L = feat.shape[2]
    j = L // 2
    g, = torch.autograd.grad(feat[0, :, j].sum(), x, retain_graph=True)
    a = dep_range(g)
    g2, = torch.autograd.grad(feat[0, :, j + 1].sum(), x, retain_graph=True)
    b = dep_range(g2)
    print(f"causal={causal}: L={L}; elemento {j}: {a}; elemento {j+1}: {b}; spostamento = {b[0]-a[0]}; unione dei due = {b[1]-a[0]+1}")

# forward rifiuta lunghezze diverse e ingressi (B,T,C)
m = tn.TEMPONet(in_channels=384, window=160, out_dim=20)
for shape in ((2, 384, 159), (2, 160, 384), (2, 384, 161)):
    try:
        m.eval()(torch.zeros(*shape)); print(shape, "accettato")
    except ValueError as e:
        print(shape, "rifiutato:", e)

# BatchNorm in training: l'uscita del campione 0 dipende dal campione 1?
torch.manual_seed(2)
m = tn.TEMPONet(in_channels=384, window=160, out_dim=20)
for mode in ("train", "eval"):
    getattr(m, mode)()
    x = torch.randn(4, 384, 160, requires_grad=True)
    # niente dropout casuale nel confronto: in train il dropout c'e', ma il gradiente incrociato nasce solo dalla BN
    y = m(x)
    g, = torch.autograd.grad(y[0].sum(), x)
    print(f"modo {mode}: |grad| dell'uscita del campione 0 rispetto all'ingresso del campione 1 =", float(g[1].abs().sum()))

# conteggi di parametri citati nel prompt di Claude
for d, T in ((384, 38), (384, 101), (384, 160), (896, 160), (16, 256)):
    od = 20 if d != 16 else 5
    print(f"parametri d={d}, T={T}, out={od}:", tn.count_parameters(tn.TEMPONet(in_channels=d, window=T, out_dim=od)))
print("TEMPONet emg2pose (16x256, 20 uscite):", tn.count_parameters(tn.TEMPONet(in_channels=16, window=256, out_dim=20)))
