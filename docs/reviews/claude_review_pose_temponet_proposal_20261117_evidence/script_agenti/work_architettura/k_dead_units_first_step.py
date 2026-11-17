"""Secondo tentativo: perche' al primo backward solo 377/640 (TCN) e 499/640 (MLP) pesi dell'ultimo strato hanno gradiente non nullo?
Ipotesi: unita' del quinto strato nascosto (32) spente su tutto il batch o accese in pochissime posizioni. Stesso setup di c_zero_init.py."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura")
import torch
from torch import nn
from heads import BUILD, D, J, P
for name in ("MLP", "TCN_post"):
    torch.manual_seed(0)
    m = BUILD[name]().train()
    g = torch.Generator().manual_seed(11)
    x = torch.randn(32, P, D, generator=g)
    acts = []
    relus = [mod for mod in m.modules() if isinstance(mod, nn.ReLU)]
    h = relus[-1].register_forward_hook(lambda mod, i, o: acts.append(o.detach()))
    with torch.no_grad():
        m.eval(); m(x); m.train()
    h.remove()
    a = acts[0]
    a = a if a.shape[-1] == 32 else a.transpose(1, 2)  # (B,T,32)
    c = a[:, 6:P - 6]
    on_frac = (c > 0).float().mean(dim=(0, 1))
    print(f"{name}: unita' del quinto hidden (32) mai accese sui 32x148 centri = {int((on_frac == 0).sum())}; accese in meno dell'1% = {int(((on_frac > 0) & (on_frac < 0.01)).sum())}; sempre accese = {int((on_frac == 1).sum())}")
    print("   quota accesa per unita' (ordinata):", " ".join(f"{v:.3f}" for v in sorted(on_frac.tolist())))
