"""Come k_dead_units_first_step.py ma in modo train (dropout attivo), replicando l'ordine di c_zero_init.py:
torch.manual_seed(0) -> costruzione -> x da generatore 11 -> primo forward in train (consuma il RNG del dropout)."""
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
        m(x)
    h.remove()
    a = acts[0]
    a = a if a.shape[-1] == 32 else a.transpose(1, 2)
    c = a[:, 6:P - 6]
    on_frac = (c > 0).float().mean(dim=(0, 1))
    print(f"{name} (train, dropout attivo): unita' del quinto hidden mai accese sui centri = {int((on_frac == 0).sum())}; sempre accese = {int((on_frac == 1).sum())}")
