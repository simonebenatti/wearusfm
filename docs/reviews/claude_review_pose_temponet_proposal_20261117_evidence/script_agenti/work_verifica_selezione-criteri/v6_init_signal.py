# Segnale all'inizializzazione Torch standard (mio): varianza fra ingressi all'ultimo strato nascosto (32 unita') di MLP e TCN.
import torch, numpy as np
from torch import nn
torch.set_num_threads(1)
D = 384
def mlp():
    return nn.Sequential(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(),
                         nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
def tcn():
    return nn.Sequential(nn.Conv1d(D, 64, 1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                         nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=3, dilation=3), nn.ReLU(),
                         nn.Conv1d(64, 32, 1), nn.ReLU())
g = torch.Generator().manual_seed(5)
# ingresso A: gaussiano iid; ingresso B: componente fissa forte + variazioni lente (piu' simile a feature apprese), entrambi con LN
xa = torch.randn(64, 160, D, generator=g)
base = torch.randn(D, generator=g); slow = torch.randn(64, 1, D, generator=g); fast = torch.randn(64, 160, D, generator=g)
xb = 1.0 * base + 0.5 * slow + 0.5 * fast
for name, x in (('A iid', xa), ('B struttura', xb)):
    x = nn.functional.layer_norm(x, (D,))
    for arch in ('MLP', 'TCN'):
        for seed in (0, 1, 2):
            torch.manual_seed(seed); net = mlp() if arch == 'MLP' else tcn()
            with torch.no_grad():
                h = net(x) if arch == 'MLP' else net(x.transpose(1, 2)).transpose(1, 2)
                h = h[:, 6:154].reshape(-1, 32)
                var_in = h.var(0).mean().item(); const = (h.mean(0) ** 2).mean().item(); dead = (h.max(0).values <= 0).float().mean().item()
                first = (net[1](net[0](x)) if arch == 'MLP' else net[1](net[0](x.transpose(1, 2))).transpose(1, 2)).reshape(-1, net[0].out_features if arch == 'MLP' else 64)
            print(f'{name:12s} {arch} seme {seed}: var dip. ingresso ultimo nascosto {var_in:.2e}, parte costante {const:.2e}, '
                  f'rapporto {var_in / const:.4f}, unita spente {dead:.2f}; var dopo il primo strato {first.var(0).mean().item():.4f}')
