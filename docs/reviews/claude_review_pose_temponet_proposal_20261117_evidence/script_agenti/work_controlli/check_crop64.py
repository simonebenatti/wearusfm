import copy, torch
from torch import nn
import check_heads as H
torch.manual_seed(1)
for name, make in (("L13", H.l13), ("TCN", H.tcn)):
    m = make().double().eval()
    for p in m.parameters():
        nn.init.normal_(p, std=0.3)
    x = torch.randn(2, H.P, H.D, dtype=torch.float64)
    with torch.no_grad():
        full = m(x)[:, 6:-6]
        m2 = copy.deepcopy(m)
        for c in m2.modules():
            if isinstance(c, nn.Conv1d):
                c.padding = (0,)
        nopad = m2(x)
        print(name, tuple(full.shape), tuple(nopad.shape), "scarto massimo float64:", float((full - nopad).abs().max()), "ampiezza massima:", float(full.abs().max()))
