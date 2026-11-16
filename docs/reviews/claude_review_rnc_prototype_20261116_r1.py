"""Review Claude del prototipo Rank-N-Contrast (16/11/2026): controesempio R1 della review GPT 924e19d rifatto sul nucleo del prototipo.
Quattro pose identiche, due per finestra, logit uniformi; positivi cross contro all. Uso: python <questo file> <albero>/src."""
import sys, torch
sys.path.insert(0, sys.argv[1])
from wearusfm.model import rank_contrast as rc
logits = torch.zeros(4, 4, dtype=torch.float64, requires_grad=True)
dist = torch.zeros(4, 4, dtype=torch.float64)
win = torch.tensor([0, 0, 1, 1])
cand = ~torch.eye(4, dtype=torch.bool)
for name, pos in (("cross", cand & (win[:, None] != win[None, :])), ("all", cand)):
    num, chance, n_pos = rc.rank_contrast_pairs(logits, dist, cand, pos)
    val = (num / chance)[0]
    g, = torch.autograd.grad(val, logits)
    print(name, "derivate della perdita dell'ancora 0 sui logit verso [se', stessa finestra, altra, altra]:", [round(float(x), 6) for x in g[0]])
