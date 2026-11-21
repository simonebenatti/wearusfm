# Simulazione CPU: stessa somma di 4 gradienti locali fp32, due ordini di riduzione (ring con rank di partenza diverso)
import torch
torch.manual_seed(0)
def ring(g, start):
    order=[(start+i)%4 for i in range(4)]
    acc=g[order[0]].clone()
    for r in order[1:]: acc=acc+g[r]
    return acc/1  # divisione esatta per 4 omessa
def rel(a,b):
    a,b=a.double(),b.double(); return float((a-b).norm()/max(a.norm(),b.norm(),1e-12))
n=147456
gen=[torch.randn(n)*1e-3 + 0.3e-3*torch.randn(n) for _ in range(4)]
print("generico     rel_l2 =", rel(ring(gen,0)/4, ring(gen,1)/4))
base=torch.randn(n)*1e-3
canc=[base*(1 if r<2 else -1)+1e-7*torch.randn(n) for r in range(4)]  # contributi che si annullano fra rank (gradiente vero ~0)
print("cancellante  rel_l2 =", rel(ring(canc,0)/4, ring(canc,1)/4))
