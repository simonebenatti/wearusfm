# Tensori piccoli, gradiente locale indipendente fra rank (media vera nulla o media=0.3 sigma); due rotazioni ad anello diverse
import torch
torch.manual_seed(1)
T=20000
def ring(g,start):
    acc=g[start].clone()
    for k in range(1,4): acc=acc+g[(start+k)%4]
    return acc
for n in [1,2,6,16,64,256]:
  for mu in [0.0,0.3]:
    g=((mu+torch.randn(T,4,n))*1e-3/4).float().transpose(0,1)  # [4,T,n]
    a,b=ring(g,0),ring(g,1)
    d=(a.double()-b.double()).norm(dim=-1); den=torch.maximum(a.double().norm(dim=-1),b.double().norm(dim=-1)).clamp(min=1e-12)
    r=d/den
    print(f"n={n:4d} media={mu} P(rel>1e-6)={float((r>1e-6).double().mean()):.4f} P(rel>0)={float((r>0).double().mean()):.3f} max={float(r.max()):.1e}")
