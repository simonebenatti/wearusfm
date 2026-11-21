# Simulazione: somma fp32 a 4 addendi (gia' divisi per 4, come allreduce_hook) in due rotazioni ad anello diverse
import torch
torch.manual_seed(0)
n=200_000
def ring(g, start):  # somma sequenziale fp32 a partire dal rank start (rotazione ciclica, come reduce-scatter ad anello)
    acc=g[start].clone()
    for k in range(1,4): acc=acc+g[(start+k)%4]
    return acc
def rel(g):
    g=(g/4).float()
    a,b=ring(g,0),ring(g,1)
    d=(a.double()-b.double()); den=max(a.double().norm(),b.double().norm())
    ratio=float(g.double().abs().sum(0).norm()/g.double().sum(0).norm())
    nratio=float(sum(g[r].double().norm() for r in range(4))/g.double().sum(0).norm())
    return float(d.norm()/den), ratio, nratio
mu=torch.randn(n)
for s in [0,1,10,100]:
    g=mu+s*torch.randn(4,n); print("indipendente sigma/media=%g rel=%.2e rapp_elem=%.1f rapp_norme=%.1f"%((s,)+rel(g)))
g=torch.randn(4,n)*1e-7; print("solo rumore (gradiente vero nullo) rel=%.2e rapp_elem=%.1f rapp_norme=%.1f"%rel(g))
for c in [20,200,2000,20000]:
    x=torch.randn(2,n)*c
    g=torch.stack([x[0],-x[0],x[1],-x[1]])+torch.randn(4,n)  # anticorrelazione costruita
    print("anticorrelato c=%d rel=%.2e rapp_elem=%.1f rapp_norme=%.1f"%((c,)+rel(g)))
