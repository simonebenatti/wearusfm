# Simulazione CPU (dichiarata): somma ad anello NCCL modellata come rotazione ciclica dell'ordine
# ((g[c+1]+g[c+2])+g[c+3])+g[c], fp32, dopo divisione esatta per 4 (allreduce_hook / reducer).
# Si confrontano due rotazioni diverse (chunk diverso dopo il cambio di offset) e si misura
# la regola del diagnostico: ||d||/max(||a||,||b||,1e-12) > 1e-6 su un tensore.
import torch
torch.manual_seed(1)
def ring(g, c):  # g: [4, n] fp32
    o=[(c+1)%4,(c+2)%4,(c+3)%4,c%4]
    acc=g[o[0]]+g[o[1]]; acc=acc+g[o[2]]; return acc+g[o[3]]
def rel(a,b):
    a,b=a.double(),b.double(); return ((a-b).norm(dim=-1)/torch.maximum(torch.maximum(a.norm(dim=-1),b.norm(dim=-1)),torch.tensor(1e-12))).numpy()
T=200000
for n in (1,5,6,20,24,54,1024,147456):
    for snr in (0.0,0.5,1.0,3.0):
        trials = T if n<=54 else (200 if n==1024 else 5)
        mu=torch.randn(trials,1,n)*snr if snr else torch.zeros(trials,1,n)
        g=((mu+torch.randn(trials,4,n))*1e-3/4).float()   # div per 4 esatta
        g=g.transpose(0,1)
        a=ring(g,0); 
        res=[]
        for c in (1,2,3):
            b=ring(g,c); res.append(rel(a,b))
        import numpy as np
        r=np.stack(res)
        print(f"n={n:6d} snr={snr:3.1f} P(rel>1e-6)={ (r>1e-6).mean():.4f}  P(rel>1e-5)={(r>1e-5).mean():.5f} mediana={np.median(r):.2e} max={r.max():.2e}")
