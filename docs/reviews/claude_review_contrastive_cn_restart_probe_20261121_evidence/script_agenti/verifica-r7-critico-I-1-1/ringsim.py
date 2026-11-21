# SIMULAZIONE CPU (non e' NCCL): somma fp32 di 4 contributi per rank con ordine ad anello
# ((g[s+1]+g[s+2])+g[s+3])+g[s]; le due riprese usano start rank diversi per elemento
# (offset di bucket diversi). Poi /4 esatto. rel_l2 come in compare_vector.
import torch
torch.manual_seed(1)
def ring(g, start):  # g: (4, T, n) fp32, start: (T,n) long
    o=[(start+k)%4 for k in (1,2,3,0)]
    take=lambda idx: torch.gather(g,0,idx.unsqueeze(0)).squeeze(0)
    acc=take(o[0]).clone()
    for idx in o[1:]: acc=acc+take(idx)
    return acc/4
def trial(n, snr, T=20000, scale=1e-3):
    g=(scale*(snr+torch.randn(4,T,n))).float()
    s1=torch.randint(0,4,(T,n)); s2=(s1+torch.randint(1,4,(T,n)))%4
    a,b=ring(g,s1).double(),ring(g,s2).double()
    d=(a-b).norm(dim=1); den=torch.maximum(torch.maximum(a.norm(dim=1),b.norm(dim=1)),torch.tensor(1e-12))
    rel=d/den
    return float((rel>1e-6).float().mean()), float(rel.median()), float(rel.max())
for n in (1,5,6,20,24,32,54,256,4096):
    for snr in (0.0,0.5,2.0):
        p,med,mx=trial(n,snr, T=20000 if n<300 else 500)
        print(f"n={n:5d} mean/sd={snr:3.1f}  P(rel>1e-6)={p:.4f}  mediana={med:.2e}  max={mx:.2e}")
