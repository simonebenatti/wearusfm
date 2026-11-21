# Simulation: fp32 ring-like reduction of 4 pre-divided local gradients, two cyclic orders.
import numpy as np
rng=np.random.default_rng(0); n=200_000; u=2.0**-24
def ring(g,start):
    # NCCL ring: chunk reduced starting at rank start, sequential fp32 adds
    idx=[(start+k)%4 for k in range(4)]
    acc=g[idx[0]].copy()
    for i in idx[1:]: acc=(acc+g[i]).astype(np.float32)
    return acc
for corr,label in [(0.0,"indep"),(0.9,"pos-corr")]:
    pass
print("alpha  R=sum||g_r||/||sum||  rel_l2(worst over start pairs)  bound 6u*R")
for alpha in [0,0.5,0.9,0.99,0.999,0.9999]:
    m=rng.standard_normal(n)
    nz=rng.standard_normal((4,n))
    nz=nz-alpha*nz.mean(0)   # anti-correlate across ranks
    g=((0.0*m+nz)/4).astype(np.float32)   # zero-mean true gradient (worst case)
    exact=g.astype(np.float64).sum(0)
    R=sum(np.linalg.norm(gi.astype(np.float64)) for gi in g)/np.linalg.norm(exact)
    outs=[ring(g,s).astype(np.float64) for s in range(4)]
    worst=max(np.linalg.norm(outs[a]-outs[b])/max(np.linalg.norm(outs[a]),np.linalg.norm(outs[b])) for a in range(4) for b in range(a+1,4))
    print(f"{alpha:<7}{R:>10.2f}{worst:>14.3e}{6*u*R:>12.3e}")
