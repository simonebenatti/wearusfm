# Simulation: rounding-only reduced-gradient ratio for small tensors, 4 ranks, fp32, zero-mean rank noise.
import numpy as np
rng=np.random.default_rng(0)
N=200000
def orders(x):  # x: (N,4,k) float32, prescaled by 1/4
    a,b,c,d=[x[:,i] for i in range(4)]
    o1=((a+b)+c)+d; o2=(a+b)+(c+d); o3=((d+c)+b)+a; o4=(a+c)+(b+d)
    return [o1,o2,o3,o4]
for k in (1,5,6,16,20):
    for mu in (0.0,0.3):
        g=(rng.standard_normal((N,4,k))+mu).astype(np.float32)
        x=(g/np.float32(4)).astype(np.float32)
        os_=orders(x); worst=np.zeros(N)
        for i in range(4):
            for j in range(i+1,4):
                d=np.linalg.norm(os_[i].astype(np.float64)-os_[j].astype(np.float64),axis=1)
                den=np.maximum(np.maximum(np.linalg.norm(os_[i].astype(np.float64),axis=1),np.linalg.norm(os_[j].astype(np.float64),axis=1)),1e-12)
                worst=np.maximum(worst,d/den)
        scal=None
        if k==1:
            meanabs=np.abs(g.astype(np.float64)).mean(1)[:,0]
            w2=np.zeros(N)
            for i in range(4):
                for j in range(i+1,4):
                    d=np.abs(os_[i][:,0].astype(np.float64)-os_[j][:,0].astype(np.float64))
                    den=np.maximum.reduce([np.abs(os_[i][:,0]).astype(np.float64),np.abs(os_[j][:,0]).astype(np.float64),meanabs,np.full(N,1e-12)])
                    w2=np.maximum(w2,d/den)
            scal=(w2.max(), (w2>1e-6).mean())
        print(f"k={k} mu={mu} P(orig>1e-6)={(worst>1e-6).mean():.2e} max={worst.max():.2e}", "screened:" , scal)
