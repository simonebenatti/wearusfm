# Controllo di algebra lineare numerica (matrice casuale, nessun dato del progetto, nessuna rete):
# rapporto del residuo normale per un RHS rappresentabile esattamente (come la base «L») con due normalizzazioni.
import numpy as np
rng=np.random.default_rng(0)
n,p=20000,384
Z=rng.standard_normal((n,p)); Z-=Z.mean(1,keepdims=True); Z/=Z.std(1,keepdims=True)   # «LN» in float64: somma per riga = 0
X=np.hstack([Z,np.ones((n,1))])
beta=rng.standard_normal(p+1)/np.sqrt(p)
def solve(X,y,cut=1e-8):
    Q,R=np.linalg.qr(X,mode='reduced'); U,s,Vt=np.linalg.svd(R)
    keep=s>cut*s[0]; c=(U.T@(Q.T@y))[keep]/s[keep]; b=Vt[keep].T@c
    return b,keep.sum(),s
nX=np.linalg.norm(X,2)
for tag,y in (('y=X*beta in float64 (consistente)',X@beta),
              ('y calcolato in float32 poi in float64',(X.astype(np.float32)@beta.astype(np.float32)).astype(np.float64)),
              ('y = X*beta + rumore (incompatibile)',X@beta+rng.standard_normal(n))):
    b,rk,s=solve(X,y); r=y-X@b; g=np.linalg.norm(X.T@r)
    print(f"{tag}: rango={rk}; ||X^T r||/(||X|| ||r||)={g/(nX*np.linalg.norm(r)):.3e}; ||X^T r||/(||X|| ||y||)={g/(nX*np.linalg.norm(y)):.3e}; sqrt(p/n)={np.sqrt((p+1)/n):.3e}")
# LN in FP32 contro float64: componente lungo (1,...,1)/sqrt(D) e rapporto sigma_null/sigma_max
x=(rng.standard_normal((n,p))*0.5+rng.standard_normal(p)*1.0+3.0).astype(np.float16).astype(np.float32)  # righe con media comune
def ln(a):
    mu=a.mean(1,keepdims=True); v=((a-mu)**2).mean(1,keepdims=True); return (a-mu)/np.sqrt(v+1e-5)
for dt in (np.float32,np.float64):
    z=ln(x.astype(dt)).astype(np.float64); Xd=np.hstack([z,np.ones((n,1))]); s=np.linalg.svd(Xd,compute_uv=False)
    print(f"LN in {np.dtype(dt).name}: sigma_min/sigma_max={s[-1]/s[0]:.2e}; RMS di sum_k z_k={np.sqrt((z.sum(1)**2).mean()):.2e}; rango a 1e-8: {(s>1e-8*s[0]).sum()} su {p+1}")
