"""Order-of-summation effect of a 4-rank ring all-reduce in fp32 (sequential sum starting at a rank
that depends on the chunk containing the element). Compares two layouts = two independent start offsets."""
import numpy as np
rng=np.random.default_rng(0)
N=4_000_000
g=(rng.standard_t(3,size=(4,N))*1e-3).astype(np.float32)/4   # per-rank grads, divided by world size (exact)
def ring(start):
    idx=(start[None,:]+np.arange(4)[:,None])%4
    v=g[idx,np.arange(N)]
    acc=v[0].copy()
    for k in range(1,4): acc=(acc+v[k]).astype(np.float32)
    return acc
chunk=lambda size: (np.arange(N)//size)%4
a=ring(chunk(131072)); b=ring((np.arange(N)+777)//(131072*3)%4)   # different offset and chunk size
d=(a.astype(np.float64)-b)
frac=np.mean(a!=b)
ulp=np.spacing(np.abs(a)).astype(np.float64)
print("fraction of elements differing", frac)
print("max |diff| in ulps", np.max(np.abs(d)/ulp))
print("||diff||/||g||", np.linalg.norm(d)/np.linalg.norm(a.astype(np.float64)))
na=np.linalg.norm(a.astype(np.float64)); nb=np.linalg.norm(b.astype(np.float64))
print("relative change of the global norm (fp64 accumulation)", abs(na-nb)/na)
na32=float(np.linalg.norm(a)); nb32=float(np.linalg.norm(b))
print("relative change of the norm in fp32 (single tensor)", abs(na32-nb32)/na32)
