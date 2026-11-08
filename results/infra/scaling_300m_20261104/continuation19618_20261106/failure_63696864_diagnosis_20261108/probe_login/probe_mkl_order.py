import sys, os
MODE = sys.argv[1]
if MODE == "torch_first":
    import torch; import numpy as np
else:
    import numpy as np; import torch
from scipy import signal as sps, fft as sfft, linalg as sla
import ctypes
def maps():
    out = set()
    for line in open("/proc/self/maps"):
        p = line.split()
        if len(p) >= 6 and ".so" in p[5]:
            out.add(os.path.basename(p[5]))
    return out
class Dl(ctypes.Structure):
    _fields_ = [("fname", ctypes.c_char_p), ("fbase", ctypes.c_void_p), ("sname", ctypes.c_char_p), ("saddr", ctypes.c_void_p)]
libc = ctypes.CDLL(None)
libc.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Dl)]
def provider(sym):
    try:
        addr = ctypes.cast(getattr(ctypes.CDLL(None), sym), ctypes.c_void_p).value
    except Exception as e:
        return f"non risolto ({type(e).__name__})"
    info = Dl()
    if libc.dladdr(addr, ctypes.byref(info)) and info.fname:
        return os.path.basename(info.fname.decode())
    return "dladdr fallito"
base = maps()
print(MODE, "| kernel MKL gia' caricati:", sorted(l for l in base if "mkl_avx512" in l or "vml" in l))
for sym in ("sgemm_", "dgemm_", "dgesv_", "cblas_sgemm", "cblas_sgemm64_", "sgemm_64_", "vsLn", "vmsLn", "LAPACKE_dgesv"):
    print(f"   simbolo globale {sym:16s} -> {provider(sym)}")
rng = np.random.default_rng(0)
sos = sps.butter(4, [20, 450], btype="bandpass", fs=2000, output="sos")
steps = [
    ("np.linalg.solve", lambda: np.linalg.solve(rng.standard_normal((4, 4)), rng.standard_normal(4))),
    ("np matmul f32", lambda: rng.standard_normal((256, 129)).astype(np.float32) @ rng.standard_normal((129, 8)).astype(np.float32)),
    ("np.log f64", lambda: np.log(np.abs(rng.standard_normal(10000)) + 1e-6)),
    ("scipy.fft.rfft f32", lambda: sfft.rfft(rng.standard_normal((16, 32, 256)).astype(np.float32), axis=-1)),
    ("iirnotch", lambda: sps.iirnotch(50, 30, fs=2000)),
    ("sosfilt_zi", lambda: sps.sosfilt_zi(sos)),
    ("sosfiltfilt", lambda: sps.sosfiltfilt(sos, rng.standard_normal((16, 8000)), axis=-1)),
    ("scipy.linalg.solve", lambda: sla.solve(rng.standard_normal((4, 4)), rng.standard_normal(4))),
    ("torch cpu log", lambda: torch.log(torch.rand(1000) + 0.1)),
    ("torch cpu matmul", lambda: torch.rand(64, 64) @ torch.rand(64, 64)),
]
prev = base
for name, fn in steps:
    fn(); now = maps(); new = sorted(now - prev); prev = now
    print(f"   dopo {name:20s}: nuove librerie {new}")
