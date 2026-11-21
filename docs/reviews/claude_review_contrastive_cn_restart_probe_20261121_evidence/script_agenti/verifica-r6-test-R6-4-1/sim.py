# SIMULAZIONE: props senza uuid (torch 2.2) + finto libcuda con l'ABI del driver
import sys, ctypes, types
from unittest.mock import patch
import torch
import contrastive_cn_restart_leg as leg
real = ctypes.CDLL
lib = sys.argv[1]
def fake_cdll(name, *a, **k):
    assert name == "libcuda.so.1", name
    return real(lib)
props = types.SimpleNamespace(name="NVIDIA A100-SXM-64GB")  # nessun attributo uuid
out = []
with patch.object(ctypes, "CDLL", fake_cdll), patch.object(torch.cuda, "get_device_properties", lambda d: props):
    for r in range(4):
        out.append(leg.device_identity(f"cuda:{r}"))
for o in out: print(o)
print("uuid distinti:", len({o["gpu_uuid"] for o in out}))
try:
    with patch.object(ctypes, "CDLL", fake_cdll), patch.object(torch.cuda, "get_device_properties", lambda d: props):
        leg.device_identity("cuda:7")
except Exception as e: print("ordinale fuori intervallo ->", type(e).__name__, e)
