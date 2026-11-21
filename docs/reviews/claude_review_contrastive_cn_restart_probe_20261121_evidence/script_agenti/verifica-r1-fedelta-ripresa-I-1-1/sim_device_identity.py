# SIMULAZIONE sul Mac: props senza uuid (come torch 2.2) e libcuda finta che imita le firme del driver.
import ctypes, sys, types
from unittest.mock import patch
sys.path[:0] = ["src", "scripts", "scripts/diagnostics"]
import contrastive_cn_restart_leg as leg
class Props: name = "NVIDIA A100-SXM-64GB"   # nessun attributo uuid
calls = []
class FakeDriver:
    def __init__(self, rc): self.rc = rc
    def cuDeviceGet(self, pdev, idx):
        calls.append(("cuDeviceGet", type(idx).__name__, idx))
        ctypes.cast(pdev, ctypes.POINTER(ctypes.c_int)).contents.value = idx
        return self.rc[0]
    def cuDeviceGetUuid(self, praw, dev):
        calls.append(("cuDeviceGetUuid", type(dev).__name__, dev.value))
        buf = ctypes.cast(praw, ctypes.POINTER(ctypes.c_ubyte * 16)).contents
        for i in range(16): buf[i] = 16 * dev.value + i
        return self.rc[1]
for rc in ((0, 0), (3, 0), (0, 101)):
    with patch.object(leg.torch.cuda, "get_device_properties", lambda d: Props()), \
         patch.object(leg.ctypes, "CDLL", lambda name: FakeDriver(rc)):
        try: print(rc, leg.device_identity("cuda:2"))
        except Exception as e: print(rc, type(e).__name__, e)
print(calls)
