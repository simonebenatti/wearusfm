# SIMULAZIONE: libcuda finta (Mac senza CUDA); props senza uuid come su torch 2.2.
import ctypes, sys, types
from unittest.mock import patch
sys.path[:0]=["src","scripts","scripts/diagnostics"]
import torch
import contrastive_cn_restart_leg as leg
class Props: name="NVIDIA A100-SXM-64GB"
def make(rc_get=0, rc_uuid=0):
    class Drv:
        def cuDeviceGet(self, p, idx):
            assert isinstance(idx,int); ctypes.cast(p, ctypes.POINTER(ctypes.c_int))[0]=idx; return rc_get
        def cuDeviceGetUuid(self, p, ordinal):
            arr=ctypes.cast(p, ctypes.POINTER(ctypes.c_ubyte*16))[0]
            for i in range(16): arr[i]=i+16*ordinal.value
            return rc_uuid
    return lambda name: Drv()
for rc in [(0,0),(0,999),(100,0)]:
    with patch.object(torch.cuda,"get_device_properties",lambda d: Props()), patch.object(ctypes,"CDLL",make(*rc)):
        try: print(rc, leg.device_identity("cuda:2"))
        except Exception as e: print(rc, type(e).__name__, e)
