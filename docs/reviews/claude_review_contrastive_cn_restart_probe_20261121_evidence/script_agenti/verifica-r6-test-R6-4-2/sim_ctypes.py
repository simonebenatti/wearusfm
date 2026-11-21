# SIMULAZIONE: props senza uuid (come torch 2.2) e libcuda finto con la semantica di cuDeviceGet/cuDeviceGetUuid.
import ctypes, sys, types, uuid
from unittest.mock import patch
import contrastive_cn_restart_leg as leg
UUIDS = [uuid.UUID(int=0x1111*(i+1) << 64 | i) for i in range(4)]
calls = []
class FakeDriver:
    def cuDeviceGet(self, dev_ref, ordinal):
        calls.append(("get", ordinal)); dev_ref._obj.value = ordinal; return 0
    def cuDeviceGetUuid(self, raw_ref, dev):
        d = dev.value if isinstance(dev, ctypes.c_int) else dev
        calls.append(("uuid", d)); ctypes.memmove(raw_ref._obj, UUIDS[d].bytes, 16); return 0
props = types.SimpleNamespace(name="NVIDIA A100-SXM-64GB")
out = []
with patch.object(leg.torch.cuda, "get_device_properties", lambda d: props), \
     patch.object(leg.ctypes, "CDLL", lambda name: (calls.append(("dlopen", name)), FakeDriver())[1]):
    for r in range(4):
        out.append(leg.device_identity(f"cuda:{r}"))
for o in out: print(o)
print(calls)
assert [o["gpu_uuid"] for o in out] == [str(u) for u in UUIDS]
assert len({o["gpu_uuid"] for o in out}) == 4 and [o["device_index"] for o in out] == [0,1,2,3]
# errore del driver -> RuntimeError
class Bad(FakeDriver):
    def cuDeviceGetUuid(self, raw_ref, dev): return 3  # CUDA_ERROR_NOT_INITIALIZED
with patch.object(leg.torch.cuda, "get_device_properties", lambda d: props), patch.object(leg.ctypes, "CDLL", lambda n: Bad()):
    try: leg.device_identity("cuda:0"); print("NO ERROR")
    except RuntimeError as e: print("RuntimeError:", e)
# device senza indice
with patch.object(leg.torch.cuda, "get_device_properties", lambda d: props), patch.object(leg.ctypes, "CDLL", lambda n: FakeDriver()):
    try: print("cuda (no index):", leg.device_identity("cuda"))
    except Exception as e: print(type(e).__name__, e)
print("SIM OK")
