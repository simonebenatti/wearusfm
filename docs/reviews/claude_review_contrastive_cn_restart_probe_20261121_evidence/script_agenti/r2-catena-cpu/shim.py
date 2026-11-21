"""SIMULATION of torch 2.2 behaviours on the Mac's torch 2.14:
 (1) torch.load(f, mmap=True) raises ValueError unless isinstance(f, str)  (serialization.py:1012-1014 on the cluster)
 (2) torch.cuda.get_device_properties(...) result without a `uuid` attribute.
"""
import ctypes, json, os, sys, types, uuid as uuidmod
from pathlib import Path
from unittest.mock import patch
CLONE = Path(os.environ["CLONE"])
sys.path[:0] = [str(CLONE / p) for p in ("src", "scripts", "scripts/diagnostics")]
import torch
_load = torch.load
def load22(f, *a, mmap=None, **k):
    if mmap and not isinstance(f, str):
        raise ValueError("f must be a string filename in order to use mmap argument")
    return _load(f, *a, mmap=mmap, **k)

base = Path(sys.argv[1])
import contrastive_cn_restart_observe as O
import contrastive_cn_restart_compare as C
import contrastive_target_resume as T
with patch.object(torch, "load", load22):
    for mode in ("ordinary", "warm"):
        try:
            O.load_observation(base / mode, 0, expected_parameters=106)
            print(mode, "load_observation under shim: OK")
        except Exception as e:
            print(mode, "load_observation under shim:", type(e).__name__, e,
                  "| caught by compare.main:", isinstance(e, (ValueError, KeyError, OSError, RuntimeError, TypeError, IndexError, EOFError)))
    s = T.checkpoint(base / "warm" / "checkpoint.pt")
    print("contrastive_target_resume.checkpoint under shim: OK, step", s["step"])
    # Minimal fix: pass str(tensor_path)
    src = (CLONE / "scripts/diagnostics/contrastive_cn_restart_observe.py").read_text()
    fixed = src.replace("torch.load(tensor_path, map_location", "torch.load(str(tensor_path), map_location")
    assert fixed != src
    mod = types.ModuleType("observe_fixed"); exec(compile(fixed, "observe_fixed", "exec"), mod.__dict__)
    rec, vec = mod.load_observation(base / "ordinary", 0, expected_parameters=106)
    print("fixed load_observation under shim: OK,", len(rec["parameters"]), "rows; offsets preserved:",
          all(vec["local"][r["id"]].storage_offset() == r["bucket_element_offset"] for r in rec["parameters"]))

# (2) device_identity with props lacking uuid, and a fake libcuda
import contrastive_cn_restart_leg as leg
props = types.SimpleNamespace(name="NVIDIA A100-SXM-64GB")
raw_uuid = uuidmod.UUID("12345678-1234-5678-1234-567812345678")
class FakeDriver:
    def cuDeviceGet(self, ordinal_ref, index):
        ctypes.cast(ordinal_ref, ctypes.POINTER(ctypes.c_int)).contents.value = index; return 0
    def cuDeviceGetUuid(self, raw_ref, ordinal):
        arr = ctypes.cast(raw_ref, ctypes.POINTER(ctypes.c_ubyte * 16)).contents
        for i, b in enumerate(raw_uuid.bytes): arr[i] = b
        return 0
class FakeDev:
    def __init__(self, s): self.index = int(str(s).split(":")[1])
with patch.object(torch.cuda, "get_device_properties", lambda d: props), patch.object(ctypes, "CDLL", lambda n: FakeDriver()), \
        patch.object(leg.torch, "device", FakeDev):
    print("device_identity (simulated, no uuid attr):", leg.device_identity("cuda:2"))
