# SIMULAZIONE: riproduce il controllo di torch 2.2 serialization.py:1012-1014
import torch
_orig = torch.load
def _load(f, *a, **k):
    if k.get("mmap") and not isinstance(f, str):
        raise ValueError("f must be a string filename in order to use mmap argument")
    return _orig(f, *a, **k)
torch.load = _load
