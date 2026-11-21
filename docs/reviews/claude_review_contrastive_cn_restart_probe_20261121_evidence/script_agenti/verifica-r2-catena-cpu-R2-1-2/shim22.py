# SIMULAZIONE del controllo di torch 2.2.0a0 serialization.py:1012-1014 (solo processo principale di pytest)
import torch
_orig = torch.load
def _load(f, *a, **k):
    if k.get("mmap") and not isinstance(f, str):
        raise ValueError("f must be a string filename in order to use mmap argument")
    return _orig(f, *a, **k)
torch.load = _load
