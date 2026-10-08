"""Opt-in A/B fingerprints. Pure hashing: never consumes a random generator.

This audits delivered batches, including signal/targets/masks, not just crop IDs.
It does not make the ordinary resumable trainer's worker stream resumable.
"""
import hashlib
import json
from dataclasses import fields, is_dataclass

import numpy as np
import torch


def _update(digest, value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("object arrays are not auditable")
        digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif is_dataclass(value):
        _update(digest, {f.name: getattr(value, f.name) for f in fields(value)})
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value):
            _update(digest, key)
            _update(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (tuple, list)):
        digest.update(b"list[")
        for item in value:
            _update(digest, item)
        digest.update(b"]")
    else:
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\0")


def state_sha256(state):
    digest = hashlib.sha256()
    _update(digest, state)
    return digest.hexdigest()


def batch_sha256(batch):
    digest = hashlib.sha256()
    _update(digest, batch)
    return digest.hexdigest()
