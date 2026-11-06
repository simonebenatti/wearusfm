"""Paired pre/post-backbone readouts without changing the model's forward path.

The pre view already includes tokenizer, local encoder and Perceiver. The post
view includes the backbone's final normalization. Both have width D, not 2D.
"""
from __future__ import annotations

import torch
import numpy as np

from wearusfm.model.spectral_step1 import pool_p1p2


def depth_readouts(model, inp):
    """One unmasked eval forward; sequences preserve time, vectors average it.

    A temporary hook observes the *actual* Perceiver output. It is removed even
    if encode fails. This helper is intentionally eval-only and does not detach:
    callers choose no_grad/inference_mode for extraction.
    """
    if model.training:
        raise ValueError("depth readouts require eval mode")
    captured = []
    hook = model.pool.register_forward_hook(lambda module, args, value: captured.append(value))
    try:
        enc = model.encode(inp, None)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape != enc.z.shape:
        raise ValueError("expected one same-shape Perceiver/backbone pair")
    valid = enc.key_time_valid
    if not bool(valid.any(dim=1).all()):
        raise ValueError("sample without valid temporal support")
    sequences = {"pre": captured[0].float().mean(dim=2), "post": enc.z.float().mean(dim=2)}
    m = valid.float()[..., None]
    vectors = {k: (v * m).sum(dim=1) / m.sum(dim=1) for k, v in sequences.items()}
    historical = pool_p1p2(enc, inp.counts, inp.qc_valid)
    if not torch.allclose(vectors["post"], historical[:, :model.cfg.dim], atol=1e-6, rtol=1e-5):
        raise ValueError("post-backbone pooling differs from historical readout")
    return {"sequences": sequences, "vectors": vectors, "valid": valid, "pool": historical}


@torch.no_grad()
def extract_windows(model, windows, subjects, fs, montage, *, batch=8, device="cpu", retain_sequences=False):
    """Common D-independent extractor for fixed-length raw windows.

    Classification vectors average the same start/end crops as historical FF.
    Optional temporal arrays keep crops on a separate axis (N,crop,patch,D):
    never average different temporal positions or call such a cache a trajectory.
    This preprocessing is for classification only. Pose callers must instead
    use depth_readouts on the original pose loader ModelInputs, preserving its
    session scales, filtering margins and exact target grid.
    """
    from wearusfm.harness.finetuning import PreparedWindows
    from wearusfm.harness.fm_features import _crops

    if len(windows) == 0 or batch < 1:
        raise ValueError("nonempty windows and positive batch required")
    model.eval()
    prepared = PreparedWindows(windows, np.asarray(subjects), fs, montage)
    n, d = len(windows), model.cfg.dim
    out = {"pre": np.zeros((n, d), dtype=np.float32), "post": np.zeros((n, d), dtype=np.float32),
           "pool": np.zeros((n, 2 * d), dtype=np.float32), "crop_bounds": _crops(windows.shape[1], fs, 4.)}
    temporal = {k: [] for k in ("pre", "post", "valid")}
    for start in range(0, n, batch):
        end = min(n, start + batch)
        part = {k: [] for k in temporal}
        for inp in prepared.batch(range(start, end), device):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                views = depth_readouts(model, inp)
            for name in ("pre", "post"):
                out[name][start:end] += views["vectors"][name].cpu().numpy() / len(prepared.crops)
                if retain_sequences:
                    part[name].append(views["sequences"][name].cpu().numpy())
            out["pool"][start:end] += views["pool"].float().cpu().numpy() / len(prepared.crops)
            if retain_sequences:
                part["valid"].append(views["valid"].cpu().numpy())
        if retain_sequences:
            for name in temporal:
                temporal[name].append(np.stack(part[name], axis=1))
    if retain_sequences:
        out["temporal"] = {k: np.concatenate(v) for k, v in temporal.items()}
    if not all(np.isfinite(out[k]).all() for k in ("pre", "post", "pool")):
        raise ValueError("nonfinite depth features")
    return out
