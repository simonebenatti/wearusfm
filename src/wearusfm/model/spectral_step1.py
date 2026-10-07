"""Experimental Step 1: shared differentiable readout, global targets and keep loss.

The 32 coordinates describe mean spectra, NOT temporal activation order.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wearusfm.model.spectral_targets import EPS, TARGET_VERSION

READOUT_VERSION = "p1p2_qc_equal_channel_v1"
FAMILIES = {"fast_shape": slice(0, 6), "fast_energy": slice(6, 7),
            "slow_shape": slice(7, 31), "slow_energy": slice(31, 32)}


def pool_p1p2(enc, counts: list[int], qc_valid: torch.Tensor) -> torch.Tensor:
    """(S,2d), equal weight QC-good channels; rejects masked views and empty samples."""
    if sum(counts) != enc.local.shape[0] or len(counts) != enc.z.shape[0] or any(c <= 0 for c in counts):
        raise ValueError("readout channel counts do not match encoding")
    if qc_valid.shape != (sum(counts),) or qc_valid.dtype != torch.bool:
        raise ValueError("qc_valid must be a boolean vector of packed channels")
    valid = enc.patch_valid & qc_valid[:, None]
    if not torch.equal(enc.visible, valid):
        raise ValueError("P1/P2 readout requires an UNMASKED encoding")
    tv = enc.key_time_valid
    if not bool(tv.any(dim=1).all()):
        raise ValueError("sample has no valid backbone time")
    z = torch.where(tv[..., None], enc.z.float().mean(dim=2), 0.).sum(dim=1) / tv.sum(dim=1, keepdim=True)
    local = torch.where(valid[..., None], enc.local.float(), 0.).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
    pooled, start = [], 0
    for c in counts:
        good = qc_valid[start:start+c] & enc.patch_valid[start:start+c].any(dim=1)
        if not bool(good.any()):
            raise ValueError("sample has no QC-valid channel with patches")
        pooled.append(local[start:start+c][good].mean(dim=0))
        start += c
    return torch.cat([z, torch.stack(pooled)], dim=1)


def global_spectral_targets(targets, counts, qc_valid) -> tuple[np.ndarray, np.ndarray]:
    """Average linear power in time per channel, then equally across good channels.

    A scale is disabled if ANY good channel lacks windows; available bands are
    the intersection across good channels. Missing values are never observations.
    """
    qc = qc_valid.detach().cpu().numpy() if isinstance(qc_valid, torch.Tensor) else np.asarray(qc_valid)
    if len(targets) != len(counts) or qc.shape != (sum(counts),) or qc.dtype != bool:
        raise ValueError("global spectral target counts/QC mismatch")
    y, mask = np.zeros((len(counts), 32), np.float32), np.zeros((len(counts), 32), bool)
    start = 0
    for s, (tg, c) in enumerate(zip(targets, counts)):
        good = qc[start:start+c]
        start += c
        for name, offset, nb in (("ms_fast", 0, 6), ("ms_slow", 7, 24)):
            lp, valid, avail = (getattr(tg, name+suffix) for suffix in ("", "_valid", "_available"))
            if lp is None or valid is None or avail is None:
                raise ValueError("keep targets require multiscale anchors")
            if lp.shape[0] != c or lp.shape[-1] != nb or valid.shape != lp.shape[:2] or avail.shape != (c, nb):
                raise ValueError("invalid multiscale target shape")
            if not good.any() or not valid[good].any(axis=1).all():
                continue
            common = avail[good].all(axis=0)
            if not common.any():
                continue
            per_channel = []
            for ch in np.flatnonzero(good):
                observed = lp[ch][valid[ch]][:, common].astype(np.float64)
                if not np.isfinite(observed).all():
                    raise ValueError("nonfinite observed spectral target")
                per_channel.append(np.maximum(np.exp(observed)-EPS, 0.).mean(axis=0))
            power = np.mean(per_channel, axis=0)
            energy = power.sum()
            if not np.isfinite(energy):
                raise ValueError("nonfinite integrated spectral energy")
            y[s, offset+nb] = np.log(energy+EPS)
            mask[s, offset+nb] = True
            if energy > EPS:
                idx = offset + np.flatnonzero(common)
                y[s, idx] = np.log(np.maximum(power/energy, EPS))
                mask[s, idx] = True
    return y, mask


class SpectralKeepHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 32)
        self.register_buffer("target_mean", torch.zeros(32))
        self.register_buffer("target_std", torch.ones(32))
        self.register_buffer("target_active", torch.zeros(32, dtype=torch.bool))
        self.register_buffer("stats_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_target_stats(self, y, valid, min_std: float = 1e-4):
        y = torch.as_tensor(y, dtype=torch.float64, device=self.target_mean.device)
        valid = torch.as_tensor(valid, dtype=torch.bool, device=y.device)
        if y.ndim != 2 or y.shape[1] != 32 or valid.shape != y.shape or not bool(torch.isfinite(y[valid]).all()):
            raise ValueError("invalid calibration targets")
        count = valid.sum(dim=0)
        mean = torch.where(valid, y, 0.).sum(0)/count.clamp(min=1)
        var = torch.where(valid, (y-mean).square(), 0.).sum(0)/count.clamp(min=1)
        std = var.sqrt()
        active = (count >= 2) & (std >= min_std)
        if not bool(active.any()):
            raise ValueError("no variable observed coordinate in calibration")
        self.target_mean.copy_(mean.float())
        self.target_std.copy_(torch.where(active, std, 1.).float())
        self.target_active.copy_(active)
        self.stats_fitted.fill_(True)

    def forward(self, features):
        return self.linear(features)

    def loss(self, prediction, y, valid):
        if not bool(self.stats_fitted):
            raise ValueError("fit target statistics on TRAIN before using keep loss")
        y = torch.as_tensor(y, dtype=torch.float32, device=prediction.device)
        valid = torch.as_tensor(valid, dtype=torch.bool, device=prediction.device) & self.target_active
        standardized = torch.where(valid, (y-self.target_mean)/self.target_std, 0.)
        error = F.smooth_l1_loss(prediction.float(), standardized, reduction="none")
        error = torch.where(valid, error, 0.)
        per_family, present, metrics = [], [], {}
        for name, cols in FAMILIES.items():
            count = valid[:, cols].sum(1)
            v = error[:, cols].sum(1)/count.clamp(min=1)
            per_family.append(v)
            present.append(count > 0)
            metrics["keep_"+name] = v[count > 0].mean().detach() if bool((count > 0).any()) else prediction.new_zeros(())
        present = torch.stack(present, dim=1)
        per_sample = torch.stack(per_family, dim=1).sum(1)/present.sum(1).clamp(min=1)
        ok = present.any(dim=1)
        loss = per_sample[ok].mean() if bool(ok.any()) else prediction.sum()*0.
        metrics["keep_valid_samples"] = ok.sum().float()
        return loss, metrics


def keep_objective(student, inp, anchor_targets, *, target_version=TARGET_VERSION):
    if any(x.device.type not in ("cpu", "cuda") for x in inp.signals):
        raise ValueError("Step 1 supports CPU/CUDA only (RNG pairing not validated on MPS)")
    if target_version != TARGET_VERSION or student.spectral_keep is None:
        raise ValueError("incompatible keep target version or missing head")
    y, valid = global_spectral_targets(anchor_targets, inp.counts, inp.qc_valid)
    # Additional dropout must not change the RNG stream of the paired JEPA run.
    devices = sorted({x.device.index for x in inp.signals if x.is_cuda})
    with torch.random.fork_rng(devices=devices):
        enc = student.encode(inp, None)
        prediction = student.spectral_keep(pool_p1p2(enc, inp.counts, inp.qc_valid))
    return student.spectral_keep.loss(prediction, y, valid)
