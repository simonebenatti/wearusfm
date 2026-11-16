"""Stateless primitives for the masked-student / EMA-teacher InfoNCE arm.

Metadata filtering, distributed gathering and memory updates belong to the
training controller. Here keys are always detached; only student queries learn.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """D -> hidden -> GELU -> out, preserving the caller's CPU RNG state."""

    def __init__(self, dim: int, hidden: int | None = None, out: int = 128):
        super().__init__()
        hidden = dim if hidden is None else hidden
        if any(type(v) is not int or v < 1 for v in (dim, hidden, out)):
            raise ValueError("contrastive projector dimensions must be positive integers")
        with torch.random.fork_rng(devices=[]):
            self.layers = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def window_readout(z: torch.Tensor, physical_valid: torch.Tensor,
                   visible_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean over slots and ALL physical patches, excluding padding.

    ``z`` is (S,P,K,D), the masks are (S,P). Hidden/slab queries remain in the
    average. ``visible_valid`` only determines whether a window has any actual
    visible key; it never changes the pooling support. Empty windows return zero.
    """
    if z.ndim != 4 or physical_valid.shape != z.shape[:2] or physical_valid.dtype != torch.bool:
        raise ValueError("contrastive pooling requires z (S,P,K,D), physical mask (S,P)")
    if visible_valid is not None and (visible_valid.shape != physical_valid.shape or visible_valid.dtype != torch.bool):
        raise ValueError("contrastive visible mask must be boolean (S,P)")
    dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=z.device.type, enabled=False):
        per_patch = z.to(dtype).mean(dim=2)
        per_patch = torch.where(physical_valid[..., None], per_patch, 0.0)
        pooled = per_patch.sum(dim=1) / physical_valid.sum(dim=1, keepdim=True).clamp(min=1)
    valid = physical_valid.any(dim=1)
    if visible_valid is not None:
        valid = valid & (physical_valid & visible_valid).any(dim=1)
    return pooled, valid


def normalize_projection(x: torch.Tensor, *, eps: float = 1e-8,
                         valid: torch.Tensor | None = None) -> torch.Tensor:
    """FP32 L2 normalization, failing on nonfinite/near-zero eligible projections.

    Invalid rows, if supplied, are replaced by zero BEFORE calculating norms;
    their NaNs and gradients cannot contaminate eligible rows.
    """
    if x.ndim != 2 or type(eps) not in (float, int) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("contrastive projection requires (N,D) and finite positive epsilon")
    if valid is None:
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    if valid.shape != x.shape[:1] or valid.dtype != torch.bool or valid.device != x.device:
        raise ValueError("contrastive projection validity must be boolean (N,)")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        clean = torch.where(valid[:, None], x.to(dtype), 0.0)
        norms = clean.norm(dim=-1, keepdim=True)
        if not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(norms).all()) or \
                bool((norms[valid] < eps).any()):
            raise ValueError("contrastive eligible projection is nonfinite or below norm floor")
        return clean / norms.clamp(min=eps)


def info_nce(query: torch.Tensor, key_positive: torch.Tensor, negative_keys: torch.Tensor,
             negative_mask: torch.Tensor, *, temperature: float = 0.2, min_negatives: int = 32,
             query_valid: torch.Tensor | None = None):
    """Return per-query loss, eligibility and detached count metrics.

    Inputs are L2-normalized projections. Positive keys are (N,D), shared negative
    keys (M,D), admissibility is (N,M). The positive is always in the denominator;
    fewer than ``min_negatives`` admissible keys means zero contribution. No
    renormalization over eligible queries happens here. All keys are detached.
    """
    if query.ndim != 2 or key_positive.shape != query.shape or negative_keys.ndim != 2 or \
            negative_keys.shape[1] != query.shape[1] or negative_mask.shape != (query.shape[0], negative_keys.shape[0]) or \
            negative_mask.dtype != torch.bool:
        raise ValueError("InfoNCE requires query/positive (N,D), negatives (M,D), boolean mask (N,M)")
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("InfoNCE temperature must be finite and positive")
    if type(min_negatives) is not int or min_negatives < 1:
        raise ValueError("InfoNCE min_negatives must be a positive integer")
    if query_valid is None:
        query_valid = torch.ones(query.shape[0], dtype=torch.bool, device=query.device)
    if query_valid.shape != query.shape[:1] or query_valid.dtype != torch.bool:
        raise ValueError("InfoNCE query validity must be boolean (N,)")
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=query.device.type, enabled=False):
        q = query.to(dtype)
        kp = key_positive.detach().to(dtype)
        kn = negative_keys.detach().to(dtype)
        if any(not bool(torch.isfinite(x).all()) for x in (q, kp, kn)):
            raise ValueError("InfoNCE projections must be finite")
        positive = (q * kp).sum(dim=-1) / temperature
        negatives = q @ kn.T / temperature
        if not bool(torch.isfinite(positive).all()) or not bool(torch.isfinite(negatives).all()):
            raise ValueError("InfoNCE logits overflowed")
        negatives = torch.where(negative_mask, negatives, -torch.inf)
        denominator = torch.logsumexp(torch.cat((positive[:, None], negatives), dim=1), dim=1)
        counts = negative_mask.sum(dim=1)
        eligible = query_valid & (counts >= min_negatives)
        value = torch.where(eligible, denominator - positive, 0.0)
    metrics = {"eligible": eligible.sum().detach(), "queries": eligible.new_tensor(eligible.numel(), dtype=torch.int64),
               "negative_counts": counts.detach()}
    return value, eligible, metrics
