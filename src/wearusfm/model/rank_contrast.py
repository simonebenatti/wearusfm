"""Rank-N-Contrast on instantaneous pose readouts, with symmetric pair selection.

This is the WearUsFM adaptation of RnC, not the paper's original recipe: cosine
similarity after parameter-free LayerNorm, RMS distance between standardized
angles, and division by each anchor's loss at uniform similarities. Every
admissible candidate is also a positive in its own ranking term. In particular,
same-window points are NEVER retained as negative-only candidates.

The caller supplies the candidate population (the experiment uses the global
batch). No collective or state update happens in this numerical module. Splitting
that population and averaging local losses does not reproduce the global loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from wearusfm.model.anchors import masked_sum_count

POSITIVES = ("all",)
_CHUNK = 256


def _validate_selection(stride: int, positives: str) -> None:
    if type(stride) is not int or stride < 1:
        raise ValueError("rank_contrast: stride must be an integer >= 1")
    if positives != "all":
        raise ValueError("rank_contrast: only symmetric 'all' positives are supported; cross is rejected")


def anchor_support(valid: torch.Tensor, stride: int, positives: str = "all",
                   pair_allowed: torch.Tensor | None = None):
    """Select anchors and symmetric candidate pairs from masks only.

    ``valid`` is (S,P,J), already intersected with physical patch existence.
    ``pair_allowed``, when supplied, is a symmetric (N,N) boolean mask on the
    selected anchors, ordered by window then patch. It removes pairs from BOTH
    roles. At least two candidates are required for an anchor to count.
    """
    if valid.ndim != 3 or valid.dtype != torch.bool:
        raise ValueError("rank_contrast: valid must be boolean (S,P,J)")
    _validate_selection(stride, positives)
    keep = torch.zeros(valid.shape[1], dtype=torch.bool, device=valid.device)
    keep[stride // 2::stride] = True
    s_idx, p_idx = (valid.any(dim=-1) & keep).nonzero(as_tuple=True)
    v = valid[s_idx, p_idx]
    with torch.autocast(device_type=valid.device.type, enabled=False):
        m = v.float()
        cand = (m @ m.T > 0) & ~torch.eye(s_idx.numel(), dtype=torch.bool, device=valid.device)
    if pair_allowed is not None:
        if pair_allowed.dtype != torch.bool or pair_allowed.shape != cand.shape or pair_allowed.device != cand.device:
            raise ValueError("rank_contrast: pair_allowed must be boolean (N,N) on the anchor device")
        if not torch.equal(pair_allowed, pair_allowed.T):
            raise ValueError("rank_contrast: pair_allowed must be symmetric")
        cand = cand & pair_allowed
    return s_idx, p_idx, v, cand, cand, cand.sum(dim=1) >= 2


def rank_contrast_count(valid: torch.Tensor, *, stride: int, positives: str = "all", per_window: bool = True,
                        pair_allowed: torch.Tensor | None = None) -> torch.Tensor:
    """Count windows or anchors before encode, on the SAME candidate population."""
    s_idx, _, _, _, _, ok = anchor_support(valid, stride, positives, pair_allowed)
    if not per_window:
        return ok.sum().detach()
    per_sample = torch.zeros(valid.shape[0], dtype=torch.int64, device=valid.device)
    return (per_sample.index_add(0, s_idx, ok.to(torch.int64)) > 0).sum().detach()


def readout_similarity(h: torch.Tensor) -> torch.Tensor:
    """Cosine similarity after coordinate centering/scaling, with no parameters."""
    if h.ndim != 2 or h.shape[-1] < 1 or not bool(torch.isfinite(h).all()):
        raise ValueError("rank_contrast: finite (N,D) readouts required")
    u = F.normalize(F.layer_norm(h, h.shape[-1:]), dim=-1)
    return u @ u.T


@torch.no_grad()
def pose_distance(y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pairwise RMS over shared valid joints, float64; masked NaNs are ignored."""
    if y.ndim != 2 or valid.shape != y.shape or valid.dtype != torch.bool:
        raise ValueError("rank_contrast: target and boolean validity must be (N,J)")
    if not bool(torch.isfinite(y[valid]).all()):
        raise ValueError("rank_contrast: nonfinite valid pose target")
    if y.shape[0] == 0:
        return torch.empty((0, 0), dtype=torch.float64, device=y.device)
    t = torch.where(valid, y.to(torch.float64), 0.0)
    m = valid.to(torch.float64)
    d2 = torch.cat([
        ((t[a:a + _CHUNK, None, :] - t[None, :, :]).square()
         * m[a:a + _CHUNK, None, :] * m[None, :, :]).sum(dim=-1)
        for a in range(0, t.shape[0], _CHUNK)])
    return (d2 / (m @ m.T).clamp(min=1.0)).sqrt()


def rank_contrast_pairs(logits: torch.Tensor, dist: torch.Tensor, cand: torch.Tensor,
                        pos: torch.Tensor | None = None):
    """Sum ranking terms, uniform-similarity reference and pair count per anchor.

    Sort distances, then use reverse cumulative logsumexp. Rows with equal
    candidate counts are batched and ONLY their valid prefix enters logsumexp:
    no finite sentinel contributes at small temperatures and no all--inf suffix
    can produce NaNs in the backward. Equal distances share the same denominator.
    ``pos`` is accepted for API clarity but must equal ``cand``.
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or dist.shape != logits.shape or \
            cand.shape != logits.shape or cand.dtype != torch.bool:
        raise ValueError("rank_contrast: square logits/dist and boolean candidate mask required")
    if pos is not None and (pos.dtype != torch.bool or not torch.equal(pos, cand)):
        raise ValueError("rank_contrast: candidates and positives must be identical")
    if bool(cand.diagonal().any()):
        raise ValueError("rank_contrast: self pairs are forbidden")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(dist[cand]).all()):
        raise ValueError("rank_contrast: nonfinite logits or candidate distance")
    d_sorted, order = torch.sort(torch.where(cand, dist, float("inf")), dim=1, stable=True)
    n_cand = cand.sum(dim=1)
    num = logits.sum(dim=1) * 0.0
    chance = torch.zeros_like(num)
    for size in torch.unique(n_cand).tolist():
        if size == 0:
            continue
        rows = (n_cand == size).nonzero(as_tuple=True)[0]
        labels = d_sorted[rows, :size].contiguous()
        scores = logits[rows].gather(1, order[rows, :size])
        tail = torch.logcumsumexp(scores.flip(1), dim=1).flip(1)
        first = torch.searchsorted(labels, labels)
        values = (tail.gather(1, first) - scores).sum(dim=1)
        ref = (size - first).to(scores.dtype).log().sum(dim=1)
        num = num.index_copy(0, rows, values)
        chance = chance.index_copy(0, rows, ref)
    return num, chance, n_cand


def pose_rank_contrast(z_mean: torch.Tensor, pose: tuple | None, time_valid: torch.Tensor, *, temperature: float,
                       stride: int, positives: str = "all", per_window: bool = True, resolution: float = 0.0,
                       pair_allowed: torch.Tensor | None = None):
    """Return (loss, numerator, int64 count, logs) on the supplied population.

    ``z_mean``: (S,P,D); ``pose``: (standardized targets, joint validity[, ...]);
    ``time_valid``: physical patches (S,P), including masked/slab queries.
    Optional pair filtering acts symmetrically. Gradients flow through all
    representations, including each representation's role as a candidate.
    """
    _validate_selection(stride, positives)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("rank_contrast: temperature must be finite and positive")
    if type(resolution) not in (int, float) or not math.isfinite(resolution) or resolution < 0:
        raise ValueError("rank_contrast: resolution must be finite and nonnegative")
    if z_mean.ndim != 3 or time_valid.shape != z_mean.shape[:2] or time_valid.dtype != torch.bool:
        raise ValueError("rank_contrast: readout (S,P,D) and boolean physical support (S,P) required")
    dtype = torch.float64 if z_mean.dtype == torch.float64 else torch.float32
    zero = z_mean.to(dtype).sum() * 0.0
    none = time_valid.new_zeros((), dtype=torch.int64)
    empty = (zero, zero, none, {"anchors": none, "pairs": none, "raw": zero.detach(), "chance": zero.detach()})
    if pose is None:
        return empty
    target, mask = pose[0], pose[1]
    if target.ndim != 3 or target.shape[:2] != z_mean.shape[:2] or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("rank_contrast: target and boolean mask must be (S,P,J)")
    s_idx, p_idx, v, cand, pos, ok = anchor_support(mask & time_valid[..., None], stride, positives, pair_allowed)
    if not bool(ok.any()):
        return empty
    y = target[s_idx, p_idx]
    with torch.autocast(device_type=z_mean.device.type, enabled=False):
        with torch.no_grad():
            dist = pose_distance(y, v)
            if resolution > 0:
                dist = torch.floor(dist / resolution) * resolution
        logits = readout_similarity(z_mean[s_idx, p_idx].to(dtype)) / temperature
        num, chance, n_pos = rank_contrast_pairs(logits, dist, cand, pos)
    value = torch.where(ok, num, 0.0) / torch.where(ok, chance, 1.0)
    r_num, count = masked_sum_count(value, ok, s_idx if per_window else None, z_mean.shape[0])
    per_pair = n_pos.clamp(min=1).to(dtype)
    log = {"anchors": ok.sum().detach(), "pairs": torch.where(ok, n_pos, 0).sum().detach(),
           "raw": torch.where(ok, num.detach() / per_pair, 0.0).sum(),
           "chance": torch.where(ok, chance / per_pair, 0.0).sum()}
    return r_num / count.clamp(min=1), r_num, count, log
