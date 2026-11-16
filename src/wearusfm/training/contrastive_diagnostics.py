"""Descriptive contrastive geometry diagnostics, with no state or random draws.

Imports that can initialize numerical libraries stay inside the call: the
trainer invokes this only after initialization, on declared diagnostic steps.
"""
from __future__ import annotations


def cross_window_rank_alignment(z_global, target_global, valid_global,
                                tv_global, stride, pair_allowed):
    """Spearman(cosine, -pose RMS) on undirected valid cross-window pairs.

    Inputs and anchor order match ``pose_rank_contrast``: window-major, then
    patches ``stride//2::stride`` with at least one valid physical pose joint.
    ``pair_allowed`` removes identical physical instants symmetrically. The
    diagnostic retains each eligible pair once, uses average ranks for ties,
    and never counts within-window pairs. Its observations are correlated;
    neither pair count nor correlation is an independent-sample significance
    test. Empty/one-pair/constant support emits valid=0 and correlation=0.
    """
    import numpy as np
    import torch
    from scipy.stats import rankdata
    from wearusfm.model.rank_contrast import anchor_support, pose_distance, readout_similarity

    prefix = "contrast_rnc_cross_window_rank_"
    result = {prefix + "correlation": 0.0, prefix + "valid": 0.0,
              prefix + "pairs": 0.0}
    if z_global.ndim != 3 or tv_global.shape != z_global.shape[:2] or tv_global.dtype != torch.bool:
        raise ValueError("rank alignment: readout (B,P,D) and boolean physical support (B,P) required")
    if target_global.ndim != 3 or target_global.shape[:2] != z_global.shape[:2] or \
            valid_global.shape != target_global.shape or valid_global.dtype != torch.bool:
        raise ValueError("rank alignment: target and boolean validity must be (B,P,J)")
    with torch.no_grad(), torch.autocast(device_type=z_global.device.type, enabled=False):
        s, p, joints, candidates, _, _ = anchor_support(
            valid_global & tv_global[..., None], stride, "all", pair_allowed)
        pairs = torch.triu(candidates & (s[:, None] != s[None, :]), diagonal=1)
        result[prefix + "pairs"] = float(pairs.sum().item())
        if result[prefix + "pairs"] < 2:
            return result
        dtype = torch.float64 if z_global.dtype == torch.float64 else torch.float32
        similarity = readout_similarity(z_global[s, p].to(dtype))[pairs]
        distance = pose_distance(target_global[s, p], joints)[pairs]
        x = similarity.detach().double().cpu().numpy()
        y = -distance.detach().double().cpu().numpy()
    # Only eligible values reach rankdata; masked target NaNs remain excluded.
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("rank alignment: nonfinite eligible similarity/distance")
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    rx, ry = rx-rx.mean(), ry-ry.mean()
    denominator = float(np.sqrt(np.dot(rx, rx)*np.dot(ry, ry)))
    if denominator == 0.0:
        return result
    result[prefix + "correlation"] = float(np.clip(np.dot(rx, ry)/denominator, -1., 1.))
    result[prefix + "valid"] = 1.0
    return result
