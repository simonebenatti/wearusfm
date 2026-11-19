"""Bounded-memory, weighted linear references for frozen pose readers.

No Gram eigenvalues are used to determine rank.  A streaming QR retains only
R, Q'Y and RHS norms.  WLS uses the SVD of the augmented design; ridge uses
its centered subfactor so the intercept is never penalized.  These are
different references, with different stationarity equations.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Iterable, Iterator

import numpy as np


class LinearReferenceError(ValueError):
    """A reference cannot certify the specified numerical contract."""


def _require(condition, message):
    if not condition:
        raise LinearReferenceError(message)


def layer_norm64(x, eps=1e-5):
    """Fixed gamma=1/beta=0 LN, recomputed in float64 from cache values."""
    x = np.asarray(x, dtype=np.float64)
    _require(x.ndim >= 2 and np.isfinite(x).all(), "nonfinite or invalid features")
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)


@dataclass
class LinearSolution:
    # Coefficients include the unpenalized intercept in row zero.
    coefficients: np.ndarray
    alpha: float
    rank: int
    singular_values: np.ndarray
    normal_residual_ratio: np.ndarray
    training_sse: np.ndarray
    weight_sum: float


class WeightedQR:
    """TSQR accumulator for a common design/weight pattern and several RHS.

    `add` receives features without an intercept. Weights are *not* normalized
    again: the caller supplies m_itj/q_i. Thus the ridge objective is the sum
    of window losses plus alpha times coefficient norm, not their mean.
    """

    def __init__(self, dimension: int, n_rhs: int, *, singular_cutoff=1e-8,
                 residual_tolerance=1e-6, max_factor_bytes=24 * 1024**3):
        _require(dimension > 0 and n_rhs > 0, "empty design")
        _require(0 < singular_cutoff < 1 and residual_tolerance > 0, "invalid tolerances")
        # R + SVD factors + stack/QR workspace. This is an explicit conservative
        # factor allowance, not an allocation measurement or a memory fallback.
        projected = 8 * (8 * (dimension + 1)**2 + 4 * (dimension + 1) * n_rhs)
        _require(projected <= max_factor_bytes, "factor storage exceeds declared bound")
        self.dimension, self.n_rhs = int(dimension), int(n_rhs)
        self.singular_cutoff = float(singular_cutoff)
        self.residual_tolerance = float(residual_tolerance)
        self.max_factor_bytes = int(max_factor_bytes)
        self.projected_factor_bytes = projected
        self.r = np.empty((0, dimension + 1), dtype=np.float64)
        self.qty = np.empty((0, n_rhs), dtype=np.float64)
        self.y_norm2 = np.zeros(n_rhs, dtype=np.float64)
        self._y_cross = np.zeros((n_rhs, n_rhs), dtype=np.float64)
        self.weight_sum = 0.0
        self.n_rows = 0
        self._aug_svd = None
        self._center_svd = None

    def add(self, x, y, weight):
        x, y, weight = (np.asarray(v, dtype=np.float64) for v in (x, y, weight))
        if y.ndim == 1:
            y = y[:, None]
        _require(x.ndim == 2 and x.shape[1] == self.dimension, "feature shape differs")
        _require(y.shape == (len(x), self.n_rhs) and weight.shape == (len(x),), "RHS/weight shape differs")
        _require(np.isfinite(weight).all() and (weight >= 0).all(), "invalid weights")
        keep = weight > 0
        if not keep.any():
            return self
        x, y, weight = x[keep], y[keep], weight[keep]
        _require(np.isfinite(x).all() and np.isfinite(y).all(), "nonfinite supported inputs")
        projected = (len(x) + len(self.r)) * (self.dimension + 1 + self.n_rhs) * 8 * 3
        _require(projected + self.projected_factor_bytes <= self.max_factor_bytes, "QR block exceeds declared bound")
        sw = np.sqrt(weight)
        a = np.concatenate((np.ones((len(x), 1)), x), axis=1) * sw[:, None]
        b = y * sw[:, None]
        stack_a = np.concatenate((self.r, a), axis=0)
        stack_b = np.concatenate((self.qty, b), axis=0)
        q, self.r = np.linalg.qr(stack_a, mode="reduced")
        self.qty = q.T @ stack_b
        self.y_norm2 += np.sum(b * b, axis=0)
        self._y_cross += b.T @ b
        self.weight_sum += float(weight.sum())
        self.n_rows += len(x)
        self._aug_svd = self._center_svd = None
        return self

    def _svd(self, centered=False):
        _require(self.n_rows > 0 and self.weight_sum > 0, "no supported rows")
        name = "_center_svd" if centered else "_aug_svd"
        if getattr(self, name) is None:
            matrix = self.r[1:, 1:] if centered else self.r
            setattr(self, name, np.linalg.svd(matrix, full_matrices=False))
        return getattr(self, name)

    def solve(self, alpha=0.0, rhs_mix=None):
        """Solve saved RHS or their exact linear combinations.

        A fixed-alpha ridge map is linear in RHS. Selecting a different alpha
        for each component and combining those solutions is not supported.
        """
        alpha = float(alpha)
        _require(np.isfinite(alpha) and alpha >= 0, "invalid alpha")
        mix = np.eye(self.n_rhs) if rhs_mix is None else np.asarray(rhs_mix, dtype=np.float64)
        if mix.ndim == 1:
            mix = mix[:, None]
        _require(mix.ndim == 2 and mix.shape[0] == self.n_rhs and np.isfinite(mix).all(), "invalid RHS mixture")
        qty = self.qty @ mix
        u, s, vt = self._svd()
        rank = int(np.count_nonzero(s > self.singular_cutoff * s[0]))
        if alpha == 0:
            inverse = np.zeros_like(s)
            inverse[:rank] = 1 / s[:rank]
            coef = (vt.T * inverse) @ (u.T @ qty)
        else:
            # First QR column is the weighted constant. Eliminating its first
            # row leaves precisely the centered weighted design and RHS.
            uc, sc, vtc = self._svd(centered=True)
            slope = (vtc.T * (sc / (sc * sc + alpha))) @ (uc.T @ qty[1:])
            intercept = (qty[0] - self.r[0, 1:] @ slope) / self.r[0, 0]
            coef = np.concatenate((intercept[None, :], slope), axis=0)
        projected_residual = self.r @ coef - qty
        normal = self.r.T @ projected_residual
        if alpha:
            normal[1:] += alpha * coef[1:]
        # For mixtures the full RHS norm is not available from diagonal norms
        # alone; compute it from the retained full RHS cross-product below.
        gram_y = getattr(self, "_y_cross", None)
        if gram_y is None:
            _require(rhs_mix is None or np.all(np.sum(mix != 0, axis=0) <= 1),
                     "mixed RHS needs cross-products; use add_with_cross")
            y2 = np.sum(self.y_norm2[:, None] * mix * mix, axis=0)
        else:
            y2 = np.einsum("ik,ij,jk->k", mix, gram_y, mix)
        scale = s[0] * np.sqrt(np.maximum(y2, 0))
        # Ridge adds its penalty scale; the unpenalized intercept contributes zero.
        if alpha:
            scale = scale + alpha * np.linalg.norm(coef[1:], axis=0)
        normal_norm = np.linalg.norm(normal, axis=0)
        ratios = np.divide(normal_norm, scale, out=np.zeros_like(scale), where=scale > 0)
        ratios[(scale == 0) & (normal_norm != 0)] = np.inf
        _require(np.isfinite(coef).all() and np.isfinite(ratios).all()
                 and (ratios <= self.residual_tolerance).all(), "normal residual failed")
        pred = self.r @ coef
        sse = y2 + np.sum(pred * pred - 2 * pred * qty, axis=0)
        tolerance = 1e-10 * np.maximum(y2, 1)
        _require((sse >= -tolerance).all(), "negative residual sum beyond roundoff")
        return LinearSolution(coef, alpha, rank, s.copy(), ratios,
                              np.maximum(sse, 0), self.weight_sum)

    def add_with_cross(self, x, y, weight):
        return self.add(x, y, weight)


def factorize(blocks: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]], **kwargs):
    factor = None
    for x, y, weight in blocks:
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        if factor is None:
            factor = WeightedQR(np.asarray(x).shape[1], y.shape[1], **kwargs)
        factor.add_with_cross(x, y, weight)
    _require(factor is not None and factor.n_rows > 0, "no supported design")
    return factor


def mask_groups(support):
    """Group identical joint masks, keeping every joint-specific pattern."""
    support = np.asarray(support, dtype=bool)
    _require(support.ndim == 3, "support must be windows,time,joints")
    groups = []
    buckets = {}
    for j in range(support.shape[-1]):
        digest = sha256(support[..., j].tobytes()).hexdigest()
        found = None
        for k in buckets.get(digest, []):
            if np.array_equal(support[..., groups[k][0]], support[..., j]):
                found = k
                break
        if found is None:
            buckets.setdefault(digest, []).append(len(groups))
            groups.append([j])
        else:
            groups[found].append(j)
    return groups


def design_block(features, *, context=0, crop=6, normalizer=layer_norm64):
    """Normalize a bounded batch and retain centers in [crop,T-crop)."""
    _require(isinstance(context, int) and 0 <= context <= crop, "invalid context/crop")
    z = normalizer(features)
    _require(z.ndim == 3 and z.shape[1] > 2 * crop, "invalid feature geometry")
    if context == 0:
        return z[:, crop:z.shape[1]-crop]
    windows = np.lib.stride_tricks.sliding_window_view(z, 2 * context + 1, axis=1)
    windows = windows[:, crop-context:windows.shape[1]-(crop-context) if crop > context else None]
    # Time-major tap ordering, with all feature coordinates inside each tap.
    return windows.transpose(0, 1, 3, 2).reshape(len(z), z.shape[1]-2*crop, -1)


def iter_window_blocks(features, targets, support, indices, joints, *, context=0,
                       crop=6, block_windows=4, normalizer=layer_norm64):
    """Yield bounded design blocks for one identical mask group.

    Features and targets have full time length; support has the cropped time
    length and must already enforce key_time_valid and actual window length.
    Targets may concatenate several basis RHS per joint outside this helper.
    """
    features, targets, support = np.asarray(features), np.asarray(targets), np.asarray(support, dtype=bool)
    indices, joints = np.asarray(indices, dtype=int), np.asarray(joints, dtype=int)
    _require(features.ndim == targets.ndim == support.ndim == 3, "invalid arrays")
    _require(targets.shape[:2] == features.shape[:2] and support.shape ==
             (len(features), features.shape[1]-2*crop, targets.shape[2]), "geometry differs")
    _require(block_windows >= 1 and len(joints) > 0, "empty batch/group")
    _require(len(np.unique(indices)) == len(indices) and ((indices >= 0) & (indices < len(features))).all(), "invalid or duplicate windows")
    _require(((joints >= 0) & (joints < targets.shape[-1])).all(), "joint outside target")
    for j in joints[1:]:
        _require(np.array_equal(support[indices, :, j], support[indices, :, joints[0]]), "joint masks differ within group")
    for start in range(0, len(indices), block_windows):
        take = indices[start:start + block_windows]
        x = design_block(features[take], context=context, crop=crop, normalizer=normalizer)
        n_valid = support[take].sum(axis=(1, 2))
        valid = support[take, :, joints[0]]
        w = np.divide(valid, n_valid[:, None], out=np.zeros(valid.shape, dtype=np.float64), where=n_valid[:, None] > 0)
        y = targets[take, crop:targets.shape[1]-crop][:, :, joints]
        yield x.reshape(-1, x.shape[-1]), y.reshape(-1, len(joints)), w.reshape(-1)


def predict_windowed(features, coefficients, *, context=0, crop=6,
                     block_windows=4, normalizer=layer_norm64):
    features, coefficients = np.asarray(features), np.asarray(coefficients, dtype=np.float64)
    _require(features.ndim == 3 and coefficients.ndim == 2, "invalid prediction arrays")
    dim = features.shape[-1] * (2 * context + 1)
    _require(coefficients.shape[0] == dim + 1, "coefficient design differs")
    prediction = np.empty((len(features), features.shape[1]-2*crop, coefficients.shape[1]), dtype=np.float64)
    for start in range(0, len(features), block_windows):
        x = design_block(features[start:start+block_windows], context=context, crop=crop, normalizer=normalizer)
        prediction[start:start+block_windows] = x @ coefficients[1:] + coefficients[0]
    _require(np.isfinite(prediction).all(), "nonfinite predictions")
    return prediction


def select_alpha(score: Callable[[float], float], *, initial_exponents=range(-2, 5),
                 extension_decades=3):
    """Finite inner-MSE selection; unresolved extreme is not calibratable.

    `score` must use fitting-outer users only. A cache avoids repeated alpha
    evaluation; exact ties choose the smaller alpha. No tolerance is silently
    substituted for the separately prescribed neural LR-edge tolerance.
    """
    initial = tuple(int(k) for k in initial_exponents)
    _require(initial and list(initial) == list(range(min(initial), max(initial)+1)), "alpha exponents must be contiguous")
    lo, hi = min(initial)-extension_decades, max(initial)+extension_decades
    scores = {}
    def evaluate(k):
        value = float(score(10.**k))
        _require(np.isfinite(value) and value >= 0, "invalid inner MSE")
        scores[k] = value
    for k in initial:
        evaluate(k)
    while True:
        winner = min(scores, key=lambda k: (scores[k], k))
        left, right = min(scores), max(scores)
        if winner == left and left > lo:
            evaluate(left-1)
        elif winner == right and right < hi:
            evaluate(right+1)
        else:
            boundary = (winner == left == lo or winner == right == hi)
            return {"alpha": 10.**winner, "mse": scores[winner],
                    "boundary_unresolved": boundary,
                    "status": "control_non_calibrated" if boundary else "selected",
                    "curve": [{"alpha": 10.**k, "mse": scores[k]} for k in sorted(scores)]}


class WindowedBasisFit:
    """Reuse each mask-pattern QR for several full-window RHS basis arrays."""

    def __init__(self, features, targets, support, indices, *, context=0, crop=6,
                 block_windows=4, normalizer=layer_norm64, max_factor_bytes=24*1024**3):
        self.features = np.asarray(features)
        self.targets = {k: np.asarray(v) for k, v in targets.items()}
        self.names = tuple(self.targets)
        self.support = np.asarray(support, dtype=bool)
        self.indices = np.asarray(indices, dtype=int)
        self.context, self.crop, self.block_windows = context, crop, block_windows
        self.normalizer = normalizer
        _require(self.names and len(self.indices) > 0, "empty fitting basis")
        shape = next(iter(self.targets.values())).shape
        _require(len(shape) == 3 and all(v.shape == shape for v in self.targets.values()), "basis shapes differ")
        _require(shape[:2] == self.features.shape[:2] and self.support.shape ==
                 (shape[0], shape[1]-2*crop, shape[2]), "basis support geometry differs")
        _require(len(np.unique(self.indices)) == len(self.indices) and
                 ((self.indices >= 0) & (self.indices < shape[0])).all(), "fitting indices invalid")
        self.n_joints = shape[-1]
        self.dimension = self.features.shape[-1]*(2*context+1)
        groups = mask_groups(self.support[self.indices])
        # Patterns are kept simultaneously because alpha paths reuse them.
        projected = sum(8*(8*(self.dimension+1)**2 + 4*(self.dimension+1)*len(g)*len(self.names)) for g in groups)
        _require(projected <= max_factor_bytes, "all mask factors exceed declared bound")
        self.factors = []
        for group in groups:
            factor = WeightedQR(self.dimension, len(group)*len(self.names), max_factor_bytes=max_factor_bytes)
            for start in range(0, len(self.indices), block_windows):
                take = self.indices[start:start+block_windows]
                x = design_block(self.features[take], context=context, crop=crop, normalizer=normalizer)
                q = self.support[take].sum(axis=(1, 2))
                valid = self.support[take, :, group[0]]
                w = np.divide(valid, q[:, None], out=np.zeros(valid.shape, dtype=np.float64), where=q[:, None] > 0)
                y = np.concatenate([v[take, crop:shape[1]-crop][:, :, group] for v in self.targets.values()], axis=-1)
                factor.add(x.reshape(-1, self.dimension), y.reshape(-1, len(group)*len(self.names)), w.reshape(-1))
            _require(factor.n_rows > 0, "joint mask pattern has no fitting support")
            self.factors.append((group, factor))

    def coefficients(self, alpha=0.0):
        coefficients = {k: np.empty((self.dimension+1, self.n_joints), dtype=np.float64) for k in self.names}
        diagnostics = []
        for group, factor in self.factors:
            solution = factor.solve(alpha)
            for j, name in enumerate(self.names):
                coefficients[name][:, group] = solution.coefficients[:, j*len(group):(j+1)*len(group)]
            diagnostics.append({"joints": list(group), "rank": solution.rank,
                                "normal_residual_max": float(solution.normal_residual_ratio.max()),
                                "weight_sum": solution.weight_sum,
                                "rows": factor.n_rows,
                                "training_sse_by_basis_joint": solution.training_sse.reshape(len(self.names), len(group)).tolist()})
        return coefficients, diagnostics

    def predictions(self, indices, alpha=0.0):
        coefficients, diagnostics = self.coefficients(alpha)
        indices = np.asarray(indices, dtype=int)
        _require(indices.ndim == 1 and ((indices >= 0) & (indices < len(self.features))).all(), "prediction index invalid")
        predictions = {k: np.empty((len(indices), self.features.shape[1]-2*self.crop, self.n_joints)) for k in self.names}
        # Copy only bounded feature batches; no full held feature expansion.
        for start in range(0, len(indices), self.block_windows):
            take = indices[start:start+self.block_windows]
            x = design_block(self.features[take], context=self.context, crop=self.crop, normalizer=self.normalizer)
            for name, coef in coefficients.items():
                predictions[name][start:start+len(take)] = x@coef[1:] + coef[0]
        return predictions, diagnostics

    def mixed_coefficients(self, weights, alpha=0.0):
        """Check the normal residual of the actual combined RHS, not just bases."""
        _require(set(weights) <= set(self.names) and all(np.isfinite(v) for v in weights.values()), "invalid basis weights")
        coefficients = np.empty((self.dimension+1, self.n_joints), dtype=np.float64)
        diagnostics = []
        for group, factor in self.factors:
            mixture = np.concatenate([np.eye(len(group))*weights.get(name, 0.) for name in self.names], axis=0)
            solution = factor.solve(alpha, rhs_mix=mixture)
            coefficients[:, group] = solution.coefficients
            diagnostics.append({"joints": list(group), "rank": solution.rank,
                                "normal_residual_max": float(solution.normal_residual_ratio.max()),
                                "training_sse": solution.training_sse.tolist(),
                                "weight_sum": solution.weight_sum})
        return coefficients, diagnostics

    def predict_mixture(self, indices, weights, alpha=0.0):
        coefficients, diagnostics = self.mixed_coefficients(weights, alpha)
        indices = np.asarray(indices, dtype=int)
        _require(indices.ndim == 1 and ((indices >= 0) & (indices < len(self.features))).all(), "prediction index invalid")
        prediction = np.empty((len(indices), self.features.shape[1]-2*self.crop, self.n_joints), dtype=np.float64)
        for start in range(0, len(indices), self.block_windows):
            take = indices[start:start+self.block_windows]
            x = design_block(self.features[take], context=self.context, crop=self.crop, normalizer=self.normalizer)
            prediction[start:start+len(take)] = x@coefficients[1:] + coefficients[0]
        return prediction, diagnostics
