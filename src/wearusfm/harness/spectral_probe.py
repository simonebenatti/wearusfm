"""Frozen ridge: train-only scaling, validation-only alpha, disjoint subjects."""
import numpy as np
from wearusfm.model.spectral_step1 import FAMILIES


def validate_splits(subject, split):
    subject, split = np.asarray(subject), np.asarray(split)
    if subject.ndim != 1 or split.shape != subject.shape or subject.dtype.kind not in "US" or split.dtype.kind not in "US":
        raise ValueError("subject/split must be string vectors")
    if set(split.tolist()) != {"train", "val", "test"}:
        raise ValueError("nonempty train/val/test required")
    if any("/" not in s for s in subject.tolist()):
        raise ValueError("subject IDs must be dataset-qualified")
    groups = {r: set(subject[split == r].tolist()) for r in ("train", "val", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("subject leakage between splits")


def r2(y, pred):
    if len(y) < 2:
        return None
    denom = np.square(y-y.mean()).sum()
    return float(1-np.square(y-pred).sum()/denom) if denom > 1e-12 else None


def spectral_probe(x, y, valid, subject, split, alphas=(.1, 1., 10., 100.)):
    x, y, valid = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(valid)
    subject, split = np.asarray(subject), np.asarray(split)
    validate_splits(subject, split)
    if x.ndim != 2 or y.shape != (len(x), 32) or valid.shape != y.shape or valid.dtype != bool or len(subject) != len(x):
        raise ValueError("invalid probe shapes/dtype")
    if not np.isfinite(x).all() or not np.isfinite(y[valid]).all():
        raise ValueError("nonfinite observed features/targets")
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("alphas must be positive finite numbers")
    rows, cache = [], {}
    for j in range(32):
        masks = {r: (split == r) & valid[:, j] for r in ("train", "val", "test")}
        a, b, c = (masks[r] for r in ("train", "val", "test"))
        row = {"coordinate": j, "counts": {r: int(m.sum()) for r, m in masks.items()}, "r2": None, "alpha": None}
        if min(row["counts"].values()) < 2 or np.std(y[a, j]) < 1e-6 or np.std(y[b, j]) < 1e-6:
            row["status"] = "insufficient_or_constant_train_validation"
            rows.append(row); continue
        key = a.tobytes()
        if key not in cache:
            mean, std = x[a].mean(0), x[a].std(0)
            z = (x-mean)/np.where(std > 1e-8, std, 1.)
            u, singular, vt = np.linalg.svd(z[a], full_matrices=False)
            cache[key] = z, u, singular, vt
        z, u, singular, vt = cache[key]
        ym, ys = y[a, j].mean(), y[a, j].std()
        rhs = u.T @ ((y[a, j]-ym)/ys)
        candidates = []
        for alpha in alphas:
            coef = vt.T @ ((singular/(singular**2+alpha))*rhs)
            candidates.append((float(np.square((z[b]@coef)*ys+ym-y[b, j]).mean()), alpha, coef))
        _, alpha, coef = min(candidates, key=lambda t: (t[0], t[1]))
        prediction = (z[c]@coef)*ys+ym
        row.update(alpha=float(alpha), r2=r2(y[c, j], prediction), status="ok", per_subject={})
        for s in sorted(set(subject[c].tolist())):
            sub = subject[c] == s
            row["per_subject"][s] = {"n": int(sub.sum()), "r2": r2(y[c, j][sub], prediction[sub])}
        if row["r2"] is None:
            row["status"] = "constant_test"
        rows.append(row)
    summary = {}
    for name, cols in FAMILIES.items():
        values = [row["r2"] for row in rows[cols] if row["r2"] is not None]
        summary[name] = {"r2_macro": float(np.mean(values)) if values else None, "n_assessable": len(values)}
    return {"families": summary, "coordinates": rows, "alphas": list(alphas), "bootstrap": "not_performed"}
