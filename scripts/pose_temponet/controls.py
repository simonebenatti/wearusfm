"""Train-only synthetic controls and finite, predeclared admission arithmetic.

Calibration receives inner-fit evaluations only. Outer held values enter
separate diagnostic/admission functions, never a grid or alpha selector.
This module does not train a neural head or open a development cache.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Callable, Mapping

import numpy as np

from .linear import WindowedBasisFit, select_alpha


SIGMA_GRID = tuple(k / 20 for k in range(121))
A_GRID = tuple(k / 5 for k in range(1, 16))
E_REAL_TOLERANCE = 0.05
TEMPORAL_GAP = (0.06, 0.10)
LINEAR_REFERENCE_RECOVERY = 0.90
LINEAR_HEAD_RECOVERY = 0.90
TEMPORAL_HEAD_RECOVERY = 0.50


class GateNotPassed(ValueError):
    """A measured scientific/operational gate, distinct from a runtime failure."""

    def __init__(self, message, *, status=None, gate=None):
        super().__init__(message)
        self.status = status
        self.gate = gate


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def window_noise(seed, metadata, n_time, n_joints, *, rho=np.exp(-1/4)):
    """Stationary AR(1), independently keyed by window metadata and joint."""
    _require(0 <= rho < 1 and n_time > 0 and n_joints > 0, "invalid noise geometry")
    out = np.empty((len(metadata), n_time, n_joints), dtype=np.float64)
    innovation_scale = np.sqrt(1-rho*rho)
    for i, meta in enumerate(metadata):
        for j in range(n_joints):
            key = canonical([int(seed), meta, j]).encode("utf-8")
            derived = int.from_bytes(sha256(key).digest()[:8], "little", signed=False)
            rng = np.random.Generator(np.random.PCG64(derived))
            innovations = rng.standard_normal(n_time)
            out[i, 0, j] = innovations[0]
            for t in range(1, n_time):
                out[i, t, j] = rho*out[i, t-1, j] + innovation_scale*innovations[t]
    return out


def validate_outer_partition(metadata, fit_indices, held_indices, allowed_train_users):
    """Reject overlap, non-train users, duplicate indices and split users."""
    users = np.asarray([str(m["user"]) for m in metadata])
    allowed = {str(x) for x in allowed_train_users}
    _require(set(users) <= allowed and allowed, "control input contains non-training users")
    fit, held = np.asarray(fit_indices, dtype=int), np.asarray(held_indices, dtype=int)
    for name, idx in (("fit", fit), ("held", held)):
        _require(idx.ndim == 1 and len(idx) > 0 and len(np.unique(idx)) == len(idx), name+" indices invalid")
        _require(((idx >= 0) & (idx < len(users))).all(), name+" index outside input")
    _require(not np.intersect1d(fit, held).size, "fit/held windows overlap")
    _require(not (set(users[fit]) & set(users[held])), "fit/held users overlap")
    return {"fitting_users": sorted(set(users[fit])), "held_users": sorted(set(users[held]))}


def inner_partitions(metadata, outer_fit_indices):
    """Alternating sorted user halves entirely within fitting-outer windows."""
    users = np.asarray([str(m["user"]) for m in metadata])
    outer = np.asarray(outer_fit_indices, dtype=int)
    _require(outer.ndim == 1 and len(np.unique(outer)) == len(outer), "invalid outer indices")
    _require(((outer >= 0) & (outer < len(users))).all(), "outer index outside input")
    unique = sorted(set(users[outer]))
    _require(len(unique) >= 2, "inner CV requires at least two fitting users")
    halves = [set(unique[::2]), set(unique[1::2])]
    return [(outer[np.isin(users[outer], sorted(halves[k]))],
             outer[np.isin(users[outer], sorted(halves[1-k]))]) for k in range(2)]


def normalize_component(component, support, fit_indices, *, crop=6):
    """Weighted outer-fitting normalization, applied to all 160 output times."""
    value = np.asarray(component, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    fit = np.asarray(fit_indices, dtype=int)
    _require(value.ndim == 3 and support.shape == (len(value), value.shape[1]-2*crop, value.shape[2]),
             "component/support geometry differs")
    _require(len(fit) > 0 and ((fit >= 0) & (fit < len(value))).all(), "invalid fitting subset")
    y, mask = value[fit, crop:value.shape[1]-crop], support[fit]
    q = mask.sum(axis=(1, 2))
    weight = np.divide(mask, q[:, None, None], out=np.zeros(mask.shape, dtype=np.float64),
                       where=q[:, None, None] > 0)
    total = weight.sum(axis=(0, 1))
    _require(np.isfinite(value).all() and (total > 0).all(), "nonfinite or unsupported teacher component")
    mean = np.sum(y*weight, axis=(0, 1))/total
    variance = np.sum((y-mean)**2*weight, axis=(0, 1))/total
    scale = np.sqrt(variance)
    if not (scale > 1e-8).all():
        raise GateNotPassed("control_non_calibrated: degenerate teacher component",
                            status="control_non_calibrated",
                            gate={"status": "control_non_calibrated", "reason": "degenerate_teacher_component",
                                  "std": scale.tolist(), "minimum_std": 1e-8,
                                  "fitting_windows": len(fit)})
    return (value-mean)/scale, {"mean": mean.tolist(), "std": scale.tolist(),
                              "weight_sum": total.tolist(), "fitting_windows": len(fit)}


def _state_hash(state):
    digest = sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(canonical([name, str(a.dtype), list(a.shape)]).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def generate_components(features, metadata, support, fit_indices, held_indices,
                        allowed_train_users, *, outer_fold, n_joints=20,
                        block_windows=16, crop=6):
    """Known affine and width-8 temporal teachers on real train latents.

    Teachers and their forward use pinned Torch CPU FP32, exactly like the
    nonlinear readers. Linear-reference designs independently recompute LN in
    float64; that numerical distinction is explicit, not a bitwise claim.
    No statistics or parameter selection use outer-held windows.
    """
    import torch
    from torch import nn
    partition = validate_outer_partition(metadata, fit_indices, held_indices, allowed_train_users)
    x = np.asarray(features)
    _require(x.ndim == 3 and len(x) == len(metadata), "features/metadata differ")
    _require(outer_fold in (0, 1) and block_windows > 0, "invalid generator fold/batch")
    base_seed = 20261118 + 1000*outer_fold
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base_seed)
        linear = nn.Linear(x.shape[-1], n_joints, bias=True)
        nn.init.normal_(linear.weight, std=x.shape[-1]**-.5)
        nn.init.zeros_(linear.bias)
        torch.manual_seed(base_seed+1)
        temporal = nn.Sequential(
            nn.Conv1d(x.shape[-1], 8, 1), nn.ReLU(),
            nn.Conv1d(8, 8, 3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 8, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv1d(8, 8, 3, dilation=3, padding=3), nn.ReLU(),
            nn.Conv1d(8, 8, 1), nn.ReLU(), nn.Conv1d(8, n_joints, 1))
        convolutions = [m for m in temporal if isinstance(m, nn.Conv1d)]
        for m in convolutions[:-1]:
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.normal_(convolutions[-1].weight, std=8**-.5)
        nn.init.zeros_(convolutions[-1].bias)
    linear.eval(); temporal.eval()
    arrays = {"L": np.empty((len(x), x.shape[1], n_joints), dtype=np.float64),
              "T": np.empty((len(x), x.shape[1], n_joints), dtype=np.float64)}
    active, total = np.zeros(5, dtype=np.int64), np.zeros(5, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(x), block_windows):
            tensor = torch.as_tensor(np.array(x[start:start+block_windows], dtype=np.float32))
            z = torch.nn.functional.layer_norm(tensor, (x.shape[-1],), eps=1e-5)
            arrays["L"][start:start+block_windows] = linear(z).numpy()
            h = z.transpose(1, 2)
            relu = 0
            for module in temporal:
                h = module(h)
                if isinstance(module, nn.ReLU):
                    # Descriptive activation counts use fitting-outer rows only.
                    selected = np.flatnonzero(np.isin(np.arange(start, start+len(z)), fit_indices))
                    if len(selected):
                        part = h[selected]
                        active[relu] += int((part > 0).sum())
                        total[relu] += part.numel()
                    relu += 1
            arrays["T"][start:start+block_windows] = h.transpose(1, 2).numpy()
    normalization = {}
    for key in ("L", "T"):
        arrays[key], normalization[key] = normalize_component(arrays[key], support, fit_indices, crop=crop)
    arrays["epsilon"] = window_noise(base_seed+2, metadata, x.shape[1], n_joints)
    return {**arrays, "normalization": normalization, "partition": partition,
            "generator_states": {"L": linear.state_dict(), "T": temporal.state_dict()},
            "generator_hashes": {"L": _state_hash(linear.state_dict()), "T": _state_hash(temporal.state_dict())},
            "hidden_active_fraction_fitting": np.divide(active, total, out=np.zeros(5), where=total > 0).tolist(),
            "seed": base_seed, "dtype": "torch.float32", "layernorm_eps": 1e-5}


def synthetic_target(components, a, sigma):
    _require(np.isfinite(a) and np.isfinite(sigma) and a >= 0 and sigma >= 0, "invalid synthetic coefficients")
    clean = np.asarray(components["L"]) + float(a)*np.asarray(components["T"])
    observed = clean + float(sigma)*np.asarray(components["epsilon"])
    _require(clean.shape == observed.shape and np.isfinite(observed).all(), "invalid synthetic target")
    return clean, observed


def calibrate_grid(real_reference: Mapping, evaluate_pair: Callable[[float, float], Mapping]):
    """Select bounded worlds using *only* inner held values from fitting outer.

    `real_reference` holds E_dyn and the alpha-selection boundary flag.
    `evaluate_pair(a,sigma)` returns E_cal, E_cleanLS, E_oracle and
    boundary_unresolved for the ridge on observed synthetic y.  A boundary
    excludes that pair, not every other pair.  No neural score is accepted.
    """
    value = real_reference["E_dyn"]
    e_real = float(value) if value is not None else float("nan")
    if not np.isfinite(e_real) or e_real < 0:
        return {"status": "control_non_diagnostic", "reason": "invalid real dynamic denominator/reference", "evaluations": 0}
    if real_reference.get("boundary_unresolved", False):
        return {"status": "control_non_calibrated", "reason": "real ridge reference unresolved", "evaluations": 0}
    rows, admitted = [], {"linear": [], "temporal": []}
    for a in (0.0, *A_GRID):
        for sigma in SIGMA_GRID:
            row = dict(evaluate_pair(a, sigma))
            _require(not ({"neural", "held_outer", "development"} & row.keys()), "forbidden calibration score source")
            for key in ("E_cal", "E_cleanLS", "E_oracle"):
                row[key] = float(row[key]) if row[key] is not None else float("nan")
            row.update(a=a, sigma=sigma)
            finite = all(np.isfinite(row[k]) and row[k] >= 0 for k in ("E_cal", "E_cleanLS", "E_oracle"))
            gain = ((row["E_cleanLS"]-row["E_oracle"])/row["E_cleanLS"]
                    if finite and row["E_cleanLS"] > 0 else None)
            row["oracle_gain_fraction"] = gain
            close = finite and abs(row["E_cal"]-e_real) <= E_REAL_TOLERANCE
            acceptable = close and not row.get("boundary_unresolved", False)
            if a > 0:
                acceptable = acceptable and gain is not None and TEMPORAL_GAP[0] <= gain <= TEMPORAL_GAP[1]
            row["admissible"] = bool(acceptable)
            if not finite:
                row["non_diagnostic_reason"] = "nonfinite or negative score/denominator"
                for key in ("E_cal", "E_cleanLS", "E_oracle"):
                    if not np.isfinite(row[key]):
                        row[key] = None
            rows.append(row)
            if acceptable:
                admitted["linear" if a == 0 else "temporal"].append(row)
    selected = {}
    if admitted["linear"]:
        selected["linear"] = min(admitted["linear"], key=lambda r: (abs(r["E_cal"]-e_real), r["sigma"]))
    if admitted["temporal"]:
        selected["temporal"] = min(admitted["temporal"], key=lambda r:
                                  (abs(r["oracle_gain_fraction"]-.08), abs(r["E_cal"]-e_real), r["a"], r["sigma"]))
    return {"status": "calibrated" if len(selected) == 2 else "control_non_calibrated",
            "E_real": e_real, "selected": selected, "evaluations": len(rows),
            "missing_worlds": sorted(set(admitted)-set(selected)), "curve": rows}


def _recovery(baseline, candidate, reference):
    if any(x is None for x in (baseline, candidate, reference)):
        return None
    baseline, candidate, reference = map(float, (baseline, candidate, reference))
    if not all(np.isfinite(x) and x >= 0 for x in (baseline, candidate, reference)):
        return None
    margin = baseline-reference
    if margin <= 0:
        return None
    return (baseline-candidate)/margin


def linear_reference_gate(records):
    """Records must include both folds and pooled, each with MSE and E_dyn."""
    _require(set(records) == {"fold0", "fold1", "pooled"}, "linear reference panel incomplete")
    result = {}
    for split, metrics in records.items():
        _require(set(metrics) == {"MSE", "E_dyn"}, "linear reference metrics differ")
        result[split] = {name: _recovery(v["constant"], v["ridge"], v["oracle"]) for name, v in metrics.items()}
    good = all(v is not None and v >= LINEAR_REFERENCE_RECOVERY for metrics in result.values() for v in metrics.values())
    return {"status": "diagnostic" if good else "control_non_diagnostic", "recovery": result}


def linear_head_gate(reference_records, head_records, *, optimization_alarms=()):
    reference = linear_reference_gate(reference_records)
    if reference["status"] != "diagnostic":
        return reference
    _require(set(head_records) == set(reference_records), "head/reference panels differ")
    recovered = {}
    for split, metrics in reference_records.items():
        _require(set(head_records[split]) == set(metrics), "head metrics differ")
        recovered[split] = {name: _recovery(v["constant"], head_records[split][name], v["ridge"])
                            for name, v in metrics.items()}
    passed = all(v is not None and v >= LINEAR_HEAD_RECOVERY for d in recovered.values() for v in d.values())
    return {"status": "passed" if passed else ("optimization_unresolved" if optimization_alarms else "recipe_not_admitted"),
            "recovery": recovered, "optimization_alarms": list(optimization_alarms)}


def temporal_reference_gate(records):
    _require(set(records) == {"fold0", "fold1", "pooled"}, "temporal reference panel incomplete")
    gain = {}
    for split, row in records.items():
        if row["clean_ls"] is None or row["oracle"] is None:
            gain[split] = None
            continue
        ls, oracle = float(row["clean_ls"]), float(row["oracle"])
        gain[split] = (ls-oracle)/ls if np.isfinite(ls) and np.isfinite(oracle) and ls > 0 and oracle >= 0 else None
    good = all(x is not None and TEMPORAL_GAP[0] <= x <= TEMPORAL_GAP[1] for x in gain.values())
    return {"status": "diagnostic" if good else "control_non_diagnostic", "oracle_gain_fraction": gain}


def temporal_head_gate(reference_records, head_records, *, optimization_alarms=(), mlp_records=None):
    reference = temporal_reference_gate(reference_records)
    if reference["status"] != "diagnostic":
        return reference
    _require(set(head_records) == set(reference_records), "temporal head panel incomplete")
    recovered = {split: _recovery(row["clean_ls"], head_records[split], row["oracle"])
                 for split, row in reference_records.items()}
    good = all(x is not None and x >= TEMPORAL_HEAD_RECOVERY for x in recovered.values())
    result = {"status": "passed" if good else ("optimization_unresolved" if optimization_alarms else "recipe_not_admitted"),
              "recovery": recovered, "optimization_alarms": list(optimization_alarms),
              "mlp_is_descriptive_only": True}
    if mlp_records is not None:
        _require(set(mlp_records) == set(reference_records), "MLP descriptive panel incomplete")
        result["mlp_recovery"] = {split: _recovery(row["clean_ls"], mlp_records[split], row["oracle"])
                                  for split, row in reference_records.items()}
        result["interpretation"] = "A nearby MLP does not qualify a specifically temporal reading."
    return result


def _quadratic_panel(error_basis, target_basis, support, *, block_windows=16):
    """Per-window 3x3 sufficient statistics; no candidate prediction caches."""
    support = np.asarray(support, dtype=bool)
    shape = support.shape
    _require(len(error_basis) == len(target_basis) > 0 and
             all(np.asarray(v).shape == shape for v in (*error_basis, *target_basis)), "quadratic basis shape differs")
    n_basis = len(error_basis)
    mse, error_dyn, target_dyn = [np.zeros((len(support), n_basis, n_basis), dtype=np.float64) for _ in range(3)]
    supported = support.sum(axis=(1, 2)) > 0
    for start in range(0, len(support), block_windows):
        sl = slice(start, start+block_windows)
        mask = support[sl]
        e = np.stack([v[sl] for v in error_basis], axis=-1).astype(np.float64)
        y = np.stack([v[sl] for v in target_basis], axis=-1).astype(np.float64)
        _require(np.isfinite(e[mask]).all() and np.isfinite(y[mask]).all(), "nonfinite supported basis")
        e, y = np.where(mask[..., None], e, 0), np.where(mask[..., None], y, 0)
        q = mask.sum(axis=(1, 2))
        mse[sl] = np.einsum("ntjk,ntjl->nkl", e, e)/np.maximum(q[:, None, None], 1)
        count = mask.sum(axis=1)
        eligible = mask & (count[:, None, :] >= 2)
        for values, out in ((e, error_dyn), (y, target_dyn)):
            mean = values.sum(axis=1)/np.maximum(count[..., None], 1)
            centered = np.where(eligible[..., None], values-mean[:, None], 0)
            out[sl] = np.einsum("ntjk,ntjl->nkl", centered, centered)
    return {"mse": mse, "dynamic_error": error_dyn, "dynamic_target": target_dyn, "supported": supported}


def _pose_variance_gram(basis, mask, *, block_windows=16):
    """Full-window mean-joint population variance as a quadratic form."""
    mask = np.asarray(mask, dtype=bool)
    result = np.full((len(mask), len(basis), len(basis)), np.nan, dtype=np.float64)
    _require(all(np.asarray(v).shape == mask.shape for v in basis), "variance basis shape differs")
    for start in range(0, len(mask), block_windows):
        sl = slice(start, start+block_windows)
        m = mask[sl]
        values = np.stack([v[sl] for v in basis], axis=-1).astype(np.float64)
        _require(np.isfinite(values[m]).all(), "nonfinite supported variance basis")
        values = np.where(m[..., None], values, 0)
        count = m.sum(axis=1)
        mean = values.sum(axis=1)/np.maximum(count[..., None], 1)
        centered = np.where(m[..., None], values-mean[:, None], 0)
        gram = np.einsum("ntjk,ntjl->njkl", centered, centered)/np.maximum(count[..., None, None], 1)
        eligible = count >= 2
        n = eligible.sum(axis=1)
        block = np.sum(np.where(eligible[..., None, None], gram, 0), axis=1)/np.maximum(n[:, None, None], 1)
        block[n == 0] = np.nan
        result[sl] = block
    return result


def _quad(matrix, coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return np.einsum("k,nkl,l->n", coefficients, matrix, coefficients)


def _threshold(variances):
    finite = np.asarray(variances)[np.isfinite(variances)]
    _require(len(finite) > 0, "control_non_diagnostic: no fitting dynamic support")
    return max(float(np.median(finite)), 1e-12)


def score_quadratic_panel(panel, coefficients, dynamic_selected):
    """Raw additive moments, pooled by summation before either ratio."""
    selected = np.asarray(dynamic_selected, dtype=bool)
    _require(selected.shape == panel["supported"].shape, "dynamic selector shape differs")
    mse = float(_quad(panel["mse"], coefficients)[panel["supported"]].sum())
    dynamic_sse = float(_quad(panel["dynamic_error"], coefficients)[selected].sum())
    dynamic_target = float(_quad(panel["dynamic_target"], coefficients)[selected].sum())
    # Roundoff in a PSD quadratic form is tolerated, not negative real error.
    scale = max(float(np.abs(panel["mse"]).sum()), float(np.abs(panel["dynamic_error"]).sum()), 1.)
    _require(min(mse, dynamic_sse, dynamic_target) >= -1e-12*scale, "negative quadratic error")
    return {"mse_sum": max(mse, 0.), "window_count": int(panel["supported"].sum()),
            "dynamic_sse": max(dynamic_sse, 0.), "dynamic_target_ss": max(dynamic_target, 0.)}


def aggregate_quadratic_panel(panel, dynamic_selected):
    """Sum fixed-a masks once; scanning sigma then touches only 3x3 matrices."""
    selected = np.asarray(dynamic_selected, dtype=bool)
    _require(selected.shape == panel["supported"].shape, "dynamic selector shape differs")
    return {"mse": panel["mse"][panel["supported"]].sum(axis=0),
            "window_count": int(panel["supported"].sum()),
            "dynamic_error": panel["dynamic_error"][selected].sum(axis=0),
            "dynamic_target": panel["dynamic_target"][selected].sum(axis=0)}


def score_aggregate(panel, coefficients):
    c = np.asarray(coefficients, dtype=np.float64)
    values = [float(c@panel[k]@c) for k in ("mse", "dynamic_error", "dynamic_target")]
    scale = max(*(float(np.abs(panel[k]).sum()) for k in ("mse", "dynamic_error", "dynamic_target")), 1.)
    _require(min(values) >= -1e-12*scale, "negative aggregate quadratic error")
    return {"mse_sum": max(values[0], 0.), "window_count": panel["window_count"],
            "dynamic_sse": max(values[1], 0.), "dynamic_target_ss": max(values[2], 0.)}


def pool_statistics(records):
    _require(len(records) > 0, "empty statistics panel")
    return {key: sum(r[key] for r in records) for key in
            ("mse_sum", "window_count", "dynamic_sse", "dynamic_target_ss")}


def statistic_values(record):
    return {"MSE": record["mse_sum"]/record["window_count"] if record["window_count"] > 0 else None,
            "E_dyn": record["dynamic_sse"]/record["dynamic_target_ss"] if record["dynamic_target_ss"] > 0 else None}


def _data_arrays(data):
    from .data import central_support, lengths_for
    _require("lat" in data and np.asarray(data["lat"]).shape[:2] == np.asarray(data["target"]).shape[:2],
             "controls require the post-backbone train view")
    support = central_support(data["mask"], lengths_for(data), data.get("key_time_valid"))
    return np.asarray(data["lat"]), np.asarray(data["target"]), np.asarray(data["mask"], dtype=bool), support


def build_positive_control_fold(data, fit_indices, held_indices, allowed_train_users, *,
                                outer_fold, block_windows=4, components=None):
    """Build one complete inner calibration and its subsequent outer reference gate.

    All alpha paths are closed-form. The 1,936 candidate worlds are scored
    from small per-window quadratic moments. Outer-held targets/predictions
    are not scored until the inner choice is irrevocably made. Components
    may be provided only to replay saved generator artifacts or small tests.
    """
    from .metrics import dynamic_variance, training_threshold
    from time import perf_counter
    started = perf_counter()
    features, real, full_mask, support = _data_arrays(data)
    fit, held = np.asarray(fit_indices, dtype=int), np.asarray(held_indices, dtype=int)
    partition = validate_outer_partition(data["meta"], fit, held, allowed_train_users)
    if components is None:
        components = generate_components(features, data["meta"], support, fit, held, allowed_train_users,
                                         outer_fold=outer_fold, n_joints=real.shape[-1], block_windows=block_windows)
    for key in ("L", "T", "epsilon"):
        _require(np.asarray(components[key]).shape == real.shape, "component geometry differs")
    basis = {"real": real, **{key: components[key] for key in ("L", "T", "epsilon")}}
    alphas = tuple(10.**k for k in range(-5, 8))
    inner_panels, diagnostics = [], []
    for inner_fit, inner_held in inner_partitions(data["meta"], fit):
        factor = WindowedBasisFit(features, basis, support, inner_fit, block_windows=block_windows)
        actual = [np.asarray(components[k])[inner_held, 6:-6] for k in ("L", "T", "epsilon")]
        crop_mask = support[inner_held]
        real_threshold = training_threshold(real[inner_fit], full_mask[inner_fit])
        real_pv = dynamic_variance(real[inner_held], full_mask[inner_held])
        real_selected = np.isfinite(real_pv) & (real_pv >= real_threshold)
        panel = {"by_alpha": {}, "real_by_alpha": {}, "dynamic_selectors": {}, "factor": factor, "aggregates": {},
                 "fit_pose_gram": _pose_variance_gram([components[k][inner_fit] for k in ("L", "T")], full_mask[inner_fit]),
                 "held_pose_gram": _pose_variance_gram([components[k][inner_held] for k in ("L", "T")], full_mask[inner_held])}
        zero = np.zeros_like(actual[0])
        panel["oracle"] = _quadratic_panel([zero, zero, -actual[2]], actual, crop_mask)
        for alpha in (0.0, *alphas):
            predictions, diag = factor.predictions(inner_held, alpha)
            if alpha == 0:
                panel["clean_ls"] = _quadratic_panel(
                    [predictions["L"]-actual[0], predictions["T"]-actual[1], -actual[2]], actual, crop_mask)
                diagnostics.append({"kind": "inner_WLS", "fit_windows": len(inner_fit), "held_windows": len(inner_held),
                                    "factors": diag})
            else:
                panel["by_alpha"][alpha] = _quadratic_panel(
                    [predictions[k]-actual[j] for j, k in enumerate(("L", "T", "epsilon"))], actual, crop_mask)
                real_panel = _quadratic_panel([predictions["real"]-real[inner_held, 6:-6]],
                                              [real[inner_held, 6:-6]], crop_mask)
                panel["real_by_alpha"][alpha] = score_quadratic_panel(real_panel, [1.], real_selected)
            del predictions
        for a in (0.0, *A_GRID):
            coefficients = [1., a]
            threshold = _threshold(_quad(panel["fit_pose_gram"], coefficients))
            pv = _quad(panel["held_pose_gram"], coefficients)
            panel["dynamic_selectors"][a] = np.isfinite(pv) & (pv >= threshold)
            panel["aggregates"][a] = {
                "ridge": {alpha: aggregate_quadratic_panel(p, panel["dynamic_selectors"][a]) for alpha, p in panel["by_alpha"].items()},
                **{kind: aggregate_quadratic_panel(panel[kind], panel["dynamic_selectors"][a]) for kind in ("clean_ls", "oracle")}}
        # The finite-grid loop only needs the small aggregated matrices.
        del panel["by_alpha"], panel["clean_ls"], panel["oracle"]
        inner_panels.append(panel)
        del factor
    real_selection = select_alpha(lambda alpha: statistic_values(pool_statistics(
        [p["real_by_alpha"][alpha] for p in inner_panels]))["MSE"])
    real_stats = pool_statistics([p["real_by_alpha"][real_selection["alpha"]] for p in inner_panels])
    real_reference = {**statistic_values(real_stats), **real_selection}
    # The fields MSE and E_dyn above are metric values; alpha selection's mse
    # carries the same MSE but remains a separate provenance entry.
    def evaluate_pair(a, sigma):
        coeff = [1., a, sigma]
        def statistics(kind, alpha=None):
            return pool_statistics([score_aggregate(
                p["aggregates"][a]["ridge"][alpha] if kind == "ridge" else p["aggregates"][a][kind], coeff)
                for p in inner_panels])
        selection = select_alpha(lambda alpha: statistic_values(statistics("ridge", alpha))["MSE"])
        return {"E_cal": statistic_values(statistics("ridge", selection["alpha"]))["E_dyn"],
                "E_cleanLS": statistic_values(statistics("clean_ls"))["E_dyn"],
                "E_oracle": statistic_values(statistics("oracle"))["E_dyn"],
                "alpha": selection["alpha"], "boundary_unresolved": selection["boundary_unresolved"],
                "alpha_curve": selection["curve"]}
    calibration = calibrate_grid(real_reference, evaluate_pair)
    for name in ("MSE", "E_dyn"):
        if real_reference[name] is not None and not np.isfinite(real_reference[name]):
            real_reference[name] = None
    if calibration["status"] == "calibrated":
        for inner_index, panel in enumerate(inner_panels):
            for world, choice in calibration["selected"].items():
                for kind, alpha, noise_weight in (("observed", choice["alpha"], choice["sigma"]), ("clean", 0., 0.)):
                    _, diag = panel["factor"].mixed_coefficients(
                        {"L": 1., "T": choice["a"], "epsilon": noise_weight}, alpha)
                    diagnostics.append({"kind": "inner_selected_combined", "inner": inner_index,
                                        "world": world, "rhs": kind, "alpha": alpha, "factors": diag})
    result = {"outer_fold": outer_fold, "partition": partition, "components": components,
              "real_reference": real_reference, "calibration": calibration,
              "factor_diagnostics": diagnostics, "timing": {"inner_seconds": perf_counter()-started},
              "outer_scored_after_inner_selection": False}
    del inner_panels
    if calibration["status"] != "calibrated":
        return result
    outer_started = perf_counter()
    result["outer_reference_statistics"] = evaluate_outer_references(
        data, components, fit, held, calibration["selected"], block_windows=block_windows)
    result["outer_scored_after_inner_selection"] = True
    result["timing"]["outer_seconds"] = perf_counter()-outer_started
    return result


def _metric_panel(data, indices, observed, clean):
    from .metrics import dynamic_variance
    indices = np.asarray(indices, dtype=int)
    mask = np.asarray(data["mask"])[indices]
    panel = {"target": np.asarray(observed)[indices], "mask": mask,
             "pose_var": dynamic_variance(np.asarray(clean)[indices], mask),
             "std": np.asarray(data["std"])[indices],
             "meta": [data["meta"][i] for i in indices]}
    if "key_time_valid" in data:
        panel["key_time_valid"] = np.asarray(data["key_time_valid"])[indices]
    return panel


def evaluate_outer_references(data, components, fit_indices, held_indices, selected, *,
                              block_windows=4, include_l13_diagnostics=True):
    """Evaluate already chosen worlds. Never selects parameters on outer held."""
    from .metrics import dynamic_variance, training_threshold, selection_statistics
    features, real, full_mask, support = _data_arrays(data)
    fit, held = np.asarray(fit_indices, dtype=int), np.asarray(held_indices, dtype=int)
    _require(set(selected) == {"linear", "temporal"}, "both inner choices must be fixed before outer scoring")
    basis = {"real": real, **{key: components[key] for key in ("L", "T", "epsilon")}}
    factor = WindowedBasisFit(features, basis, support, fit, block_windows=block_windows)
    results = {}
    for world in ("linear", "temporal"):
        a, sigma = selected[world]["a"], selected[world]["sigma"]
        clean, observed = synthetic_target(components, a, sigma)
        threshold = training_threshold(clean[fit], full_mask[fit])
        panel = _metric_panel(data, held, observed, clean)
        q = support[fit].sum(axis=(1, 2))
        weight = np.divide(support[fit], q[:, None, None], out=np.zeros(support[fit].shape), where=q[:, None, None] > 0)
        weight_sum = weight.sum(axis=(0, 1))
        _require((weight_sum > 0).all(), "unsupported fitting-outer joint")
        constant = np.sum(np.where(support[fit], observed[fit, 6:-6], 0)*weight, axis=(0, 1))/weight_sum
        clean_prediction, clean_diagnostic = factor.predict_mixture(held, {"L": 1., "T": a}, 0.)
        predictions = {"constant": np.broadcast_to(constant, support[held].shape), "oracle": clean[held, 6:-6],
                       "clean_ls": clean_prediction}
        if world == "linear":
            predictions["ridge"], ridge_diagnostic = factor.predict_mixture(
                held, {"L": 1., "epsilon": sigma}, float(selected["linear"]["alpha"]))
        results[world] = {name: selection_statistics(pred, panel, threshold) for name, pred in predictions.items()}
        results[world+"_threshold"] = threshold
        results[world+"_normal_residuals"] = {"clean": clean_diagnostic}
        if world == "linear":
            results[world+"_normal_residuals"]["observed_ridge"] = ridge_diagnostic
    results["linear_diagnostics"] = _outer_linear_diagnostics(
        data, components, fit, held, selected, factor, block_windows=block_windows,
        include_l13=include_l13_diagnostics)
    return results


def _outer_linear_diagnostics(data, components, fit, held, selected, l0_factor, *, block_windows, include_l13):
    """The four prescribed outer designs: L0/L13, two folds, reused RHS."""
    from .metrics import training_threshold, selection_statistics
    features, real, full_mask, support = _data_arrays(data)
    basis = {"real": real, **{key: components[key] for key in ("L", "T", "epsilon")}}
    real_threshold = training_threshold(real[fit], full_mask[fit])
    result = {}
    clean, observed = synthetic_target(components, selected["temporal"]["a"], selected["temporal"]["sigma"])
    threshold = training_threshold(clean[fit], full_mask[fit])
    fit_clean_pred, _ = l0_factor.predict_mixture(fit, {"L": 1., "T": selected["temporal"]["a"]})
    fit_panel = _metric_panel(data, fit, observed, clean)
    baseline = statistic_values(selection_statistics(fit_clean_pred, fit_panel, threshold))["E_dyn"]
    oracle = statistic_values(selection_statistics(clean[fit, 6:-6], fit_panel, threshold))["E_dyn"]
    for name, context in (("L0", 0), ("L13", 6)):
        if name == "L13" and not include_l13:
            continue
        factor = l0_factor if name == "L0" else WindowedBasisFit(features, basis, support, fit, context=context,
                                                                 block_windows=block_windows)
        real_statistics = {}
        real_normals = {}
        world_statistics = {}
        for split, indices in (("fitting", fit), ("held", held)):
            pred, diag = factor.predict_mixture(indices, {"real": 1.})
            real_statistics[split] = selection_statistics(pred, _metric_panel(data, indices, real, real), real_threshold)
            real_normals[split] = diag
        for world, choice in selected.items():
            ystar, y = synthetic_target(components, choice["a"], choice["sigma"])
            world_threshold = training_threshold(ystar[fit], full_mask[fit])
            world_statistics[world] = {}
            for split, indices in (("fitting", fit), ("held", held)):
                pred, diag = factor.predict_mixture(indices, {"L": 1., "T": choice["a"], "epsilon": choice["sigma"]})
                stats = selection_statistics(pred, _metric_panel(data, indices, y, ystar), world_threshold)
                world_statistics[world][split] = {"statistics": stats, "normal_residuals": diag}
        pred_t, _ = factor.predict_mixture(fit, {"T": 1.})
        t_statistics = selection_statistics(pred_t, _metric_panel(data, fit, components["T"], components["T"]), 0.)
        # This margin measures accessibility of the clean teacher, not fitting
        # of the realized noise. Both predictions are scored on the same noisy y.
        clean_pred, clean_diag = factor.predict_mixture(fit, {"L": 1., "T": selected["temporal"]["a"]})
        clean_statistics = selection_statistics(clean_pred, fit_panel, threshold)
        temporal_fit_error = statistic_values(clean_statistics)["E_dyn"]
        result[name] = {"real": real_statistics, "real_normal_residuals": real_normals,
                        "worlds": world_statistics, "T_fitting_statistics": t_statistics,
                        "T_unread_fraction_fitting": statistic_values(t_statistics)["E_dyn"],
                        "temporal_margin_recovery_fitting": _recovery(baseline, temporal_fit_error, oracle),
                        "temporal_margin_clean_fit_statistics": clean_statistics,
                        "temporal_margin_clean_normal_residuals": clean_diag,
                        "temporal_margin_fit_target": "clean_y_star", "temporal_margin_score_target": "observed_y",
                        "descriptive_only": True}
        if name == "L13":
            del factor
    return result


def combine_reference_folds(fold_results):
    """Construct mandatory fold0/fold1/pooled gates from additive moments."""
    _require(len(fold_results) == 2 and {r["outer_fold"] for r in fold_results} == {0, 1}, "two outer folds required")
    if any(r["calibration"]["status"] != "calibrated" for r in fold_results):
        states = [r["calibration"]["status"] for r in fold_results]
        return {"status": "control_non_calibrated" if "control_non_calibrated" in states else "control_non_diagnostic"}
    ordered = sorted(fold_results, key=lambda r: r["outer_fold"])
    _require(all(r["outer_scored_after_inner_selection"] for r in ordered), "outer gate unavailable")
    linear, temporal = {}, {}
    panels = {"fold0": ordered[0]["outer_reference_statistics"], "fold1": ordered[1]["outer_reference_statistics"]}
    panels["pooled"] = {world: {name: pool_statistics([r["outer_reference_statistics"][world][name] for r in ordered])
                                 for name in ("constant", "oracle", "clean_ls", "ridge") if name in panels["fold0"][world]}
                         for world in ("linear", "temporal")}
    for split, panel in panels.items():
        values = {name: statistic_values(stats) for name, stats in panel["linear"].items()}
        linear[split] = {metric: {name: values[name][metric] for name in ("constant", "ridge", "oracle")}
                         for metric in ("MSE", "E_dyn")}
        temporal[split] = {name: statistic_values(panel["temporal"][name])["E_dyn"] for name in ("clean_ls", "oracle")}
    lg, tg = linear_reference_gate(linear), temporal_reference_gate(temporal)
    return {"status": "diagnostic" if lg["status"] == tg["status"] == "diagnostic" else "control_non_diagnostic",
            "linear_records": linear, "temporal_records": temporal, "linear_gate": lg, "temporal_gate": tg}
