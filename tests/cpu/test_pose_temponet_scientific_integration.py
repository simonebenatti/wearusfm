"""Independent numerical oracles across reference, loss, masks and final report."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet import controls as C, data as D, linear as L, metrics as M, report as R


def manual_design(x, context):
    x = np.asarray(x, dtype=np.float64)
    rows = []
    for window in x:
        normalized = []
        for time in window:
            centered = time - sum(time) / len(time)
            normalized.append(centered / np.sqrt(sum(centered**2) / len(time) + 1e-5))
        normalized = np.asarray(normalized)
        rows.append([np.r_[1., normalized[t-context:t+context+1].flatten()]
                     for t in range(6, len(window)-6)])
    return np.asarray(rows)


def manual_statistics(pred, target, mask, clean, threshold, users):
    """No historical helper; explicit valid times and per-window weighting."""
    per_user = {u: {"error": 0., "target": 0., "mae": 0., "elements": 0} for u in set(users)}
    mse_sum, count = 0., 0
    variances = []
    for i in range(len(target)):
        jointvars = [float(np.var(clean[i, mask[i, :, j], j])) for j in range(target.shape[-1])
                     if mask[i, :, j].sum() >= 2]
        variances.append(float(np.mean(jointvars)) if jointvars else float("nan"))
        q = int(mask[i, 6:-6].sum())
        if q:
            mse_sum += sum((pred[i, t-6, j]-target[i, t, j])**2 for t in range(6, target.shape[1]-6)
                           for j in range(target.shape[-1]) if mask[i, t, j]) / q
            count += 1
        if not np.isfinite(variances[-1]) or variances[-1] < threshold:
            continue
        row = per_user[users[i]]
        for j in range(target.shape[-1]):
            times = [t for t in range(6, target.shape[1]-6) if mask[i, t, j]]
            if len(times) < 2:
                continue
            p = np.asarray([pred[i, t-6, j] for t in times])
            y = np.asarray([target[i, t, j] for t in times])
            row["error"] += float(sum(((p-p.mean())-(y-y.mean()))**2))
            row["target"] += float(sum((y-y.mean())**2))
            row["mae"] += float(sum(abs(p-y)))
            row["elements"] += len(times)
    return {"mse_sum": mse_sum, "window_count": count,
            "dynamic_sse": sum(row["error"] for row in per_user.values()),
            "dynamic_target_ss": sum(row["target"] for row in per_user.values())}, per_user, np.asarray(variances)


@pytest.mark.parametrize("context,alpha", [(0, 0.), (0, .7), (6, 0.), (6, .7)])
def test_multijoint_closed_form_matches_independent_augmented_design_and_loss(context, alpha):
    rng = np.random.default_rng(902)
    features = rng.normal(size=(8, 24, 4)).astype(np.float16)
    target = rng.normal(size=(8, 24, 3))
    nuisance = rng.normal(size=target.shape)
    mask = rng.random(size=target.shape) > .25
    mask[0, :, 1] = False
    mask[1, 7:, 2] = False
    support = D.central_support(mask, np.full(8, 24, dtype=np.int64))
    q = support.sum((1, 2))
    fit, held = np.arange(6), np.arange(6, 8)
    factor = L.WindowedBasisFit(features, {"y": target, "nuisance": nuisance}, support, fit,
                               context=context, block_windows=2)
    mixed = target + .3*nuisance
    prediction, diagnostics = factor.predict_mixture(held, {"y": 1., "nuisance": .3}, alpha)
    fit_prediction, _ = factor.predict_mixture(fit, {"y": 1., "nuisance": .3}, alpha)
    design = manual_design(features, context)
    independent = np.empty_like(prediction)
    objective = 0.
    for joint in range(3):
        x, y, weights = [], [], []
        for i in fit:
            for centered_t in range(support.shape[1]):
                if support[i, centered_t, joint]:
                    x.append(design[i, centered_t]); y.append(mixed[i, centered_t+6, joint]); weights.append(1./q[i])
        a = np.asarray(x) * np.sqrt(weights)[:, None]
        b = np.asarray(y) * np.sqrt(weights)
        if alpha:
            penalty = np.eye(a.shape[1])*np.sqrt(alpha)
            penalty[0, 0] = 0
            augmented = np.vstack((a, penalty))
            rhs = np.r_[b, np.zeros(a.shape[1])]
            coef = np.linalg.lstsq(augmented, rhs, rcond=1e-14)[0]
        else:
            coef = np.linalg.lstsq(a, b, rcond=1e-8)[0]
        independent[..., joint] = design[held] @ coef
        objective += float(np.sum((a@coef-b)**2))
        normal = a.T @ (a@coef-b)
        if alpha:
            normal[1:] += alpha*coef[1:]
        assert np.linalg.norm(normal) < 1e-9
    np.testing.assert_allclose(prediction, independent, atol=2e-11, rtol=2e-11)
    torch_loss = M.per_window_mse(torch.from_numpy(fit_prediction), torch.from_numpy(mixed[fit, 6:-6]),
                                 torch.from_numpy(support[fit])).item()
    assert torch_loss*len(fit) == pytest.approx(objective, rel=1e-11, abs=1e-11)
    assert all(d["normal_residual_max"] <= 1e-6 for d in diagnostics)


def test_raw_oracle_matches_metrics_and_quadratic_control_with_singleton_and_empty_joints():
    rng = np.random.default_rng(62)
    shape = (7, 24, 3)
    clean_basis = [rng.normal(size=shape), rng.normal(size=shape)]
    noise = rng.normal(size=shape)*5
    coeff = [1., .4, .7]
    target_basis = [*clean_basis, noise]
    clean = clean_basis[0]+.4*clean_basis[1]
    target = clean+.7*noise
    pred_basis = [rng.normal(size=(7, 12, 3)) for _ in range(3)]
    pred = sum(c*v for c, v in zip(coeff, pred_basis))
    mask = rng.random(shape) > .3
    mask[0] = False
    mask[1, :, 0] = False; mask[1, 9, 0] = True  # One element enters MSE, never centered sums.
    mask[2, 8:, 1] = False
    target[~mask] = np.nan
    users = ["u0", "u0", "u1", "u1", "u2", "u2", "u3"]
    threshold = .8
    expected, _, pv = manual_statistics(pred, target, mask, clean, threshold, users)
    data = {"target": target, "mask": mask, "pose_var": pv, "std": np.ones((7, 3)),
            "meta": [{"user": u, "n_patches": 24} for u in users]}
    actual = M.selection_statistics(pred, data, threshold)
    support = D.central_support(mask, np.full(7, 24, dtype=np.int64))
    panel = C._quadratic_panel([p-y[:, 6:-6] for p, y in zip(pred_basis, target_basis)],
                               [y[:, 6:-6] for y in target_basis], support)
    quadratic = C.score_quadratic_panel(panel, coeff, np.isfinite(pv) & (pv >= threshold))
    assert actual["window_count"] == expected["window_count"] == 6
    for key in ("mse_sum", "dynamic_sse", "dynamic_target_ss"):
        assert actual[key] == pytest.approx(expected[key], rel=2e-14)
        assert quadratic[key] == pytest.approx(expected[key], rel=2e-14)
    # If y rather than y* selected dynamics this fixture produces a different estimand.
    wrong = {**data, "pose_var": M.dynamic_variance(target, mask)}
    assert M.selection_statistics(pred, wrong, threshold)["dynamic_target_ss"] != actual["dynamic_target_ss"]


def test_final_paired_bootstrap_matches_raw_prediction_oracle_not_metric_helpers():
    rng = np.random.default_rng(912)
    users = [f"u{i:02}" for i in range(15)]
    target = rng.normal(size=(15, 20, 20)) * np.linspace(.9, 1.3, 15)[:, None, None]
    mask = rng.random(target.shape) > .15
    data = {"target": target, "mask": mask, "pose_var": np.ones(15), "std": np.ones((15, 20)),
            "meta": [{"user": u, "n_patches": 20} for u in users]}
    factors = {"L0": .2, "MLP": .3, "L13": .4, "TCN_post": .55, "TCN_pre": .1}
    cells, raw_error = {}, {}
    for head, factor in factors.items():
        cells[head], raw_error[head] = {}, []
        for seed in R.SEEDS:
            pred = target[:, 6:-6]*(factor+seed*.05)+rng.normal(size=(15, 8, 20))*.03
            cells[head][seed] = {"complete": True, "analysis": M.evaluate(pred, data, per_joint=True)}
            _, by_user, _ = manual_statistics(pred, target, mask, target, M.THRESHOLD, users)
            raw_error[head].append([by_user[u]["error"] for u in users])
    refs = {}
    for name, factor in zip(R.REFERENCES, (.15, .17)):
        pred = target[:, 6:-6]*factor
        refs[name] = {"complete": True, "analysis": M.evaluate(pred, data)}
        _, by_user, _ = manual_statistics(pred, target, mask, target, M.THRESHOLD, users)
        raw_error[name] = [[by_user[u]["error"] for u in users]]
    assessment = {h: {"optimizer_alarms": [], "objective_tradeoff": {"flag": False}} for h in R.HEADS}
    result = R.aggregate(cells, refs, assessments=assessment, common_gates={g: True for g in R.COMMON_GATES})
    assert result["complete"], result
    draws = np.random.default_rng(20261102).integers(0, 15, size=(2000, 15))
    candidate = np.mean(raw_error["TCN_post"], axis=0)
    for i, head in enumerate(R.COMPARATORS, 1):
        comparator = np.mean(raw_error[head], axis=0)
        gains = np.array([100.*(1.-sum(candidate[j] for j in draw)/sum(comparator[j] for j in draw))
                          for draw in draws])
        effect = result["contrasts"][str(i)]["relative_E_dyn"]
        assert effect["point"] == pytest.approx(100*(1-candidate.sum()/comparator.sum()), abs=1e-12)
        np.testing.assert_allclose(effect["ci"], np.percentile(gains, [.5, 99.5]), rtol=0, atol=1e-12)
        assert effect["undefined_replicates"] == effect["replicates_discarded"] == 0
