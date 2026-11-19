from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet.linear import (WeightedQR, LinearReferenceError, factorize,
                                  layer_norm64, design_block, iter_window_blocks,
                                  mask_groups, predict_windowed, select_alpha)
from pose_temponet.linear import WindowedBasisFit


def _fixture():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(97, 7))
    # Exact rank deficiency, exercising SVD cutoff without Gram eigenvalues.
    x[:, -1] = x[:, 0] + x[:, 1]
    y = rng.normal(size=(97, 3))
    w = rng.uniform(.1, 1, size=97)
    w[::7] = 0
    return x, y, w


def test_streamed_wls_matches_independent_lstsq_and_rhs_mixtures():
    x, y, w = _fixture()
    qr = factorize((x[k:k+11], y[k:k+11], w[k:k+11]) for k in range(0, len(x), 11))
    a = np.column_stack((np.ones(len(x)), x)) * np.sqrt(w[:, None])
    b = y * np.sqrt(w[:, None])
    expected = np.linalg.lstsq(a, b, rcond=1e-8)[0]
    solved = qr.solve()
    np.testing.assert_allclose(solved.coefficients, expected, atol=1e-12)
    np.testing.assert_allclose(solved.training_sse, np.sum((a@expected-b)**2, axis=0), atol=1e-11)
    mix = np.array([[1., 0], [.2, -1], [-.5, 3]])
    combined = qr.solve(rhs_mix=mix)
    np.testing.assert_allclose(combined.coefficients, expected@mix, atol=1e-12)
    np.testing.assert_allclose(combined.training_sse, np.sum((a@expected@mix-b@mix)**2, axis=0), atol=1e-10)
    assert solved.rank == 7


@pytest.mark.parametrize("alpha", [1e-5, .5, 100., 1e7])
def test_ridge_intercept_unpenalized_stationarity_and_direct_solution(alpha):
    x, y, w = _fixture()
    qr = factorize([(x[:40], y[:40], w[:40]), (x[40:], y[40:], w[40:])])
    solution = qr.solve(alpha)
    a = np.column_stack((np.ones(len(x)), x)) * np.sqrt(w[:, None])
    b = y * np.sqrt(w[:, None])
    penalty = np.eye(a.shape[1])*alpha
    penalty[0, 0] = 0
    expected = np.linalg.solve(a.T@a+penalty, a.T@b)
    np.testing.assert_allclose(a@solution.coefficients, a@expected, rtol=1e-8, atol=1e-9)
    residual = a.T@(a@solution.coefficients-b)+penalty@solution.coefficients
    assert np.linalg.norm(residual) < 1e-9
    assert solution.normal_residual_ratio.max() < 1e-12


def test_exact_rhs_does_not_use_residual_as_denominator():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 4))
    y = np.column_stack((np.ones(len(x)), x)) @ rng.normal(size=(5, 2))
    factor = factorize([(x, y, np.ones(len(x)))])
    solution = factor.solve()
    assert np.isfinite(solution.normal_residual_ratio).all()
    assert solution.normal_residual_ratio.max() < 1e-12
    zero = factorize([(x, np.zeros((len(x), 1)), np.ones(len(x)))]).solve()
    assert zero.normal_residual_ratio[0] == 0


def test_window_weights_use_total_valid_elements_and_joint_masks():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3, 16, 4)).astype(np.float16)
    y = rng.normal(size=(3, 16, 3))
    support = np.ones((3, 4, 3), dtype=bool)
    support[1, :, 0] = False
    support[2, 0, :] = False
    assert mask_groups(support) == [[0], [1, 2]]
    blocks = list(iter_window_blocks(x, y, support, [0, 1, 2], [1, 2], block_windows=1))
    np.testing.assert_allclose(blocks[0][2], np.full(4, 1/12))
    np.testing.assert_allclose(blocks[1][2], np.full(4, 1/8))
    np.testing.assert_allclose(blocks[2][2], [0, 1/9, 1/9, 1/9])
    with pytest.raises(LinearReferenceError, match="masks differ"):
        list(iter_window_blocks(x, y, support, [0, 1, 2], [0, 1]))


def test_temporal_design_is_time_major_and_keeps_original_center_alignment():
    x = np.arange(2*18*3).reshape(2, 18, 3).astype(float)
    identity = lambda z: np.asarray(z, dtype=float)
    design = design_block(x, context=6, crop=6, normalizer=identity)
    assert design.shape == (2, 6, 39)
    np.testing.assert_array_equal(design[0, 0], x[0, :13].reshape(-1))
    np.testing.assert_array_equal(design[1, -1], x[1, 5:18].reshape(-1))
    coef = np.zeros((40, 1)); coef[1+6*3, 0] = 1
    pred = predict_windowed(x, coef, context=6, normalizer=identity, block_windows=1)
    np.testing.assert_array_equal(pred[..., 0], x[:, 6:-6, 0])


def test_ln64_has_exact_channel_constraint_within_roundoff():
    x = np.random.default_rng(0).normal(size=(4, 160, 384)).astype(np.float16)
    z = layer_norm64(x)
    assert z.dtype == np.float64
    np.testing.assert_allclose(z.sum(-1), 0, atol=4e-14)


def test_alpha_selection_bounded_and_exact_tie_lowest():
    optimum = select_alpha(lambda alpha: (np.log10(alpha)+3)**2+1)
    assert optimum["alpha"] == 1e-3 and not optimum["boundary_unresolved"]
    assert min(row["alpha"] for row in optimum["curve"]) == 1e-4
    tied = select_alpha(lambda alpha: 1.)
    assert tied["alpha"] == 1e-5 and tied["boundary_unresolved"]
    assert len(tied["curve"]) == 10
    high = select_alpha(lambda alpha: 1/alpha)
    assert high["alpha"] == 1e7 and high["boundary_unresolved"]


def test_bad_weights_nonfinite_supported_values_and_memory_fail_explicitly():
    x, y, w = _fixture()
    with pytest.raises(LinearReferenceError, match="factor storage"):
        WeightedQR(100, 2, max_factor_bytes=1000)
    with pytest.raises(LinearReferenceError, match="invalid weights"):
        factorize([(x, y, -w)])
    y[1, 0] = np.nan
    with pytest.raises(LinearReferenceError, match="nonfinite"):
        factorize([(x, y, w)])
    y[1, 0] = 0; y[0, 0] = np.nan  # zero weight row is irrelevant
    factorize([(x, y, w)]).solve()


def test_basis_factor_reuses_qr_and_matches_separate_joint_mask_solves():
    rng = np.random.default_rng(102)
    x = rng.normal(size=(6, 20, 4)).astype(np.float16)
    y, noise = rng.normal(size=(6, 20, 2)), rng.normal(size=(6, 20, 2))
    support = np.ones((6, 8, 2), dtype=bool); support[0, :2, 1] = False
    factor = WindowedBasisFit(x, {"y": y, "noise": noise}, support, [0, 1, 2, 3], block_windows=1)
    coefficients, diagnostics = factor.coefficients(.2)
    assert len(diagnostics) == 2
    predictions, _ = factor.predictions([4, 5], .2)
    for j in range(2):
        reference = factorize(iter_window_blocks(x, y, support, [0, 1, 2, 3], [j], block_windows=2)).solve(.2)
        np.testing.assert_allclose(coefficients["y"][:, j], reference.coefficients[:, 0], atol=1e-12)
        expected = predict_windowed(x[[4, 5]], reference.coefficients)
        np.testing.assert_allclose(predictions["y"][:, :, j], expected[:, :, 0], atol=1e-12)
