"""Independent oracle and regression tests for the corrected RnC core."""

import itertools
import math

import pytest
import torch

from wearusfm.model.rank_contrast import (anchor_support, pose_distance, pose_rank_contrast,
                                        rank_contrast_count, rank_contrast_pairs)


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(4, 7, 8, dtype=torch.float64, generator=g, requires_grad=True)
    y = torch.randint(0, 4, (4, 7, 3), generator=g).double() / 4
    mask = torch.rand(4, 7, 3, generator=g) > .25
    tv = torch.rand(4, 7, generator=g) > .1
    y[~mask] = float("nan")
    return z, y, mask, tv


def _oracle(z, y, mask, tv, temperature, stride, per_window, resolution=0., allowed=None):
    points = [(s, p) for s in range(z.shape[0]) for p in range(stride // 2, z.shape[1], stride)
              if bool(tv[s, p]) and bool(mask[s, p].any())]
    values = {}
    for i, (si, pi) in enumerate(points):
        hi = z[si, pi] - z[si, pi].mean()
        hi = hi / hi.norm()
        ds, scores = {}, {}
        for k, (sk, pk) in enumerate(points):
            common = mask[si, pi] & mask[sk, pk]
            if i == k or not common.any() or (allowed is not None and not allowed[i, k]):
                continue
            d = math.sqrt(sum(float(y[si, pi, j] - y[sk, pk, j]) ** 2
                              for j in range(y.shape[-1]) if common[j]) / int(common.sum()))
            ds[k] = math.floor(d / resolution) * resolution if resolution else d
            hk = z[sk, pk] - z[sk, pk].mean()
            hk = hk / hk.norm()
            scores[k] = (hi * hk).sum() / temperature
        if len(ds) < 2:
            continue
        num, chance = [], 0.
        for j in ds:
            keys = [k for k in ds if ds[k] >= ds[j]]
            num.append(torch.logsumexp(torch.stack([scores[k] for k in keys]), 0) - scores[j])
            chance += math.log(len(keys))
        values.setdefault(si, []).append(torch.stack(num).sum() / chance)
    if not values:
        return z.sum() * 0, 0
    if per_window:
        return torch.stack([torch.stack(v).mean() for v in values.values()]).mean(), len(values)
    flat = [v for group in values.values() for v in group]
    return torch.stack(flat).mean(), len(flat)


@pytest.mark.parametrize("seed", range(8))
def test_independent_forward_backward_mask_tie_filter_oracle(seed):
    z, y, mask, tv = _data(seed)
    for per_window, resolution, filtered in itertools.product((False, True), (0., .5), (False, True)):
        kwargs = dict(temperature=.07 + seed * .02, stride=1 + seed % 3,
                      per_window=per_window, resolution=resolution)
        s, _, _, _, _, _ = anchor_support(mask & tv[..., None], kwargs["stride"])
        # The physical filtering policy belongs to the caller. This exercises a
        # symmetric mask with some rows having fewer than two eligible candidates.
        allowed = (s[:, None] != s[None, :]) if filtered else None
        loss, num, count, _ = pose_rank_contrast(z, (y, mask), tv, pair_allowed=allowed, **kwargs)
        ref, expected_count = _oracle(z, y, mask, tv, allowed=allowed, **kwargs)
        assert count.dtype == torch.int64 and int(count) == expected_count
        assert torch.allclose(loss, ref, atol=1e-11, rtol=1e-10)
        assert torch.allclose(num, loss * count, atol=1e-11, rtol=1e-10)
        assert int(rank_contrast_count(mask & tv[..., None], stride=kwargs["stride"],
                                      per_window=per_window, pair_allowed=allowed)) == expected_count
        ga, = torch.autograd.grad(loss, z)
        ge, = torch.autograd.grad(ref, z)
        assert torch.allclose(ga, ge, atol=1e-10, rtol=1e-9)


def test_identical_poses_never_have_window_specific_repulsion():
    logits = torch.zeros(4, 4, dtype=torch.float64, requires_grad=True)
    cand = ~torch.eye(4, dtype=torch.bool)
    num, chance, _ = rank_contrast_pairs(logits, torch.zeros_like(logits), cand)
    grad, = torch.autograd.grad(num[0] / chance[0], logits)
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-14)
    groups = torch.tensor([0, 0, 1, 1])
    cross = cand & (groups[:, None] != groups[None, :])
    with pytest.raises(ValueError, match="identical"):
        rank_contrast_pairs(logits, torch.zeros_like(logits), cand, cross)
    z, y, mask, tv = _data()
    with pytest.raises(ValueError, match="cross is rejected"):
        pose_rank_contrast(z, (y, mask), tv, temperature=.1, stride=2, positives="cross")


@pytest.mark.parametrize("temperature", [0.1, 5e-5, 1e-5])
def test_masked_candidates_do_not_enter_small_temperature_denominators(temperature):
    z = torch.tensor([[[1., -1., 0.]], [[0., 1., -1.]], [[-1., 0., 1.]]], dtype=torch.float64,
                     requires_grad=True)
    y = torch.arange(3, dtype=torch.float64).reshape(3, 1, 1)
    mask, tv = torch.ones_like(y, dtype=torch.bool), torch.ones(3, 1, dtype=torch.bool)
    kwargs = dict(temperature=temperature, stride=1, per_window=True)
    loss, _, _, _ = pose_rank_contrast(z, (y, mask), tv, **kwargs)
    ref, _ = _oracle(z, y, mask, tv, **kwargs)
    assert torch.allclose(loss, ref, atol=1e-8, rtol=1e-8)
    assert torch.isfinite(torch.autograd.grad(loss, z)[0]).all()


def test_finite_difference_gradcheck():
    z, y, mask, tv = _data(4)
    z, y, mask, tv = z[:2, :3].detach().requires_grad_(True), y[:2, :3], mask[:2, :3], tv[:2, :3]
    assert torch.autograd.gradcheck(lambda x: pose_rank_contrast(x, (y, mask), tv, temperature=.2, stride=1)[0],
                                    (z,), atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_insufficient_support_keeps_zero_gradient(n):
    z = torch.randn(n, 1, 8, requires_grad=True)
    y = torch.zeros(n, 1, 2)
    mask, tv = torch.ones_like(y, dtype=torch.bool), torch.ones(n, 1, dtype=torch.bool)
    loss, num, count, log = pose_rank_contrast(z, (y, mask), tv, temperature=.1, stride=1)
    assert loss == 0 and num == 0 and count == 0 and log["anchors"] == 0
    assert torch.equal(torch.autograd.grad(num, z)[0], torch.zeros_like(z))


def test_masked_targets_padding_and_no_pose():
    z, y, mask, tv = _data()
    mask.zero_()
    y.fill_(float("nan"))
    for pose in (None, (y, mask)):
        loss, num, count, _ = pose_rank_contrast(z, pose, tv, temperature=.1, stride=2)
        assert loss == 0 and count == 0
        assert torch.equal(torch.autograd.grad(num, z)[0], torch.zeros_like(z))
    mask[:, :3] = True
    y[:, :3] = 0
    tv[:, :3] = True
    tv[:, 3:] = False
    loss, *_ = pose_rank_contrast(z, (y, mask), tv, temperature=.1, stride=1)
    grad, = torch.autograd.grad(loss, z)
    assert torch.isfinite(grad).all() and (grad[:, 3:] == 0).all()


def test_cpu_autocast_global_rng_and_count_types():
    z, y, mask, tv = _data(1)
    z = z.float().detach().requires_grad_(True)
    kwargs = dict(temperature=.1, stride=2)
    state = torch.random.get_rng_state()
    plain = pose_rank_contrast(z, (y, mask), tv, **kwargs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp = pose_rank_contrast(z, (y, mask), tv, **kwargs)
    assert torch.equal(state, torch.random.get_rng_state())
    assert amp[0].dtype == torch.float32 and amp[2].dtype == torch.int64
    assert torch.equal(plain[0], amp[0])
    assert torch.equal(torch.autograd.grad(plain[0], z)[0], torch.autograd.grad(amp[0], z)[0])


def test_pair_allowed_must_be_symmetric_and_pose_distance_empty_is_defined():
    mask = torch.ones(2, 2, 3, dtype=torch.bool)
    bad = torch.ones(4, 4, dtype=torch.bool)
    bad[0, 1] = False
    with pytest.raises(ValueError, match="symmetric"):
        anchor_support(mask, 1, pair_allowed=bad)
    assert pose_distance(torch.empty(0, 3), torch.empty(0, 3, dtype=torch.bool)).shape == (0, 0)


@pytest.mark.parametrize("temperature", [0., -1., float("inf"), float("nan")])
def test_invalid_temperature_is_rejected(temperature):
    z, y, mask, tv = _data()
    with pytest.raises(ValueError, match="temperature"):
        pose_rank_contrast(z, (y, mask), tv, temperature=temperature, stride=2)
