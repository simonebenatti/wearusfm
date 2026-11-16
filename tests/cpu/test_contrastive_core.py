"""Tests of readout support, RNG isolation and asymmetric InfoNCE gradients."""

import math

import pytest
import torch

from wearusfm.model.contrastive import ProjectionHead, info_nce, normalize_projection, window_readout


def test_projector_preserves_rng_and_is_reproducible_without_changing_base_parameters():
    torch.manual_seed(17)
    first = torch.nn.Linear(8, 8)
    state = torch.random.get_rng_state()
    head = ProjectionHead(8, out=4)
    assert torch.equal(state, torch.random.get_rng_state())
    second = ProjectionHead(8, out=4)
    assert all(torch.equal(a, b) for a, b in zip(head.parameters(), second.parameters()))
    next_layer = torch.nn.Linear(8, 8)
    torch.manual_seed(17)
    base_first, base_next = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), base_first.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(next_layer.parameters(), base_next.parameters()))
    assert head(torch.ones(2, 8)).shape == (2, 4)


def test_readout_includes_hidden_patches_and_excludes_padding_with_exact_gradients():
    z = torch.arange(2 * 4 * 3 * 2, dtype=torch.float64).reshape(2, 4, 3, 2).requires_grad_(True)
    physical = torch.tensor([[True, True, True, False], [True, True, False, False]])
    visible = torch.tensor([[True, False, False, False], [False, False, False, False]])
    pooled, eligible = window_readout(z, physical, visible)
    assert eligible.tolist() == [True, False]
    assert torch.equal(pooled[0], z[0, :3].mean((0, 1)))
    assert torch.equal(pooled[1], z[1, :2].mean((0, 1)))
    grad, = torch.autograd.grad(pooled.sum(), z)
    assert torch.allclose(grad[0, :3], torch.full_like(grad[0, :3], 1 / 9))
    assert torch.allclose(grad[1, :2], torch.full_like(grad[1, :2], 1 / 6))
    assert (grad[0, 3:] == 0).all() and (grad[1, 2:] == 0).all()


def test_readout_ignores_nonfinite_padding_and_empty_support():
    z = torch.ones(2, 3, 2, 4, requires_grad=True)
    with torch.no_grad():
        z[0, 2] = float("nan")
        z[1] = float("nan")
    physical = torch.tensor([[True, True, False], [False, False, False]])
    pooled, valid = window_readout(z, physical)
    assert torch.isfinite(pooled).all() and valid.tolist() == [True, False]
    assert torch.equal(pooled[1], torch.zeros(4))
    grad, = torch.autograd.grad(pooled.sum(), z)
    assert torch.isfinite(grad).all() and (grad[1] == 0).all()


def test_projection_norm_floor_and_invalid_rows():
    bad = torch.tensor([[3., 4.], [float("nan"), 0.]], requires_grad=True)
    valid = torch.tensor([True, False])
    result = normalize_projection(bad, valid=valid)
    assert torch.allclose(result, torch.tensor([[.6, .8], [0., 0.]]))
    grad, = torch.autograd.grad(result.sum(), bad)
    assert torch.isfinite(grad).all() and (grad[1] == 0).all()
    with pytest.raises(ValueError, match="norm floor"):
        normalize_projection(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="nonfinite"):
        normalize_projection(bad)


def _nce_data():
    g = torch.Generator().manual_seed(17)
    q = normalize_projection(torch.randn(4, 8, dtype=torch.float64, generator=g)).requires_grad_(True)
    kp = normalize_projection(torch.randn(4, 8, dtype=torch.float64, generator=g)).requires_grad_(True)
    kn = normalize_projection(torch.randn(7, 8, dtype=torch.float64, generator=g)).requires_grad_(True)
    mask = torch.tensor([[True] * 7, [True, True] + [False] * 5, [False] * 7, [True] * 7])
    return q, kp, kn, mask


def test_info_nce_matches_direct_denominators_and_only_query_gets_gradients():
    q, kp, kn, mask = _nce_data()
    valid = torch.tensor([True, True, True, False])
    losses, eligible, metrics = info_nce(q, kp, kn, mask, temperature=.2, min_negatives=2, query_valid=valid)
    refs = []
    for i in range(4):
        if valid[i] and mask[i].sum() >= 2:
            p = (q[i] * kp[i].detach()).sum() / .2
            vals = [(q[i] * kn[k].detach()).sum() / .2 for k in range(7) if mask[i, k]]
            refs.append(torch.logsumexp(torch.stack([p, *vals]), dim=0) - p)
        else:
            refs.append(q[i].sum() * 0.)
    ref = torch.stack(refs)
    assert eligible.tolist() == [True, True, False, False]
    assert torch.allclose(losses, ref, atol=1e-12, rtol=1e-12)
    assert metrics["eligible"] == 2 and metrics["queries"] == 4
    gq, gkp, gkn = torch.autograd.grad(losses.sum(), (q, kp, kn), allow_unused=True)
    gr, = torch.autograd.grad(ref.sum(), q)
    assert torch.allclose(gq, gr, atol=1e-12, rtol=1e-12)
    assert gkp is None and gkn is None and (gq[2:] == 0).all()


def test_uniform_projections_loss_is_log_number_of_admissible_keys():
    q = torch.ones(3, 4) / 2
    kn = torch.ones(8, 4) / 2
    mask = torch.ones(3, 8, dtype=torch.bool)
    mask[1, 3:] = False
    mask[2] = False
    loss, valid, _ = info_nce(q, q, kn, mask, min_negatives=1)
    assert torch.allclose(loss, torch.tensor([math.log(9), math.log(4), 0.]))
    assert valid.tolist() == [True, True, False]


def test_empty_memory_is_finite_zero_and_no_query_gradient():
    q = torch.randn(3, 5, requires_grad=True)
    loss, valid, _ = info_nce(q, q.detach(), torch.empty(0, 5), torch.empty(3, 0, dtype=torch.bool))
    assert torch.equal(loss, torch.zeros(3)) and not valid.any()
    assert torch.equal(torch.autograd.grad(loss.sum(), q)[0], torch.zeros_like(q))


def test_cpu_autocast_and_rng_are_unchanged():
    q, kp, kn, mask = _nce_data()
    q, kp, kn = q.float().detach().requires_grad_(True), kp.float(), kn.float()
    state = torch.random.get_rng_state()
    plain = info_nce(q, kp, kn, mask, min_negatives=2)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp = info_nce(q, kp, kn, mask, min_negatives=2)[0]
    assert torch.equal(state, torch.random.get_rng_state())
    assert amp.dtype == torch.float32 and torch.equal(amp, plain)
    assert torch.equal(torch.autograd.grad(amp.sum(), q)[0], torch.autograd.grad(plain.sum(), q)[0])


def test_info_nce_gradcheck():
    q, kp, kn, mask = _nce_data()
    assert torch.autograd.gradcheck(lambda x: info_nce(x, kp, kn, mask, min_negatives=2)[0], (q,))


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_bad_nce_temperature_is_rejected(temperature):
    q, kp, kn, mask = _nce_data()
    with pytest.raises(ValueError, match="temperature"):
        info_nce(q, kp, kn, mask, temperature=temperature)


def test_nce_protocol_threshold_31_vs_32_and_temperature():
    # Independent regression from Claude 0ab9d98, prop_nce_killers.py.
    generator = torch.Generator().manual_seed(3)
    q, kp = [normalize_projection(torch.randn(3, 16, generator=generator)) for _ in range(2)]
    kn = normalize_projection(torch.randn(40, 16, generator=generator))
    mask = torch.arange(40)[None, :] < torch.tensor([31, 32, 40])[:, None]
    value, eligible, _ = info_nce(q, kp, kn, mask)
    assert eligible.tolist() == [False, True, True] and value[0] == 0
    explicit = info_nce(q, kp, kn, mask, temperature=.2, min_negatives=32)[0]
    changed = info_nce(q, kp, kn, mask, temperature=.1, min_negatives=32)[0]
    assert torch.equal(value, explicit) and not torch.allclose(value[1:], changed[1:])
