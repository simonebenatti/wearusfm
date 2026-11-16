"""Test di `model/rank_contrast.py` (CPU; float64 dove serve il confronto esatto)."""

import math

import pytest
import torch
import torch.nn.functional as F

from wearusfm.model.rank_contrast import pose_rank_contrast, rank_contrast_count

KW = dict(temperature=0.2, stride=2)


def _data(s=4, p=12, j=5, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(s, p, d, generator=g, dtype=torch.float64)
    target = torch.randn(s, p, j, generator=g, dtype=torch.float64)
    mask = torch.rand(s, p, j, generator=g) > 0.2
    mask[0, :4] = False  # un tratto senza posa
    mask[1, 1] = True
    target[1, 3], mask[1, 3] = target[1, 1], mask[1, 1]  # pose identiche nella stessa finestra...
    target[2, 5], mask[2, 5] = target[1, 1], mask[1, 1]  # ...e in un'altra: parita' esatte
    tv = torch.ones(s, p, dtype=torch.bool)
    tv[3, 8:] = False  # finestra piu' corta: riempimento
    return z, (target, mask), tv


def reference(z, target, mask, tv, temperature, stride, positives, per_window, resolution=0.0):
    """La definizione coppia per coppia, senza ordinamenti. Ritorna (perdita, conteggio)."""
    anchors = [(s, p) for s in range(z.shape[0]) for p in range(stride // 2, z.shape[1], stride)
               if bool(tv[s, p]) and bool(mask[s, p].any())]
    h = torch.stack([z[s, p] for s, p in anchors]).double()
    u = F.normalize(F.layer_norm(h, h.shape[-1:]), dim=-1)
    sim = u @ u.T / temperature
    per_sample = {}
    for i, (si, pi) in enumerate(anchors):
        d = {}
        for k, (sk, pk) in enumerate(anchors):
            m = mask[si, pi] & mask[sk, pk]
            if k != i and bool(m.any()):
                v = float((target[si, pi, m] - target[sk, pk, m]).double().square().mean().sqrt())
                d[k] = math.floor(v / resolution) * resolution if resolution > 0 else v
        pos = [j for j in d if positives == "all" or anchors[j][0] != si]
        if len(pos) < 2:
            continue
        num = sum(torch.logsumexp(sim[i, [k for k in d if d[k] >= d[j]]], dim=0) - sim[i, j] for j in pos)
        chance = sum(math.log(sum(d[k] >= d[j] for k in d)) for j in pos)
        per_sample.setdefault(si, []).append(num / chance)
    if per_window:
        return torch.stack([torch.stack(v).mean() for v in per_sample.values()]).mean(), len(per_sample)
    flat = [x for v in per_sample.values() for x in v]
    return torch.stack(flat).mean(), len(flat)


@pytest.mark.parametrize("positives", ["cross", "all"])
@pytest.mark.parametrize("per_window", [True, False])
@pytest.mark.parametrize("resolution", [0.0, 0.5])
def test_matches_pairwise_definition_and_contract_count(positives, per_window, resolution):
    z, pose, tv = _data()
    loss, num, count, _ = pose_rank_contrast(z, pose, tv, positives=positives, per_window=per_window,
                                             resolution=resolution, **KW)
    ref, n_ref = reference(z, pose[0], pose[1], tv, 0.2, 2, positives, per_window, resolution)
    assert torch.allclose(loss, ref, rtol=1e-9, atol=1e-12) and int(count) == n_ref
    assert torch.allclose(num, loss * count)
    # contratto: il conteggio dalla sola maschera (senza encode ne' target) e' quello del forward
    pre = rank_contrast_count(pose[1] & tv[..., None], stride=2, positives=positives, per_window=per_window)
    assert pre.dtype == torch.int64 and torch.equal(pre, count)


def test_uniform_similarity_is_one():
    z, pose, tv = _data()
    z = z[:1, :1].expand_as(z).clone()  # embedding tutti uguali
    for per_window in (True, False):
        loss, *_ = pose_rank_contrast(z, pose, tv, positives="cross", per_window=per_window, **KW)
        assert torch.allclose(loss, torch.ones((), dtype=torch.float64), atol=1e-9)


def test_no_pose_or_single_window_gives_zero_count_and_null_gradient():
    z, pose, tv = _data()
    z = z.float().requires_grad_(True)
    m1 = pose[1].clone()
    m1[1:] = False  # la posa in una finestra sola
    m1[0, 4:] = True
    for p in (None, (pose[0], m1)):  # con "cross" nessuna coppia positiva
        loss, num, count, log = pose_rank_contrast(z, p, tv, positives="cross", per_window=True, **KW)
        assert float(loss) == 0.0 and int(count) == 0 and int(log["anchors"]) == 0
        (g,) = torch.autograd.grad(num, z)
        assert bool((g == 0).all())
    assert int(pose_rank_contrast(z, (pose[0], m1), tv, positives="all", per_window=True, **KW)[2]) == 1


def test_gradient_is_finite_and_descends():
    z, pose, tv = _data()
    z = z.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    history = []
    for _ in range(40):
        loss, *_ = pose_rank_contrast(z, pose, tv, positives="cross", per_window=True, **KW)
        opt.zero_grad()
        loss.backward()
        assert bool(torch.isfinite(z.grad).all())
        opt.step()
        history.append(float(loss))
    assert history[-1] < history[0]


def test_global_rng_untouched():
    z, pose, tv = _data()
    state = torch.random.get_rng_state()
    pose_rank_contrast(z, pose, tv, positives="all", per_window=True, **KW)
    rank_contrast_count(pose[1] & tv[..., None], stride=2, positives="all", per_window=True)
    assert torch.equal(state, torch.random.get_rng_state())
