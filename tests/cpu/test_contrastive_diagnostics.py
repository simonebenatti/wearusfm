"""Independent cross-window geometry oracles and observational invariance."""
import numpy as np
import pytest
from scipy.stats import spearmanr
import torch

from wearusfm.training.contrastive_diagnostics import cross_window_rank_alignment


KEY = "contrast_rnc_cross_window_rank_"


def fixture():
    values = torch.arange(3*5*4, dtype=torch.float64).reshape(3,5,4)
    z = (values.sin() + (values/3).cos()).requires_grad_()
    target = torch.arange(3*5*2, dtype=torch.float64).reshape(3,5,2)/4
    valid = torch.ones_like(target, dtype=torch.bool)
    tv = torch.ones((3,5), dtype=torch.bool)
    allowed = torch.ones((6,6), dtype=torch.bool)
    return z,target,valid,tv,allowed


def independent_oracle(z,target,valid,tv,stride,allowed):
    anchors = [(b,p) for b in range(z.shape[0]) for p in range(stride//2,z.shape[1],stride)
               if tv[b,p] and valid[b,p].any()]
    x,y = [],[]
    for i,(b,p) in enumerate(anchors):
        for j,(c,q) in enumerate(anchors[i+1:],i+1):
            common = (valid[b,p] & valid[c,q]).numpy()
            if b == c or not allowed[i,j] or not common.any():
                continue
            u = z[b,p].detach().numpy(); u = u-u.mean()
            v = z[c,q].detach().numpy(); v = v-v.mean()
            x.append(float(u@v / (np.linalg.norm(u)*np.linalg.norm(v))))
            delta = (target[b,p]-target[c,q]).numpy()[common]
            y.append(-float(np.sqrt(np.mean(delta*delta))))
    return len(x), float(spearmanr(x,y).statistic)


def test_matches_independent_upper_triangle_oracle_and_excludes_physical_duplicates():
    z,target,valid,tv,allowed = fixture()
    allowed[0,2] = allowed[2,0] = False
    valid[0,1,0] = False
    target[0,1,0] = float("nan")  # masked NaN must not contaminate shared-joint RMS
    out = cross_window_rank_alignment(z,target,valid,tv,2,allowed)
    count,corr = independent_oracle(z,target,valid,tv,2,allowed)
    assert count == 11
    assert out[KEY+"pairs"] == count
    assert out[KEY+"valid"] == 1
    assert out[KEY+"correlation"] == pytest.approx(corr,abs=1e-12)


def test_average_tie_ranks_are_used():
    z = torch.tensor([[[1.,0.,-1.]], [[0.,1.,-1.]], [[-1.,0.,1.]], [[0.,-1.,1.]]],dtype=torch.float64)
    target = torch.tensor([[[0.]],[[1.]],[[1.]],[[2.]]],dtype=torch.float64)
    valid = torch.ones_like(target,dtype=torch.bool)
    tv = torch.ones((4,1),dtype=torch.bool)
    allowed = torch.ones((4,4),dtype=torch.bool)
    out = cross_window_rank_alignment(z,target,valid,tv,1,allowed)
    count,corr = independent_oracle(z,target,valid,tv,1,allowed)
    assert count == 6 and out[KEY+"valid"] == 1
    assert out[KEY+"correlation"] == pytest.approx(corr,abs=1e-12)


@pytest.mark.parametrize("mode",["empty","single_window","one_pair","constant_feature","constant_target"])
def test_undefined_correlations_have_explicit_validity_and_finite_sentinel(mode):
    z,target,valid,tv,allowed = fixture()
    if mode == "empty":
        valid[:] = False; allowed = torch.ones((0,0),dtype=torch.bool)
    elif mode == "single_window":
        valid[1:] = False; allowed = torch.ones((2,2),dtype=torch.bool)
    elif mode == "one_pair":
        allowed[:] = False; allowed[0,2] = allowed[2,0] = True
    elif mode == "constant_feature":
        z = torch.ones_like(z)
    else:
        target[:] = 0
    out = cross_window_rank_alignment(z,target,valid,tv,2,allowed)
    assert out[KEY+"valid"] == 0 and out[KEY+"correlation"] == 0
    assert out[KEY+"pairs"] == (1 if mode == "one_pair" else 12 if mode.startswith("constant") else 0)
    assert all(type(v) is float and np.isfinite(v) for v in out.values())


def test_observation_preserves_gradients_rng_and_inputs():
    z,target,valid,tv,allowed = fixture()
    before = [a.detach().clone() for a in (z,target,valid,tv,allowed)]
    state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    loss = z.square().sum()
    expected = torch.autograd.grad(loss,z,retain_graph=True)[0]
    cross_window_rank_alignment(z,target,valid,tv,2,allowed)
    actual = torch.autograd.grad(loss,z)[0]
    assert torch.equal(actual,expected)
    assert torch.equal(state,torch.random.get_rng_state())
    after_numpy_state = np.random.get_state()
    assert all(np.array_equal(a,b) for a,b in zip(numpy_state,after_numpy_state))
    assert all(torch.equal(a.detach(),b) for a,b in zip((z,target,valid,tv,allowed),before))
    assert z.grad is None


def test_physical_patch_mask_controls_anchor_population():
    z,target,valid,tv,_ = fixture()
    tv[0,1] = False
    allowed = torch.ones((5,5),dtype=torch.bool)
    out = cross_window_rank_alignment(z,target,valid,tv,2,allowed)
    count,corr = independent_oracle(z,target,valid,tv,2,allowed)
    assert count == 8 and out[KEY+"pairs"] == 8
    assert out[KEY+"correlation"] == pytest.approx(corr,abs=1e-12)


def test_asymmetric_physical_mask_rejected():
    z,target,valid,tv,allowed = fixture()
    allowed[0,2] = False
    with pytest.raises(ValueError,match="symmetric"):
        cross_window_rank_alignment(z,target,valid,tv,2,allowed)
