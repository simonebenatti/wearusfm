"""Architecture, numerical policy and usable gradients of the new frozen readers."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import heads


@pytest.mark.parametrize("name", heads.HEADS)
def test_parameter_count_zero_output_and_time_preservation(name):
    model = heads.make_head(name).eval()
    assert sum(p.numel() for p in model.parameters()) == heads.PARAMETERS[name]
    x = torch.randn(2, 160, 384, dtype=torch.float16)
    result = model(x)
    assert result.shape == (2, 160, 20) and result.dtype == torch.float32
    assert torch.equal(result, torch.zeros_like(result))
    assert model.norm.eps == 1e-5 and torch.equal(model.norm.weight, torch.ones(384))
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.MaxPool1d, nn.AvgPool1d, nn.Dropout1d))
                   for m in model.modules())


@pytest.mark.parametrize("name", ("MLP", "TCN_post", "TCN_pre"))
def test_zero_deep_path_equals_l0_with_shared_affine_normalization(name):
    linear = heads.make_head("L0").eval()
    model = heads.make_head(name).eval()
    with torch.no_grad():
        linear.out.weight.normal_(std=.2)
        linear.out.bias.normal_(std=.2)
        linear.norm.weight.normal_(mean=1., std=.1)
        linear.norm.bias.normal_(std=.1)
        model.affine.load_state_dict(linear.out.state_dict())
        model.norm.load_state_dict(linear.norm.state_dict())
    x = torch.randn(2, 19, 384)
    assert torch.equal(model(x), linear(x))


def test_factory_preserves_global_rng_and_lr_fold_initial_identity():
    torch.manual_seed(145)
    state = torch.random.get_rng_state().clone()
    first = heads.make_head("TCN_post", 2)
    assert torch.equal(state, torch.random.get_rng_state())
    second = heads.make_head("TCN_post", 2)
    assert heads.state_sha256(first) == heads.state_sha256(second)
    assert heads.state_sha256(first) != heads.state_sha256(heads.make_head("TCN_post", 1))
    assert heads.state_sha256(heads.make_head("L0", 0)) == heads.state_sha256(heads.make_head("L0", 2))
    assert heads.state_sha256(heads.make_head("TCN_pre", 2)) == heads.state_sha256(first)
    assert heads.trajectory_seeds(2, 1) == {"initialization":100002,"dropout":200102,"shuffle":300102}


def test_factory_does_not_seed_cuda_even_when_constructing_on_cpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    heads.make_head("TCN_post", 1)
    assert calls == []


@pytest.mark.parametrize("name", ("MLP", "TCN_post"))
def test_zero_output_initialization_learns_deep_output_then_hidden(name):
    model = heads.make_head(name).train()
    x, y = torch.randn(2, 16, 384), torch.randn(2, 16, 20)
    transforms = [m for m in model.deep if isinstance(m, (nn.Linear, nn.Conv1d))]
    optimizer = torch.optim.AdamW(heads.optimizer_groups(model), lr=1e-3)
    loss = (model(x)-y).square().mean()
    loss.backward()
    assert transforms[-1].weight.grad.abs().sum() > 0
    assert model.affine.weight.grad.abs().sum() > 0
    assert transforms[0].weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (model(x)-y).square().mean().backward()
    assert transforms[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ("L13", "TCN_post"))
def test_exact_thirteen_patch_receptive_field(name):
    model = heads.make_head(name).eval()
    with torch.no_grad():
        if name == "L13":
            model.out.weight.normal_(std=.1)
        else:
            model.deep[-1].weight.normal_(std=.1)
    x = torch.randn(1, 41, 384, requires_grad=True)
    gradient = torch.autograd.grad(model(x)[0, 20, 0], x)[0]
    active = set(torch.nonzero(gradient.abs().sum(dim=-1)[0] > 0).flatten().tolist())
    assert active == set(range(14, 27))


def test_dropout_is_elementwise_only_deep_and_disabled_in_eval():
    model = heads.make_head("MLP")
    drops = [m for m in model.deep if isinstance(m, nn.Dropout)]
    assert len(drops) == 2 and all(m.p == .1 for m in drops)
    with torch.no_grad():
        model.deep[-1].weight.normal_(std=.1)
    x = torch.randn(2, 16, 384)
    model.train()
    assert not torch.equal(model(x), model(x))
    model.eval()
    assert torch.equal(model(x), model(x))


def test_decay_applies_to_kernels_not_layernorm_bias_and_fp32_policy():
    model = heads.make_head("TCN_post")
    groups = heads.optimizer_groups(model)
    decayed = {id(p) for p in groups[0]["params"]}
    untouched = {id(p) for p in groups[1]["params"]}
    assert decayed.isdisjoint(untouched)
    assert decayed | untouched == {id(p) for p in model.parameters()}
    for name, parameter in model.named_parameters():
        assert (id(parameter) in decayed) == (parameter.ndim >= 2 and name.endswith("weight"))
    model.half()
    with pytest.raises(ValueError, match="FP32"):
        model(torch.zeros(1, 16, 384))


def test_first_projection_contains_no_future_context_for_instantaneous_mlp():
    model = heads.make_head("MLP").eval()
    with torch.no_grad():
        model.deep[-1].weight.normal_(std=.1)
    x = torch.randn(1, 9, 384, requires_grad=True)
    grad = torch.autograd.grad(model(x)[0, 4, 0], x)[0]
    assert set(torch.nonzero(grad.abs().sum(-1)[0] > 0).flatten().tolist()) == {4}
