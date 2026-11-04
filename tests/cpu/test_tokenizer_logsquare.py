"""Shared initialization, hidden-content invariance and real branch learning."""
from dataclasses import asdict, replace

import pytest
import torch

from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.model.patch_tokens import PatchTokenizer
from wearusfm.harness.fm_features import load_model
from wearusfm.training import run as R
from test_tokenizer_activation import CFG, inputs


def paired_models():
    torch.manual_seed(0)
    linear = WearUsFM(CFG).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(0)
    branch = WearUsFM(replace(CFG, tokenizer_logsquare=True)).eval()
    assert torch.equal(rng, torch.get_rng_state())
    return linear, branch


@pytest.mark.parametrize("fs", [200., 1000., 2000.])
def test_initial_full_model_and_all_shared_weights_exactly_match(fs):
    linear, branch = paired_models()
    old, new = linear.state_dict(), branch.state_dict()
    assert set(new) - set(old) == {"tokenizer.logsquare_weight"}
    assert all(torch.equal(value, new[key]) for key, value in old.items())
    assert branch.tokenizer.logsquare_weight.count_nonzero() == 0
    assert sum(p.numel() for p in branch.parameters()) - sum(p.numel() for p in linear.parameters()) == 64 * CFG.dim
    inp = inputs(fs)
    with torch.no_grad():
        before, after = linear.encode(inp), branch.encode(inp)
    assert torch.equal(before.local, after.local)
    assert torch.equal(before.z, after.z)
    assert branch.tokenizer.window_s == .1 and branch.tokenizer.patch_s == .025


def test_zero_new_weights_receive_gradient_and_then_change_output():
    linear, branch = paired_models()
    x = inputs(200.)
    a, b = linear.tokenizer(x.signals, x.fs)[0], branch.tokenizer(x.signals, x.fs)[0]
    a.square().mean().backward()
    b.square().mean().backward()
    for name, param in linear.tokenizer.named_parameters():
        other = dict(branch.tokenizer.named_parameters())[name]
        assert torch.equal(param.grad, other.grad), name
    grad = branch.tokenizer.logsquare_weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    torch.optim.AdamW(branch.parameters(), lr=1e-4).step()
    with torch.no_grad():
        assert not torch.equal(a, branch.tokenizer(x.signals, x.fs)[0])


@pytest.mark.parametrize("fs", [200., 1000., 2000.])
def test_active_branch_cannot_observe_hidden_raw_samples(fs):
    _, model = paired_models()
    with torch.no_grad():
        model.tokenizer.logsquare_weight.fill_(.02)
    inp = inputs(fs)
    visible = torch.ones(8, 40, dtype=torch.bool)
    visible[:, 10:20] = False
    visible[2] = False
    hidden = (~visible).repeat_interleave(int(fs / 40), dim=1)
    altered = inp.signals[0].clone()
    altered[hidden] += 1000
    with torch.no_grad():
        first = model.encode(inp, visible)
        second = model.encode(replace(inp, signals=[altered]), visible)
    assert torch.equal(first.local, second.local) and torch.equal(first.z, second.z)


def test_log_definition_evenness_fp32_under_autocast_and_finite_derivative():
    tokenizer = PatchTokenizer(4, tokenizer_logsquare=True)
    x = torch.tensor([-100., -1., -.031, 0., .031, 1., 100.], requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = tokenizer.logsquare_features(x)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, torch.log(x.square() + 1e-3), rtol=0, atol=0)
    assert torch.equal(y, y.flip(0))
    y.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach() / (1e-3 + x.detach().square()))
    assert torch.isfinite(x.grad).all() and x.grad[3] == 0


def test_new_and_legacy_checkpoints_load_strictly_and_recipe_switch_is_rejected(tmp_path):
    linear, branch = paired_models()
    for name, model in (("linear", linear), ("log", branch)):
        config = asdict(model.cfg)
        if name == "linear":
            config.pop("tokenizer_logsquare")
            config.pop("tokenizer_logsquare_eps")
        path = tmp_path / f'{name}.pt'
        torch.save({'config': {'model': config}, 'student': model.state_dict()}, path)
        restored = load_model(str(path), which='student')
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    with pytest.raises(RuntimeError):
        branch.load_state_dict(linear.state_dict(), strict=True)
    cfg = R.sanity_config()
    changed = replace(cfg, model=replace(cfg.model, tokenizer_logsquare=True))
    with pytest.raises(ValueError):
        R._check_resume_recipe(changed, {'config': R.config_to_dict(cfg)})


@pytest.mark.parametrize('enabled,eps,activation', [(1,1e-3,'identity'), (True,0,'identity'),
    (True,float('nan'),'identity'), (True,float('inf'),'identity'), (True,-1,'identity'),
    (True,True,'identity'), (True,1e-3,'gelu')])
def test_invalid_config_rejected_before_rng_is_consumed(enabled, eps, activation):
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        PatchTokenizer(4, tokenizer_activation=activation, tokenizer_logsquare=enabled, tokenizer_logsquare_eps=eps)
    with pytest.raises(ValueError):
        replace(CFG, tokenizer_activation=activation, tokenizer_logsquare=enabled, tokenizer_logsquare_eps=eps)
    assert torch.equal(rng, torch.get_rng_state())
