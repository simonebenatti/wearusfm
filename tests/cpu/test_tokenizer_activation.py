"""Identity/GELU tokenizers: fixed support, legacy parity and actual learnability."""
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.harness import fm_features as FF
from wearusfm.model import channel_codes as CC
from wearusfm.model import frontend as FE
from wearusfm.model.channel_identity import codes_to_tensors
from wearusfm.model.fm import FMConfig, ModelInputs, WearUsFM
from wearusfm.model.local_encoder import sets_to_tensors
from wearusfm.model.patch_tokens import PatchTokenizer
from wearusfm.training.jepa import hidden_queries
from test_frontend import _rel, _sig

CFG = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1,
               muscle_dropout=0., window_ms=100.)


def inputs(fs, seed=14):
    montage = FF.myo8_montage("epn612", fs)
    layout, code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
    sets = CC.attention_sets(layout, 8)
    signal = torch.randn(8, int(fs), generator=torch.Generator().manual_seed(seed))
    return ModelInputs([signal], [fs], codes_to_tensors(CC.pack_codes([code])),
                       sets_to_tensors(CC.pack_attention_sets([sets])), [8], torch.as_tensor(layout.qc_valid))


@pytest.mark.parametrize("window_ms", [None, 100.])
def test_identity_preserves_legacy_forward_state_and_rng(window_ms):
    kw = FE.window_kwargs(25., window_ms)
    # Independent legacy construction: frontend then Linear; gain calibration
    # has no random draws from torch's global generator.
    torch.manual_seed(81)
    legacy_frontend = FE.ContinuousKernelFrontEnd(patch_ms=25., **kw)
    legacy_projection = torch.nn.Linear(64, 16)
    legacy_rng = torch.random.get_rng_state().clone()
    tokenizers = []
    for activation in (None, "identity", "gelu"):
        torch.manual_seed(81)
        extra = {} if activation is None else {"tokenizer_activation": activation}
        tokenizer = PatchTokenizer(16, patch_ms=25., **kw, **extra)
        assert torch.equal(torch.random.get_rng_state(), legacy_rng)
        assert all(torch.equal(v, tokenizer.frontend.state_dict()[k]) for k, v in legacy_frontend.state_dict().items())
        assert all(torch.equal(v, tokenizer.proj.state_dict()[k]) for k, v in legacy_projection.state_dict().items())
        tokenizers.append(tokenizer)
    base, explicit, nonlinear = tokenizers
    assert base.tokenizer_activation == explicit.tokenizer_activation == "identity"
    assert base.state_dict().keys() == explicit.state_dict().keys() == nonlinear.state_dict().keys()
    assert all(torch.equal(v, nonlinear.state_dict()[k]) for k, v in base.state_dict().items())
    assert torch.equal(base.fe_gain, explicit.fe_gain) and torch.equal(base.fe_gain, nonlinear.fe_gain)
    generator = torch.Generator().manual_seed(13)
    signals = [torch.randn(2, 200, generator=generator), torch.randn(3, 500, generator=generator)]
    rates = [200., 1000.]
    with torch.no_grad():
        actual, pv, tv = base(signals, rates)
        explicit_out = explicit(signals, rates)
        expected_parts = []
        for signal, fs in zip(signals, rates):
            features = legacy_frontend(signal[None], fs)[0]
            projected = legacy_projection((features * base.fe_gain).to(legacy_projection.weight.dtype))
            expected_parts.append(torch.cat((projected, projected.new_zeros(len(signal), 40 - projected.shape[1], 16)), 1))
        assert torch.equal(actual, torch.cat(expected_parts))
        assert torch.equal(actual, explicit_out[0]) and torch.equal(pv, explicit_out[1]) and torch.equal(tv, explicit_out[2])
        assert actual.shape == (5, 40, 16) and not pv[2:, 20:].any() and not tv[1, 20:].any()
        assert not actual[2:, 20:].any()


def test_model_activation_changes_no_parameters_gain_or_rng():
    torch.manual_seed(21)
    linear = WearUsFM(CFG).eval()
    linear_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(21)
    nonlinear = WearUsFM(replace(CFG, tokenizer_activation="gelu")).eval()
    assert torch.equal(linear_rng, torch.random.get_rng_state())
    assert linear.state_dict().keys() == nonlinear.state_dict().keys()
    assert all(torch.equal(v, nonlinear.state_dict()[k]) for k, v in linear.state_dict().items())
    assert sum(p.numel() for p in linear.parameters()) == sum(p.numel() for p in nonlinear.parameters())
    assert nonlinear.tokenizer.tokenizer_activation == "gelu"
    assert torch.equal(linear.tokenizer.fe_gain, nonlinear.tokenizer.fe_gain)
    assert linear.cfg.core_radius == nonlinear.cfg.core_radius == 2


@pytest.mark.parametrize("activation", ["identity", "gelu"])
def test_signal_map_before_projection_is_linear_only_for_identity(activation):
    torch.manual_seed(3)
    tokenizer = PatchTokenizer(16, tokenizer_activation=activation, **FE.window_kwargs(25., 100.))
    signals = torch.randn(2, 2, 1000, generator=torch.Generator().manual_seed(2))
    captured = []
    hook = tokenizer.proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            for signal in (signals[0], signals[1], signals.sum(0)):
                tokenizer([signal], [2000.])
    finally:
        hook.remove()
    first, second, combined = captured
    assert first.dtype == torch.float32 and first.shape[-1] == 64
    if activation == "identity":
        torch.testing.assert_close(combined, first + second, rtol=2e-5, atol=2e-6)
    else:
        relative_error = torch.linalg.vector_norm(combined - first - second) / torch.linalg.vector_norm(combined)
        assert relative_error > .05
    with torch.no_grad():
        scaled = (tokenizer.frontend(signals[0][None], 2000.)[0] * tokenizer.fe_gain).float()
        expected = torch.nn.functional.gelu(scaled) if activation == "gelu" else scaled
        assert torch.equal(first, expected)  # activation follows gain, before the 64 -> D projection


@pytest.mark.parametrize("activation", ["identity", "gelu"])
def test_activation_retains_40_tokens_per_second_and_finite_gradients_to_all_kernel_families(activation):
    torch.manual_seed(8)
    tokenizer = PatchTokenizer(16, tokenizer_activation=activation, **FE.window_kwargs(25., 100.))
    signals = [torch.randn(2, fs, generator=torch.Generator().manual_seed(fs)).requires_grad_() for fs in (200, 1000, 2000)]
    tokens, patch_valid, time_valid = tokenizer(signals, [200., 1000., 2000.])
    assert tokens.shape == (6, 40, 16) and patch_valid.all() and time_valid.all()
    assert torch.isfinite(tokens).all()
    assert tokenizer.window_s == pytest.approx(.1) and tokenizer.patch_s == .025
    assert tokenizer.frontend.context_s == .1  # 100-ms kernel + two 100-ms antialias tails = 300-ms numerical support
    tokens.square().mean().backward()
    for parameter in (tokenizer.frontend.fourier_coef, tokenizer.frontend.spline_coef,
                      tokenizer.frontend.mlp.l1.weight, tokenizer.frontend.mlp.l2.weight,
                      tokenizer.frontend.mlp.l3.weight, tokenizer.proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(signal.grad is not None and torch.isfinite(signal.grad).all() and signal.grad.abs().sum() > 0 for signal in signals)
    assert not tokenizer.fe_gain.requires_grad


@pytest.mark.parametrize("fs_b,fmax", [(1000, 440), (200, 88)])
def test_gelu_preserves_cross_sampling_rate_features_with_the_declared_d8_tolerance(fs_b, fmax):
    """The same band-limited signal, after gain + GELU and before projection."""
    torch.manual_seed(3)
    tokenizer = PatchTokenizer(16, tokenizer_activation="gelu", **FE.window_kwargs(25., 100.))
    captured = []
    hook = tokenizer.proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            for fs in (2000, fs_b):
                tokenizer([_sig(fs, 2., np.linspace(5, fmax, 40))[0]], [float(fs)])
    finally:
        hook.remove()
    first, second = (value[:, 8:-8] for value in captured)  # 200 ms per edge, beyond the numerical support radius
    for name, section in tokenizer.frontend.family_slices().items():
        error = _rel(first, second, section)
        assert error < .02, (fs_b, name, error)  # same preregistered 2% tolerance as the 100-ms linear frontend


@pytest.mark.parametrize("fs", [200., 1000., 2000.])
def test_gelu_student_never_sees_perturbed_hidden_raw_content(fs):
    torch.manual_seed(27)
    model = WearUsFM(replace(CFG, tokenizer_activation="gelu")).eval()
    inp = inputs(fs)
    visible = torch.ones(8, 40, dtype=torch.bool)
    visible[:, 12:20] = False
    visible[2] = False
    visible[5, 28:32] = False
    hidden_samples = (~visible).repeat_interleave(int(fs / 40), dim=1)
    signal = inp.signals[0].clone()
    signal[hidden_samples] += 100 * torch.randn(int(hidden_samples.sum()), generator=torch.Generator().manual_seed(5))
    assert torch.equal(signal[~hidden_samples], inp.signals[0][~hidden_samples])
    noisy = replace(inp, signals=[signal])
    q_ch, q_t, q_off = hidden_queries(~visible, inp.counts)
    with torch.no_grad():
        first, second = model.encode(inp, visible), model.encode(noisy, visible)
        assert torch.equal(first.local, second.local) and torch.equal(first.z, second.z)
        assert torch.equal(model.decode(first, q_ch, q_t, q_off), model.decode(second, q_ch, q_t, q_off))
        assert not torch.allclose(model.encode(inp, None).z, model.encode(noisy, None).z)


@pytest.mark.parametrize("activation", [None, "identity", "gelu"])
def test_checkpoint_config_selects_activation_and_missing_field_means_legacy(tmp_path, activation):
    cfg = CFG if activation is None else replace(CFG, tokenizer_activation=activation)
    torch.manual_seed(7)
    model = WearUsFM(cfg).eval()
    config = asdict(cfg)
    if activation is None:
        del config["tokenizer_activation"]
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": {"model": config}, "student": model.state_dict()}, path)
    restored = FF.load_model(str(path), which="student")
    assert restored.cfg.tokenizer_activation == (activation or "identity")
    assert restored.tokenizer.tokenizer_activation == restored.cfg.tokenizer_activation
    inp = inputs(200.)
    with torch.no_grad():
        assert torch.equal(model.encode(inp).z, restored.encode(inp).z)
    if activation == "gelu":
        linear = WearUsFM(CFG).eval()
        linear.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            assert not torch.allclose(linear.encode(inp).z, restored.encode(inp).z)


@pytest.mark.parametrize("invalid", ["relu", "GELU", "", None, True, []])
def test_invalid_activation_is_rejected_before_parameter_initialization(invalid):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="tokenizer_activation"):
        replace(CFG, tokenizer_activation=invalid)
    with pytest.raises(ValueError, match="tokenizer_activation"):
        PatchTokenizer(16, tokenizer_activation=invalid)
    assert torch.equal(before, torch.random.get_rng_state())
