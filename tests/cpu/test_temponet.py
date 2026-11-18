"""TEMPONet per regressione (Zanghieri et al., COINS 2021): conteggi del paper, forme, causalita' e MAC."""

import pytest
import torch

from wearusfm.model.temponet import (
    TEMPONet,
    _TemporalConv,
    count_macs,
    count_parameters,
    output_length,
    receptive_field,
)


def test_parameter_count_matches_paper_70_9k():
    # [Z21]: 70,9 kB in int8 con Block III a 64 canali; parametri addestrabili 70.917 con la lettura del documento
    assert count_parameters(TEMPONet()) == 70_917


def test_largest_explored_model_below_500k():
    # [Z21] sez. II-C2: «less than 500 k parameters» anche con Block III a 192 canali
    assert count_parameters(TEMPONet(block_channels=(16, 32, 192), fc_units=(192, 32))) == 481_541


def test_shapes_receptive_field_and_macs():
    m = TEMPONet().eval()
    assert m.out_length == 4 == output_length(256)
    assert receptive_field() == 378  # [Z20] Fig. 3: 377, contato in modo leggermente diverso
    with torch.no_grad():
        assert m(torch.randn(7, 16, 256)).shape == (7, 5)
    # convoluzioni: 720.896 + 917.504 + 753.664; densi: 16.384 + 2.048 + 160
    assert count_macs(m) == 2_410_656


def test_input_shape_is_checked():
    m = TEMPONet()
    with pytest.raises(ValueError):
        m(torch.randn(2, 16, 255))
    with pytest.raises(ValueError):
        m(torch.randn(2, 8, 256))


@pytest.mark.parametrize("causal", [True, False])
def test_temporal_conv_keeps_length(causal):
    layer = _TemporalConv(4, 6, dilation=8, causal=causal).eval()
    assert layer(torch.randn(2, 4, 50)).shape == (2, 6, 50)


def test_causal_conv_ignores_future_samples():
    torch.manual_seed(0)
    layer = _TemporalConv(3, 5, dilation=4, causal=True).eval()
    x = torch.randn(1, 3, 40)
    y = x.clone()
    y[:, :, 30:] += 10.0
    with torch.no_grad():
        a, b = layer(x), layer(y)
    assert torch.equal(a[:, :, :30], b[:, :, :30])
    assert not torch.equal(a[:, :, 30:], b[:, :, 30:])


def test_symmetric_padding_sees_future_samples():
    torch.manual_seed(0)
    layer = _TemporalConv(3, 5, dilation=4, causal=False).eval()
    x = torch.randn(1, 3, 40)
    y = x.clone()
    y[:, :, 30:] += 10.0
    with torch.no_grad():
        assert not torch.equal(layer(x)[:, :, 26:30], layer(y)[:, :, 26:30])


def test_backward_runs_in_train_mode():
    m = TEMPONet().train()
    loss = m(torch.randn(8, 16, 256)).abs().mean()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


# --- variante residua (18/11/2026, docs/temponet_emg2pose_residual_plan_20261118.md) -------------------------------------------------------

def _state(seed, **kw):
    torch.manual_seed(seed)
    return TEMPONet(**kw).state_dict()


def test_default_structure_is_the_plain_network():
    m = TEMPONet()
    assert not m.residual and not hasattr(m, "skips")
    assert [type(x).__name__ for x in m.features] == ["_TemporalConv", "_TemporalConv", "_StridedConv"] * 3
    assert all(isinstance(m.features[i].relu, torch.nn.ReLU) for i in (2, 5, 8))
    assert all(k.startswith(("features.", "head.", "out.")) for k in m.state_dict())


def test_residual_counts_and_receptive_field():
    # salti: AvgPool + 1x1 senza bias + BN; 16x16+32, 16x32+64, 32x64+128 = 3.040 parametri; MAC 16x128x16 + 32x32x16 + 64x4x32 = 57.344
    assert count_parameters(TEMPONet(residual=True)) == 70_917 + 3_040 == 73_957
    assert count_parameters(TEMPONet(out_dim=20, causal=False, residual=True)) == 71_412 + 3_040 == 74_452
    assert count_macs(TEMPONet(residual=True).eval()) == 2_410_656 + 57_344
    assert receptive_field() == 378  # i salti 1x1 non allargano il campo ricettivo


def test_residual_shares_the_initial_weights_of_the_plain_network():
    for seed in (0, 777):
        plain, res = _state(seed, out_dim=20, causal=False), _state(seed, out_dim=20, causal=False, residual=True)
        assert set(plain) < set(res) and all(k.startswith("skips.") for k in set(res) - set(plain))
        assert all(torch.equal(plain[k], res[k]) for k in plain)


def test_residual_with_zero_skips_is_the_plain_network():
    # con gamma e beta delle BN dei salti a zero il salto vale 0: la rete residua coincide con quella senza salti, stessi pesi
    torch.manual_seed(1)
    res = TEMPONet(out_dim=20, causal=False, residual=True)
    plain = TEMPONet(out_dim=20, causal=False)
    plain.load_state_dict({k: v for k, v in res.state_dict().items() if not k.startswith("skips.")})
    with torch.no_grad():
        for s in res.skips:
            s.bn.weight.zero_()
            s.bn.bias.zero_()
    x = torch.randn(6, 16, 256)
    for mode in ("train", "eval"):
        res.train(mode == "train")
        plain.train(mode == "train")
        torch.manual_seed(2)
        a = res(x)
        torch.manual_seed(2)
        b = plain(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        res.skips[1].bn.bias.fill_(0.5)
    assert not torch.equal(res(x), plain(x))  # un salto non nullo cambia l'uscita


def test_residual_skip_lengths_match_the_blocks():
    m = TEMPONet(residual=True).eval()
    x = torch.randn(2, 16, 256)
    lengths = []
    with torch.no_grad():
        for i, s in enumerate(m.skips):
            y = s(x)
            main = m.features[3 * i + 2](m.features[3 * i + 1](m.features[3 * i](x)))
            assert y.shape == main.shape
            lengths.append(y.shape[2])
            x = m.relu(main + y)
    assert lengths == [128, 32, 4]
    TEMPONet(window=262)  # senza salti va bene
    with pytest.raises(ValueError):
        TEMPONet(window=262, residual=True)  # blocco 2: 33 campioni, salto 32


@pytest.mark.parametrize("residual", [False, True])
def test_invariant_to_the_global_input_scale(residual):
    # convoluzioni senza bias seguite da BatchNorm (anche nei salti): la scala 10^k dell'ingresso non cambia l'uscita
    torch.manual_seed(0)
    m = TEMPONet(out_dim=20, causal=False, dropout=0.0, residual=residual).train()
    x = torch.randn(32, 16, 256)
    torch.testing.assert_close(m(10.0 * x), m(x), rtol=1e-4, atol=1e-4)


def test_residual_forward_backward():
    m = TEMPONet(out_dim=20, causal=False, residual=True).train()
    out = m(torch.randn(8, 16, 256))
    assert out.shape == (8, 20)
    out.abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert all(float(s.conv.weight.grad.abs().sum()) > 0 for s in m.skips)
    with torch.no_grad():
        assert m.eval()(torch.randn(3, 16, 256)).shape == (3, 20)
