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
