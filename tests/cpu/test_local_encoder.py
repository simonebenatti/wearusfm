"""Encoder spaziale locale (v10 §5.3-5.4) sui montaggi veri degli adattatori. Richiede torch: si salta se non c'e'
(gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.ingest import camargo, capgmyo, emg2pose, ninapro_std  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model.local_encoder import GeometricBias, LocalEncoder, sets_to_tensors  # noqa: E402

DIM, HEADS, P = 16, 4, 3


def _sets(montage_dict, k=4):
    return CC.attention_sets(CC.layout_from_montage(montage_dict), k)


def _enc(seed=0, **kw):
    torch.manual_seed(seed)
    enc = LocalEncoder(DIM, HEADS, 2, **kw).eval()
    with torch.no_grad():  # bias appreso non nullo, cosi' i test vedono anche la componente appresa
        for layer in enc.layers:
            for prm in (layer.bias.ring_w, layer.bias.row_w, layer.bias.col_w, layer.bias.type_bias):
                prm.normal_(0, 0.5)
    return enc


def _x(c, seed=1):
    return torch.randn(c, P, DIM, generator=torch.Generator().manual_seed(seed))


def test_ring_rotation_changes_nothing_and_permutation_permutes():
    m = montage_to_dict(emg2pose.build_montage_metadata("u1", "s", "left"), [True] * 16)
    enc, x = _enc(), _x(16)
    base = enc(x, sets_to_tensors(_sets(m)))
    rot = copy.deepcopy(m)
    for ch in rot["groups"][0]["channels"]:
        ch["sensor_coords"]["ring_angle_deg"] = (ch["sensor_coords"]["ring_angle_deg"] + 5 * 22.5) % 360.0
    assert torch.allclose(enc(x, sets_to_tensors(_sets(rot))), base, atol=1e-6)
    perm = np.random.default_rng(0).permutation(16)
    sh = copy.deepcopy(m)
    sh["groups"][0]["channels"] = [m["groups"][0]["channels"][i] for i in perm]
    assert torch.allclose(enc(x[perm], sets_to_tensors(_sets(sh))), base[perm], atol=1e-6)


def test_packed_montages_do_not_see_each_other():
    a = montage_to_dict(ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"), [True] * 12)
    b = montage_to_dict(capgmyo.build_montage_metadata(1), [True] * 128)
    g = sets_to_tensors(CC.pack_attention_sets([_sets(a), _sets(b)]))
    enc, x = _enc(), _x(140)
    out = enc(x, g)
    x2 = x.clone()
    x2[12:] = torch.randn(128, P, DIM)  # cambia tutto il secondo montaggio
    assert torch.allclose(enc(x2, g)[:12], out[:12], atol=1e-6)
    assert torch.allclose(enc(x[:12], sets_to_tensors(_sets(a))), out[:12], atol=1e-6)  # come da solo


def test_sparse_channels_see_the_whole_montage_ring_channels_do_not_see_sparse():
    m = montage_to_dict(ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"), [True] * 12)  # 8 di anello + 4 mirati
    enc, x = _enc(), _x(12)
    g = sets_to_tensors(_sets(m, k=2))
    out = enc(x, g)
    x2 = x.clone()
    x2[10] += torch.randn(P, DIM)  # un elettrodo mirato (rumore, non una costante: il LayerNorm toglie la media)
    out2 = enc(x2, g)
    assert torch.allclose(out2[:8], out[:8], atol=1e-6)  # l'anello non guarda i mirati
    assert not torch.allclose(out2[8], out[8])  # un altro mirato si'
    cam = montage_to_dict(camargo.build_montage_metadata(1, "d"), [True] * 11)
    gc = sets_to_tensors(_sets(cam))
    assert (gc["pair_type"][:, 1:] == CC.PAIR_SET).all()  # Camargo: solo insieme, nessuna geometria
    xc = _x(11)
    xc2 = xc.clone()
    xc2[3] += torch.randn(P, DIM)
    assert not torch.allclose(enc(xc2, gc)[0], enc(xc, gc)[0])


def test_qc_invalid_channel_is_ignored_by_the_others():
    m = montage_to_dict(emg2pose.build_montage_metadata("u1", "s", "left"), [i != 1 for i in range(16)])
    enc, x = _enc(), _x(16)
    g = sets_to_tensors(_sets(m))
    out = enc(x, g)
    x2 = x.clone()
    x2[1] = 100.0 * torch.randn(P, DIM)
    assert torch.allclose(torch.cat([enc(x2, g)[:1], enc(x2, g)[2:]]), torch.cat([out[:1], out[2:]]), atol=1e-6)


def test_bias_starts_from_the_physical_prior_and_learns():
    m = montage_to_dict(capgmyo.build_montage_metadata(1), [True] * 128)
    g = sets_to_tensors(_sets(m))
    b = GeometricBias(HEADS)
    out = b(g)
    expected = -torch.nan_to_num(g["dist"], nan=0.0)[..., None].expand(-1, -1, HEADS)
    assert torch.allclose(out, expected)  # pesi appresi a zero: solo il decadimento con la distanza
    enc = LocalEncoder(DIM, HEADS, 1)
    enc(_x(128), g).sum().backward()
    assert enc.layers[0].bias.row_w.grad.abs().sum() > 0 and enc.layers[0].bias.ring_w.grad.abs().sum() == 0  # griglia: niente anello


def test_wrong_keys_are_rejected_and_flops_are_counted():
    m = montage_to_dict(emg2pose.build_montage_metadata("u1", "s", "left"), [True] * 16)
    g = sets_to_tensors(_sets(m))
    with pytest.raises(ValueError, match="colonna 0"):
        LocalEncoder(DIM, HEADS, 1)(_x(15), g)
    f = LocalEncoder.flops_per_layer(16, 5, 160, 384)
    assert f == 2 * (16 * 160 * 384 * 384 * 4 + 16 * 160 * 5 * 384 * 2 + 16 * 160 * 384 * 1536 * 2)
