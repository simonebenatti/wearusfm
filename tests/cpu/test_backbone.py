"""Backbone temporale (v10 §5) e modello in fila fino al backbone su un batch con le tre topologie. Richiede torch: si salta se non c'e'
(gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.ingest import camargo, capgmyo, emg2pose, ninapro_std  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model.backbone import TemporalBackbone, rope  # noqa: E402
from wearusfm.model.channel_identity import ChannelIdentity  # noqa: E402
from wearusfm.model.local_encoder import LocalEncoder, sets_to_tensors  # noqa: E402
from wearusfm.model.perceiver import PerceiverPooling, offsets_from_counts  # noqa: E402

DIM, HEADS, K, P = 16, 4, 6, 7


def _bb(seed=0, layers=2):
    torch.manual_seed(seed)
    return TemporalBackbone(DIM, HEADS, layers).eval()


def _z(s=2, seed=1):
    return torch.randn(s, P, K, DIM, generator=torch.Generator().manual_seed(seed))


def test_rope_is_relative():
    x = torch.randn(1, 5, 2, 8)
    pos = torch.arange(5)[None]
    a, b = rope(x, pos), rope(x, pos + 37)
    dots_a = torch.einsum("blhd,bmhd->bhlm", a, a)
    dots_b = torch.einsum("blhd,bmhd->bhlm", b, b)
    assert torch.allclose(dots_a, dots_b, atol=1e-4)  # il prodotto scalare dipende solo dalla differenza di posizione
    with pytest.raises(ValueError, match="pari"):
        rope(torch.randn(1, 5, 2, 7), pos)


def test_shape_relative_positions_and_latent_permutation():
    bb, z = _bb(), _z()
    out = bb(z)
    assert out.shape == z.shape
    pos = torch.arange(P)[None].expand(2, P) + 100
    assert torch.allclose(bb(z, positions=pos), out, atol=1e-4)  # spostare tutte le posizioni non cambia nulla
    perm = torch.randperm(K, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(bb(z[:, :, perm]), out[:, :, perm], atol=1e-5)  # nessuna posizione sui latenti


def test_invalid_instants_are_not_keys_and_samples_are_independent():
    bb, z = _bb(), _z()
    valid = torch.ones(2, P, dtype=torch.bool)
    valid[0, 5:] = False  # padding del campione 0 (finestra piu' corta)
    out = bb(z, valid)
    z2 = z.clone()
    z2[0, 5:] = 50 * torch.randn(2, K, DIM)
    out2 = bb(z2, valid)
    assert torch.allclose(out2[0, :5], out[0, :5], atol=1e-5) and torch.allclose(out2[1], out[1])
    z3 = z.clone()
    z3[1] = torch.randn(P, K, DIM)
    assert torch.allclose(bb(z3, valid)[0], out[0], atol=1e-6)


def test_flops_counted():
    f = TemporalBackbone.flops_per_block(1, 160, 64, 384)
    tokens = 160 * 64
    assert f == 2 * (2 * tokens * 384 * 4 * 384 + 64 * 160 * 160 * 384 * 2 + 160 * 64 * 64 * 384 * 2 + tokens * 384 * 1536 * 2)


def test_three_topologies_forward_and_backward_through_the_backbone():
    """Criterio di chiusura del passo 6, parte «un batch con le tre topologie insieme fa forward e backward»: identita' -> encoder locale ->
    Perceiver -> backbone, con un canale scartato, un canale nascosto e uno slab temporale; manca il decoder a query (D11)."""
    torch.manual_seed(0)
    montages = [emg2pose.build_montage_metadata("u1", "s", "left"), camargo.build_montage_metadata(1, "d"),
                capgmyo.build_montage_metadata(1), ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right")]
    counts = [m.n_channels for m in montages]
    valids = [[True] * c for c in counts]
    valids[0][3] = False  # un canale scartato dal QC nell'anello
    dicts = [montage_to_dict(m, v) for m, v in zip(montages, valids)]
    codes = [CC.anatomy_codes(d) for d in dicts]
    packed = {f: torch.as_tensor(np.concatenate([getattr(c, f) for c in codes])) for f in ("region", "compartment_weights", "muscle",
                                                                                            "muscle_known", "topology")}
    packed["compartment_weights"] = packed["compartment_weights"].float()
    sets = sets_to_tensors(CC.pack_attention_sets([CC.attention_sets(CC.layout_from_montage(d), 4) for d in dicts]))
    ident, enc = ChannelIdentity(DIM, muscle_dropout=0.3), LocalEncoder(DIM, HEADS, 2)
    pool, bb = PerceiverPooling(DIM, HEADS, K), TemporalBackbone(DIM, HEADS, 2)
    tokens = torch.randn(sum(counts), P, DIM) + ident(packed)[:, None, :]
    visible = torch.as_tensor(np.concatenate(valids))[:, None].expand(-1, P).clone()
    visible[20] = False  # un canale di Camargo nascosto (masking spaziale)
    visible[:, 4] = False  # uno slab su tutti i canali
    lat = pool(enc(tokens, sets), offsets_from_counts(counts), visible)
    time_valid = torch.ones(4, P, dtype=torch.bool)
    time_valid[:, 4] = False  # l'istante nascosto non e' una chiave
    out = bb(lat, time_valid)
    assert out.shape == (4, P, K, DIM) and torch.isfinite(out).all()
    out.square().mean().backward()
    for module in (ident, enc, pool, bb):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
