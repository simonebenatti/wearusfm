"""Decoder a query e diagnostica del collasso da query (v10 §5.6, §7.1 punto 4); modello in fila fino al decoder sulle tre topologie.
Richiede torch: si salta se non c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.ingest import camargo, capgmyo, emg2pose, ninapro_std  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model.backbone import TemporalBackbone  # noqa: E402
from wearusfm.model.channel_identity import ChannelIdentity, codes_to_tensors  # noqa: E402
from wearusfm.model.local_encoder import LocalEncoder, sets_to_tensors  # noqa: E402
from wearusfm.model.perceiver import PerceiverPooling, offsets_from_counts  # noqa: E402
from wearusfm.model.query_decoder import QueryDecoder, probe_query_collapse, query_collapse_stats  # noqa: E402

DIM, HEADS, K, P = 16, 4, 6, 7


def _dec(seed=0):
    torch.manual_seed(seed)
    return QueryDecoder(DIM, HEADS, 2).eval()


def _z(s=3, seed=1):
    return torch.randn(s, P, K, DIM, generator=torch.Generator().manual_seed(seed))


def test_queries_see_only_their_sample_and_only_valid_instants():
    dec, z = _dec(), _z()
    q = torch.randn(9, DIM)
    t = torch.tensor([0, 3, 6, 1, 1, 2, 5, 4, 0])
    off = torch.tensor([0, 3, 5, 9])
    out = dec(z, q, t, off)
    assert out.shape == (9, DIM)
    z2 = z.clone()
    z2[1] = torch.randn(P, K, DIM)
    out2 = dec(z2, q, t, off)
    assert torch.allclose(out2[:3], out[:3]) and torch.allclose(out2[5:], out[5:]) and not torch.allclose(out2[3:5], out[3:5])
    valid = torch.ones(3, P, dtype=torch.bool)
    valid[0, 4] = False  # istante nascosto allo studente
    z3 = z.clone()
    z3[0, 4] = 50 * torch.randn(K, DIM)
    assert torch.allclose(dec(z3, q, t, off, valid)[:3], dec(z, q, t, off, valid)[:3], atol=1e-5)


def test_relative_time_and_bad_offsets():
    dec, z = _dec(), _z(1)
    q, t = torch.randn(4, DIM), torch.tensor([0, 2, 4, 6])
    off = torch.tensor([0, 4])
    out = dec(z, q, t, off)
    shifted = torch.arange(P)[None] + 50
    assert torch.allclose(dec(z, q, t, off, positions=shifted), out, atol=1e-4)  # conta solo la distanza fra istanti
    with pytest.raises(ValueError, match="query_offsets"):
        dec(z, q, t, torch.tensor([0, 3]))


def test_collapse_stats_detect_a_query_only_decoder():
    out = torch.randn(1, 5, 8).expand(4, 5, 8)  # stessa uscita per ogni campione: collasso
    assert query_collapse_stats(out)["ratio"] == pytest.approx(0.0, abs=1e-12)
    healthy = torch.randn(4, 5, 8)
    assert query_collapse_stats(healthy)["ratio"] > 0.5
    with pytest.raises(ValueError):
        query_collapse_stats(torch.randn(1, 5, 8))
    dec, z = _dec(), _z(4)
    probe_q, probe_t = torch.randn(5, DIM), torch.tensor([0, 1, 2, 3, 4])
    alive = probe_query_collapse(dec, z, probe_q, probe_t)
    with torch.no_grad():
        for block in dec.blocks:  # il contenuto non arriva piu' alle query: l'uscita dipende solo dalla query
            block.out.weight.zero_()
            block.out.bias.zero_()
    dead = probe_query_collapse(dec, z, probe_q, probe_t)
    assert alive["ratio"] > 1e-3 and dead["ratio"] < 1e-10


def test_full_chain_to_the_decoder_on_three_topologies():
    """Criterio di chiusura del passo 6 («un batch con le tre topologie insieme fa forward e backward»), con il decoder a query: identita' ->
    encoder locale -> Perceiver -> backbone -> decoder, query sui canali nascosti e sull'istante nascosto; front-end ancora sostituito da token
    casuali."""
    torch.manual_seed(0)
    montages = [emg2pose.build_montage_metadata("u1", "s", "left"), camargo.build_montage_metadata(1, "d"),
                capgmyo.build_montage_metadata(1), ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right")]
    counts = [m.n_channels for m in montages]
    dicts = [montage_to_dict(m, [True] * m.n_channels) for m in montages]
    codes = [CC.anatomy_codes(d) for d in dicts]
    packed = codes_to_tensors(CC.pack_codes(codes))
    packed["compartment_weights"] = packed["compartment_weights"].float()
    sets = sets_to_tensors(CC.pack_attention_sets([CC.attention_sets(CC.layout_from_montage(d), 4) for d in dicts]))
    ident, enc = ChannelIdentity(DIM, muscle_dropout=0.4), LocalEncoder(DIM, HEADS, 2)
    pool, bb, dec = PerceiverPooling(DIM, HEADS, K), TemporalBackbone(DIM, HEADS, 2), QueryDecoder(DIM, HEADS, 2)
    ids = ident(packed)
    visible = torch.ones(sum(counts), P, dtype=torch.bool)
    visible[3] = False  # un canale dell'anello nascosto
    visible[:, 5] = False  # uno slab
    lat = pool(enc(torch.randn(sum(counts), P, DIM) + ids[:, None, :], sets), offsets_from_counts(counts), visible)
    time_valid = torch.ones(4, P, dtype=torch.bool)
    time_valid[:, 5] = False
    z = bb(lat, time_valid)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    q_ch = torch.tensor([3, 3, 5, starts[1] + 2, starts[1] + 2, starts[2] + 40, starts[3] + 9, starts[3] + 1])  # canale (indice impacchettato)
    q_t = torch.tensor([2, 5, 5, 0, 5, 5, 5, 3])
    q_off = torch.tensor([0, 3, 5, 6, 8])
    pred = dec(z, ids[q_ch], q_t, q_off, time_valid)
    assert pred.shape == (8, DIM) and torch.isfinite(pred).all()
    pred.square().mean().backward()
    for module in (ident, enc, pool, bb, dec):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
