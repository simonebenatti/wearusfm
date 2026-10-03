"""Identita' di canale (v10 §5.2). Richiede torch: si salta se non c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.ingest import camargo, emg2pose, ninapro_std  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.metadata import taxonomy as T  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model.channel_identity import AnatomicalEmbedding, ChannelIdentity, codes_to_tensors  # noqa: E402


def _codes(m):
    return codes_to_tensors(CC.anatomy_codes(montage_to_dict(m, [True] * m.n_channels)))


def test_hierarchical_sum_unk_and_soft_weights():
    torch.manual_seed(0)
    emb = AnatomicalEmbedding(8, muscle_dropout=0.0).eval()
    c = _codes(ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"))
    out = emb(c["region"], c["compartment_weights"], c["muscle"], c["muscle_known"])
    unk = emb.region.weight[0] + emb.compartment.weight[0] + emb.muscle.weight[0]
    assert torch.allclose(out[:8], unk.expand(8, -1))  # anello senza orientamento: UNK appreso a ogni livello, non zero
    i = CC.MUSCLE_KEYS.index("FDS")
    expect = emb.region.weight[CC.REGION_KEYS.index("forearm_proximal")] + emb.compartment.weight[CC.COMPARTMENT_KEYS.index("flexor_pronator_ulnar")] \
        + emb.muscle.weight[i]
    assert torch.allclose(out[8], expect)
    w = torch.zeros(1, len(CC.COMPARTMENT_KEYS))
    w[0, 1], w[0, 2] = 0.25, 0.75  # pesi soft: somma pesata degli embedding di compartimento
    soft = emb(torch.tensor([1]), w, torch.tensor([0]), torch.tensor([False]))
    assert torch.allclose(soft[0], emb.region.weight[1] + 0.25 * emb.compartment.weight[1] + 0.75 * emb.compartment.weight[2] + emb.muscle.weight[0])


def test_same_compartment_from_different_datasets_shares_the_vector():
    emb = AnatomicalEmbedding(4, muscle_dropout=0.0).eval()
    m = montage_to_dict(emg2pose.build_montage_metadata("u1", "s", "left"), [True] * 16)
    a = m["groups"][0]["channels"]
    a[0]["anatomical_identity"].update(region="forearm_proximal", compartment="dorsal_extensors", precision="compartment")
    a[1]["anatomical_identity"].update(region="forearm_proximal", compartment="dorsal_extensors", precision="compartment")
    c = codes_to_tensors(CC.anatomy_codes(m))
    out = emb(c["region"], c["compartment_weights"], c["muscle"], c["muscle_known"])
    assert torch.equal(out[0], out[1])


def test_muscle_dropout_only_in_training_and_only_where_known():
    torch.manual_seed(0)
    emb = AnatomicalEmbedding(4, muscle_dropout=0.5)
    c = _codes(camargo.build_montage_metadata(1, "d"))  # 11 muscoli noti
    args = (c["region"], c["compartment_weights"], c["muscle"], c["muscle_known"])
    full = emb.eval()(*args)
    assert torch.equal(full, emb(*args))  # in valutazione niente dropout
    g = torch.Generator().manual_seed(1)
    train = emb.train()(*args, generator=g)
    dropped = ~torch.isclose(train, full).all(dim=1)
    assert 0 < int(dropped.sum()) < 11  # con p = 0,5 qualcuno si', qualcuno no
    d = dropped.nonzero()[0, 0]
    expect = full[d] - emb.muscle.weight[c["muscle"][d]] + emb.muscle.weight[0]  # il muscolo diventa UNK, il resto non cambia
    assert torch.allclose(train[d], expect, atol=1e-6)
    ring = _codes(emg2pose.build_montage_metadata("u1", "s", "left"))  # nessun muscolo noto: il dropout non tocca nulla
    r_args = (ring["region"], ring["compartment_weights"], ring["muscle"], ring["muscle_known"])
    assert torch.equal(emb.train()(*r_args, generator=g), emb.eval()(*r_args))


def test_muscle_dropout_must_be_chosen():
    with pytest.raises(TypeError):
        AnatomicalEmbedding(4)  # nessun valore di default: v10 dice «da fissare»
    with pytest.raises(ValueError):
        AnatomicalEmbedding(4, muscle_dropout=1.0)


def test_channel_identity_concatenates_and_projects_with_gradients():
    torch.manual_seed(0)
    model = ChannelIdentity(16, muscle_dropout=0.3)
    c = _codes(ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"))
    out = model(c)
    assert out.shape == (12, 16) and torch.isfinite(out).all()
    out.sum().backward()
    assert model.anatomy.muscle.weight.grad is not None and model.topology.weight.grad is not None
    assert not torch.allclose(model.eval()(c)[0], model.eval()(c)[8])  # anello ignoto e muscolo mirato: identita' diverse
    assert len(CC.REGION_KEYS) == len(T.REGIONS) + 1 and np.unique(c["topology"].numpy()).tolist() == [0, 2]


def test_ring_channels_and_the_two_hands_have_distinct_identities():
    """Review del 03/10: su emg2qwerty 32 canali avevano 1 identita' (anatomia ignota, nessuna posizione): il decoder a query dava la stessa
    predizione per tutti i canali allo stesso istante e non distingueva la mano sinistra dalla destra."""
    from wearusfm.ingest import emg2qwerty

    torch.manual_seed(0)
    ident = ChannelIdentity(16, muscle_dropout=0.0).eval()
    c = _codes(emg2qwerty.build_montage_metadata("u1", "s"))
    out = ident(c)
    d = torch.cdist(out, out)
    assert out.shape[0] == 32 and bool((d + torch.eye(32) * 1e9).min() > 1e-3)  # 32 identita' distinte
    assert c["side"][:16].tolist() == [1] * 16 and c["side"][16:].tolist() == [2] * 16 and c["group"].tolist() == [0] * 16 + [1] * 16
    pos = CC.anatomy_codes(montage_to_dict(emg2qwerty.build_montage_metadata("u1", "s"), [True] * 32)).sensor_pos
    assert np.allclose(pos[1, :4], np.cos(np.arange(1, 5) * 2 * np.pi / 16))  # anello: cos/sin(m * angolo)
    cm = camargo.build_montage_metadata(1, "d")
    cam = CC.anatomy_codes(montage_to_dict(cm, [True] * cm.n_channels))
    assert not cam.sensor_pos.any()  # sparsi: nessuna posizione, l'identita' e' anatomica
