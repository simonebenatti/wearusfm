"""Pooling Perceiver (v10 §5, §5.1) e primo pezzo del modello in fila: identita' di canale -> encoder locale -> Perceiver, su un batch con le
tre topologie insieme. Richiede torch: si salta se non c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.ingest import camargo, capgmyo, emg2pose, ninapro_std  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model.channel_identity import ChannelIdentity  # noqa: E402
from wearusfm.model.local_encoder import LocalEncoder, sets_to_tensors  # noqa: E402
from wearusfm.model.perceiver import PerceiverPooling, offsets_from_counts  # noqa: E402

DIM, HEADS, K, P = 16, 4, 6, 5


def _model(seed=0):
    torch.manual_seed(seed)
    return PerceiverPooling(DIM, HEADS, K).eval()


def _x(c, seed=1):
    return torch.randn(c, P, DIM, generator=torch.Generator().manual_seed(seed))


def test_shape_and_order_invariance_within_a_sample():
    m, x = _model(), _x(10)
    out = m(x, offsets_from_counts([10]))
    assert out.shape == (1, P, K, DIM)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(m(x[perm], offsets_from_counts([10])), out, atol=1e-5)


def test_packed_samples_are_isolated():
    m, x = _model(), _x(30)
    off = offsets_from_counts([8, 12, 10])
    out = m(x, off)
    x2 = x.clone()
    x2[8:20] = torch.randn(12, P, DIM)
    out2 = m(x2, off)
    assert torch.allclose(out2[0], out[0]) and torch.allclose(out2[2], out[2]) and not torch.allclose(out2[1], out[1])
    assert torch.allclose(m(x[:8], offsets_from_counts([8]))[0], out[0], atol=1e-6)


def test_hidden_tokens_do_not_count_and_a_fully_hidden_instant_gives_the_learned_latents():
    m, x = _model(), _x(10)
    off = offsets_from_counts([10])
    vis = torch.ones(10, P, dtype=torch.bool)
    vis[3] = False  # un canale nascosto (masking spaziale o QC)
    vis[:, 2] = False  # un istante nascosto su tutti i canali (slab)
    out = m(x, off, vis)
    x2 = x.clone()
    x2[3] = 50 * torch.randn(P, DIM)
    x2[:, 2] = 50 * torch.randn(10, DIM)
    assert torch.allclose(m(x2, off, vis), out, atol=1e-5)
    base = m.latents + m.out.bias  # proiezione d'uscita di un contributo nullo = il suo bias: costante, nessun contenuto
    empty = base + m.mlp(m.norm_mlp(base))
    assert torch.allclose(out[0, 2], empty, atol=1e-6)


def test_bad_offsets_are_rejected_and_flops_counted():
    m = _model()
    for bad in ([0, 5, 5, 10], [1, 10], [0, 9]):
        with pytest.raises(ValueError, match="channel_offsets"):
            m(_x(10), torch.tensor(bad))
    assert PerceiverPooling.flops(100, 4, 160, 64, 384) > 0


def test_identity_local_encoder_perceiver_on_a_batch_with_three_topologies():
    """Primo pezzo del criterio di chiusura del passo 6 («un batch con le tre topologie insieme fa forward e backward»): anello (emg2pose),
    sparso (Camargo), griglia (CapgMyo), misto (NinaPro DB2), impacchettati; manca ancora il backbone temporale."""
    torch.manual_seed(0)
    montages = [m for m in (emg2pose.build_montage_metadata("u1", "s", "left"), camargo.build_montage_metadata(1, "d"),
                            capgmyo.build_montage_metadata(1), ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"))]
    dicts = [montage_to_dict(m, [True] * m.n_channels) for m in montages]
    counts = [m.n_channels for m in montages]
    codes = [CC.anatomy_codes(d) for d in dicts]
    packed_codes = {f: torch.as_tensor(np.concatenate([getattr(c, f) for c in codes]))
                    for f in ("region", "compartment_weights", "muscle", "muscle_known", "topology")}
    packed_codes["compartment_weights"] = packed_codes["compartment_weights"].float()
    sets = sets_to_tensors(CC.pack_attention_sets([CC.attention_sets(CC.layout_from_montage(d), 4) for d in dicts]))
    ident, enc, pool = ChannelIdentity(DIM, muscle_dropout=0.3), LocalEncoder(DIM, HEADS, 2), PerceiverPooling(DIM, HEADS, K)
    patches = torch.randn(sum(counts), P, DIM)  # al posto delle uscite del front-end
    tokens = patches + ident(packed_codes)[:, None, :]
    lat = pool(enc(tokens, sets), offsets_from_counts(counts))
    assert lat.shape == (4, P, K, DIM) and torch.isfinite(lat).all()
    lat.square().mean().backward()
    for module in (ident, enc, pool):
        assert all(p.grad is not None for p in module.parameters() if p.requires_grad and p is not ident.anatomy.muscle.weight)
    assert ident.anatomy.muscle.weight.grad.abs().sum() > 0  # i muscoli di Camargo e dei mirati arrivano fino al gradiente
