"""Front-end collegato al modello e teste delle ancore; test d'insieme dal segnale alle perdite delle ancore. Richiede torch: si salta se non
c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.ingest import emg2pose, ninapro_db5  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model import anchor_targets as AT  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model.anchors import AnchorHeads, RVQHead, anchor_losses, rvq_loss  # noqa: E402
from wearusfm.model.backbone import TemporalBackbone  # noqa: E402
from wearusfm.model.channel_identity import ChannelIdentity, codes_to_tensors  # noqa: E402
from wearusfm.model.local_encoder import LocalEncoder, sets_to_tensors  # noqa: E402
from wearusfm.model.patch_tokens import PatchTokenizer  # noqa: E402
from wearusfm.model.perceiver import PerceiverPooling, offsets_from_counts  # noqa: E402
from wearusfm.model.query_decoder import QueryDecoder  # noqa: E402

DIM, HEADS, K = 16, 4, 6


def test_tokenizer_handles_two_rates_and_shorter_windows():
    torch.manual_seed(0)
    tok = PatchTokenizer(DIM)
    a = torch.randn(3, 2000)  # 1 s a 2 kHz -> 40 patch
    b = torch.randn(2, 100)  # 0,5 s a 200 Hz (Myo) -> 20 patch
    tokens, patch_valid, time_valid = tok([a, b], [2000.0, 200.0])
    assert tokens.shape == (5, 40, DIM) and torch.isfinite(tokens).all()
    assert patch_valid[:3].all() and patch_valid[3:, :20].all() and not patch_valid[3:, 20:].any()
    assert time_valid.tolist() == [[True] * 40, [True] * 20 + [False] * 20]
    assert (tokens[3:, 20:] == 0).all()  # padding
    support = math.ceil(0.025 * 2000 + 1e-9) + 2 * math.ceil(0.1 * 2000 - 1e-9)  # 51 + 400: gli stessi campioni che il front-end raccoglie
    assert tok.flops(3, 40, 2000.0) == 2 * 3 * 40 * (support * tok.frontend.d_out + tok.frontend.d_out * DIM)


def test_anchor_losses_ignore_what_has_no_target():
    heads = AnchorHeads(DIM)
    h = torch.randn(4, DIM, requires_grad=True)
    pred = heads(h)
    tgt = {"log_rms": torch.zeros(4), "log_env": torch.zeros(4), "band_shape": torch.zeros(4, 5),
           "rms_valid": torch.tensor([True, True, False, False]), "env_valid": torch.zeros(4, dtype=torch.bool),
           "spec_valid": torch.tensor([True, False, False, False]), "band_available": torch.tensor([[True, True, True, False, False]] * 4)}
    loss = anchor_losses(pred, tgt)
    assert loss["log_env"].item() == 0.0
    assert loss["log_rms"].item() == pytest.approx((pred["log_rms"][:2] ** 2).mean().item(), rel=1e-6)
    assert loss["band_shape"].item() == pytest.approx((pred["band_shape"][0, :3] ** 2).mean().item(), rel=1e-6)
    sum(loss.values()).backward()
    assert h.grad[3].abs().sum() == 0  # la query senza nessun target non riceve gradiente
    rvq = RVQHead(DIM, n_codes=32)
    logits = rvq(torch.randn(2, 8, DIM))
    assert logits.shape == (2, 32) and rvq_loss(logits, torch.tensor([1, 5])) > 0
    assert rvq_loss(logits[:0], torch.tensor([], dtype=torch.long)) == 0
    with pytest.raises(ValueError):
        rvq(torch.randn(2, 7, DIM))


def test_from_signal_to_anchor_losses_on_ring_and_myo():
    """Dal segnale vero al gradiente: due montaggi a frequenze diverse (bracciale Meta a 2 kHz, Myo di DB5 a 200 Hz), front-end vero, identita',
    encoder locale, Perceiver, backbone, decoder, teste delle ancore; uno slab sulla griglia del tokenizer."""
    torch.manual_seed(0)
    mont = [emg2pose.build_montage_metadata("u1", "s", "left"), ninapro_db5.build_montage_metadata(1, "right")]
    dicts = [montage_to_dict(m, [True] * m.n_channels) for m in mont]
    counts = [m.n_channels for m in mont]
    fs = [2000.0, 200.0]
    rng = np.random.default_rng(0)
    signals = [rng.normal(size=(counts[0], 1600)), rng.normal(size=(counts[1], 160))]  # 0,8 s ciascuno: 32 patch
    tok = PatchTokenizer(DIM)
    tokens, patch_valid, time_valid = tok([torch.as_tensor(s) for s in signals], fs)
    p = tokens.shape[1]
    codes = [CC.anatomy_codes(d) for d in dicts]
    packed = codes_to_tensors(CC.AnatomyCodes(*[np.concatenate([getattr(c, f) for c in codes]) for f in
                                                 ("region", "compartment_weights", "muscle", "muscle_known", "topology")]))
    ident, enc = ChannelIdentity(DIM, muscle_dropout=0.4), LocalEncoder(DIM, HEADS, 1)
    pool, bb, dec, heads = PerceiverPooling(DIM, HEADS, K), TemporalBackbone(DIM, HEADS, 1), QueryDecoder(DIM, HEADS, 1), AnchorHeads(DIM)
    ids = ident(packed)
    visible = patch_valid.clone()
    visible[:, 8:16] = False  # slab sulla finestra 1 del tokenizer
    sets = sets_to_tensors(CC.pack_attention_sets([CC.attention_sets(CC.layout_from_montage(d), 2) for d in dicts]))
    tv = time_valid & visible[[0, counts[0]]]  # istanti nascosti fuori dalle chiavi temporali
    z = bb(pool(enc(tokens + ids[:, None, :], sets), offsets_from_counts(counts), visible), tv)
    q_ch = torch.arange(sum(counts)).repeat_interleave(8)  # ogni canale, le 8 patch nascoste
    q_t = torch.arange(8, 16).repeat(sum(counts))
    h = dec(z, ids[q_ch], q_t, offsets_from_counts([c * 8 for c in counts]), tv)
    tgts = [AT.anchor_targets(s, f, lim) for s, f, lim in zip(signals, fs, (850.0, 100.0))]  # banda del bracciale Meta 20-850 Hz; Myo 100 Hz
    def take(name):  # noqa: E306
        return torch.as_tensor(np.concatenate([getattr(t, name)[:, 8:16].reshape(-1, *getattr(t, name).shape[2:]) for t in tgts]))
    target = {n: take(n) for n in ("log_rms", "rms_valid", "band_shape", "spec_valid", "log_env", "env_valid")}
    target["band_available"] = torch.as_tensor(np.concatenate([np.repeat(t.band_available, 8, axis=0) for t in tgts]))
    target = {k: (v.float() if v.dtype == torch.float64 else v) for k, v in target.items()}
    losses = anchor_losses(heads(h), target)
    assert all(torch.isfinite(v) for v in losses.values())
    assert target["band_available"][-1].tolist() == [True, True, False, False, False]  # Myo: bande oltre 100 Hz senza target
    sum(losses.values()).backward()
    assert tok.proj.weight.grad.abs().sum() > 0 and tok.frontend.fourier_coef.grad is not None  # il gradiente arriva fino al front-end
    win = AT.rvq_target_windows(visible.numpy(), patch_valid.numpy(), np.array([True] * counts[0] + [False] * counts[1]))
    assert win[:counts[0], 1].all() and not win[counts[0]:].any()  # DB5 ha l'ancora RVQ spenta (D5b)
