"""Modello intero e ciclo JEPA (v10 §6.2, §5.6): niente fuga del contenuto nascosto verso lo studente, target (a) e (b), EMA, ancore e RVQ.
Richiede torch: si salta se non c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.ingest import camargo, emg2pose  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model import anchor_targets as AT  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model.channel_identity import codes_to_tensors  # noqa: E402
from wearusfm.model.fm import FMConfig, ModelInputs, WearUsFM  # noqa: E402
from wearusfm.model.local_encoder import sets_to_tensors  # noqa: E402
from wearusfm.training.jepa import JEPAConfig, effective_rank, ema_update, hidden_queries, jepa_losses, make_teacher, train_step  # noqa: E402

CFG = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4, rvq_codes=32)
FS = [2000.0, 1000.0]


def _inputs(seed=0):
    mont = [emg2pose.build_montage_metadata("u1", "s", "left"), camargo.build_montage_metadata(1, "d")]
    dicts = [montage_to_dict(m, [True] * m.n_channels) for m in mont]
    counts = [m.n_channels for m in mont]
    codes = [CC.anatomy_codes(d) for d in dicts]
    packed = codes_to_tensors(CC.pack_codes(codes))
    sets = sets_to_tensors(CC.pack_attention_sets([CC.attention_sets(CC.layout_from_montage(d), 2) for d in dicts]))
    rng = np.random.default_rng(seed)
    signals = [torch.as_tensor(rng.normal(size=(counts[0], 1600))), torch.as_tensor(rng.normal(size=(counts[1], 800)))]  # 0,8 s: 32 patch
    return ModelInputs(signals, FS, packed, sets, counts, torch.ones(sum(counts), dtype=torch.bool))


def _visible(inp, p=32):
    vis = torch.ones(sum(inp.counts), p, dtype=torch.bool)
    vis[:, 8:16] = False  # slab sulla finestra 1 della griglia RVQ
    vis[3] = False  # un canale dell'anello nascosto
    vis[16 + 2, 20:28] = False  # un muscolo di Camargo nascosto per 200 ms
    return vis


def test_student_sees_nothing_of_the_hidden_content():
    torch.manual_seed(0)
    model = WearUsFM(CFG).eval()
    inp = _inputs()
    vis = _visible(inp)
    hidden = ~vis
    q_ch, q_t, q_off = hidden_queries(hidden, inp.counts)
    h = model.decode(model.encode(inp, vis), q_ch, q_t, q_off)
    noisy = _inputs()
    x0 = noisy.signals[0].clone()
    x0[:, 8 * 50:16 * 50] = 100 * torch.randn(16, 400)  # dentro lo slab (2 kHz: 50 campioni per patch)
    x0[3] = 100 * torch.randn(1600)  # il canale nascosto, per intero
    x1 = noisy.signals[1].clone()
    x1[:, 8 * 25:16 * 25] = 100 * torch.randn(11, 200)
    x1[2, 20 * 25:28 * 25] = 100 * torch.randn(200)
    noisy.signals = [x0, x1]
    h2 = model.decode(model.encode(noisy, vis), q_ch, q_t, q_off)
    assert torch.allclose(h2, h, atol=1e-5)  # nessuna fuga: ne' dal contesto del front-end, ne' dall'encoder locale, ne' dal Perceiver


@pytest.mark.parametrize("target", ["a", "b"])
def test_jepa_losses_both_targets_with_anchors_and_rvq(target):
    torch.manual_seed(0)
    student = WearUsFM(CFG)
    teacher = make_teacher(student)
    inp = _inputs()
    vis = _visible(inp)
    tg = [AT.anchor_targets(x.numpy(), f, lim) for x, f, lim in zip(inp.signals, FS, (850.0, 500.0))]
    rvq_on = torch.tensor([True] * 16 + [True] * 11)
    seen = []

    def codes(c, w):
        seen.append(int(c.numel()))
        return (c * 7 + w) % 32

    losses = jepa_losses(student, teacher, inp, vis, JEPAConfig(target, 0.2, 0.99), anchor_targets=tg, rvq_on=rvq_on, rvq_codes=codes)
    assert {"jepa", "log_rms", "band_shape", "log_env", "rvq", "total"} <= set(losses) and all(torch.isfinite(v) for v in losses.values())
    assert seen[0] == 27  # lo slab copre la finestra 1 di tutti i 27 canali (la seconda chiamata e' la diagnostica sul masking di canale)
    assert {"rvq_acc", "rvq_channel", "rvq_channel_acc"} <= set(losses) and not losses["rvq_channel"].requires_grad
    expected = losses["jepa"] + 0.2 * (losses["log_rms"] + losses["band_shape"] + losses["log_env"] + losses["rvq"])
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()
    assert all(p.grad is None for p in teacher.parameters())  # stop-gradient
    assert student.tokenizer.frontend.fourier_coef.grad is not None


def test_ema_and_train_step():
    torch.manual_seed(0)
    student = WearUsFM(CFG)
    teacher = make_teacher(student)
    before_t = [p.clone() for p in teacher.parameters()]
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    inp = _inputs()
    out = train_step(student, teacher, opt, inp, _visible(inp), JEPAConfig("b", 0.2, 0.9))
    assert np.isfinite(out["total"])
    for pt, b, ps in zip(teacher.parameters(), before_t, student.parameters()):
        assert torch.allclose(pt, 0.9 * b + 0.1 * ps.detach(), atol=1e-6)  # EMA esatta
    t2 = make_teacher(student)
    ema_update(t2, student, 0.0)
    assert all(torch.equal(a, b) for a, b in zip(t2.parameters(), student.parameters()))


def test_effective_rank_and_config_checks():
    assert effective_rank(torch.eye(8)) == pytest.approx(7.0, rel=0.05)  # centrata: 8 punti in 8 dimensioni hanno rango 7
    assert effective_rank(torch.randn(50, 1) @ torch.randn(1, 8)) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        JEPAConfig("c", 0.2, 0.99)
    with pytest.raises(ValueError):
        JEPAConfig("a", 0.2, 1.0)
    with pytest.raises(TypeError):
        JEPAConfig("a", 0.2)  # niente default per il momento EMA
