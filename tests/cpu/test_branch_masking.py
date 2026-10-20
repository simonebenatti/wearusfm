"""Obiettivo mascherato dei rami (`model/branch_masking.py`, protocollo `rami_pilota_v1`): token dell'encoder di scala uguali al forward,
maschere e ruoli dei frame, indipendenza esatta dello studente dai campioni nascosti (PSD e wavelet), distorsione dell'azzeramento sui frame di
contesto, riferimento del predittore costante, monitor e allarme di collasso relativi al passo 0."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from wearusfm.model import branch_masking as BM
from wearusfm.model import spectral_branches as SB

from test_spectral_branches import _inputs, _noise

FS = 2000.0


def _old_forward(layer, logp, valid):
    """Il forward di `ScaleEncoder` com'era prima di `tokens` (copiato alla lettera)."""
    if logp.shape[1] == 0:
        return logp.new_zeros(logp.shape[0], layer.output.out_features), torch.zeros(logp.shape[0], dtype=torch.bool, device=logp.device)
    use = valid & layer.observed
    x = torch.where(use, (logp - layer.input_mean) / layer.input_scale, 0.0)
    x = torch.cat((x, use.to(x.dtype)), -1)
    ok = use.any(-1)
    h = torch.where(ok[..., None], F.gelu(layer.project(x)), 0.0)
    for conv in layer.temporal:
        v = conv(h.transpose(1, 2)).transpose(1, 2)
        h = torch.where(ok[..., None], F.gelu(h + v), 0.0)
    den = ok.sum(1).clamp(min=1)[:, None].to(h.dtype)
    mean = h.sum(1) / den
    dev = torch.where(ok[..., None], h - mean[:, None, :], 0.0)
    std = (dev.square().sum(1) / den + 1e-6).sqrt()
    return layer.output(torch.cat((mean, std), -1)), ok.any(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tokens_refactor_leaves_the_forward_bit_identical(dtype):
    torch.manual_seed(0)
    layer = SB.ScaleEncoder(6, 8, 16).to(dtype)
    layer.observed.fill_(True)
    layer.observed[2] = False
    logp = torch.randn(5, 12, 6, dtype=dtype)
    valid = torch.rand(5, 12, 6) > 0.3
    valid[1] = False  # un canale senza frame validi
    new, ok_new = layer(logp, valid)
    old, ok_old = _old_forward(layer, logp, valid)
    assert torch.equal(new, old) and torch.equal(ok_new, ok_old)
    h, ok = layer.tokens(logp, valid)
    assert h.shape == (5, 12, 8) and torch.equal(h[~ok], torch.zeros_like(h[~ok]))
    empty = layer.tokens(logp[:, :0], valid[:, :0])
    assert empty[0].shape == (5, 0, 8) and empty[1].shape == (5, 0)


def test_spans_follow_the_frozen_rules():
    rng = np.random.default_rng(0)
    t, guard = int(4.0 * FS), (100, 300)
    for _ in range(50):
        sp = BM.sample_spans(t, FS, guard, rng)
        s, m = int(0.2 * FS), int(0.1 * FS)
        usable = t - guard[0] - guard[1] - 2 * m
        assert len(sp) == max(1, int(np.floor(0.25 * usable / s + 0.5)))
        assert all(b - a == s for a, b in sp) and all(sp[i][1] <= sp[i + 1][0] for i in range(len(sp) - 1))
        assert sp[0][0] >= guard[0] + m and sp[-1][1] <= t - guard[1] - m
    assert BM.sample_spans(int(0.35 * FS), FS, (0, 0), rng) == []  # parte utile (150 ms) piu' corta di un tratto
    assert len(BM.sample_spans(int(0.45 * FS), FS, (0, 0), rng)) == 1  # almeno uno se entra
    inp = _inputs([_noise(2, 3.0, FS, 1), _noise(3, 4.0, FS, 2)], [FS, FS], [np.full(2, 450.0), np.full(3, 450.0)])
    assert BM.batch_spans(inp, 0, 5) == BM.batch_spans(inp, 0, 5) != BM.batch_spans(inp, 0, 6)


def test_frame_roles_use_the_full_support_per_band():
    spans = [(4000, 4400)]
    radius = np.array([0, 50, 400])
    ctx, tgt = BM.frame_roles(int(4.0 * FS), FS, 50.0, 25.0, radius, spans)
    n, starts = SB.frame_starts(int(4.0 * FS), FS, 50.0, 25.0)
    centers = starts + n / 2
    assert np.array_equal(tgt, (centers >= 4000) & (centers < 4400))
    for b, r in enumerate(radius):
        touches = (starts - r < 4400) & (starts + n + r > 4000)
        assert np.array_equal(ctx[:, b], ~touches & ~tgt)
    assert not (ctx & tgt[:, None]).any() and ctx[:, 2].sum() < ctx[:, 0].sum()  # la banda col raggio lungo perde piu' contesto


def _branches(kinds=("psd", "wavelet")):
    torch.manual_seed(0)
    sbs = SB.SpectralBranches(kinds, out_dim=16, hidden=8)
    sigs = [_noise(3, 4.0, FS, 10), _noise(2, 3.0, FS, 11)]
    sbs.calibrate([_inputs(sigs, [FS, FS], [np.full(3, 450.0), np.full(2, 450.0)])])
    return sbs


@pytest.mark.parametrize("kind", SB.BRANCH_KINDS)
def test_student_side_never_depends_on_the_hidden_samples(kind):
    """Il punto del protocollo: cambiando a piacere il segnale nei tratti nascosti, spettri, token di contesto e predizioni dello studente
    restano identici bit per bit (anche per le wavelet, che filtrano in frequenza); cambiandolo fuori, cambiano."""
    sbs = _branches((kind,))
    sb = sbs.branches[kind]
    predictors = BM.make_predictors((kind,), hidden=8, seed=0)
    sigs = [_noise(3, 4.0, FS, 20), _noise(2, 3.0, FS, 21)]
    inp = _inputs(sigs, [FS, FS], [np.full(3, 450.0), np.full(2, 450.0)])
    spans = BM.batch_spans(inp, 0, 1)
    assert all(spans)

    def student(i):
        side = BM.student_context(sb, i, spans)
        preds = {}
        for name, st in side.items():
            rows = st["ok"].any(1)
            preds[name] = predictors[f"{kind}_{name}"](st["h"][rows], st["ok"][rows], st["times"][rows])
        return side, preds

    base, pred = student(inp)
    hidden = [x.copy() for x in sigs]
    outside = [x.copy() for x in sigs]
    for x, sp in zip(hidden, spans):
        for a, b in sp:
            x[:, a:b] = 1000.0 * np.random.default_rng(a).standard_normal((x.shape[0], b - a))  # un artefatto enorme nel tratto nascosto
    for x, sp in zip(outside, spans):
        x[:, sp[0][0] - 10] += 5.0  # un campione appena fuori dal primo tratto
    same, pred_same = student(_inputs(hidden, [FS, FS], [np.full(3, 450.0), np.full(2, 450.0)]))
    diff, pred_diff = student(_inputs(outside, [FS, FS], [np.full(3, 450.0), np.full(2, 450.0)]))
    for name in base:
        for key in ("logp", "valid", "ctx", "tgt", "h", "ok"):
            assert torch.equal(base[name][key], same[name][key]), (name, key)
        assert torch.equal(pred[name], pred_same[name])
    assert any(not torch.equal(base[n]["logp"], diff[n]["logp"]) for n in base)


def test_zeroing_distorts_the_context_frames_only_within_the_declared_bound():
    """Sui frame di contesto la regola del supporto tiene lontano l'azzeramento: la differenza fra le log-potenze dello studente (segnale
    azzerato) e del teacher (segnale completo) resta sotto 1e-2 (limite dichiarato nel protocollo); per la PSD e' zero."""
    sbs = _branches()
    sigs = [_noise(4, 4.0, FS, 30)]
    inp = _inputs(sigs, [FS], [np.full(4, 450.0)])
    spans = BM.batch_spans(inp, 3, 7)
    for kind, sb in sbs.branches.items():
        side = BM.student_context(sb, inp, spans)
        full = sb.spectra(inp)
        for name, st in side.items():
            m = st["valid"] & st["ctx"]
            d = (st["logp"] - full[name][0]).abs()[m]
            assert d.numel() > 0
            if kind == "psd":
                assert float(d.max()) == 0.0
            else:
                assert float(d.max()) < 1e-2, (kind, name, float(d.max()))


def test_objective_reference_monitors_and_rows_without_context():
    sbs = _branches()
    student = SimpleNamespace(spectral_branches=sbs)
    teacher = SimpleNamespace(spectral_branches=copy.deepcopy(sbs))
    predictors = BM.make_predictors(("psd", "wavelet"), hidden=8, seed=0)
    inp = _inputs([_noise(3, 4.0, FS, 40), _noise(2, 4.0, FS, 41)], [FS, FS], [np.full(3, 450.0), np.full(2, 450.0)])
    loss, m = BM.masked_branch_objective(student, teacher, predictors, inp, BM.batch_spans(inp, 0, 1))
    assert torch.isfinite(loss) and loss.requires_grad
    for key in ("psd_fast", "psd_slow", "wavelet_fast"):
        assert m[f"masked_{key}_targets"] > 0 and 0 < m[f"masked_{key}_normalized"] < 10
        assert all(np.isfinite(m[f"masked_{key}_{mon}"]) for mon in BM.MONITORS)
    loss.backward()
    grads = {n for n, p in list(sbs.named_parameters()) + list(predictors.named_parameters()) if p.grad is not None}
    assert any(".project." in n for n in grads) and not any(".output." in n or "combine" in n for n in grads)  # strati finali non usati
    short = _inputs([_noise(2, 0.3, FS, 42)], [FS], [np.full(2, 450.0)])  # nessun tratto: nessun bersaglio -> errore esplicito
    with pytest.raises(ValueError, match="nessun frame bersaglio"):
        BM.masked_branch_objective(student, teacher, predictors, short, BM.batch_spans(short, 0, 1))
    with pytest.raises(ValueError, match="senza frame di contesto"):
        predictors["psd_fast"](torch.zeros(1, 3, 8), torch.zeros(1, 3, dtype=torch.bool), torch.zeros(1, 3))


def test_collapse_stats_see_a_constant_teacher():
    rng = torch.Generator().manual_seed(0)
    raw = torch.randn(200, 8, generator=rng)
    s = BM.collapse_stats(BM.target_tokens(raw), raw)
    assert s["diversity"] > 0.5 and s["erank"] > 4 and s["variance"] > 0.5 and 0 < s["baseline"] <= 1
    const = torch.randn(1, 8, generator=rng).repeat(200, 1)
    c = BM.collapse_stats(BM.target_tokens(const), const)
    assert c["diversity"] == pytest.approx(0, abs=1e-6) and c["erank"] == 0.0 and c["variance"] == pytest.approx(0, abs=1e-9)
    # token quasi costanti con coordinate quasi uguali: la LayerNorm per token amplifica il residuo fino a farlo sembrare diverso (diversita'
    # e rango alti), la varianza assoluta lo vede (review del critico del 20/10)
    # (residuo sopra la sqrt(eps) della LayerNorm, ~3e-3: sotto, i target vanno a zero e la diversita' stessa lo vede)
    tiny = 3.0 * torch.ones(200, 8) + 1e-2 * torch.randn(200, 8, generator=rng)
    t = BM.collapse_stats(BM.target_tokens(tiny), tiny)
    assert t["diversity"] > 0.5 and t["erank"] > 4 and t["variance"] < 1e-3 < s["variance"]


def test_collapse_alarm_relative_to_the_first_updates():
    alarm = BM.CollapseAlarm()
    good = {"masked_psd_fast_diversity": 0.8, "masked_psd_fast_erank": 6.0, "masked_psd_fast_variance": 2.0}
    for step in range(1, 11):
        assert alarm.update(step, good) is None
    assert alarm.reference() == pytest.approx(good)
    bad = {**good, "masked_psd_fast_variance": 0.01}  # sotto 0,05 x 2,0
    assert alarm.update(50, bad) is None  # prima di 100 aggiornamenti: nessun controllo
    assert alarm.update(110, bad) is None and alarm.update(120, bad) is None
    assert alarm.update(125, bad) is None  # fuori cadenza (ogni 10)
    assert "variance" in alarm.update(130, bad)
    fresh = BM.CollapseAlarm()
    for step in range(1, 11):
        fresh.update(step, good)
    fresh.update(110, bad)
    fresh.update(120, bad)
    assert fresh.update(130, good) is None and fresh.update(140, bad) is None  # una ripresa azzera il conteggio
