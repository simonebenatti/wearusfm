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


def _heterogeneous_targets(inp):
    targets = [AT.anchor_targets(x.numpy(), f, lim, multiscale=True)
               for x, f, lim in zip(inp.signals, inp.fs, (850.0, 100.0))]
    targets[0].rms_valid[:] = False  # one shard has no targets for this component
    targets[1].band_available[:, -2:] = False
    return targets


@pytest.mark.parametrize("per_window", [False, True])
@pytest.mark.parametrize("target", ["a", "b"])
@pytest.mark.parametrize("rvq_selection", ["slab", "channel"])
def test_additive_terms_match_legacy_and_precounts(per_window, target, rvq_selection):
    from dataclasses import replace
    from wearusfm.training.jepa import jepa_loss_terms, loss_counts
    torch.manual_seed(7)
    model_cfg = replace(CFG, multiscale_anchor=True)
    student = WearUsFM(model_cfg).eval()
    teacher = make_teacher(student)
    inp = _inputs()
    inp.signals[1] = inp.signals[1][:, :300]  # 12 real patches, 32 packed: slow targets absent
    inp.qc_valid[3] = False
    visible = _visible(inp)
    tg = _heterogeneous_targets(inp)
    rvq_on = torch.ones(sum(inp.counts), dtype=torch.bool)
    codes = lambda ch, w: torch.where(ch % 4 == 0, -1, (ch + w) % 32)
    cfg = JEPAConfig(target, .2, .99, loss_per_window=per_window, rvq_targets=rvq_selection)
    kwargs = dict(anchor_targets=tg, rvq_on=rvq_on, rvq_codes=codes, kinds=(~visible).long())
    rng = torch.get_rng_state().clone()
    counts = loss_counts(inp, visible, cfg, tg, rvq_on, codes, model_cfg=model_cfg)
    assert torch.equal(rng, torch.get_rng_state())
    old = jepa_losses(student, teacher, inp, visible, cfg, **kwargs)
    terms = jepa_loss_terms(student, teacher, inp, visible, cfg, **kwargs)
    assert set(counts) == set(terms.num) == set(terms.den) == set(terms.weight)
    for k in counts:
        assert counts[k].dtype == terms.den[k].dtype == torch.int64
        assert torch.equal(counts[k], terms.den[k]), k
        torch.testing.assert_close(terms.num[k] / terms.den[k].clamp(min=1), old[k], rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(terms.backward_loss(), old['total'], rtol=2e-6, atol=1e-7)
    assert any(v == 0 for v in terms.log_den.values())  # fixed keys for locally absent diagnostic selections
    assert all(not v.requires_grad for v in [*terms.den.values(), *terms.log_num.values(), *terms.log_den.values(), *terms.log_sum.values()])
    terms.backward_loss().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def _regions_of(visible, inp, per_token: bool):
    """Regioni della maschera di `_visible`: per token (ogni token nascosto la sua) oppure le quattro maschere vere (slab e canale sul primo
    campione, slab e tubo sul secondo; il canale 3 dentro lo slab resta dello slab, come nel generatore)."""
    hidden = ~visible
    region = torch.zeros_like(visible, dtype=torch.int16)
    if per_token:
        region[hidden] = torch.arange(1, int(hidden.sum()) + 1, dtype=torch.int16)
        return region
    c0 = inp.counts[0]
    region[:c0, 8:16] = 1
    region[3][region[3] == 0] = 2
    region[c0:, 8:16] = 1
    region[c0 + 2, 20:28] = 2
    assert torch.equal(region > 0, hidden)
    return region


@pytest.mark.parametrize("per_window", [False, True])
@pytest.mark.parametrize("target", ["a", "b"])
def test_region_target_reduces_to_the_token_target_on_single_token_regions_and_keeps_the_loss_contract(per_window, target):
    """R2: con una regione per token la perdita per regione e' quella per token (stesse medie, in float32); con le regioni vere i numeratori e
    i conteggi del contratto coincidono con le medie e con `loss_counts`, il gradiente c'e', il default e' intoccato."""
    from dataclasses import replace
    from wearusfm.training.jepa import jepa_loss_terms, loss_counts
    torch.manual_seed(3)
    model_cfg = replace(CFG, multiscale_anchor=True)
    student = WearUsFM(model_cfg).eval()
    teacher = make_teacher(student)
    inp = _inputs()
    visible = _visible(inp)
    tg = _heterogeneous_targets(inp)
    tok = JEPAConfig(target, .2, .99, loss_per_window=per_window)
    reg = replace(tok, region_target=True)
    kinds = (~visible).long()
    base = jepa_losses(student, teacher, inp, visible, tok, anchor_targets=tg, kinds=kinds)
    same = jepa_losses(student, teacher, inp, visible, tok, anchor_targets=tg, kinds=kinds, regions=_regions_of(visible, inp, True))
    assert all(torch.equal(base[k], same[k]) for k in base)  # senza region_target le regioni si ignorano: bit per bit
    one = jepa_losses(student, teacher, inp, visible, reg, anchor_targets=tg, kinds=kinds, regions=_regions_of(visible, inp, True))
    torch.testing.assert_close(one["jepa"], base["jepa"], rtol=1e-5, atol=1e-6)
    assert one["jepa_regions"] == int((~visible).sum()) and one["jepa_tokens_per_region"] == 1.0
    for k in ("log_rms", "band_shape", "log_env"):  # le ancore restano per token
        assert torch.equal(one[k], base[k])
    regions = _regions_of(visible, inp, False)
    kw = dict(anchor_targets=tg, kinds=kinds, regions=regions)
    four = jepa_losses(student, teacher, inp, visible, reg, **kw)
    assert four["jepa_regions"] == 4 and four["jepa"] != base["jepa"] and torch.isfinite(four["jepa"])
    assert four["jepa_tokens_per_region"] == pytest.approx(int((~visible).sum()) / 4)
    terms = jepa_loss_terms(student, teacher, inp, visible, reg, **kw)
    counts = loss_counts(inp, visible, reg, tg, model_cfg=model_cfg, regions=regions)
    assert set(counts) == set(terms.num) == set(terms.den)
    for k in counts:
        assert torch.equal(counts[k], terms.den[k]), k
        torch.testing.assert_close(terms.num[k] / terms.den[k].clamp(min=1), four[k], rtol=2e-6, atol=1e-7)
    n_hidden = int((~visible).sum())
    assert int(terms.den["jepa"]) == (2 if per_window else n_hidden * CFG.dim)  # ogni token porta l'errore della sua regione: conteggi per token
    assert int(terms.log_sum["jepa_regions"]) == 4 and int(terms.log_sum["jepa_hidden_tokens"]) == n_hidden
    assert torch.equal(loss_counts(inp, visible, reg, tg, model_cfg=model_cfg)["jepa"], terms.den["jepa"])  # senza regioni: stessi conteggi
    torch.testing.assert_close(terms.backward_loss(), four["total"], rtol=2e-6, atol=1e-7)
    # le quote di D10 restano: la perdita e' la media per token dell'errore della regione di ogni token
    from wearusfm.training.jepa import hidden_queries, region_means
    enc = student.encode(inp, visible)
    q_ch, q_t, q_off = hidden_queries(~enc.visible & enc.patch_valid & inp.qc_valid[:, None], inp.counts)
    with torch.no_grad():
        hh = student.decode(enc, q_ch, q_t, q_off)
        enc_t = teacher.encode(inp, None)
        rr = enc_t.local[q_ch, q_t] if target == "a" else teacher.decode(enc_t, q_ch, q_t, q_off)
        h_r, t_r, inv, _ = region_means(hh, rr, regions, visible.shape, q_ch, q_t, q_off)
        per_token = (h_r - t_r).pow(2).mean(-1)[inv]
        if not per_window:
            torch.testing.assert_close(four["jepa"], per_token.mean(), rtol=1e-5, atol=1e-7)
    terms.backward_loss().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.decoder.parameters())
    # il log per tipo di maschera esiste anche per regione (ogni token porta l'errore della sua regione)
    assert "jepa_corta" in four and torch.isfinite(four["jepa_corta"])
    # senza regioni, o con un token nascosto senza regione: errore, non un numero
    with pytest.raises(ValueError, match="regioni"):
        jepa_losses(student, teacher, inp, visible, reg, anchor_targets=tg)
    broken = regions.clone()
    broken[3, 0] = 0
    with pytest.raises(ValueError, match="senza regione"):
        jepa_losses(student, teacher, inp, visible, reg, anchor_targets=tg, regions=broken)


def test_loss_counts_apply_the_core_radius_of_a_longer_kernel_window():
    """R5: con `window_ms` > patch la perdita conta solo i token interamente nascosti; `loss_counts` deve contare gli stessi (prima del 24/10 non
    lo faceva e il contratto delle loss avrebbe fermato il run al primo passo). Con ancore, un canale scartato dal QC e un campione corto
    (12 patch vere su 32 impacchettate: oltre il segnale conta come nascosto)."""
    from dataclasses import replace
    from wearusfm.training.jepa import jepa_loss_terms, loss_counts
    torch.manual_seed(5)
    model_cfg = replace(CFG, window_ms=150.0, multiscale_anchor=True)
    student = WearUsFM(model_cfg).eval()
    teacher = make_teacher(student)
    inp = _inputs()
    inp.signals[1] = inp.signals[1][:, :300]
    inp.qc_valid[3] = False
    tg = _heterogeneous_targets(inp)
    visible = torch.ones(sum(inp.counts), 32, dtype=torch.bool)
    visible[:, 4:24] = False  # 20 patch nascoste su tutti i canali: interamente nascoste 7..20 (3 per lato); sul campione corto 7..11
    for per_window in (False, True):
        cfg = JEPAConfig("b", .2, .99, loss_per_window=per_window)
        terms = jepa_loss_terms(student, teacher, inp, visible, cfg, anchor_targets=tg)
        counts = loss_counts(inp, visible, cfg, tg, model_cfg=model_cfg)
        assert set(counts) == set(terms.den) and all(torch.equal(counts[k], terms.den[k]) for k in counts)
        if not per_window:
            assert int(counts["jepa"]) == (15 * 14 + 11 * 5) * CFG.dim < sum(inp.counts) * 20 * CFG.dim


def _one_sample(inp, s):
    a, b = sum(inp.counts[:s]), sum(inp.counts[:s+1])
    sets = {k: v[a:b].clone() for k, v in inp.sets.items()}
    sets['index'] = torch.where(sets['index'] >= 0, sets['index'] - a, sets['index'])
    return ModelInputs([inp.signals[s]], [inp.fs[s]], {k: v[a:b].clone() for k, v in inp.codes.items()},
                       sets, [inp.counts[s]], inp.qc_valid[a:b].clone())


@pytest.mark.parametrize("per_window", [False, True])
@pytest.mark.parametrize("region_target", [False, True])
def test_global_batch_equals_microbatches_loss_gradients_optimizer_and_ema(per_window, region_target):
    import copy
    from dataclasses import replace
    from wearusfm.training.jepa import jepa_loss_terms, loss_counts
    torch.manual_seed(8)
    model_cfg = replace(CFG, rvq_codes=None, multiscale_anchor=True, muscle_dropout=0.)
    full = WearUsFM(model_cfg).double().eval()
    split = copy.deepcopy(full)
    tf, ts = make_teacher(full), make_teacher(split)
    inp, visible = _inputs(), _visible(_inputs())
    inp.signals[1] = inp.signals[1][:, :300]
    tg = _heterogeneous_targets(inp)
    cfg = JEPAConfig('b', .2, .99, loss_per_window=per_window, region_target=region_target)
    regions = _regions_of(visible, inp, False) if region_target else None  # R2: le regioni seguono i micro-batch come le maschere
    whole = jepa_loss_terms(full, tf, inp, visible, cfg, tg, regions=regions)
    pieces = []
    a = 0
    for s, c in enumerate(inp.counts):
        one = _one_sample(inp, s)
        n = split.n_patches(one)
        v = visible[a:a+c, :n]
        piece = jepa_loss_terms(split, ts, one, v, cfg, [tg[s]], regions=regions[a:a+c, :n] if region_target else None)
        precount = loss_counts(one, v, cfg, [tg[s]], model_cfg=model_cfg)
        assert all(torch.equal(piece.den[k], precount[k]) for k in piece.den)
        pieces.append(piece)
        a += c
    assert pieces[0].den['log_rms'] == 0 and pieces[1].den['log_rms'] > 0
    global_den = {k: sum(p.den[k] for p in pieces) for k in whole.den}
    assert all(torch.equal(global_den[k], whole.den[k]) for k in global_den)
    # Simulate DDP's average of R=2 ranks; equivalent to sequential accumulation over two micro-batches without dividing by k again.
    part_losses = [p.backward_loss(global_den, world_size=2) / 2 for p in pieces]
    torch.testing.assert_close(sum(part_losses), whole.backward_loss(), rtol=1e-10, atol=1e-10)
    whole.backward_loss().backward()
    for loss in part_losses:
        loss.backward()
    for (name, p), (_, q) in zip(full.named_parameters(), split.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-10, msg=name)
    for model, teacher in ((full, tf), (split, ts)):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        ema_update(teacher, model, .99)
    for a, b in ((full, split), (tf, ts)):
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[k], rtol=1e-10, atol=1e-10, msg=k)


def test_empty_local_hidden_selection_is_a_zero_term_not_a_legacy_nan():
    from wearusfm.training.jepa import jepa_loss_terms, loss_counts
    student = WearUsFM(CFG).eval()
    teacher = make_teacher(student)
    inp = _inputs()
    visible = torch.ones(sum(inp.counts), 32, dtype=torch.bool)
    cfg = JEPAConfig('b', .2, .99, loss_per_window=True)
    with pytest.raises(ValueError, match='nessun token nascosto'):
        jepa_losses(student, teacher, inp, visible, cfg)
    terms = jepa_loss_terms(student, teacher, inp, visible, cfg)
    assert terms.den['jepa'] == loss_counts(inp, visible, cfg, model_cfg=CFG)['jepa'] == 0
    assert terms.backward_loss() == 0 and torch.isfinite(terms.backward_loss())
    terms.backward_loss().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())


def test_counts_require_geometry_and_keep_calibration_and_validate_component_keys():
    from dataclasses import replace
    from wearusfm.training.jepa import LossTerms, loss_counts
    inp, visible = _inputs(), _visible(_inputs())
    cfg = JEPAConfig('b', .2, .99)
    with pytest.raises(TypeError):
        loss_counts(inp, visible, cfg)
    with pytest.raises(ValueError, match='geometry'):
        loss_counts(inp, visible[:, :-1], cfg, model_cfg=CFG)
    with pytest.raises(ValueError, match='calibrated'):
        loss_counts(inp, visible, replace(cfg, keep_weight=.05), _heterogeneous_targets(inp), model_cfg=replace(CFG, keep_readout=True))
    with pytest.raises(ValueError, match='component keys'):
        LossTerms({'jepa': torch.tensor(1.)}, {'jepa': torch.tensor(1)}, {'jepa': 1.}).backward_loss({})


def test_loss_terms_pytree_roundtrip_exposes_differentiable_outputs_to_ddp():
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from wearusfm.training.jepa import LossTerms
    p = torch.tensor(2., requires_grad=True)
    terms = LossTerms({'jepa': p.square()}, {'jepa': torch.tensor(2)}, {'jepa': 1.})
    leaves, spec = tree_flatten(terms)
    assert any(x is terms.num['jepa'] for x in leaves)
    restored = tree_unflatten(leaves, spec)
    assert isinstance(restored, LossTerms) and restored.num['jepa'] is terms.num['jepa']
    restored.backward_loss().backward()
    assert p.grad == 2.
