"""Rami spettrali in parallelo al raw (step 2 PSD e step 3 wavelet; Simone, 13/10/2026): frontend contro i target dello Step 1 e contro la
teoria, supporto e disponibilita', invarianze dell'encoder, calibrazione, loss sul readout (gradienti, encoder congelato, contratto delle loss),
modello accoppiato a quello senza rami, trainer (congiunto, ripresa, encoder congelato), estrazione P1/P2, equivalenza DDP e accumulo."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from wearusfm.data import pretraining_loader as L
from wearusfm.harness import fm_features as FF
from wearusfm.model import anchor_targets as AT
from wearusfm.model import spectral_branches as SB
from wearusfm.model.fm import FMConfig, ModelInputs, WearUsFM
from wearusfm.model.spectral_step1 import keep_objective
from wearusfm.training import run as R

from test_spectral_step1 import _calibration, _setup


def _inputs(signals, fs, limits, guards=None, qc=None):
    """ModelInputs minimi per i rami (codici e insiemi non servono)."""
    counts = [x.shape[0] for x in signals]
    qc = torch.ones(sum(counts), dtype=torch.bool) if qc is None else qc
    return ModelInputs([torch.as_tensor(x, dtype=torch.float32) for x in signals], list(fs), {}, {}, counts, qc,
                       band_limit=[np.asarray(v, dtype=np.float64) for v in limits], edge_guard=guards or [(0, 0)] * len(signals))


def _noise(c, seconds, fs, seed=0):
    return np.random.default_rng(seed).standard_normal((c, int(seconds * fs))).astype(np.float32)


# --- frontend ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fs", [200.0, 1000.0, 1500.0, 2000.0, 2048.0])
def test_psd_frontend_equals_step1_targets(fs):
    """Stesse finestre, stessa PSD, stesse bande e stessa disponibilita' dei target multiscala dello Step 1 (`anchor_targets`)."""
    x = _noise(3, 3.1, fs, seed=int(fs))
    limit = np.array([450.0, 300.0, fs / 2])
    tg = AT.anchor_targets(x, fs, limit, multiscale=True)
    out = SB.PSDFrontend()(torch.from_numpy(x), fs, np.minimum(limit, fs / 2), (0, 0))
    for name, attr, per in (("fast", "ms_fast", 1), ("slow", "ms_slow", 4)):
        power, valid = (v.numpy() for v in out[name])
        ref, ref_ok, ref_av = getattr(tg, attr)[:, ::per], getattr(tg, attr + "_valid")[0, ::per], getattr(tg, attr + "_available")
        p = min(power.shape[1], ref.shape[1])
        both = valid[:, :p] & ref_ok[None, :p, None] & ref_av[:, None, :]
        assert both.sum() > 0
        np.testing.assert_allclose(np.log(power[:, :p] + 1e-8)[both], ref[:, :p][both], atol=2e-4)
        np.testing.assert_array_equal(valid[:, :p].any(1), ref_av & ref_ok[:p].any())  # disponibilita' delle bande come i target


def test_branch_support_near_edges_is_stricter_than_the_targets():
    """Review GPT del 13/10: vicino ai bordi del tratto il ramo e i target non hanno lo stesso supporto. I target dello Step 1 escludono solo le
    patch che toccano la zona di bordo (la finestra da 500 ms centrata su una patch valida puo' entrarci); il ramo vuole l'INTERA finestra fuori
    (guida del pacchetto: supporto completo). Finestra da 1 s con 0,1 s di bordo: 6 frame lenti nei target, 4 nel ramo; senza bordo, uguali."""
    fs = 2000.0
    x = _noise(2, 1.0, fs)
    for guard, n_targets, n_branch in (((0, 0), 6, 6), ((200, 200), 6, 4)):
        tg = AT.anchor_targets(x, fs, 450.0, guard=guard, multiscale=True)
        slow_ok = tg.ms_slow_valid[0]  # per patch, ogni frame lento ripetuto sulle sue 4 patch
        frames = {p // 4 for p in np.flatnonzero(slow_ok)}
        _, valid = SB.PSDFrontend()(torch.from_numpy(x), fs, np.full(2, 450.0), guard)["slow"]
        assert len(frames) == n_targets and int(valid[0].any(-1).sum()) == n_branch


def test_wavelet_energy_is_band_power_for_white_noise_and_tone_gain():
    """Rumore bianco: energia di ogni filtro = PSD * larghezza della banda (normalizzazione a banda equivalente di rumore); la PSD da' PSD *
    (bin nella banda) * fs / n, quantizzata dalla risoluzione della finestra (a 50 ms un bin ogni 20 Hz). Sinusoide di ampiezza 1 al centro di
    un filtro: energia |H(centro)|^2 / 2."""
    fs = 2000.0
    x = _noise(4, 30.0, fs, seed=1)
    lim = np.full(4, 450.0)
    wav = SB.WaveletFrontend()(torch.from_numpy(x), fs, lim, (0, 0))
    psd = SB.PSDFrontend()(torch.from_numpy(x), fs, lim, (0, 0))
    density = 2.0 / fs  # PSD monolatera di un rumore bianco di varianza 1
    for name, win_ms, _, nb in SB.SCALES:
        e = SB.band_edges(nb)
        p, v = wav[name]
        np.testing.assert_allclose(((p * v).sum((0, 1)) / v.sum((0, 1))).numpy(), density * np.diff(e), rtol=0.05)
        n = int(round(win_ms / 1000 * fs))
        f = np.fft.rfftfreq(n, 1 / fs)
        bins = np.array([((f >= lo) & (f < hi)).sum() for lo, hi in zip(e[:-1], e[1:])])
        q, w = psd[name]
        np.testing.assert_allclose(((q * w).sum((0, 1)) / w.sum((0, 1))).numpy(), density * bins * fs / n, rtol=0.1)  # bande da 1-2 bin
    bank = SB.WaveletBank(24)
    c, s = (t.numpy() for t in bank.physical())
    k = 10
    tone = np.sin(2 * np.pi * c[k] * np.arange(int(4 * fs)) / fs).astype(np.float32)[None, :]
    p, v = SB.WaveletFrontend()(torch.from_numpy(tone), fs, np.array([450.0]), (0, 0))["slow"]
    gain2 = bank.width.numpy()[k] / (s[k] * np.sqrt(np.pi))
    assert float(p[0, :, k][v[0, :, k]].mean()) == pytest.approx(gain2 / 2, rel=1e-3)


def test_wavelet_same_signal_at_two_sampling_rates():
    """Filtri definiti in Hz: lo stesso segnale (toni sotto 450 Hz) a 1 e 2 kHz da' le stesse energie medie nel tempo (frame per frame i
    battimenti fra toni vicini, sommati su 50 o 100 campioni, differiscono di qualche per cento)."""
    rng = np.random.default_rng(2)
    f, a, ph = rng.uniform(25, 440, 12), rng.uniform(0.2, 1, 12), rng.uniform(0, 6.3, 12)
    out = {}
    for fs in (1000.0, 2000.0):
        t = np.arange(int(3 * fs)) / fs
        x = (a[:, None] * np.sin(2 * np.pi * f[:, None] * t + ph[:, None])).sum(0).astype(np.float32)[None, :]
        out[fs] = SB.WaveletFrontend()(torch.from_numpy(x), fs, np.array([450.0]), (0, 0))
    for name in ("fast", "slow"):
        (p1, v1), (p2, v2) = out[1000.0][name], out[2000.0][name]
        both = (v1 & v2)[0]
        m1, m2 = ((p[0] * both).sum(0) / both.sum(0) for p in (p1, p2))
        np.testing.assert_allclose(m1.numpy(), m2.numpy(), rtol=1e-2, atol=1e-6)


def test_support_guard_and_availability():
    """Un frame vale solo se finestra e raggio del filtro stanno nel segnale fuori dalle zone di bordo; le wavelet basse hanno il supporto
    piu' lungo. Bande sopra il limite del canale e canali scartati dal QC: mai validi."""
    fs = 1000.0
    x = torch.from_numpy(_noise(2, 4.0, fs))
    lim = np.array([450.0, 100.0])
    for front in (SB.PSDFrontend(), SB.WaveletFrontend()):
        free = front(x, fs, lim, (0, 0))
        guarded = front(x, fs, lim, (300, 300))
        for name, win_ms, hop_ms, nb in SB.SCALES:
            v0, v1 = free[name][1], guarded[name][1]
            assert bool((v0 >= v1).all()) and v1.sum() < v0.sum()  # il bordo toglie frame, non ne aggiunge
            upper = SB.band_edges(nb)[1:]
            assert not v0[1][:, upper > 100.0].any() and v0[1][:, upper <= 100.0].any()
    wav = SB.WaveletFrontend()(x, fs, lim, (0, 0))["slow"][1][0]
    assert wav[:, 0].sum() < wav[:, -1].sum()  # 20-23 Hz: raggio ~0,36 s; la banda alta: ~0,02 s
    branch = SB.SpectralBranch("psd", hidden=4)
    sp = branch.spectra(_inputs([x.numpy()], [fs], [lim], qc=torch.tensor([True, False])))
    assert not sp["fast"][1][1].any() and sp["fast"][1][0].any()


# --- encoder e calibrazione -------------------------------------------------------------------------------------------------------------

def _calibrated(kind, learnable=False, seed=0):
    torch.manual_seed(seed)
    b = SB.SpectralBranch(kind, out_dim=16, hidden=8, learnable=learnable)
    sigs = [_noise(3, 2.5, 2000.0, 10), _noise(2, 4.0, 1000.0, 11), _noise(4, 3.0, 2000.0, 12)]
    b.set_stats(b.accumulate_stats(_inputs(sigs, [2000.0, 1000.0, 2000.0], [np.full(3, 450.0), np.full(2, 450.0), np.full(4, 450.0)])))
    return b


@pytest.mark.parametrize("kind", SB.BRANCH_KINDS)
def test_branch_packing_and_channel_permutation_invariance(kind):
    """Un campione da solo o impacchettato con altri (fs e durate diverse, frame in coda di padding) da' lo stesso vettore; l'ordine dei canali
    non conta."""
    b = _calibrated(kind)
    a, other = _noise(3, 3.3, 2000.0, 20), _noise(5, 1.7, 1000.0, 21)
    lim_a, lim_o = np.array([450.0, 450.0, 300.0]), np.full(5, 450.0)
    alone = b(_inputs([a], [2000.0], [lim_a], guards=[(40, 10)]))
    packed = b(_inputs([other, a], [1000.0, 2000.0], [lim_o, lim_a], guards=[(0, 0), (40, 10)]))
    torch.testing.assert_close(packed[1:], alone, rtol=1e-5, atol=1e-6)
    perm = [2, 0, 1]
    torch.testing.assert_close(b(_inputs([a[perm]], [2000.0], [lim_a[perm]], guards=[(40, 10)])), alone, rtol=1e-5, atol=1e-6)


def test_unobserved_coordinates_are_masked_like_missing_bands():
    """Calibrato solo a 200 Hz (bande fino a 100 Hz): a 2 kHz le coordinate mai viste restano fuori, come se il canale avesse banda 100 Hz."""
    torch.manual_seed(0)
    b = SB.SpectralBranch("psd", out_dim=8, hidden=4)
    low = [_noise(2, 3.0, 200.0, s) for s in range(3)]
    b.set_stats(b.accumulate_stats(_inputs(low, [200.0] * 3, [np.full(2, 100.0)] * 3)))
    x = _noise(2, 3.0, 2000.0, 9)
    torch.testing.assert_close(b(_inputs([x], [2000.0], [np.full(2, 450.0)])), b(_inputs([x], [2000.0], [np.full(2, 100.0)])))
    fresh = SB.SpectralBranch("psd", out_dim=8, hidden=4)
    with pytest.raises(RuntimeError, match="calibrare"):
        fresh(_inputs([x], [2000.0], [np.full(2, 450.0)]))


def test_no_support_gives_flags_not_errors():
    """Un ritaglio troppo corto per ogni frame: vettore finito coi flag di presenza spenti (nessuna eccezione a meta' training)."""
    b = _calibrated("wavelet")
    z = b(_inputs([_noise(2, 0.04, 2000.0)], [2000.0], [np.full(2, 450.0)]))
    assert torch.isfinite(z).all()


def test_learnable_wavelet_gradients_and_bounds():
    b = _calibrated("wavelet", learnable=True)
    fixed = _calibrated("wavelet", learnable=False)
    inp = _inputs([_noise(2, 2.0, 2000.0, 30)], [2000.0], [np.full(2, 450.0)])
    b(inp).square().sum().backward()
    for bank in b.frontend.banks.values():
        assert bank.center_logits.grad is not None and bank.center_logits.grad.abs().sum() > 0
        with torch.no_grad():
            bank.center_logits.fill_(50.0)
            bank.sigma_logits.fill_(-50.0)
        c, s = bank.physical()
        assert torch.allclose(c, bank.center_hi) and torch.allclose(s, bank.sigma_lo)
    assert not any(p.requires_grad for p in fixed.frontend.parameters())
    np.testing.assert_array_equal(b.frontend.banks["fast"].radius(2000.0), fixed.frontend.banks["fast"].radius(2000.0))


# --- modello, loss sul readout ----------------------------------------------------------------------------------------------------------

def test_config_rules_and_pairing_with_the_model_without_branches():
    base = R.small_config().model
    with pytest.raises(ValueError, match="keep_readout"):
        replace(base, spectral_branches=("psd",))
    for bad in (("gabor",), ("psd", "psd")):
        with pytest.raises(ValueError, match="rami spettrali"):
            replace(base, keep_readout=True, spectral_branches=bad)
    with pytest.raises(ValueError, match="wavelet_learnable"):
        replace(base, keep_readout=True, spectral_branches=("psd",), wavelet_learnable=True)
    assert FMConfig(**{**R.config_to_dict(R.small_config())["model"], "keep_readout": True, "spectral_branches": ["psd"]}).spectral_branches == ("psd",)
    plain = replace(base, keep_readout=True)
    states, rngs = [], []
    for cfg in (plain, replace(plain, spectral_branches=("psd", "wavelet"))):
        torch.manual_seed(5)
        states.append(WearUsFM(cfg).state_dict())
        rngs.append(torch.get_rng_state())
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])  # stesso encoder raw
    assert torch.equal(rngs[0], rngs[1])  # stesso generatore dopo: dropout del muscolo accoppiato
    assert {k.split(".")[0] for k in set(states[1]) - set(states[0])} == {"spectral_branches", "branch_heads"}


def test_branch_init_does_not_reseed_accelerators(monkeypatch):
    """Review GPT del 13/10: l'inizializzazione dei rami non deve toccare i generatori CUDA/MPS (fork_rng(devices=[]) ripristina solo la CPU),
    altrimenti sulla GPU il dropout del modello con i rami non e' piu' accoppiato a quello senza."""
    calls = []
    for mod, name in ((torch.cuda, "manual_seed_all"), (torch.cuda, "manual_seed"), (torch.mps, "manual_seed")):
        monkeypatch.setattr(mod, name, lambda *a, _n=name, **k: calls.append(_n))
    WearUsFM(replace(R.small_config().model, keep_readout=True, spectral_branches=("psd", "wavelet")))
    assert calls == []


def _branch_model(tmp_path, kinds=("psd", "wavelet"), learnable=False, alpha=0.5):
    root, manifest, cfg, batch, y, m = _setup(tmp_path)
    mcfg = replace(cfg.model, spectral_branches=kinds, wavelet_learnable=learnable, branch_alpha=alpha)
    torch.manual_seed(0)
    s = WearUsFM(mcfg)
    inp, visible, _ = L.to_model_inputs(batch)
    s.spectral_branches.calibrate([inp])
    for h in [s.spectral_keep, *s.branch_heads.values()]:
        h.fit_target_stats(y, m)
    return s, inp, visible, batch


def test_raw_keep_unchanged_and_branch_term_from_one_encode(tmp_path, monkeypatch):
    """Parere GPT d5c53c1: la keep dello Step 1 sulla testa raw resta identica anche col modello coi rami; il termine dei rami e' (1 - alpha)
    testa fusa + alpha media delle teste dei rami; un solo encode non mascherato per entrambi; RNG globale intatto."""
    s, inp, _, batch = _branch_model(tmp_path, alpha=0.3)
    ref_loss, ref_met = keep_objective(s, inp, batch.anchor_targets)  # Step 1, invariato (nessun aggancio ai rami)
    calls = []
    encode = s.encode
    monkeypatch.setattr(s, "encode", lambda *a, **k: (calls.append(1), encode(*a, **k))[1])
    rng = torch.get_rng_state().clone()
    parts = SB.readout_objectives(s, inp, batch.anchor_targets, keep=True)
    assert calls == [1] and torch.equal(torch.get_rng_state(), rng)
    (keep, keep_met), (kb, kb_met) = parts["keep"], parts["keep_branches"]
    assert torch.equal(keep, ref_loss) and all(torch.equal(keep_met[k], ref_met[k]) for k in ref_met)
    assert float(kb.detach()) == pytest.approx(0.7 * float(kb_met["keep_fused"]) + 0.3 * (float(kb_met["keep_psd"]) +
                                                                                           float(kb_met["keep_wavelet"])) / 2, rel=1e-6)
    lt, _, num, den, log_num, log_den = SB.readout_objectives(s, inp, batch.anchor_targets, keep=False, terms=True)["keep_branches"]
    assert float((num / den).detach()) == pytest.approx(float(lt.detach()), rel=1e-6)
    assert int(den) == int(kb_met["keep_branches_valid_samples"])
    assert {"keep_fused", "keep_fused_fast_shape", "keep_psd_slow_shape", "keep_wavelet_fast_energy"} <= set(log_num)
    assert set(log_num) == set(log_den) and not set(log_num) & set(ref_met)  # nessuna collisione coi nomi della keep raw
    (keep + kb).backward()
    for name in ("tokenizer", "backbone", "spectral_keep", "spectral_branches", "branch_heads"):
        grads = [p.grad for p in getattr(s, name).parameters() if p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0, name
    assert all(p.grad is None for p in s.spectral_branches.branches["wavelet"].frontend.parameters())  # filtri fissi


def test_heads_with_different_calibration_are_refused(tmp_path):
    s, inp, _, batch = _branch_model(tmp_path)
    with torch.no_grad():
        s.branch_heads["psd"].target_active[0] = ~s.branch_heads["psd"].target_active[0]
    with pytest.raises(ValueError, match="target_active"):
        SB.readout_objectives(s, inp, batch.anchor_targets, keep=True)


def test_jepa_losses_two_additive_terms_and_counts(tmp_path):
    """jepa_losses: totale = JEPA + ancore + keep_weight keep + branch_weight keep_branches; `loss_counts` conta le due componenti."""
    from wearusfm.training.jepa import JEPAConfig, jepa_loss_terms, jepa_losses, loss_counts, make_teacher

    s, inp, visible, batch = _branch_model(tmp_path)
    teacher = make_teacher(s)
    base = JEPAConfig("b", .2, .99, loss_per_window=True)
    torch.manual_seed(3)
    plain = jepa_losses(s, teacher, inp, visible, replace(base, keep_weight=.05), anchor_targets=batch.anchor_targets)
    torch.manual_seed(3)
    both = jepa_losses(s, teacher, inp, visible, replace(base, keep_weight=.05, branch_weight=.2), anchor_targets=batch.anchor_targets)
    assert torch.equal(plain["keep"], both["keep"])
    torch.testing.assert_close(both["total"], plain["total"] + .2 * both["keep_branches"])
    cfg = replace(base, keep_weight=.05, branch_weight=.2)
    counts = loss_counts(inp, visible, cfg, batch.anchor_targets, model_cfg=s.cfg, keep_target_active=s.spectral_keep.target_active)
    torch.manual_seed(3)
    terms = jepa_loss_terms(s, teacher, inp, visible, cfg, anchor_targets=batch.anchor_targets)
    assert torch.equal(counts["keep_branches"], counts["keep"]) and all(torch.equal(terms.den[k], counts[k]) for k in ("keep", "keep_branches"))
    assert terms.weight["keep_branches"] == .2 and terms.weight["keep"] == .05
    with pytest.raises(ValueError, match="branch_weight"):
        JEPAConfig("b", .2, .99, branch_weight=-1.)


def test_frozen_encoder_trains_only_branches(tmp_path):
    s, inp, _, batch = _branch_model(tmp_path, learnable=True)
    SB.freeze_raw(s)
    s.train()
    with pytest.raises(ValueError, match="congelato"):
        SB.readout_objectives(s, inp, batch.anchor_targets, keep=True)
    l1, _ = SB.readout_objectives(s, inp, batch.anchor_targets, keep=False)["keep_branches"]
    l2, _ = SB.readout_objectives(s, inp, batch.anchor_targets, keep=False)["keep_branches"]
    assert torch.equal(l1, l2)  # encode in eval: niente dropout del muscolo
    l1.backward()
    for name, p in s.named_parameters():
        assert (p.grad is not None) == name.startswith(SB.BRANCH_PREFIXES), name
    assert s.training


def test_readout_is_step1_without_branches_and_extends_with_them(tmp_path):
    s, inp, _, _ = _branch_model(tmp_path)
    z = SB.readout(s, s.encode(inp, None), inp)
    assert z.shape == (len(inp.counts), 2 * s.cfg.dim + 2 * s.cfg.branch_dim)
    plain = WearUsFM(replace(s.cfg, spectral_branches=()))
    from wearusfm.model.spectral_step1 import pool_p1p2
    enc = plain.encode(inp, None)
    assert torch.equal(SB.readout(plain, enc, inp), pool_p1p2(enc, inp.counts, inp.qc_valid))


# --- trainer ----------------------------------------------------------------------------------------------------------------------------

def _branch_cfg(tmp_path, **kw):
    root, manifest, cfg, _, y, m = _setup(tmp_path)
    cal = tmp_path / "cal.npz"
    _calibration(cal, cfg, manifest, y, m)
    return root, manifest, R.with_spectral_branches(R.with_spectral_keep(cfg, cal, kw.pop("keep", 0.05)), ("psd", "wavelet"),
                                                    weight=kw.pop("weight", 0.05), calibration_batches=2, **kw)


def test_trainer_joint_branches_checkpoint_and_resume(tmp_path):
    root, manifest, cfg = _branch_cfg(tmp_path)
    out = tmp_path / "run"
    assert R.train(replace(cfg, keep_grad_every=1), manifest, [root], out, log=lambda _: None)["steps"] == 1
    cfg = replace(cfg, keep_grad_every=1)
    rec = json.loads((out / "metrics.jsonl").read_text().splitlines()[0])
    assert rec["keep"] > 0 and rec["keep_branches"] > 0 and rec["keep_psd"] > 0 and rec["keep_wavelet"] > 0 and rec["jepa"] > 0
    assert all(rec["shared_grad_" + k] > 0 for k in ("jepa", "keep", "keep_branches", "anchors"))  # contributi separati sul raw
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    assert state["student"]["spectral_branches.branches.psd.stats_ready"] and state["teacher"]["spectral_branches.branches.wavelet.stats_ready"]
    prov = json.loads((out / "spectral_branches.json").read_text())
    assert prov["calibration"]["batches"] == 2 and prov["frozen_init"] is None
    stats = state["student"]["spectral_branches.branches.psd.scales.slow.input_mean"].clone()
    summary = R.train(replace(cfg, max_steps=2), manifest, [root], out, log=lambda _: None)
    assert summary["steps"] == 2 and summary["spectral_branches"]["kinds"] == ["psd", "wavelet"]
    assert torch.equal(torch.load(out / "checkpoint.pt", weights_only=False)["student"]["spectral_branches.branches.psd.scales.slow.input_mean"],
                       stats)  # la ripresa non ricalibra
    with pytest.raises(ValueError, match="identity changed"):
        R.train(replace(cfg, model=replace(cfg.model, branch_alpha=0.2)), manifest, [root], out, log=lambda _: None)


def test_branches_at_weight_zero_leave_the_raw_training_of_b_unchanged(tmp_path):
    """Parere GPT d5c53c1: con branch_weight 0 la formula della loss e' quella di B; l'encoder raw (pesi, teacher, metriche) evolve bit per bit
    come nel modello senza rami. Il controllo B resta pero' il modello SENZA rami: questo, con rami non allenati nell'embedding, e' un altro
    controllo."""
    root, manifest, cfg = _branch_cfg(tmp_path, weight=0.0)
    plain = replace(cfg, model=replace(cfg.model, spectral_branches=()))
    states, recs = [], []
    for name, c in (("b", plain), ("rami0", cfg)):
        out = tmp_path / name
        R.train(replace(c, max_steps=2), manifest, [root], out, log=lambda _: None)
        states.append(torch.load(out / "checkpoint.pt", weights_only=False))
        recs.append([json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()])
    for part in ("student", "teacher"):
        assert all(torch.equal(states[1][part][k], v) for k, v in states[0][part].items()), part
    for a, b in zip(*recs):
        assert all(a[k] == b[k] for k in ("total", "jepa", "keep", "grad_norm"))
        assert "keep_branches" not in b


def test_branches_off_trainer_unchanged_by_new_fields(tmp_path):
    """Un checkpoint dello Step 1 scritto prima dei rami (config senza i campi nuovi) si riprende."""
    root, manifest, cfg, _, y, m = _setup(tmp_path)
    cal = tmp_path / "cal.npz"
    _calibration(cal, cfg, manifest, y, m)
    cfg = R.with_spectral_keep(cfg, cal, 0.05)
    out = tmp_path / "old"
    R.train(cfg, manifest, [root], out, log=lambda _: None)
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    for k in ("branch_calibration_batches", "freeze_encoder"):
        state["config"].pop(k)
    for k in R.BRANCH_MODEL_FIELDS:
        state["config"]["model"].pop(k)
    state["config"]["jepa"].pop("branch_weight")
    torch.save(state, out / "checkpoint.pt")
    assert R.train(replace(cfg, max_steps=2), manifest, [root], out, log=lambda _: None)["steps"] == 2


def test_trainer_frozen_encoder_then_p1p2_features(tmp_path):
    root, manifest, cfg_b = _branch_cfg(tmp_path, freeze_encoder=True, keep=0.0)
    src_cfg = R.with_spectral_keep(replace(cfg_b, model=replace(cfg_b.model, spectral_branches=()), freeze_encoder=False,
                                           jepa=replace(cfg_b.jepa, branch_weight=0.0)), cfg_b.keep_calibration, 0.0)
    with pytest.raises(ValueError, match="keep_weight 0"):
        R.train(replace(cfg_b, jepa=replace(cfg_b.jepa, keep_weight=0.05)), manifest, [root], tmp_path / "kw", log=lambda _: None)
    src = tmp_path / "src"
    R.train(src_cfg, manifest, [root], src, log=lambda _: None)
    with pytest.raises(ValueError, match="init_from"):
        R.train(cfg_b, manifest, [root], tmp_path / "noinit", log=lambda _: None)
    out = tmp_path / "frozen"
    summary = R.train(replace(cfg_b, max_steps=2), manifest, [root], out, log=lambda _: None, init_from=src / "checkpoint.pt")
    assert summary["steps"] == 2
    source = torch.load(src / "checkpoint.pt", weights_only=False)["teacher"]
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    raw = [k for k in source if not k.startswith(("spectral_keep.",)) and not k.endswith("num_batches_tracked")]
    assert all(torch.equal(state["student"][k], source[k]) and torch.equal(state["teacher"][k], source[k]) for k in raw)
    fresh = WearUsFM(cfg_b.model)
    SB.freeze_raw(fresh)
    trainable = sum(p.numel() for p in fresh.parameters() if p.requires_grad)
    assert summary["spectral_branches"]["trainable_parameters"] == trainable
    assert trainable == sum(p.numel() for n, p in fresh.named_parameters() if n.startswith(SB.BRANCH_PREFIXES) and not n.endswith("_logits"))
    prov = json.loads((out / "spectral_branches.json").read_text())["frozen_init"]
    assert prov["weights"] == "teacher" and prov["step"] == 1
    model = FF.load_model(str(out / "checkpoint.pt"))
    fs = 1000.0
    windows = np.random.default_rng(0).standard_normal((3, int(1.5 * fs), 8)).astype(np.float32)
    feats = FF.extract_features(model, windows, fs, FF.myo8_montage("x", fs), np.array(["a", "a", "b"]))
    assert feats.shape == (3, 2 * model.cfg.dim + 2 * model.cfg.branch_dim) and np.isfinite(feats).all()


def test_feature_sets_of_the_2a_diagnostic():
    """Test preliminare dello step 2: statistiche spettrali fisse e maschera accanto all'embedding, con lo stesso preprocessing."""
    cfg = replace(R.small_config().model, keep_readout=True)
    torch.manual_seed(0)
    model = WearUsFM(cfg)
    fs = 2000.0
    rng = np.random.default_rng(1)
    windows = rng.standard_normal((4, int(1.0 * fs), 8)).astype(np.float32)
    subjects = np.array(["a", "a", "b", "b"])
    montage = FF.myo8_montage("x", fs)
    f = {s: FF.extract_features(model, windows, fs, montage, subjects, feature_set=s) for s in FF.FEATURE_SETS}
    d = 2 * cfg.dim
    assert f["model"].shape == (4, d) and f["stats"].shape == (4, 64) and f["mask"].shape == (4, 32)
    np.testing.assert_array_equal(f["model+stats"], np.concatenate([f["model"], f["stats"]], 1))
    np.testing.assert_array_equal(f["model+mask"], np.concatenate([f["model"], f["mask"]], 1))
    np.testing.assert_array_equal(f["stats"][:, 32:], f["mask"])
    assert f["mask"].all()  # 2 kHz, bracciale fino a 1 kHz: tutte le bande e le due scale in 1 s
    scales = FF.subject_scales(windows, subjects, fs, (20.0, 450.0))
    x = FF._filtered(windows[2], fs, (20.0, 450.0)) / np.float32(scales["b"])
    y, m = FF.spectral_stats(x, fs, FF.band_limit_hz(montage, fs), np.ones(8, bool), 200)
    np.testing.assert_allclose(f["stats"][2, :32], np.where(m, y, 0), rtol=1e-6)
    with pytest.raises(ValueError, match="feature_set"):
        FF.extract_features(model, windows, fs, montage, subjects, feature_set="raw")


# --- contratto delle loss: accumulo e DDP con i rami -----------------------------------------------------------------------------------

def test_branches_one_process_accumulation_and_ddp_equal_the_global_batch(tmp_path):
    from test_ddp_equivalence import _config, _ddp, _global_batch, _reference, _same, _update
    from wearusfm.model.spectral_step1 import global_spectral_targets

    cfg = _config(True, True)
    cfg = replace(cfg, model=replace(cfg.model, keep_readout=True, spectral_branches=("psd", "wavelet")),
                  jepa=replace(cfg.jepa, keep_weight=0.05, branch_weight=0.05))
    batch = _global_batch(tmp_path, cfg)
    torch.manual_seed(0)
    model = WearUsFM(cfg.model).double()
    y, m = global_spectral_targets(batch.anchor_targets, batch.counts, batch.qc_valid)
    for h in [model.spectral_keep, *model.branch_heads.values()]:
        h.fit_target_stats(np.concatenate([y, y + 0.1]), np.concatenate([m, m]))
    model.spectral_branches.calibrate([L.to_model_inputs(batch)[0]])
    init = model.state_dict()
    ref = _reference(cfg, init, batch)
    assert ref["metrics"]["keep_psd"] > 0
    _same(ref, _update(cfg, init, batch, 2, R.DistContext(), lambda s: None), "1 processo x 2 micro-batch, rami")
    res = _ddp(tmp_path, "rami", cfg, init, batch, 2, (1,))
    _same(ref, res[0][1], "2 processi x 1, rami")
