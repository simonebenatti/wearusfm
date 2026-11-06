"""Finestra del kernel piu' lunga del passo (150/125 ms e 100/75 ms, sempre passo di 25 ms): il banco di
sempre resta bit per bit lo stesso senza finestra; con la finestra la griglia dei token non cambia, le basi scalano, le feature restano coerenti
fra frequenze di campionamento (il senso del gate D8), e la perdita JEPA conta solo i token interamente nascosti."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_frontend import _rel, _sig  # noqa: E402
from wearusfm.model import frontend as FE  # noqa: E402
from wearusfm.model.fm import FMConfig, WearUsFM  # noqa: E402
from wearusfm.model.patch_tokens import PatchTokenizer  # noqa: E402
from wearusfm.training.jepa import core_hidden  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
CFG = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4)


@pytest.mark.parametrize("window_ms,harmonics,knots,omega", [(100., 104, 80, 24.), (150., 156, 120, 36.)])
def test_window_kwargs_scale_the_bases_and_reject_bad_windows(window_ms, harmonics, knots, omega):
    assert FE.window_kwargs(25.0, None) == {}
    kw = FE.window_kwargs(25.0, window_ms)
    assert kw == {"window_ms": window_ms, "n_harmonics": harmonics, "n_knots": knots, "mlp_omega0": omega}
    for bad in (25.0, 10.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            FE.window_kwargs(25.0, bad)


def test_without_a_window_the_frontend_is_bit_identical():
    with torch.no_grad():
        a, b = FE.ContinuousKernelFrontEnd(seed=0), FE.ContinuousKernelFrontEnd(seed=0, window_ms=None)
        assert a.window_s == a.patch_s and a.lead_s == 0.0
        for fs in (2000.0, 1000.0, 200.0):
            x = _sig(int(fs), 1.0, np.linspace(5, 0.4 * fs, 30))
            assert torch.equal(a(x, fs), b(x, fs))
        # il tokenizer: stesso guadagno d'ingresso e stesse uscite
        torch.manual_seed(0)
        t1 = PatchTokenizer(8, patch_ms=25.0)
        torch.manual_seed(0)
        t2 = PatchTokenizer(8, patch_ms=25.0, **FE.window_kwargs(25.0, None))
        assert torch.equal(t1.fe_gain, t2.fe_gain) and t1.window_s == 0.025


@pytest.mark.parametrize("window_ms,lead_s,n_ref", [(100., .0375, 1600), (150., .0625, 2400)])
def test_long_window_keeps_the_token_grid_and_is_consistent_across_sampling_rates(window_ms, lead_s, n_ref):
    with torch.no_grad():
        fe = FE.ContinuousKernelFrontEnd(seed=0, **FE.window_kwargs(25.0, window_ms))
        assert fe.window_s == pytest.approx(window_ms / 1000) and fe.patch_s == 0.025 and fe.lead_s == pytest.approx(lead_s)
        assert fe.kernels_ref().shape == (64, n_ref) and fe.d_out == 64
        for fs in (200., 1000., 2000.):
            out = fe(torch.zeros(2, 3, int(fs), dtype=torch.float64), fs)
            assert out.shape == (2, 3, 40, 64)  # 40 token/s a tutte le frequenze, anche con overlap di 75 ms
        # stesso segnale a banda limitata a 2 kHz e a 1 kHz / 200 Hz: stesse feature lontano dai bordi (la proprieta' del gate D8)
        for fs_b, fmax in ((1000, 440), (200, 88)):
            fr = np.linspace(5, fmax, 40)
            a = fe(_sig(2000, 2.0, fr), 2000.0)[:, :, 8:-8]
            b = fe(_sig(fs_b, 2.0, fr), float(fs_b))[:, :, 8:-8]
            for name, s in fe.family_slices().items():
                assert _rel(a, b, s) < 0.02, (fs_b, name)
        # Un impulso a 50 ms dalla patch cambia il token; col banco da 25 ms, contesto a parte, quasi no.
        x = torch.zeros(1, 1, 2000, dtype=torch.float64)
        x[..., 1000] = 1.0  # impulso al campione 1000 = 500 ms = inizio della patch 20
        short = FE.ContinuousKernelFrontEnd(seed=0)
        resp_long = fe(x, 2000.0)[0, 0].abs().sum(-1)  # (40,) risposta per patch
        resp_short = short(x, 2000.0)[0, 0].abs().sum(-1)
        assert resp_long[18] > 0.1 * resp_long[20] and resp_short[18] < 0.02 * resp_short[20]  # due patch prima: dentro la finestra lunga, fuori da quella corta
        # il costo dichiarato cresce col supporto
        tl, ts = PatchTokenizer(8, patch_ms=25.0, **FE.window_kwargs(25.0, window_ms)), PatchTokenizer(8, patch_ms=25.0)
        assert tl.flops(8, 40, 2000.0) > ts.flops(8, 40, 2000.0)


@pytest.mark.parametrize("window_ms,kernel_bounds,full_bounds", [
    (100., (.4625, .5625), (.3625, .6625)),
    (150., (.4375, .5875), (.3375, .6875)),
])
def test_antialias_context_remains_100ms_each_side_and_has_real_signal_support(window_ms, kernel_bounds, full_bounds):
    """Token 20: 500..525 ms; independent physical intervals, not FE indexing helpers."""
    fs, token = 200., 20
    t = np.arange(int(fs)) / fs
    inside = (t >= full_bounds[0]) & (t < full_bounds[1])
    kernel = (t >= kernel_bounds[0]) & (t < kernel_bounds[1])
    noise = torch.tensor(np.random.default_rng(11).normal(size=int(fs)), dtype=torch.float64)
    x = torch.zeros(1, 2, int(fs), dtype=torch.float64)
    x[0, 0, ~inside] = noise[~inside]  # tutto il segnale esterno al supporto reale
    x[0, 1, inside & ~kernel] = noise[inside & ~kernel]  # solo le code dell'anti-aliasing
    with torch.no_grad():
        fe = FE.ContinuousKernelFrontEnd(seed=0, **FE.window_kwargs(25., window_ms))
        no_context = FE.ContinuousKernelFrontEnd(seed=0, context_ms=0., **FE.window_kwargs(25., window_ms))
        assert fe.context_s == FE.ContinuousKernelFrontEnd().context_s == .1
        assert full_bounds[1] - full_bounds[0] == pytest.approx(window_ms / 1000 + .2)
        response = fe(x, fs)[0, :, token]
        assert torch.equal(response[0], torch.zeros_like(response[0]))
        assert response[1].abs().sum() > 1e-10  # finestra nominale + filtro: 300 ms per window_ms=100
        assert torch.equal(no_context(x, fs)[0, 1, token], torch.zeros_like(response[1]))


def test_core_hidden_keeps_only_tokens_whose_window_is_fully_hidden():
    visible = torch.ones(2, 12, dtype=torch.bool)
    visible[0, 3:10] = False  # canale 0: nascoste le patch 3..9 (7 patch)
    visible[1, :] = False  # canale 1: tutto nascosto
    hidden = ~visible
    assert torch.equal(core_hidden(hidden, visible, 0), hidden)  # senza finestra: invariato
    core = core_hidden(hidden, visible, 3)
    assert core[0].nonzero().flatten().tolist() == [6]  # solo la patch 6 ha 3 patch nascoste per lato (3..5 e 7..9)
    assert core[1].all()  # oltre i bordi del segnale conta come nascosto
    assert not core_hidden(hidden, visible, 4)[0].any()  # con raggio 4 nessun token di 7 patch e' interamente nascosto
    assert core_hidden(hidden, visible, 2)[0].nonzero().flatten().tolist() == [5, 6, 7]
    assert FMConfig(**{**CFG.__dict__}).core_radius == 0 and replace(CFG, window_ms=150.0).core_radius == 3
    assert replace(CFG, window_ms=75.0).core_radius == 1 and replace(CFG, window_ms=125.0).core_radius == 2
    assert replace(CFG, window_ms=100.0).core_radius == 2


@pytest.mark.parametrize("window_ms,n_core", [(100., 16), (150., 14)])
def test_model_with_a_window_runs_and_the_config_is_validated(window_ms, n_core):
    with pytest.raises(ValueError):
        FMConfig(**{**CFG.__dict__, "window_ms": 25.0})
    with pytest.raises(ValueError, match="rami spettrali"):
        FMConfig(**{**CFG.__dict__, "window_ms": window_ms, "spectral_branches": ("psd",), "keep_readout": True})
    cfg = replace(CFG, window_ms=window_ms)
    torch.manual_seed(0)
    model = WearUsFM(cfg).eval()
    assert model.tokenizer.window_s == pytest.approx(window_ms / 1000) and model.tokenizer.patch_s == 0.025
    from wearusfm.harness import fm_features as FF

    w = np.random.default_rng(0).normal(size=(3, 1000, 8))
    f = FF.extract_features(model, w, 200.0, FF.myo8_montage("epn612", 200.0), np.array(["a", "a", "a"]), batch=2)
    assert f.shape == (3, 32) and np.isfinite(f).all()
    # la perdita JEPA conta solo i token interamente nascosti: con una maschera di 2 patch nessuna query
    from wearusfm.model import channel_codes as CC
    from wearusfm.model.channel_identity import codes_to_tensors
    from wearusfm.model.fm import ModelInputs
    from wearusfm.model.local_encoder import sets_to_tensors
    from wearusfm.training.jepa import JEPAConfig, jepa_losses, jepa_loss_terms, loss_counts, make_teacher

    montage = FF.myo8_montage("epn612", 200.0)
    layout, code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
    sets = CC.attention_sets(layout, 8)
    sig = [torch.from_numpy(np.ascontiguousarray(w[0].T).astype(np.float32))]
    inp = ModelInputs(sig, [200.0], codes_to_tensors(CC.pack_codes([code])), sets_to_tensors(CC.pack_attention_sets([sets])), [8],
                      torch.as_tensor(layout.qc_valid))
    teacher = make_teacher(model)
    jcfg = JEPAConfig(target="b", anchor_weight=0.0, ema_momentum=0.996)
    visible = torch.ones(8, 200, dtype=torch.bool)
    visible[:, 100:102] = False  # 2 patch su tutti i canali: nessun token interamente nascosto con raggio 2 o 3
    with pytest.raises(ValueError, match="nessun token nascosto"):
        jepa_losses(model, teacher, inp, visible, jcfg)
    visible[:, 100:120] = False  # 20 patch: core 102..117 (100 ms) oppure 103..116 (150 ms)
    losses = jepa_losses(model, teacher, inp, visible, jcfg)
    assert torch.isfinite(losses["jepa"]) and losses["core_hidden_fraction"] == pytest.approx(n_core / 20)
    for per_window in (False, True):
        loss_cfg = replace(jcfg, loss_per_window=per_window)
        terms = jepa_loss_terms(model, teacher, inp, visible, loss_cfg)
        counts = loss_counts(inp, visible, loss_cfg, model_cfg=cfg)
        assert set(counts) == set(terms.den) and all(torch.equal(counts[k], terms.den[k]) for k in counts)
        assert int(counts["jepa"]) == (1 if per_window else 8 * n_core * CFG.dim)


@pytest.mark.parametrize("window_ms", [100., 150.])
@pytest.mark.parametrize("fs", [200., 1000., 2000.])
def test_long_window_student_is_invariant_to_changes_in_hidden_raw_samples(window_ms, fs):
    """Perturb raw samples before encode; masking must precede the overlapping tokenizer."""
    from wearusfm.harness import fm_features as FF
    from wearusfm.model import channel_codes as CC
    from wearusfm.model.channel_identity import codes_to_tensors
    from wearusfm.model.fm import ModelInputs
    from wearusfm.model.local_encoder import sets_to_tensors
    from wearusfm.training.jepa import hidden_queries

    torch.manual_seed(17)
    model = WearUsFM(replace(CFG, window_ms=window_ms)).eval()
    montage = FF.myo8_montage("epn612", fs)
    layout, code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
    sets = CC.attention_sets(layout, 8)
    signal = torch.randn(8, int(fs))
    inp = ModelInputs([signal], [fs], codes_to_tensors(CC.pack_codes([code])),
                      sets_to_tensors(CC.pack_attention_sets([sets])), [8], torch.as_tensor(layout.qc_valid))
    visible = torch.ones(8, 40, dtype=torch.bool)
    visible[:, 12:20] = False
    visible[2] = False
    visible[5, 28:32] = False
    # A queste tre fs ciascuna patch ha esattamente 5/25/50 campioni: oracolo indipendente dal codice di azzeramento.
    hidden_samples = (~visible).repeat_interleave(int(fs / 40), dim=1)
    changed = signal.clone()
    changed[hidden_samples] += 100 * torch.randn(int(hidden_samples.sum()))
    assert torch.equal(signal[~hidden_samples], changed[~hidden_samples])
    noisy = replace(inp, signals=[changed])
    q_ch, q_t, q_off = hidden_queries(~visible, inp.counts)
    with torch.no_grad():
        enc, enc_changed = model.encode(inp, visible), model.encode(noisy, visible)
        assert torch.equal(enc.local, enc_changed.local) and torch.equal(enc.z, enc_changed.z)
        assert torch.equal(model.decode(enc, q_ch, q_t, q_off), model.decode(enc_changed, q_ch, q_t, q_off))
        # Controllo positivo: la perturbazione e' rilevabile se non si occultano i campioni grezzi.
        assert not torch.allclose(model.encode(inp, None).z, model.encode(noisy, None).z)


@pytest.mark.parametrize("window_ms,radius", [(100., 2), (150., 3)])
def test_trainer_flags_for_target_a_and_window(window_ms, radius):
    sp = importlib.util.spec_from_file_location("train_jepa_window", REPO / "scripts" / "train_jepa.py")
    TJ = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(TJ)
    base = ["--preset", "sanity", "--manifest", "-", "--root", "-", "--out-dir", "-"]
    cfg = TJ.build_config(TJ.build_parser().parse_args(base))
    assert cfg.jepa.target == "b" and cfg.model.window_ms is None  # i default del preset, invariati
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--jepa-target", "a", "--window-ms", str(window_ms)]))
    assert cfg.jepa.target == "a" and cfg.model.window_ms == window_ms and cfg.model.core_radius == radius
    with pytest.raises(ValueError, match="--window-ms"):
        TJ.build_config(TJ.build_parser().parse_args(base + ["--window-ms", "25"]))
    # R2 e lo scostamento da D10 di R5: target per regione e quote delle maschere; default intoccati
    assert cfg.jepa.region_target is False and (cfg.loader.mask.share_short, cfg.loader.mask.share_spatial) == (0.15, 0.40)
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--jepa-region-target"]))
    assert cfg.jepa.region_target is True and cfg.loader.mask.share_short == 0.15
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--window-ms", str(window_ms), "--mask-shares", "0:0.294:0.235:0.471"]))
    m = cfg.loader.mask
    assert (m.share_short, m.share_medium, m.share_long, m.share_spatial) == (0.0, 0.294, 0.235, 0.471) and m.ratio == 0.5
    for bad in ("0.5:0.5", "0.1:0.2:0.3:0.5", "a:b:c:d"):
        with pytest.raises(ValueError, match="--mask-shares"):
            TJ.build_config(TJ.build_parser().parse_args(base + ["--mask-shares", bad]))
