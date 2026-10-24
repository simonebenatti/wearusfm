"""Finestra del kernel piu' lunga del passo (R5 del pilot degli obiettivi; Simone, 23/10/2026: «150ms con overlap di 125ms»): il banco di
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


def test_window_kwargs_scale_the_bases_and_reject_bad_windows():
    assert FE.window_kwargs(25.0, None) == {}
    kw = FE.window_kwargs(25.0, 150.0)
    assert kw == {"window_ms": 150.0, "n_harmonics": 156, "n_knots": 120, "mlp_omega0": 36.0}  # x6: stesso contenuto in frequenza fisico
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


def test_long_window_keeps_the_token_grid_and_is_consistent_across_sampling_rates():
    with torch.no_grad():
        fe = FE.ContinuousKernelFrontEnd(seed=0, **FE.window_kwargs(25.0, 150.0))
        assert fe.window_s == pytest.approx(0.150) and fe.patch_s == 0.025 and fe.lead_s == pytest.approx(0.0625)
        assert fe.kernels_ref().shape == (64, 2400) and fe.d_out == 64  # 150 ms a 16 kHz, stesse 64 uscite
        out = fe(torch.zeros(2, 3, 2000, dtype=torch.float64), 2000.0)
        assert out.shape == (2, 3, 40, 64)  # la griglia e' ancora il passo: 1 s / 25 ms
        # stesso segnale a banda limitata a 2 kHz e a 1 kHz / 200 Hz: stesse feature lontano dai bordi (la proprieta' del gate D8)
        for fs_b, fmax in ((1000, 440), (200, 88)):
            fr = np.linspace(5, fmax, 40)
            a = fe(_sig(2000, 2.0, fr), 2000.0)[:, :, 8:-8]
            b = fe(_sig(fs_b, 2.0, fr), float(fs_b))[:, :, 8:-8]
            for name, s in fe.family_slices().items():
                assert _rel(a, b, s) < 0.02, (fs_b, name)
        # la finestra vede davvero 150 ms: un impulso a 50 ms dalla patch cambia il token, con il banco da 25 ms (contesto a parte) quasi no
        x = torch.zeros(1, 1, 2000, dtype=torch.float64)
        x[..., 1000] = 1.0  # impulso al campione 1000 = 500 ms = inizio della patch 20
        short = FE.ContinuousKernelFrontEnd(seed=0)
        resp_long = fe(x, 2000.0)[0, 0].abs().sum(-1)  # (40,) risposta per patch
        resp_short = short(x, 2000.0)[0, 0].abs().sum(-1)
        assert resp_long[18] > 0.1 * resp_long[20] and resp_short[18] < 0.02 * resp_short[20]  # due patch prima: dentro la finestra lunga, fuori da quella corta
        # il costo dichiarato cresce col supporto
        tl, ts = PatchTokenizer(8, patch_ms=25.0, **FE.window_kwargs(25.0, 150.0)), PatchTokenizer(8, patch_ms=25.0)
        assert tl.flops(8, 40, 2000.0) > ts.flops(8, 40, 2000.0)


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
    assert FMConfig(**{**CFG.__dict__}).core_radius == 0 and replace(CFG, window_ms=150.0).core_radius == 3
    assert replace(CFG, window_ms=75.0).core_radius == 1 and replace(CFG, window_ms=125.0).core_radius == 2


def test_model_with_a_window_runs_and_the_config_is_validated():
    with pytest.raises(ValueError):
        FMConfig(**{**CFG.__dict__, "window_ms": 25.0})
    with pytest.raises(ValueError, match="rami spettrali"):
        FMConfig(**{**CFG.__dict__, "window_ms": 150.0, "spectral_branches": ("psd",), "keep_readout": True})
    cfg = replace(CFG, window_ms=150.0)
    torch.manual_seed(0)
    model = WearUsFM(cfg).eval()
    assert model.tokenizer.window_s == pytest.approx(0.150) and model.tokenizer.patch_s == 0.025
    from wearusfm.harness import fm_features as FF

    w = np.random.default_rng(0).normal(size=(3, 1000, 8))
    f = FF.extract_features(model, w, 200.0, FF.myo8_montage("epn612", 200.0), np.array(["a", "a", "a"]), batch=2)
    assert f.shape == (3, 32) and np.isfinite(f).all()
    # la perdita JEPA conta solo i token interamente nascosti: con una maschera di 2 patch nessuna query
    from wearusfm.model import channel_codes as CC
    from wearusfm.model.channel_identity import codes_to_tensors
    from wearusfm.model.fm import ModelInputs
    from wearusfm.model.local_encoder import sets_to_tensors
    from wearusfm.training.jepa import JEPAConfig, jepa_losses, make_teacher

    montage = FF.myo8_montage("epn612", 200.0)
    layout, code = CC.layout_from_montage(montage), CC.anatomy_codes(montage)
    sets = CC.attention_sets(layout, 8)
    sig = [torch.from_numpy(np.ascontiguousarray(w[0].T).astype(np.float32))]
    inp = ModelInputs(sig, [200.0], codes_to_tensors(CC.pack_codes([code])), sets_to_tensors(CC.pack_attention_sets([sets])), [8],
                      torch.as_tensor(layout.qc_valid))
    teacher = make_teacher(model)
    jcfg = JEPAConfig(target="b", anchor_weight=0.0, ema_momentum=0.996)
    visible = torch.ones(8, 200, dtype=torch.bool)
    visible[:, 100:102] = False  # 2 patch su tutti i canali: nessun token interamente nascosto con raggio 3
    with pytest.raises(ValueError, match="nessun token nascosto"):
        jepa_losses(model, teacher, inp, visible, jcfg)
    visible[:, 100:120] = False  # 20 patch: i token 103..116 sono interamente nascosti
    losses = jepa_losses(model, teacher, inp, visible, jcfg)
    assert torch.isfinite(losses["jepa"]) and losses["core_hidden_fraction"] == pytest.approx(14 / 20)


def test_trainer_flags_for_target_a_and_window():
    sp = importlib.util.spec_from_file_location("train_jepa_window", REPO / "scripts" / "train_jepa.py")
    TJ = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(TJ)
    base = ["--preset", "sanity", "--manifest", "-", "--root", "-", "--out-dir", "-"]
    cfg = TJ.build_config(TJ.build_parser().parse_args(base))
    assert cfg.jepa.target == "b" and cfg.model.window_ms is None  # i default del preset, invariati
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--jepa-target", "a", "--window-ms", "150"]))
    assert cfg.jepa.target == "a" and cfg.model.window_ms == 150.0 and cfg.model.core_radius == 3
    with pytest.raises(ValueError, match="--window-ms"):
        TJ.build_config(TJ.build_parser().parse_args(base + ["--window-ms", "25"]))
    # R2 e lo scostamento da D10 di R5: target per regione e quote delle maschere; default intoccati
    assert cfg.jepa.region_target is False and (cfg.loader.mask.share_short, cfg.loader.mask.share_spatial) == (0.15, 0.40)
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--jepa-region-target"]))
    assert cfg.jepa.region_target is True and cfg.loader.mask.share_short == 0.15
    cfg = TJ.build_config(TJ.build_parser().parse_args(base + ["--window-ms", "150", "--mask-shares", "0:0.294:0.235:0.471"]))
    m = cfg.loader.mask
    assert (m.share_short, m.share_medium, m.share_long, m.share_spatial) == (0.0, 0.294, 0.235, 0.471) and m.ratio == 0.5
    for bad in ("0.5:0.5", "0.1:0.2:0.3:0.5", "a:b:c:d"):
        with pytest.raises(ValueError, match="--mask-shares"):
            TJ.build_config(TJ.build_parser().parse_args(base + ["--mask-shares", bad]))
