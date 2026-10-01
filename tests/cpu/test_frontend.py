"""Front-end a kernel continui (v10 §4.2). Richiede torch: si salta se non c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.model.frontend import ContinuousKernelFrontEnd  # noqa: E402


def _sig(fs, dur, freqs, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(dur * fs))) / fs
    amps, ph = rng.normal(size=len(freqs)), rng.uniform(0, 2 * np.pi, len(freqs))
    return torch.tensor(sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, ph)))[None, None, :]


def _rel(a, b, s=slice(None)):
    return float(torch.linalg.norm(a[..., s] - b[..., s]) / torch.linalg.norm(a[..., s]))


@pytest.fixture(scope="module")
def fe():
    torch.set_grad_enabled(False)
    return ContinuousKernelFrontEnd(seed=0)


def test_shapes_and_families(fe):
    out = fe(torch.zeros(2, 3, 2000, dtype=torch.float64), 2000.0)
    assert out.shape == (2, 3, 40, fe.d_out) and fe.d_out == 64  # 1 s / 25 ms = 40 patch
    sl = fe.family_slices()
    assert [sl[f].stop - sl[f].start for f in ("fourier", "spline", "mlp")] == [24, 24, 16]
    with pytest.raises(ValueError):
        fe(torch.zeros(1, 1, 10, dtype=torch.float64), 2000.0)


@pytest.mark.parametrize("fs_b, fmax", [(1000, 440), (200, 88)])
def test_same_features_on_different_grids(fe, fs_b, fmax):
    """Lo stesso segnale a banda limitata, a 2 kHz e a fs_b: stesse feature, per famiglia, sulle patch lontane dai bordi."""
    fr = np.linspace(5, fmax, 40)
    a = fe(_sig(2000, 2.0, fr), 2000.0)[:, :, 5:-5]
    b = fe(_sig(fs_b, 2.0, fr), float(fs_b))[:, :, 5:-5]
    for name, s in fe.family_slices().items():
        assert _rel(a, b, s) < 0.01, name


def test_non_integer_samples_per_patch(fe):
    fr = np.linspace(5, 440, 40)
    a = fe(_sig(2000, 2.0, fr), 2000.0)[:, :, 5:-5]
    b = fe(_sig(2048, 2.0, fr), 2048.0)[:, :, 5:-5]  # 2048 Hz x 25 ms = 51,2 campioni per patch
    n = min(a.shape[2], b.shape[2])
    assert _rel(a[:, :, :n], b[:, :, :n]) < 0.01


def test_delta_t_factor_is_what_makes_amplitude_rate_independent(fe):
    """Senza il fattore Δt l'uscita a 2 kHz sarebbe il doppio di quella a 1 kHz (identificatore di dataset gratuito, v10 §4.2)."""
    fr = np.linspace(5, 440, 40)
    a = fe(_sig(2000, 2.0, fr), 2000.0)[:, :, 5:-5]
    b = fe(_sig(1000, 2.0, fr), 1000.0)[:, :, 5:-5]
    ratio = float(torch.linalg.norm(a * 2000.0) / torch.linalg.norm(b * 1000.0))  # le somme senza Δt
    assert ratio == pytest.approx(2.0, rel=0.02)
    assert _rel(a, b) < 0.01  # con Δt: uguali


def test_anti_aliasing_removes_content_above_nyquist(fe):
    """Il kernel valutato per fs = 200 Hz non contiene energia sopra 100 Hz, per ciascuna famiglia."""
    t = torch.arange(-0.3, 0.325, 1 / 16000.0, dtype=torch.float64)
    k = fe.kernels_at(t, 200.0)  # (d, n_t) su una griglia fine
    spec = torch.fft.rfft(k, dim=1).abs() ** 2
    f = torch.fft.rfftfreq(k.shape[1], d=1 / 16000.0)
    for name, s in fe.family_slices().items():
        frac = float(spec[s][:, f > 100.0].sum() / spec[s].sum())
        assert frac < 1e-6, (name, frac)
    k2 = fe.kernels_at(t, 2000.0)  # a 2 kHz il kernel conserva contenuto fra 100 e 900 Hz
    spec2 = torch.fft.rfft(k2, dim=1).abs() ** 2
    assert float(spec2[:, (f > 100.0) & (f < 900.0)].sum() / spec2.sum()) > 0.05


def test_seed_controls_initialisation():
    x = torch.randn(1, 1, 4000, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b, c = (ContinuousKernelFrontEnd(seed=s)(x, 2000.0) for s in (0, 0, 1))
    assert torch.equal(a, b) and not torch.allclose(a, c)
