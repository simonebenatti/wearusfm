import numpy as np
import pytest
from scipy.signal import periodogram

from wearusfm.model.anchor_targets import _band_logpower
from wearusfm.model.spectral_targets import hann_periodogram


@pytest.mark.parametrize("n", [49, 50, 499, 500])
@pytest.mark.parametrize("dtype,rtol", [(np.float32, 5e-6), (np.float64, 1e-12)])
def test_psd_matches_scipy(n, dtype, rtol):
    x = np.random.default_rng(10).normal(size=(2, 3, n)).astype(dtype)
    actual, freqs = hann_periodogram(x, 1000.)
    f, expected = periodogram(x, fs=1000., window=np.hanning(n).astype(dtype), detrend=False, scaling="density")
    np.testing.assert_array_equal(freqs, f)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-10)


@pytest.mark.parametrize("n", [49, 50])
def test_parseval_including_dc_and_nyquist(n):
    x = np.random.default_rng(8).normal(size=(4, n)) + 3
    psd, _ = hann_periodogram(x, 1000.)
    w = np.hanning(n)
    np.testing.assert_allclose(psd.sum(-1) * 1000/n, (x*x*w*w).sum(-1)/(w*w).sum(), rtol=1e-12)


@pytest.mark.parametrize("win_ms,n_bands", [(50., 6), (500., 24)])
def test_same_tone_sampling_rate_and_gain(win_ms, n_bands):
    powers = []
    for fs in [1000., 2000.]:
        t = np.arange(int(2*fs))/fs
        x = np.sin(2*np.pi*160*t)[None, :].astype(np.float32)
        y, ok, avail = _band_logpower(x, fs, np.array([1.]), win_ms, n_bands, np.array([450.]))
        z, _, _ = _band_logpower(x*3, fs, np.array([1.]), win_ms, n_bands, np.array([450.]))
        p = (np.exp(y)-1e-8)[..., avail[0]].sum()
        q = (np.exp(z)-1e-8)[..., avail[0]].sum()
        np.testing.assert_allclose(q/p, 9, rtol=2e-6)
        powers.append(p)
        assert ok.all()
    np.testing.assert_allclose(powers, [.5, .5], rtol=.002)
    np.testing.assert_allclose(powers[0], powers[1], rtol=.002)


def test_missing_bands_and_boundaries_unchanged():
    _, ok, avail = _band_logpower(np.ones((1, 400), np.float32), 200., np.array([0., 1., 2.]), 50., 6, np.array([100.]))
    assert ok.tolist() == [False, True, False]
    assert not avail[0, -1]


@pytest.mark.parametrize("n", [0, 1, 2])
def test_degenerate_hann_rejected(n):
    with pytest.raises(ValueError):
        hann_periodogram(np.ones(n), 1000.)


def test_old_multiscale_resume_rejected_without_writes(tmp_path):
    import torch
    from wearusfm.training import run as R
    out = tmp_path / "old_run"
    out.mkdir()
    path = out / "checkpoint.pt"
    torch.save({"config": {}}, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="NEW run"):
        R.train(R.with_window1_rules(R.small_config()), tmp_path / "absent", [], out)
    assert path.read_bytes() == before
    assert list(out.iterdir()) == [path]
