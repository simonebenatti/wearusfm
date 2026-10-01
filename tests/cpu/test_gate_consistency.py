"""Gate D8 su dati sintetici: con il front-end vero deve passare; con un front-end rotto (senza Δt) deve fallire. Richiede torch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.gate import consistency as G  # noqa: E402
from wearusfm.model.frontend import ContinuousKernelFrontEnd  # noqa: E402


def _sessions(n_subjects=6, seconds=30, n_ch=2, seed=0):
    """Rumore filtrato fra 20 e 400 Hz a 2 kHz, ampiezza diversa per soggetto."""
    from scipy import signal

    rng = np.random.default_rng(seed)
    sos = signal.butter(4, [20, 400], btype="band", fs=2000, output="sos")
    out = []
    for i in range(n_subjects):
        x = signal.sosfilt(sos, rng.normal(size=(int(seconds * 2000), n_ch)), axis=0) * (1 + i)
        out.append((f"s{i}", x, []))
    return out


def test_make_versions_same_content_on_two_grids():
    x = np.random.default_rng(0).normal(size=(2, 4000))
    a, b = G.make_versions(x, 90.0, 10)
    assert a.shape == (2, 4000) and b.shape == (2, 400) and np.array_equal(a[:, ::10], b)


def test_draw_windows_alignment_and_avoid():
    s = [("s0", np.zeros((60000, 2)), [(0, 30000)])]  # dopo il buco restano 30000 campioni: c'e' posto per finestre da 10000
    w = G.draw_windows(s, 3, np.random.default_rng(0))
    total = int((G.WINDOW_S + 2 * G.MARGIN_S) * G.FS_A)
    assert len(w) == 3 and all(x.data.shape == (2, total) for x in w)
    w2 = G.draw_windows([("s0", np.arange(80000.0)[:, None].repeat(2, 1), [(0, 30000)])], 5, np.random.default_rng(1))
    assert all(x.data[0, 0] >= 30000 and x.data[0, 0] % 10 == 0 for x in w2)  # dopo il buco e allineate a 10 campioni


def test_features_case_real_frontend_vs_frontend_without_delta_t():
    torch.set_grad_enabled(False)
    windows = G.draw_windows(_sessions(), 4, np.random.default_rng(0))
    ok = G.features_case(ContinuousKernelFrontEnd(seed=0), windows, 450.0, 2)
    assert ok["rel_error"]["total"] < 0.01 and all(v < 0.01 for v in ok["rel_error"].values())
    assert ok["x"].shape[0] == 2 * len(windows) and set(ok["y"].tolist()) == {0, 1}

    class NoDeltaT(ContinuousKernelFrontEnd):
        def forward(self, x, fs):
            return super().forward(x, fs) * fs  # toglie il fattore Δt

    bad = G.features_case(NoDeltaT(seed=0), windows, 450.0, 2)
    assert bad["rel_error"]["total"] > 0.4
