"""Feature congelate del nostro FM per la sonda (passo 5). Richiede torch: si salta se non c'e'."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.harness import fm_features as F  # noqa: E402
from wearusfm.model.fm import FMConfig, WearUsFM  # noqa: E402

CFG = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4)


def _windows(n=6, t=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, t, 8)) * np.linspace(0.5, 2.0, 8)[None, None, :], np.array(["a", "a", "a", "b", "b", "b"][:n])


def test_features_shape_determinism_and_batch_independence():
    torch.manual_seed(0)
    model = WearUsFM(CFG)
    m = F.myo8_montage("epn612", 200.0)
    w, s = _windows()
    f1 = F.extract_features(model, w, 200.0, m, s, batch=4)
    f2 = F.extract_features(model, w, 200.0, m, s, batch=2)
    assert f1.shape == (6, 32) and np.isfinite(f1).all() and np.allclose(f1, f2, atol=1e-5)  # non dipende dal batch
    assert not np.allclose(f1[0], f1[1])  # finestre diverse, feature diverse
    assert not model.training


def test_long_windows_are_cropped_and_scale_is_per_subject():
    torch.manual_seed(0)
    model = WearUsFM(CFG)
    m = F.myo8_montage("epn612", 200.0)
    w, s = _windows(t=1000)  # 5 s a 200 Hz: due ritagli da 4 s
    assert F._crops(1000, 200.0, 4.0) == [(0, 800), (200, 1000)] and F._crops(600, 200.0, 4.0) == [(0, 600)]
    sc = F.subject_scales(w, s, 200.0, (20.0, 450.0), scale_windows=2)
    assert set(sc) == {"a", "b"} and all(v > 0 for v in sc.values())
    w2 = w.copy()
    w2[3:] *= 10.0  # il soggetto b registrato con un guadagno diverso: la scala di soggetto lo assorbe
    assert np.allclose(F.extract_features(model, w, 200.0, m, s)[3:], F.extract_features(model, w2, 200.0, m, s)[3:], atol=1e-4)


def test_load_model_from_a_training_checkpoint(tmp_path):
    from wearusfm.training import run as R

    from test_pretraining_loader import _tree

    root, mpath = _tree(tmp_path)
    R.train(R.small_config(datasets=None, max_steps=1), mpath, [root], tmp_path / "r", log=lambda msg: None)
    model = F.load_model(str(tmp_path / "r" / "checkpoint.pt"))
    w, s = _windows()
    assert F.extract_features(model, w, 200.0, F.myo8_montage("x", 200.0), s).shape == (6, 2 * model.cfg.dim)
