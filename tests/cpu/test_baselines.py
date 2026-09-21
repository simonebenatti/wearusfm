import numpy as np

from wearusfm.harness.baselines import HudginsLDABaseline


def _make_separable_windows(rng, n_per_class=40, n_classes=3, t=200, c=4):
    windows, labels = [], []
    for k in range(n_classes):
        amplitude = 1.0 + 3.0 * k  # classi con ampiezza ben distinta -> MAV separabile
        for _ in range(n_per_class):
            w = rng.normal(loc=0.0, scale=amplitude, size=(t, c))
            windows.append(w)
            labels.append(k)
    windows = np.stack(windows)
    labels = np.array(labels)
    idx = rng.permutation(len(labels))
    return windows[idx], labels[idx]


def test_hudgins_lda_learns_separable_classes():
    rng = np.random.default_rng(0)
    windows, labels = _make_separable_windows(rng)
    n_train = int(0.8 * len(labels))
    model = HudginsLDABaseline().fit(windows[:n_train], labels[:n_train])
    acc = model.evaluate(windows[n_train:], labels[n_train:])
    assert acc > 0.8  # ben sopra il caso (1/3 per 3 classi)


def test_hudgins_lda_predict_before_fit_raises():
    model = HudginsLDABaseline()
    try:
        model.predict(np.zeros((1, 10, 2)))
        assert False, "doveva sollevare RuntimeError"
    except RuntimeError:
        pass
