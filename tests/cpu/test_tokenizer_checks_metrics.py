import numpy as np
import pytest

from wearusfm.tokenizer_checks import metrics as m


def test_token_nmse_perfect_and_zero_reconstruction():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 200))
    assert np.allclose(m.token_nmse(x, x), 0.0)
    assert np.allclose(m.token_nmse(x, np.zeros_like(x)), 1.0)


def test_token_nmse_rejects_bad_inputs():
    with pytest.raises(ValueError):
        m.token_nmse(np.ones((2, 200)), np.ones((3, 200)))
    with pytest.raises(ValueError):
        m.token_nmse(np.zeros((2, 200)), np.zeros((2, 200)))


def test_v2_threshold_is_x_equals_2_inclusive():
    ref = np.array([0.1, 0.1, 0.1])
    assert m.v2_passes(m.median_ratio(np.array([0.2, 0.2, 0.2]), ref))
    assert not m.v2_passes(m.median_ratio(np.array([0.21, 0.21, 0.21]), ref))


def test_v1_threshold_is_1p5_inclusive():
    multi = np.array([0.2, 0.2, 0.2])
    assert m.v1_passes(np.array([0.3, 0.3, 0.3]), multi)
    assert not m.v1_passes(np.array([0.31, 0.31, 0.31]), multi)


def test_noise_floor_is_median_rms_of_lowest_decile():
    scales = np.linspace(0.1, 1.0, 100)  # 100 token, RMS = scala
    tokens = np.stack([np.full(200, s) for s in scales])
    floor = m.noise_floor_rms(tokens)
    low = scales[scales <= np.quantile(scales, 0.10)]
    assert floor == pytest.approx(np.median(low))
    assert floor < 0.25


def test_code_stability_fractions():
    clean = np.zeros((4, 16, 100), dtype=int)
    noisy = clean.copy()
    noisy[:, 10:, :20] = 1  # dal livello 10 in su cambia il 20%
    f = m.code_stability(clean, noisy)
    assert f.shape == (4, 16)
    assert np.allclose(f[:, :10], 1.0) and np.allclose(f[:, 10:], 0.8)


def test_stable_levels_needs_all_branches_and_all_datasets_and_boundary_inclusive():
    a = np.full((4, 16), 0.9)
    b = np.full((4, 16), 0.9)
    a[2, 5] = 0.74  # un solo ramo sotto soglia: il livello 5 non e' stabile su 'a'
    b[:, 7] = 0.75  # esattamente al limite: stabile
    b[0, 9] = 0.2
    per, overall = m.stable_levels({"a": a, "b": b})
    assert not per["a"][5] and per["a"][7] and not per["b"][9]
    assert not overall[5] and overall[7] and not overall[9] and overall[0]
    assert m.v4_passes(overall)
    assert not m.v4_passes(np.zeros(16, dtype=bool))


def test_code_histogram_features_shape_normalization_and_duplicates():
    codes = np.zeros((4, 16, 8), dtype=int)  # tutti i token con codice 0
    codes[:, :, 4:] = 7
    X = m.code_histogram_features([codes, codes])
    assert X.shape == (2, 4 * 16 * 8192)
    assert X.sum(axis=1).A.ravel() == pytest.approx([64.0, 64.0])  # 64 coppie, ciascuna somma 1
    assert X[0, 0] == pytest.approx(0.5) and X[0, 7] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        m.code_histogram_features([np.full((4, 16, 3), 8192)])
    with pytest.raises(ValueError):
        m.code_histogram_features([np.zeros((3, 16, 3), dtype=int)])


def test_band_power_features_put_tone_in_right_band():
    t = np.arange(200) / 1000.0
    tok100 = np.tile(np.sin(2 * np.pi * 100 * t), (16, 1))
    tok300 = np.tile(np.sin(2 * np.pi * 300 * t), (16, 1))
    f = m.band_power_features([tok100, tok300])
    assert f.shape == (2, 5)
    assert np.argmax(f[0]) == 1  # 60-125 Hz
    assert np.argmax(f[1]) == 4  # 250-400 Hz


def test_split_subjects_disjoint_and_nonempty():
    rng = np.random.default_rng(0)
    sp = m.split_subjects({"d1": list(range(10)), "d2": list(range(100, 105))}, rng)
    for d, subs in {"d1": range(10), "d2": range(100, 105)}.items():
        parts = [sp[(d, s)] for s in subs]
        assert set(parts) == {"train", "val", "test"}
    with pytest.raises(ValueError):
        m.split_subjects({"tiny": [1, 2]}, rng)


def test_balance_classes_equalizes_counts():
    rng = np.random.default_rng(0)
    y = np.array([0] * 10 + [1] * 4 + [2] * 7)
    keep = m.balance_classes(y, rng)
    assert np.bincount(y[keep]).tolist() == [4, 4, 4]


def _make_units(n_per_unit=10, n_units=6, classes=(0, 1, 2)):
    y, units, part = [], [], []
    plan = ["train", "train", "train", "val", "test", "test"]
    for c in classes:
        for u in range(n_units):
            y += [c] * n_per_unit
            units += [f"{c}-{u}"] * n_per_unit
            part += [plan[u]] * n_per_unit
    return np.array(y), np.array(units), np.array(part)


def test_probe_finds_separable_datasets_and_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y, units, part = _make_units()
    X = rng.normal(size=(len(y), 5)) + y[:, None] * 3.0
    r = m.dataset_id_probe(X, y, units, part, rng)
    assert r.balanced_accuracy > 0.95 and r.chance == pytest.approx(1 / 3)
    assert r.ci95[0] <= r.balanced_accuracy <= r.ci95[1]


def test_probe_is_near_chance_when_features_carry_no_dataset_information():
    rng = np.random.default_rng(1)
    y, units, part = _make_units(n_per_unit=20)
    X = rng.normal(size=(len(y), 5))
    r = m.dataset_id_probe(X, y, units, part, rng)
    assert r.balanced_accuracy < 0.55


def test_probe_works_on_sparse_code_histograms():
    rng = np.random.default_rng(2)
    y, units, part = _make_units(n_per_unit=8)
    samples = []
    for c in y:  # ogni dataset usa un proprio sottoinsieme di codici nel livello 0 del ramo 0
        codes = rng.integers(0, 8192, size=(4, 16, 32))
        codes[0, 0] = rng.integers(c * 100, c * 100 + 50, size=32)
        samples.append(codes)
    X = m.code_histogram_features(samples)
    r = m.dataset_id_probe(X, y, units, part, rng, top_k_columns=300)
    assert r.balanced_accuracy > 0.9


def test_v3_threshold_is_10_points_inclusive():
    assert m.v3_passes(0.60, 0.50)
    assert not m.v3_passes(0.61, 0.50)
