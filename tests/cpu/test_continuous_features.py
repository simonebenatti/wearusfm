import numpy as np
import pytest

from wearusfm.tokenizer_checks import continuous_features as C
from wearusfm.tokenizer_checks import metrics as M


def _fake_fn(scale=1.0):
    """Modello finto: 4 rami, 128 feature = proiezione fissa della potenza di 8 sotto-bande del token; deterministico."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 8, 128)).astype(np.float32)

    def fn(windows):
        toks = windows.reshape(-1, 200)
        p = np.log10(np.abs(np.fft.rfft(toks, axis=-1))[:, 1:97].reshape(len(toks), 8, 12).mean(-1) ** 2 + 1e-6).astype(np.float32)
        return np.stack([p @ w[k] * scale for k in range(4)])

    return fn


def _tokens(n, rng, amp=1.0):
    return (rng.normal(size=(n, 200)) * amp).astype(np.float32)


def test_token_features_shape_order_and_input_checks():
    rng = np.random.default_rng(0)
    tok = _tokens(64, rng)
    f = C.token_features(_fake_fn(), tok, batch=3)  # batch che non divide: 4 finestre
    assert f.shape == (4, 64, 128)
    assert np.allclose(f, C.token_features(_fake_fn(), tok, batch=32))  # il raggruppamento non cambia il risultato
    with pytest.raises(ValueError):
        C.token_features(_fake_fn(), tok[:60])
    with pytest.raises(ValueError):
        C.token_features(lambda w: np.zeros((4, 3, 128)), tok)  # feature_fn con forma sbagliata


def test_pool_units_is_mean_then_branch_concat():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(4, 512, 128)).astype(np.float32)
    p = C.pool_units(f)
    assert p.shape == (2, 512)
    assert np.allclose(p[1, 128:256], f[1, 256:512].mean(axis=0))  # unita' 1, ramo 1
    ps = C.pool_units(f, with_std=True)
    assert ps.shape == (2, 1024) and np.allclose(ps[0, 512 : 512 + 128], f[0, :256].std(axis=0), atol=1e-6)
    with pytest.raises(ValueError):
        C.pool_units(f[:, :300])


def test_standardize_uses_only_train_statistics():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [100.0, 5.0], [200.0, 5.0]])
    part = np.array(["train", "train", "test", "val"])
    z = C.standardize_with_train(x, part)
    assert np.allclose(z[:2, 0], [-1.0, 1.0]) and np.allclose(z[:, 1], 0.0)  # costante: zero, non NaN
    assert z[2, 0] > 50  # test non influenza media e std


def _synthetic_run(rng, n_groups=8, datasets=("a", "b", "c")):
    per = {}
    for di, name in enumerate(datasets):
        n = n_groups * 256
        per[name] = {
            "codes": rng.integers(0, M.N_CODE, size=(4, 16, n)).astype(np.int32),
            "tokens": _tokens(n, rng, amp=1.0 + di),  # ampiezza diversa per dataset: identificabile
            "group_subject": np.array([f"s{g % 4}" for g in range(n_groups)]),
        }
    return per


def test_v3_setup_matches_run_v3_split():
    """Lo split ricostruito dagli array e' quello di pipeline.run_v3: stessi rng, soggetti disgiunti fra le parti."""
    rng = np.random.default_rng(0)
    per = _synthetic_run(rng)
    names, codes_l, tokens_l, y, units, part = C.v3_setup(per, seed=0)
    assert names == ["a", "b", "c"] and len(codes_l) == len(tokens_l) == len(y) == 24
    assert codes_l[0].shape == (4, 16, 256) and tokens_l[0].shape == (256, 200)
    by_subject = {}
    for u, p in zip(units, part):
        by_subject.setdefault(u, set()).add(p)
    assert all(len(v) == 1 for v in by_subject.values())  # un soggetto sta in una sola parte
    part_map = M.split_subjects({n: sorted(set(per[n]["group_subject"].tolist())) for n in names}, np.random.default_rng([0, 3, 0]))
    assert all(part_map[(names[yy], u.split("/", 1)[1])] == p for yy, u, p in zip(y, units, part))


def test_dataset_id_probe_continuous_runs_and_finds_planted_signal():
    rng = np.random.default_rng(0)
    per = _synthetic_run(rng, n_groups=12)
    feats = {n: C.token_features(_fake_fn(), per[n]["tokens"], batch=64) for n in per}
    r = C.dataset_id_probe_continuous(per, feats, seed=0)
    assert r["datasets"] == ["a", "b", "c"] and r["n_samples"] == 36
    for k in ("frozen_bands", "frozen_codes", "extra_continuous_mean", "extra_continuous_mean_std", "extra_continuous_mean_branch3"):
        assert 0.0 <= r[k]["balanced_accuracy"] <= 1.0 and r[k]["chance"] == pytest.approx(1 / 3)
    assert r["extra_continuous_mean"]["balanced_accuracy"] > 0.9  # l'ampiezza per dataset e' nelle feature (potenza)
    assert r["frozen_codes"]["balanced_accuracy"] < 0.9  # i codici finti sono casuali


def test_cosine_rows():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    assert np.allclose(C.cosine_rows(a, a), 1.0)
    assert np.allclose(C.cosine_rows(a, -a), -1.0)
    assert np.allclose(C.cosine_rows(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])), 0.0)


def test_stability_more_noise_moves_features_more_and_reference_is_lower():
    rng = np.random.default_rng(0)
    tok = _tokens(32 * 16, rng)
    r = C.stability_continuous(_fake_fn(), tok, noise_floor_rms=1.0, n_windows=20, seed=0)
    assert r["n_windows"] == 20 and set(r["noise"]) == {"1x_floor", "0.1x_floor"}
    hi, lo = r["noise"]["1x_floor"]["median_per_branch"], r["noise"]["0.1x_floor"]["median_per_branch"]
    assert all(l > h for l, h in zip(lo, hi)) and all(l > 0.9 for l in lo)  # 0,1 x soglia: quasi invariate
    assert all(q <= m for q, m in zip(r["noise"]["1x_floor"]["q05_per_branch"], hi))
    assert set(r["between_different_tokens"]) == {"median_per_branch", "q95_per_branch"}
    # riproducibile: stesso seed, stesso risultato
    assert r == C.stability_continuous(_fake_fn(), tok, noise_floor_rms=1.0, n_windows=20, seed=0)
