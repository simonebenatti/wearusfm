import numpy as np

from wearusfm.harness.protocol import run_hudgins_lda_protocol


def _make_synthetic_dataset(rng, n_subjects=12, n_classes=3, per_subject_per_class=15, t=200, c=4):
    """Soggetti sintetici con un piccolo offset di ampiezza per soggetto (variabilita'
    fra soggetti, realistica) ma la classe resta il segnale dominante e separabile."""
    windows, labels, subjects = [], [], []
    for s in range(n_subjects):
        subject_id = f"synth_s{s:02d}"
        subject_offset = rng.uniform(0.8, 1.2)  # variabilita' fra soggetti
        for k in range(n_classes):
            amplitude = (1.0 + 3.0 * k) * subject_offset
            for _ in range(per_subject_per_class):
                w = rng.normal(loc=0.0, scale=amplitude, size=(t, c))
                windows.append(w)
                labels.append(k)
                subjects.append(subject_id)
    return np.stack(windows), np.array(labels), np.array(subjects)


def test_protocol_generalizes_to_held_out_subjects():
    rng = np.random.default_rng(0)
    windows, labels, subjects = _make_synthetic_dataset(rng)

    result = run_hudgins_lda_protocol(windows, labels, subjects, seed=0)

    assert result.n_train_windows > 0
    assert result.n_test_windows > 0
    assert result.test_balanced_accuracy > 0.7  # sopra il caso (1/3), su soggetti mai visti
    # nessun soggetto in piu' di uno split (SubjectSplit lo garantisce gia' da solo,
    # ma lo ri-verifichiamo qui come proprieta' del protocollo end-to-end)
    train_set = set(result.split.train)
    test_set = set(result.split.test)
    assert not (train_set & test_set)


def test_protocol_is_deterministic_given_seed():
    rng = np.random.default_rng(1)
    windows, labels, subjects = _make_synthetic_dataset(rng)
    r1 = run_hudgins_lda_protocol(windows, labels, subjects, seed=7)
    r2 = run_hudgins_lda_protocol(windows, labels, subjects, seed=7)
    assert r1.split == r2.split
    assert r1.test_balanced_accuracy == r2.test_balanced_accuracy


def test_protocol_accepts_precomputed_split():
    from wearusfm.harness.splits import split_subjects

    rng = np.random.default_rng(2)
    windows, labels, subjects = _make_synthetic_dataset(rng)
    fixed_split = split_subjects(np.unique(subjects), seed=99)

    result = run_hudgins_lda_protocol(windows, labels, subjects, split=fixed_split)
    assert result.split == fixed_split
