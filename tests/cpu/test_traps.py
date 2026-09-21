import numpy as np

from wearusfm.harness.traps import label_run_lengths, warn_if_rest_looks_like_padding


def test_label_run_lengths_finds_all_runs():
    labels = np.array(["g1", "rest", "rest", "g2", "rest", "g1", "g1", "g1"])
    runs = label_run_lengths(labels, "rest")
    assert sorted(runs.tolist()) == [1, 2]


def test_label_run_lengths_empty_when_label_absent():
    labels = np.array(["g1", "g2", "g1"])
    runs = label_run_lengths(labels, "rest")
    assert runs.size == 0


def test_warn_if_rest_looks_like_padding_flags_short_uniform_runs():
    # "rest" appare sempre per 2 campioni esatti fra un gesto e l'altro: tipico pattern
    # di padding automatico, non riposo genuino registrato
    labels = []
    for _ in range(20):
        labels += ["g1"] * 10 + ["rest"] * 2
    warnings = warn_if_rest_looks_like_padding(
        np.array(labels), "rest", short_run_threshold_samples=5
    )
    assert len(warnings) == 1
    assert "padding" in warnings[0]


def test_warn_if_rest_looks_like_padding_silent_for_genuine_long_rest():
    labels = []
    for _ in range(10):
        labels += ["g1"] * 10 + ["rest"] * 200  # riposo lungo, plausibilmente genuino
    warnings = warn_if_rest_looks_like_padding(
        np.array(labels), "rest", short_run_threshold_samples=5
    )
    assert warnings == []


def test_warn_if_rest_looks_like_padding_silent_when_label_absent():
    labels = np.array(["g1", "g2", "g1"])
    warnings = warn_if_rest_looks_like_padding(labels, "rest", short_run_threshold_samples=5)
    assert warnings == []
