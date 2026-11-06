"""PSD3 pairing and uncertainty, checked on explicit synthetic predictions."""

from copy import deepcopy
from itertools import product
import json

import numpy as np
import pytest

from wearusfm.harness import psd3_compare as C


def _shards():
    y = [0, 1, 2] * 6
    subjects = [f"s{i // 6}" for i in range(len(y))]
    out = []
    for model, ds, view in product(C.MODELS, C.DATASETS, C.VIEWS):
        out.append({
            "status": "completed", "protocol": "psd3_test_v1", "recipe_sha256": "a" * 64,
            "model": model, "dataset": ds, "view": view,
            "input_sha256": "b" * 64, "roles_sha256": "c" * 64,
            "checkpoint_sha256": ("d" if model == "B" else "e") * 64,
            "rows": list(range(len(y))), "y": y.copy(), "subjects": subjects.copy(),
            "predictions": {"logistic": {"0": y.copy()}, "mlp": {"0": y.copy(), "1": y.copy()}},
            "additional_metadata": {"ignored": True},
        })
    return out


def _set_arrays(shards, y, subjects, pred):
    for shard in shards:
        shard.update(rows=list(range(len(y))), y=list(y), subjects=list(subjects))
        shard["predictions"] = {reader: {seed: list(pred) for seed in seeds}
                                for reader, seeds in C.READOUT_SEEDS.items()}


def _all_summaries(result):
    return list(result["datasets"].values()) + list(result["aggregates"].values())


def test_perfect_grid_and_zero_contrasts():
    result = C.compare_shards(_shards(), n_boot=40)
    assert result["descriptive_only"] is True
    assert result["identity"]["n_shards"] == 20
    for summary in _all_summaries(result):
        assert len(summary["cells"]) == 8
        assert len(summary["contrasts"]) == 14
        for cell in summary["cells"].values():
            assert cell["bacc"] == pytest.approx(1)
            assert cell["ci95"] == pytest.approx([1, 1])
        for contrast in summary["contrasts"].values():
            assert contrast["delta_pp"] == pytest.approx(0)
            assert contrast["ci95_pp"] == pytest.approx([0, 0])
    json.dumps(result, allow_nan=False)


def test_mlp_is_mean_of_seed_scores_not_voting_or_best_seed():
    shards = _shards()
    for shard in shards:
        # Seed scores are 2/3 and 1/3. Breaking a voting tie by the smaller
        # predicted class would score 2/3; selecting the best seed also 2/3.
        shard["predictions"]["mlp"] = {
            "0": [[0, 2, 2][y] for y in shard["y"]],
            "1": [[1, 1, 0][y] for y in shard["y"]],
        }
    result = C.compare_shards(shards, n_boot=40)
    for summary in _all_summaries(result):
        for model, view in product(C.MODELS, C.VIEWS):
            cell = summary["cells"][f"{model}/{view}/mlp"]
            assert cell["by_readout_seed"] == pytest.approx({"0": 2 / 3, "1": 1 / 3})
            assert cell["bacc"] == pytest.approx(0.5)
            assert cell["ci95"] == pytest.approx([0.5, 0.5])
            assert summary["contrasts"][f"mlp_minus_logistic/{model}/{view}"]["delta_pp"] == pytest.approx(-50)


def test_pairing_cancels_subject_uncertainty_across_all_cells():
    shards = _shards()
    for shard in shards:
        pred = [y if s == "s0" else (y + 1) % 3 for y, s in zip(shard["y"], shard["subjects"])]
        shard["predictions"] = {reader: {seed: pred.copy() for seed in seeds}
                                for reader, seeds in C.READOUT_SEEDS.items()}
    result = C.compare_shards(shards, n_boot=150)
    # Each cell has sampling uncertainty, but identical paired predictions do not.
    assert result["datasets"][C.DATASETS[0]]["cells"]["B/pool/logistic"]["ci95"] == pytest.approx([0, 1])
    for summary in _all_summaries(result):
        for contrast in summary["contrasts"].values():
            assert contrast["ci95_pp"] == pytest.approx([0, 0])
            assert contrast["paired_subject_se_pp"] == pytest.approx(0)


def test_balanced_accuracy_pools_class_counts_not_subject_scores():
    shards = _shards()
    _set_arrays(shards, [0, 0, 0, 1, 0, 1], ["a"] * 4 + ["b"] * 2, [0, 0, 0, 1, 1, 0])
    result = C.compare_shards(shards, n_boot=40)
    # Mean subject BAcc would be .5; pooled class recalls are .75 and .5.
    assert result["datasets"][C.DATASETS[0]]["cells"]["B/pool/logistic"]["bacc"] == pytest.approx(0.625)


def test_fixed_dataset_weights_and_percentage_point_units():
    shards = _shards()
    for shard in shards:
        correct = shard["model"] == "B" and shard["dataset"] in ("ninapro_db2", "epn612")
        pred = shard["y"] if correct else [(y + 1) % 3 for y in shard["y"]]
        shard["predictions"] = {reader: {seed: pred.copy() for seed in seeds}
                                for reader, seeds in C.READOUT_SEEDS.items()}
    result = C.compare_shards(shards, n_boot=40)
    for view, reader in product(C.VIEWS, C.READERS):
        name = f"B_minus_R3c/{view}/{reader}"
        p1 = result["aggregates"]["p1"]["contrasts"][name]
        p2 = result["aggregates"]["p2"]["contrasts"][name]
        assert p1["delta_pp"] == pytest.approx(800 / 13)
        assert p1["ci95_pp"] == pytest.approx([800 / 13] * 2)
        assert p2["delta_pp"] == pytest.approx(50)
        assert p2["ci95_pp"] == pytest.approx([50, 50])


def _manual_scores(shards, n_boot, seed):
    """Independent row-level bootstrap; no production count/summarization helpers."""
    rng = np.random.default_rng(seed)
    results = {}
    for ds in C.DATASETS:
        rr = [s for s in shards if s["dataset"] == ds]
        y, subjects = np.asarray(rr[0]["y"]), np.asarray(rr[0]["subjects"])
        names = np.unique(subjects)
        rows = [np.flatnonzero(subjects == name) for name in names]
        picks = rng.integers(len(names), size=(n_boot, len(names)))
        point, draws = {}, {}
        for shard in rr:
            for reader, seed_pred in shard["predictions"].items():
                def score(idx):
                    target = y[idx]
                    return np.mean([
                        np.mean([np.mean(np.asarray(pred)[idx][target == cl] == cl) for cl in np.unique(target)])
                        for pred in seed_pred.values()
                    ])
                cell = f"{shard['model']}/{shard['view']}/{reader}"
                point[cell] = score(np.arange(len(y)))
                draws[cell] = np.asarray([score(np.concatenate([rows[i] for i in p])) for p in picks])
        results[ds] = point, draws
    return results


def test_joint_bootstrap_and_interactions_against_independent_row_oracle():
    shards = _shards()
    rng = np.random.default_rng(12)
    for shard in shards:
        for seeds in shard["predictions"].values():
            for seed in seeds:
                seeds[seed] = rng.integers(3, size=len(shard["y"])).tolist()
    expected = _manual_scores(shards, n_boot=50, seed=19)
    for task, weights in C.TASK_WEIGHTS.items():
        expected[task] = (
            {cell: sum(w * expected[ds][0][cell] for ds, w in weights.items()) for cell in C.CELLS},
            {cell: sum(w * expected[ds][1][cell] for ds, w in weights.items()) for cell in C.CELLS},
        )
    actual = C.compare_shards(shards, n_boot=50, seed=19)
    summaries = {**actual["datasets"], **actual["aggregates"]}
    for name, summary in summaries.items():
        point, draws = expected[name]
        for cell, result in summary["cells"].items():
            assert result["bacc"] == pytest.approx(point[cell])
            assert result["ci95"] == pytest.approx(np.quantile(draws[cell], [.025, .975]))
        # Check the four-way interaction explicitly, rather than reusing its
        # reported coefficients to compute the oracle.
        for model in C.MODELS:
            names = [f"{model}/dec/mlp", f"{model}/pool/mlp", f"{model}/dec/logistic", f"{model}/pool/logistic"]
            signs = [1, -1, -1, 1]
            estimate = sum(sign * point[cell] for cell, sign in zip(names, signs)) * 100
            samples = sum(sign * draws[cell] for cell, sign in zip(names, signs)) * 100
            result = summary["contrasts"][f"interaction/{model}"]
            assert result["delta_pp"] == pytest.approx(estimate)
            assert result["ci95_pp"] == pytest.approx(np.quantile(samples, [.025, .975]))


def test_single_class_bootstrap_draws_are_retained_and_counted():
    shards = _shards()
    _set_arrays(shards, [0, 0, 1, 1], ["a", "a", "b", "b"], [0, 0, 1, 1])
    result = C.compare_shards(shards, n_boot=100)
    assert result["n_boot"] == 100
    for ds in result["datasets"].values():
        assert 0 < ds["bootstrap_single_class_draws"] < 100
        assert ds["bootstrap_draws_with_missing_classes"] == ds["bootstrap_single_class_draws"]
        assert ds["cells"]["B/pool/mlp"]["ci95"] == pytest.approx([1, 1])


def test_numeric_and_epn_text_labels_work_without_casting_between_them():
    shards = _shards()
    for shard in shards:
        if shard["dataset"] == "epn612":
            shard["y"] = [f"gesture{y}" for y in shard["y"]]
            shard["predictions"] = {reader: {seed: [f"gesture{p}" for p in pred] for seed, pred in seeds.items()}
                                    for reader, seeds in shard["predictions"].items()}
    result = C.compare_shards(shards, n_boot=20)
    assert result["datasets"]["epn612"]["classes"] == ["gesture0", "gesture1", "gesture2"]
    assert result["datasets"]["epn612"]["cells"]["B/dec/mlp"]["bacc"] == 1


def test_row_and_shard_order_do_not_change_result_or_modify_inputs():
    shards = _shards()
    original = deepcopy(shards)
    shuffled = deepcopy(shards[::-1])
    for shard in shuffled[::2]:
        for key in ("rows", "y", "subjects"):
            shard[key].reverse()
        for seeds in shard["predictions"].values():
            for pred in seeds.values():
                pred.reverse()
    assert C.compare_shards(shards, n_boot=20) == C.compare_shards(shuffled, n_boot=20)
    assert shards == original


def test_global_numpy_rng_is_not_consumed():
    before = np.random.get_state()
    C.compare_shards(_shards(), n_boot=10)
    after = np.random.get_state()
    np.testing.assert_equal(before, after)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra"])
def test_incomplete_or_duplicate_grid_rejected(kind):
    shards = _shards()
    if kind == "missing":
        shards.pop()
    elif kind == "duplicate":
        shards.append(deepcopy(shards[0]))
    else:
        shards[0]["view"] = "local"
    with pytest.raises(ValueError, match="grid"):
        C.compare_shards(shards, n_boot=10)


@pytest.mark.parametrize("field,value", [
    ("protocol", "other_protocol"), ("recipe_sha256", "f" * 64),
    ("input_sha256", "f" * 64), ("roles_sha256", "f" * 64),
    ("checkpoint_sha256", "f" * 64),
])
def test_identity_mismatch_rejected(field, value):
    shards = _shards()
    shards[0][field] = value
    with pytest.raises(ValueError, match="mismatch|inconsistent"):
        C.compare_shards(shards, n_boot=10)


@pytest.mark.parametrize("field", ["rows", "y", "subjects"])
def test_paired_rows_targets_subjects_must_match_exactly(field):
    shards = _shards()
    shards[0][field][0] = {"rows": 100, "y": 1, "subjects": "other"}[field]
    with pytest.raises(ValueError, match="paired.*mismatch"):
        C.compare_shards(shards, n_boot=10)


@pytest.mark.parametrize("bad", [False, -1, 0.5, 2**80])
def test_invalid_row_indices_rejected(bad):
    shards = _shards()
    shards[0]["rows"][0] = bad
    with pytest.raises(ValueError, match="rows"):
        C.compare_shards(shards, n_boot=10)


def test_duplicate_row_rejected_even_if_repeated_in_every_cell():
    shards = _shards()
    for shard in shards:
        shard["rows"][1] = shard["rows"][0]
    with pytest.raises(ValueError, match="duplicate rows"):
        C.compare_shards(shards, n_boot=10)


@pytest.mark.parametrize("location,bad", [
    ("y", np.nan), ("y", np.inf), ("pred", np.nan), ("pred", -np.inf),
    ("y", "0"), ("pred", "0"), ("pred", None), ("pred", 1j),
])
def test_invalid_or_mixed_label_types_rejected(location, bad):
    shards = _shards()
    value = shards[0]["y"] if location == "y" else shards[0]["predictions"]["mlp"]["1"]
    value[0] = bad
    with pytest.raises(ValueError, match="label|predictions"):
        C.compare_shards(shards, n_boot=10)


def test_prediction_absent_from_global_targets_is_counted_as_an_error():
    shards = _shards()
    shards[0]["predictions"]["logistic"]["0"][0] = 7
    result = C.compare_shards(shards, n_boot=20)
    ds = result["datasets"][shards[0]["dataset"]]
    assert ds["cells"]["B/pool/logistic"]["bacc"] == pytest.approx(17 / 18)
    assert ds["predictions_outside_observed_targets"]["B/pool/logistic"] == {"0": 1}
    assert ds["predictions_outside_observed_targets"]["B/pool/mlp"] == {"0": 0, "1": 0}


@pytest.mark.parametrize("mutation", ["reader", "missing_seed", "extra_seed", "bad_length"])
def test_exact_reader_seed_grid_and_prediction_lengths_required(mutation):
    shards = _shards()
    pred = shards[0]["predictions"]
    if mutation == "reader":
        pred.pop("logistic")
    elif mutation == "missing_seed":
        pred["mlp"].pop("1")
    elif mutation == "extra_seed":
        pred["logistic"]["1"] = pred["logistic"]["0"].copy()
    else:
        pred["mlp"]["1"].pop()
    with pytest.raises(ValueError, match="predictions|seeds|length"):
        C.compare_shards(shards, n_boot=10)


@pytest.mark.parametrize("status", [None, "timing_only", "failed", "running"])
def test_only_completed_shards_are_accepted(status):
    shards = _shards()
    if status is None:
        shards[0].pop("status")
    else:
        shards[0]["status"] = status
    with pytest.raises(ValueError, match="status"):
        C.compare_shards(shards, n_boot=10)


@pytest.mark.parametrize("field", ["recipe_sha256", "input_sha256", "roles_sha256", "checkpoint_sha256"])
def test_malformed_hash_rejected(field):
    shards = _shards()
    shards[0][field] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        C.compare_shards(shards, n_boot=10)


@pytest.mark.parametrize("kwargs", [{"n_boot": 0}, {"n_boot": 1}, {"n_boot": True},
                                    {"n_boot": 3.5}, {"seed": -1}, {"seed": False}])
def test_invalid_bootstrap_controls_rejected(kwargs):
    with pytest.raises(ValueError, match="integer"):
        C.compare_shards(_shards(), **kwargs)
