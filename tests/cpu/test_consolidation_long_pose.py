"""Original-loader parity, frozen group-CV identity and paired ratio oracles."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import consolidation_long_pose as P
from test_tokenizer_pose_readout import data as ridge_data


def data(seed=61, users=("a", "b", "c", "d")):
    result = ridge_data(seed=seed, users=users)
    result.update(pre=result["lat"].copy(), key_time_valid=np.ones(result["lat"].shape[:2], bool),
                  pred=result["target"].copy(), has_head=True, dim=result["lat"].shape[-1], config={})
    for i, meta in enumerate(result["meta"]):
        meta.update(session="s"+meta["user"], stage="A", side="left", stem=f"r{i//4}",
                    recording=i//4, start=(i % 4)*16000, fs=2000.)
    return result


def cells():
    d = data()
    output = {}
    for dose, f in zip(P.DOSES, (.6, .7, .8)):
        for reader, increment in zip(P.READERS, (0., .05, .1)):
            output[f"{dose}/{reader}"] = {
                "complete": True, "which": "teacher", "checkpoint_step": dose, "reader": reader,
                "checkpoint_sha256": str(P.DOSES.index(dose)+1)*64,
                "fit": {"selection": {"boundary_unresolved": False}},
                "analysis": P.summarize(d, d["target"][:, 6:-6].astype(float)*(f+increment))}
    return output


def test_original_loader_real_model_one_forward_exact_parity_and_no_grad(tmp_path, monkeypatch):
    from test_pose_eval import _setup, CFG
    from wearusfm.data import pose_targets as PT
    from wearusfm.model.fm import WearUsFM

    root, pose_root, csv = _setup(tmp_path)
    recs = P.select_recordings(csv, root, "val")[:1]
    model = WearUsFM(CFG).eval()
    with torch.no_grad():
        model.pose_head.out.weight.normal_(std=.1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    store = PT.PoseStore(pose_root)
    ev = P.PE.EvalConfig(batch=2)
    historical = P.PE.extract(model, recs, store, ev, log=lambda _: None)
    encode = model.encode
    calls = []
    def observed(inp, mask):
        assert not torch.is_grad_enabled() and not model.training and mask is None
        calls.append(len(inp.signals))
        return encode(inp, mask)
    monkeypatch.setattr(model, "encode", observed)
    extracted = P.extract_pose(model, recs, store, ev, reference=historical, log=lambda _: None)
    assert sum(calls) == len(extracted["meta"])
    for key in ("lat", "pred", "target", "mask", "std", "pose_var"):
        np.testing.assert_array_equal(extracted[key], historical[key])
    assert extracted["meta"] == historical["meta"] and extracted["config"] == historical["config"]
    assert extracted["key_time_valid"].all() and extracted["pre"].shape == extracted["lat"].shape
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())
    path = tmp_path/"depth.npz"
    P.save_cache(path, extracted)
    loaded = P.read_cache(path, historical)
    np.testing.assert_array_equal(loaded["pre"], extracted["pre"])
    np.testing.assert_array_equal(loaded["key_time_valid"], extracted["key_time_valid"])
    with pytest.raises(FileExistsError):
        P.save_cache(path, extracted)


def test_original_recording_selection_is_reused(monkeypatch):
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return ["original"]
    monkeypatch.setattr(P.PE, "recordings_for", original)
    assert P.select_recordings("meta.csv", "processed", "train") == ["original"]
    P.select_recordings("meta.csv", "processed", "val")
    assert calls[0][1] == dict(n_users=40, n_recordings=200, seed=20261026)
    assert calls[1][1] == dict(n_users=None, n_recordings=None, seed=20261026)


@pytest.mark.parametrize("change", ["mask", "target", "std", "pose_var", "metadata", "time_valid"])
def test_validation_fails_instead_of_changing_original_support(change):
    original = data()
    changed = copy.deepcopy(original)
    if change == "metadata":
        changed["meta"][0]["start"] += 400
    elif change == "time_valid":
        changed["key_time_valid"][0, 4] = False
    elif change == "mask":
        changed["mask"][0, 7, 0] ^= True
    elif change == "target":
        changed["target"][0, 7, 0] += .1
    else:
        changed[change].flat[0] += .1
    with pytest.raises(ValueError, match="historical|key_time_valid"):
        P.validate_cache(changed, original)


def test_short_window_padding_retains_original_masks():
    d = data()
    d["meta"][0]["n_patches"] = 19
    d["mask"][0, 19:] = False
    d["key_time_valid"][0, 19:] = False
    d["pre"][0, 19:] = np.nan
    d["lat"][0, 19:] = np.nan
    original_mask = d["mask"].copy()
    P.validate_cache(d)
    view = P.view_data(d, "pre")
    np.testing.assert_array_equal(d["mask"], original_mask)
    np.testing.assert_array_equal(view["central_mask"][0, :7], original_mask[0, 6:13])
    assert not view["central_mask"][0, 7:].any()


def test_ridge_reuses_exact_train_group_selection_scaler_alpha_and_coefficients():
    d = data()
    folds = [["a", "c"], ["b", "d"]]
    expected = P.R.fit_reader(P.view_data(d, "post"), folds)
    for depth in P.DEPTHS:
        weights, bias, fit = P.fit_reader(d, depth, folds)
        np.testing.assert_array_equal(weights, expected[0])
        np.testing.assert_array_equal(bias, expected[1])
        assert fit["selection"] == expected[2]["selection"]
        assert fit["selection"]["fold_users"] == folds
    with pytest.raises(ValueError, match="fold users"):
        P.fit_reader(d, "pre", [["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError, match="leakage"):
        P.evaluate_reader(d, d, "pre")


def test_prediction_cannot_use_held_pose_or_masks_and_fit_ignores_unsupported_targets():
    train, held = data(), data(seed=90, users=("held_a", "held_b"))
    weights, bias, fit = P.fit_reader(train, "pre")
    expected = P.predict_reader(held, "pre", weights, bias)
    held["target"][:] = np.nan
    held["mask"][:] = False
    held["pose_var"][:] = 1e30
    np.testing.assert_array_equal(P.predict_reader(held, "pre", weights, bias), expected)
    train["target"][~train["mask"]] = np.nan
    train["target"][:, :6] = 1e20
    train["target"][:, -6:] = -1e20
    got = P.fit_reader(train, "pre")
    np.testing.assert_array_equal(got[0], weights)
    np.testing.assert_array_equal(got[1], bias)
    assert got[2]["selection"] == fit["selection"]


def test_e2_historical_pairing_common_mask_and_original_dynamic_selection():
    d = data()
    d["mask"][0, :8] = False
    d["mask"][1, -8:] = False
    # Exact historical full-window partners; only evaluation support is cropped.
    partner, info = P.PE.misaligned_pairs(d["meta"], d["target"], d["mask"])
    support = P.e2_support(d)
    np.testing.assert_array_equal(support["partner"], partner)
    central = P.view_data(d, "pre")["central_mask"]
    dynamic = d["pose_var"] >= P.R.THRESHOLD
    expected_mask = central & central[np.maximum(partner, 0)] & (partner >= 0)[:, None, None] & dynamic[:, None, None]
    np.testing.assert_array_equal(support["mask"], expected_mask)
    prediction = d["target"][:, 6:-6].astype(float).copy()
    summary = P.summarize(d, prediction)
    assert summary["E2"]["mae_aligned"] == 0
    assert summary["E2"]["n_valid_elements"] == int(expected_mask.sum())
    predicted_wrong = prediction.copy()
    predicted_wrong[~expected_mask] = 1e9
    assert P.summarize(d, predicted_wrong)["E2"] == summary["E2"]
    assert summary["groups"]["dynamic"]["n_selected_windows"] == int(dynamic.sum())
    assert summary["E2"]["gap_mae"] == summary["E2"]["mae_misaligned"]
    assert support["signature"]["pairing"]["n_same_recording"] == info["n_same_recording"]


def test_e2_std_mismatch_is_explicit_and_does_not_relax_pairing():
    d = data()
    d["std"][0] *= 2
    support = P.e2_support(d)
    assert support["pairing"]["n_excluded_std_mismatch"] > 0
    assert support["partner"][0] == -1 and not support["mask"][0].any()


def test_paired_bootstrap_full_ratio_exact_oracle_and_all_contrasts():
    c = cells()
    got = P.aggregate(c)
    assert len(got["contrasts"]) == 12
    key = "49046/trained_head_minus_19618/trained_head"
    row = got["contrasts"][key]["groups"]["dynamic"]
    users = got["bootstrap_groups"]["dynamic"]["users"]
    draws = np.random.default_rng(20261102).integers(0, len(users), (2000, len(users)))
    assert got["bootstrap_groups"]["dynamic"]["draws_sha256"] == P.R.arrays_hash(draws)
    def error(dose):
        rows = c[f"{dose}/trained_head"]["analysis"]["groups"]["dynamic"]["centered"]["per_user"]
        return np.asarray([rows[u]["sums"]["centered"] for u in users])
    base, new = error(19618), error(49046)
    expected = 100*(1-new[draws].sum(1)/base[draws].sum(1))
    np.testing.assert_allclose(row["relative_improvement_percent"]["ci95"], np.percentile(expected, [2.5, 97.5]), atol=1e-13)
    assert row["relative_improvement_percent"]["point"] == pytest.approx(100*(1-new.sum()/base.sum()))
    assert row["assessment"] == "practical_benefit"
    # The converged tokenizer oracle uses exactly the same pooled-error delta.
    mapping = dict(zip(P.R.ARMS, P.DOSES))
    historical = {arm: {**c[f"{dose}/trained_head"], "arm": arm, "checkpoint_step": 19618} for arm, dose in mapping.items()}
    old = P.R.paired_comparison(historical, 19618)["dynamic"]["contrasts"]["T100-G_minus_T25-L"]
    np.testing.assert_allclose(row["delta"]["E_dyn"]["ci95"], old["ci95"], atol=1e-16)
    assert got["contrasts"][key]["primary_dose_contrast"] is True
    assert all("trained_head" not in name for name, value in got["contrasts"].items() if value["kind"] == "depth")
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize("mutation", ["missing", "identity", "checkpoint", "support", "denominator", "e2", "boundary"])
def test_aggregate_fails_closed_on_unpaired_or_incomplete_cells(mutation):
    c = cells()
    key = "49046/post_ridge"
    if mutation == "missing":
        del c[key]
    elif mutation == "identity":
        c[key]["reader"] = "pre_ridge"
    elif mutation == "checkpoint":
        c[key]["checkpoint_sha256"] = "f"*64
    elif mutation == "support":
        c[key]["analysis"]["support_sha256"] = "0"*64
    elif mutation == "denominator":
        c[key]["analysis"]["groups"]["dynamic"]["centered"]["per_user"]["a"]["sums"]["vary"] += 1
    elif mutation == "e2":
        c[key]["analysis"]["E2"]["support"]["common_mask_sha256"] = "0"*64
    else:
        c[key]["fit"]["selection"]["boundary_unresolved"] = True
    with pytest.raises(ValueError):
        P.aggregate(c)


def test_zero_prediction_variance_preserves_undefined_correlation_without_discarding_draws():
    c = cells()
    d = data()
    c["19618/pre_ridge"]["analysis"] = P.summarize(d, np.zeros_like(d["target"][:, 6:-6]))
    got = P.aggregate(c, n_boot=30)
    correlation = got["cells"]["19618/pre_ridge"]["groups"]["dynamic"]["correlation"]
    assert correlation["point"] is None and correlation["ci95"] is None and not correlation["evaluable"]
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize("point,ci,status", [(2., [.1, 4.], "practical_benefit"), (-2., [-4., -.1], "practical_worsening"),
                                             (1., [.1, 1.5], "small"), (4., [-1., 7.], "inconclusive")])
def test_two_sided_practical_rule(point, ci, status):
    assert P.practical_assessment(dict(point=point, ci95=ci, evaluable=True)) == status


def test_reference_selection_defers_clear_tradeoff():
    result = P.aggregate(cells(), n_boot=20)
    reader = "trained_head"
    contrasts = result["contrasts"]
    selected = P.select_reference(contrasts, reader)
    assert selected["candidate_from_E_dyn"] == 49046
    contrasts["49046/trained_head_minus_19618/trained_head"]["clear_regressions"] = ["correlation"]
    assert P.select_reference(contrasts, reader)["selected_reference"] is None
    assert P.select_reference(contrasts, reader)["status"] == "defer_tradeoff"


def test_train_only_benchmark_delegates_exact_recipe_and_projection_is_not_launch_gate(monkeypatch):
    d = data()
    def benchmark(train):
        assert np.array_equal(train["lat"], d["pre"])
        assert {m["user"] for m in train["meta"]} == {"a", "b", "c", "d"}
        return {"statistics_seconds": 2., "three_eigensystems_seconds": 3.}
    monkeypatch.setattr(P.R, "benchmark_train", benchmark)
    result = P.project_fit_seconds(P.benchmark_train(d))
    assert result["fit_seconds"] == 2*6*(2+27*3)+120
    assert result["complete_resource_plan"] is False and result["excludes"]
