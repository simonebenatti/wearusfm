"""Natural-width regressions, S5-before-LN, leakage and paired-support tests."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import scaling_pose_readout as S
import scaling_projection as P
import tokenizer_pose_readout as Q
import test_tokenizer_pose_readout as OLD


@pytest.fixture(scope="module")
def projection(tmp_path_factory):
    return P.load(P.save(tmp_path_factory.mktemp("pose_s5") / "matrices"))


def data(dim, seed=61, users=("a", "b", "c", "d")):
    d = OLD.data(seed=seed, users=users, windows=5, dim=dim, joints=1)
    x = Q.T.normalized_inputs(d["lat"], d["lengths"])
    rng = np.random.default_rng(seed + 8)
    d["target"] = (.4 * x[..., :1] + .3 * x[..., 1:2] + .7 * rng.normal(size=d["target"].shape)).astype(np.float32)
    return d


@pytest.mark.parametrize("arm", S.ARMS)
def test_natural_fit_and_predictions_match_frozen_reader_at_both_widths(arm):
    d = data(S.V.dimension(arm))
    w0, b0, old = Q.fit_reader(d)
    w, b, new = S.fit_reader(d, arm, "natural", expected_folds=[["a", "c"], ["b", "d"]])
    np.testing.assert_array_equal(w, w0)
    np.testing.assert_array_equal(b, b0)
    assert new["selection"] == old["selection"]
    assert new["fit_support_sha256"] == old["fit_support_sha256"]
    np.testing.assert_array_equal(S.predict(d["lat"], d["lengths"], w, b, arm, "natural"),
                                  Q.predict(d["lat"], d["lengths"], w0, b0))


def test_matched_projection_is_before_layernorm_and_matches_direct_frozen_fit(projection):
    d = data(896)
    converted = {**d, "lat": P.project_pose(d["lat"], projection)}
    expected = Q.T.normalized_inputs(converted["lat"], d["lengths"])
    actual = S.normalized_inputs(d["lat"], d["lengths"], S.ARMS[1], "matched", projection)
    np.testing.assert_array_equal(actual, expected)
    wrong_order = P.project_pose(Q.T.normalized_inputs(d["lat"], d["lengths"]), projection)
    assert not np.allclose(actual, wrong_order)
    w0, b0, old = Q.fit_reader(converted)
    w, b, new = S.fit_reader(d, S.ARMS[1], "matched", projection)
    np.testing.assert_array_equal(w, w0)
    np.testing.assert_array_equal(b, b0)
    assert new["selection"] == old["selection"] and new["n_features"] == 384


def test_30m_matched_control_is_exactly_unchanged_and_needs_no_projection():
    d = data(384)
    np.testing.assert_array_equal(S.normalized_inputs(d["lat"], d["lengths"], S.ARMS[0], "matched", None),
                                  Q.T.normalized_inputs(d["lat"], d["lengths"]))
    assert (S.ARMS[0], "matched") not in S.CELLS.values()


def test_projected_prediction_cannot_use_validation_targets_or_pose_masks(projection):
    d = data(896, users=("v", "w"))
    rng = np.random.default_rng(18)
    weights, bias = rng.normal(size=(384, 1)), np.zeros(1)
    before = S.predict(d["lat"], d["lengths"], weights, bias, S.ARMS[1], "matched", projection)
    d["target"][:] = np.nan
    d["mask"][:] = False
    d["pose_var"][:] = 1e9
    np.testing.assert_array_equal(before, S.predict(d["lat"], d["lengths"], weights, bias,
                                                  S.ARMS[1], "matched", projection))
    d["lengths"][0] = 19
    d["lat"][0, 19:] = np.nan
    assert np.isfinite(S.predict(d["lat"], d["lengths"], weights, bias, S.ARMS[1], "matched", projection)).all()
    d["lat"][0, 17] = np.nan
    with pytest.raises(ValueError, match="real input"):
        S.predict(d["lat"], d["lengths"], weights, bias, S.ARMS[1], "matched", projection)


def test_masked_and_border_targets_do_not_change_matched_training_statistics(projection):
    d = data(896)
    first, dim, joints = S.grouped_stats(d, S.ARMS[1], "matched", projection)
    changed = copy.deepcopy(d)
    changed["target"][:, :6] = 1e12
    changed["target"][:, -6:] = -1e12
    changed["target"][~changed["mask"]] = np.nan
    second, _, _ = S.grouped_stats(changed, S.ARMS[1], "matched", projection)
    assert dim == 384 and joints == 1 and first.keys() == second.keys()
    for user in first:
        for a, b in zip(first[user], second[user]):
            assert a.keys() == b.keys()
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])


def cells():
    d = OLD.data()
    result = {}
    for (name, (arm, view)), factor in zip(S.CELLS.items(), (.6, .8, .9)):
        result[name] = {"complete": True, "arm": arm, "view": view, "which": "teacher", "checkpoint_step": 5000,
            "checkpoint_sha256": ("a" if arm == S.ARMS[0] else "b") * 64,
            "projection_manifest_sha256": "c" * 64 if view == "matched" else None,
            "fit": {"n_features": 384 if view == "matched" else S.V.dimension(arm),
                    "selection": {"boundary_unresolved": False}},
            "analysis": Q.summarize(d, d["target"][:, 6:-6] * factor)}
    return result


def test_primary_and_secondary_contrasts_share_subject_draws_and_direct_oracle():
    got = S.paired_comparison(cells())["dynamic"]
    expected = {"natural_300m_minus_30m": -.12, "matched_300m_minus_30m": -.15,
                "natural_300m_minus_matched_300m": .03}
    users = got["users"]
    draws = np.random.default_rng(20261102).integers(0, len(users), (2000, len(users)))
    assert got["draws_sha256"] == Q.arrays_hash(draws)
    for name, delta in expected.items():
        row = got["contrasts"][name]
        assert row["delta_ratio"] == pytest.approx(delta, abs=1e-7)
        num = np.asarray([row["per_user"][u]["delta_centered_sse"] for u in users])
        den = np.asarray([row["per_user"][u]["target_variance_sum"] for u in users])
        np.testing.assert_array_equal(row["ci95"], np.percentile(num[draws].sum(1) / den[draws].sum(1), [2.5, 97.5]))
    old = Q.paired_comparison(OLD.cells(), 5000)["dynamic"]
    assert got["draws_sha256"] == old["draws_sha256"]
    np.testing.assert_array_equal(got["contrasts"]["natural_300m_minus_30m"]["ci95"],
                                  old["contrasts"]["T100-L_minus_T25-L"]["ci95"])


@pytest.mark.parametrize("mutation", ("missing", "incomplete", "dose", "support", "denominator", "checkpoint", "projection", "solver"))
def test_unpaired_or_unresolved_comparison_is_rejected(mutation):
    c = cells()
    last = c["T100-G-300M-matched"]
    if mutation == "missing": c.pop("T100-G-30M-natural")
    if mutation == "incomplete": last["complete"] = False
    if mutation == "dose": last["checkpoint_step"] = 19618
    if mutation == "support": last["analysis"]["support_sha256"] = "different"
    if mutation == "denominator": last["analysis"]["groups"]["dynamic"]["centered"]["per_user"]["a"]["sums"]["vary"] += 1
    if mutation == "checkpoint": last["checkpoint_sha256"] = "e" * 64
    if mutation == "projection": last["projection_manifest_sha256"] = None
    if mutation == "solver": last["fit"]["selection"]["boundary_unresolved"] = True
    with pytest.raises(ValueError):
        S.paired_comparison(c)


def fake_plan(tmp_path, projection):
    old = OLD.fake_plan(tmp_path)
    caches = {a: copy.deepcopy(old["caches"][b]) for a, b in zip(S.ARMS, ("T25-L", "T100-G"))}
    source = {"path": "/source_contract.json", "sha256": "d" * 64, "bytes": 1}
    return S.make_plan(caches, old["references"], old["fold_users"], projection.pin, source)


def test_plan_freezes_two_arms_three_views_projection_and_recipe(tmp_path, projection):
    plan = fake_plan(tmp_path, projection)
    S.validate_plan(plan)
    for mutate in (lambda p: p.update(step=19618), lambda p: p["recipe"].update(offset=4),
                   lambda p: p["caches"].pop(S.ARMS[0]), lambda p: p["projection"].update(sha256="0" * 64),
                   lambda p: p["caches"][S.ARMS[1]]["val"].update(checkpoint_sha256="f" * 64)):
        changed = copy.deepcopy(plan)
        mutate(changed)
        with pytest.raises(ValueError):
            S.validate_plan(changed)


def artifact(tmp_path, arm, legacy=False):
    d, pin, plan, ref, prov, receipt = OLD.artifact_fixture(tmp_path)
    dim = S.V.dimension(arm)
    d["lat"] = np.resize(d["lat"], (len(d["lat"]), 160, dim)).copy().astype(np.float16)
    protocol, source_arm = (S.V.E.PROTOCOL, "T100-G") if legacy else (S.V.PROTOCOL, arm)
    contract_sha = "d" * 64
    plan["training_source_contract"] = {"sha256": contract_sha}
    config = prov["config"]
    model = json.loads((S.ROOT / "results/infra/tokenizer_review_20261103/config_T100-G.json").read_text())["model"]
    model.update(S.V.R.LADDER["30m" if arm == S.ARMS[0] else "300m"])
    np.savez(pin["path"], **{k: d[k] for k in ("lat", "target", "mask", "std", "pose_var")},
             meta_json=np.asarray(json.dumps(d["meta"])), config_json=np.asarray(json.dumps(config)),
             n_recordings=d["n_recordings"], dim=dim, has_head=True)
    pin["sha256"], pin["bytes"] = Q.D.file_sha256(pin["path"]), Path(pin["path"]).stat().st_size
    prov.update(protocol=protocol, arm=source_arm, model_config=model, cache_sha256=pin["sha256"],
                cache_bytes=pin["bytes"], training_source_contract_sha256=contract_sha)
    Path(pin["provenance_path"]).write_text(json.dumps(prov))
    pin["provenance_sha256"] = Q.D.file_sha256(pin["provenance_path"])
    receipt.update(protocol=protocol, arm=source_arm, stage="extract-pose", training_source_contract_sha256=contract_sha,
                   outputs={"train": {"path": pin["path"], "sha256": pin["sha256"], "bytes": pin["bytes"],
                       "provenance_path": pin["provenance_path"], "provenance_sha256": pin["provenance_sha256"]}})
    Path(pin["receipt_path"]).write_text(json.dumps(receipt))
    pin["receipt_sha256"] = Q.D.file_sha256(pin["receipt_path"])
    return d, pin, plan, ref, prov, receipt


@pytest.mark.parametrize("arm,legacy", ((S.ARMS[0], True), (S.ARMS[0], False), (S.ARMS[1], False)))
def test_reader_accepts_only_explicit_historical_or_scaling_provenance(tmp_path, arm, legacy):
    _, pin, plan, ref, _, _ = artifact(tmp_path, arm, legacy)
    assert S.validate_artifact(pin, arm, "train", plan, ref)["users"] == ["a", "b"]
    if legacy:
        assert Q.validate_artifact(pin, "T100-G", "train", plan, ref)["targets"] == ref["targets"]


@pytest.mark.parametrize("mutation", ("complete", "source", "width", "target", "protocol"))
def test_reader_rejects_rehashed_but_semantically_wrong_provenance(tmp_path, mutation):
    _, pin, plan, ref, prov, receipt = artifact(tmp_path, S.ARMS[1])
    if mutation == "complete": receipt["complete"] = False
    if mutation == "source": prov["training_source_contract_sha256"] = "e" * 64
    if mutation == "width": prov["model_config"]["dim"] = 384
    if mutation == "target": ref["targets"]["arrays"] = "0" * 64
    if mutation == "protocol": prov["protocol"] = S.V.E.PROTOCOL
    Path(pin["provenance_path"]).write_text(json.dumps(prov))
    pin["provenance_sha256"] = Q.D.file_sha256(pin["provenance_path"])
    receipt["outputs"]["train"]["provenance_sha256"] = pin["provenance_sha256"]
    Path(pin["receipt_path"]).write_text(json.dumps(receipt))
    pin["receipt_sha256"] = Q.D.file_sha256(pin["receipt_path"])
    with pytest.raises(ValueError):
        S.validate_artifact(pin, S.ARMS[1], "train", plan, ref)


def test_benchmark_dispatch_never_reads_validation_or_selects_alpha(tmp_path, monkeypatch, projection):
    plan = fake_plan(tmp_path, projection)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    identity = {"sources": plan["sources"]}
    monkeypatch.setattr(Q, "runtime_identity", lambda _: copy.deepcopy(identity))
    source = tmp_path / "source.json"
    source.write_text('{}')
    monkeypatch.setattr(S.ST, "resolve_pin", lambda _: {"path": str(source)})
    monkeypatch.setattr(S.ST, "check_source_contract", lambda _: {"states": {}, "contract": {
        "reference_checkpoint_sha256": plan["caches"][S.ARMS[0]]["train"]["checkpoint_sha256"]}})
    users, touched = sorted(sum(plan["fold_users"], [])), []
    def reference(pin):
        assert "ref_train" in pin["path"]
        touched.append("reference_train")
        return {"users": users, "states": {}}
    def validate(pin, arm, split, *args):
        assert split == "train"
        touched.append(arm + "/" + split)
        return {"states": {}}
    monkeypatch.setattr(Q, "validate_reference", reference)
    monkeypatch.setattr(S, "validate_artifact", validate)
    monkeypatch.setattr(Q, "read_data", lambda pin: touched.append(pin["path"]) or {})
    monkeypatch.setattr(S, "benchmark_train", lambda *args: {"seconds": 1})
    monkeypatch.setattr(Q.E, "cross_validate", lambda *args: pytest.fail("no alpha selection in benchmark"))
    out = tmp_path / "benchmark"
    assert S.main(["benchmark", "--plan", str(plan_path), "--expected-code-sha", "a" * 40,
                   "--cell", "T100-G-300M-matched", "--out", str(out)]) == 0
    assert len(touched) == 3 and touched[1] == "T100-G-300M/train"
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "train_only_timing" and report["projection_pin"] == projection.pin


def test_runner_fits_train_before_reading_val_and_writes_all_three_cells(tmp_path, monkeypatch, projection):
    train, val = data(896), data(896, seed=72, users=("v", "w"))
    plan = {"fold_users": [["a", "c"], ["b", "d"]], "caches": {}}
    for arm in S.ARMS:
        plan["caches"][arm] = {split: {"path": arm + "/" + split,
            "checkpoint_sha256": ("a" if arm == S.ARMS[0] else "b") * 64} for split in ("train", "val")}
    events = []
    def read(pin):
        arm, split = pin["path"].split("/")
        events.append(split)
        if split == "val": assert events[-2] == "fit_train"
        d = train if split == "train" else val
        return {**d, "lat": d["lat"][..., :S.V.dimension(arm)]}
    def fit(d, arm, view, *args):
        assert set(m["user"] for m in d["meta"]) == {"a", "b", "c", "d"}
        events.append("fit_train")
        dim = 384 if view == "matched" else S.V.dimension(arm)
        return np.zeros((dim, 1)), np.zeros(1), {"n_features": dim, "selection": {"boundary_unresolved": False}}
    monkeypatch.setattr(Q, "read_data", read)
    monkeypatch.setattr(S, "fit_reader", fit)
    out = tmp_path / "ridge"
    out.mkdir()
    got = S.run(plan, out, projection)
    assert got["complete"] and set(got["cells"]) == set(S.CELLS)
    assert events == ["train", "fit_train", "val"] * 3
    for name, cell in got["cells"].items():
        for key in ("coefficients", "predictions"):
            assert Q.D.file_sha256(cell[key]["path"]) == cell[key]["sha256"]
        assert cell["analysis"]["prediction_sha256"] == Q.arrays_hash(np.load(cell["predictions"]["path"], allow_pickle=False))


def test_unresolved_alpha_cannot_produce_a_fit_report(monkeypatch, projection):
    monkeypatch.setattr(Q.E, "cross_validate", lambda *args: {"boundary_unresolved": True})
    with pytest.raises(ValueError, match="alpha boundary unresolved"):
        S.fit_reader(data(896), S.ARMS[1], "matched", projection)
