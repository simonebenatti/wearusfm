"""Diagnostics of the failed temporal control: the coverage gaps found by mutation in the independent review.

Each test fixes one point that the first test file left free; the fixtures are the same small synthetic control.
"""
import copy
import dataclasses
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[2]
for sub in ("tests/cpu", "scripts", "src"):
    sys.path.insert(0, str(TREE / sub))
from pose_temponet import controls as C, metrics as M, provenance as P, selection as S, train as T, workflow as W

import test_pose_temponet_workflow as WF
import test_tcn_temporal_diag as TD
from test_tcn_temporal_diag import world  # noqa: F401  (module fixture of the reviewed tests)
from tcn_temporal_diag import core
import tcn_temporal_diag.__main__ as entry

RECIPE = TD.RECIPE


# ----------------------------------------------------------------- T1: references of the right population
def test_summary_uses_the_references_of_its_own_population(world):
    cb, selection, states, refs = TD._tcn(world)
    record = core.replay_head(cb, core.HEAD, selection, states, refs, device="cpu", recipe=RECIPE,
                              scope="selected_best")["records"][0]
    recoveries = {}
    for split in ("held", "fitting"):
        observed = [f[split]["observed"] for f in record["per_fold"]]
        clean = [f[split]["clean"] for f in record["per_fold"]]
        for index, name in ((0, "fold0"), (1, "fold1"), (None, "pooled")):
            pick = lambda rows: C.pool_statistics(list(rows)) if index is None else rows[index]
            e = {k: C.statistic_values(pick(v))["E_dyn"] for k, v in (
                ("reader", observed), ("clean_ls", refs[split]["clean_ls"]), ("oracle", refs[split]["oracle"]))}
            got = record["summary"][split]["recovery_on_observed"][name]
            assert got["E_dyn"] == e
            assert got["recovery"] == pytest.approx((e["clean_ls"] - e["reader"]) / (e["clean_ls"] - e["oracle"]), rel=1e-12)
            recoveries[split, name] = got["recovery"]
            oracle = pick(refs[split]["oracle"])
            alignment = record["summary"][split]["noise_alignment"][name]
            assert alignment["noise_sse"] == oracle["dynamic_sse"]
            assert alignment["error_shares"]["noise_level"] == oracle["dynamic_sse"] / pick(observed)["dynamic_target_ss"]
            assert alignment["cross_term"] == pytest.approx(
                (pick(clean)["dynamic_sse"] + oracle["dynamic_sse"] - pick(observed)["dynamic_sse"]) / 2, rel=1e-12)
    # The fixture must separate the two populations, otherwise the assertions above prove nothing.
    assert all(refs["held"][k][f] != refs["fitting"][k][f] for k in ("clean_ls", "oracle") for f in (0, 1))
    assert all(recoveries["held", n] != recoveries["fitting", n] for n in core.PANELS)


# ----------------------------------------------------------------- T2: oracle against the frozen closed-form path
def test_oracle_statistics_equals_the_frozen_outer_reference_definition(world):
    cb, aa = world["bundle"], world["artifacts"]
    for fold in (0, 1):
        choice = aa[fold]["calibration"]["selected"]["temporal"]
        clean, observed = C.synthetic_target(aa[fold]["components"], choice["a"], choice["sigma"])
        threshold = M.training_threshold(clean[cb.fitting[fold]], cb.cache["mask"][cb.fitting[fold]])
        for indices in (cb.held[fold], cb.fitting[fold]):
            # Same calls as controls.evaluate_outer_references / _outer_linear_diagnostics.
            expected = M.selection_statistics(clean[indices, 6:-6], C._metric_panel(cb.cache, indices, observed, clean), threshold)
            assert core.oracle_statistics(cb, fold, indices) == expected


# ----------------------------------------------------------------- T3: per-fold parity of the selected state
def test_per_fold_parity_of_the_selected_state_exists_and_fails_closed(world):
    cb, selection, states, refs = TD._tcn(world)
    replay = core.replay_head(cb, core.HEAD, selection, states, refs, device="cpu", recipe=RECIPE)
    labels = [p["panel"] for p in replay["parity"] if p["panel"] != "pooled_curve_point"]
    assert sorted(labels) == sorted("fold%d_%s" % (f, l) for f in (0, 1)
                                    for l in ("held_observed", "held_clean", "fitting_observed"))
    assert len([p for p in replay["parity"] if p["panel"] == "pooled_curve_point"]) == 2 * len(selection["curves"])
    for key, name, label in (("selected_outer_statistics", "fold1", "fold1_held_observed"),
                             ("selected_outer_clean_statistics", "fold0", "fold0_held_clean"),
                             ("selected_fitting_diagnostics", "1", "fold1_fitting_observed")):
        panel = copy.deepcopy(selection[key])
        panel[name]["dynamic_sse"] *= 1 + 1e-9
        changed = WF.relock(selection, **{key: panel})
        relabelled = {**states, "selection_sha256": changed["selection_sha256"]}
        again = core.replay_head(cb, core.HEAD, changed, relabelled, refs, device="cpu", recipe=RECIPE,
                                 scope="selected_best")
        assert not again["parity_passed"]
        assert [p["panel"] for p in again["parity"] if not p["passed"]] == [label]


# ----------------------------------------------------------------- T4: wiring of the reading rule in run_d1
def _d0_then(monkeypatch, world, root):
    TD._patch_entry(monkeypatch, world, root)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 0
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d1"])


def test_d1_passes_the_selected_recoveries_late_flag_and_gates_to_the_reading_rule(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    _d0_then(monkeypatch, world, root)
    calls, original = [], core.reading_rule
    monkeypatch.setattr(core, "reading_rule", lambda *a: calls.append(a) or original(*a))
    assert entry.main(["d1"]) == 0
    result = json.loads((root / "d1/clean_fit.json").read_text())
    summary = result["clean_fit"]["selected_summary"]
    expected = {split: {n: summary[split]["recovery_on_observed"][n]["recovery"] for n in core.PANELS}
                for split in ("held", "fitting")}
    assert result["recovery_held"] == expected["held"] and result["recovery_fitting"] == expected["fitting"]
    assert expected["held"] != expected["fitting"]
    assert calls == [(expected["held"], expected["fitting"], result["clean_fit"]["late"], True)]
    assert result["reading"] == original(*calls[0])
    assert result["clean_fit"]["lr"] == entry.LEARNING_RATE


def test_d1_reading_is_not_evaluable_when_a_gate_after_the_fit_fails(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    _d0_then(monkeypatch, world, root)
    original = core.clean_fit
    monkeypatch.setattr(core, "clean_fit", lambda *a, **k: {**original(*a, **k), "selected_reproduced": False})
    assert entry.main(["d1"]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS["d1"] + ".json")).read_text())
    assert receipt["status"] == "gate_failed" and receipt["gates"]["selected_point_reproduced"] is False
    assert json.loads((root / "d1/clean_fit.json").read_text())["reading"] == {"priority": 0, "outcome": "not_evaluable"}


# ----------------------------------------------------------------- T5: held-oracle self-check, negative
def test_held_oracle_self_check_fails_closed_and_stops_both_stages(world, tmp_path, monkeypatch):
    cb = world["bundle"]
    aa = copy.deepcopy(world["artifacts"])
    for fold in (0, 1):  # both folds scaled alike: gate records stay diagnostic, the archive no longer matches the arrays
        aa[fold]["outer_reference_statistics"]["temporal"]["oracle"]["dynamic_sse"] *= 1 + 1e-6
    refs = core.reference_statistics(cb, aa)
    assert not refs["held_oracle_self_check_passed"]
    assert [c["within_reference_tolerance"] for c in refs["held_oracle_self_check"]] == [False, False]
    assert core.REFERENCE_TOLERANCE == 1e-9 and core.TOLERANCE == 1e-12
    assert core.close(1 + 1.5e-12, 1) and not core.close(1 + 3e-12, 1)

    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    context = TD._context(world)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 0            # a completed D0 with the good references
    context["references"] = refs
    monkeypatch.setattr(entry, "setup", lambda: context)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d1"])
    monkeypatch.setattr(core, "clean_fit", lambda *a, **k: pytest.fail("clean fit must not run"))
    assert entry.main(["d1"]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS["d1"] + ".json")).read_text())
    # Today the receipt of a D1 stopped before the fit has no gates (defect T-6); with the two-line fix it lists them.
    assert receipt["status"] == "gate_failed"
    assert receipt["gates"] in ({}, {"driver_parity": True, "held_oracle_self_check": False})

    root = tmp_path.resolve() / "diag2"
    monkeypatch.setattr(entry, "OUTPUT_ROOT", str(root))
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS["d0"] + ".json")).read_text())
    assert receipt["status"] == "gate_failed" and receipt["gates"]["held_oracle_self_check"] is False
    assert receipt["gates"]["state_parity"] is True


# ----------------------------------------------------------------- T6: a fixture with the shape of the real calendar
def unequal_cache():
    """Four users, 7/4/7/7 windows: fitting folds of 14 and 11 windows -> 4 and 3 steps per epoch at batch 4."""
    c = WF.cache(windows=7)
    users = [m["user"] for m in c["meta"]]
    drop = [i for i, u in enumerate(users) if u == "u1"][-3:]
    keep = np.asarray([i for i in range(len(users)) if i not in drop])
    out = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    out["meta"] = [c["meta"][i] for i in keep]
    return out


@pytest.fixture(scope="module")
def calendar_world():
    b = WF.bundle(unequal_cache())
    aa = TD.control_artifacts(b)
    cb = WF.control_bundle(b, aa, "temporal")
    run = W._run_selection(cb, core.HEAD, recipe=RECIPE)
    return {"bundle": cb, "artifacts": aa, "run": run, "references": core.reference_statistics(cb, aa)["references"]}


def test_real_calendar_shape_fractional_exposures_unequal_folds_and_partial_epochs(calendar_world):
    cb, run, refs = calendar_world["bundle"], calendar_world["run"], calendar_world["references"]
    selection, states = run["selection"], run["states"]
    sets = [W._training_set(cb, ix, "post", fold) for fold, ix in enumerate(cb.fitting)]
    steps = [int(np.ceil(d.size / RECIPE.batch_size)) for d in sets]
    assert steps[0] != steps[1] and min(steps) > 1
    first = [p["exposure"] for p in selection["curves"][0]["points"]][:5]
    assert first == [0.0, 0.25, 0.5, 0.75, 1.0]
    replay = core.replay_head(cb, core.HEAD, selection, states, refs, device="cpu", recipe=RECIPE)
    assert replay["parity_passed"] and len(replay["records"]) == 2 * len(selection["curves"])
    for curve in selection["curves"]:
        parity = core.first_epoch_parity(cb, selection, states, refs, device="cpu", recipe=RECIPE, lr=curve["lr"])
        assert parity["passed"], parity
        assert [p["exposure"] for p in parity["points"]] == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert [p["fold_steps"] for p in parity["points"]][-1] == steps
    lr = selection["selected"]["lr"]
    fit = core.clean_fit(cb, selection, refs, device="cpu", recipe=RECIPE, lr=lr, snapshots=(1, 2, 3))
    clean_world = dataclasses.replace(cb, targets=cb.clean_targets)
    clean_sets = [W._training_set(clean_world, ix, "post", fold) for fold, ix in enumerate(cb.fitting)]
    plain_evaluator = lambda model, fold: W._score_model(model, cb, fold, cb.held[fold], "post", "cpu")
    trajectories = [T.Trajectory(core.default_factory(), data, lr=lr, seed=0, fold=fold, device="cpu", _recipe=RECIPE)
                    for fold, data in enumerate(clean_sets)]
    pair = T.PairedCandidate(trajectories, plain_evaluator, fit["evaluation_identity"])
    pair.advance(RECIPE.initial_epochs)
    if S.needs_epoch_extension(pair.curve(), RECIPE.initial_epochs, RECIPE.last_epochs):
        pair.advance(RECIPE.maximum_epochs)  # the frozen search continues in exactly this case, and so does the clean fit
    assert asdict(pair.curve()) == fit["curve"] and fit["selected_reproduced"]
    assert fit["horizon"] == pair.curve().horizon
    for fold in (0, 1):
        assert pair.best_states[fold]["payload_sha256"] == fit["states"]["best_states"][fold]["payload_sha256"]
        assert trajectories[fold].state()["payload_sha256"] == fit["states"]["final_states"][fold]["payload_sha256"]
    for record in fit["records"]:
        for fold, loss in enumerate(record["train_mode_loss"]):
            done = record["point"]["fold_steps"][fold]
            assert (loss is not None) == (done > 0 and done % steps[fold] == 0)
    # Snapshots hold the state of their own exposure, not another one.
    for exposure, pair_states in fit["states"]["snapshots"].items():
        point = next(r["point"] for r in fit["records"] if r["point"]["exposure"] == exposure)
        assert [s["updates"] for s in pair_states] == list(point["fold_steps"])
    assert sorted(fit["states"]["snapshots"]) == [1.0, 2.0, 3.0]
    chosen = next(r for r in fit["records"] if r["point"]["exposure"] == fit["selected"]["exposure"])
    assert chosen["per_fold"] == fit["selected_per_fold"]
    json.dumps(core.plain({k: v for k, v in fit.items() if k != "states"}), allow_nan=False)


def test_first_epoch_parity_detects_another_shuffle_when_an_epoch_has_several_steps(calendar_world, monkeypatch):
    cb, run, refs = calendar_world["bundle"], calendar_world["run"], calendar_world["references"]
    selection, states = run["selection"], run["states"]
    lr = selection["selected"]["lr"]
    original = T.heads.trajectory_seeds
    monkeypatch.setattr(T.heads, "trajectory_seeds",
                        lambda seed, fold: {**original(seed, fold), "shuffle": original(seed, fold)["shuffle"] + 7})
    parity = core.first_epoch_parity(cb, selection, states, refs, device="cpu", recipe=RECIPE, lr=lr)
    assert not parity["passed"] and parity["parity_level"] == "failed"
    failed = {c["check"] for c in parity["checks"] if not c["passed"]}
    assert {"fold0_first_shuffle", "fold1_first_shuffle"} <= failed
    assert any(not p["passed"] for p in parity["points"])


# ----------------------------------------------------------------- T7: a clean fit whose best pair is not the final one
def test_clean_fit_selects_the_best_pair_not_the_final_one_and_is_not_late(world):
    cb, selection, _, refs = TD._tcn(world)
    fit = core.clean_fit(cb, selection, refs, device="cpu", recipe=RECIPE, lr=0.01, snapshots=(1, 2, 3))
    last = fit["curve"]["points"][-1]["exposure"]
    for exposure, pair_states in fit["states"]["snapshots"].items():
        point = next(r["point"] for r in fit["records"] if r["point"]["exposure"] == exposure)
        assert [s["updates"] for s in pair_states] == list(point["fold_steps"])
    assert fit["late"] is False and fit["selected"]["exposure"] == 1.0 < last == RECIPE.initial_epochs
    best, final = fit["states"]["best_states"], fit["states"]["final_states"]
    assert [s["updates"] for s in best] == list(fit["selected"]["fold_steps"]) != [s["updates"] for s in final]
    by_exposure = {r["point"]["exposure"]: r for r in fit["records"]}
    assert fit["selected_per_fold"] == by_exposure[1.0]["per_fold"] != by_exposure[last]["per_fold"]
    assert fit["selected_summary"] == by_exposure[1.0]["summary"] and fit["selected_reproduced"]


# ----------------------------------------------------------------- T8: selected_reproduced fails closed
def test_clean_fit_reports_a_selected_pair_that_does_not_reproduce_its_curve_point(world, monkeypatch):
    cb, selection, _, refs = TD._tcn(world)
    original = core.score_splits

    def shifted(model, bundle, fold, device, depth=core.DEPTH):
        out = original(model, bundle, fold, device, depth)
        out["held"]["observed"] = {**out["held"]["observed"], "mse_sum": out["held"]["observed"]["mse_sum"] * (1 + 1e-9)}
        return out

    monkeypatch.setattr(core, "score_splits", shifted)
    fit = core.clean_fit(cb, selection, refs, device="cpu", recipe=RECIPE, lr=selection["selected"]["lr"])
    assert fit["selected_reproduced"] is False


# ----------------------------------------------------------------- T9: every identity check of the first-epoch parity counts alone
def test_first_epoch_parity_fails_on_each_identity_check_alone(world):
    cb, selection, states, refs = TD._tcn(world)
    lr = selection["selected"]["lr"]
    wrong = {**selection, "initial_sha256_per_fold": ["0" * 64] * 2}
    parity = core.first_epoch_parity(cb, wrong, states, refs, device="cpu", recipe=RECIPE, lr=lr)
    assert not parity["passed"] and parity["parity_level"] == "failed" and all(r["passed"] for r in parity["points"])
    assert [c["check"] for c in parity["checks"] if not c["passed"]] == ["initial_weights"]
    for field, check in (("epoch_order_sha256", "fold1_first_shuffle"), ("epoch_losses", "fold1_first_epoch_train_loss")):
        changed = copy.deepcopy(states)
        final = changed["candidate_states"][lr]["current_states"][1]
        if field == "epoch_order_sha256":
            final[field][0] = "0" * 64
        else:
            final[field][0]["loss_sum"] *= 1 + 1e-9
        parity = core.first_epoch_parity(cb, selection, changed, refs, device="cpu", recipe=RECIPE, lr=lr)
        assert not parity["passed"] and all(r["passed"] for r in parity["points"])
        assert [c["check"] for c in parity["checks"] if not c["passed"]] == [check]
        # A different hash is a failure; a loss beyond the frozen tolerance but within rounding is the numerical tier.
        assert parity["parity_level"] == ("failed" if field == "epoch_order_sha256" else "numerical")
    shifted = copy.deepcopy(selection)
    next(c for c in shifted["curves"] if c["lr"] == lr)["points"][1]["edyn"] *= 1 + 1e-9
    near = core.first_epoch_parity(cb, shifted, states, refs, device="cpu", recipe=RECIPE, lr=lr)
    assert not near["passed"] and near["parity_level"] == "numerical"
    next(c for c in shifted["curves"] if c["lr"] == lr)["points"][1]["edyn"] *= 1 + 1e-3
    assert core.first_epoch_parity(cb, shifted, states, refs, device="cpu", recipe=RECIPE, lr=lr)["parity_level"] == "failed"


# ----------------------------------------------------------------- T10: D0 gates at the entry point
def test_d0_gate_includes_the_per_fold_parity_of_the_other_heads(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    selection = world["runs"]["MLP"]["selection"]
    panel = copy.deepcopy(selection["selected_outer_clean_statistics"])
    panel["fold0"]["dynamic_sse"] *= 1 + 1e-3  # beyond the numerical tier: a real mismatch
    changed = WF.relock(selection, selected_outer_clean_statistics=panel)
    context = TD._context(world)
    context["controls"]["MLP"]["selection"] = changed
    monkeypatch.setattr(entry, "setup", lambda: context)
    monkeypatch.setattr(entry, "_states", lambda c, head: {**world["runs"][head]["states"],
                        "selection_sha256": c["controls"][head]["selection"]["selection_sha256"]})
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS["d0"] + ".json")).read_text())
    assert receipt["status"] == "gate_failed" and receipt["gates"]["state_parity"] is False


def test_d0_fails_when_the_replayed_gate_status_is_not_the_archived_one(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    monkeypatch.setattr(entry, "ARCHIVED_GATE_STATUS", "another_status")
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS["d0"] + ".json")).read_text())
    assert receipt["gates"] == {"state_parity": True, "held_oracle_self_check": True, "gate_arithmetic": False}


# ----------------------------------------------------------------- T11: sign and terms of the held-minus-fitting gap
def test_gap_decomposition_is_held_minus_fitting_term_by_term(world):
    cb, selection, states, refs = TD._tcn(world)
    summary = core.replay_head(cb, core.HEAD, selection, states, refs, device="cpu", recipe=RECIPE,
                               scope="selected_best")["records"][0]["summary"]
    gaps = core.gap_decomposition(summary)
    for name in core.PANELS:
        held, fitting = (summary[s]["noise_alignment"][name]["error_shares"] for s in ("held", "fitting"))
        assert gaps[name] == {k: held[k] - fitting[k] for k in ("E_dyn", "clean_error", "noise_level", "alignment")}
        assert gaps[name]["E_dyn"] == summary["held"]["observed"][name]["E_dyn"] - summary["fitting"]["observed"][name]["E_dyn"] != 0


# ----------------------------------------------------------------- T12: a directory without receipt is refused too
def test_entry_point_refuses_an_existing_stage_directory_without_a_receipt(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    (root / "d0").mkdir(parents=True)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    with pytest.raises(ValueError, match="do not duplicate or retry"):
        entry.main(["d0"])
    assert not (root / "receipts").exists()


# ----------------------------------------------------------------- T13: clean_fit refuses another archive
def test_clean_fit_refuses_another_bundle_or_initialization(world):
    cb, selection, _, refs = TD._tcn(world)
    lr = selection["selected"]["lr"]
    for changed, message in (({"initial_sha256_per_fold": ["0" * 64] * 2}, "archived initialization"),
                             ({"feature_array_sha256": "0" * 64}, "feature array differs"),
                             ({"bundle": {**selection["bundle"], "world": "linear"}}, "rebuilt bundle differs"),
                             ({"evaluation_identity": "0" * 64}, "evaluation identity differs")):
        with pytest.raises(ValueError, match=message):
            core.clean_fit(cb, {**selection, **changed}, refs, device="cpu", recipe=RECIPE, lr=lr)
