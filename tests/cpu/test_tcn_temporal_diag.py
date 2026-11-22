"""Diagnostics of the failed temporal control: parity gates, clean-target fit, reading rule.

The fixtures are the small synthetic control of test_pose_temponet_workflow,
trained with the real readers and the private test recipe. No number here is a
scientific result.
"""
import copy
import dataclasses
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from pose_temponet import controls as C, metrics as M, provenance as P, selection as S, train as T, workflow as W

import test_pose_temponet_workflow as WF
from tcn_temporal_diag import core
import tcn_temporal_diag.__main__ as entry

RECIPE = T._TestRecipe()


def control_artifacts(b):
    """Workflow fixtures plus the two references that the diagnostics read from the real artifacts."""
    aa = WF.artifacts(b)
    cb = WF.control_bundle(b, aa, "temporal")
    for fold, artifact in enumerate(aa):
        outer = artifact["outer_reference_statistics"]
        held, fit = (core.oracle_statistics(cb, fold, ix) for ix in (cb.held[fold], cb.fitting[fold]))
        outer["temporal"]["oracle"] = held
        outer["temporal"]["clean_ls"] = {**held, "dynamic_sse": held["dynamic_sse"] * 1.08,
                                         "mse_sum": held["mse_sum"] * 1.08}
        outer["linear_diagnostics"] = {"L0": {
            "temporal_margin_clean_fit_statistics": {**fit, "dynamic_sse": fit["dynamic_sse"] * 1.08,
                                                     "mse_sum": fit["mse_sum"] * 1.08},
            "temporal_margin_fit_target": "clean_y_star", "temporal_margin_score_target": "observed_y"}}
    return aa


@pytest.fixture(scope="module")
def world():
    b = WF.bundle()
    aa = control_artifacts(b)
    cb = WF.control_bundle(b, aa, "temporal")
    runs = {head: W._run_selection(cb, head, recipe=RECIPE) for head in W.CONTROL_HEADS}
    return {"bundle": cb, "artifacts": aa, "runs": runs, "references": core.reference_statistics(cb, aa)}


def _tcn(world):
    run = world["runs"][core.HEAD]
    return world["bundle"], run["selection"], run["states"], world["references"]["references"]


def test_frozen_reader_identity_is_untouched_by_the_diagnostic_package():
    manifest = P.source_manifest(ROOT)
    assert P.digest(manifest) == entry.SOURCE_SHA256
    assert not any("tcn_temporal_diag" in name for name in manifest)
    plan, pin, code = entry.reviewed_plan()
    assert pin["sha256"] == entry.PLAN_SHA256 and code["reviewed_code_sha"] == entry.REVIEWED_CODE_SHA
    assert [j["id"] for j in plan["jobs"] if j["stage"] == "controls" and j["world"] == "temporal"] == [
        "TCNH-controls-temporal-%s-20261120" % head for head in W.CONTROL_HEADS]
    mirror = ROOT / "results/infra/pose_temponet_execution_20261121_A1/launcher"
    for identifier, expected in entry.RECEIPT_SHA256.items():
        assert P.file_pin(mirror / (identifier + ".json"))["sha256"] == expected


def test_launcher_environment_equals_the_frozen_launcher():
    def environment(name):
        lines = (ROOT / "scripts/slurm" / name).read_text().splitlines()
        start = lines.index('source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"')
        stop = lines.index("export CUBLAS_WORKSPACE_CONFIG=:4096:8")
        return lines[start:stop + 1]
    assert environment("tcn_temporal_diag.sbatch") == environment("pose_temponet.sbatch")
    frozen = (ROOT / "scripts/slurm/pose_temponet_dispatch.sbatch").read_text().splitlines()
    ours = (ROOT / "scripts/slurm/tcn_temporal_diag_dispatch.sbatch").read_text().splitlines()
    assert ours[2:-1] == frozen[2:-1] and ours[-1].endswith('tcn_temporal_diag.sbatch"')


def test_references_come_from_artifacts_and_the_held_oracle_is_rebuilt(world):
    cb, aa, refs = world["bundle"], world["artifacts"], world["references"]
    assert refs["held_oracle_self_check_passed"]
    for fold in (0, 1):
        outer = aa[fold]["outer_reference_statistics"]
        assert refs["references"]["held"]["clean_ls"][fold] == outer["temporal"]["clean_ls"]
        assert refs["references"]["fitting"]["clean_ls"][fold] == \
            outer["linear_diagnostics"]["L0"]["temporal_margin_clean_fit_statistics"]
        assert refs["references"]["fitting"]["oracle"][fold]["window_count"] == \
            int(cb.support[cb.fitting[fold]].any(axis=(1, 2)).sum())
    values = {k: core.panel_values(refs["references"]["held"][k]) for k in ("clean_ls", "oracle")}
    for name in core.PANELS:
        assert {k: values[k][name]["E_dyn"] for k in values} == cb.reference_gate["temporal_records"][name]
    broken = copy.deepcopy(aa)
    broken[0]["outer_reference_statistics"]["linear_diagnostics"]["L0"]["temporal_margin_fit_target"] = "observed_y"
    with pytest.raises(ValueError, match="fitting reference definition"):
        core.reference_statistics(cb, broken)


def test_replay_reproduces_the_archive_and_scores_both_targets_on_both_populations(world):
    cb, selection, states, refs = _tcn(world)
    replay = core.replay_head(cb, core.HEAD, selection, states, refs, device="cpu", recipe=RECIPE)
    assert replay["parity_passed"] and len(replay["records"]) == 2 * len(selection["curves"])
    # One pooled curve point per replayed pair, plus six per-fold accumulator rows of the selected pair.
    assert replay["parity_level"] == "exact" and len(replay["parity"]) == 2 * len(selection["curves"]) + 6
    assert sorted(p["panel"] for p in replay["parity"] if p["panel"] != "pooled_curve_point") == sorted(
        "fold%d_%s" % (fold, label) for fold in (0, 1) for label in ("held_observed", "held_clean", "fitting_observed"))
    selected = [r for r in replay["records"] if r["selected"]]
    assert len(selected) == 1 and selected[0]["kind"] == "best_states"
    record = selected[0]
    for fold in (0, 1):
        assert record["per_fold"][fold]["held"]["observed"] == selection["selected_outer_statistics"]["fold%d" % fold]
        assert record["per_fold"][fold]["held"]["clean"] == selection["selected_outer_clean_statistics"]["fold%d" % fold]
        assert record["per_fold"][fold]["fitting"]["observed"] == selection["selected_fitting_diagnostics"][str(fold)]
    gate = core.frozen_gate(record["summary"], cb)
    gaps = core.gap_decomposition(record["summary"])
    references = world["references"]["references"]
    for fold in (0, 1):  # every split is read against its own references and its own oracle
        for split in ("held", "fitting"):
            name, scores = "fold%d" % fold, record["per_fold"][fold][split]
            e = {k: C.statistic_values(v)["E_dyn"] for k, v in (
                ("reader", scores["observed"]), ("clean_ls", references[split]["clean_ls"][fold]),
                ("oracle", references[split]["oracle"][fold]))}
            assert record["summary"][split]["recovery_on_observed"][name]["recovery"] == \
                C._recovery(e["clean_ls"], e["reader"], e["oracle"])
            assert record["summary"][split]["noise_alignment"][name]["noise_sse"] == \
                references[split]["oracle"][fold]["dynamic_sse"]
        assert references["fitting"]["oracle"][fold] != references["held"]["oracle"][fold]
    held_edyn, fit_edyn = (record["summary"][split]["observed"]["pooled"]["E_dyn"] for split in ("held", "fitting"))
    assert gaps["pooled"]["E_dyn"] == pytest.approx(held_edyn - fit_edyn, abs=1e-15) and held_edyn != fit_edyn
    for name in core.PANELS:
        assert record["summary"]["held"]["recovery_on_observed"][name]["recovery"] == gate["recovery"][name]
        assert gaps[name]["clean_error"] + gaps[name]["noise_level"] + gaps[name]["alignment"] == \
            pytest.approx(gaps[name]["E_dyn"], abs=1e-12)
        for split in ("held", "fitting"):
            shares = record["summary"][split]["noise_alignment"][name]["error_shares"]
            assert shares["E_dyn"] == record["summary"][split]["observed"][name]["E_dyn"]
    finals = [r for r in replay["records"] if r["kind"] == "current_states"]
    assert all(len(r["train_mode_epoch_losses"]) == 2 and
               len(r["train_mode_epoch_losses"][0]) == r["exposure"] for r in finals)
    for head in ("L0", "MLP"):
        run = world["runs"][head]
        other = core.replay_head(cb, head, run["selection"], run["states"], refs, device="cpu", recipe=RECIPE,
                                 scope="selected_best")
        assert other["parity_passed"] and len(other["records"]) == 1 and other["records"][0]["selected"]
    json.dumps(core.plain(replay), allow_nan=False)


def test_replay_fails_closed_on_any_changed_archive(world):
    cb, selection, states, refs = _tcn(world)
    curves = copy.deepcopy(selection["curves"])
    victim = next(c for c in curves if c["lr"] != selection["selected"]["lr"])
    victim["points"][-1]["edyn"] *= 1 + 1e-9
    changed = WF.relock(selection, curves=curves)
    relabelled = {**states, "selection_sha256": changed["selection_sha256"]}
    replay = core.replay_head(cb, core.HEAD, changed, relabelled, refs, device="cpu", recipe=RECIPE)
    # Beyond the frozen tolerance but inside the registered numerical tier: reported as such, never as exact.
    assert not replay["parity_passed"] and replay["parity_level"] == "numerical"
    worse = copy.deepcopy(curves)
    next(c for c in worse if c["lr"] == victim["lr"])["points"][-1]["edyn"] *= 1 + 1e-3
    broken = WF.relock(selection, curves=worse)
    assert core.replay_head(cb, core.HEAD, broken, {**states, "selection_sha256": broken["selection_sha256"]}, refs,
                            device="cpu", recipe=RECIPE)["parity_level"] == "failed"
    failed = [p for p in replay["parity"] if not p["passed"]]
    # Only the changed point fails: the final state, and the best state when it is that same point.
    assert failed and "current_states" in {p["kind"] for p in failed}
    assert all(p["lr"] == victim["lr"] and p["exposure"] == victim["points"][-1]["exposure"] for p in failed)
    with pytest.raises(ValueError, match="states do not belong"):
        core.replay_head(cb, core.HEAD, changed, states, refs, device="cpu", recipe=RECIPE)
    tampered = copy.deepcopy(states)
    lr = selection["selected"]["lr"]
    next(iter(tampered["candidate_states"][lr]["best_states"][0]["model"].values())).add_(1e-3)
    with pytest.raises(ValueError, match="checksum"):
        core.replay_head(cb, core.HEAD, selection, tampered, refs, device="cpu", recipe=RECIPE)
    other_targets = dataclasses.replace(cb, targets=cb.clean_targets)
    with pytest.raises(ValueError, match="synthetic targets changed|provenance"):
        core.replay_head(other_targets, core.HEAD, selection, states, refs, device="cpu", recipe=RECIPE)


def test_branch_probe_only_reads_and_counts_what_a_manual_forward_counts(world):
    cb = world["bundle"]
    fold, indices = 1, cb.held[1]
    model = core.default_factory()(0, fold)
    with torch.no_grad():  # a reader that is not at its all-zero output initialization
        for parameter in model.parameters():
            parameter.add_(.05 * torch.randn(parameter.shape, generator=torch.Generator().manual_seed(parameter.numel())))
        hidden = [layer for layer in model.deep if isinstance(layer, torch.nn.Conv1d)][4]
        hidden.weight[3].zero_()
        hidden.bias[3] = -1.
    plain_scores = W._score_model(model, cb, fold, indices, "post", "cpu", include_clean=True)
    probed = core.score_probed(model, cb, fold, indices, "cpu")
    assert (probed["observed"], probed["clean"]) == plain_scores
    assert all(not module._forward_hooks for module in model.modules())
    layers = probed["branches"]["relu_layers"]
    assert [row["units"] for row in layers] == [64, 64, 64, 64, 32]
    assert layers[-1]["dead_units"] == 1 and layers[-1]["least_active_unit_fraction"] == 0
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(W.D.view_inputs(cb.cache, "post", indices))
        h = model.norm(x)
        value, expected = h.transpose(1, 2), []
        for module in model.deep:
            value = module(value)
            if isinstance(module, torch.nn.ReLU):
                expected.append(float((value > 0).double().mean()))
        deep, affine = value.transpose(1, 2)[:, 6:-6].double(), model.affine(h)[:, 6:-6].double()
    assert [row["active_fraction"] for row in layers] == pytest.approx(expected, rel=1e-12)
    assert probed["branches"]["deep_mean_square"] == pytest.approx(float((deep ** 2).mean()), rel=1e-9)
    assert probed["branches"]["affine_mean_square"] == pytest.approx(float((affine ** 2).mean()), rel=1e-9)
    assert probed["branches"]["affine_deep_mean_product"] == pytest.approx(float((affine * deep).mean()), rel=1e-9)
    assert probed["branches"]["elements"] == deep.numel()
    centred = deep - deep.mean(dim=1, keepdim=True)
    assert probed["branches"]["deep_centred_mean_square"] == pytest.approx(float((centred ** 2).mean()), rel=1e-9)
    assert 0 < probed["branches"]["deep_share_of_centred_energy"] < 1
    with torch.no_grad():  # a deep branch that only outputs its bias has no time-centred energy at all
        dead = core.default_factory()(0, fold)
        list(dead.deep)[-1].bias.fill_(.3)
        dead.affine.weight.normal_(generator=torch.Generator().manual_seed(1))
    silent = core.score_probed(dead, cb, fold, indices, "cpu")["branches"]
    assert silent["deep_mean_square"] == pytest.approx(.09, rel=1e-6) and silent["deep_centred_mean_square"] < 1e-12
    assert silent["deep_share_of_centred_energy"] < 1e-9
    drift = core.layer_norm_drift(model)
    assert drift["weight_rms_deviation_from_one"] > 0 and drift["bias_rms"] > 0
    # L0 has no deep branch; the MLP is channel-last.
    for head, units in (("L0", None), ("MLP", [96, 96, 96, 64, 32])):
        other = core.score_probed(core.default_factory(head)(0, fold), cb, fold, indices, "cpu")["branches"]
        assert (other is None) if units is None else ([row["units"] for row in other["relu_layers"]] == units)


def test_teacher_written_into_the_reader_reproduces_the_generated_clean_function():
    rng = np.random.default_rng(20261122)
    x = rng.standard_normal((6, 160, 384)).astype(np.float16)
    meta = [{"user": "user%d" % (i // 2), "window": i} for i in range(6)]
    support = np.ones((6, 148, 20), dtype=bool)
    components = C.generate_components(x, meta, support, np.array([0, 1, 2, 3]), np.array([4, 5]),
                                       ["user0", "user1", "user2"], outer_fold=1)
    artifact = {"components": components, "calibration": {"selected": {"temporal": {"a": .4, "sigma": 1.3}}}}
    model = core.teacher_reader(artifact)
    assert not model.training
    with torch.no_grad(), core.branch_probe(model) as probe:
        prediction = model(torch.from_numpy(x).float()).numpy()
    expected = components["L"] + .4 * components["T"]
    error = prediction[:, 6:-6] - expected[:, 6:-6]
    assert np.abs(error).max() < 1e-4 and np.sqrt((error ** 2).mean() / (expected[:, 6:-6] ** 2).mean()) < 1e-5
    layers = probe.summary()["relu_layers"]
    # Only the first eight channels of each hidden layer carry the generator.
    assert [row["units"] - row["dead_units"] <= 8 for row in layers] == [True] * 5
    assert core.generator_summary([{**artifact, "outer_fold": 0}])[0]["hidden_active_fraction_fitting"] == \
        pytest.approx(components["hidden_active_fraction_fitting"])
    with pytest.raises(ValueError, match="depths differ"):
        broken = {**artifact, "components": {**components, "generator_states": {
            "L": components["generator_states"]["L"],
            "T": {k: v for k, v in components["generator_states"]["T"].items() if not k.startswith("0.")}}}}
        core.teacher_reader(broken)


def test_generator_summary_and_clean_scale_reference_are_descriptive_extras(world):
    cb, aa = world["bundle"], world["artifacts"]
    summary = core.generator_summary(aa)
    assert [row["outer_fold"] for row in summary] == [0, 1] and summary[0]["selected"] == {"a": .3, "sigma": .8}
    assert summary[0]["hidden_active_fraction_fitting"] is None  # the small fixtures carry bare components
    first = core.clean_scale_references(cb, aa, block_windows=4)
    assert not first["self_check_passed"]  # fixture references are placeholders, not this clean-LS
    assert core.clean_scale_panels([], first, "held") is None
    real = copy.deepcopy(aa)
    for fold, artifact in enumerate(real):
        outer = artifact["outer_reference_statistics"]
        outer["temporal"]["clean_ls"] = first["per_fold"][fold]["held"]["observed"]
        outer["linear_diagnostics"]["L0"]["temporal_margin_clean_fit_statistics"] = \
            first["per_fold"][fold]["fitting"]["observed"]
    second = core.clean_scale_references(cb, real, block_windows=4)
    assert second["self_check_passed"] and second["per_fold"] == first["per_fold"]
    for fold in (0, 1):
        for split, indices in (("held", cb.held[fold]), ("fitting", cb.fitting[fold])):
            row = second["per_fold"][fold][split]
            assert row["clean"]["window_count"] == row["observed"]["window_count"]
            assert row["clean"]["dynamic_sse"] < row["observed"]["dynamic_sse"]
    references = {**world["references"]["references"], "clean_scale": second}
    run = world["runs"][core.HEAD]
    replay = core.replay_head(cb, core.HEAD, run["selection"], run["states"], references, device="cpu", recipe=RECIPE,
                              scope="selected_best")
    scale = replay["records"][0]["summary"]["held"]["clean_scale"]
    reader = replay["records"][0]["per_fold"][0]["held"]["clean"]["dynamic_sse"]
    linear = second["per_fold"][0]["held"]["clean"]["dynamic_sse"]
    assert scale["fold0"]["share_of_clean_ls_error_removed"] == pytest.approx(1 - reader / linear, rel=1e-12)
    assert set(scale) == set(core.PANELS)
    without = core.replay_head(cb, core.HEAD, run["selection"], run["states"], world["references"]["references"],
                               device="cpu", recipe=RECIPE, scope="selected_best")
    assert without["records"][0]["summary"]["held"]["clean_scale"] is None


def test_noise_alignment_equals_the_direct_cross_term_of_the_frozen_metric(world):
    cb = world["bundle"]
    fold, indices = 0, world["bundle"].held[0]
    observed_panel, clean_panel = W._metric_panel(cb, fold, indices), W._metric_panel(cb, fold, indices, clean=True)
    threshold = cb.thresholds[fold]
    clean = np.asarray(cb.clean_targets[fold])[indices][:, 6:-6]
    noisy = np.asarray(cb.targets[fold])[indices][:, 6:-6]
    prediction = clean + .3 * np.random.default_rng(5).normal(size=clean.shape) + .2 * (noisy - clean)
    stats = {name: M.selection_statistics(p, panel, threshold) for name, p, panel in (
        ("observed", prediction, observed_panel), ("clean", prediction, clean_panel), ("oracle", clean, observed_panel))}
    measured = core.noise_alignment(stats["observed"], stats["clean"], stats["oracle"])
    _, _, support = M._cropped(prediction, observed_panel)
    count = support.sum(axis=1)

    def centred(values):
        mean = np.where(support, values, 0).sum(axis=1) / np.maximum(count, 1)
        return np.where(support, values - mean[:, None, :], 0)

    dynamic = np.isfinite(observed_panel["pose_var"]) & (observed_panel["pose_var"] >= threshold)
    # The frozen accumulators are element-weighted sums over window/joint pairs with at least two valid samples.
    per_window_joint = (centred(prediction - clean) * centred(noisy - clean)).sum(axis=1)
    direct = float(np.where(count >= 2, per_window_joint, 0)[dynamic].sum())
    assert direct > 0 and measured["cross_term"] == pytest.approx(direct, rel=1e-9)
    shares = core.error_shares(stats["observed"], stats["clean"], stats["oracle"])
    assert shares["clean_error"] + shares["noise_level"] + shares["alignment"] == pytest.approx(shares["E_dyn"], rel=1e-12)
    assert shares["E_dyn"] == C.statistic_values(stats["observed"])["E_dyn"] and shares["alignment"] < 0
    # Limits: the clean function ignores the noise, the observed target reproduces it entirely.
    oracle = stats["oracle"]
    zero = M.selection_statistics(clean, clean_panel, threshold)
    assert core.noise_alignment(oracle, zero, oracle)["noise_fit_fraction"] == pytest.approx(0, abs=1e-12)
    perfect = {name: M.selection_statistics(noisy, panel, threshold)
               for name, panel in (("observed", observed_panel), ("clean", clean_panel))}
    assert core.noise_alignment(perfect["observed"], perfect["clean"], oracle)["noise_fit_fraction"] == \
        pytest.approx(1, rel=1e-9)


def test_first_epoch_parity_retraces_the_archive_and_rejects_another_training_target(world):
    cb, selection, states, refs = _tcn(world)
    for curve in selection["curves"][:2]:
        parity = core.first_epoch_parity(cb, selection, states, refs, device="cpu", recipe=RECIPE, lr=curve["lr"])
        assert parity["passed"] and all(c["passed"] for c in parity["checks"])
        assert [p["exposure"] for p in parity["points"]] == [p["exposure"] for p in curve["points"] if p["exposure"] <= 1]
    lr = selection["selected"]["lr"]
    clean_world = dataclasses.replace(cb, targets=cb.clean_targets)
    parity = core.first_epoch_parity(clean_world, selection, states, refs, device="cpu", recipe=RECIPE, lr=lr)
    assert not parity["passed"] and parity["parity_level"] == "failed"
    assert not next(c for c in parity["checks"] if c["check"] == "training_set_identity")["passed"]
    shifted = copy.deepcopy(selection)
    next(c for c in shifted["curves"] if c["lr"] == lr)["points"][1]["mse"] *= 1 + 1e-9
    near = core.first_epoch_parity(cb, shifted, states, refs, device="cpu", recipe=RECIPE, lr=lr)
    assert not near["passed"] and near["parity_level"] == "numerical"
    next(c for c in shifted["curves"] if c["lr"] == lr)["points"][1]["mse"] *= 1 + 1e-3
    assert core.first_epoch_parity(cb, shifted, states, refs, device="cpu", recipe=RECIPE, lr=lr)["parity_level"] == "failed"
    # A discrete identity that differs is a failure whatever the scores say: no numerical tier for hashes.
    other_init = {**selection, "initial_sha256_per_fold": ["0" * 64, "0" * 64]}
    wrong = core.first_epoch_parity(cb, other_init, states, refs, device="cpu", recipe=RECIPE, lr=lr)
    assert all(row["passed"] for row in wrong["points"]) and wrong["parity_level"] == "failed" and not wrong["passed"]


def test_noisy_extension_continues_the_archived_trajectory_exactly(world):
    cb, selection, states, refs = _tcn(world)
    short = [c for c in selection["curves"] if c["horizon"] == RECIPE.initial_epochs]
    assert short, "the fixture needs one learning rate that stopped at the initial horizon"
    lr = short[0]["lr"]
    extension = core.extend_noisy_fit(cb, selection, states, refs, device="cpu", recipe=RECIPE, lr=lr,
                                      snapshots=(RECIPE.maximum_epochs, 31))
    assert extension["start_parity"]["passed"] and extension["start_exposure"] == RECIPE.initial_epochs
    for fold in (0, 1):  # the archived states are left intact by the continued training
        archived_state = dict(states["candidate_states"][lr]["current_states"][fold])
        assert T.payload_sha256({k: v for k, v in archived_state.items() if k != "payload_sha256"}) == \
            archived_state["payload_sha256"]
    assert extension["new_points"] == RECIPE.maximum_epochs - RECIPE.initial_epochs
    assert extension["records"][0]["point"]["exposure"] == RECIPE.initial_epochs
    assert {k: (list(v) if k == "fold_steps" else v) for k, v in extension["records"][0]["point"].items()} == \
        short[0]["points"][-1]
    # The continuation equals one uninterrupted run of the frozen trajectories on the same observed target.
    sets = [W._training_set(cb, ix, "post", fold) for fold, ix in enumerate(cb.fitting)]
    plain_evaluator = lambda model, fold: W._score_model(model, cb, fold, cb.held[fold], "post", "cpu")
    trajectories = [T.Trajectory(core.default_factory(), data, lr=lr, seed=0, fold=fold, device="cpu", _recipe=RECIPE)
                    for fold, data in enumerate(sets)]
    points = []
    for exposure, fold_steps in S.checkpoint_schedule([t.steps_per_epoch for t in trajectories], RECIPE.maximum_epochs):
        for trajectory, step in zip(trajectories, fold_steps):
            trajectory.train_to_step(step)
        points.append(asdict(S.pool_scores([t.evaluate(plain_evaluator) for t in trajectories], exposure, fold_steps)))
    assert [r["point"] for r in extension["records"]] == points[-len(extension["records"]):]
    for fold in (0, 1):
        assert trajectories[fold].state()["payload_sha256"] == \
            extension["states"]["final_states"][fold]["payload_sha256"]
    assert extension["snapshot_exposures"] == [float(RECIPE.maximum_epochs)]
    assert extension["minimum_mse_new_point"] == min(points[-extension["new_points"]:],
                                                      key=lambda q: (q["mse"], q["exposure"]))
    assert extension["minimum_new_pooled_edyn"] == min(q["edyn"] for q in points[-extension["new_points"]:])
    start, end = extension["fitting_noise_fit_fraction_start_end"]
    assert start == extension["records"][0]["summary"]["fitting"]["noise_alignment"]["pooled"]["noise_fit_fraction"]
    assert end == extension["records"][-1]["summary"]["fitting"]["noise_alignment"]["pooled"]["noise_fit_fraction"]
    # A changed archive stops before any new training.
    shifted = copy.deepcopy(selection)
    next(c for c in shifted["curves"] if c["lr"] == lr)["points"][-1]["mse"] *= 1 + 1e-3
    stopped = core.extend_noisy_fit(cb, shifted, states, refs, device="cpu", recipe=RECIPE, lr=lr)
    assert stopped["start_parity"]["parity_level"] == "failed" and stopped["records"] == [] and stopped["states"] is None
    done = next(c for c in selection["curves"] if c["horizon"] == RECIPE.maximum_epochs)
    with pytest.raises(ValueError, match="frozen cap"):
        core.extend_noisy_fit(cb, selection, states, refs, device="cpu", recipe=RECIPE, lr=done["lr"])


def test_clean_fit_changes_only_the_training_target(world):
    cb, selection, _, refs = _tcn(world)
    lr = selection["selected"]["lr"]
    fit = core.clean_fit(cb, selection, refs, device="cpu", recipe=RECIPE, lr=lr, snapshots=(1, 3, 31))
    clean_world = dataclasses.replace(cb, targets=cb.clean_targets)
    clean_sets = [W._training_set(clean_world, ix, "post", fold) for fold, ix in enumerate(cb.fitting)]
    noisy_sets = [W._training_set(cb, ix, "post", fold) for fold, ix in enumerate(cb.fitting)]
    assert fit["training_set_sha256"] == [d.identity for d in clean_sets] != [d.identity for d in noisy_sets]
    assert fit["initial_sha256_per_fold"] == selection["initial_sha256_per_fold"]
    # Reference run from the frozen pieces only: no extra evaluation, plain held evaluator on observed y.
    plain_evaluator = lambda model, fold: W._score_model(model, cb, fold, cb.held[fold], "post", "cpu")
    trajectories = [T.Trajectory(core.default_factory(), data, lr=lr, seed=0, fold=fold, device="cpu", _recipe=RECIPE)
                    for fold, data in enumerate(clean_sets)]
    pair = T.PairedCandidate(trajectories, plain_evaluator, fit["evaluation_identity"])
    pair.advance(RECIPE.initial_epochs)
    extended = S.needs_epoch_extension(pair.curve(), RECIPE.initial_epochs, RECIPE.last_epochs)
    if extended:  # the frozen search continues to the maximum horizon in exactly this case
        pair.advance(RECIPE.maximum_epochs)
    assert asdict(pair.curve()) == fit["curve"] and fit["extended_by_frozen_rule"] == extended
    assert fit["horizon"] == (RECIPE.maximum_epochs if extended else RECIPE.initial_epochs)
    for fold in (0, 1):
        assert pair.best_states[fold]["payload_sha256"] == fit["states"]["best_states"][fold]["payload_sha256"]
        assert trajectories[fold].state()["payload_sha256"] == fit["states"]["final_states"][fold]["payload_sha256"]
    # Selection: minimum pooled MSE against observed y, positive exposure, first on ties.
    points = [S.pool_scores([f["held"]["observed"] for f in r["per_fold"]], r["point"]["exposure"],
                            r["point"]["fold_steps"]) for r in fit["records"]]
    assert [asdict(p) for p in points] == [r["point"] for r in fit["records"]] == list(fit["curve"]["points"])
    best = min((p for p in points if p.exposure > 0), key=lambda p: (p.mse, p.exposure))
    assert asdict(best) == fit["selected"] and fit["selected_reproduced"]
    assert fit["minimum_exposures"] == {m: pair.curve().best(m).exposure for m in ("mse", "edyn")}
    edge = RECIPE.maximum_epochs - RECIPE.last_epochs + 1
    assert fit["late"] == (extended and max(fit["minimum_exposures"].values()) >= edge)
    chosen = next(r for r in fit["records"] if r["point"]["exposure"] == fit["selected"]["exposure"])
    assert chosen["per_fold"] == fit["selected_per_fold"] and chosen["summary"] == fit["selected_summary"]
    assert fit["snapshot_exposures"] == [1.0, 3.0] and set(fit["states"]["snapshots"]) == {1.0, 3.0}
    assert len(fit["records"]) == len(fit["curve"]["points"]) == len(pair.points)
    # Same initialization and same scorer: the step-zero point is the archived one; later points are not.
    archived = next(c for c in selection["curves"] if c["lr"] == lr)["points"]
    ours = [{**q, "fold_steps": list(q["fold_steps"])} for q in fit["curve"]["points"]]
    assert ours[0] == archived[0] and ours[1:] != archived[1:len(ours)]
    again = core.clean_fit(cb, selection, refs, device="cpu", recipe=RECIPE, lr=lr)
    assert again["curve"] == fit["curve"] and again["selected_per_fold"] == fit["selected_per_fold"]
    readings = core.fixed_readings(fit, (1, 3, 31))
    assert list(readings) == ["1.0", "3.0", "selected"]
    json.dumps(core.plain({k: v for k, v in fit.items() if k != "states"}), allow_nan=False)


def test_clean_fit_reports_a_selected_pair_that_does_not_reproduce_its_point(world, monkeypatch):
    cb, selection, _, refs = _tcn(world)
    original = core.score_splits

    def drifting(model, bundle, fold, device, depth=core.DEPTH):
        scores = original(model, bundle, fold, device, depth)
        scores["held"]["observed"] = {**scores["held"]["observed"],
                                      "mse_sum": scores["held"]["observed"]["mse_sum"] * (1 + 1e-9)}
        return scores

    monkeypatch.setattr(core, "score_splits", drifting)
    fit = core.clean_fit(cb, selection, refs, device="cpu", recipe=RECIPE, lr=selection["selected"]["lr"])
    assert fit["selected_reproduced"] is False


def test_reading_rule_is_ordered_exhaustive_and_uses_the_frozen_threshold():
    threshold = C.TEMPORAL_HEAD_RECOVERY
    assert threshold == 0.5
    panel = lambda values: dict(zip(core.PANELS, values))
    seen = set()
    for held in itertools.product((threshold - .01, threshold), repeat=3):
        for fitting in itertools.product((threshold - .01, threshold), repeat=3):
            for late in (False, True):
                result = core.reading_rule(panel(held), panel(fitting), late, True)
                passed_held, passed_fit = [h >= threshold for h in held], [f >= threshold for f in fitting]
                if all(passed_held):
                    expected = (1, "sensitivity_recovered_when_trained_on_clean_target")
                elif late:
                    expected = (2, "unresolved_within_horizon")
                elif all(passed_fit):
                    expected = (3, "3a_fitting_reaches_threshold_but_not_every_held_panel")
                elif not any(passed_fit):
                    expected = (3, "3b_recovery_insufficient_also_on_fitting")
                else:
                    expected = (3, "3c_heterogeneous_fitting_panels")
                assert (result["priority"], result["outcome"]) == expected and result["late"] == late
                seen.add(expected)
    assert len(seen) == 5
    good = panel((.6, .6, .6))
    assert core.reading_rule(good, good, False, False) == {"priority": 0, "outcome": "not_evaluable"}
    for bad in (None, float("nan"), float("inf")):
        assert core.reading_rule(panel((.6, bad, .6)), good, False, True)["priority"] == 0
        assert core.reading_rule(good, panel((.6, .6, bad)), False, True)["priority"] == 0


def _context(world):
    cb = world["bundle"]
    selection = world["runs"][core.HEAD]["selection"]
    return {"plan": {}, "plan_pin": {}, "code": {"code_sha": "c" * 40, "reviewed_code_sha": entry.REVIEWED_CODE_SHA},
            "numeric": {"kind": "test"}, "pins": {k: {"path": "/" + k, "sha256": v, "bytes": 1}
                                                   for k, v in entry.RECEIPT_SHA256.items()},
            "bundle": cb, "calibrated": world["artifacts"],
            "controls": {head: {"receipt": {"job_id": head}, "selection": world["runs"][head]["selection"]}
                         for head in W.CONTROL_HEADS},
            "evaluation_identity": core.check_bundle(cb, selection),
            "references": {**world["references"], "generator": core.generator_summary(world["artifacts"]),
                           "references": {**world["references"]["references"],
                                          "clean_scale": core.clean_scale_references(cb, world["artifacts"],
                                                                                     block_windows=4)}}}


def _patch_entry(monkeypatch, world, root):
    selection = world["runs"][core.HEAD]["selection"]
    record = core.replay_head(world["bundle"], core.HEAD, selection, world["runs"][core.HEAD]["states"],
                              world["references"]["references"], device="cpu", recipe=RECIPE)["records"]
    status = core.frozen_gate(next(r for r in record if r["selected"])["summary"], world["bundle"])["status"]
    gate = core.frozen_gate(next(r for r in record if r["selected"])["summary"], world["bundle"])
    for name, value in (("OUTPUT_ROOT", str(root)), ("DEVICE", "cpu"), ("RECIPE", RECIPE),
                        ("LEARNING_RATE", selection["selected"]["lr"]), ("READING_EXPOSURES", (1, 2, 3)),
                        ("EXTENSION_SNAPSHOTS", (RECIPE.maximum_epochs,)),
                        ("ARCHIVED_GATE_STATUS", status), ("ARCHIVED_RECOVERY", gate["recovery"])):
        monkeypatch.setattr(entry, name, value)
    monkeypatch.setattr(entry, "allocation", lambda: {"slurm_job_id": "1", "scontrol": "test"})
    monkeypatch.setattr(entry, "setup", lambda: _context(world))
    monkeypatch.setattr(entry, "_states", lambda context, head: world["runs"][head]["states"])


def test_entry_point_writes_receipts_chains_d1_after_d0_and_never_retries(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    _patch_entry(monkeypatch, world, root)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d1"])
    with pytest.raises(FileNotFoundError):
        entry.main(["d1"])
    failed = json.loads((root / "receipts" / (entry.JOBS["d1"] + ".json")).read_text())
    assert failed["status"] == "technical_failure" and not failed["complete"] and not (root / "d1").exists()
    with pytest.raises(ValueError, match="do not duplicate or retry"):
        entry.main(["d1"])

    root = tmp_path.resolve() / "diag2"
    monkeypatch.setattr(entry, "OUTPUT_ROOT", str(root))
    with pytest.raises(ValueError, match="logical job ID"):
        entry.main(["d0"])
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 0
    receipt = json.loads((root / "receipts" / (entry.JOBS["d0"] + ".json")).read_text())
    assert receipt["status"] == "completed" and receipt["complete"] and all(receipt["gates"].values())
    assert receipt["receipt_sha256"] == P.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    replay = json.loads((root / "d0/replay.json").read_text())
    assert P.file_pin(root / "d0/replay.json")["sha256"] == receipt["artifacts"]["replay"]["sha256"]
    assert set(replay["replays"]) == set(W.CONTROL_HEADS) and replay["gates"] == receipt["gates"]
    assert set(replay["selected_comparison"]["held_minus_fitting_selected"]) == set(core.PANELS)
    # The small fixtures carry no generator states: the descriptive extra records its own failure and gates nothing.
    assert "error" in replay["teacher_embedded_in_reader"]
    with pytest.raises(ValueError, match="do not duplicate or retry"):
        entry.main(["d0"])

    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d1"])
    assert entry.main(["d1"]) == 0
    receipt = json.loads((root / "receipts" / (entry.JOBS["d1"] + ".json")).read_text())
    assert receipt["status"] == "completed" and set(receipt["artifacts"]) == {"parity", "states", "clean_fit"}
    assert receipt["predecessors"][entry.JOBS["d0"]]["sha256"] == \
        P.file_pin(root / "receipts" / (entry.JOBS["d0"] + ".json"))["sha256"]
    result = json.loads((root / "d1/clean_fit.json").read_text())
    horizon = result["clean_fit"]["horizon"]
    expected_readings = {"1.0", "2.0", "3.0", "selected"} | ({"%.1f" % horizon} if result["clean_fit"]["extended_by_frozen_rule"] else set())
    assert result["reading"]["priority"] in (1, 2, 3) and set(result["fixed_readings"]) == expected_readings
    assert result["recovery_held"] == result["frozen_gate_on_selected_point"]["recovery"]
    saved = torch.load(root / "d1/clean_fit_states.pt", map_location="cpu", weights_only=False)
    assert saved["schema"] == "tcn-temporal-diag-clean-fit-states-v1" and len(saved["best_states"]) == 2

    # D2 is independent of D1 and needs only the completed D0; here the selected learning rate may already be at the cap.
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d2"])
    short = [c for c in world["runs"][core.HEAD]["selection"]["curves"] if c["horizon"] == RECIPE.initial_epochs]
    monkeypatch.setattr(entry, "LEARNING_RATE", short[0]["lr"])
    assert entry.main(["d2"]) == 0
    receipt = json.loads((root / "receipts" / (entry.JOBS["d2"] + ".json")).read_text())
    assert receipt["status"] == "completed" and set(receipt["artifacts"]) == {"states", "extension"}
    extension = json.loads((root / "d2/noisy_extension.json").read_text())
    assert extension["extension"]["start_parity"]["passed"] and extension["gates"]["restored_state_parity"]
    assert extension["extension"]["new_points"] == RECIPE.maximum_epochs - RECIPE.initial_epochs


def test_entry_point_stops_before_the_clean_fit_when_driver_parity_fails(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    _patch_entry(monkeypatch, world, root)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    assert entry.main(["d0"]) == 0
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d1"])
    original = core.first_epoch_parity

    def failing(*args, **kwargs):
        return {**original(*args, **kwargs), "passed": False, "parity_level": "failed"}

    monkeypatch.setattr(core, "first_epoch_parity", failing)
    monkeypatch.setattr(core, "clean_fit", lambda *a, **k: pytest.fail("clean fit must not run"))
    assert entry.main(["d1"]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS["d1"] + ".json")).read_text())
    assert receipt["status"] == "gate_failed" and set(receipt["artifacts"]) == {"parity"}
    assert not (root / "d1/clean_fit.json").exists()


def test_d0_reports_a_parity_failure_as_a_failed_gate_with_all_evidence(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    _patch_entry(monkeypatch, world, root)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    selection = world["runs"][core.HEAD]["selection"]
    curves = copy.deepcopy(selection["curves"])
    next(c for c in curves if c["lr"] != selection["selected"]["lr"])["points"][-1]["mse"] *= 1 + 1e-3
    changed = WF.relock(selection, curves=curves)
    context = _context(world)
    context["controls"][core.HEAD]["selection"] = changed
    context["evaluation_identity"] = core.check_bundle(world["bundle"], changed)
    monkeypatch.setattr(entry, "setup", lambda: context)
    monkeypatch.setattr(entry, "_states", lambda c, head: {**world["runs"][head]["states"],
                        "selection_sha256": c["controls"][head]["selection"]["selection_sha256"]})
    assert entry.main(["d0"]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS["d0"] + ".json")).read_text())
    assert receipt["status"] == "gate_failed" and receipt["gates"]["state_parity"] is False
    assert (root / "d0/replay.json").exists()
    for stage in ("d1", "d2"):
        monkeypatch.setenv("TCND_JOB_ID", entry.JOBS[stage])
        with pytest.raises(ValueError, match="D0 did not complete"):
            entry.main([stage])


def test_d0_completes_on_the_numerical_parity_tier_and_marks_it(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    _patch_entry(monkeypatch, world, root)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    selection = world["runs"][core.HEAD]["selection"]
    curves = copy.deepcopy(selection["curves"])
    next(c for c in curves if c["lr"] != selection["selected"]["lr"])["points"][-1]["mse"] *= 1 + 1e-9
    changed = WF.relock(selection, curves=curves)
    context = _context(world)
    context["controls"][core.HEAD]["selection"] = changed
    context["evaluation_identity"] = core.check_bundle(world["bundle"], changed)
    monkeypatch.setattr(entry, "setup", lambda: context)
    monkeypatch.setattr(entry, "_states", lambda c, head: {**world["runs"][head]["states"],
                        "selection_sha256": c["controls"][head]["selection"]["selection_sha256"]})
    assert entry.main(["d0"]) == 0
    replay = json.loads((root / "d0/replay.json").read_text())
    assert replay["parity_level"] == "numerical" and all(replay["gates"].values())
    assert replay["replays"][core.HEAD]["parity_level"] == "numerical"
    assert not replay["replays"][core.HEAD]["parity_passed"]


def test_numerical_parity_bound_on_the_recoveries_follows_the_score_tolerance(world, tmp_path, monkeypatch):
    selection = world["runs"][core.HEAD]["selection"]
    records = world["bundle"].reference_gate["temporal_records"]
    exact = entry.recovery_bounds("exact", selection, world["bundle"])
    numerical = entry.recovery_bounds("numerical", selection, world["bundle"])
    for name in core.PANELS:
        e, margin = selection["selected_outer_values"][name]["E_dyn"], records[name]["clean_ls"] - records[name]["oracle"]
        # A relative change 1e-4 of E_dyn moves the recovery by 1e-4 E / margin: it must stay inside the bound.
        assert exact[name] < 1e-11 and numerical[name] > core.NUMERICAL_TOLERANCE * e / margin * 2
    root = tmp_path.resolve() / "diag"
    _patch_entry(monkeypatch, world, root)
    curves = copy.deepcopy(selection["curves"])
    next(c for c in curves if c["lr"] != selection["selected"]["lr"])["points"][-1]["mse"] *= 1 + 1e-9
    changed = WF.relock(selection, curves=curves)
    context = _context(world)
    context["controls"][core.HEAD]["selection"] = changed
    context["evaluation_identity"] = core.check_bundle(world["bundle"], changed)
    monkeypatch.setattr(entry, "setup", lambda: context)
    monkeypatch.setattr(entry, "_states", lambda c, head: {**world["runs"][head]["states"],
                        "selection_sha256": c["controls"][head]["selection"]["selection_sha256"]})
    base = dict(entry.ARCHIVED_RECOVERY)
    e, margin = selection["selected_outer_values"]["pooled"]["E_dyn"], records["pooled"]["clean_ls"] - records["pooled"]["oracle"]
    for shift, code in ((2e-4 * e / margin, 0), (3e-4 * e / margin, 3)):
        for leftover in ("d0",):
            for path in (root / leftover, root / "receipts" / (entry.JOBS[leftover] + ".json")):
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
        monkeypatch.setattr(entry, "ARCHIVED_RECOVERY", {**base, "pooled": base["pooled"] + shift})
        monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
        assert entry.main(["d0"]) == code


def test_d0_gate_compares_the_replayed_recoveries_with_the_archived_ones(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    _patch_entry(monkeypatch, world, root)
    monkeypatch.setenv("TCND_JOB_ID", entry.JOBS["d0"])
    monkeypatch.setattr(entry, "ARCHIVED_RECOVERY", {**entry.ARCHIVED_RECOVERY,
                                                     "pooled": entry.ARCHIVED_RECOVERY["pooled"] * (1 + 1e-9)})
    assert entry.main(["d0"]) == 3
    receipt = json.loads((root / "receipts" / (entry.JOBS["d0"] + ".json")).read_text())
    assert receipt["gates"] == {"state_parity": True, "held_oracle_self_check": True, "gate_arithmetic": False}
