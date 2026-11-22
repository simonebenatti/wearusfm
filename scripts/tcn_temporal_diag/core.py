"""Scientific core of the temporal-control diagnostics; no files, no scheduler.

D0 replays archived trajectory states of the temporal control and scores them
against the observed target ``y`` and the clean target ``y*``, on the fitting
and on the inner-held users, with the frozen scorer. Parity with the archived
numbers is checked before any new number is trusted.

D1 repeats the frozen training of the TCN reader at one learning rate with a
single intervention: the *training* target is ``y*``. The checkpoint calendar,
the selection (minimum pooled MSE against the observed ``y`` of the held users),
the initialization, the shuffle and dropout streams, the recipe and the scorer
are the frozen ones. A parity gate first reproduces the archived first epoch
with the training target ``y``.

Every function takes the frozen ``FoldBundle``; nothing here builds targets,
masks, thresholds or partitions on its own.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict

import numpy as np

from pose_temponet import controls as C, data as D, heads, metrics as M, selection as S, train as T, workflow as W

WORLD = "temporal"
HEAD = "TCN_post"
DEPTH = "post"
PANELS = ("fold0", "fold1", "pooled")
TOLERANCE = 1e-12
REFERENCE_TOLERANCE = 1e-9
ACCUMULATORS = ("mse_sum", "dynamic_sse", "dynamic_target_ss")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(new, old, tolerance=TOLERANCE):
    """The scoring tolerance of the frozen selection: absolute plus relative."""
    return abs(float(new) - float(old)) <= tolerance + tolerance * abs(float(old))


def same_statistics(new, old, tolerance=TOLERANCE):
    return (int(new["window_count"]) == int(old["window_count"]) and
            all(close(new[k], old[k], tolerance) for k in ACCUMULATORS))


def statistics_gap(new, old):
    """Largest relative difference of the accumulators, for the parity record."""
    gaps = [abs(float(new[k]) - float(old[k])) / max(abs(float(old[k])), 1e-300) for k in ACCUMULATORS]
    return {"window_count_equal": int(new["window_count"]) == int(old["window_count"]),
            "max_relative_difference": max(gaps)}


def plain(value):
    """JSON-ready copy: tuples to lists, NumPy scalars to Python numbers."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def panels(per_fold):
    """fold0, fold1 and the pooled panel from summed accumulators, never from averaged ratios."""
    require(len(per_fold) == 2, "two folds required")
    return {"fold0": dict(per_fold[0]), "fold1": dict(per_fold[1]), "pooled": C.pool_statistics(list(per_fold))}


def panel_values(per_fold):
    return {name: C.statistic_values(stats) for name, stats in panels(per_fold).items()}


def default_factory(head=HEAD):
    return lambda seed, fold: heads.make_head(head, seed=seed)


def score_splits(model, bundle, fold, device, depth=DEPTH):
    """Frozen scorer, same 32-window blocks: observed and clean targets, held and fitting users."""
    result = {}
    for split, indices in (("held", bundle.held[fold]), ("fitting", bundle.fitting[fold])):
        observed, clean = W._score_model(model, bundle, fold, indices, depth, device, include_clean=True)
        result[split] = {"observed": observed, "clean": clean}
    return result


def oracle_statistics(bundle, fold, indices):
    """The known clean function scored against the observed target, as the closed-form reference did."""
    prediction = np.asarray(bundle.clean_targets[fold])[indices][:, D.CROP:-D.CROP]
    return M.selection_statistics(prediction, W._metric_panel(bundle, fold, indices), bundle.thresholds[fold])


def reference_statistics(bundle, artifacts):
    """Clean-LS and oracle accumulators against observed ``y``, for held and fitting users.

    Held: the authenticated closed-form references that the gate itself uses.
    Fitting: clean-LS from the authenticated linear diagnostic (fitted on ``y*``,
    scored on ``y`` over the fitting panel); oracle computed here from the arrays.
    The held oracle is recomputed as a self-check of the panel reconstruction.
    """
    ordered = sorted(artifacts, key=lambda a: a["outer_fold"])
    require([a["outer_fold"] for a in ordered] == [0, 1], "two calibrated outer folds required")
    references = {"held": {"clean_ls": [], "oracle": []}, "fitting": {"clean_ls": [], "oracle": []}}
    checks = []
    for fold, artifact in enumerate(ordered):
        outer = artifact["outer_reference_statistics"]
        references["held"]["clean_ls"].append(dict(outer[WORLD]["clean_ls"]))
        references["held"]["oracle"].append(dict(outer[WORLD]["oracle"]))
        diagnostic = outer["linear_diagnostics"]["L0"]
        require(diagnostic["temporal_margin_fit_target"] == "clean_y_star" and
                diagnostic["temporal_margin_score_target"] == "observed_y", "unexpected fitting reference definition")
        references["fitting"]["clean_ls"].append(dict(diagnostic["temporal_margin_clean_fit_statistics"]))
        references["fitting"]["oracle"].append(oracle_statistics(bundle, fold, bundle.fitting[fold]))
        recomputed = oracle_statistics(bundle, fold, bundle.held[fold])
        archived = references["held"]["oracle"][-1]
        checks.append({"fold": fold, "recomputed_held_oracle": recomputed, **statistics_gap(recomputed, archived),
                       "within_reference_tolerance": same_statistics(recomputed, archived, REFERENCE_TOLERANCE)})
        # The fitting clean-LS and the fitting oracle must describe the same panel.
        require(int(references["fitting"]["clean_ls"][-1]["window_count"]) ==
                int(references["fitting"]["oracle"][-1]["window_count"]), "fitting reference panels differ")
        require(close(references["fitting"]["clean_ls"][-1]["dynamic_target_ss"],
                      references["fitting"]["oracle"][-1]["dynamic_target_ss"], REFERENCE_TOLERANCE),
                "fitting reference dynamic denominators differ")
    return {"references": references, "held_oracle_self_check": checks,
            "held_oracle_self_check_passed": all(c["within_reference_tolerance"] for c in checks)}


def recovery_panels(reader_per_fold, reference):
    """Gate formula (E_cleanLS - E_reader)/(E_cleanLS - E_oracle) on E_dyn against observed ``y``."""
    out = {}
    readers, clean_ls, oracle = (panel_values(v) for v in (reader_per_fold, reference["clean_ls"], reference["oracle"]))
    for name in PANELS:
        e = {"reader": readers[name]["E_dyn"], "clean_ls": clean_ls[name]["E_dyn"], "oracle": oracle[name]["E_dyn"]}
        out[name] = {"E_dyn": e, "recovery": C._recovery(e["clean_ls"], e["reader"], e["oracle"])}
    return out


def noise_alignment(observed, clean, oracle):
    """Cross term of SSE_y = SSE_y* + SSE_oracle - 2<e,n> on one dynamic support.

    ``e = prediction - y*`` and ``n = y - y*``, both centred in time as the
    metric does. ``noise_fit_fraction = <e,n>/<n,n>`` is 0 when the prediction
    error ignores the realised noise and 1 when the prediction reproduces it.
    """
    noise = float(oracle["dynamic_sse"])
    cross = (float(clean["dynamic_sse"]) + noise - float(observed["dynamic_sse"])) / 2
    return {"cross_term": cross, "noise_sse": noise,
            "noise_fit_fraction": cross / noise if noise > 0 else None}


def error_shares(observed, clean, oracle):
    """Exact split of E_dyn against observed ``y`` over its own denominator.

    E_dyn = clean_error + noise_level + alignment, with alignment = -2<e,n>/TSS:
    the error against the clean function, the level of the realised noise and
    what the prediction gains by reproducing that noise.
    """
    total = float(observed["dynamic_target_ss"])
    cross = noise_alignment(observed, clean, oracle)["cross_term"]
    if total <= 0:
        return {"clean_error": None, "noise_level": None, "alignment": None, "E_dyn": None}
    return {"clean_error": float(clean["dynamic_sse"]) / total, "noise_level": float(oracle["dynamic_sse"]) / total,
            "alignment": -2 * cross / total, "E_dyn": float(observed["dynamic_sse"]) / total}


def alignment_panels(scores_per_fold, oracle_per_fold):
    """Per fold and pooled, for one split; pooling sums the three accumulators first."""
    observed, clean, oracle = (panels(v) for v in ([s["observed"] for s in scores_per_fold],
                                                    [s["clean"] for s in scores_per_fold], oracle_per_fold))
    return {name: {**noise_alignment(observed[name], clean[name], oracle[name]),
                   "error_shares": error_shares(observed[name], clean[name], oracle[name])} for name in PANELS}


def gap_decomposition(summary):
    """Held minus fitting, term by term, for the exact error shares of one model pair."""
    out = {}
    for name in PANELS:
        held, fitting = (summary[split]["noise_alignment"][name]["error_shares"] for split in ("held", "fitting"))
        out[name] = {key: (None if held[key] is None or fitting[key] is None else held[key] - fitting[key])
                     for key in ("E_dyn", "clean_error", "noise_level", "alignment")}
    return out


def summarize(per_fold_scores, references):
    """All readings of one model pair: values, gate-scale recoveries and noise alignment."""
    out = {}
    for split in ("held", "fitting"):
        scores = [f[split] for f in per_fold_scores]
        out[split] = {
            "observed": panel_values([s["observed"] for s in scores]),
            "clean": panel_values([s["clean"] for s in scores]),
            "recovery_on_observed": recovery_panels([s["observed"] for s in scores], references[split]),
            "noise_alignment": alignment_panels(scores, references[split]["oracle"]),
        }
    return out


def check_bundle(bundle, selection):
    """The rebuilt bundle must be the one that the archived selection used, lock for lock."""
    W._assert_bundle(bundle)
    require(bundle.world == WORLD and selection["world"] == WORLD, "temporal control bundle required")
    require(dict(bundle.manifest) == selection["bundle"], "rebuilt bundle differs from the archived selection bundle")
    feature_hash = W._array_hash(W._features(bundle.cache, DEPTH))
    require(feature_hash == selection["feature_array_sha256"], "feature array differs from the archived selection")
    identity = W._identity({"bundle": bundle.manifest, "depth": DEPTH, "feature_array_sha256": feature_hash})
    require(identity == selection["evaluation_identity"], "inner evaluation identity differs from the archive")
    return identity


def replay_head(bundle, head, selection, states, references, *, device, recipe=T.Recipe(),
                scope="all", factory=None, log=None):
    """D0: restore archived states, score them four ways and compare with the archive.

    ``scope="all"`` reads the best and the final state of every learning rate;
    ``scope="selected_best"`` reads only the globally selected pair.
    """
    require(scope in ("all", "selected_best"), "unknown replay scope")
    W._check_selection(selection, production=not recipe.test_only)
    require(selection["head"] == head and selection["world"] == bundle.world == WORLD, "wrong archived selection")
    require(states.get("schema") == "pose-head-selection-states-v1" and
            states.get("selection_sha256") == selection["selection_sha256"], "states do not belong to this selection")
    depth = W._feature_depth(head)
    require(depth == DEPTH, "post-backbone readers only")
    factory = factory or default_factory(head)
    fitting = [W._training_set(bundle, ix, depth, fold) for fold, ix in enumerate(bundle.fitting)]
    curves = {float(c["lr"]): c for c in selection["curves"]}
    require(set(curves) == {float(lr) for lr in states["candidate_states"]}, "archived curves and states differ")
    selected_lr = float(selection["selected"]["lr"])
    records, parity = [], []
    for lr in sorted(states["candidate_states"]):
        if scope == "selected_best" and float(lr) != selected_lr:
            continue
        pair, curve = states["candidate_states"][lr], curves[float(lr)]
        points = {float(p["exposure"]): p for p in curve["points"]}
        for kind in ("best_states", "current_states"):
            if scope == "selected_best" and kind != "best_states":
                continue
            exposure = float(pair["best_exposure"] if kind == "best_states" else curve["points"][-1]["exposure"])
            per_fold, updates, losses, identities = [], [], [], []
            for fold in (0, 1):
                state = pair[kind][fold]
                trajectory = T.Trajectory(factory, fitting[fold], lr=float(lr), seed=0, fold=fold,
                                          device=device, _recipe=recipe)
                # restore() authenticates payload, recipe, numerics, initialization and the training set itself.
                trajectory.restore(state)
                per_fold.append(score_splits(trajectory.model, bundle, fold, device, depth))
                updates.append(trajectory.updates)
                losses.append(list(trajectory.epoch_losses))
                identities.append({"payload_sha256": state["payload_sha256"],
                                   "model_sha256": T.payload_sha256(state["model"]),
                                   "dataset_sha256": state["dataset_sha256"], "updates": trajectory.updates})
                del trajectory
            point = S.pool_scores([f["held"]["observed"] for f in per_fold], exposure, updates)
            archived = points[exposure]
            check = {"head": head, "lr": float(lr), "kind": kind, "exposure": exposure, "panel": "pooled_curve_point",
                     "fold_steps_equal": list(point.fold_steps) == list(archived["fold_steps"]),
                     "mse": [point.mse, archived["mse"]], "edyn": [point.edyn, archived["edyn"]],
                     "passed": (list(point.fold_steps) == list(archived["fold_steps"]) and
                                close(point.mse, archived["mse"]) and close(point.edyn, archived["edyn"]))}
            parity.append(check)
            selected = kind == "best_states" and float(lr) == selected_lr
            if selected:
                require(exposure == float(selection["selected"]["exposure"]), "selected exposure differs from archive")
                for fold in (0, 1):
                    for label, new, old in (
                            ("held_observed", per_fold[fold]["held"]["observed"],
                             selection["selected_outer_statistics"]["fold" + str(fold)]),
                            ("held_clean", per_fold[fold]["held"]["clean"],
                             selection["selected_outer_clean_statistics"]["fold" + str(fold)]),
                            ("fitting_observed", per_fold[fold]["fitting"]["observed"],
                             selection["selected_fitting_diagnostics"][str(fold)])):
                        parity.append({"head": head, "lr": float(lr), "kind": kind, "exposure": exposure,
                                       "panel": "fold%d_%s" % (fold, label), **statistics_gap(new, old),
                                       "passed": same_statistics(new, old)})
            record = {"head": head, "lr": float(lr), "kind": kind, "exposure": exposure, "selected": selected,
                      "fold_steps": updates, "states": identities, "per_fold": per_fold,
                      "summary": summarize(per_fold, references)}
            if kind == "current_states":
                record["train_mode_epoch_losses"] = losses
            records.append(record)
            if log is not None:
                log({"event": "replayed_state", "head": head, "lr": float(lr), "kind": kind, "exposure": exposure,
                     "parity": check["passed"]})
    require(records, "no archived state was replayed")
    return {"head": head, "scope": scope, "selected": dict(selection["selected"]), "records": records,
            "parity": parity, "parity_passed": all(p["passed"] for p in parity)}


def first_epoch_parity(bundle, selection, states, *, device, recipe=T.Recipe(), lr, factory=None, log=None):
    """D1 gate: this driver, trained on the observed ``y``, must retrace the archived first epoch."""
    factory = factory or default_factory(HEAD)
    fitting = [W._training_set(bundle, ix, DEPTH, fold) for fold, ix in enumerate(bundle.fitting)]
    evaluator = lambda model, fold: W._score_model(model, bundle, fold, bundle.held[fold], DEPTH, device)
    trajectories = [T.Trajectory(factory, data, lr=lr, seed=0, fold=fold, device=device, _recipe=recipe)
                    for fold, data in enumerate(fitting)]
    pair = states["candidate_states"][lr]
    curve = next(c for c in selection["curves"] if float(c["lr"]) == float(lr))
    archived = [p for p in curve["points"] if p["exposure"] <= 1]
    schedule = S.checkpoint_schedule([t.steps_per_epoch for t in trajectories], 1)
    checks = [{"check": "initial_weights", "passed": [t.initial_sha256 for t in trajectories] ==
               list(selection["initial_sha256_per_fold"])},
              {"check": "training_set_identity", "passed": [d.identity for d in fitting] ==
               [pair["current_states"][fold]["dataset_sha256"] for fold in (0, 1)]},
              {"check": "schedule_length", "passed": len(schedule) == len(archived)}]
    rows = []
    for (exposure, fold_steps), old in zip(schedule, archived):
        for trajectory, step in zip(trajectories, fold_steps):
            trajectory.train_to_step(step)
        point = S.pool_scores([t.evaluate(evaluator) for t in trajectories], exposure, fold_steps)
        row = {"exposure": exposure, "fold_steps": list(fold_steps), "archived_fold_steps": list(old["fold_steps"]),
               "mse": [point.mse, old["mse"]], "edyn": [point.edyn, old["edyn"]],
               "passed": (exposure == old["exposure"] and list(fold_steps) == list(old["fold_steps"]) and
                          close(point.mse, old["mse"]) and close(point.edyn, old["edyn"]))}
        rows.append(row)
        if log is not None:
            log({"event": "parity_point", **row})
    for fold, trajectory in enumerate(trajectories):
        final = pair["current_states"][fold]
        require(trajectory.updates == trajectory.steps_per_epoch and len(trajectory.epoch_losses) == 1,
                "parity run did not stop after exactly one epoch")
        checks.append({"check": "fold%d_first_shuffle" % fold,
                       "passed": trajectory.epoch_order_sha256[0] == final["epoch_order_sha256"][0]})
        new, old = trajectory.epoch_losses[0], final["epoch_losses"][0]
        checks.append({"check": "fold%d_first_epoch_train_loss" % fold, "loss_sum": [new["loss_sum"], old["loss_sum"]],
                       "passed": new["window_count"] == old["window_count"] and close(new["loss_sum"], old["loss_sum"])})
    passed = all(c["passed"] for c in checks) and all(r["passed"] for r in rows) and len(rows) == len(archived)
    return {"lr": lr, "training_target": "observed_y", "checks": checks, "points": rows, "passed": passed}


def clean_fit(bundle, selection, references, *, device, recipe=T.Recipe(), lr, snapshots=(), factory=None, log=None):
    """D1: the frozen paired training at one learning rate, fitted on the clean target ``y*``.

    Selection is untouched: minimum pooled MSE against the observed ``y`` of the
    held users over the frozen calendar, first exposure on ties. The clean
    curves are descriptive. Fitting users are scored in evaluation mode at
    every checkpoint; no extra evaluation draws random numbers.
    """
    factory = factory or default_factory(HEAD)
    identity = check_bundle(bundle, selection)
    clean_bundle = dataclasses.replace(bundle, targets=bundle.clean_targets)
    fitting = [W._training_set(clean_bundle, ix, DEPTH, fold) for fold, ix in enumerate(bundle.fitting)]
    held = {}

    def evaluator(model, fold):
        observed, clean = W._score_model(model, bundle, fold, bundle.held[fold], DEPTH, device, include_clean=True)
        held[fold] = {"observed": observed, "clean": clean}
        return observed

    trajectories = [T.Trajectory(factory, data, lr=lr, seed=0, fold=fold, device=device, _recipe=recipe)
                    for fold, data in enumerate(fitting)]
    require([t.initial_sha256 for t in trajectories] == list(selection["initial_sha256_per_fold"]),
            "clean fit does not start from the archived initialization")
    pair = T.PairedCandidate(trajectories, evaluator, identity)
    records, saved = [], {}
    wanted = {float(e) for e in snapshots}

    def checkpoint(candidate):
        point = candidate.points[-1]
        per_fold = []
        for fold, trajectory in enumerate(candidate.trajectories):
            observed, clean = W._score_model(trajectory.model, bundle, fold, bundle.fitting[fold], DEPTH, device,
                                             include_clean=True)
            per_fold.append({"held": {k: dict(v) for k, v in held[fold].items()},
                             "fitting": {"observed": observed, "clean": clean}})
        records.append({"point": asdict(point), "per_fold": per_fold, "summary": summarize(per_fold, references),
                        "train_mode_loss": [t.epoch_losses[-1] if t.epoch_losses and
                                            t.updates % t.steps_per_epoch == 0 else None
                                            for t in candidate.trajectories]})
        if float(point.exposure) in wanted:
            saved[float(point.exposure)] = [t.state() for t in candidate.trajectories]
        if log is not None:
            log({"event": "clean_fit_checkpoint", "lr": lr, "point": asdict(point),
                 "held_recovery_pooled": records[-1]["summary"]["held"]["recovery_on_observed"]["pooled"]["recovery"]})

    pair.advance(recipe.initial_epochs, checkpoint)
    curve = pair.curve()
    best = curve.best("mse")
    minima = {metric: curve.best(metric).exposure for metric in ("mse", "edyn")}
    late = S.needs_epoch_extension(curve, recipe.initial_epochs, recipe.last_epochs)
    selected = []
    for fold in (0, 1):
        trajectory = T.Trajectory(factory, fitting[fold], lr=lr, seed=0, fold=fold, device=device, _recipe=recipe)
        trajectory.restore(pair.best_states[fold])
        require(trajectory.updates == best.fold_steps[fold], "selected pair exposure mismatch")
        selected.append(score_splits(trajectory.model, bundle, fold, device))
        del trajectory
    point = S.pool_scores([s["held"]["observed"] for s in selected], best.exposure, best.fold_steps)
    # Same check as the frozen selection: the restored selected pair reproduces its curve point.
    reproduced = close(point.mse, best.mse) and close(point.edyn, best.edyn)
    selected_summary = summarize(selected, references)
    return {"lr": lr, "training_target": "clean_y_star", "selection_target": "observed_y",
            "evaluation_identity": identity, "training_set_sha256": [d.identity for d in fitting],
            "initial_sha256_per_fold": [t.initial_sha256 for t in trajectories],
            "horizon": recipe.initial_epochs, "recipe": asdict(recipe),
            "curve": asdict(curve), "minimum_exposures": minima, "late": bool(late),
            "selected": asdict(best), "selected_reproduced": bool(reproduced),
            "selected_per_fold": selected, "selected_summary": selected_summary,
            "records": records, "snapshot_exposures": sorted(saved),
            "states": {"schema": "tcn-temporal-diag-clean-fit-states-v1", "lr": lr,
                       "best_exposure": pair.best_exposure, "best_states": pair.best_states,
                       "snapshots": saved, "final_states": [t.state() for t in trajectories]}}


def frozen_gate(selected_summary, bundle):
    """The frozen temporal head gate on the D1 selected point, with the gate's own references."""
    head = {name: selected_summary["held"]["observed"][name]["E_dyn"] for name in PANELS}
    return C.temporal_head_gate(bundle.reference_gate["temporal_records"], head)


def reading_rule(held_recovery, fitting_recovery, late, valid):
    """Ordered and non-overlapping outcome, registered before the fits.

    Recoveries are the gate formula against observed ``y`` at the selected
    point, for fold0, fold1 and pooled. The threshold is the frozen one.
    """
    threshold = C.TEMPORAL_HEAD_RECOVERY
    values = [held_recovery.get(p) for p in PANELS] + [fitting_recovery.get(p) for p in PANELS]
    if not valid or any(v is None or not np.isfinite(v) for v in values):
        return {"priority": 0, "outcome": "not_evaluable"}
    held = [held_recovery[p] >= threshold for p in PANELS]
    fitting = [fitting_recovery[p] >= threshold for p in PANELS]
    if all(held):
        return {"priority": 1, "outcome": "sensitivity_recovered_when_trained_on_clean_target", "late": bool(late)}
    if late:
        return {"priority": 2, "outcome": "unresolved_within_horizon", "late": True}
    if all(fitting):
        return {"priority": 3, "outcome": "3a_fitting_reaches_threshold_but_not_every_held_panel", "late": False}
    if not any(fitting):
        return {"priority": 3, "outcome": "3b_recovery_insufficient_also_on_fitting", "late": False}
    return {"priority": 3, "outcome": "3c_heterogeneous_fitting_panels", "late": False}


def fixed_readings(result, exposures):
    """Readings at the exposures declared before the fits, plus the selected point."""
    by_exposure = {float(r["point"]["exposure"]): r for r in result["records"]}
    rows = {}
    for exposure in sorted({float(e) for e in exposures}):
        if exposure in by_exposure:
            rows[str(exposure)] = {"point": by_exposure[exposure]["point"], "summary": by_exposure[exposure]["summary"]}
    rows["selected"] = {"point": result["selected"], "summary": result["selected_summary"]}
    return rows
