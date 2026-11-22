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

from contextlib import contextmanager
import dataclasses
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from pose_temponet import controls as C, data as D, heads, metrics as M, selection as S, train as T, workflow as W

WORLD = "temporal"
HEAD = "TCN_post"
DEPTH = "post"
PANELS = ("fold0", "fold1", "pooled")
TOLERANCE = 1e-12
# Second parity tier, registered before the jobs: FP32 rounding between two nodes of the same model.
# A different state, target, support, shuffle or dropout stream moves the scores by far more than this.
NUMERICAL_TOLERANCE = 1e-4
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


def relative_difference(new, old):
    return abs(float(new) - float(old)) / max(abs(float(old)), 1e-300)


def statistics_gap(new, old):
    """Largest relative difference of the accumulators, for the parity record."""
    return {"window_count_equal": int(new["window_count"]) == int(old["window_count"]),
            "max_relative_difference": max(relative_difference(new[k], old[k]) for k in ACCUMULATORS)}


def tiers(exact, discrete, differences):
    """One parity row: exact tier, numerical tier and the largest relative difference seen."""
    worst = max(differences) if differences else 0.0
    return {"passed": bool(exact and discrete), "numerical": bool(discrete and worst <= NUMERICAL_TOLERANCE),
            "max_relative_difference": worst}


def parity_level(rows):
    """'exact' when every row passes the frozen tolerance, 'numerical' when every row passes the second tier."""
    rows = list(rows)
    if rows and all(r["passed"] for r in rows):
        return "exact"
    if rows and all(r["passed"] or r.get("numerical", False) for r in rows):
        return "numerical"
    return "failed"


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


class BranchProbe:
    """Read-only forward hooks on one reader: nothing is modified, no extra forward is run.

    While attached, every forward pass adds to (i) how often each unit of each
    ReLU of the deep branch is positive and (ii) the energy of the affine and of
    the deep branch on the central crop. A unit is *dead* on the scored windows
    when it is never positive there. L0 has no deep branch: the summary is None.
    """

    def __init__(self, model):
        self.model, self.handles, self.layers = model, [], []
        self.energy = {key: 0.0 for key in ("affine", "deep", "cross", "affine_centred", "deep_centred",
                                             "cross_centred")}
        self.energy["count"] = 0
        self._deep = None
        deep = getattr(model, "deep", None)
        if deep is None:
            return
        channel_last = getattr(model, "name", None) == "MLP"
        for index, module in enumerate(deep):
            if isinstance(module, nn.ReLU):
                row = {"module": index, "positive": None, "count": 0}
                self.layers.append(row)
                self.handles.append(module.register_forward_hook(self._activity(row, channel_last)))
        self.handles.append(deep.register_forward_hook(self._remember(channel_last)))
        self.handles.append(model.affine.register_forward_hook(self._energy))

    @staticmethod
    def _activity(row, channel_last):
        def hook(module, inputs, output):
            value = output.detach()
            positive = (value > 0).sum(dim=(0, 1) if channel_last else (0, 2)).to(dtype=torch.float64, device="cpu")
            row["positive"] = positive if row["positive"] is None else row["positive"] + positive
            row["count"] += value.shape[0] * (value.shape[1] if channel_last else value.shape[2])
        return hook

    def _remember(self, channel_last):
        def hook(module, inputs, output):
            value = output.detach()
            self._deep = value if channel_last else value.transpose(1, 2)
        return hook

    def _energy(self, module, inputs, output):
        # PoseReadout.forward computes the deep branch first, then affine(h) + deep.
        affine, deep = output.detach()[:, D.CROP:-D.CROP].double(), self._deep[:, D.CROP:-D.CROP].double()
        for suffix, a, d in (("", affine, deep), ("_centred", affine - affine.mean(dim=1, keepdim=True),
                                                  deep - deep.mean(dim=1, keepdim=True))):
            self.energy["affine" + suffix] += float((a * a).sum().cpu())
            self.energy["deep" + suffix] += float((d * d).sum().cpu())
            self.energy["cross" + suffix] += float((a * d).sum().cpu())
        self.energy["count"] += affine.numel()
        self._deep = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summary(self):
        if not self.layers:
            return None
        layers = []
        for row in self.layers:
            if row["positive"] is None or row["count"] == 0:
                return None
            fraction = row["positive"] / row["count"]
            layers.append({"module": row["module"], "units": int(fraction.numel()),
                           "active_fraction": float(fraction.mean()), "dead_units": int((row["positive"] == 0).sum()),
                           "least_active_unit_fraction": float(fraction.min())})
        count = self.energy["count"]
        affine, deep, cross = (self.energy[k] / count for k in ("affine", "deep", "cross"))
        affine_c, deep_c, cross_c = (self.energy[k + "_centred"] / count for k in ("affine", "deep", "cross"))
        return {"relu_layers": layers, "elements": count,
                "affine_mean_square": affine, "deep_mean_square": deep, "affine_deep_mean_product": cross,
                # Centred in time inside each window and joint: what the dynamic error can see of each branch.
                "affine_centred_mean_square": affine_c, "deep_centred_mean_square": deep_c,
                "affine_deep_centred_mean_product": cross_c,
                "deep_share_of_centred_energy": deep_c / (affine_c + deep_c) if affine_c + deep_c > 0 else None}


@contextmanager
def branch_probe(model):
    probe = BranchProbe(model)
    try:
        yield probe
    finally:
        probe.close()


def layer_norm_drift(model):
    """Learnable LayerNorm of the reader against its identity initialization."""
    weight, bias = model.norm.weight.detach().double().cpu(), model.norm.bias.detach().double().cpu()
    return {"weight_mean": float(weight.mean()), "weight_min": float(weight.min()), "weight_max": float(weight.max()),
            "weight_rms_deviation_from_one": float(((weight - 1) ** 2).mean().sqrt()),
            "bias_rms": float((bias ** 2).mean().sqrt())}


def score_probed(model, bundle, fold, indices, device, depth=DEPTH):
    """One frozen scoring pass against observed and clean targets, with the read-only branch probe attached."""
    with branch_probe(model) as probe:
        observed, clean = W._score_model(model, bundle, fold, indices, depth, device, include_clean=True)
    return {"observed": observed, "clean": clean, "branches": probe.summary()}


def score_splits(model, bundle, fold, device, depth=DEPTH):
    """Frozen scorer, same 32-window blocks: observed and clean targets, held and fitting users."""
    return {"held": score_probed(model, bundle, fold, bundle.held[fold], device, depth),
            "fitting": score_probed(model, bundle, fold, bundle.fitting[fold], device, depth),
            "layer_norm": layer_norm_drift(model)}


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


def teacher_reader(artifact):
    """The known clean function written into the frozen reader. Evaluation only: never trained, never a candidate.

    L goes to the affine branch and the width-8 generator T to the first eight
    channels of the deep branch; means, scales and the coefficient ``a`` are
    absorbed by the two output layers. Scoring it with the frozen scorer shows
    what the scoring path gives for the true function on the real inputs.
    """
    components, a = artifact["components"], float(artifact["calibration"]["selected"][WORLD]["a"])
    model = heads.make_head(HEAD, 0).eval()
    layers = [module for module in model.deep if isinstance(module, nn.Conv1d)]
    teacher, linear = components["generator_states"]["T"], components["generator_states"]["L"]
    keys = [key for key in teacher if key.endswith(".weight")]
    require(len(keys) == len(layers), "generator and reader depths differ")
    scale = {name: {k: torch.tensor(components["normalization"][name][k], dtype=torch.float32)
                    for k in ("mean", "std")} for name in ("L", "T")}
    with torch.no_grad():
        for layer, key in zip(layers, keys):
            weight, bias = teacher[key].float(), teacher[key[:-len("weight")] + "bias"].float()
            require(weight.shape[2:] == layer.weight.shape[2:] and weight.shape[0] <= layer.weight.shape[0] and
                    weight.shape[1] <= layer.weight.shape[1], "generator layer does not fit the reader")
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight[:weight.shape[0], :weight.shape[1]].copy_(weight)
            layer.bias[:len(bias)].copy_(bias)
        model.affine.weight.copy_(linear["weight"].float() / scale["L"]["std"][:, None])
        model.affine.bias.copy_((linear["bias"].float() - scale["L"]["mean"]) / scale["L"]["std"])
        layers[-1].weight.mul_((a / scale["T"]["std"])[:, None, None])
        layers[-1].bias.copy_((layers[-1].bias - scale["T"]["mean"]) * a / scale["T"]["std"])
    return model


def teacher_scores(bundle, artifacts, references, *, device):
    """Frozen scorer applied to the embedded clean function, per fold and pooled: recovery should be one."""
    per_fold = []
    for fold, artifact in enumerate(sorted(artifacts, key=lambda a: a["outer_fold"])):
        model = teacher_reader(artifact).to(device=device, dtype=torch.float32)
        per_fold.append(score_splits(model, bundle, fold, device))
        del model
    return {"per_fold": per_fold, "summary": summarize(per_fold, references)}


def generator_summary(artifacts):
    """Small descriptive fields of the frozen synthetic generator, kept only inside the calibration artifacts."""
    rows = []
    for artifact in sorted(artifacts, key=lambda a: a["outer_fold"]):
        components = artifact.get("components") or {}
        rows.append({"outer_fold": artifact["outer_fold"], "seed": components.get("seed"),
                     "generator_hashes": components.get("generator_hashes"),
                     "hidden_active_fraction_fitting": components.get("hidden_active_fraction_fitting"),
                     "normalization": components.get("normalization"),
                     "selected": artifact["calibration"]["selected"].get(WORLD)})
    return plain(rows)


def clean_scale_references(bundle, artifacts, *, block_windows=32):
    """Clean-LS (fitted on ``y*`` of the fitting users) scored against ``y*`` and ``y``, held and fitting.

    The score against ``y`` repeats the authenticated references and is only a
    self-check; the score against ``y*`` is the noise-free linear reference that
    the archive does not contain. Descriptive: no gate reads it.
    """
    features = np.asarray(W._features(bundle.cache, DEPTH))
    ordered = sorted(artifacts, key=lambda a: a["outer_fold"])
    rows, checks = [], []
    for fold, artifact in enumerate(ordered):
        components, a = artifact["components"], artifact["calibration"]["selected"][WORLD]["a"]
        factor = C.WindowedBasisFit(features, {key: components[key] for key in ("L", "T")}, bundle.support,
                                    bundle.fitting[fold], block_windows=block_windows)
        row = {}
        for split, indices in (("held", bundle.held[fold]), ("fitting", bundle.fitting[fold])):
            prediction, _ = factor.predict_mixture(indices, {"L": 1., "T": a}, 0.)
            row[split] = {target: M.selection_statistics(prediction, W._metric_panel(bundle, fold, indices, clean=clean),
                                                         bundle.thresholds[fold])
                          for target, clean in (("observed", False), ("clean", True))}
        outer = artifact["outer_reference_statistics"]
        for split, archived in (("held", outer[WORLD]["clean_ls"]),
                                ("fitting", outer["linear_diagnostics"]["L0"]["temporal_margin_clean_fit_statistics"])):
            checks.append({"fold": fold, "split": split, **statistics_gap(row[split]["observed"], archived),
                           "within_reference_tolerance": same_statistics(row[split]["observed"], archived,
                                                                          REFERENCE_TOLERANCE)})
        rows.append(row)
        del factor
    return {"per_fold": rows, "self_check": checks,
            "self_check_passed": all(c["within_reference_tolerance"] for c in checks)}


def clean_scale_panels(scores_per_fold, clean_scale, split):
    """Share of the clean-LS error against ``y*`` that the reader removes; None without the reference."""
    if not clean_scale or not clean_scale.get("self_check_passed"):
        return None
    reader = panels([s["clean"] for s in scores_per_fold])
    linear = panels([row[split]["clean"] for row in clean_scale["per_fold"]])
    out = {}
    for name in PANELS:
        r, l = float(reader[name]["dynamic_sse"]), float(linear[name]["dynamic_sse"])
        out[name] = {"reader_clean_sse": r, "clean_ls_clean_sse": l,
                     "clean_ls_clean_E_dyn": C.statistic_values(linear[name])["E_dyn"],
                     "share_of_clean_ls_error_removed": 1 - r / l if l > 0 else None}
    return out


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
            "clean_scale": clean_scale_panels(scores, references.get("clean_scale"), split),
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
            same_steps = list(point.fold_steps) == list(archived["fold_steps"])
            check = {"head": head, "lr": float(lr), "kind": kind, "exposure": exposure, "panel": "pooled_curve_point",
                     "fold_steps_equal": same_steps,
                     "mse": [point.mse, archived["mse"]], "edyn": [point.edyn, archived["edyn"]],
                     **tiers(close(point.mse, archived["mse"]) and close(point.edyn, archived["edyn"]), same_steps,
                             [relative_difference(point.mse, archived["mse"]),
                              relative_difference(point.edyn, archived["edyn"])])}
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
                        gap = statistics_gap(new, old)
                        parity.append({"head": head, "lr": float(lr), "kind": kind, "exposure": exposure,
                                       "panel": "fold%d_%s" % (fold, label), "window_count_equal": gap["window_count_equal"],
                                       **tiers(same_statistics(new, old), gap["window_count_equal"],
                                               [gap["max_relative_difference"]])})
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
            "parity": parity, "parity_passed": all(p["passed"] for p in parity), "parity_level": parity_level(parity)}


def held_evaluator(bundle, device, held):
    """The frozen held-user scorer on observed ``y``; clean scores and branch census are kept aside in ``held``."""
    def evaluator(model, fold):
        held[fold] = score_probed(model, bundle, fold, bundle.held[fold], device)
        return held[fold]["observed"]
    return evaluator


def checkpoint_record(point, trajectories, held, bundle, references, device):
    """Everything read at one checkpoint. Evaluation mode only: no random number is drawn."""
    per_fold = []
    for fold, trajectory in enumerate(trajectories):
        per_fold.append({"held": dict(held[fold]),
                         "fitting": score_probed(trajectory.model, bundle, fold, bundle.fitting[fold], device),
                         "layer_norm": layer_norm_drift(trajectory.model)})
    return {"point": asdict(point), "per_fold": per_fold, "summary": summarize(per_fold, references),
            "train_mode_loss": [t.epoch_losses[-1] if t.epoch_losses and t.updates % t.steps_per_epoch == 0 else None
                                for t in trajectories]}


def first_epoch_parity(bundle, selection, states, references, *, device, recipe=T.Recipe(), lr, factory=None,
                       log=None):
    """D1 gate: this driver, trained on the observed ``y``, must retrace the archived first epoch.

    The same extra evaluations of the clean fit run at every checkpoint, so a
    perturbation of the trajectory by those evaluations would break the parity.
    """
    factory = factory or default_factory(HEAD)
    fitting = [W._training_set(bundle, ix, DEPTH, fold) for fold, ix in enumerate(bundle.fitting)]
    held = {}
    evaluator = held_evaluator(bundle, device, held)
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
        checkpoint_record(point, trajectories, held, bundle, references, device)
        row = {"exposure": exposure, "fold_steps": list(fold_steps), "archived_fold_steps": list(old["fold_steps"]),
               "mse": [point.mse, old["mse"]], "edyn": [point.edyn, old["edyn"]],
               **tiers(close(point.mse, old["mse"]) and close(point.edyn, old["edyn"]),
                       exposure == old["exposure"] and list(fold_steps) == list(old["fold_steps"]),
                       [relative_difference(point.mse, old["mse"]), relative_difference(point.edyn, old["edyn"])])}
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
                       **tiers(close(new["loss_sum"], old["loss_sum"]), new["window_count"] == old["window_count"],
                               [relative_difference(new["loss_sum"], old["loss_sum"])])})
    complete = len(rows) == len(archived)
    level = parity_level(checks + rows) if complete else "failed"
    return {"lr": lr, "training_target": "observed_y", "checks": checks, "points": rows,
            "passed": level == "exact", "parity_level": level}


def clean_fit(bundle, selection, references, *, device, recipe=T.Recipe(), lr, snapshots=(), factory=None, log=None):
    """D1: the frozen paired training at one learning rate, fitted on the clean target ``y*``.

    Selection is untouched: minimum pooled MSE against the observed ``y`` of the
    held users over the frozen calendar, first exposure on ties. The horizon
    follows the frozen rule of the search: 150 epochs, continued to 300 only
    when the pooled minimum of MSE or E_dyn falls in the last five epochs.
    The clean curves are descriptive. Fitting users are scored in evaluation
    mode at every checkpoint; no extra evaluation draws random numbers.
    """
    factory = factory or default_factory(HEAD)
    identity = check_bundle(bundle, selection)
    clean_bundle = dataclasses.replace(bundle, targets=bundle.clean_targets)
    fitting = [W._training_set(clean_bundle, ix, DEPTH, fold) for fold, ix in enumerate(bundle.fitting)]
    held = {}
    trajectories = [T.Trajectory(factory, data, lr=lr, seed=0, fold=fold, device=device, _recipe=recipe)
                    for fold, data in enumerate(fitting)]
    require([t.initial_sha256 for t in trajectories] == list(selection["initial_sha256_per_fold"]),
            "clean fit does not start from the archived initialization")
    pair = T.PairedCandidate(trajectories, held_evaluator(bundle, device, held), identity)
    records, saved = [], {}
    wanted = {float(e) for e in snapshots}

    def checkpoint(candidate):
        point = candidate.points[-1]
        records.append(checkpoint_record(point, candidate.trajectories, held, bundle, references, device))
        if float(point.exposure) in wanted:
            saved[float(point.exposure)] = [t.state() for t in candidate.trajectories]
        if log is not None:
            log({"event": "clean_fit_checkpoint", "lr": lr, "point": asdict(point),
                 "held_recovery_pooled": records[-1]["summary"]["held"]["recovery_on_observed"]["pooled"]["recovery"]})

    pair.advance(recipe.initial_epochs, checkpoint)
    extended = S.needs_epoch_extension(pair.curve(), recipe.initial_epochs, recipe.last_epochs)
    if extended:  # exactly the frozen search: train._run_search
        pair.advance(recipe.maximum_epochs, checkpoint)
    curve = pair.curve()
    best = curve.best("mse")
    minima = {metric: curve.best(metric).exposure for metric in ("mse", "edyn")}
    # After the frozen extension the only edge left is the one of the maximum horizon (selection.assessment).
    boundary = max(1, recipe.maximum_epochs - recipe.last_epochs + 1)
    late = curve.horizon == recipe.maximum_epochs and any(e >= boundary for e in minima.values())
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
            "horizon": curve.horizon, "extended_by_frozen_rule": bool(extended), "recipe": asdict(recipe),
            "curve": asdict(curve), "minimum_exposures": minima, "late": bool(late),
            "selected": asdict(best), "selected_reproduced": bool(reproduced),
            "selected_per_fold": selected, "selected_summary": selected_summary,
            "records": records, "snapshot_exposures": sorted(saved),
            "states": {"schema": "tcn-temporal-diag-clean-fit-states-v1", "lr": lr,
                       "best_exposure": pair.best_exposure, "best_states": pair.best_states,
                       "snapshots": saved, "final_states": [t.state() for t in trajectories]}}


def extend_noisy_fit(bundle, selection, states, references, *, device, recipe=T.Recipe(), lr, snapshots=(),
                     factory=None, log=None):
    """D2: the archived noisy trajectories at one learning rate, continued up to the frozen cap of epochs.

    Descriptive only: nothing is selected and no gate is evaluated. Both
    trajectories are restored from the archived final states, optimizer,
    shuffle and dropout streams included, and advanced through the frozen
    calendar on the same observed target. The restored pair must first
    reproduce the archived last point of its curve.
    """
    factory = factory or default_factory(HEAD)
    check_bundle(bundle, selection)
    fitting = [W._training_set(bundle, ix, DEPTH, fold) for fold, ix in enumerate(bundle.fitting)]
    pair = states["candidate_states"][lr]
    curve = next(c for c in selection["curves"] if float(c["lr"]) == float(lr))
    archived = curve["points"][-1]
    start, horizon = float(archived["exposure"]), recipe.maximum_epochs
    require(start == curve["horizon"] and start < horizon, "archived curve already reaches the frozen cap")
    trajectories = []
    for fold in (0, 1):
        trajectory = T.Trajectory(factory, fitting[fold], lr=lr, seed=0, fold=fold, device=device, _recipe=recipe)
        # A private copy: on CPU the optimizer keeps the restored tensors and further training would
        # rewrite the caller's archived state in place.
        trajectory.restore(T._cpu_copy(pair["current_states"][fold]))
        trajectories.append(trajectory)
    held, records, saved = {}, [], {}
    evaluator = held_evaluator(bundle, device, held)
    wanted = {float(e) for e in snapshots}
    steps = [t.steps_per_epoch for t in trajectories]
    schedule = [(start, tuple(t.updates for t in trajectories))]
    schedule += S.checkpoint_schedule(steps, horizon, after_exposure=start)
    parity = None
    for exposure, fold_steps in schedule:
        for trajectory, step in zip(trajectories, fold_steps):
            trajectory.train_to_step(step)
        point = S.pool_scores([t.evaluate(evaluator) for t in trajectories], exposure, fold_steps)
        if parity is None:
            same_steps = list(fold_steps) == list(archived["fold_steps"])
            parity = {"exposure": exposure, "fold_steps_equal": same_steps,
                      "mse": [point.mse, archived["mse"]], "edyn": [point.edyn, archived["edyn"]],
                      **tiers(close(point.mse, archived["mse"]) and close(point.edyn, archived["edyn"]), same_steps,
                              [relative_difference(point.mse, archived["mse"]),
                               relative_difference(point.edyn, archived["edyn"])])}
            parity["parity_level"] = parity_level([parity])
            if parity["parity_level"] == "failed":
                return {"lr": lr, "training_target": "observed_y", "start_parity": parity, "records": [],
                        "states": None}
        records.append(checkpoint_record(point, trajectories, held, bundle, references, device))
        if float(exposure) in wanted:
            saved[float(exposure)] = [t.state() for t in trajectories]
        if log is not None:
            log({"event": "noisy_extension_checkpoint", "lr": lr, "point": asdict(point),
                 "held_recovery_pooled": records[-1]["summary"]["held"]["recovery_on_observed"]["pooled"]["recovery"]})
    recovery = [(r["point"]["exposure"], r["summary"]["held"]["recovery_on_observed"]) for r in records[1:]]
    selected = float(selection["selected"]["edyn"])
    best = min(records[1:], key=lambda r: (r["point"]["mse"], r["point"]["exposure"]))
    return {"lr": lr, "training_target": "observed_y", "start_exposure": start, "horizon": horizon,
            "start_parity": parity, "records": records, "new_points": len(records) - 1,
            "minimum_mse_new_point": best["point"],
            "any_new_pooled_edyn_below_archived_selected": any(r["point"]["edyn"] < selected for r in records[1:]),
            "minimum_new_pooled_edyn": min(r["point"]["edyn"] for r in records[1:]),
            # Prediction 2 of the protocol: the fitting noise-fit fraction, pooled, at the start and at the end.
            "fitting_noise_fit_fraction_start_end": [
                r["summary"]["fitting"]["noise_alignment"]["pooled"]["noise_fit_fraction"] for r in (records[0], records[-1])],
            "any_new_point_reaches_threshold_in_every_panel": any(
                all(v[name]["recovery"] is not None and v[name]["recovery"] >= C.TEMPORAL_HEAD_RECOVERY
                    for name in PANELS) for _, v in recovery),
            "snapshot_exposures": sorted(saved),
            "states": {"schema": "tcn-temporal-diag-noisy-extension-states-v1", "lr": lr, "snapshots": saved,
                       "final_states": [t.state() for t in trajectories]}}


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
