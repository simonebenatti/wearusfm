"""Entry point: python -m tcn_temporal_diag {d0,d1}. Never submits, never retries.

Both stages run inside one SLURM GPU allocation, from a frozen campaign
checkout that is a metadata descendant of the reviewed reader sources. They
authenticate the completed A1 control chain exactly as the frozen stage
controller does, rebuild the same temporal control bundle and write their
results under their own output root. Nothing under the A1 output root is
written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

# Same first imports, in the same order, as the frozen entry point
# (python -m pose_temponet run): the numerical context is authenticated below.
from pose_temponet import execution as E, provenance as P
from pose_temponet import runner as R
from pose_temponet import data as D, train as T, workflow as W

from . import core

# Pure Python and an already loaded NumPy: imported after the frozen package, the numerical census is unchanged.
from dataclasses import asdict
import numpy as np

PLAN_REPO_PATH = "results/infra/pose_temponet_execution_20261121_A1/plan_controls.json"
PLAN_SHA256 = "b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3"
SOURCE_SHA256 = "9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea"
REVIEWED_CODE_SHA = "a1ae9607681c1a181034d7f72770537054af419d"
NUMERIC_GPU8 = "12cb2a0b3207b66e260d3ca490db93ee0c029bb8f03d766df89a9edba6b7eb3f"
# The archived temporal control of the TCN reader: its ancestor chain is authenticated exactly as that job did.
ARCHIVED_CONTROL = "TCNH-controls-temporal-TCN_post-20261120"
RECEIPT_SHA256 = {
    "TCNH-controls-temporal-TCN_post-20261120": "f95c7a5dceb80458998c92a50e4d1efbe3248c972c781055a2689be58fafd84d",
    "TCNH-controls-temporal-L0-20261120": "d299b001c83015dd37b7dc6630e571a663755d6e95a4544ff4b8c634de36b25d",
    "TCNH-controls-temporal-MLP-20261120": "81f6c38eb097a87d5ab2ffd78a597904c86f29627e228f53c9634d00c9b20fcc",
    "TCNH-wls-calibration-0-20261120": "2e78439b72d15a57da56660d64d18d163d2d98e02879cf8cdc3372832a554634",
    "TCNH-wls-calibration-1-20261120": "697d52d894f9c759f64c95488011780a95ebf388db843b2e907476dc799a2dfa",
    "TCNH-preflight-20261120": "0a3314138a98ea3b9e6a8475297c38d5fed575dc112eb52dc4ae4d14d51ee16b",
}
# The archived linear-world controls, read by the weight-decay exploration only.
LINEAR_RECEIPT_SHA256 = {
    "TCNH-controls-linear-L0-20261120": "543d7ba1cd0feceedc6115b3ff9053b0a307a9aa8d02458e6f6f34131ce780c0",
    "TCNH-controls-linear-MLP-20261120": "57cb766e0ebb4fcf97ec2aeb3a052bbb1e9c8dca51f40df75cfe852e35228be2",
    "TCNH-controls-linear-TCN_post-20261120": "0576c92fd784051fc0cf7b03445462200c54f8a66d1bbc313ab382e016047946",
}
OUTPUT_ROOT = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_diag_20261122"
CAMPAIGN = "pose_temponet_diag_20261122"
JOBS = {"d0": "TCND-d0-replay-20261122", "d1": "TCND-d1-cleanfit-20261122", "d2": "TCND-d2-extend-20261122",
        "d3t": "TCND-d3t-wd-20261123", "d3l": "TCND-d3l-wd-20261123", "d3s": "TCND-d3s-wd-20261123"}
# Weight-decay exploration (D3): which reader and which world each job fits, at every dose of core.VARIANT_DOSES.
D3_CELLS = {"d3t": (("TCN_post", "temporal"),), "d3l": (("TCN_post", "linear"),),
            "d3s": (("L0", "linear"), ("MLP", "linear"))}
PHASE2_PROTOCOL = "tcn-temporal-control-weight-decay-exploration-20261123-v1"
CAMPAIGN_D3 = "pose_temponet_diag_20261123"
# Confirmation on a new world (B): two CPU calibrations, six control searches with the chosen dose, one CPU gate.
PHASE3_PROTOCOL = "tcn-temporal-control-new-world-confirmation-20261123-v1"
B_CALIBRATIONS = {"bc0": 0, "bc1": 1}
B_CONTROLS = {"b-%s-%s" % (world, head): (world, head) for world in W.WORLDS for head in W.CONTROL_HEADS}
B_GATE = "bg"
JOBS.update({"bc0": "TCND-b-calib-0-20261123", "bc1": "TCND-b-calib-1-20261123",
             **{stage: "TCND-b-controls-%s-%s-20261123" % cell for stage, cell in B_CONTROLS.items()},
             B_GATE: "TCND-b-gate-20261123"})
CPU_STAGES = {"bc0": 8, "bc1": 8, B_GATE: 2}
SELECTED = {"lr": 0.003, "exposure": 31.0}
# Recoveries of the archived gate (positive_controls_gate_20261121.json, temporal_head_gate.recovery).
ARCHIVED_RECOVERY = {"fold0": 0.24173052888786836, "fold1": 0.1887545039561508, "pooled": 0.21710109906955868}
LEARNING_RATE = 0.003
READING_EXPOSURES = (1, 10, 31, 75, 150)
EXTENSION_SNAPSHOTS = (200, 250, 300)
PROTOCOL = "tcn-temporal-control-diagnostics-20261122-v1"
ARCHIVED_GATE_STATUS = "recipe_not_admitted"
DEVICE = "cuda"
RECIPE = T.Recipe()


class GateFailed(RuntimeError):
    """A registered validity gate did not pass; the chain stops, nothing is relaxed."""


def log(record):
    print(json.dumps(core.plain(record), sort_keys=True, allow_nan=False), flush=True)


def allocation():
    environment = os.environ
    P.require(str(environment.get("SLURM_JOB_ID", "")).isdigit(), "not a SLURM job")
    P.require(environment.get("SLURM_RESTART_COUNT", "0") == "0", "requeue/restart forbidden")
    P.require(int(environment.get("SLURM_CPUS_PER_TASK", 0)) == 8, "the frozen GPU profile uses eight CPUs")
    import torch
    P.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one allocated GPU required")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", environment["SLURM_JOB_ID"]], text=True)
    return {"slurm_job_id": environment["SLURM_JOB_ID"], "scontrol": raw.strip()}


def allocation_cpu(cpus):
    environment = os.environ
    P.require(str(environment.get("SLURM_JOB_ID", "")).isdigit(), "not a SLURM job")
    P.require(environment.get("SLURM_RESTART_COUNT", "0") == "0", "requeue/restart forbidden")
    P.require(int(environment.get("SLURM_CPUS_PER_TASK", 0)) == cpus, "unexpected CPU allocation")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", environment["SLURM_JOB_ID"]], text=True)
    return {"slurm_job_id": environment["SLURM_JOB_ID"], "scontrol": raw.strip()}


def reviewed_plan():
    """The released controls-phase plan, the untouched reader sources and this descendant checkout."""
    plan_path = P.ROOT / PLAN_REPO_PATH
    plan_pin = P.file_pin(plan_path)
    P.require(plan_pin["sha256"] == PLAN_SHA256, "controls phase plan differs from the pinned one")
    plan = P.load_plan(plan_path, require_review=True)
    P.require(plan["reader"]["source_sha256"] == SOURCE_SHA256 and P.source_identity() == SOURCE_SHA256 and
              plan["reader"]["code_sha"] == REVIEWED_CODE_SHA, "frozen reader sources differ from the reviewed ones")
    P.require(plan["numeric_pins"]["gpu8"] == NUMERIC_GPU8 and
              E.find_job(plan, ARCHIVED_CONTROL)["stage"] == "controls", "unexpected controls phase plan")
    return plan, plan_pin, P.checkout_identity(plan)


def setup():
    """Authenticate plan, sources, numerics, receipts and inputs; rebuild the archived bundle."""
    plan, plan_pin, code = reviewed_plan()
    T.configure_numerics()
    numeric = P.numeric_context("gpu")
    P.verify_numeric(plan, "controls", numeric)
    P.require(P.digest(numeric) == NUMERIC_GPU8 == plan["numeric_pins"]["gpu8"],
              "numerical context differs from the six controls")
    # The same call as the archived control job: preflight, the two benchmarks and the two calibrations.
    receipts, pins = R._ancestors(plan, E.find_job(plan, ARCHIVED_CONTROL))
    memo, control_rows = {}, {}
    for head in W.CONTROL_HEADS:
        identifier = E.job_id("controls", head=head, world=core.WORLD)
        control_rows[head], pins[identifier] = R._receipt(plan, identifier, verification_cache=memo)
    for identifier, expected in RECEIPT_SHA256.items():
        P.require(pins[identifier]["sha256"] == expected, "receipt differs from its archived mirror: " + identifier)
    previous = R._single(receipts, "preflight")["details"]["preflight"]
    P.require(previous["integrity"] is True and bool(previous["historical_R1"].get("passed", False)),
              "preflight gates are not passed")
    calibrated = R._calibrations(receipts)
    closed = W.combined_closed_form_gate(calibrated)
    P.require(closed["status"] == "diagnostic" and
              all(c["calibration"]["status"] == "calibrated" for c in calibrated),
              "closed-form references are not diagnostic")
    cache = P.read_verified_inputs(plan, "train", stage="controls")
    bundle = W.prepare_control_folds(cache, calibrated, core.WORLD,
                                     allowed_train_users=plan["producers"]["train"]["users"],
                                     source_identity=R._input_identity(plan),
                                     expected_folds=previous["train"]["folds"])
    controls = {}
    for head, row in control_rows.items():
        selection = R._json_artifact(row, "selection")
        P.require(selection["world"] == core.WORLD and selection["head"] == head and row["stage"] == "controls",
                  "temporal control selection label mismatch")
        controls[head] = {"receipt": row, "selection": selection, "pin": pins[row["job_id"]]}
    P.require(set(controls) == set(W.CONTROL_HEADS), "three temporal control selections are required")
    selected = controls[core.HEAD]["selection"]["selected"]
    P.require(selected["lr"] == SELECTED["lr"] == LEARNING_RATE and selected["exposure"] == SELECTED["exposure"] and
              len(controls[core.HEAD]["selection"]["curves"]) == 5, "archived TCN selection is not the diagnosed one")
    identity = core.check_bundle(bundle, controls[core.HEAD]["selection"])
    references = core.reference_statistics(bundle, calibrated)
    # Descriptive extras: recorded with their own self-check, never a gate.
    try:
        clean_scale = core.clean_scale_references(bundle, calibrated)
    except Exception as error:
        clean_scale = {"per_fold": [], "self_check": [], "self_check_passed": False,
                       "error": type(error).__name__ + ": " + str(error)}
    references["references"]["clean_scale"] = clean_scale
    references["generator"] = core.generator_summary(calibrated)
    gate_records = bundle.reference_gate["temporal_records"]
    held = {name: {k: core.panel_values(references["references"]["held"][k])[name]["E_dyn"]
                   for k in ("clean_ls", "oracle")} for name in core.PANELS}
    P.require(all(held[name][k] == gate_records[name][k] for name in core.PANELS for k in ("clean_ls", "oracle")),
              "held references differ from the frozen gate records")
    return {"plan": plan, "plan_pin": plan_pin, "code": code, "numeric": numeric, "pins": pins,
            "bundle": bundle, "calibrated": calibrated, "controls": controls,
            "evaluation_identity": identity, "references": references, "folds": previous["train"]["folds"]}


def setup_calibration():
    """Authentication for the CPU calibration of the confirmation world, exactly as the archived calibrations."""
    plan, plan_pin, code = reviewed_plan()
    T.configure_numerics()
    numeric = P.numeric_context("cpu")
    P.verify_numeric(plan, "wls_calibration", numeric)
    P.require(P.digest(numeric) == plan["numeric_pins"]["cpu8"], "numerical context differs from the calibrations")
    receipts, pins = R._ancestors(plan, E.find_job(plan, "TCNH-wls-calibration-0-20261120"))
    P.require(pins["TCNH-preflight-20261120"]["sha256"] == RECEIPT_SHA256["TCNH-preflight-20261120"],
              "preflight receipt differs from its archived mirror")
    previous = R._single(receipts, "preflight")["details"]["preflight"]
    P.require(previous["integrity"] is True and bool(previous["historical_R1"].get("passed", False)),
              "preflight gates are not passed")
    cache = P.read_verified_inputs(plan, "train", stage="wls_calibration")
    cache.pop("pre", None)
    return {"plan": plan, "plan_pin": plan_pin, "code": code, "numeric": numeric, "pins": pins,
            "cache": cache, "folds": previous["train"]["folds"], "users": plan["producers"]["train"]["users"]}


def setup_variant():
    """The setup of D0-D2, plus the three archived linear-world controls and the rebuilt linear bundle."""
    context = setup()
    plan, pins, memo, controls = context["plan"], context["pins"], {}, {}
    for head in W.CONTROL_HEADS:
        identifier = E.job_id("controls", head=head, world="linear")
        row, pins[identifier] = R._receipt(plan, identifier, verification_cache=memo)
        P.require(pins[identifier]["sha256"] == LINEAR_RECEIPT_SHA256[identifier],
                  "receipt differs from its archived mirror: " + identifier)
        selection = R._json_artifact(row, "selection")
        P.require(selection["world"] == "linear" and selection["head"] == head and row["stage"] == "controls",
                  "linear control selection label mismatch")
        controls[head] = {"receipt": row, "selection": selection, "pin": pins[identifier]}
    linear = W.prepare_control_folds(context["bundle"].cache, context["calibrated"], "linear",
                                     allowed_train_users=plan["producers"]["train"]["users"],
                                     source_identity=R._input_identity(plan), expected_folds=context["folds"])
    for head, control in controls.items():
        core.check_world_bundle(linear, control["selection"], head)
    return {**context, "linear_bundle": linear, "linear_controls": controls}


def _states(context, head):
    return R._torch_artifact(context["controls"][head]["receipt"], "states")


def _recoveries(summary, split):
    return {name: summary[split]["recovery_on_observed"][name]["recovery"] for name in core.PANELS}


def recovery_bounds(level, archived, bundle):
    """How far the replayed recoveries may sit from the archived ones, for each panel.

    Exact parity: the frozen tolerance. Numerical parity: a relative difference
    of NUMERICAL_TOLERANCE on both dynamic accumulators moves E_dyn by at most
    about twice that, hence the recovery by 2 eps E_dyn / (E_cleanLS - E_oracle);
    the factor 2.2 covers the second-order terms.
    """
    records, values = bundle.reference_gate["temporal_records"], archived["selected_outer_values"]
    bounds = {}
    for name in core.PANELS:
        bounds[name] = core.TOLERANCE * (1 + abs(ARCHIVED_RECOVERY[name]))
        if level != "exact":
            margin = records[name]["clean_ls"] - records[name]["oracle"]
            bounds[name] += 2.2 * core.NUMERICAL_TOLERANCE * values[name]["E_dyn"] / margin
    return bounds


def run_d0(context, directory, root, artifacts):
    """Replay the archived states: 20 of the TCN reader, the four selected of L0 and MLP."""
    replays = {}
    for head, scope in ((core.HEAD, "all"), ("L0", "selected_best"), ("MLP", "selected_best")):
        states = _states(context, head)
        replays[head] = core.replay_head(context["bundle"], head, context["controls"][head]["selection"], states,
                                         context["references"]["references"], device=DEVICE, recipe=RECIPE,
                                         scope=scope, log=log)
        del states
    # Best and final pair of every learning rate for the TCN reader; the selected pair of L0 and MLP.
    P.require(len(replays[core.HEAD]["records"]) == 2 * len(context["controls"][core.HEAD]["selection"]["curves"]) and
              all(len(replays[h]["records"]) == 1 for h in ("L0", "MLP")), "unexpected number of replayed pairs")
    selected = next(r for r in replays[core.HEAD]["records"] if r["selected"])
    archived = context["controls"][core.HEAD]["selection"]
    gate = core.frozen_gate(selected["summary"], context["bundle"])
    final = next(r for r in replays[core.HEAD]["records"]
                 if r["kind"] == "current_states" and r["lr"] == archived["selected"]["lr"])
    comparison = {"archived_selected": archived["selected"], "archived_recovery": ARCHIVED_RECOVERY,
                  "recovery_held": _recoveries(selected["summary"], "held"),
                  "recovery_fitting": _recoveries(selected["summary"], "fitting"),
                  "frozen_gate_on_replayed_selected_point": gate,
                  "held_minus_fitting_selected": core.gap_decomposition(selected["summary"]),
                  "held_minus_fitting_final": core.gap_decomposition(final["summary"]),
                  "final_exposure": final["exposure"]}
    level = core.parity_level([p for r in replays.values() for p in r["parity"]])
    comparison["recovery_bounds"] = recovery_bounds(level, archived, context["bundle"])
    gates = {"state_parity": level != "failed",
             "held_oracle_self_check": context["references"]["held_oracle_self_check_passed"],
             # The replayed selected point gives back the archived recoveries and the archived gate status.
             "gate_arithmetic": all(comparison["recovery_held"][name] is not None and
                                    core.close(comparison["recovery_held"][name], gate["recovery"][name]) and
                                    abs(comparison["recovery_held"][name] - ARCHIVED_RECOVERY[name]) <=
                                    comparison["recovery_bounds"][name]
                                    for name in core.PANELS) and gate["status"] == ARCHIVED_GATE_STATUS}
    # Descriptive extra, never a gate: the known clean function written into the reader and scored as a reader.
    try:
        teacher = core.teacher_scores(context["bundle"], context["calibrated"], context["references"]["references"],
                                      device=DEVICE)
    except Exception as error:
        teacher = {"error": type(error).__name__ + ": " + str(error)}
    result = {"protocol": PROTOCOL, "stage": "d0", "tolerance": core.TOLERANCE,
              "reference_tolerance": core.REFERENCE_TOLERANCE,
              "numerical_tolerance": core.NUMERICAL_TOLERANCE, "parity_level": level,
              "references": context["references"], "replays": replays, "selected_comparison": comparison,
              "teacher_embedded_in_reader": teacher, "gates": gates}
    artifacts["replay"] = P.atomic_json(directory / "replay.json", core.plain(result), root=root)
    return gates


def run_d1(context, directory, root, artifacts):
    """Driver parity on the observed target, then the two fits on the clean target."""
    archived = context["controls"][core.HEAD]["selection"]
    states = _states(context, core.HEAD)
    parity = core.first_epoch_parity(context["bundle"], archived, states, context["references"]["references"],
                                     device=DEVICE, recipe=RECIPE, lr=LEARNING_RATE, log=log)
    del states
    artifacts["parity"] = P.atomic_json(directory / "parity.json", core.plain(parity), root=root)
    gates = {"driver_parity": parity["parity_level"] != "failed",
             "held_oracle_self_check": context["references"]["held_oracle_self_check_passed"]}
    if not all(gates.values()):
        # No clean fit: the failed gates go into the receipt, the stage closes as gate_failed.
        log({"event": "d1_stopped_before_clean_fit", "gates": gates})
        return gates
    fit = core.clean_fit(context["bundle"], archived, context["references"]["references"], device=DEVICE,
                         recipe=RECIPE, lr=LEARNING_RATE, snapshots=READING_EXPOSURES + EXTENSION_SNAPSHOTS, log=log)
    saved = fit.pop("states")
    artifacts["states"] = P.atomic_torch(directory / "clean_fit_states.pt", saved, root=root)
    held, fitting = _recoveries(fit["selected_summary"], "held"), _recoveries(fit["selected_summary"], "fitting")
    gate = core.frozen_gate(fit["selected_summary"], context["bundle"])
    gates["selected_point_reproduced"] = bool(fit["selected_reproduced"])
    gates["gate_arithmetic"] = all(held[name] is not None and gate["recovery"][name] is not None and
                                   core.close(held[name], gate["recovery"][name]) for name in core.PANELS)
    reading = core.reading_rule(held, fitting, fit["late"], all(gates.values()))
    result = {"protocol": PROTOCOL, "stage": "d1", "tolerance": core.TOLERANCE,
              "numerical_tolerance": core.NUMERICAL_TOLERANCE, "driver_parity_level": parity["parity_level"],
              "reading": reading, "recovery_held": held, "recovery_fitting": fitting,
              "frozen_gate_on_selected_point": gate, "archived_noisy_selection": archived["selected"],
              "fixed_readings": core.fixed_readings(fit, READING_EXPOSURES + EXTENSION_SNAPSHOTS), "clean_fit": fit,
              "gates": gates,
              "references": context["references"],
              "note": "Training target y*, selection and scores on observed y; a pass does not admit the recipe."}
    artifacts["clean_fit"] = P.atomic_json(directory / "clean_fit.json", core.plain(result), root=root)
    log({"event": "d1_reading", "reading": reading, "recovery_held": held, "recovery_fitting": fitting,
         "selected": fit["selected"], "late": fit["late"]})
    return gates


def run_d2(context, directory, root, artifacts):
    """Descriptive: the archived noisy trajectories at the selected learning rate, continued to the frozen cap."""
    archived = context["controls"][core.HEAD]["selection"]
    states = _states(context, core.HEAD)
    extension = core.extend_noisy_fit(context["bundle"], archived, states, context["references"]["references"],
                                      device=DEVICE, recipe=RECIPE, lr=LEARNING_RATE, snapshots=EXTENSION_SNAPSHOTS,
                                      log=log)
    del states
    saved = extension.pop("states")
    gates = {"restored_state_parity": extension["start_parity"]["parity_level"] != "failed",
             "held_oracle_self_check": context["references"]["held_oracle_self_check_passed"]}
    if saved is not None:
        artifacts["states"] = P.atomic_torch(directory / "noisy_extension_states.pt", saved, root=root)
    result = {"protocol": PROTOCOL, "stage": "d2", "tolerance": core.TOLERANCE, "extension": extension,
              "archived_selected": archived["selected"], "archived_recovery": ARCHIVED_RECOVERY, "gates": gates,
              "references": context["references"],
              "note": "Descriptive only: no selection and no gate; A1 stays recipe_not_admitted whatever these values."}
    artifacts["extension"] = P.atomic_json(directory / "noisy_extension.json", core.plain(result), root=root)
    return gates


def run_d3(context, directory, root, artifacts, cells):
    """Weight-decay exploration: every dose, one learning rate, frozen calendar and selection; results written as they come."""
    rows = {}
    for dose in core.VARIANT_DOSES:
        recipe = core.variant_recipe(dose, RECIPE)
        for head, world in cells:
            temporal = world == core.WORLD
            bundle = context["bundle"] if temporal else context["linear_bundle"]
            selection = (context["controls"] if temporal else context["linear_controls"])[head]["selection"]
            result = core.variant_fit(bundle, selection, context["references"]["references"] if temporal else None,
                                      device=DEVICE, recipe=recipe, lr=LEARNING_RATE, head=head,
                                      fitting_exposures=READING_EXPOSURES + EXTENSION_SNAPSHOTS, log=log)
            states = result.pop("states")
            key = "%s_%s_wd%s" % (world, head, repr(float(dose)))
            artifacts[key] = P.atomic_json(directory / (key + ".json"),
                                           core.plain({"protocol": PHASE2_PROTOCOL, **result}), root=root)
            if head == core.HEAD:
                artifacts[key + "_states"] = P.atomic_torch(directory / (key + "_states.pt"), states, root=root)
            rows[key] = {"head": head, "world": world, "weight_decay": float(dose), "gate": result["gate"],
                         "selected": result["selected"], "late": result["late"],
                         "extended": result["extended_by_frozen_rule"], "reproduced": result["selected_reproduced"]}
            log({"event": "variant_done", **rows[key]})
    gates = {"held_oracle_self_check": context["references"]["held_oracle_self_check_passed"],
             "selected_points_reproduced": all(row["reproduced"] for row in rows.values())}
    artifacts["summary"] = P.atomic_json(directory / "variant_summary.json",
                                         core.plain({"protocol": PHASE2_PROTOCOL, "lr": LEARNING_RATE,
                                                     "doses": core.VARIANT_DOSES, "rows": rows, "gates": gates}),
                                         root=root)
    return gates


def own_receipt(root, stage):
    """A completed receipt of an earlier stage of this chain, read through its pin."""
    pin = P.file_pin(root / "receipts" / (JOBS[stage] + ".json"))
    receipt = P.read_json_pin(pin)
    P.require(receipt["status"] == "completed" and receipt["job_id"] == JOBS[stage], stage + " did not complete")
    return receipt, pin


def own_torch(receipt, name):
    import torch
    check = P.verify_pin(receipt["artifacts"][name])
    value = torch.load(receipt["artifacts"][name]["path"], map_location="cpu", weights_only=False)
    P.unchanged({receipt["artifacts"][name]["path"]: check["state"]})
    return value


def d3_decision(root, predecessors):
    """The pre-registered dose rule applied to the three authenticated D3 summaries."""
    rows = {}
    for stage in D3_CELLS:
        receipt, pin = own_receipt(root, stage)
        predecessors[JOBS[stage]] = pin
        rows.update(P.read_json_pin(receipt["artifacts"]["summary"])["rows"])
    return core.choose_dose(core.d3_results(rows))


def new_world_artifacts(root, predecessors):
    artifacts = []
    for stage, fold in sorted(B_CALIBRATIONS.items(), key=lambda item: item[1]):
        receipt, pin = own_receipt(root, stage)
        predecessors[JOBS[stage]] = pin
        artifact = own_torch(receipt, "calibration_state")
        P.require(artifact["outer_fold"] == fold and artifact["components"]["seed"] == core.new_world_seed(fold),
                  "calibration of another fold or world")
        artifacts.append(artifact)
    return artifacts


def run_bc(context, fold, directory, root, artifacts):
    """Calibration of one outer fold of the confirmation world, as the archived wls_calibration stage."""
    cache = context["cache"]
    groups = D.alternating_user_folds(cache, context["folds"])
    fit_users = set(groups[fold])
    fit = np.asarray([i for i, m in enumerate(cache["meta"]) if m["user"] in fit_users])
    held = np.asarray([i for i, m in enumerate(cache["meta"]) if m["user"] not in fit_users])
    result = core.calibrate_new_world(cache, fit, held, context["users"], fold, block_windows=32)
    artifacts["calibration_state"] = P.atomic_torch(directory / "calibration.pt", result, root=root)
    summary = {k: v for k, v in result.items() if k != "components"}
    artifacts["calibration"] = P.atomic_json(directory / "calibration.json", R._plain(summary), root=root)
    log({"event": "new_world_calibration", "fold": fold, "status": result["calibration"]["status"],
         "selected": result["calibration"].get("selected")})
    return {"calibrated": result["calibration"]["status"] == "calibrated",
            "new_world_seed": result["components"]["seed"] == core.new_world_seed(fold)}


def run_b_control(context, cell, directory, root, artifacts, predecessors):
    """One frozen control search on the confirmation world with the dose chosen by the pre-registered rule."""
    world, head = cell
    decision = d3_decision(root, predecessors)
    if not decision["confirm"] or decision["chosen"] is None:
        raise GateFailed("the pre-registered dose rule does not propose a confirmation")
    calibrated = new_world_artifacts(root, predecessors)
    closed = W.combined_closed_form_gate(calibrated)
    if closed["status"] != "diagnostic":
        raise GateFailed("confirmation world is not diagnostic: " + str(closed["status"]))
    plan = context["plan"]
    bundle = W.prepare_control_folds(context["bundle"].cache, calibrated, world,
                                     allowed_train_users=plan["producers"]["train"]["users"],
                                     source_identity=R._input_identity(plan), expected_folds=context["folds"])
    recipe = core.variant_recipe(decision["chosen"], RECIPE)
    progress = lambda pair: log({"event": "selection_checkpoint", "lr": pair.lr, "point": core.plain(asdict(pair.points[-1]))}) \
        if pair.points else None
    result = W._run_selection(bundle, head, recipe=recipe, device=DEVICE, on_checkpoint=progress)
    R._save_results(root, directory, result, artifacts)
    artifacts["decision"] = P.atomic_json(directory / "dose_decision.json", core.plain(decision), root=root)
    log({"event": "b_control_done", "world": world, "head": head, "dose": decision["chosen"],
         "selected": result["selection"]["selected"], "alarms": result["selection"]["optimizer_alarms"]})
    return {"d3_confirm": True, "new_world_diagnostic": True}


def run_b_gate(directory, root, artifacts, predecessors):
    """The frozen positive-control gate on the confirmation world, from the six authenticated searches."""
    calibrated = new_world_artifacts(root, predecessors)
    selections = {world: {} for world in W.WORLDS}
    for stage, (world, head) in B_CONTROLS.items():
        receipt, pin = own_receipt(root, stage)
        predecessors[JOBS[stage]] = pin
        selections[world][head] = P.read_json_pin(receipt["artifacts"]["selection"])
    gate = W.qualify_controls(calibrated, selections)
    artifacts["gate"] = P.atomic_json(directory / "gate.json", core.plain(gate), root=root)
    log({"event": "new_world_gate", "status": gate["status"],
         "temporal": gate.get("temporal_head_gate", {}).get("recovery"),
         "linear": {h: g.get("status") for h, g in gate.get("linear_head_gates", {}).items()}})
    return {"gate_computed": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(JOBS))
    stage = parser.parse_args(argv).stage
    job_id = JOBS[stage]
    P.require(os.environ.get("TCND_JOB_ID") == job_id, "logical job ID differs from the launcher")
    root = P.safe_path(OUTPUT_ROOT)
    directory = root / stage
    receipt_path = root / "receipts" / (job_id + ".json")
    P.require(not directory.exists() and not receipt_path.exists(),
              "existing stage output or receipt: do not duplicate or retry")
    started = time.perf_counter()
    artifacts, gates, details, predecessors = {}, {}, {}, {}
    status, context, failure = "technical_failure", None, None
    try:
        details["allocation"] = allocation_cpu(CPU_STAGES[stage]) if stage in CPU_STAGES else allocation()
        if stage != "d0":
            previous = P.file_pin(root / "receipts" / (JOBS["d0"] + ".json"))
            P.require(P.read_json_pin(previous)["status"] == "completed",
                      "D0 did not complete: no later stage may run")
            predecessors[JOBS["d0"]] = previous
        if stage in B_CONTROLS:
            for calibration in B_CALIBRATIONS:  # fail before any GPU work if a calibration did not complete
                predecessors[JOBS[calibration]] = own_receipt(root, calibration)[1]
        if stage == B_GATE:
            plan, plan_pin, code = reviewed_plan()
            context = {"plan": plan, "plan_pin": plan_pin, "code": code}
        elif stage in B_CALIBRATIONS:
            context = setup_calibration()
        else:
            context = setup_variant() if stage in D3_CELLS else setup()
        log({"event": "setup_complete", "stage": stage, "code": context["code"],
             "bundle_sha256": context["bundle"].manifest["bundle_sha256"] if "bundle" in context else None,
             "evaluation_identity": context.get("evaluation_identity")})
        if stage in D3_CELLS:
            gates = run_d3(context, directory, root, artifacts, D3_CELLS[stage])
        elif stage in B_CALIBRATIONS:
            gates = run_bc(context, B_CALIBRATIONS[stage], directory, root, artifacts)
        elif stage in B_CONTROLS:
            gates = run_b_control(context, B_CONTROLS[stage], directory, root, artifacts, predecessors)
        elif stage == B_GATE:
            gates = run_b_gate(directory, root, artifacts, predecessors)
        else:
            gates = {"d0": run_d0, "d1": run_d1, "d2": run_d2}[stage](context, directory, root, artifacts)
        status = "completed" if gates and all(gates.values()) else "gate_failed"
    except GateFailed as error:
        status, failure = "gate_failed", error
    except Exception as error:  # recorded below, then re-raised
        status, failure = "technical_failure", error
    details.update(elapsed_seconds=time.perf_counter() - started,
                   finished_at=datetime.now(timezone.utc).isoformat())
    if failure is not None:
        details.update(error_type=type(failure).__name__, error=str(failure))
    phase = PHASE3_PROTOCOL if stage in (*B_CALIBRATIONS, *B_CONTROLS, B_GATE) else (
        PHASE2_PROTOCOL if stage in D3_CELLS else PROTOCOL)
    receipt = {"schema": "tcn-temporal-diag-receipt-v1", "protocol": phase, "job_id": job_id, "stage": stage,
               "status": status, "complete": status == "completed",
               "campaign": CAMPAIGN if stage in ("d0", "d1", "d2") else CAMPAIGN_D3,
               "source_sha256": SOURCE_SHA256, "reviewed_code_sha": REVIEWED_CODE_SHA,
               "plan": {"path": PLAN_REPO_PATH, "sha256": PLAN_SHA256},
               "predecessors": predecessors, "artifacts": artifacts, "gates": gates, "details": details}
    if context is not None:
        receipt["code"] = context["code"]
        if "numeric" in context:
            receipt.update(numeric_context=context["numeric"], numeric_sha256=P.digest(context["numeric"]))
        if "bundle" in context:
            receipt.update(bundle_sha256=context["bundle"].manifest["bundle_sha256"],
                           evaluation_identity=context["evaluation_identity"])
        if "pins" in context:
            receipt["a1_receipts"] = {k: context["pins"][k] for k in sorted(RECEIPT_SHA256) if k in context["pins"]}
    receipt = core.plain(receipt)
    receipt["receipt_sha256"] = P.digest(receipt)
    try:
        log({"event": "receipt", "status": status, **P.atomic_json(receipt_path, receipt, root=root)})
    except Exception as recording_error:
        log({"event": "receipt_unavailable", "status": status, "recording_error": str(recording_error),
             "original_error": None if failure is None else str(failure)})
        if failure is None:
            raise
    if failure is not None and not isinstance(failure, GateFailed):
        raise failure
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
