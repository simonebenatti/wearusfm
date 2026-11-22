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
from pose_temponet import train as T, workflow as W

from . import core

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
OUTPUT_ROOT = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_diag_20261122"
CAMPAIGN = "pose_temponet_diag_20261122"
JOBS = {"d0": "TCND-d0-replay-20261122", "d1": "TCND-d1-cleanfit-20261122"}
SELECTED = {"lr": 0.003, "exposure": 31.0}
LEARNING_RATE = 0.003
READING_EXPOSURES = (1, 10, 31, 75, 150)
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
    gate_records = bundle.reference_gate["temporal_records"]
    held = {name: {k: core.panel_values(references["references"]["held"][k])[name]["E_dyn"]
                   for k in ("clean_ls", "oracle")} for name in core.PANELS}
    P.require(all(held[name][k] == gate_records[name][k] for name in core.PANELS for k in ("clean_ls", "oracle")),
              "held references differ from the frozen gate records")
    return {"plan": plan, "plan_pin": plan_pin, "code": code, "numeric": numeric, "pins": pins,
            "bundle": bundle, "calibrated": calibrated, "controls": controls,
            "evaluation_identity": identity, "references": references}


def _states(context, head):
    return R._torch_artifact(context["controls"][head]["receipt"], "states")


def _recoveries(summary, split):
    return {name: summary[split]["recovery_on_observed"][name]["recovery"] for name in core.PANELS}


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
    comparison = {"archived_selected": archived["selected"],
                  "recovery_held": _recoveries(selected["summary"], "held"),
                  "recovery_fitting": _recoveries(selected["summary"], "fitting"),
                  "frozen_gate_on_replayed_selected_point": gate,
                  "held_minus_fitting_selected": core.gap_decomposition(selected["summary"]),
                  "held_minus_fitting_final": core.gap_decomposition(final["summary"]),
                  "final_exposure": final["exposure"]}
    gates = {"state_parity": all(r["parity_passed"] for r in replays.values()),
             "held_oracle_self_check": context["references"]["held_oracle_self_check_passed"],
             "gate_arithmetic": all(core.close(comparison["recovery_held"][name], gate["recovery"][name])
                                    for name in core.PANELS) and gate["status"] == ARCHIVED_GATE_STATUS}
    result = {"protocol": PROTOCOL, "stage": "d0", "tolerance": core.TOLERANCE,
              "reference_tolerance": core.REFERENCE_TOLERANCE,
              "references": context["references"], "replays": replays, "selected_comparison": comparison,
              "gates": gates}
    artifacts["replay"] = P.atomic_json(directory / "replay.json", core.plain(result), root=root)
    return gates


def run_d1(context, directory, root, artifacts):
    """Driver parity on the observed target, then the two fits on the clean target."""
    archived = context["controls"][core.HEAD]["selection"]
    states = _states(context, core.HEAD)
    parity = core.first_epoch_parity(context["bundle"], archived, states, device=DEVICE, recipe=RECIPE,
                                     lr=LEARNING_RATE, log=log)
    del states
    artifacts["parity"] = P.atomic_json(directory / "parity.json", core.plain(parity), root=root)
    gates = {"driver_parity": bool(parity["passed"]),
             "held_oracle_self_check": context["references"]["held_oracle_self_check_passed"]}
    if not all(gates.values()):
        raise GateFailed("driver parity or reference self-check failed: no clean fit was run")
    fit = core.clean_fit(context["bundle"], archived, context["references"]["references"], device=DEVICE,
                         recipe=RECIPE, lr=LEARNING_RATE, snapshots=READING_EXPOSURES, log=log)
    saved = fit.pop("states")
    artifacts["states"] = P.atomic_torch(directory / "clean_fit_states.pt", saved, root=root)
    held, fitting = _recoveries(fit["selected_summary"], "held"), _recoveries(fit["selected_summary"], "fitting")
    gate = core.frozen_gate(fit["selected_summary"], context["bundle"])
    gates["selected_point_reproduced"] = bool(fit["selected_reproduced"])
    gates["gate_arithmetic"] = all(held[name] is not None and gate["recovery"][name] is not None and
                                   core.close(held[name], gate["recovery"][name]) for name in core.PANELS)
    reading = core.reading_rule(held, fitting, fit["late"], all(gates.values()))
    result = {"protocol": PROTOCOL, "stage": "d1", "tolerance": core.TOLERANCE,
              "reading": reading, "recovery_held": held, "recovery_fitting": fitting,
              "frozen_gate_on_selected_point": gate, "archived_noisy_selection": archived["selected"],
              "fixed_readings": core.fixed_readings(fit, READING_EXPOSURES), "clean_fit": fit, "gates": gates,
              "note": "Training target y*, selection and scores on observed y; a pass does not admit the recipe."}
    artifacts["clean_fit"] = P.atomic_json(directory / "clean_fit.json", core.plain(result), root=root)
    log({"event": "d1_reading", "reading": reading, "recovery_held": held, "recovery_fitting": fitting,
         "selected": fit["selected"], "late": fit["late"]})
    return gates


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
        details["allocation"] = allocation()
        if stage == "d1":
            previous = P.file_pin(root / "receipts" / (JOBS["d0"] + ".json"))
            P.require(P.read_json_pin(previous)["status"] == "completed", "D0 did not complete: D1 must not run")
            predecessors[JOBS["d0"]] = previous
        context = setup()
        log({"event": "setup_complete", "stage": stage, "code": context["code"],
             "bundle_sha256": context["bundle"].manifest["bundle_sha256"],
             "evaluation_identity": context["evaluation_identity"]})
        gates = (run_d0 if stage == "d0" else run_d1)(context, directory, root, artifacts)
        status = "completed" if gates and all(gates.values()) else "gate_failed"
    except GateFailed as error:
        status, failure = "gate_failed", error
    except Exception as error:  # recorded below, then re-raised
        status, failure = "technical_failure", error
    details.update(elapsed_seconds=time.perf_counter() - started,
                   finished_at=datetime.now(timezone.utc).isoformat())
    if failure is not None:
        details.update(error_type=type(failure).__name__, error=str(failure))
    receipt = {"schema": "tcn-temporal-diag-receipt-v1", "protocol": PROTOCOL, "job_id": job_id, "stage": stage,
               "status": status, "complete": status == "completed", "campaign": CAMPAIGN,
               "source_sha256": SOURCE_SHA256, "reviewed_code_sha": REVIEWED_CODE_SHA,
               "plan": {"path": PLAN_REPO_PATH, "sha256": PLAN_SHA256},
               "predecessors": predecessors, "artifacts": artifacts, "gates": gates, "details": details}
    if context is not None:
        receipt.update(code=context["code"], numeric_context=context["numeric"],
                       numeric_sha256=P.digest(context["numeric"]),
                       bundle_sha256=context["bundle"].manifest["bundle_sha256"],
                       evaluation_identity=context["evaluation_identity"],
                       a1_receipts={k: context["pins"][k] for k in sorted(RECEIPT_SHA256)})
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
