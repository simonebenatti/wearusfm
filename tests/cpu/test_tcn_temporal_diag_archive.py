"""Diagnostics of the failed temporal control, on the real archived A1 chain mirrored in the repository.

Turns into a CPU check what setup() and run_d0 assume about the archive; every entry-point test replaces them by fixtures.
"""
import json
import os
from pathlib import Path
import sys

TREE = Path(__file__).resolve().parents[2]
for sub in ("tests/cpu", "scripts", "src"):
    sys.path.insert(0, str(TREE / sub))
from pose_temponet import controls as C, execution as E, provenance as P, runner as R, selection as S, workflow as W
from tcn_temporal_diag import core
import tcn_temporal_diag.__main__ as entry

MIRROR = TREE / "results/infra/pose_temponet_execution_20261121_A1/launcher"


def test_setup_assumptions_hold_on_the_archived_a1_chain():
    plan, _, _ = entry.reviewed_plan()
    job = E.find_job(plan, entry.ARCHIVED_CONTROL)
    closure, stack = set(), list(job["after"])
    while stack:
        identifier = stack.pop()
        if identifier not in closure:
            closure.add(identifier)
            stack.extend(E.find_job(plan, identifier)["after"])
    controls = {head: E.job_id("controls", head=head, world=core.WORLD) for head in W.CONTROL_HEADS}
    assert set(entry.RECEIPT_SHA256) <= closure | set(controls.values())      # every pinned receipt is one that setup() reads
    assert plan["output_root"] != entry.OUTPUT_ROOT and Path(plan["output_root"]) not in Path(entry.OUTPUT_ROOT).parents
    memo, receipts = {}, {}
    for identifier in sorted(closure | set(controls.values())):
        pin = plan.get("receipt_pins", {}).get(identifier) or P.file_pin(MIRROR / (identifier + ".json"))
        receipts[identifier] = P.verified_receipt(pin, plan, verify_artifacts=False, receipt_mirror_dir=MIRROR, _memo=memo)
    previous = R._single({k: receipts[k] for k in closure}, "preflight")["details"]["preflight"]
    assert previous["integrity"] is True and previous["historical_R1"]["passed"] is True
    assert [len(f) for f in previous["train"]["folds"]] == [20, 20]
    selections = {}
    for head, identifier in controls.items():
        row = receipts[identifier]
        assert row["stage"] == "controls" and {"selection", "states"} <= set(row["artifacts"])
        path = MIRROR / ("selection_%s_%s.json" % (core.WORLD, head))
        assert P.file_pin(path)["sha256"] == row["artifacts"]["selection"]["sha256"]
        selections[head] = json.loads(path.read_text())
        W._check_selection(selections[head], production=True)
        assert selections[head]["world"] == core.WORLD and selections[head]["head"] == head
    tcn = selections[core.HEAD]
    assert tcn["selected"]["lr"] == entry.SELECTED["lr"] == entry.LEARNING_RATE
    assert tcn["selected"]["exposure"] == entry.SELECTED["exposure"] and len(tcn["curves"]) == 5
    curve = next(c for c in tcn["curves"] if c["lr"] == entry.LEARNING_RATE)
    exposures = {p["exposure"] for p in curve["points"]}
    assert all(float(e) in exposures for e in entry.READING_EXPOSURES) and curve["horizon"] == entry.RECIPE.initial_epochs
    first = [(p["exposure"], tuple(p["fold_steps"])) for p in curve["points"] if p["exposure"] <= 1]
    assert first == S.checkpoint_schedule(list(first[-1][1]), 1) and len(first) == 5
    # References and gate arithmetic exactly as run_d0 computes them, on the archived numbers.
    calibrations = [json.loads((MIRROR / ("calibration_%d.json" % fold)).read_text()) for fold in (0, 1)]
    gate = C.combine_reference_folds(calibrations)
    assert gate["status"] == "diagnostic" and gate == tcn["bundle"]["reference_gate"]
    for artifact in calibrations:
        diagnostic = artifact["outer_reference_statistics"]["linear_diagnostics"]["L0"]
        assert (diagnostic["temporal_margin_fit_target"], diagnostic["temporal_margin_score_target"]) == ("clean_y_star", "observed_y")
    held = {k: [dict(a["outer_reference_statistics"][core.WORLD][k]) for a in calibrations] for k in ("clean_ls", "oracle")}
    reader = [tcn["selected_outer_statistics"]["fold%d" % fold] for fold in (0, 1)]
    recovery = core.recovery_panels(reader, held)
    frozen = C.temporal_head_gate(gate["temporal_records"], {n: core.panel_values(reader)[n]["E_dyn"] for n in core.PANELS})
    assert frozen["status"] == entry.ARCHIVED_GATE_STATUS == "recipe_not_admitted"
    assert all(core.close(recovery[n]["recovery"], frozen["recovery"][n]) for n in core.PANELS)
    assert [round(100 * recovery[n]["recovery"], 2) for n in core.PANELS] == [24.17, 18.88, 21.71]
    fitting = [dict(a["outer_reference_statistics"]["linear_diagnostics"]["L0"]["temporal_margin_clean_fit_statistics"])
               for a in calibrations]
    for fold in (0, 1):  # the archived fitting clean-LS and the archived fitting scores of the reader describe one panel
        scored = tcn["selected_fitting_diagnostics"][str(fold)]
        assert scored["window_count"] == fitting[fold]["window_count"]
        assert core.close(scored["dynamic_target_ss"], fitting[fold]["dynamic_target_ss"], core.REFERENCE_TOLERANCE)
