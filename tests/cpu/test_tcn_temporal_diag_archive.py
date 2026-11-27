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
    for own in (entry.OUTPUT_ROOT, entry.PHASE1_OUTPUT_ROOT):  # neither root of the diagnostics is inside the frozen A1 root
        assert plan["output_root"] != own and Path(plan["output_root"]) not in Path(own).parents
        assert Path(own) not in Path(plan["output_root"]).parents
    assert entry.PHASE1_OUTPUT_ROOT == "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_diag_20261122"
    assert entry.OUTPUT_ROOT == \
        "/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/pose_temponet_diag_20261123"
    assert [entry.stage_root(s) == Path(entry.PHASE1_OUTPUT_ROOT) for s in ("d0", "d1", "d2")] == [True] * 3
    assert all(entry.stage_root(s) == Path(entry.OUTPUT_ROOT) for s in entry.JOBS if s not in ("d0", "d1", "d2"))
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


def test_setup_variant_assumptions_hold_on_the_archived_linear_controls():
    """What setup_variant and variant_fit read from the three archived linear-world controls, on the mirrored receipts."""
    plan, _, _ = entry.reviewed_plan()
    memo, manifests = {}, {}
    assert set(entry.LINEAR_RECEIPT_SHA256) == {E.job_id("controls", head=h, world="linear") for h in W.CONTROL_HEADS}
    for head in W.CONTROL_HEADS:
        identifier = E.job_id("controls", head=head, world="linear")
        pin = plan.get("receipt_pins", {}).get(identifier) or P.file_pin(MIRROR / (identifier + ".json"))
        assert pin["sha256"] == entry.LINEAR_RECEIPT_SHA256[identifier]
        row = P.verified_receipt(pin, plan, verify_artifacts=False, receipt_mirror_dir=MIRROR, _memo=memo)
        assert row["job_id"] == identifier and row["stage"] == "controls" and {"selection", "states"} <= set(row["artifacts"])
        path = MIRROR / ("selection_linear_%s.json" % head)
        assert P.file_pin(path)["sha256"] == row["artifacts"]["selection"]["sha256"]
        selection = json.loads(path.read_text())
        W._check_selection(selection, production=True)
        assert selection["world"] == "linear" and selection["head"] == head
        # Every field that check_world_bundle, variant_fit and start_point_parity read, on the real artifact.
        identity = W._identity({"bundle": selection["bundle"], "depth": W._feature_depth(head),
                                "feature_array_sha256": selection["feature_array_sha256"]})
        assert identity == selection["evaluation_identity"]
        assert len(selection["initial_sha256_per_fold"]) == 2
        curves = [c for c in selection["curves"] if float(c["lr"]) == entry.LEARNING_RATE]
        assert len(curves) == 1 and curves[0]["horizon"] == entry.RECIPE.initial_epochs
        start = curves[0]["points"][0]
        assert start["exposure"] == 0 and [int(s) for s in start["fold_steps"]] == [0, 0]
        assert start["mse"] > 0 and start["edyn"] > 0
        manifests[head] = selection["bundle"]
    # One rebuilt linear bundle serves the three cells; it shares the temporal world's calibration and reference gate.
    assert manifests["L0"] == manifests["MLP"] == manifests["TCN_post"]
    temporal = json.loads((MIRROR / ("selection_%s_%s.json" % (core.WORLD, core.HEAD))).read_text())["bundle"]
    assert manifests["L0"]["world"] == "linear" and temporal["world"] == core.WORLD
    assert manifests["L0"]["reference_gate"] == temporal["reference_gate"]
    assert manifests["L0"]["artifacts_sha256"] == temporal["artifacts_sha256"]
