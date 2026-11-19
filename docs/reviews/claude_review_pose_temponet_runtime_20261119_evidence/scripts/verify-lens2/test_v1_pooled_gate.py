"""Verifier check for L2-1: label of a closed-form (pooled/fold) gate failure in the runner."""
import sys
from pathlib import Path
WT = Path("/Users/simonebenatti/dev/wearusfm/.claude/worktrees/review-pose-heads-2f98211")
sys.path.insert(0, str(WT/"tests/cpu"))
sys.path.insert(0, str(WT/"scripts"))
import numpy as np
import pytest
from pose_temponet import execution as E, provenance as P, runner as R, controls as C
from test_pose_temponet_provenance import prepared  # noqa: F401  (fixture)
from test_pose_temponet_runner import runtime, initial_chain  # noqa: F401
REAL_COMBINE = C.combine_reference_folds  # captured before fixtures patch the module


def _stats(m, n, d, s):
    return {"mse_sum": m, "window_count": n, "dynamic_sse": d, "dynamic_target_ss": s}


def _fold(f, ridge_mse):
    # constant=10, oracle=0 -> recovery = (10-ridge)/10 ; temporal band 8% ok
    lin = {"constant": _stats(100., 10, 10., 10.), "oracle": _stats(0., 10, 0., 10.),
           "clean_ls": _stats(50., 10, 5., 10.), "ridge": _stats(10.*ridge_mse, 10, 0.5, 10.)}
    tmp = {"constant": _stats(100., 10, 10., 10.), "oracle": _stats(92., 10, 9.2, 10.),
           "clean_ls": _stats(100., 10, 10., 10.)}
    return {"outer_fold": f, "calibration": {"status": "calibrated"}, "outer_scored_after_inner_selection": True,
            "outer_reference_statistics": {"linear": lin, "temporal": tmp}}


def test_real_combine_reports_non_diagnostic_for_fold_failure():
    gate = REAL_COMBINE([_fold(0, 1.2), _fold(1, 0.5)])  # fold0 MSE recovery 0.88
    print("\nreal combine_reference_folds status:", gate["status"], gate["linear_gate"]["recovery"])
    assert gate["status"] == "control_non_diagnostic"


def test_runner_labels_closed_form_failure(runtime, monkeypatch):
    repo, plan, opened, _ = initial_chain(runtime)
    for fold in (0, 1):
        R.run_stage(plan, E.job_id("wls_calibration", fold=fold), source_root=repo)
    # Use the REAL combiner on synthetic fold states that are each 'calibrated'
    monkeypatch.setattr(R.controls, "combine_reference_folds", REAL_COMBINE)
    monkeypatch.setattr(R, "_torch_artifact", lambda r, name: _fold(int(r["job_id"][-10]), 1.2 if r["job_id"][-10] == "0" else 0.5)
                        if name == "calibration_state" else pytest.fail("unexpected artifact"))
    for world in E.WORLDS:
        for head in E.CONTROL_HEADS:
            ident = E.job_id("controls", head=head, world=world)
            job = E.find_job(plan, ident)
            with pytest.raises(Exception) as info:
                R.run_stage(plan, ident, source_root=repo)
            receipt = P.read_json_pin(P.file_pin(P.receipt_path(plan, ident)))
            print(ident, "gpus=", job["resources"]["gpus"], "| raised", type(info.value).__name__,
                  "GateNotPassed?", isinstance(info.value, R.GateNotPassed), "|", info.value,
                  "| receipt.status=", receipt["status"], "| artifacts=", sorted(receipt["artifacts"]))
    # no gate artifact anywhere in the output tree
    names = sorted(p.name for p in Path(plan["output_root"]).rglob("*") if p.is_file() and "gate" in p.name)
    print("gate-like files in output root:", names)
    print("cache opens:", opened)


REAL_NORMALIZE = C.normalize_component


def test_calibration_stage_labels(runtime, monkeypatch):
    repo, plan, opened, _ = initial_chain(runtime)
    # (a) calibrate_grid itself returns control_non_diagnostic for invalid E_real
    out = C.calibrate_grid({"E_dyn": None}, lambda a, s: pytest.fail("no pair evaluated"))
    print("\ncalibrate_grid(E_dyn=None) ->", out["status"])
    monkeypatch.setattr(R.controls, "build_positive_control_fold",
                        lambda cache, *a, outer_fold, **k: {"outer_fold": outer_fold, "components": {},
                                                             "calibration": {"status": "control_non_diagnostic"}})
    ident = E.job_id("wls_calibration", fold=0)
    with pytest.raises(Exception) as info:
        R.run_stage(plan, ident, source_root=repo)
    rec = P.read_json_pin(P.file_pin(P.receipt_path(plan, ident)))
    print("(a)", type(info.value).__name__, "|", info.value, "| receipt.status=", rec["status"],
          "| details.calibration_status=", rec["details"].get("calibration_status"))
    # (b) degenerate teacher component inside the calibration builder
    def degenerate(cache, *a, outer_fold, **k):
        value = np.ones((3, 20, 2)); support = np.ones((3, 8, 2), dtype=bool)
        REAL_NORMALIZE(value, support, np.arange(3))
    monkeypatch.setattr(R.controls, "build_positive_control_fold", degenerate)
    ident = E.job_id("wls_calibration", fold=1)
    with pytest.raises(Exception) as info:
        R.run_stage(plan, ident, source_root=repo)
    rec = P.read_json_pin(P.file_pin(P.receipt_path(plan, ident)))
    print("(b)", type(info.value).__name__, "|", info.value, "| receipt.status=", rec["status"])
