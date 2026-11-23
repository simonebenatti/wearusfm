"""Gaps found by the mutation review of 6386bd0b (23/11): step-zero exposure, step-zero failure and numerical tier through
run_d3, linear pins in the D3 receipt, summaries without a receipt in summarize.py, users of the new-world calibration.

Same fixtures as test_tcn_temporal_diag_d3_rows and test_tcn_temporal_diag_confirmation; no number is a result.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/cpu", "scripts", "src"):
    sys.path.insert(0, str(ROOT / sub))
from pose_temponet import selection as S

import test_tcn_temporal_diag as TD
from test_tcn_temporal_diag import world  # noqa: F401  (module fixture of the first test file)
from test_tcn_temporal_diag_variant import _variant_context, linear_runs  # noqa: F401
from test_tcn_temporal_diag_d3_rows import _run, _summarize, _write_rows
from test_tcn_temporal_diag_confirmation import _patch_b
from tcn_temporal_diag import core
import tcn_temporal_diag.__main__ as entry


def test_step_zero_parity_rejects_an_archived_point_at_another_exposure(world):
    _, selection, _, _ = TD._tcn(world)
    lr = selection["selected"]["lr"]
    curve = S.curve_from_dict(next(c for c in selection["curves"] if c["lr"] == lr))
    moved = copy.deepcopy(selection)
    next(c for c in moved["curves"] if c["lr"] == lr)["points"][0]["exposure"] = 0.5
    assert core.parity_level([core.start_point_parity(curve, moved, lr)]) == "failed"


def test_step_zero_tiers_reach_rows_gate_and_receipt_of_a_d3_job(world, linear_runs, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    TD._patch_entry(monkeypatch, world, root)
    context = _variant_context(world, linear_runs)
    for head, factor in (("L0", 1 + 1e-3), ("MLP", 1 + 1e-6)):  # above both tiers; inside the numerical tier only
        moved = copy.deepcopy(context["linear_controls"][head]["selection"])
        next(c for c in moved["curves"] if float(c["lr"]) == float(entry.LEARNING_RATE))["points"][0]["mse"] *= factor
        context["linear_controls"][head] = {**context["linear_controls"][head], "selection": moved}
    # As on Leonardo, setup_variant leaves the three verified linear pins in the context.
    context["pins"].update({k: {"path": "/" + k, "sha256": v, "bytes": 1} for k, v in entry.LINEAR_RECEIPT_SHA256.items()})
    monkeypatch.setattr(entry, "setup_variant", lambda: context)
    monkeypatch.setattr(core, "VARIANT_DOSES", (0.3,))
    assert _run("d0", monkeypatch) == 0
    assert _run("d3s", monkeypatch) == 3
    summary = json.loads((root / "d3s" / "variant_summary.json").read_text())
    l0, mlp = summary["rows"]["linear_L0_wd0.3"], summary["rows"]["linear_MLP_wd0.3"]
    assert (l0["start_parity_level"], l0["start_reproduced"]) == ("failed", False)
    assert (mlp["start_parity_level"], mlp["start_reproduced"]) == ("numerical", True)
    assert summary["gates"] == {"held_oracle_self_check": True, "selected_points_reproduced": True,
                                "start_points_reproduced": False}
    receipt = json.loads((root / "receipts" / (entry.JOBS["d3s"] + ".json")).read_text())
    assert receipt["status"] == "gate_failed" and receipt["gates"] == summary["gates"]
    assert set(receipt["a1_receipts"]) == set(entry.RECEIPT_SHA256) | set(entry.LINEAR_RECEIPT_SHA256)


def test_summarize_ignores_a_summary_without_its_receipt(tmp_path, capsys):
    four = {0.1: (.30, .31, .305), 0.3: (.60, .62, .61), 1.0: (.55, .56, .555), 3.0: (.20, .21, .205)}
    _write_rows(tmp_path / "f", four)
    (tmp_path / "f" / "receipts" / (entry.JOBS["d3l"] + ".json")).unlink()
    _summarize().d3(tmp_path / "f")
    out = capsys.readouterr().out
    assert "sintesi escluse, job non completati: d3l (ricevuta assente)" in out
    assert "dose scelta None, conferma su un mondo nuovo: False" in out


def test_new_world_calibration_receives_the_users_of_its_own_fold(world, tmp_path, monkeypatch):
    root = tmp_path.resolve() / "diag"
    artifacts = _patch_b(monkeypatch, world, root, {"chosen": 0.3, "confirm": True, "rows": []})
    captured = {}

    def capture(cache, fit, held, users, fold, block_windows):
        captured[fold] = (np.sort(fit), np.sort(held))
        return artifacts[fold]

    monkeypatch.setattr(core, "calibrate_new_world", capture)
    assert _run("d0", monkeypatch) == 0
    for stage, fold in entry.B_CALIBRATIONS.items():
        assert _run(stage, monkeypatch) == 0
        fit, held = captured[fold]
        assert np.array_equal(fit, np.sort(world["bundle"].fitting[fold]))
        assert np.array_equal(held, np.sort(world["bundle"].held[fold]))
