#!/usr/bin/env python3
"""Reproduce the temporal-control arithmetic from archived local JSON only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUT = ROOT / "results/infra/pose_temponet_execution_20261121_A1/launcher"
NAMES = (
    "positive_controls_gate_20261121.json",
    "selection_temporal_L0.json",
    "selection_temporal_MLP.json",
    "selection_temporal_TCN_post.json",
    "calibration_0.json",
    "calibration_1.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = {name: INPUT / name for name in NAMES}
    records = {name: json.loads(path.read_text()) for name, path in paths.items()}
    gate = records["positive_controls_gate_20261121.json"]["gate"]
    selections = {
        head: records[f"selection_temporal_{head}.json"]
        for head in ("L0", "MLP", "TCN_post")
    }
    calibrations = {
        f"fold{fold}": records[f"calibration_{fold}.json"]["calibration"]
        for fold in (0, 1)
    }
    for calibration in calibrations.values():
        selected_world = calibration["selected"]["temporal"]
        assert selected_world["a"] == 0.4 and selected_world["sigma"] == 1.3
        assert calibration["status"] == "calibrated"
    summary = {
        "scope": "Independent arithmetic on archived train-only synthetic controls. No training, jobs, or external held data.",
        "analysis_source_sha256": sha256(Path(__file__)),
        "source_inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "gate_status": gate["status"],
        "held": {},
        "fitting": {},
        "inner_calibration": {
            panel: {
                "E_real": calibration["E_real"],
                **{key: calibration["selected"]["temporal"][key]
                   for key in ("a", "sigma", "E_cal", "E_cleanLS", "E_oracle", "oracle_gain_fraction", "boundary_unresolved")},
            }
            for panel, calibration in calibrations.items()
        },
        "synthetic_world": {
            "a": 0.4,
            "sigma": 1.3,
            "noise_rho": math.exp(-0.25),
            "nominal_noise_variance_to_T_variance_ratio": 1.3**2 / 0.4**2,
            "caveat": "L and T are normalized separately on fitting support. This ratio is marginal component variance bookkeeping, not measured dynamic SNR; it excludes covariance and temporal-centering effects.",
        },
        "limits": [
            "Fitting recovery can benefit from fitting realized noise; it does not prove accurate recovery of the clean generator.",
            "Fitting oracle is reconstructed algebraically from the archived L13 clean-fitting margin, not independently evaluated from the arrays.",
            "The neural fitting-clean score is not archived here; recover it from selected checkpoint inference without replaying training.",
            "A held oracle advantage establishes room for improvement, not finite-sample learnability of a random nonlinear teacher at this noise level.",
            "All panels use train-only synthetic splits. These results say nothing directly about the fifteen external held users or missing FM information.",
        ],
    }
    for panel in ("fold0", "fold1", "pooled"):
        reference = gate["reference_gate"]["temporal_records"][panel]
        baseline, oracle = reference["clean_ls"], reference["oracle"]
        tcn = selections["TCN_post"]["selected_outer_values"][panel]["E_dyn"]
        linear = selections["L0"]["selected_outer_values"][panel]["E_dyn"]
        tcn_clean = selections["TCN_post"]["selected_outer_clean_values"][panel]["E_dyn"]
        linear_clean = selections["L0"]["selected_outer_clean_values"][panel]["E_dyn"]
        recovery = (baseline - tcn) / (baseline - oracle)
        saved = gate["temporal_head_gate"]["recovery"][panel]
        assert math.isclose(recovery, saved, rel_tol=0, abs_tol=1e-14)
        summary["held"][panel] = {
            "clean_ls": baseline,
            "oracle": oracle,
            "TCN": tcn,
            "recovery_recomputed": recovery,
            "recovery_saved": saved,
            "admission_maximum_Edyn": (baseline + oracle) / 2,
            "gap_to_admission_Edyn": tcn - (baseline + oracle) / 2,
            "learned_L0": linear,
            "TCN_relative_improvement_vs_learned_L0": (linear - tcn) / linear,
            "L0_clean_target_Edyn": linear_clean,
            "TCN_clean_target_Edyn": tcn_clean,
            "TCN_relative_improvement_vs_L0_on_clean_target": (linear_clean - tcn_clean) / linear_clean,
        }
    pooled = dict(baseline_sse=0.0, oracle_sse=0.0, tcn_sse=0.0, target_ss=0.0)
    for fold in (0, 1):
        diagnostic = gate["linear_diagnostics"][f"fold{fold}"]
        l0 = diagnostic["L0"]["temporal_margin_clean_fit_statistics"]
        l13 = diagnostic["L13"]["temporal_margin_clean_fit_statistics"]
        neural = selections["TCN_post"]["selected_fitting_diagnostics"][str(fold)]
        denominator = l0["dynamic_target_ss"]
        assert denominator == l13["dynamic_target_ss"] == neural["dynamic_target_ss"]
        baseline = l0["dynamic_sse"] / denominator
        contextual = l13["dynamic_sse"] / denominator
        contextual_recovery = diagnostic["L13"]["temporal_margin_recovery_fitting"]
        assert contextual_recovery > 0
        oracle = baseline - (baseline - contextual) / contextual_recovery
        tcn = neural["dynamic_sse"] / denominator
        summary["fitting"][f"fold{fold}"] = {
            "clean_L0": baseline,
            "clean_L13": contextual,
            "L13_recovery": contextual_recovery,
            "oracle_reconstructed_from_clean_L13_margin": oracle,
            "TCN": tcn,
            "TCN_recovery": (baseline - tcn) / (baseline - oracle),
            "TCN_MSE": neural["mse_sum"] / neural["window_count"],
            "TCN_windows": neural["window_count"],
        }
        pooled["baseline_sse"] += l0["dynamic_sse"]
        pooled["oracle_sse"] += oracle * denominator
        pooled["tcn_sse"] += neural["dynamic_sse"]
        pooled["target_ss"] += denominator
    summary["fitting"]["pooled"] = {
        **pooled,
        "TCN_recovery": (pooled["baseline_sse"] - pooled["tcn_sse"]) / (pooled["baseline_sse"] - pooled["oracle_sse"]),
    }
    output = HERE / "independent_calibration_gate_arithmetic.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "fitting_recovery": summary["fitting"]["pooled"]["TCN_recovery"],
        "held_recovery": summary["held"]["pooled"]["recovery_recomputed"],
    }, indent=2))


if __name__ == "__main__":
    main()
