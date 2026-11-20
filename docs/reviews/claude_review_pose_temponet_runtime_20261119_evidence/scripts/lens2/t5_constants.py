import json, math
from dataclasses import asdict
from pose_temponet import selection as S, controls as C, train as T, heads as H, linear as Lm, workflow as W
j = json.load(open("results/infra/pose_temponet_proposal_20261117/proposal.json"))
fit, pc = j["fit"], j["positive_controls"]
chk = {
 "LRs": list(S.INITIAL_LRS) == fit["learning_rates"],
 "lower/upper": (S.LOWER_LR, S.UPPER_LR) == (fit["extensions"]["lower_lr"], fit["extensions"]["upper_lr"]),
 "epochs": (S.INITIAL_EPOCHS, S.MAX_EPOCHS) == (fit["initial_epochs"], fit["max_epochs"]),
 "lower tol": S.LOWER_EDGE_RELATIVE_TOLERANCE == fit["extensions"]["lower_edge_relative_mse_gain_threshold"],
 "recipe": (T.Recipe().batch_size, T.Recipe().clip_norm, T.Recipe().weight_decay, list(T.Recipe().betas), T.Recipe().eps)
           == (fit["batch_windows"], fit["clip_norm"], fit["weight_decay"], fit["betas"], fit["eps"]),
 "a grid": list(C.A_GRID) == pc["calibration"]["a_grid"],
 "sigma grid": list(C.SIGMA_GRID) == pc["calibration"]["sigma_grid"],
 "E tol": C.E_REAL_TOLERANCE == pc["calibration"]["real_E_dyn_abs_tolerance"],
 "gap": list(C.TEMPORAL_GAP) == pc["calibration"]["temporal_oracle_gain_range_fraction"],
 "lin ref/head": (C.LINEAR_REFERENCE_RECOVERY, C.LINEAR_HEAD_RECOVERY) == (pc["gates"]["linear"]["closed_form_diagnostic"]["min_fraction"], pc["gates"]["linear"]["min_fraction"]),
 "temporal head": C.TEMPORAL_HEAD_RECOVERY == pc["gates"]["temporal"]["min_fraction"],
 "alpha exps": list(range(-2, 5)) == pc["calibration"]["ridge"]["alpha_exponents_initial"],
 "alpha bounds": (10.**(-2-3), 10.**(4+3)) == tuple(pc["calibration"]["ridge"]["bounds"]),
 "S2": (5, 0.02) == (fit["objective_tradeoff"]["epoch_distance_gt"], fit["objective_tradeoff"]["relative_E_dyn_regret_percent_ge"]/100),
 "seeds": H.trajectory_seeds(1, 2) == {"initialization": 100001, "dropout": 200201, "shuffle": 300201},
 "fractions": fit["first_epoch"]["fractions"] == [0.25, 0.5, 0.75, 1.0],
 "count grid": len(C.A_GRID)*len(C.SIGMA_GRID) + len(C.SIGMA_GRID) == pc["calibration"]["combination_count_per_outer"],
 "max evals": (60+72)*(1+3+300) == fit["internal_fold_evaluations_max"],
}
print(json.dumps(chk, indent=0))
print("all:", all(chk.values()))
