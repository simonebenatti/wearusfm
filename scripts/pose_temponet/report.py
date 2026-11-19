"""Prespecified paired readout contrasts; never select a model on development data."""
from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np
import pose_dynamics as historical
from wearusfm.ingest.emg2pose_pose import JOINTS

from .heads import HEADS
from .metrics import THRESHOLD

N_BOOT, BOOTSTRAP_SEED = 2000, 20261102
SEEDS = (0, 1, 2)
GROUPS = ("all", "dynamic", "static")
REFERENCES = ("historical_post_ridge", "historical_trained_head")
COMPARATORS = ("L0", "MLP", "L13", *REFERENCES)
CANDIDATE_E_DYN_MAX = 0.7630564046643697
COMMON_GATES = ("integrity", "historical_R1", "resources", "positive_controls")
BENEFIT = "beneficio_pratico_risolto"
HARM = "deterioramento_pratico"
UNSUPPORTED = "beneficio_2pct_non_sostenuto"
INCONCLUSIVE = "inconclusivo"
COUNTS = ("n_windows", "n_window_joints", "n_valid_elements")
TARGET_FIELDS = ("vary", "baseline_total", "mean_target_squared")
METRICS = ("E_dyn", "level_mse", "total_mse", "predicted_variance", "target_variance",
           "variance_ratio", "correlation", "mae_rad", "mae_degrees")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def outcome(gain, lower, upper):
    """First true condition; do not reuse the long-dose practical_assessment."""
    require(all(math.isfinite(float(x)) for x in (gain, lower, upper)) and lower <= upper,
            "invalid effect/interval")
    if gain >= 2 and lower > 0:
        return BENEFIT
    if gain <= -2 and upper < 0:
        return HARM
    if upper < 2:
        return UNSUPPORTED
    return INCONCLUSIVE


def monte_carlo_border(interval):
    return interval is not None and any(abs(float(x)-threshold) <= .1
                                       for x in interval for threshold in (0., 2.))


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _interval(point, samples, confidence):
    samples = np.asarray(samples, dtype=np.float64)
    bad = int((~np.isfinite(samples)).sum())
    valid = np.isfinite(point) and bad == 0
    q = (100-confidence)/2
    return {"point":_finite(point), "ci":np.percentile(samples,[q,100-q],method="linear").tolist() if valid else None,
            "confidence_percent":confidence,"evaluable":bool(valid),"undefined_replicates":bad,
            "replicates_discarded":0}


def _moments(rows, users):
    require(set(rows) <= set(users), "unexpected subject in paired rows")
    result = {key:np.zeros(len(users),dtype=np.float64) for key in (*historical.CENTERED_FIELDS,*COUNTS)}
    for i,user in enumerate(users):
        if user not in rows:
            continue
        row = rows[user]
        for key in COUNTS:
            value = row[key]
            require(type(value) is int and value >= 0, "invalid support count")
            result[key][i] = value
        for key in historical.CENTERED_FIELDS:
            value = float(row["sums"][key])
            require(np.isfinite(value), "nonfinite per-user accumulator")
            if key in ("centered","vary","varp","total","mean_error"):
                require(value >= 0, "negative error/variance accumulator")
            result[key][i] = value
    return result


def _arrays(group, users):
    result = _moments(group["centered"]["per_user"],users)
    maes = group["mae_per_user"]
    require(set(maes) <= set(users), "unexpected subject in MAE")
    result.update(mae_error=np.zeros(len(users)),mae_count=np.zeros(len(users)))
    for i,user in enumerate(users):
        if user not in maes:
            continue
        row = maes[user]
        require(type(row["n_valid_elements"]) is int and row["n_valid_elements"] >= 0 and
                np.isfinite(row["error_sum_rad"]) and row["error_sum_rad"] >= 0,
                "invalid MAE accumulator")
        result["mae_error"][i] = row["error_sum_rad"]
        result["mae_count"][i] = row["n_valid_elements"]
    return result


def _paired(candidate, control):
    for key in (*COUNTS,*TARGET_FIELDS,"mae_count"):
        require(np.array_equal(candidate[key],control[key]), "paired target/support differs: "+key)


def _mean_seed_arrays(arrays):
    for other in arrays[1:]:
        _paired(arrays[0],other)
    # Counts and target moments must remain the identical source arrays. Averaging
    # them is unnecessary and could introduce a floating-point pairing mismatch.
    fixed = set(COUNTS) | set(TARGET_FIELDS) | {"mae_count"}
    return {key:(arrays[0][key].copy() if key in fixed else np.mean([a[key] for a in arrays],axis=0))
            for key in arrays[0]}


def _metric_values(sums):
    with np.errstate(divide="ignore",invalid="ignore"):
        result = {"E_dyn":sums["centered"]/sums["vary"],
                  "level_mse":sums["mean_error"]/sums["n_valid_elements"],
                  "total_mse":sums["total"]/sums["n_valid_elements"],
                  "predicted_variance":sums["varp"]/sums["n_valid_elements"],
                  "target_variance":sums["vary"]/sums["n_valid_elements"],
                  "variance_ratio":sums["varp"]/sums["vary"],
                  "correlation":sums["cov"]/np.sqrt(sums["varp"]*sums["vary"]),
                  "mae_rad":sums["mae_error"]/sums["mae_count"]}
        result["mae_degrees"] = result["mae_rad"]*180/np.pi
    return result


def _points(arrays):
    return _metric_values({key:values.sum(dtype=np.float64) for key,values in arrays.items()})


def _draw_metrics(arrays, draws):
    return _metric_values({key:values[draws].sum(axis=1,dtype=np.float64) for key,values in arrays.items()})


def _gain(candidate,control):
    with np.errstate(divide="ignore",invalid="ignore"):
        return 100*(control-candidate)/control


def _relative(candidate,control,draws,confidence):
    _paired(candidate,control)
    cp,rp = _points(candidate),_points(control)
    cs,rs = _draw_metrics(candidate,draws),_draw_metrics(control,draws)
    return _interval(_gain(cp["E_dyn"],rp["E_dyn"]),
                     _gain(cs["E_dyn"],rs["E_dyn"]),confidence)


def _difference(candidate,control,draws,metric):
    _paired(candidate,control)
    cp,rp = _points(candidate),_points(control)
    cs,rs = _draw_metrics(candidate,draws),_draw_metrics(control,draws)
    interval = _interval(cp[metric]-rp[metric],cs[metric]-rs[metric],95)
    interval.update(positive_means="candidate worse",estimated_worsening=interval["point"] is not None and interval["point"] > 0,
                    resolved_worsening=interval["ci"] is not None and interval["ci"][0] > 0)
    return interval


def _assessment(value):
    alarms = value["optimizer_alarms"]
    require(isinstance(alarms,list) and all(isinstance(x,str) and x for x in alarms), "invalid optimizer alarms")
    flag = value["objective_tradeoff"]["flag"]
    require(type(flag) is bool, "missing explicit S-2 flag")
    if "optimization_unresolved" in value:
        require(value["optimization_unresolved"] is bool(alarms), "optimizer alarm summary differs")
    return {"optimizer_alarms":alarms.copy(),"optimization_unresolved":bool(alarms),"objective_tradeoff":flag}


def directional_reading(label, candidate_alarm, comparator_alarm):
    """Optimization uncertainty is directional; statistical values remain visible."""
    permitted = label is not None
    reasons = []
    if label == BENEFIT and comparator_alarm:
        permitted = False
        reasons.append("comparator optimization unresolved: no pro-candidate benefit claim")
    if label != BENEFIT and candidate_alarm:
        permitted = False
        reasons.append("candidate optimization unresolved: null/negative conclusion suspended")
    return {"scientific_label":label if permitted else None,"statistical_label":label,
            "interpretation_suspended":not permitted,"reasons":reasons,
            "may_use_absence_of_practical_veto":not comparator_alarm}


def candidacy(contrasts, assessments, common_gates, *, candidate_e_dyn):
    common = all(common_gates.get(k) is True for k in COMMON_GATES)
    comparators = all(not assessments[h]["optimization_unresolved"] for h in ("L0","MLP","L13"))
    required = all(contrasts[str(i)]["statistical_label"] == BENEFIT for i in (1,4,5))
    vetoes = [i for i in (2,3) if contrasts[str(i)]["statistical_label"] == HARM]
    all_evaluable = all(contrasts[str(i)]["relative_E_dyn"]["evaluable"] for i in range(1,6))
    absolute_value = _finite(candidate_e_dyn) if candidate_e_dyn is not None else None
    absolute_passed = absolute_value is not None and absolute_value <= CANDIDATE_E_DYN_MAX
    candidate = common and comparators and required and not vetoes and all_evaluable and absolute_passed
    mixed = [COMPARATORS[i-1] for i in (2,3)
             if contrasts[str(i)]["relative_E_dyn"]["ci"] is not None and
             contrasts[str(i)]["relative_E_dyn"]["ci"][1] < 0 and
             contrasts[str(i)]["relative_E_dyn"]["point"] > -2]
    tradeoffs = [h for h in ("L0","MLP","L13") if assessments[h]["objective_tradeoff"]]
    qualifiers = []
    name = "candidata alla lettura della dinamica centrata" if candidate else "non candidata secondo la regola prespecificata"
    if candidate and mixed:
        name = "candidata, con "+"/".join(mixed)+" risolti migliori ma stima sotto il 2%"
        qualifiers.append("Caso misto; non è un'affermazione di non inferiorità")
    if candidate and tradeoffs:
        qualifiers.append("sotto la selezione MSE comune, con compromesso S-2 sul comparatore: "+", ".join(tradeoffs))
    if candidate and assessments["TCN_post"]["optimization_unresolved"]:
        qualifiers.append("Allarme TCN-post: beneficio misurato conservato; letture nulle/negative sospese")
    return {"candidate":candidate,"name":name,"qualifiers":qualifiers,
            "absolute_E_dyn":{"value":absolute_value,"maximum":CANDIDATE_E_DYN_MAX,"passed":absolute_passed},
            "required_benefit_contrasts_passed":required,"comparators_admitted":comparators,
            "practical_veto_contrasts":vetoes,"common_gates_passed":common,"all_contrasts_evaluable":all_evaluable,
            "mixed_better_comparators":mixed,"S2_comparators":tradeoffs,
            "noncandidacy_is_not_a_negative_scientific_verdict":True,
            "scope":"development centered-dynamics readout; no adoption, pretraining claim or level B"}


def _seed_summary(arrays):
    values = [_points(a) for a in arrays]
    output = {"per_seed":{str(s):{k:_finite(v) for k,v in p.items()} for s,p in zip(SEEDS,values)},
              "mean_accumulators_metrics":{k:_finite(v) for k,v in _points(_mean_seed_arrays(arrays)).items()}}
    output["arithmetic_mean_seed_metrics"] = {}
    output["sample_sd_seed_metrics"] = {}
    for metric in METRICS:
        points = np.asarray([p[metric] for p in values])
        finite = np.isfinite(points).all()
        output["arithmetic_mean_seed_metrics"][metric] = float(np.mean(points)) if finite else None
        output["sample_sd_seed_metrics"][metric] = float(np.std(points,ddof=1)) if finite else None
    return output


def _user_accumulators(arrays, users):
    """Compact sufficient statistics for independent reproduction of paired CIs."""
    return {u:{key:(int(value[i]) if key in (*COUNTS,"mae_count") else float(value[i]))
               for key,value in arrays.items()} for i,u in enumerate(users)}


def _cell_analysis(cell):
    require(cell.get("complete") is True, "incomplete readout cell")
    analysis = cell["analysis"]
    require(analysis["dynamic_threshold"] == THRESHOLD, "external dynamic threshold changed")
    require(set(analysis["groups"]) == set(GROUPS), "missing prespecified group")
    return analysis


def _aggregate(cells, references, assessments, common_gates):
    require(set(cells) == set(HEADS) and set(references) == set(REFERENCES), "readout matrix differs")
    require(set(assessments) == set(HEADS), "missing readout optimization assessment")
    assessments = {h:_assessment(assessments[h]) for h in HEADS}
    analyses = {}
    for head,values in cells.items():
        require(isinstance(values,Mapping) and len(values) == 3 and
                all(type(s) in (int,str) and s in (*SEEDS,"0","1","2") for s in values), "expected all three head seeds")
        normalized = {int(s):cell for s,cell in values.items()}
        require(set(normalized) == set(SEEDS), "seed identities are not unique and complete")
        analyses[head] = [_cell_analysis(normalized[s]) for s in SEEDS]
    refs = {name:_cell_analysis(cell) for name,cell in references.items()}
    base = analyses["L0"][0]
    users = sorted(base["groups"]["dynamic"]["centered"]["per_user"])
    require(len(users) == 15, "expected the 15 prespecified development users")
    for analysis in [a for values in analyses.values() for a in values]+list(refs.values()):
        require(analysis["support_sha256"] == base["support_sha256"], "paired crop support hash differs")
        require(sorted(analysis["groups"]["dynamic"]["centered"]["per_user"]) == users,
                "paired dynamic users differ; no intersections")
    grouped = {h:{g:[_arrays(a["groups"][g],users) for a in analyses[h]] for g in GROUPS} for h in HEADS}
    averaged = {h:{g:_mean_seed_arrays(grouped[h][g]) for g in GROUPS} for h in HEADS}
    averaged.update({h:{g:_arrays(a["groups"][g],users) for g in GROUPS} for h,a in refs.items()})
    for group in GROUPS:
        for value in averaged.values():
            _paired(averaged["L0"][group],value[group])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(users),(N_BOOT,len(users)))
    summaries = {h:{g:_seed_summary(grouped[h][g]) for g in GROUPS} for h in HEADS}
    joint_summaries = {}
    for values in analyses.values():
        for analysis in values:
            for group in GROUPS:
                require(set(analysis["groups"][group].get("by_joint",{})) == set(JOINTS),
                        "missing all-joint final descriptive metrics")
    for h in HEADS:
        joint_summaries[h] = {}
        for group in GROUPS:
            joint_summaries[h][group] = {}
            for joint in JOINTS:
                values = [_arrays(a["groups"][group]["by_joint"][joint],users) for a in analyses[h]]
                baseline = _arrays(analyses["L0"][0]["groups"][group]["by_joint"][joint],users)
                for value in values:
                    _paired(baseline,value)
                joint_summaries[h][group][joint] = _seed_summary(values)
    contrasts = {}
    for i,comparator in enumerate(COMPARATORS,1):
        candidate = averaged["TCN_post"]["dynamic"]
        control = averaged[comparator]["dynamic"]
        effect = _relative(candidate,control,draws,99)
        label = outcome(effect["point"],*effect["ci"]) if effect["evaluable"] else None
        comparator_alarm = assessments[comparator]["optimization_unresolved"] if comparator in assessments else False
        reading = directional_reading(label,assessments["TCN_post"]["optimization_unresolved"],comparator_alarm)
        per_seed = {}
        for seed in SEEDS:
            candidate_s = grouped["TCN_post"]["dynamic"][seed]
            control_s = grouped[comparator]["dynamic"][seed] if comparator in grouped else control
            per_seed[str(seed)] = _relative(candidate_s,control_s,draws,99)
        gains = [per_seed[str(s)]["point"] for s in SEEDS]
        secondary = {}
        for group in GROUPS:
            ca,co = averaged["TCN_post"][group],averaged[comparator][group]
            secondary[group] = {"relative_E_dyn_ci95":_relative(ca,co,draws,95),
                                "MAE_delta_ci95":_difference(ca,co,draws,"mae_rad"),
                                "level_delta_ci95":_difference(ca,co,draws,"level_mse")}
        contrasts[str(i)] = {"candidate":"TCN_post","comparator":comparator,"primary":i == 1,
                             "relative_E_dyn":effect,**reading,"al_bordo_MC":monte_carlo_border(effect["ci"]),
                             "per_seed_same_seed_contrasts":per_seed,
                             "sample_sd_seed_gains":float(np.std(gains,ddof=1)) if all(g is not None for g in gains) else None,
                             "secondary_descriptive":secondary}
        if comparator in grouped:
            contrasts[str(i)]["cross_seed_points_descriptive_only"] = {
                f"{a}/{b}":_finite(_gain(_points(grouped["TCN_post"]["dynamic"][a])["E_dyn"],
                                         _points(grouped[comparator]["dynamic"][b])["E_dyn"]))
                for a in SEEDS for b in SEEDS}
    descriptive = {}
    for ca,co in (("L0",REFERENCES[0]),("L0",REFERENCES[1]),("MLP","L0"),("L13","L0"),("TCN_post","TCN_pre")):
        effect = _relative(averaged[ca]["dynamic"],averaged[co]["dynamic"],draws,95)
        ca_alarm = assessments[ca]["optimization_unresolved"]
        co_alarm = assessments[co]["optimization_unresolved"] if co in assessments else False
        direction = ("candidate" if effect["ci"] and effect["ci"][0] > 0 else
                     "comparator" if effect["ci"] and effect["ci"][1] < 0 else "unresolved")
        suspended = ((direction == "candidate" and co_alarm) or
                     (direction == "comparator" and ca_alarm) or
                     (direction == "unresolved" and (ca_alarm or co_alarm)))
        descriptive[ca+"_vs_"+co] = {"relative_E_dyn":effect,"resolved_direction":direction,
                                    "interpretation_suspended":suspended,
                                    "MAE_delta_ci95":_difference(averaged[ca]["dynamic"],averaged[co]["dynamic"],draws,"mae_rad"),
                                    "level_delta_ci95":_difference(averaged[ca]["dynamic"],averaged[co]["dynamic"],draws,"level_mse")}
    return {"complete":True,"status":"descriptive_development_results","common_gates":dict(common_gates),
            "users":users,"head_seeds":list(SEEDS),"assessments":assessments,
            "bootstrap":{"replicates":N_BOOT,"seed":BOOTSTRAP_SEED,"unit":"user","same_draws_everywhere":True,
                         "draws_sha256":hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest(),
                         "family_confidence_percent":95,"individual_confidence_percent":99,
                         "coverage":"nominal Bonferroni approximation; not guaranteed with 15 users",
                         "hierarchical_seed_bootstrap":False},
            "seed_aggregation":"mean per-user moment accumulators before metric/bootstrap; not averaged predictions",
            "heads":summaries,"per_joint":joint_summaries,
            "paired_user_accumulators":{h:{g:_user_accumulators(a,users) for g,a in groups.items()}
                                        for h,groups in averaged.items()},
            "references":{h:{g:{k:_finite(v) for k,v in _points(a).items()} for g,a in value.items()}
                          for h,value in averaged.items() if h in REFERENCES},
            "contrasts":contrasts,"descriptive":descriptive,
            "candidacy":candidacy(contrasts,assessments,common_gates,
                                  candidate_e_dyn=_points(averaged["TCN_post"]["dynamic"])["E_dyn"]),
            "historical_pre_ridge":{"E_dyn":1.0321280032044744,"paired_CI":None,"summary_anchor_only":True,
                                    "TCN_pre_relative_gain_percent":_finite(_gain(
                                        _points(averaged["TCN_pre"]["dynamic"])["E_dyn"],1.0321280032044744)),
                                    "per_seed_relative_gain_percent":{str(s):_finite(_gain(
                                        _points(grouped["TCN_pre"]["dynamic"][s])["E_dyn"],1.0321280032044744))
                                        for s in SEEDS}},
            "limits":["one pretrained encoder; development users previously consulted",
                      "head seed SD is separate from paired-user intervals",
                      "no automatic adoption, FM claim, new experiment or level B"]}


def aggregate(cells, references, *, assessments, common_gates):
    """Fail closed on missing cells/quality gates; preserve all five valid contrasts."""
    if not isinstance(common_gates,Mapping) or any(common_gates.get(k) is not True for k in COMMON_GATES):
        return {"complete":False,"status":"quality_gate_not_passed","contrasts":{},
                "common_gates":dict(common_gates) if isinstance(common_gates,Mapping) else None,
                "candidacy":{"candidate":False},"scientific_verdict":None}
    try:
        return _aggregate(cells,references,assessments,common_gates)
    except (ValueError,TypeError,KeyError,IndexError) as error:
        return {"complete":False,"status":"technical_failure","reason":str(error),"contrasts":{},
                "candidacy":{"candidate":False},"scientific_verdict":None}
