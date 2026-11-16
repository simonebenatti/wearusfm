"""Offline, paired three-arm contrastive comparison. Never submits or trains.

Input OOF arrays use the existing tokenizer probe schema; pose cells use the
existing instantaneous ridge reader's per-user sufficient statistics. The CLI
requires SHA256-pinned files and all solver audit records. There is no scalar
score that mixes regression and classification.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness import downstream_ab as DA

PROTOCOL = "contrastive-comparison-20261116-v2"
ARMS = ("C0", "CN", "CR")
PAIRS = (("CN", "C0"), ("CR", "C0"), ("CN", "CR"))
TASKS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
WEIGHTS = {"p1": (8/13, 3/13, 2/13), "p2": (.5, .5)}
DATASETS = sum(TASKS.values(), ())
C_GRID = (.001, .01, .1, 1., 10., 100.)
N_BOOT, SEED = 100_000, 20261116
FAMILY_SIZE, ALPHA = 9, .05
SIMULTANEOUS_LEVEL = 1 - ALPHA / FAMILY_SIZE
PRIMARY_ENDPOINTS = ("E_dyn", "p1", "epn612")
DESCRIPTIVE_ENDPOINTS = ("p2", "uci_emg")
MARGINS = {"p1": {"gain": .5, "ni": 1.}, "epn612": {"gain": .5, "ni": 1.},
           "E_dyn": {"gain": 2., "ni": 2.}}
MC_BOUNDARY_SE = 3.0


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _labels(value, name):
    a = np.asarray(value)
    require(a.ndim == 1 and a.dtype.kind in "biufUS" and len(a), "invalid " + name)
    if a.dtype.kind in "biuf":
        require(np.isfinite(a).all(), "nonfinite " + name)
    return a


def validate_oof(a):
    require(set(a) == {"row", "y", "pred", "s", "fold"}, "OOF schema mismatch")
    a = {k: np.asarray(v) for k, v in a.items()}
    row = a["row"]
    require(row.ndim == 1 and row.dtype.kind in "iu" and len(row) == len(np.unique(row))
            and (row >= 0).all(), "duplicate/invalid OOF rows")
    for k in ("y", "pred", "s"):
        require(len(_labels(a[k], k)) == len(row), "OOF length mismatch")
    # Int/float label arrays are compatible; numeric/string arrays are not.
    # Comparing 1 with "1" would silently report every prediction as wrong.
    family = lambda x: "numeric" if x.dtype.kind in "biuf" else x.dtype.kind
    require(family(a["y"]) == family(a["pred"]), "OOF target/prediction label families differ")
    fold = a["fold"]
    require(fold.shape == row.shape and fold.dtype.kind in "iu" and np.isin(fold, range(4)).all(), "invalid OOF folds")
    for subject in np.unique(a["s"]):
        require(len(np.unique(fold[a["s"] == subject])) == 1, "subject split between held folds")
    order = np.argsort(row)
    return {k: v[order] for k, v in a.items()}


def intervals(samples):
    samples = np.asarray(samples, dtype=np.float64)
    require(samples.ndim == 1 and len(samples) >= 100 and np.isfinite(samples).all(),
            "invalid/nonfinite bootstrap draws; none discarded")
    q = ALPHA / (2 * FAMILY_SIZE)
    ordered = np.sort(samples)
    def tail_error(p):
        # Conditional Monte Carlo error of a percentile, from Binomial(n,p)
        # order-statistic uncertainty. No extra bootstrap or random seed.
        ranks = np.clip(binom.ppf([.15865525393145707, .8413447460685429], len(samples), p).astype(int), 0, len(samples)-1)
        one_sigma = ordered[ranks]
        return {"mcse": float((one_sigma[1]-one_sigma[0])/2),
                "one_sigma_order_statistic_range": one_sigma.tolist(), "ranks_zero_based": ranks.tolist()}
    return {"ci95": np.quantile(samples, [.025, .975]).tolist(),
            "ci_simultaneous": np.quantile(samples, [q, 1-q]).tolist(),
            "monte_carlo": {"method": "Binomial order-statistic one-sigma half-range; conditional on bootstrap distribution",
                            "boundary_standard_errors": MC_BOUNDARY_SE,
                            "lower": tail_error(q), "upper": tail_error(1-q)}}


def classification_statistics(cells, n_boot, seed):
    require(set(cells) == set(ARMS) and all(set(c) == set(DATASETS) for c in cells.values()), "incomplete classification grid")
    rng, points, samples, reports = np.random.default_rng(seed), {}, {}, {}
    for ds in sorted(DATASETS):
        aligned = [validate_oof(cells[a][ds]) for a in ARMS]
        base = aligned[0]
        require(all(all(np.array_equal(a[k], base[k]) for k in ("row", "y", "s", "fold")) for a in aligned), "OOF pairing differs")
        # Reuse the existing pooled subject/class sufficient statistics. This is
        # macro recall from pooled counts, NOT an average of subject accuracies.
        stats = [DA.subject_counts(a["y"], a["pred"], a["s"]) for a in aligned]
        users, denominator, _ = DA._arrays(stats[0])
        require(len(users) >= 2, "need at least two subjects")
        hits = [DA._arrays(s)[2] for s in stats]
        draws = rng.integers(0, len(users), size=(n_boot, len(users)))
        values = np.empty((n_boot, len(ARMS)), dtype=np.float64)
        missing, single = 0, 0
        # Bound temporary [replicate, subject, class] arrays for 100k draws.
        for start in range(0, n_boot, 256):
            chunk = draws[start:start+256]
            den = denominator[chunk].sum(1)
            present = den > 0
            nc = present.sum(1)
            require((nc > 0).all(), "empty target draw")
            missing += int((nc < denominator.shape[1]).sum())
            single += int((nc == 1).sum())
            for arm, correct in enumerate(hits):
                values[start:start+len(chunk), arm] = np.divide(correct[chunk].sum(1), den, out=np.zeros_like(den), where=present).sum(1)/nc
        den = denominator.sum(0)
        point = np.asarray([np.mean(correct.sum(0)[den > 0]/den[den > 0]) for correct in hits])
        points[ds], samples[ds] = point, values
        reports[ds] = {"balanced_accuracy": dict(zip(ARMS, point.tolist())), "subjects": users,
                       "paired_counts": dict(zip(ARMS, stats)), "draws_sha256": DA.fingerprint(draws),
                       "replicates_missing_classes": missing, "single_class_replicates": single,
                       "missing_class_rule": "macro recall over target classes present in each draw; all draws retained",
                       "role": "decision_endpoint" if ds == "epn612" else "descriptive_dataset"}
    task_points = {t: sum(w*points[d] for w, d in zip(WEIGHTS[t], TASKS[t])) for t in TASKS}
    task_samples = {t: sum(w*samples[d] for w, d in zip(WEIGHTS[t], TASKS[t])) for t in TASKS}
    # P2 remains the historical 50/50 aggregate. EPN612 has its own identity and
    # replaces P2 only in the three-endpoint decision family.
    for ds in ("epn612", "uci_emg"):
        task_points[ds], task_samples[ds] = points[ds], samples[ds]
    return task_points, task_samples, reports


def pose_statistics(cells, step, n_boot, seed):
    require(set(cells) == set(ARMS), "incomplete pose grid")
    ref = cells[ARMS[0]]["analysis"]
    rows = {}
    for arm, cell in cells.items():
        require(cell.get("complete") is True and cell.get("arm") == arm and cell.get("checkpoint_step") == step
                and cell.get("which") == "teacher", "incomplete/mixed-dose/nonteacher pose cell")
        analysis = cell["analysis"]
        require(all(analysis[k] == ref[k] for k in ("support_sha256", "targets_sha256", "dynamic_threshold")), "pose support/target/threshold differs")
        rows[arm] = analysis["groups"]["dynamic"]["centered"]["per_user"]
        require(analysis["groups"]["dynamic"]["n_selected_windows"] == ref["groups"]["dynamic"]["n_selected_windows"], "pose selected support differs")
    users = sorted(rows[ARMS[0]])
    require(len(users) >= 2 and all(sorted(r) == users for r in rows.values()), "unpaired pose users")
    for arm in ARMS:
        for user in users:
            require(all(rows[arm][user][k] == rows[ARMS[0]][user][k] for k in ("n_windows", "n_window_joints", "n_valid_elements")), "pose counts differ")
            require(rows[arm][user]["sums"]["vary"] == rows[ARMS[0]][user]["sums"]["vary"], "pose denominator differs")
    denominator = np.asarray([rows[ARMS[0]][u]["sums"]["vary"] for u in users], dtype=np.float64)
    errors = np.asarray([[rows[a][u]["sums"]["centered"] for a in ARMS] for u in users], dtype=np.float64)
    require(np.isfinite(denominator).all() and (denominator >= 0).all()
            and np.isfinite(errors).all() and (errors >= 0).all(), "invalid pose sums")
    draws = np.random.default_rng(seed).integers(0, len(users), (n_boot, len(users)))
    den = denominator[draws].sum(1)
    require((den > 0).all(), "zero pose denominator; no draws discarded")
    values = errors[draws].sum(1)/den[:, None]
    require((values > 0).all(), "zero pose error prevents relative comparison")
    point = errors.sum(0)/denominator.sum()
    report = {"users": users, "E_dyn": dict(zip(ARMS, point.tolist())), "draws_sha256": DA.fingerprint(draws),
              "per_user": rows, "support_sha256": ref["support_sha256"], "targets_sha256": ref["targets_sha256"]}
    return point, values, report


def verify_audits(records):
    """Recompute convergence and exact 3×5×4×7 fit coverage; no Boolean-only gate."""
    require(isinstance(records, list) and len(records) == 3*5*4*7, "incomplete/extra probe fit audit")
    unresolved = []
    for arm in ARMS:
        for ds in DATASETS:
            for fold in range(4):
                cell = [r for r in records if r.get("arm") == arm and r.get("dataset") == ds and r.get("fold") == fold]
                candidates = [r for r in cell if r.get("fit_phase") == "candidate"]
                refits = [r for r in cell if r.get("fit_phase") == "refit"]
                require(len(cell) == 7 and sorted(r.get("C") for r in candidates) == list(C_GRID)
                        and len(refits) == 1 and refits[0]["C"] in C_GRID, "probe candidate/refit grid differs")
                for r in cell:
                    require(r.get("exception") is None and r.get("max_iter") == 12000
                            and type(r.get("n_iter")) is int and 0 < r["n_iter"] <= 12000
                            and type(r.get("convergence_warnings")) is int and r["convergence_warnings"] >= 0, "invalid/failed fit record")
                    converged = r["n_iter"] < 12000 and r["convergence_warnings"] == 0
                    require(r.get("converged_by_criterion") is converged, "fit convergence flag differs from evidence")
                    if not converged:
                        unresolved.append({k: r[k] for k in ("arm", "dataset", "fold", "fit_phase", "C", "n_iter")})
    return {"complete": True, "fits": len(records), "converged": not unresolved, "unresolved": unresolved}


def summarize_pair(candidate, control, points, samples):
    i, j = ARMS.index(candidate), ARMS.index(control)
    out = {}
    for endpoint in PRIMARY_ENDPOINTS:
        if endpoint == "E_dyn":
            point = 100*(1-points[endpoint][i]/points[endpoint][j])
            draw = 100*(1-samples[endpoint][:, i]/samples[endpoint][:, j])
            unit = "relative_error_reduction_percent"
        else:
            point = 100*(points[endpoint][i]-points[endpoint][j])
            draw = 100*(samples[endpoint][:, i]-samples[endpoint][:, j])
            unit = "percentage_points"
        ci = intervals(draw)
        lo, hi = ci["ci_simultaneous"]
        gain, ni = MARGINS[endpoint]["gain"], MARGINS[endpoint]["ni"]
        lower_error, upper_error = ci["monte_carlo"]["lower"]["mcse"], ci["monte_carlo"]["upper"]["mcse"]
        boundaries = {"superiority_zero": abs(lo) <= MC_BOUNDARY_SE*lower_error,
                      "noninferiority": abs(lo+ni) <= MC_BOUNDARY_SE*lower_error,
                      "material_harm": abs(hi+ni) <= MC_BOUNDARY_SE*upper_error,
                      "equivalence_upper": abs(hi-gain) <= MC_BOUNDARY_SE*upper_error}
        out[endpoint] = {"benefit": float(point), "unit": unit, **ci,
                         "boundary": boundaries,
                         "useful_superiority": bool(point >= gain and lo > 0 and not boundaries["superiority_zero"]),
                         "noninferiority": bool(lo > -ni and not boundaries["noninferiority"]),
                         "demonstrated_material_harm": bool(hi < -ni and not boundaries["material_harm"]),
                         "practical_equivalence": bool(lo > -ni and hi < gain and not boundaries["noninferiority"] and not boundaries["equivalence_upper"])}
    out["useful_without_material_loss"] = bool(any(out[e]["useful_superiority"] for e in MARGINS)
                                                 and all(out[e]["noninferiority"] for e in MARGINS))
    out["descriptive"] = {
        e: {"benefit": float(100*(points[e][i]-points[e][j])), "unit": "percentage_points",
            "ci95": np.quantile(100*(samples[e][:,i]-samples[e][:,j]), [.025,.975]).tolist(),
            "role": "descriptive_only_not_in_decision_family"} for e in DESCRIPTIVE_ENDPOINTS
    }
    out["profile"] = pair_profile(out)
    return out


def pair_profile(pair):
    benefits = [e for e in PRIMARY_ENDPOINTS if pair[e]["useful_superiority"]]
    harms = [e for e in PRIMARY_ENDPOINTS if pair[e]["demonstrated_material_harm"]]
    unresolved_ni = [e for e in PRIMARY_ENDPOINTS if not pair[e]["noninferiority"] and e not in harms]
    equivalent = all(pair[e]["practical_equivalence"] for e in PRIMARY_ENDPOINTS)
    if benefits and harms:
        label = "tradeoff_benefit_and_demonstrated_material_harm"
    elif harms:
        label = "demonstrated_material_harm"
    elif benefits and unresolved_ni:
        label = "benefit_with_unresolved_noninferiority"
    elif benefits:
        label = "benefit_without_material_loss_on_decision_endpoints"
    elif equivalent:
        label = "practically_equivalent_on_decision_endpoints"
    else:
        label = "unresolved"
    return {"label": label, "useful_endpoints": benefits, "material_harm_endpoints": harms,
            "unresolved_noninferiority_endpoints": unresolved_ni,
            "all_endpoints_practically_equivalent": equivalent,
            "no_practical_advantage_supported": equivalent}


def scientific_reading(contrasts, reverse):
    cn = contrasts["CN_minus_C0"]["useful_without_material_loss"]
    cr = contrasts["CR_minus_C0"]["useful_without_material_loss"]
    cn_direct = contrasts["CN_minus_CR"]["useful_without_material_loss"]
    cr_direct = reverse["useful_without_material_loss"]
    # Both directions can meet gain + NI on different classification endpoints.
    # Evaluate both before selecting any recipe for a confirmation proposal.
    if cn_direct and cr_direct:
        return "bidirectional_benefit_tradeoff_no_winner"
    if cn and cn_direct:
        return "CN_supported_for_confirmation"
    if cr and cr_direct:
        return "CR_supported_for_confirmation"
    profiles = [p["profile"] for p in [*contrasts.values(), reverse]]
    if any(p["material_harm_endpoints"] and p["useful_endpoints"] for p in profiles):
        return "tradeoff_with_demonstrated_material_harm"
    if cn and cr:
        direct_profiles = [contrasts["CN_minus_CR"]["profile"], reverse["profile"]]
        if not any(p["useful_endpoints"] or p["material_harm_endpoints"] for p in direct_profiles):
            return "common_benefit_without_resolved_recipe_difference"
        return "common_benefit_with_unresolved_recipe_preference"
    if all(p["all_endpoints_practically_equivalent"] for p in profiles):
        return "practically_equivalent_on_decision_endpoints"
    return "unresolved_or_tradeoff"


def compare(classification, pose, *, audit_records=None, n_boot=N_BOOT, seed=SEED, step=19618):
    require(type(n_boot) is int and n_boot >= 100 and type(seed) is int, "invalid bootstrap settings")
    require(step == 19618, "final three-endpoint comparison is only at 19618")
    cp, cs, datasets = classification_statistics(classification, n_boot, seed)
    pp, ps, pose_report = pose_statistics(pose, step, n_boot, seed)
    points, samples = {"E_dyn": pp, **cp}, {"E_dyn": ps, **cs}
    audits = verify_audits(audit_records) if audit_records is not None else {"complete": False, "converged": False}
    if audit_records is not None:
        for arm in ARMS:
            checkpoint = pose[arm].get("checkpoint_sha256")
            require(isinstance(checkpoint, str) and len(checkpoint) == 64
                    and all(c in "0123456789abcdef" for c in checkpoint), "pose checkpoint pin missing")
            require(all(r.get("checkpoint_sha256") == checkpoint for r in audit_records if r["arm"] == arm),
                    "classification/pose checkpoint identity differs")
    ridge_ok = all(c.get("fit", {}).get("selection", {}).get("boundary_unresolved") is False for c in pose.values())
    valid = bool(audits["complete"] and audits["converged"] and ridge_ok and n_boot == N_BOOT and seed == SEED)
    contrasts = {a+"_minus_"+b: summarize_pair(a, b, points, samples) for a, b in PAIRS}
    reverse = summarize_pair("CR", "CN", points, samples)
    # Reverse intervals are monotone transformations of the same head-to-head
    # comparison, not three extra hypotheses. No weighted score or forced winner.
    reading = scientific_reading(contrasts, reverse) if valid else "interpretation_withheld"
    if not valid:
        for pair in [*contrasts.values(), reverse]:
            pair["useful_without_material_loss"] = None
            pair["profile"] = {"label": "interpretation_withheld"}
            for endpoint in MARGINS:
                pair[endpoint]["useful_superiority"] = pair[endpoint]["noninferiority"] = None
                pair[endpoint]["demonstrated_material_harm"] = pair[endpoint]["practical_equivalence"] = None
    return {"protocol": PROTOCOL, "complete": True, "checkpoint_step": step, "which": "teacher",
            "arms": list(ARMS), "datasets": datasets, "pose": pose_report, "contrasts": contrasts,
            "reverse_head_to_head": reverse, "decision_valid": valid, "reading": reading,
            "decision_endpoints": list(PRIMARY_ENDPOINTS), "descriptive_endpoints": list(DESCRIPTIVE_ENDPOINTS),
            "historical_P2": {"definition": "0.5 EPN612 + 0.5 UCI", "role": "descriptive; unchanged historical name and composition"},
            "practical_equivalence_ranges": {e: [-MARGINS[e]["ni"], MARGINS[e]["gain"]] for e in PRIMARY_ENDPOINTS},
            "fit_audit": audits, "ridge_selection_resolved": ridge_ok,
            "bootstrap": {"replicates": n_boot, "seed": seed, "family_size": FAMILY_SIZE,
                          "family_alpha": ALPHA, "nominal_simultaneous_confidence": SIMULTANEOUS_LEVEL,
                          "coverage_caveat": "Nominal Bonferroni-adjusted percentile intervals; finite-sample joint coverage is not guaranteed or measured here.",
                          "monte_carlo_boundary_rule": "A limit within three estimated Monte Carlo standard errors of its threshold is at_boundary and does not support that decision flag.",
                          "unit": "paired subjects within dataset; paired pose users", "all_draws_retained": True},
            "uncertainty": "Conditional subject bootstrap, one pretraining seed per arm, development data already observed. Does not estimate between-training variance or prove general FM utility.",
            "scope": "Endpoint-scoped recipe screening for a confirmation proposal only; no universal winner and no automatic adoption, extension, or submission. Descriptive P2/UCI cannot establish broad classification noninferiority."}


def read_pin(pin, base):
    require(isinstance(pin, dict) and set(pin) == {"path", "sha256"}, "need exact path/SHA256 pin")
    path = Path(pin["path"])
    path = path if path.is_absolute() else base / path
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin["sha256"], "input hash mismatch: " + str(path))
    return path, raw


def from_manifest(path):
    path = Path(path)
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get("protocol") == PROTOCOL and manifest.get("step") == 19618
            and manifest.get("which") == "teacher", "manifest protocol/dose/teacher mismatch")
    require(set(manifest.get("pose", {})) == set(ARMS) and set(manifest.get("classification", {})) == set(ARMS), "manifest grid incomplete")
    pose, classification, audits = {}, {}, []
    for arm in ARMS:
        pp, raw = read_pin(manifest["pose"][arm], path.parent)
        pose[arm] = json.loads(gzip.decompress(raw) if pp.suffix == ".gz" else raw)
        classification[arm] = {}
        require(set(manifest["classification"][arm]) == set(TASKS), "manifest task grid incomplete")
        for task in TASKS:
            cell = manifest["classification"][arm][task]
            require(set(cell) == {"oof", "audit"}, "classification cell schema mismatch")
            _, raw = read_pin(cell["oof"], path.parent)
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                for ds in TASKS[task]:
                    classification[arm][ds] = {k: arrays[f"{ds}_{k}"].copy() for k in ("row", "y", "pred", "s", "fold")}
            _, raw = read_pin(cell["audit"], path.parent)
            rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            require(all(r.get("arm") == arm and r.get("task") == task and r.get("dataset") in TASKS[task] for r in rows), "audit identity differs")
            audits.extend(rows)
    result = compare(classification, pose, audit_records=audits)
    result["input_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    result["input_pins"] = manifest
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "refuse to overwrite a previous comparison")
    result = from_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
