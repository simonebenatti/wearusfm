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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness import downstream_ab as DA

PROTOCOL = "contrastive-comparison-20261116-v1"
ARMS = ("C0", "CN", "CR")
PAIRS = (("CN", "C0"), ("CR", "C0"), ("CN", "CR"))
TASKS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
WEIGHTS = {"p1": (8/13, 3/13, 2/13), "p2": (.5, .5)}
DATASETS = sum(TASKS.values(), ())
C_GRID = (.001, .01, .1, 1., 10., 100.)
N_BOOT, SEED = 100_000, 20261116
FAMILY_SIZE, ALPHA = 9, .05
SIMULTANEOUS_LEVEL = 1 - ALPHA / FAMILY_SIZE
MARGINS = {"p1": {"gain": .5, "ni": 1.}, "p2": {"gain": .5, "ni": 1.},
           "E_dyn": {"gain": 2., "ni": 2.}}


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
    fold = a["fold"]
    require(fold.shape == row.shape and fold.dtype.kind in "iu" and np.isin(fold, range(4)).all(), "invalid OOF folds")
    for subject in np.unique(a["s"]):
        require(len(np.unique(fold[a["s"] == subject])) == 1, "subject split between held folds")
    order = np.argsort(row)
    return {k: v[order] for k, v in a.items()}


def intervals(samples):
    require(np.isfinite(samples).all(), "nonfinite bootstrap draws; none discarded")
    q = ALPHA / (2 * FAMILY_SIZE)
    return {"ci95": np.quantile(samples, [.025, .975]).tolist(),
            "ci_simultaneous": np.quantile(samples, [q, 1-q]).tolist()}


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
                       "role": "descriptive_dataset_not_an_additional_primary"}
    task_points = {t: sum(w*points[d] for w, d in zip(WEIGHTS[t], TASKS[t])) for t in TASKS}
    task_samples = {t: sum(w*samples[d] for w, d in zip(WEIGHTS[t], TASKS[t])) for t in TASKS}
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
    for endpoint in ("E_dyn", "p1", "p2"):
        if endpoint == "E_dyn":
            point = 100*(1-points[endpoint][i]/points[endpoint][j])
            draw = 100*(1-samples[endpoint][:, i]/samples[endpoint][:, j])
            unit = "relative_error_reduction_percent"
        else:
            point = 100*(points[endpoint][i]-points[endpoint][j])
            draw = 100*(samples[endpoint][:, i]-samples[endpoint][:, j])
            unit = "percentage_points"
        ci = intervals(draw)
        out[endpoint] = {"benefit": float(point), "unit": unit, **ci,
                         "useful_superiority": bool(point >= MARGINS[endpoint]["gain"] and ci["ci_simultaneous"][0] > 0),
                         "noninferiority": bool(ci["ci_simultaneous"][0] > -MARGINS[endpoint]["ni"])}
    out["useful_without_material_loss"] = bool(any(out[e]["useful_superiority"] for e in MARGINS)
                                                 and all(out[e]["noninferiority"] for e in MARGINS))
    return out


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
    winner = "unresolved_or_tradeoff"
    if valid:
        if contrasts["CN_minus_CR"]["useful_without_material_loss"] and contrasts["CN_minus_C0"]["useful_without_material_loss"]:
            winner = "CN_supported_for_confirmation"
        elif reverse["useful_without_material_loss"] and contrasts["CR_minus_C0"]["useful_without_material_loss"]:
            winner = "CR_supported_for_confirmation"
    if not valid:
        winner = "interpretation_withheld"
        for pair in [*contrasts.values(), reverse]:
            pair["useful_without_material_loss"] = None
            for endpoint in MARGINS:
                pair[endpoint]["useful_superiority"] = pair[endpoint]["noninferiority"] = None
    return {"protocol": PROTOCOL, "complete": True, "checkpoint_step": step, "which": "teacher",
            "arms": list(ARMS), "datasets": datasets, "pose": pose_report, "contrasts": contrasts,
            "reverse_head_to_head": reverse, "decision_valid": valid, "reading": winner,
            "fit_audit": audits, "ridge_selection_resolved": ridge_ok,
            "bootstrap": {"replicates": n_boot, "seed": seed, "family_size": FAMILY_SIZE,
                          "family_alpha": ALPHA, "nominal_simultaneous_confidence": SIMULTANEOUS_LEVEL,
                          "unit": "paired subjects within dataset; paired pose users", "all_draws_retained": True},
            "uncertainty": "Conditional subject bootstrap, one pretraining seed per arm, development data already observed. Does not estimate between-training variance or prove general FM utility.",
            "scope": "Screening for a confirmation proposal only; no automatic adoption, extension, or submission."}


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
