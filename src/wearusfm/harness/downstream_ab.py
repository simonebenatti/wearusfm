"""Paired downstream summaries; preserves the signed P1/P2 point estimands."""
import hashlib
import json

import numpy as np


def fingerprint(*values):
    """Content hash of actual ordered inputs, without consuming RNG."""
    h = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object input cannot be fingerprinted")
            h.update(json.dumps([value.dtype.str, value.shape]).encode())
            h.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        else:
            h.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        h.update(b"\0")
    return h.hexdigest()


def subject_counts(y, pred, subjects):
    """Per-subject/class denominators and correct predictions reconstruct pooled BAcc."""
    y, pred, subjects = map(np.asarray, (y, pred, subjects))
    if y.ndim != 1 or not (y.shape == pred.shape == subjects.shape) or not len(y):
        raise ValueError("invalid prediction arrays")
    classes = np.unique(y)
    return {"classes": classes.tolist(), "subjects": {
        str(s): {"n": [int(np.sum((subjects == s) & (y == c))) for c in classes],
                 "correct": [int(np.sum((subjects == s) & (y == c) & (pred == y))) for c in classes]}
        for s in np.unique(subjects)}}


def _arrays(stats):
    subjects = sorted(stats["subjects"])
    n = np.asarray([stats["subjects"][s]["n"] for s in subjects], dtype=float)
    hits = np.asarray([stats["subjects"][s]["correct"] for s in subjects], dtype=float)
    if (not subjects or len(set(stats["classes"])) != len(stats["classes"]) or
            n.shape != (len(subjects), len(stats["classes"])) or hits.shape != n.shape or
            not np.isfinite(n).all() or not np.isfinite(hits).all() or
            np.any(n < 0) or np.any(hits < 0) or np.any(hits > n) or
            np.any(n != np.floor(n)) or np.any(hits != np.floor(hits)) or np.any(n.sum(1) == 0)):
        raise ValueError("invalid subject counts")
    return subjects, n, hits


def bacc(n, hits):
    totals, correct = np.sum(n, axis=0), np.sum(hits, axis=0)
    present = totals > 0
    if np.sum(present) < 2:
        raise ValueError("bootstrap sample has fewer than two classes")
    return float(np.mean(correct[present] / totals[present]))


def paired_component(stats, rng, n_boot=1000):
    """stats order A0,B0,A1,B1. One subject draw shared by all four models."""
    arrays = [_arrays(s) for s in stats]
    subs, n, _ = arrays[0]
    for s, (names, den, _) in zip(stats, arrays):
        if names != subs or s["classes"] != stats[0]["classes"] or not np.array_equal(den, n):
            raise ValueError("unpaired subjects/classes/denominators")
    point = np.array([bacc(a[1], a[2]) for a in arrays])
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(len(subs), size=len(subs))
        v = [bacc(n[pick], a[2][pick]) for a in arrays]
        draws.append(((v[1] - v[0]) + (v[3] - v[2])) / 2)
    deltas = point[[1, 3]] - point[[0, 2]]
    result = {"A": float(point[[0, 2]].mean()), "B": float(point[[1, 3]].mean()),
              "delta": float(deltas.mean()), "delta_by_model_seed": deltas.tolist(),
              "ci95": np.quantile(draws, [.025, .975]).tolist(),
              "paired_subject_se": float(np.std(draws, ddof=1))}
    return result, np.asarray(draws)


def compare_reports(reports, n_boot=1000, seed=20261009):
    """P2 overlapping splits: RMS paired SE, without division by sqrt(3)."""
    rng = np.random.default_rng(seed)
    output = {"bootstrap_seed": seed, "n_boot": n_boot, "tasks": {},
              "uncertainty": "Conditional on the two trained models. P1 percentile subject bootstrap; "
              "P2 descriptive normal interval from RMS split paired SE, no independence assumption between splits."}
    for task, datasets in (("p1", ("ninapro_db2", "ninapro_db3", "ninapro_db6")),
                           ("p2", ("epn612", "uci_emg"))):
        rr = [reports[task][k] for k in ("A_seed0", "B_seed0", "A_seed1", "B_seed1")]
        ds_out, ds_draws = {}, []
        for ds in datasets:
            runs = [[r["datasets"][ds]] if task == "p1" else r["datasets"][ds]["splits"] for r in rr]
            if task == "p2" and any([v["split_seed"] for v in run] != [0, 1, 2] for run in runs):
                raise ValueError("P2 split seeds changed")
            components, draws = [], []
            for j in range(len(runs[0])):
                stats = [run[j]["paired_counts"] for run in runs]
                result, sample = paired_component(stats, rng, n_boot)
                for run, stat in zip(runs, stats):
                    _, n, hits = _arrays(stat)
                    if not np.isclose(run[j]["test_bacc"], bacc(n, hits), atol=1e-12, rtol=0):
                        raise ValueError("reported accuracy differs from counts")
                components.append(result)
                draws.append(sample)
            for r, run in zip(rr, runs):
                if not np.isclose(r["datasets"][ds]["test_bacc"], np.mean([v["test_bacc"] for v in run]), atol=1e-12, rtol=0):
                    raise ValueError("dataset accuracy differs from split mean")
            if task == "p1":
                ds_out[ds] = components[0]
                ds_draws.append(draws[0])
            else:
                value = {k: float(np.mean([c[k] for c in components])) for k in ("A", "B", "delta")}
                value["delta_by_model_seed"] = np.mean([c["delta_by_model_seed"] for c in components], axis=0).tolist()
                value["paired_subject_se"] = float(np.sqrt(np.mean([c["paired_subject_se"]**2 for c in components])))
                value["ci95"] = [value["delta"] + a*1.96*value["paired_subject_se"] for a in (-1, 1)]
                value["splits"] = components
                ds_out[ds] = value
        weights = np.array([8, 3, 2], dtype=float)/13 if task == "p1" else np.array([.5, .5])
        for r in rr:
            expected = float(weights @ np.array([r["datasets"][ds]["test_bacc"] for ds in datasets]))
            if not np.isclose(expected, r[task], atol=1e-12, rtol=0):
                raise ValueError("aggregate metric differs from fixed weights")
        row = {k: float(weights @ np.array([ds_out[ds][k] for ds in datasets])) for k in ("A", "B", "delta")}
        row["delta_by_model_seed"] = (weights @ np.array([ds_out[ds]["delta_by_model_seed"] for ds in datasets])).tolist()
        if task == "p1":
            samples = weights @ np.asarray(ds_draws)
            row["paired_subject_se"] = float(np.std(samples, ddof=1))
            row["ci95"] = np.quantile(samples, [.025, .975]).tolist()
        else:
            row["paired_subject_se"] = float(np.sqrt(sum((weights[i]*ds_out[ds]["paired_subject_se"])**2 for i, ds in enumerate(datasets))))
            row["ci95"] = [row["delta"] + a*1.96*row["paired_subject_se"] for a in (-1, 1)]
        row["datasets"] = ds_out
        row["practical_guard_pass"] = bool(row["delta"] >= -.02)
        output["tasks"][task] = row
    output["both_practical_guards_pass"] = all(r["practical_guard_pass"] for r in output["tasks"].values())
    output["guard_note"] = "Point estimate >= -0.02 separately for P1 and P2; not statistical non-inferiority or automatic adoption."
    return output
