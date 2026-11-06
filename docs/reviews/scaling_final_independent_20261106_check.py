"""Independent arithmetic audit: published subject counts, no project imports.

Reconstructs classification points and paired subject bootstrap using multiplicity
weights (rather than production's repeated-index reduction). Reconstructs pose
contrasts from per-user SSE differences and target variance. Raw predictions and
fits are not rerun. All printed results exclude subject identifiers.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
raw = args.report.read_bytes()
report = json.loads(raw)
cells = ("T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched")
contrasts = {"natural_300m_minus_30m": (1, 0), "matched_300m_minus_30m": (2, 0),
             "natural_300m_minus_matched_300m": (1, 2)}
task_parts = {"p1": (("ninapro_db2", 8/13), ("ninapro_db3", 3/13), ("ninapro_db6", 2/13)),
              "p2": (("epn612", .5), ("uci_emg", .5))}
audit = {"source_sha256": hashlib.sha256(raw).hexdigest(), "checks": 0,
         "maximum_absolute_difference": 0., "classification": {}, "pose": {}}

def equal(got, expected):
    g, e = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
    delta = float(np.max(np.abs(g-e)))
    audit["maximum_absolute_difference"] = max(audit["maximum_absolute_difference"], delta)
    assert g.shape == e.shape and np.allclose(g, e, rtol=0., atol=1e-10), (g, e)
    audit["checks"] += 1

def draw_weights(rng, n, repeats):
    draw = rng.integers(n, size=(repeats, n))
    weight = np.asarray([np.bincount(row, minlength=n) for row in draw], dtype=float)
    return draw, weight

def draw_hash(draw, trailing_nul=True):
    h = hashlib.sha256()
    h.update(json.dumps([draw.dtype.str, draw.shape]).encode())
    h.update(draw.tobytes(order="C"))
    if trailing_nul:
        h.update(b"\0")
    return h.hexdigest()

def verify_summary(block, point, samples):
    result = {"scores_percent": dict(zip(cells, (100*point).tolist())), "contrasts": {}}
    for i, name in enumerate(cells):
        equal(point[i], block["cells"][name]["balanced_accuracy"])
        equal(np.percentile(samples[:, i]*100, [2.5, 97.5]), block["cells"][name]["ci95_percent"])
    for key, (a,b) in contrasts.items():
        delta = 100*(point[a]-point[b])
        ci = np.percentile(100*(samples[:, a]-samples[:, b]), [2.5, 97.5])
        equal(delta, block["contrasts"][key]["delta_pp"])
        equal(ci, block["contrasts"][key]["ci95_pp"])
        result["contrasts"][key] = {"delta_pp": float(delta), "ci95_pp": ci.tolist()}
    return result

classification = report["secondary"]["classification"]
assert classification["seed"] == 20261102 and classification["n_bootstrap"] == 2000
rng = np.random.default_rng(20261102)
points, boot = {}, {}
for ds, block in sorted(classification["datasets"].items()):
    users = sorted(block["paired_counts"][cells[0]]["subjects"])
    assert users == block["subjects"]
    denominators, correct = [], []
    for name in cells:
        counts = block["paired_counts"][name]
        assert counts["classes"] == block["classes"] and sorted(counts["subjects"]) == users
        denominators.append(np.asarray([counts["subjects"][u]["n"] for u in users], dtype=float))
        correct.append(np.asarray([counts["subjects"][u]["correct"] for u in users], dtype=float))
    assert all(np.array_equal(denominators[0], other) for other in denominators[1:])
    denominator = denominators[0]
    assert all(np.all((h >= 0) & (h <= denominator)) for h in correct)
    draw, weights = draw_weights(rng, len(users), 2000)
    assert draw_hash(draw) == block["draws_sha256"]
    nboot = weights @ denominator
    valid = nboot > 0
    no_classes = valid.sum(axis=1)
    totals = denominator.sum(axis=0)
    point = np.array([np.mean(h.sum(axis=0)[totals > 0]/totals[totals > 0]) for h in correct])
    samples = np.column_stack([np.divide(weights @ h, nboot, out=np.zeros_like(nboot), where=valid).sum(axis=1)/no_classes for h in correct])
    equal(int(np.sum(no_classes < len(totals))), block["replicates_missing_classes"])
    equal(int(np.sum(no_classes == 1)), block["single_class_replicates"])
    result = verify_summary(block, point, samples)
    result["n_subjects"] = len(users)
    result["n_classes"] = len(totals)
    result["n_windows"] = int(totals.sum())
    # Descriptive userwise differences, not new tests or signed endpoints.
    recalls = [np.divide(h, denominator, out=np.zeros_like(h), where=denominator>0).sum(axis=1)/(denominator>0).sum(axis=1) for h in correct]
    diff = recalls[1]-recalls[0]
    result["descriptive_userwise_bacc_delta"] = {"positive": int(np.sum(diff>1e-15)), "negative": int(np.sum(diff < -1e-15)), "ties": int(np.sum(np.abs(diff)<=1e-15)), "median_pp": float(np.median(diff)*100)}
    audit["classification"][ds] = result
    points[ds], boot[ds] = point, samples

for task, parts in task_parts.items():
    point = sum(w*points[ds] for ds,w in parts)
    samples = sum(w*boot[ds] for ds,w in parts)
    audit["classification"][task] = verify_summary(classification["tasks"][task], point, samples)

for group, block in report["secondary"]["pose"].items():
    assert block["seed"] == 20261102 and block["n_bootstrap"] == 2000
    users = sorted(block["users"])
    _, weights = draw_weights(np.random.default_rng(20261102), len(users), 2000)
    audit["pose"][group] = {}
    for key, row in block["contrasts"].items():
        numerator = np.array([row["per_user"][u]["delta_centered_sse"] for u in users])
        denominator = np.array([row["per_user"][u]["target_variance_sum"] for u in users])
        point = numerator.sum()/denominator.sum()
        values = (weights @ numerator)/(weights @ denominator)
        ci = np.percentile(values, [2.5,97.5])
        equal(point, row["delta_ratio"])
        equal(ci, row["ci95"])
        audit["pose"][group][key] = {"delta_ratio": float(point), "ci95": ci.tolist(), "n_subjects": len(users)}

assert report["reading"]["primary_reading"] == {"epn": "better", "pose": "unresolved"}
assert report["reading"]["classification_noninferiority"] == {"p1":"supported", "p2":"supported"}
assert report["reading"]["recommendation"] == "propose_19618_confirmation"
assert report["reading"]["continuation_authorized"] is False
audit["scope"] = "Independent arithmetic from published aggregate counts and pose SSE differences; no raw OOF/prediction audit, no model imports, no causal interpretation of compression control. Subject IDs omitted."
args.out.write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
print(json.dumps(audit, indent=2, allow_nan=False))
