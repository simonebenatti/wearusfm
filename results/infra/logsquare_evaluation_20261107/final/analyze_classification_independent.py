"""Offline audit from saved counts and candidate OOF; no production imports or fits.

Run from any directory with the WearUsFM CPU Python environment. The only output
is classification_independent_analysis.json alongside this script.
"""
from pathlib import Path
import gzip
import hashlib
import io
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARMS = ("T25-L", "T100-L", "T100-G", "T100-LQ")
WEIGHTS = {"p1": {"ninapro_db2": 8/13, "ninapro_db3": 3/13, "ninapro_db6": 2/13},
           "p2": {"epn612": .5, "uci_emg": .5}}
HIST = ROOT / "results/infra/tokenizer_evaluation_20261103/probe_cpu_19618"


def read_json(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def checked_original(folder):
    meta = read_json(folder / "verification.json")
    raw = gzip.decompress((folder / "report.json.gz").read_bytes())
    assert len(raw) == meta["bytes"]
    assert hashlib.sha256(raw).hexdigest() == meta["sha256"]
    for name, item in meta["sidecars"].items():
        path = folder / (Path(item["path"]).name + ".gz")
        if path.exists():
            data = gzip.decompress(path.read_bytes())
            assert len(data) == item["bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
    return json.loads(raw), meta["sha256"]


def same(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-12)


def score(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def audit_records(records, arm, task, aggregate):
    assert len(records) == 28 * len(WEIGHTS[task])
    assert all(r["converged_by_criterion"] and not r["reached_limit"]
               and r["convergence_warnings"] == 0 and not r["anomalous_stop"]
               and r["exception"] is None for r in records)
    for ds in WEIGHTS[task]:
        for fold in aggregate["datasets"][ds]["folds"]:
            chosen = [r for r in records if r["dataset"] == ds and r["fold"] == fold["fold"]]
            trials = [r for r in chosen if r["fit_phase"] == "candidate"]
            refits = [r for r in chosen if r["fit_phase"] == "refit"]
            assert len(chosen) == 7 and len(refits) == 1
            assert sorted(r["C"] for r in trials) == [.001, .01, .1, 1., 10., 100.]
            best = sorted(trials, key=lambda r: (-r["val_bacc"], r["C"]))[0]
            assert best["C"] == refits[0]["C"] == fold["c"]
            for r in chosen:
                assert r["arm"] == arm and r["task"] == task
                assert r["n_features"] == 768 and r["dtype"] == "float32"
                assert r["max_iter"] == 12000 and r["n_iter"] < 12000
                assert r["roles_sha256"] == fold["roles_sha256"]
                assert r["scaler_sha256"] == fold["scaler_sha256"]
                assert r["code_sha"] == aggregate["code_sha"]
                assert r["plan_sha256"] == aggregate["plan_sha256"]
                assert r["solver_params"]["solver"] == "lbfgs"
                assert r["solver_params"]["class_weight"] == "balanced"
                assert r["solver_params"]["tol"] == 1e-4
                assert r["solver_params"]["random_state"] == 0
            sets = [set(fold[k]) for k in ("train", "val", "held")]
            assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    return {"fits": len(records), "converged": len(records),
            "max_iterations_observed": max(r["n_iter"] for r in records),
            "limit": 12000}


report, report_sha = checked_original(HERE / "classification")
old = read_json(HIST / "final/comparison.json")
assert report["complete"] and report["step"] == 19618
assert report["n_bootstrap"] == 2000 and report["seed"] == 20261102
audits, candidate_oof = {}, {}
for task, weights in WEIGHTS.items():
    folder = HERE / ("aggregate-" + task)
    aggregate, _ = checked_original(folder)
    assert aggregate["weights"] == weights
    oof = np.load(io.BytesIO(gzip.decompress((folder / ("aggregate-" + task + ".oof.npz.gz")).read_bytes())), allow_pickle=False)
    records = [json.loads(line) for line in gzip.decompress((folder / ("aggregate-" + task + ".audit.jsonl.gz")).read_bytes()).splitlines()]
    audits["T100-LQ/" + task] = audit_records(records, "T100-LQ", task, aggregate)
    for ds in weights:
        a = {k: oof[ds + "_" + k] for k in ("row", "y", "pred", "s", "fold")}
        assert len(np.unique(a["row"])) == len(a["row"])
        classes, cindex = np.unique(a["y"], return_inverse=True)
        subjects, sindex = np.unique(a["s"], return_inverse=True)
        flat = sindex * len(classes) + cindex
        totals = np.bincount(flat, minlength=len(subjects)*len(classes)).reshape(len(subjects), len(classes))
        hits = np.bincount(flat[a["y"] == a["pred"]], minlength=len(subjects)*len(classes)).reshape(totals.shape)
        rebuilt = {"classes": classes.tolist(), "subjects": {str(s): {"n": totals[i].tolist(), "correct": hits[i].tolist()} for i, s in enumerate(subjects)}}
        assert rebuilt == report["datasets"][ds]["paired_counts"]["T100-LQ"]
        same(score(a["y"], a["pred"]), aggregate["datasets"][ds]["balanced_accuracy"])
        assert len(subjects) == aggregate["datasets"][ds]["n_oof_subjects"]
        assert len(flat) == aggregate["datasets"][ds]["n_oof_windows"]
        for fold in aggregate["datasets"][ds]["folds"]:
            mask = a["fold"] == fold["fold"]
            assert set(a["s"][mask]) == set(fold["held"])
            same(score(a["y"][mask], a["pred"][mask]), fold["held_bacc"])
        candidate_oof[ds] = {"windows": len(flat), "subjects": len(subjects), "classes": len(classes), "all_subject_class_counts_exact": True}

for arm in ARMS[:-1]:
    for task in WEIGHTS:
        pin = report["historical_reports"][arm + "/" + task]
        path = ROOT / pin["local_archive"]
        # Archives can preserve parsed JSON rather than byte-identical remote formatting.
        aggregate = read_json(path)
        receipt = read_json(ROOT / pin["local_hash_receipt"])
        assert pin["sha256"] == receipt["report_sha256"]
        for field in ("code_sha", "plan_sha256", "plan_content_sha256", "numeric_sha256"):
            assert aggregate[field] == pin[field]
        audit_path = path.with_suffix(".audit.jsonl")
        assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == receipt["actual_hashes"]["audit"] == aggregate["audit_sha256"]
        records = [json.loads(line) for line in audit_path.read_text().splitlines()]
        audits[arm + "/" + task] = audit_records(records, arm, task, aggregate)
        for ds in WEIGHTS[task]:
            assert report["datasets"][ds]["paired_counts"][arm] == old["datasets"][ds]["paired_counts"][arm]
            same(report["datasets"][ds]["cells"][arm]["balanced_accuracy"], aggregate["datasets"][ds]["balanced_accuracy"])

rng = np.random.default_rng(20261102)
points, draws, out = {}, {}, {"datasets": {}, "tasks": {}}


def summarize(point, sample, original, ni=False):
    block = {"cells_percent": {}, "contrasts": {}}
    for arm in ARMS:
        interval = np.percentile(sample[arm], [2.5, 97.5]) * 100
        same(point[arm], original["cells"][arm]["balanced_accuracy"])
        same(interval, original["cells"][arm]["ci95_percent"])
        block["cells_percent"][arm] = {"estimate": 100*point[arm], "ci95": interval.tolist()}
    for arm in ARMS[:-1]:
        name = "T100-LQ_minus_" + arm
        delta = 100 * (point["T100-LQ"] - point[arm])
        interval = np.percentile(sample["T100-LQ"] - sample[arm], [2.5, 97.5]) * 100
        same(delta, original["contrasts"][name]["delta_pp"])
        same(interval, original["contrasts"][name]["ci95_pp"])
        block["contrasts"][name] = {"delta_pp": delta, "ci95_pp": interval.tolist()}
        if ni:
            verdict = "supported" if interval[0] > -1 else "loss_beyond_margin" if interval[1] < -1 else "unresolved"
            assert verdict == original["contrasts"][name]["noninferiority_1pp"]
            block["contrasts"][name]["noninferiority_1pp"] = verdict
    return block


for ds in sorted(report["datasets"]):
    block = report["datasets"][ds]
    subjects = sorted(block["paired_counts"]["T100-LQ"]["subjects"])
    assert subjects == block["subjects"]
    n = np.array([block["paired_counts"]["T100-LQ"]["subjects"][s]["n"] for s in subjects], dtype=np.int64)
    indices = rng.integers(len(subjects), size=(2000, len(subjects)))
    # Independent implementation: transform draw indices into subject multiplicities,
    # then multiply class counts. No production bootstrap or scoring helper is called.
    multiplicity = np.array([np.bincount(row, minlength=len(subjects)) for row in indices])
    denominators = multiplicity @ n
    present = denominators > 0
    points[ds], draws[ds] = {}, {}
    for arm in ARMS:
        counts = block["paired_counts"][arm]["subjects"]
        assert np.array_equal(n, np.array([counts[s]["n"] for s in subjects]))
        hits = np.array([counts[s]["correct"] for s in subjects], dtype=np.int64)
        assert ((0 <= hits) & (hits <= n)).all()
        numerators = multiplicity @ hits
        recalls = np.divide(numerators, denominators, out=np.zeros(numerators.shape), where=present)
        draws[ds][arm] = recalls.sum(axis=1) / present.sum(axis=1)
        points[ds][arm] = float(np.mean(hits.sum(0) / n.sum(0)))
    assert int((present.sum(1) < n.shape[1]).sum()) == block["replicates_missing_classes"]
    out["datasets"][ds] = summarize(points[ds], draws[ds], block)
    out["datasets"][ds].update(candidate_oof[ds])

for task, weights in WEIGHTS.items():
    point = {arm: sum(w*points[ds][arm] for ds, w in weights.items()) for arm in ARMS}
    sample = {arm: sum(w*draws[ds][arm] for ds, w in weights.items()) for arm in ARMS}
    out["tasks"][task] = summarize(point, sample, report["tasks"][task], ni=True)

out.update({"status": "independent_recalculation_passed", "source_report_sha256": report_sha,
            "method": "candidate OOF bincount reconstruction, archived historical subject counts, independent multiplicity-matrix paired bootstrap; no production imports",
            "n_bootstrap": 2000, "seed": 20261102, "weights": WEIGHTS,
            "audit": audits, "total_fits": sum(v["fits"] for v in audits.values()),
            "candidate_fits": 140, "historical_fits": 420,
            "scope": "one pretraining seed; fixed checkpoints and fitted probes; repeated development data; no multiple-comparison adjustment"})
(HERE / "classification_independent_analysis.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": out["status"], "fits": out["total_fits"], "datasets": {k:v["contrasts"] for k,v in out["datasets"].items()}, "tasks": out["tasks"]}, indent=2))
