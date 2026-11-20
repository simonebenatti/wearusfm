"""First statements of cpu_benchmark / gpu_benchmark / wls_calibration / controls on the real-shaped train cache,
through the unmocked reader (depth=None), with the producer user list of the new plan (cache IDs)."""
import json, sys, time, tempfile
from pathlib import Path
import numpy as np
sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P, data as D, benchmark, controls, workflow as W, runner as R, metrics as M

root = Path(sys.argv[1]).resolve()
plan = json.loads((root/"plan.json").read_text()); plan.pop("historical_reference_fixture", None)
allowed = plan["producers"]["train"]["users"]
identity = {"execution_sha256": "e"*64}
loader = lambda: P.read_verified_inputs(plan, "train", stage="cpu_benchmark")
cache, first = benchmark._load_measurement(loader, identity)
print("numpy", np.__version__, "| metadata keys:", sorted(cache["meta"][0]), "| arrays:", sorted(k for k, v in cache.items() if isinstance(v, np.ndarray)))
folds, population = benchmark._validate_train(cache, allowed, identity)
print("population:", population, "| io:", benchmark._input_io(cache, loader, identity, first)["load_scope"])
try:
    benchmark._validate_train(cache, plan["producers"]["train"]["storage_directories"], identity)
except ValueError as error:
    print("first delivery's list (directories) at the benchmark:", error)
partition = controls.validate_outer_partition(cache["meta"], folds[0], folds[1], allowed)
inner = controls.inner_partitions(cache["meta"], folds[0])
previous_folds = json.loads(json.dumps(D.alternating_user_folds(cache)))          # as read back from the preflight receipt
wls = dict(cache); wls.pop("pre", None)
groups = D.alternating_user_folds(wls, previous_folds)
fit = np.asarray([i for i, m in enumerate(wls["meta"]) if m["user"] in set(groups[0])])
held = np.asarray([i for i, m in enumerate(wls["meta"]) if m["user"] not in set(groups[0])])
print("partition users", len(partition["fitting_users"]), len(partition["held_users"]), "| inner", [(len(a), len(b)) for a, b in inner],
      "| wls fold 0 equals benchmark fold 0:", np.array_equal(fit, folds[0]) and np.array_equal(held, folds[1]))
ids = W._metadata_ids(cache)
bundle = W.prepare_real_folds(cache, allowed_train_users=allowed, source_identity=identity, expected_folds=previous_folds)
datasets = [W._training_set(bundle, bundle.fitting[0], depth, 0) for depth in ("post", "pre")]
print("window ids unique:", len(set(ids)) == len(ids), "| real fold bundle windows/excluded:",
      [(f["fitting_windows"], f["excluded_fitting_windows"]) for f in bundle.manifest["folds"]],
      "| training sets supported:", [d.size for d in datasets], [str(d.features.dtype) for d in datasets])
support = D.central_support(cache["mask"], D.lengths_for(cache), cache["key_time_valid"])
started = time.perf_counter()
components = controls.generate_components(cache["lat"], cache["meta"], support, folds[0], folds[1], allowed, outer_fold=0, n_joints=20, block_windows=32)
print("generate_components s", round(time.perf_counter()-started, 1), "| partition equal:", components["partition"] == partition,
      "| types", components["L"].dtype, type(partition["fitting_users"][0]).__name__)
model_score = W._score_model(__import__("pose_temponet.heads", fromlist=["x"]).make_head("TCN_post", 0), benchmark._scoring_bundle(cache, folds), 0, folds[1][:64], "post", "cpu", include_clean=True)
print("production score on 64 held windows:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in model_score[0].items()})
selected = sorted({row["user"] for row in cache["meta"]})[:15]
first_ids = [next(i for i, row in enumerate(cache["meta"]) if row["user"] == user) for user in selected]
print("first windows of the 15 surrogate users have central support:", support[first_ids].any(axis=(1, 2)).tolist())
with tempfile.TemporaryDirectory(dir=str(root)) as tmp:
    try:
        report = benchmark._report_measurement(cache, output_dir=Path(tmp), input_load_seconds=first["input_load_seconds"])
        print("report surrogate OK", report["bootstrap_replicates"])
    except ValueError as error:
        print("report surrogate FAILED:", error)
