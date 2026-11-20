"""wls_calibration branch of runner.run_stage on the real-size, real-support train cache (fold given on the command line).

Same statements as runner.py:281-298. ONE deviation, for run time on a single Mac thread: the L13 (4992-wide) outer
diagnostic is switched off (include_l13_diagnostics=False); it differs from L0 only by the context of the design.
"""
import io, json, resource, sys, time
from pathlib import Path
import numpy as np, torch

sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P, data, controls, runner as R, workflow as W, metrics as M

root, fold = Path(sys.argv[1]).resolve(), int(sys.argv[2])
plan = json.loads((root/"plan.json").read_text())
original_outer = controls.evaluate_outer_references
controls.evaluate_outer_references = lambda *a, **k: original_outer(*a, **{**k, "include_l13_diagnostics": False})
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30

cache = P.read_verified_inputs(plan, "train", stage="wls_calibration")
cache.pop("pre", None)
previous_folds = json.loads(json.dumps(data.alternating_user_folds(cache)))     # as read back from the preflight receipt
groups = data.alternating_user_folds(cache, previous_folds)
fit_users = set(groups[fold])
fit = np.asarray([i for i, m in enumerate(cache["meta"]) if m["user"] in fit_users])
held = np.asarray([i for i, m in enumerate(cache["meta"]) if m["user"] not in fit_users])
support = data.central_support(cache["mask"], data.lengths_for(cache), cache["key_time_valid"])
print("fold", fold, "| fit/held windows", len(fit), len(held), "| unsupported in fit/held", int((~support[fit].any(axis=(1, 2))).sum()),
      int((~support[held].any(axis=(1, 2))).sum()), "| one-centre windows", int(((support.sum(axis=1).max(axis=1)) == 1).sum()), flush=True)
pv = M.dynamic_variance(cache["target"], cache["mask"])
print("dynamic_variance: NaN windows", int(np.isnan(pv).sum()), "| thresholds", [M.training_threshold(cache["target"][ix], cache["mask"][ix]) for ix in (fit, held)], flush=True)
started = time.perf_counter()
try:
    result = controls.build_positive_control_fold(cache, fit, held, plan["producers"]["train"]["users"], outer_fold=fold, block_windows=32)
except Exception as error:
    import traceback
    frame = traceback.extract_tb(error.__traceback__)[-1]
    print("build_positive_control_fold FAILED after", round(time.perf_counter()-started, 1), "s:", type(error).__name__, error,
          "| at", Path(frame.filename).name, frame.lineno, frame.name, flush=True)
    raise
seconds = time.perf_counter()-started
calibration = result["calibration"]
print("build_positive_control_fold OK | seconds", round(seconds, 1), "| status", calibration["status"], "| evaluations", calibration.get("evaluations"),
      "| real_reference", {k: result["real_reference"][k] for k in ("MSE", "E_dyn", "alpha", "boundary_unresolved")},
      "| selected", {w: {k: c[k] for k in ("a", "sigma", "alpha", "E_cal")} for w, c in calibration.get("selected", {}).items()},
      "| outer scored", result["outer_scored_after_inner_selection"], "| timing", {k: round(v, 1) for k, v in result["timing"].items()}, flush=True)
# what the runner writes: torch state and strict JSON summary
buffer = io.BytesIO(); torch.save(result, buffer)
summary = {k: v for k, v in result.items() if k != "components"}
text = json.dumps(R._plain(summary), sort_keys=True, indent=2, allow_nan=False)
print("calibration.pt bytes", buffer.getbuffer().nbytes, "| calibration.json bytes", len(text), "| peak RSS GiB", round(rss(), 2), flush=True)
reloaded = torch.load(io.BytesIO(buffer.getvalue()), map_location="cpu", weights_only=False)
expected = controls.validate_outer_partition(cache["meta"], fit, held, plan["producers"]["train"]["users"])
print("partition after torch round trip equals a fresh validate_outer_partition:", reloaded["partition"] == expected,
      "| JSON round trip partition equals:", json.loads(text)["partition"] == expected)
if not result["outer_scored_after_inner_selection"]:
    # the inner choice did not calibrate on these synthetic values: exercise the outer path with a fixed choice (cost/wiring only)
    selected = {"linear": {"a": 0., "sigma": 1., "alpha": 1.}, "temporal": {"a": 1., "sigma": 1., "alpha": 1.}}
    started = time.perf_counter()
    outer = controls.evaluate_outer_references(cache, result["components"], fit, held, selected, block_windows=32)
    json.dumps(R._plain(outer), sort_keys=True, allow_nan=False)
    print("evaluate_outer_references (fixed choice, L13 off) OK | seconds", round(time.perf_counter()-started, 1),
          "| linear ridge held", controls.statistic_values(outer["linear"]["ridge"]), "| windows", outer["linear"]["ridge"]["window_count"], flush=True)
