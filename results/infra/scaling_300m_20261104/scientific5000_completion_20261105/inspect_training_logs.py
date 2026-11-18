"""Read-only accounting of the completed 5000-step training logs (Python 3.6+)."""
import hashlib
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
files = {}
rows = {}
for name in ("metrics.jsonl", "fixed_eval.jsonl", "pose_panel.jsonl"):
    data = (run / name).read_bytes()
    files[name] = {"path": str(run / name), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    rows[name] = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
metrics = rows["metrics.jsonl"]
expected_panel = list(range(0, 5001, 500))
errors = []
if [r.get("step") for r in metrics] != list(range(1, 5001)):
    errors.append("metrics steps not contiguous 1..5000")
if any(r.get("windows") != 32 for r in metrics):
    errors.append("global batch differs from32")
if any(not all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in ("lr", "grad_norm", "total", "pose")) for r in metrics):
    errors.append("missing or nonfinite essential training metric")
if any(not math.isclose(r["lr"], 1e-4 * min(r["step"] / 1962., 1.), rel_tol=1e-12, abs_tol=1e-15) for r in metrics):
    errors.append("LR differs from registered warmup/plateau")
for name in ("fixed_eval.jsonl", "pose_panel.jsonl"):
    if [r.get("step") for r in rows[name]] != expected_panel:
        errors.append(name + " cadence differs")
panels = rows["pose_panel.jsonl"]
if len(set(r.get("panel_sha256") for r in panels)) != 1:
    errors.append("training pose panel identity changed")
if (run / "STOP").exists():
    errors.append("STOP present")
report = {"protocol": "scaling-training-log-audit-20261105-v1", "run_dir": str(run),
          "files": files, "errors": errors, "passed": not errors,
          "metric_count": len(metrics), "global_windows": sum(r["windows"] for r in metrics),
          "fixed_eval_steps": [r["step"] for r in rows["fixed_eval.jsonl"]],
          "pose_panel_steps": [r["step"] for r in panels],
          "panel_sha256": panels[0].get("panel_sha256"),
          "last_train_metrics": {k: metrics[-1][k] for k in ("step", "lr", "grad_norm", "pose", "total", "jepa")},
          "scope": "Training logs and dose only; no checkpoint load or held-out scientific comparison."}
print(json.dumps(report, indent=2, sort_keys=True))
if errors:
    sys.exit(1)
