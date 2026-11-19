"""Compare contrastive_prepare outputs with PLAN = new plan vs PLAN = historical plan (read-only)."""
import copy, importlib.util, json, sys, tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(root / "src"), str(root / "scripts"), str(root / "tests/cpu")]
spec = importlib.util.spec_from_file_location("contrastive_prepare", root / "scripts/contrastive_prepare.py")
P = importlib.util.module_from_spec(spec); sys.modules["contrastive_prepare"] = P; spec.loader.exec_module(P)
from wearusfm.training.contrastive_objectives import ContrastiveRuntime

NEW = P.PLAN
OLD = root / "results/infra/contrastive_comparison_20261116/plan.json"
print("PLAN in code:", NEW.relative_to(root))

def fixture(path):
    path.mkdir()
    cfg = P.build_review_configs(json.loads(P.BASELINE.read_text()))["calibration"]
    inputs = P._read(P.PLAN)["input_sha256"]
    datasets = list(cfg["contrastive"]["coverage_quotas"])
    counts = {d: 1 for d in datasets}; counts[datasets[0]] += 32 - len(datasets)
    rt = ContrastiveRuntime(cfg["contrastive"]); rt.expected_datasets = tuple(datasets)
    rt.step = 511; rt.queue.last_step = 511
    rt.coverage = {"updates": 256, "rnc_active": 256, "datasets": {d: [n * 256, n * 256] for d, n in counts.items()}}
    summary = {"steps": 512, "stopped": "max_steps", "world_size": 4, "batch_per_process": 8, "workers_per_rank": 6,
               "code": {"sha": "f" * 40}, "contrastive": {"config": cfg["contrastive"], "optimizer_updates": 0,
               "queue_last_step": 511, "initial_sha256": "a" * 64, "final_student_sha256": "a" * 64,
               "final_teacher_sha256": "a" * 64, "provenance": {"source_sha256": P.source_digest(), "input_sha256": inputs},
               "expected_datasets": datasets, "coverage_completed_blocks": [], "coverage_partial_block": rt.coverage_report()}}
    rows = []
    for step in range(1, 513):
        r = {"step": step, "contrastive_calibration_only": True, "contrast_nce_eligible": 32, "contrast_rnc_windows": 4}
        for d, n in counts.items():
            r["contrast_nce_queries__" + d] = n; r["contrast_nce_eligible__" + d] = n
        if step > 256:
            r.update(contrast_global_grad_base=1., contrast_global_grad_nce=.5, contrast_global_grad_rnc=.25,
                     contrast_global_cos_nce_base=.1, contrast_global_cos_rnc_base=-.2)
        rows.append(r)
    (path / "config.json").write_text(json.dumps(cfg)); (path / "summary.json").write_text(json.dumps(summary))
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

def diff(a, b, prefix=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, "<missing>"), b.get(k, "<missing>"), prefix + "/" + str(k))
    elif a != b:
        out.append(prefix)
    return out

run = work / "fixture_run"; fixture(run)
results = {}
for label, plan in (("new", NEW), ("old", OLD)):
    P.PLAN = plan
    r = {"configs": P.build_review_configs(json.loads(P.BASELINE.read_text())),
         "policy_cov": P.policy_reference("coverage", "manifest_reference"),
         "policy_hist": P.policy_reference("calibration", "historical_reference"),
         "multiplier": P.historical_gradient_multiplier(),
         "report": P.calibration_report(run)}
    inputs = P._read(P.PLAN)["input_sha256"]; P.validate_input_bindings(inputs)
    bad = dict(inputs); bad["scales"] = "0" * 64
    try:
        P.validate_input_bindings(bad); r["rejects_bad_scales"] = False
    except ValueError as e:
        r["rejects_bad_scales"] = str(e)
    results[label] = r
P.PLAN = NEW
d = diff(results["new"], results["old"])
print("differing paths:", d)
print("new report plan pin:", results["new"]["report"]["reader_provenance"]["plan"])
print("old report plan pin:", results["old"]["report"]["reader_provenance"]["plan"])
print("status new/old:", results["new"]["report"]["status"], results["old"]["report"]["status"])
print("weights new/old:", results["new"]["report"]["weights"], results["old"]["report"]["weights"])
print("rejects bad scales new/old:", results["new"]["rejects_bad_scales"], "|", results["old"]["rejects_bad_scales"])
# A receipt produced under the historical PLAN authenticates under that same reader (self-consistent; nothing checks plan).
P.PLAN = OLD
receipt_old = copy.deepcopy(results["old"]["report"])
print("authenticate receipt(old PLAN) under old PLAN:", P.authenticate_calibration(receipt_old, run)["status"])
P.PLAN = NEW
try:
    P.authenticate_calibration(receipt_old, run); print("authenticate receipt(old PLAN) under NEW PLAN: accepted")
except ValueError as e:
    print("authenticate receipt(old PLAN) under NEW PLAN: rejected:", e)
