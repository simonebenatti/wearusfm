"""Skeptic probe with EXACT historical audit bytes (for the widened-whitelist clones).

Usage: python -B my_probe_w.py <repo_root> <work_dir>
The real data inputs live on Leonardo; their pins equal the new plan's input_sha256, so
contrastive_run.sha is patched ONLY for the four stand-in input paths to return those pins.
"""
import gzip
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "tests/cpu"))
import contrastive_prepare as P
import contrastive_run as R
import contrastive_benchmark_readout as B
import contrastive_source_compat as S

out = {"root": str(root), "source_digest": P.source_digest()}
raw = gzip.decompress((root / "results/infra/contrastive_comparison_20261116/audit_completed_20261117/audit.json.gz").read_bytes())
historical = json.loads(raw)
plan_pins = P._read(P.PLAN)["input_sha256"]
out["audit_inputs_equal_new_plan_pins"] = historical["inputs"] == plan_pins
cal_new = P.build_review_configs(P._read(P.BASELINE))["calibration"]
exact = work / "audit.json"
exact.write_bytes(raw)


def attempt(label, fn):
    try:
        out[label] = {"accepted": True, "value": fn()}
    except Exception as exc:
        out[label] = {"accepted": False, "error": type(exc).__name__ + ": " + str(exc)[:200]}


attempt("D0_compat_guard_exact_audit", lambda: sorted(S.validate_historical_evidence("audit", exact)))

stand_in = {}
for name in historical["inputs"]:
    p = work / (name + ".bin")
    p.write_text("stand-in\n")
    stand_in[name] = p
real_sha = R.sha
by_path = {str(p): plan_pins[name] for name, p in stand_in.items()}
R.sha = lambda path: by_path.get(str(path)) or real_sha(path)
attempt("D1_validate_audit_exact_bytes_new_recipe", lambda: R.validate_audit(exact, stand_in, cal_new) is not None)
R.sha = real_sha


class AfterAuditGate(Exception):
    pass


cohort_path, receipt_path = work / "cohort.json", work / "receipt.json"
cohort_path.write_text(json.dumps({"endpoint": 576, "source_sha256": P.source_digest(),
                                   "configs": {arm: {"path": arm + ".json"} for arm in B.ARMS}}))
receipt_path.write_text(json.dumps({"input_sha256": plan_pins}))
B.authenticate_calibration = lambda *a: None
B.calibrated_configs = lambda *a: {}
R.validate_cohort = lambda *a: None


def stop(*a):
    raise AfterAuditGate("reached CUDA evidence: audit gate passed")


B._cuda_evidence = stop
attempt("D2_read_preflight_exact_bytes_new_recipe", lambda: B.read_preflight(
    cohort_path, {arm: work / arm for arm in B.ARMS}, {arm: work / (arm + "-r") for arm in ("CN", "CR")},
    work / "cuda.json", cpu_audit_path=exact, calibration_path=receipt_path, calibration_run_dir=work / "calrun"))
print(json.dumps(out, indent=1))
