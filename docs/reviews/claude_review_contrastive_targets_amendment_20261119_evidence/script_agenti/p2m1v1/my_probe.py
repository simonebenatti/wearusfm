"""Skeptic probe P2-M1: does the audit recipe barrier (config_sha256) act, and is it the only one left?

Usage: python -B my_probe.py <repo_root> <work_dir>
Reads the real historical audit (25/50); writes only below <work_dir>.
"""
import gzip
import hashlib
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
base = root / "results/infra/contrastive_comparison_20261116"
raw = gzip.decompress((base / "audit_completed_20261117/audit.json.gz").read_bytes())
historical = json.loads(raw)
cal_new = P.build_review_configs(P._read(P.BASELINE))["calibration"]
canon_new = hashlib.sha256(json.dumps(cal_new, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
out["audit_config_sha256"] = historical["config_sha256"]
out["new_calibration_canonical_sha256"] = canon_new
out["new_calibration_matches_published_review_config"] = canon_new == hashlib.sha256(json.dumps(
    P._read(root / "results/infra/contrastive_targets_20261118/review_configs/calibration.config.json"),
    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
out["recipe_windows_new"] = [cal_new["loader"]["rms_window_ms"], cal_new["loader"]["ms_fast_window_ms"], cal_new["target_version"]]


def attempt(label, fn):
    try:
        value = fn()
        out[label] = {"accepted": True, "value": value}
    except Exception as exc:  # record the exact gate that stopped it
        out[label] = {"accepted": False, "error": type(exc).__name__ + ": " + str(exc)[:200]}


# A. Exact historical bytes through the current validate_audit (inputs cannot match here: we only
#    show which gate fires FIRST). Uses fake input files; first gate is the source/compat guard.
exact = work / "audit_exact.json"
exact.write_bytes(raw)
fake_inputs = {}
for name in historical["inputs"]:
    p = work / (name + ".json")
    p.write_text("{}\n")
    fake_inputs[name] = p
attempt("A_exact_historical_validate_audit", lambda: R.validate_audit(exact, fake_inputs, cal_new) and "accepted")

# B. Historical audit with source relabelled to the current digest and input pins made coherent
#    (simulates: compat whitelist widened / audit treated as current-source). Only config_sha256
#    differs from the new recipe.
relabelled = dict(historical)
relabelled["source_sha256"] = P.source_digest()
relabelled["inputs"] = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in fake_inputs.items()}
rel_path = work / "audit_relabelled.json"
rel_path.write_text(json.dumps(relabelled))
attempt("B_relabelled_validate_audit", lambda: R.validate_audit(rel_path, fake_inputs, cal_new) and "accepted")

# B2. Same, but with config_sha256 set to the NEW recipe: must pass (control: shows only config differs).
control = dict(relabelled)
control["config_sha256"] = canon_new
ctl_path = work / "audit_control.json"
ctl_path.write_text(json.dumps(control))
attempt("B2_control_new_config_validate_audit", lambda: R.validate_audit(ctl_path, fake_inputs, cal_new) and "accepted")


# C. read_preflight with the relabelled audit; everything before the audit gate is stubbed,
#    real build_review_configs and real _coverage; stop right after the audit gate.
class AfterAuditGate(Exception):
    pass


cohort_path = work / "cohort.json"
receipt_path = work / "receipt.json"
cohort_path.write_text(json.dumps({"endpoint": 576, "source_sha256": P.source_digest(),
                                   "configs": {arm: {"path": arm + ".json"} for arm in B.ARMS}}))
receipt_path.write_text(json.dumps({"input_sha256": relabelled["inputs"]}))
saved = (B.authenticate_calibration, B.calibrated_configs, R.validate_cohort, B._cuda_evidence)
B.authenticate_calibration = lambda *a: None
B.calibrated_configs = lambda *a: {}
R.validate_cohort = lambda *a: None


def stop(*a):
    raise AfterAuditGate("reached CUDA evidence: audit gate passed")


B._cuda_evidence = stop
guard_calls = []
orig_guard = S.validate_historical_evidence


def guard(kind, payload):
    guard_calls.append(kind)
    return orig_guard(kind, payload)


S.validate_historical_evidence = guard


def preflight(audit_path):
    return B.read_preflight(cohort_path, {arm: work / arm for arm in B.ARMS},
                            {arm: work / (arm + "-r") for arm in ("CN", "CR")}, work / "cuda.json",
                            cpu_audit_path=audit_path, calibration_path=receipt_path,
                            calibration_run_dir=work / "calrun")


attempt("C_relabelled_read_preflight", lambda: preflight(rel_path))
attempt("C2_control_new_config_read_preflight", lambda: preflight(ctl_path))
out["guard_calls_during_C"] = guard_calls
B.authenticate_calibration, B.calibrated_configs, R.validate_cohort, B._cuda_evidence = saved
S.validate_historical_evidence = orig_guard
print(json.dumps(out, indent=1))
