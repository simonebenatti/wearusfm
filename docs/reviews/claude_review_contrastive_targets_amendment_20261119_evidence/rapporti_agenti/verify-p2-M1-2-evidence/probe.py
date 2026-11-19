"""Independent probe of the audit recipe barrier (skeptic 2, P2-M1).

Usage: python -B probe.py <root> <outdir> <label>
Reads only; writes temporary files under <outdir>/<label>/.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve() / sys.argv[3]
out.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
import contrastive_prepare as P
import contrastive_run as R
import contrastive_benchmark_readout as B
import contrastive_source_compat as S

assert Path(P.ROOT).resolve() == root and Path(R.ROOT).resolve() == root
result = {"root": str(root), "source_digest": P.source_digest()}


def canon(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


cal = P.build_review_configs(P._read(P.BASELINE))["calibration"]
result["calibration_recipe"] = {"rms": cal["loader"]["rms_window_ms"], "fast": cal["loader"]["ms_fast_window_ms"],
                                "version": cal["target_version"], "canonical_sha256": canon(cal)}
raw = gzip.decompress((root / "results/infra/contrastive_comparison_20261116/audit_completed_20261117/audit.json.gz").read_bytes())
hist = json.loads(raw)
result["historical_audit"] = {"bytes_sha256": hashlib.sha256(raw).hexdigest(), "source_sha256": hist["source_sha256"],
                              "config_sha256": hist["config_sha256"]}
hist_path = out / "audit_hist.json"
hist_path.write_bytes(raw)
relabel = dict(hist)
relabel["source_sha256"] = P.source_digest()
relabel_path = out / "audit_relabel.json"
relabel_path.write_text(json.dumps(relabel))

# Physical inputs are not in the repository: dummy files whose sha is mapped to
# the audit's own pins, so the input gate is coherent and only later gates act.
inputs, mapping = {}, {}
for name, digest in hist["inputs"].items():
    p = out / (name + ".json")
    p.write_text("{}")
    inputs[name] = p
    mapping[str(p.resolve())] = digest
real_sha = R.sha
R.sha = lambda path: mapping.get(str(Path(path).resolve()), None) or real_sha(path)


def attempt(fn):
    try:
        value = fn()
    except Exception as exc:  # report the exact message
        return {"accepted": False, "error": type(exc).__name__ + ": " + str(exc)[:200]}
    return {"accepted": True, "returned_passed": value.get("passed") if isinstance(value, dict) else repr(value)[:80]}


# A: unmodified historical bytes, real guard.
result["A_validate_audit_hist_real_guard"] = attempt(lambda: R.validate_audit(hist_path, inputs, cal))
# B: relabelled source (guard skipped), coherent inputs.
result["B_validate_audit_relabel"] = attempt(lambda: R.validate_audit(relabel_path, inputs, cal))
# B': original bytes, guard forced to accept (stands for a widened whitelist).
orig_guard = S.validate_historical_evidence
S.validate_historical_evidence = lambda kind, payload: {"kind": kind}
result["B2_validate_audit_hist_guard_accepts"] = attempt(lambda: R.validate_audit(hist_path, inputs, cal))
S.validate_historical_evidence = orig_guard
# Control: same relabelled audit against the HISTORICAL calibration config is accepted by the
# config barrier -> the barrier is what separates 25/50 from 100/125.
hist_cal = P._read(root / "results/infra/contrastive_comparison_20261116/configs/calibration.config.json")
result["historical_calibration_config_canonical_sha256"] = canon(hist_cal)
result["C_control_relabel_vs_historical_config"] = attempt(lambda: R.validate_audit(relabel_path, inputs, hist_cal))


# D: read_preflight audit branch with real BASELINE/build_review_configs/_coverage.
class AfterAuditGate(Exception):
    pass


def preflight(audit_path, guard_accepts=False):
    saved = (B.authenticate_calibration, B.calibrated_configs, R.validate_cohort, B._cuda_evidence, S.validate_historical_evidence)
    cohort_path, receipt_path = out / "cohort.json", out / "receipt.json"
    cohort_path.write_text(json.dumps({"endpoint": 576, "source_sha256": P.source_digest(),
                                       "configs": {arm: {"path": arm + ".json"} for arm in B.ARMS}}))
    receipt_path.write_text(json.dumps({"input_sha256": hist["inputs"]}))
    B.authenticate_calibration = lambda *a: None
    B.calibrated_configs = lambda *a: {}
    R.validate_cohort = lambda *a: None

    def stop(*a):
        raise AfterAuditGate("passed the CPU audit gate")
    B._cuda_evidence = stop
    if guard_accepts:
        S.validate_historical_evidence = lambda kind, payload: {"kind": kind}
    try:
        return attempt(lambda: B.read_preflight(cohort_path, {a: out / a for a in B.ARMS},
                                                {a: out / (a + "-r") for a in ("CN", "CR")}, out / "cuda.json",
                                                cpu_audit_path=audit_path, calibration_path=receipt_path,
                                                calibration_run_dir=out / "calrun"))
    finally:
        (B.authenticate_calibration, B.calibrated_configs, R.validate_cohort, B._cuda_evidence,
         S.validate_historical_evidence) = saved


result["D_read_preflight_hist_real_guard"] = preflight(hist_path)
result["E_read_preflight_relabel"] = preflight(relabel_path)
result["E2_read_preflight_hist_guard_accepts"] = preflight(hist_path, guard_accepts=True)
print(json.dumps(result, indent=1))
