"""Skeptic probe for P2-N2: do the current readers see SLURM state or audit duration?

Read-only on the clone: every file written goes to the work dir given as argv[2].
"""
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()
WORK = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(CLONE / "scripts"))
sys.path.insert(0, str(CLONE / "src"))
import contrastive_prepare as P  # noqa: E402
import contrastive_run as runner  # noqa: E402

out = {}
TIME_WORDS = ("time", "elapsed", "slurm", "state", "wall", "second", "job")

# 1. Schema of a real audit report (the historical 64652559 report, TIMEOUT job).
raw = gzip.decompress((CLONE / "results/infra/contrastive_comparison_20261116/audit_completed_20261117/audit.json.gz").read_bytes())
assert hashlib.sha256(raw).hexdigest() == "5c720fadbcb12fd5458bc0e19cd03f28fec30ad5973ca72739f01bf1b3d7547b"
hist = json.loads(raw)
out["historical_top_keys"] = sorted(hist)
out["historical_time_like_top_keys"] = [k for k in hist if any(w in k.lower() for w in TIME_WORDS)]
rec_keys = set().union(*[set(r) for r in hist["records"]])
out["historical_time_like_record_keys"] = sorted(k for k in rec_keys if any(w in k.lower() for w in TIME_WORDS))

# 2. A content-valid audit for the NEW recipe (current source, 100/125/v3 config), no time info.
cfg = P.build_review_configs(P._read(P.BASELINE))["calibration"]
out["new_calibration_loader"] = {k: cfg["loader"].get(k) for k in ("rms_window_ms", "ms_fast_window_ms")}
out["new_calibration_target_version"] = [v for k, v in cfg.items() if k == "target_version"] or \
    [v for k, v in cfg.get("loader", {}).items() if k == "target_version"] or \
    [json.dumps(cfg).count("fast125_v3")]
inputs = {}
for name in ("manifest", "scales", "window_seconds", "pose_stats"):
    p = WORK / (name + ".json")
    p.write_text("{}")
    inputs[name] = p
new = copy.deepcopy(hist)
new["source_sha256"] = P.source_digest()
new["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
new["inputs"] = {k: runner.sha(v) for k, v in inputs.items()}
a1 = WORK / "audit_new_no_time.json"
a1.write_text(json.dumps(new))
try:
    out["validate_audit_new_no_time"] = "ACCEPTED passed=%s" % runner.validate_audit(a1, inputs, cfg)["passed"]
except Exception as e:  # noqa: BLE001
    out["validate_audit_new_no_time"] = "REJECTED: %s" % e

# 3. Same report carrying explicit TIMEOUT markers: readers ignore them.
new2 = dict(new, slurm_state="TIMEOUT", elapsed_s=7386, requested_s=7200, batch_state="COMPLETED")
a2 = WORK / "audit_new_timeout_markers.json"
a2.write_text(json.dumps(new2))
try:
    out["validate_audit_new_timeout_markers"] = "ACCEPTED passed=%s" % runner.validate_audit(a2, inputs, cfg)["passed"]
except Exception as e:  # noqa: BLE001
    out["validate_audit_new_timeout_markers"] = "REJECTED: %s" % e

# 4. Calibration release mechanics on the TIMEOUT-marked audit (fixture pattern of test_contrastive_run.py):
#    validate_release only pins audit bytes + validate_audit content; no state/duration.
sys.path.insert(0, str(CLONE / "tests/cpu"))
from test_contrastive_prepare import fixture_run  # noqa: E402
folder = WORK / "calibration"
fcfg, summary, _ = fixture_run(folder)
out["fixture_config_equals_new_calibration_config"] = (fcfg == cfg)
runner.validate_input_bindings = lambda value: None  # tiny files, as in release_fixture
rel = {"status": "APPROVED_FOR_PREPARATION", "stage": "calibration", "source_sha256": P.source_digest(),
       "config_sha256": runner.sha(folder / "config.json"), "world_size": 4, "workers_per_rank": 6,
       "inputs": {k: runner.sha(v) for k, v in inputs.items()}, "review_pin": "reviewed-commit",
       "time_limit_s": 1770, "audit_sha256": runner.sha(a2)}
try:
    runner.validate_release(rel, "calibration", folder / "config.json", inputs, time_limit_s=1770, audit_path=a2)
    out["validate_release_calibration_with_timeout_audit"] = "ACCEPTED"
except Exception as e:  # noqa: BLE001
    out["validate_release_calibration_with_timeout_audit"] = "REJECTED: %s" % e

# 5. read_preflight audit gate (lines 320-334) on the TIMEOUT-marked audit, upstream evidence mocked.
import contrastive_benchmark_readout as B  # noqa: E402


class ReachedCuda(Exception):
    pass


def stop(*a, **k):
    raise ReachedCuda("audit gate passed; stopped at _cuda_evidence")


B.authenticate_calibration = lambda receipt, run_dir: None
B.calibrated_configs = lambda receipt, run_dir, endpoint: {}
runner.validate_cohort = lambda *a, **k: None
B._cuda_evidence = stop
cohort = WORK / "cohort.json"
cohort.write_text(json.dumps({"endpoint": 576, "source_sha256": P.source_digest(),
                              "configs": {a: {"path": "x"} for a in ("C0", "CN", "CR")}}))
receipt = WORK / "receipt.json"
receipt.write_text(json.dumps({"input_sha256": new["inputs"]}))
try:
    B.read_preflight(cohort, {a: WORK for a in ("C0", "CN", "CR")}, {a: WORK for a in ("CN", "CR")}, WORK / "cuda.json",
                     cpu_audit_path=a2, calibration_path=receipt, calibration_run_dir=WORK)
    out["read_preflight_audit_gate_timeout_audit"] = "no exception (unexpected)"
except ReachedCuda as e:
    out["read_preflight_audit_gate_timeout_audit"] = "ACCEPTED (%s)" % e
except Exception as e:  # noqa: BLE001
    out["read_preflight_audit_gate_timeout_audit"] = "REJECTED: %s" % e

# 6. Control: an audit for the wrong recipe is still rejected by the same gate (probe is not vacuous).
bad = dict(new2, config_sha256="0" * 64)
a3 = WORK / "audit_bad_config.json"
a3.write_text(json.dumps(bad))
try:
    runner.validate_audit(a3, inputs, cfg)
    out["control_validate_audit_wrong_config"] = "ACCEPTED (unexpected)"
except Exception as e:  # noqa: BLE001
    out["control_validate_audit_wrong_config"] = "REJECTED: %s" % e
try:
    B.read_preflight(cohort, {a: WORK for a in ("C0", "CN", "CR")}, {a: WORK for a in ("CN", "CR")}, WORK / "cuda.json",
                     cpu_audit_path=a3, calibration_path=receipt, calibration_run_dir=WORK)
except ReachedCuda as e:
    out["control_read_preflight_wrong_config"] = "ACCEPTED (unexpected)"
except Exception as e:  # noqa: BLE001
    out["control_read_preflight_wrong_config"] = "REJECTED: %s" % e

print(json.dumps(out, indent=1))
