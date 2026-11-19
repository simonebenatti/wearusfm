"""Build the test fixture calibration run under one checkout and dump calibration_report + key reader facts."""
import json, sys, tempfile
from pathlib import Path
out = Path(sys.argv[1])
import test_contrastive_prepare as T  # loads contrastive_prepare from that checkout's ROOT
P = T.P
work = Path(tempfile.mkdtemp(dir=sys.argv[2]))
T.fixture_run(work / "run")
report = P.calibration_report(work / "run")
facts = {"PLAN": str(P.PLAN.relative_to(P.ROOT)),
         "reader_provenance_plan": report["reader_provenance"]["plan"],
         "weights": report["weights"], "status": report["status"], "failure_reasons": report["failure_reasons"],
         "historical_gradient_multiplier": report["historical_gradient_multiplier"],
         "input_sha256": report["input_sha256"],
         "review_configs": P.build_review_configs(T.baseline()),
         "calibrated_C0_target": None}
receipt = json.loads(json.dumps(report))
cal = P.calibrated_configs(receipt, work / "run", 576)
facts["calibrated"] = {a: (c["loader"]["rms_window_ms"], c["loader"]["ms_fast_window_ms"], c["target_version"], c["contrastive"]["nce_weight"], c["contrastive"]["rnc_weight"]) for a, c in cal.items()}
out.write_text(json.dumps({"facts": facts, "report": report}, indent=1, sort_keys=True))
print(json.dumps({k: facts[k] for k in ("PLAN", "reader_provenance_plan", "weights", "status", "calibrated")}, indent=1))
