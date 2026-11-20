"""Integration probe (disposable): real observe_control + real R.train (CPU, C0) -> real finish() report -> readout binding fields.

CPU single process, 4 updates: the prescribed points 128/256 are never reached, so no
probe backward runs. This checks only the wiring (patches, finish, files) and the
fields the readout binds (source/config/summary/initial/provenance), not the gate.
"""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

W = Path(sys.argv[1]); TMP = Path(sys.argv[2]); TMP.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(W / "src"), str(W / "scripts"), str(W / "tests/cpu")]
import contrastive_target_parity as P
import contrastive_benchmark_readout as B
from contrastive_prepare import source_digest
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.training import run as R

data, manifest, pose = _pose_setup(TMP)
cfg = replace(_config(pose, "C0"), max_steps=4)
provenance = {"source_sha256": source_digest(), "calibration_sha256": "c" * 64, "cohort_sha256": "b" * 64, "input_sha256": {}}
out = TMP / "C0"
with P.observe_control(out) as observer:
    summary = R.train(cfg, manifest, [data], out, log=lambda _: None, contrastive_provenance=provenance)
report = json.loads((out / "zero_weight_parity.json").read_text())
config = json.loads((out / "config.json").read_text())
canonical = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
patched_back = (P.ContrastiveRuntime.prepare.__name__, P.trainer._accumulate_terms.__name__)
result = {
    "files": sorted(p.name for p in out.iterdir() if p.name.startswith("zero_weight")),
    "patches_restored": patched_back,
    "points": len(report["points"]),
    "config_sha256_in_report": report["config_sha256"],
    "config_sha256_readout_expects": canonical,
    "summary_sha256_matches": report["summary_sha256"] == B.sha(out / "summary.json"),
    "initial_matches_summary": report["initial_state_sha256"] == summary["contrastive"]["initial_sha256"],
    "provenance_matches_summary": report["provenance"] == summary["contrastive"]["provenance"] == provenance,
    "source_matches_current": report["source_sha256"] == source_digest(),
    "gate": P.validate_parity_report(report)["status"],
    "stored_status": report["status"],
}
# The readout's binding test on this real file (with matching cohort/provenance):
cohort = {"source_sha256": source_digest(), "initial_state_sha256": summary["contrastive"]["initial_sha256"]}
binding_ok = (report.get("source_sha256") == cohort["source_sha256"] and report.get("config_sha256") == canonical and
              report.get("summary_sha256") == B.sha(out / "summary.json") and
              report.get("initial_state_sha256") == cohort["initial_state_sha256"] and report.get("provenance") == provenance)
result["readout_binding_would_pass"] = binding_ok
print(json.dumps(result, indent=1))
