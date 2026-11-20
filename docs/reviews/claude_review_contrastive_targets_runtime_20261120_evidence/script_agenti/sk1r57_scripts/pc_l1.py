"""Positive control L1: what finish() writes vs what readout.py:511-514 binds."""
import hashlib, json, sys, shutil
from pathlib import Path
from types import SimpleNamespace
out = Path(sys.argv[1])
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
import contrastive_target_parity as P
print("module:", P.__file__)
provenance = {"source_sha256": "s" * 64, "calibration_sha256": "c" * 64, "cohort_sha256": "h" * 64, "input_sha256": {"manifest": "m" * 64}}
(out / "summary.json").write_text(json.dumps({"contrastive": {"initial_sha256": "i" * 64, "provenance": provenance}}))
obs = P.ControlParityObserver(out)
obs.config_sha256 = "f" * 64  # as set by measure() at a prescribed point
report = obs.finish(SimpleNamespace(active=False, rank=0, is_main=True))
written = json.loads((out / "zero_weight_parity.json").read_text())
summary_sha = hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
print("keys:", sorted(written))
# Same predicate as contrastive_benchmark_readout.py:511-514, summary/provenance/initial part only.
bound = (written.get("summary_sha256") == summary_sha and written.get("initial_state_sha256") == "i" * 64
         and written.get("provenance") == provenance and written.get("config_sha256") == "f" * 64)
print("readout binding would accept:", bound)
