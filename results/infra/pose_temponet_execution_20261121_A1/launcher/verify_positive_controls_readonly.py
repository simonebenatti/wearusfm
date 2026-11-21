"""Read-only evaluation of the frozen six-control gate; no fitting or writes."""
import json, time
from pathlib import Path
from pose_temponet import provenance as P, execution as E, runner as R
started = time.monotonic()
assert P.source_identity() == "9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea"
plan_pin = {"path": "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117/manifests/b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3.json", "sha256": "b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3"}
plan = P.read_json_pin(plan_pin)
receipts, pins = R._ancestors(plan, E.find_job(plan, "TCNH-selection-L0-20261120"))
assert len([r for r in receipts.values() if r["stage"] == "controls"]) == 6
calibrations = R._calibrations(receipts)
gate = R._control_gate(receipts, calibrations)
print(json.dumps({"scope": "read-only frozen gate, complete calibration states and six selections; no signal cache or held data read; no training, allocation or remote write", "source_sha256": P.source_identity(), "plan": plan_pin, "verified_predecessor_receipts": pins, "gate": gate, "elapsed_seconds": time.monotonic()-started}, sort_keys=True, allow_nan=False))
