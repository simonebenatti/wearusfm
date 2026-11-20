"""R6-6 probe: calibration numeric_profile receipts are never read.

Read-only with respect to the clone: writes only below the tmp dir given as argv[1].
"""
import copy
import json
import sys
from pathlib import Path

tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=True)

import test_contrastive_prepare as T  # fixture_run, its own instance of contrastive_prepare
P = T.P
import contrastive_prepare as P2  # module used by runner/readout
import contrastive_benchmark_readout as B

folder = tmp / "calibration"
T.fixture_run(folder)
base = P2.calibration_report(folder)
print("no receipts -> status", base["status"], "inputs pinned", sorted(base["inputs"]))

wrong = {"profile": "bf16_auto_tf32_on", "matmul_tf32": True, "cudnn_tf32": True,
         "math_sdp_enabled": True, "flash_sdp_enabled": True, "mem_efficient_sdp_enabled": True,
         "cudnn_sdp_enabled": True, "deterministic_algorithms": False, "cudnn_deterministic": False,
         "cudnn_benchmark": True, "float32_matmul_precision": "high", "torch_version": "other"}
for rank in range(4):
    (folder / f"numeric_profile.rank{rank}.json").write_text(json.dumps(
        {"flags": wrong, "source_sha256": "0" * 64, "config_sha256": "0" * 64,
         "config_canonical_sha256": "0" * 64, "phase": "garbage", "rank": 9, "world_size": 1}))
with_wrong = P2.calibration_report(folder)
print("wrong receipts -> status", with_wrong["status"], "identical report:", with_wrong == base)
measured = P2.authenticate_calibration(copy.deepcopy(base), folder)
print("authenticate_calibration accepts receipt with wrong profile receipts:", measured == base)
cfgs = P2.calibrated_configs(base, folder, 576)
print("calibrated_configs arms:", sorted(cfgs), "weights:", {a: (c["contrastive"]["nce_weight"], c["contrastive"]["rnc_weight"]) for a, c in cfgs.items()})

# Reader of the 28 receipts cannot take the calibration as an eighth run.
full = {a: tmp / a for a in ("C0", "CN", "CR")}
res = {a: tmp / a for a in ("CN_A", "CN_B", "CR_A", "CR_B")}
try:
    B.numeric_context_across_runs(full, {**res, "CALIB": folder})
    print("numeric_context_across_runs accepted calibration: UNEXPECTED")
except ValueError as exc:
    print("numeric_context_across_runs with calibration added ->", exc)
ctx = B.numeric_context_across_runs(full, res)
print("seven empty runs ->", ctx["status"], "missing", len(ctx["missing"]), "required", ctx["required_rank_receipts"])
