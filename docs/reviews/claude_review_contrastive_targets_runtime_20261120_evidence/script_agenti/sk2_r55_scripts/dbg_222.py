import json, sys
from pathlib import Path
import contrastive_benchmark_readout as B
from test_contrastive_targets_readout import numeric_folder, write
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=False)
cfg, provenance, cohort = numeric_folder(root)
p1 = root / "numeric_profile.rank1.json"
row = json.loads(p1.read_text())
print("before keys", len(row["flags"]), {k: v for k, v in row["flags"].items() if v is None})
del row["flags"]["cublas_workspace_config"]
write(p1, row)
r0 = json.loads((root / "numeric_profile.rank0.json").read_text())["flags"]
r1 = json.loads(p1.read_text())["flags"]
print("rank0 has key", "cublas_workspace_config" in r0, "rank1 has key", "cublas_workspace_config" in r1, "dict differ", r0 != r1)
try:
    B._run_evidence(root, "CN", cohort, cfg, provenance)
    print("RUN_EVIDENCE: no exception")
except Exception as exc:
    print("RUN_EVIDENCE:", type(exc).__name__, exc)
