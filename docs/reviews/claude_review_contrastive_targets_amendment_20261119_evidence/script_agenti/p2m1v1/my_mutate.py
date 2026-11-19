"""Independent mutations for skeptic check of P2-M1 (exact, single-occurrence replacements)."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
which = set(sys.argv[2].split(","))

MUT = {
    # X1: validate_audit ignores the audit's config_sha256 (recipe barrier removed)
    "X1": ("scripts/contrastive_run.py",
           '            or report.get("config_sha256") != hashlib.sha256(json.dumps(calibration_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()):\n',
           '            or False):\n'),
    # X2: read_preflight ignores the audit's config_sha256
    "X2": ("scripts/contrastive_benchmark_readout.py",
           "            audit.get('workers_per_rank') != 6 or audit.get('config_sha256') != canonical_calibration):\n",
           "            audit.get('workers_per_rank') != 6):\n"),
}

for key in sorted(which):
    rel, old, new = MUT[key]
    path = root / rel
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{key}: expected exactly one occurrence in {rel}, found {n}")
    path.write_text(text.replace(old, new))
    print(f"{key}: applied to {rel}")
