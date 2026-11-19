"""Apply the two barrier-removal mutations (independent re-implementation).

MD: validate_audit no longer compares audit config_sha256 with the calibration recipe.
MG: read_preflight no longer compares audit config_sha256 with the calibration recipe.
Usage: python mutate.py <root> MD,MG
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
wanted = set(sys.argv[2].split(","))
MUTATIONS = {
    "MD": ("scripts/contrastive_run.py",
           '    if (report.get("steps") != 1024 or report.get("world_size") != 4 or report.get("workers_per_rank") != 6\n'
           '            or report.get("config_sha256") != hashlib.sha256(json.dumps(calibration_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()):\n',
           '    if (report.get("steps") != 1024 or report.get("world_size") != 4 or report.get("workers_per_rank") != 6):\n'),
    "MG": ("scripts/contrastive_benchmark_readout.py",
           "            audit.get('workers_per_rank') != 6 or audit.get('config_sha256') != canonical_calibration):\n",
           "            audit.get('workers_per_rank') != 6):\n"),
}
for name in sorted(wanted):
    rel, old, new = MUTATIONS[name]
    path = root / rel
    text = path.read_text()
    assert text.count(old) == 1, (name, text.count(old))
    path.write_text(text.replace(old, new))
    print("applied", name, rel)
