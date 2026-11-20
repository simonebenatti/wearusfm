"""The 10 mutations the reviewer reports as killed, re-expressed independently (sk1r63)."""
import json, sys
from pathlib import Path
B = "scripts/contrastive_benchmark_readout.py"; RN = "scripts/contrastive_run.py"; PR = "scripts/contrastive_prepare.py"
M = [
 ("B03", B, "numeric context not PASSED: early return removed",
  "        # unqualified or mixed numerical context.\n        return result\n", "        # unqualified or mixed numerical context.\n        pass\n"),
 ("B06", B, "reload read from update 1", "first_update=545, expected_phase", "first_update=1, expected_phase"),
 ("B07", B, "parity binding raise removed", "            raise ValueError('zero-weight parity receipt is not bound to the full control run')", "            pass"),
 ("B08", B, "missing parity not inconclusive", "        result['inconclusive_reasons'].append('missing required production zero-weight profile')", "        pass"),
 ("B09", B, "parity non-PASSED ignored", "        elif status != 'PASSED':", "        elif False:"),
 ("B10", B, "read_preflight: accounting call removed",
  "    accounting = validate_audit_accounting(cpu_audit_path, cpu_audit_sacct_path, expected_source=cohort['source_sha256'])", "    accounting = {}"),
 ("B15", B, "_run_evidence: receipt phase not compared", "                row.get('phase') != expected_phase):", "                False):"),
 ("R12", RN, "validate_cohort: scientific keep reset removed",
  "        if raw[\"max_steps\"] in (5000, 19618):\n            split_expected[\"schedule\"][\"keep_checkpoints\"] = [5000, 19618]\n", ""),
 ("P01", PR, "prepare: 544 not added for 544/576", "        if endpoint in (544, 576):\n            cfg[\"schedule\"]", "        if endpoint in ():\n            cfg[\"schedule\"]"),
 ("P02", PR, "prepare main: keep reset removed",
  "                    if endpoint in (5000, 19618):\n                        candidate[\"schedule\"][\"keep_checkpoints\"] = [5000, 19618]\n", ""),
]
clone, out = Path(sys.argv[1]), Path(sys.argv[2]); rows = []
for mid, f, desc, old, new in M:
    text = (clone / f).read_text(); n = text.count(old)
    if n != 1: raise SystemExit(f"{mid}: {n}")
    rows.append({"id": mid, "file": f, "line": text[:text.index(old)].count("\n") + 1, "desc": desc, "old": old, "new": new})
out.write_text(json.dumps(rows, indent=1)); print(len(rows), "ok")
