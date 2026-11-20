"""Equivalence probes for MU36 (git_bytes containment) and MU37 (numeric job_id).

argv[1]: clone at 0efa4d45 (unmodified on disk); argv[2]: scratch dir.
Loads contrastive_release twice from source text: original and mutated, in memory only.
"""
import hashlib
import json
import sys
import types
from pathlib import Path

root = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(root / "src"), str(root / "scripts")]
source = (root / "scripts/contrastive_release.py").read_text()

MU36 = (' or Path(relative).is_absolute() or ".." in Path(relative).parts:', ':')
MU37 = ('not isinstance(job_id, str) or not job_id.isdigit() or ', '')


def load(name, mutation=None):
    text = source if mutation is None else source.replace(*mutation)
    assert mutation is None or source.count(mutation[0]) == 1
    module = types.ModuleType(name)
    module.__file__ = str(root / "scripts/contrastive_release.py")
    exec(compile(text, module.__file__, "exec"), module.__dict__)
    return module


def outcome(fn, *args, **kw):
    try:
        fn(*args, **kw)
        return "ACCEPTED"
    except Exception as error:  # noqa: BLE001 - we classify every outcome
        return type(error).__name__ + ": " + str(error)[:90]


commit = "0efa4d45cdd3c314d2c0b28ffd8fdd14b9461aba"
orig, m36 = load("rel_orig"), load("rel_m36", MU36)
print("== MU36 git_bytes")
for rel in ["../clone/README.md", "../x", "/etc/passwd", str(root / "scripts/contrastive_release.py"), "scripts/../scripts/contrastive_release.py",
            "./../scripts/contrastive_release.py", "scripts/contrastive_release.py"]:
    a, b = outcome(orig.git_bytes, commit, rel), outcome(m36.git_bytes, commit, rel)
    print(f"{rel!r}: orig={a[:60]} | mut={b[:60]} | same_class={a.split(':')[0] == b.split(':')[0]}")

print("== MU37 numeric job_id")
m37 = load("rel_m37", MU37)
for module in (orig, m37):
    pre = work / "preflight.json"
    pre.write_text(json.dumps({"cells": [{"id": "new-audit", "mode": "audit"}]}))
    module.PREFLIGHT = pre
    module.source_digest = lambda: "a" * 64
audit = work / "audit.json"
audit.write_text(json.dumps({"source_sha256": "a" * 64, "passed": True}))
cases = {
    # realistic sacct: step rows have an empty TimelimitRaw (as in the historical 64652559 receipt)
    "realistic_steps": "JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n123|TIMEOUT|0:0|7300|8|120\n123.batch|COMPLETED|0:0|7300|8|\n123.extern|COMPLETED|0:0|7300|8|\n",
    # fixture-like step row carrying a time limit (what the unit fixture writes)
    "fixture_like_steps": "JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw\n123|TIMEOUT|0:0|7300|8|120\n123.batch|COMPLETED|0:0|7199|8|120\n",
}
for label, tsv_text in cases.items():
    tsv = work / (label + ".tsv")
    tsv.write_text(tsv_text)
    for job_id in ("123", "123.batch"):
        receipt = work / f"acc_{label}_{job_id}.json"
        receipt.write_text(json.dumps({"schema": orig.AUDIT_SCHEMA, "source_sha256": "a" * 64, "audit_sha256": orig.file_sha(audit),
                                       "logical_id": "new-audit", "job_id": job_id, "sacct": orig.pin(tsv)}))
        a = outcome(orig.validate_audit_accounting, audit, receipt)
        b = outcome(m37.validate_audit_accounting, audit, receipt)
        print(f"{label} job_id={job_id}: orig={a[:75]} | mut={b[:75]}")
