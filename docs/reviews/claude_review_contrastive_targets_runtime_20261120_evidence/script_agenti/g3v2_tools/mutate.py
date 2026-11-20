"""Apply exact-string mutations to a temporary clone; fail if any pattern is not found exactly once."""
import sys
from pathlib import Path

MUT = {
    "A": [  # review binding and review comparison in dispatcher and runtime
        ("scripts/contrastive_job.py",
         "    validate_review_binding(job.get('package_review'))\n"
         "    if job.get('review') != job['package_review'].get('review'):\n"
         "        raise ValueError('job review differs from its bound package review')\n",
         "    pass  # MUT-A\n"),
        ("scripts/contrastive_run.py",
         "    from contrastive_release import validate_review_binding\n"
         "    validate_review_binding(release.get(\"package_review\"))\n"
         "    if release[\"review_pin\"] != release[\"package_review\"][\"review_commit\"]:\n"
         "        raise ValueError(\"release review commit differs from its bound package review\")\n",
         "    pass  # MUT-A\n"),
    ],
    "B": [  # three audit-accounting calls
        ("scripts/contrastive_job.py",
         "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))\n",
         "        pass  # MUT-B\n"),
        ("scripts/contrastive_run.py",
         "    validate_audit_accounting(audit_path, release[\"audit_sacct\"])\n",
         "    pass  # MUT-B\n"),
        ("scripts/contrastive_release.py",
         "        validate_audit_accounting(audit, job[\"audit_sacct\"])\n",
         "        pass  # MUT-B\n"),
    ],
}

root = Path(sys.argv[1])
for name in sys.argv[2:]:
    for rel, old, new in MUT[name]:
        path = root / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"pattern count {text.count(old)} in {rel} for {name}")
        path.write_text(text.replace(old, new))
        print("mutated", name, rel)
