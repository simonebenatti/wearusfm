"""Synthetic review commit + binding inside a TEMPORARY clone (never the reference copy)."""
import json
import subprocess
import sys
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(CLONE / "scripts"))
import contrastive_release as C  # noqa: E402

assert C.ROOT == CLONE, (C.ROOT, CLONE)
pin = json.loads((CLONE / "results/infra/contrastive_targets_20261118/amendment_20261119/delivery_pin.json").read_text())
payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": pin["content_commit"],
           "source_sha256": pin["source_sha256"], "science_plan_sha256": pin["science_plan_sha256"],
           "preflight_plan_sha256": pin["preflight_plan_sha256"]}
rel = "docs/reviews/vn3_synthetic_release_review.md"
review = CLONE / rel
review.write_text("# Synthetic review (scettico N3, solo clone temporaneo)\n\n```contrastive-release-review\n"
                  + json.dumps(payload, indent=2) + "\n```\n")


def git(*args):
    return subprocess.check_output(["git", "-c", "user.name=vn3", "-c", "user.email=vn3@example.invalid", *args],
                                   cwd=CLONE).decode().strip()


git("add", rel)
git("commit", "-qm", "synthetic review (temporary clone only)")
commit = git("rev-parse", "HEAD")
binding = C.bind_review(commit, rel, review)
out = CLONE / "results/infra/contrastive_targets_20261118/vn3_binding.json"
C.write_new(out, binding)
print("review commit", commit)
print("binding review pin", binding["review"])
print("binding science pin", binding["science_plan"]["path"], "preflight pin", binding["preflight_plan"]["path"])
print("source_digest", C.source_digest() == pin["source_sha256"])
