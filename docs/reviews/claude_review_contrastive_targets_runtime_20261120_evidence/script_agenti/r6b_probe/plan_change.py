"""r6 (d): a plan edited after the review is noticed by the review binding (verify_git=False isolates the hash gate)."""
import json, sys
from pathlib import Path
import contrastive_release as REL
import contrastive_prepare as P
root = Path(sys.argv[1])
review = root / "docs/reviews/r6_probe_review.md"
payload = {"schema": REL.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": "0efa4d45cdd3c314d2c0b28ffd8fdd14b9461aba",
           "source_sha256": P.source_digest(), "science_plan_sha256": REL.file_sha(P.PLAN),
           "preflight_plan_sha256": REL.file_sha(REL.PREFLIGHT)}
review.write_text("x\n```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
binding = {"schema": REL.BINDING_SCHEMA, "review_commit": payload["reviewed_commit"],
           "review": REL.pin(review, relative=True), "science_plan": REL.pin(P.PLAN, relative=True),
           "preflight_plan": REL.pin(REL.PREFLIGHT, relative=True), "review_payload": payload,
           "original_review_sha256": REL.file_sha(review)}
print("unchanged plan accepted:", REL.validate_review_binding(binding, verify_git=False)["status"])
plan = json.loads(REL.PREFLIGHT.read_text()); plan["cells"][3]["seconds"] = 3600
REL.PREFLIGHT.write_text(json.dumps(plan, indent=2) + "\n")
try:
    REL.validate_review_binding(binding, verify_git=False); print("modified plan accepted: True")
except ValueError as e:
    print("modified plan rejected:", e)
