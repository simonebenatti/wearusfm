"""Skeptic-2: validate_review_binding on the REAL package in a disposable clone (no commits).

usage: python -B real_package_probe.py <clone>
Writes an untracked review copy at the clone root, builds bindings by hand (bind_review would need a
review commit) and removes the file at the end.
"""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

clone = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(clone / "scripts"))
import contrastive_release as R  # noqa: E402

assert Path(R.__file__).resolve().is_relative_to(clone), R.__file__
pin = json.loads((clone / "results/infra/contrastive_targets_20261118/amendment_20261119/delivery_pin.json").read_text())
base = {"schema": R.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": pin["content_commit"],
        "source_sha256": pin["source_sha256"], "science_plan_sha256": pin["science_plan_sha256"],
        "preflight_plan_sha256": pin["preflight_plan_sha256"]}
print("source_digest() == delivery pin:", R.source_digest() == pin["source_sha256"])
review = clone / "sk2_review_probe.md"


def block(payload):
    return "```contrastive-release-review\n" + json.dumps(payload, indent=2) + "\n```\n"


def binding_for(text):
    review.write_text(text)
    try:
        declared = R.review_payload(text)
    except Exception:
        declared = None
    return {"schema": R.BINDING_SCHEMA, "review_commit": "0" * 40, "original_review_path": "x.md",
            "original_review_sha256": R.file_sha(review), "review": R.pin(review, relative=True),
            "review_payload": declared, "science_plan": R.pin(R.PLAN, relative=True),
            "preflight_plan": R.pin(R.PREFLIGHT, relative=True)}


def attempt(name, text, mutate=None):
    binding = binding_for(text)
    if mutate:
        mutate(binding)
    try:
        R.validate_review_binding(binding)
        print(f"{name}: ACCEPTED")
    except Exception as error:
        print(f"{name}: {type(error).__name__}: {str(error)[:90]}")


try:
    attempt("valid APPROVED fbfde797", block(base))
    attempt("valid APPROVED 0efa4d45", block(dict(base, reviewed_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip())))
    attempt("status REVIEW_REQUIRED", block(dict(base, status="REVIEW_REQUIRED")))
    attempt("example + real block", "Esempio:\n" + block(base) + "\nVerdetto:\n" + block(base))
    attempt("reviewed_commit base 438c8d65", block(dict(base, reviewed_commit=pin["base_commit"])))
    attempt("declared preflight hash wrong", block(dict(base, preflight_plan_sha256="f" * 64)))
    attempt("binding payload altered", block(base), lambda b: b["review_payload"].update(status="X"))
    attempt("binding original sha altered", block(base), lambda b: b.update(original_review_sha256="0" * 64))
    attempt("binding review_commit abbreviated", block(base), lambda b: b.update(review_commit="0" * 12))
finally:
    review.unlink(missing_ok=True)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=clone, text=True).strip()
    print("clone clean after probe:", status == "")
