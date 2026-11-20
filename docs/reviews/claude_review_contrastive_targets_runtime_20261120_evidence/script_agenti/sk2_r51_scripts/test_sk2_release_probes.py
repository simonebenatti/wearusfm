"""Skeptic-2 negative probes for R5-1 (not part of the package; run against the clone)."""
import copy
import hashlib
import json

import pytest

import contrastive_release as C
from test_contrastive_release import dump, package  # noqa: F401  (fixture reuse)


def write_review(package, payload, *, prefix="# Actual review\n\n", extra=""):
    review = package["root"] / "review.md"
    review.write_text(prefix + "```contrastive-release-review\n" + json.dumps(payload) + "\n```\n" + extra)
    package["git"]("add", "review.md")
    package["git"]("commit", "-qm", "Probe review")
    return review, package["git"]("rev-parse", "HEAD")


def current_payload(package, **changes):
    payload = dict(package["payload"], science_plan_sha256=C.file_sha(package["science"]),
                   preflight_plan_sha256=C.file_sha(package["preflight"]), source_sha256=C.source_digest())
    payload.update(changes)
    return payload


def test_p01_status_review_required_is_rejected(package):
    review, commit = write_review(package, current_payload(package, status="REVIEW_REQUIRED"))
    with pytest.raises(ValueError, match="not an approved"):
        C.bind_review(commit, "review.md", review)


def test_p02_two_blocks_are_rejected(package):
    example = dict(package["payload"])
    block = "```contrastive-release-review\n" + json.dumps(example) + "\n```\n"
    review, commit = write_review(package, current_payload(package), extra="\nEsempio citato:\n\n" + block)
    with pytest.raises(ValueError, match="exactly one"):
        C.bind_review(commit, "review.md", review)


def test_p02b_approved_example_before_actual_review_required_block(package):
    example = "Formato (esempio):\n\n```contrastive-release-review\n" + json.dumps(current_payload(package)) + "\n```\n\n"
    review, commit = write_review(package, current_payload(package, status="REVIEW_REQUIRED"), prefix="# Review\n\n" + example)
    with pytest.raises(ValueError, match="exactly one"):
        C.bind_review(commit, "review.md", review)


def test_p03_binding_to_identical_plans_in_another_folder_is_rejected(package):
    other = package["root"] / "other"
    other.mkdir()
    (other / "science.json").write_bytes(package["science"].read_bytes())
    (other / "preflight.json").write_bytes(package["preflight"].read_bytes())
    package["git"]("add", "other")
    package["git"]("commit", "-qm", "Other cohort copy")
    reviewed = package["git"]("rev-parse", "HEAD")
    review, commit = write_review(package, current_payload(package, reviewed_commit=reviewed))
    binding = C.bind_review(commit, "review.md", review)
    binding["science_plan"] = C.pin(other / "science.json", relative=True)
    binding["preflight_plan"] = C.pin(other / "preflight.json", relative=True)
    with pytest.raises(ValueError, match="another cohort"):
        C.validate_review_binding(binding)


def test_p03b_review_of_another_cohort_cannot_release_this_one(package):
    other = package["root"] / "other"
    other.mkdir()
    science = dump(other / "science.json", {"input_sha256": {"manifest": "9" * 64}})
    preflight = dump(other / "preflight.json", {"source_sha256": C.source_digest(),
                     "science_plan": {"path": "other/science.json", "sha256": C.file_sha(science)},
                     "cells": [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200}]})
    package["git"]("add", "other")
    package["git"]("commit", "-qm", "Another cohort")
    reviewed = package["git"]("rev-parse", "HEAD")
    review, commit = write_review(package, current_payload(package, reviewed_commit=reviewed,
                                  science_plan_sha256=C.file_sha(science), preflight_plan_sha256=C.file_sha(preflight)))
    with pytest.raises(ValueError, match="another cohort"):
        C.bind_review(commit, "review.md", review, science_plan=science, preflight_plan=preflight)


def test_p04_declared_preflight_hash_must_match(package):
    review, commit = write_review(package, current_payload(package, preflight_plan_sha256="f" * 64))
    binding = copy.deepcopy(package["binding"])
    binding.update(review=C.pin(review, relative=True), review_payload=current_payload(package, preflight_plan_sha256="f" * 64),
                   original_review_sha256=C.file_sha(review))
    with pytest.raises(ValueError, match="current sources and both plans"):
        C.validate_review_binding(binding)


def test_p05_preflight_internal_source_pin_must_match(package):
    data = json.loads(package["preflight"].read_text())
    data["source_sha256"] = "e" * 64
    dump(package["preflight"], data)
    package["git"]("add", "preflight.json")
    package["git"]("commit", "-qm", "Stale internal source")
    review, commit = write_review(package, current_payload(package, reviewed_commit=package["git"]("rev-parse", "HEAD")))
    with pytest.raises(ValueError, match="preflight source/science-plan binding differs"):
        C.bind_review(commit, "review.md", review)


def test_p06_reviewed_commit_with_other_sources_and_same_plans_is_rejected(package):
    new_source = b"# reviewed execution, changed after review\n"
    digest = hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(new_source).digest()).hexdigest()
    data = json.loads(package["preflight"].read_text())
    data["source_sha256"] = digest
    dump(package["preflight"], data)
    package["git"]("add", "preflight.json")
    package["git"]("commit", "-qm", "Plans for the new source, old source still committed")
    reviewed = package["git"]("rev-parse", "HEAD")
    package["source"].write_bytes(new_source)
    package["git"]("add", "scripts/contrastive_one.py")
    package["git"]("commit", "-qm", "New source")
    review, commit = write_review(package, current_payload(package, reviewed_commit=reviewed))
    with pytest.raises(ValueError, match="execution sources differ from reviewed commit"):
        C.bind_review(commit, "review.md", review)


def test_p06_generic_reject_only(package):
    """Same as p06 without message: distinguishes redundant lines 83-84/86."""
    new_source = b"# reviewed execution, changed after review\n"
    digest = hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(new_source).digest()).hexdigest()
    data = json.loads(package["preflight"].read_text())
    data["source_sha256"] = digest
    dump(package["preflight"], data)
    package["git"]("add", "preflight.json")
    package["git"]("commit", "-qm", "Plans for the new source, old source still committed")
    reviewed = package["git"]("rev-parse", "HEAD")
    package["source"].write_bytes(new_source)
    package["git"]("add", "scripts/contrastive_one.py")
    package["git"]("commit", "-qm", "New source")
    review, commit = write_review(package, current_payload(package, reviewed_commit=reviewed))
    with pytest.raises(ValueError):
        C.bind_review(commit, "review.md", review)


def test_p07_reviewed_commit_with_other_plan_bytes_is_rejected(package):
    reviewed = package["payload"]["reviewed_commit"]
    data = json.loads(package["preflight"].read_text())
    data["note"] = "changed after the reviewed commit"
    dump(package["preflight"], data)
    package["git"]("add", "preflight.json")
    package["git"]("commit", "-qm", "Plan changed after review")
    review, commit = write_review(package, current_payload(package, reviewed_commit=reviewed))
    with pytest.raises(ValueError, match="reviewed plan bytes differ"):
        C.bind_review(commit, "review.md", review)


def test_p08a_binding_payload_copy_altered(package):
    binding = copy.deepcopy(package["binding"])
    binding["review_payload"]["status"] = "ALTERED"
    with pytest.raises(ValueError, match="review copy differs"):
        C.validate_review_binding(binding)


def test_p08b_binding_original_sha_altered(package):
    binding = copy.deepcopy(package["binding"])
    binding["original_review_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="review copy differs"):
        C.validate_review_binding(binding)


def test_p09_make_job_rejects_invalid_binding(package):
    binding = copy.deepcopy(package["binding"])
    binding["original_review_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        C.make_job(binding, "new-audit", {}, "/remote/new-output")


def test_p10_make_job_rejects_inputs_outside_science_plan(package):
    with pytest.raises(ValueError, match="input pins differ"):
        C.make_job(package["binding"], "new-audit", {"inputs": {"manifest": {"path": "/x", "sha256": "2" * 64}}}, "/remote/o")


def test_p11_review_copy_outside_checkout_is_rejected(package, tmp_path):
    outside = tmp_path / "outside_review.md"
    outside.write_bytes(package["review"].read_bytes())
    binding = copy.deepcopy(package["binding"])
    binding["review"] = C.pin(outside)
    with pytest.raises(ValueError, match="inside the frozen checkout"):
        C.validate_review_binding(binding)


def test_p12_abbreviated_review_commit_in_bind(package):
    with pytest.raises(ValueError):
        C.bind_review(package["binding"]["review_commit"][:12], "review.md", package["review"])


def test_p12_specific_message(package):
    with pytest.raises(ValueError, match="full commit and contained repository path required"):
        C.bind_review(package["binding"]["review_commit"][:12], "review.md", package["review"])


def test_p13_extra_payload_key_is_rejected(package):
    review, commit = write_review(package, current_payload(package, comment="extra"))
    with pytest.raises(ValueError, match="not an approved"):
        C.bind_review(commit, "review.md", review)


def test_p14_bad_hash_format_generic(package):
    review, commit = write_review(package, current_payload(package, science_plan_sha256="XYZ"))
    with pytest.raises(ValueError):
        C.bind_review(commit, "review.md", review)


def test_p14_bad_hash_format_specific(package):
    review, commit = write_review(package, current_payload(package, science_plan_sha256="XYZ"))
    with pytest.raises(ValueError, match="exact package hash"):
        C.bind_review(commit, "review.md", review)


def test_p15_binding_with_abbreviated_review_commit(package):
    binding = copy.deepcopy(package["binding"])
    binding["review_commit"] = binding["review_commit"][:12]
    with pytest.raises(ValueError, match="full commit"):
        C.validate_review_binding(binding)
