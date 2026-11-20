"""Skeptic R5-1: throwaway negative tests (outside the clone) built on the receipt's `package` fixture.

They must pass on the original code and fail when the corresponding barrier is disabled.
"""
import json
import shutil

import pytest

from test_contrastive_release import C, dump, package  # noqa: F401  (fixture re-export)


def _commit_review(package, payload, *, extra=""):
    package["review"].write_text("# Review variant\n\n" + extra + "```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    package["git"]("add", "review.md")
    package["git"]("commit", "-qm", "review variant")
    return package["git"]("rev-parse", "HEAD")


def test_sk1_status_must_be_approved(package):
    commit = _commit_review(package, dict(package["payload"], status="REVIEW_REQUIRED"))
    with pytest.raises(ValueError, match="approved target-cohort"):
        C.bind_review(commit, "review.md", package["review"])


def test_sk1_two_blocks_are_rejected(package):
    quoted = "````text\n```contrastive-release-review\n" + json.dumps(package["payload"]) + "\n```\n````\n\n"
    commit = _commit_review(package, package["payload"], extra=quoted)
    with pytest.raises(ValueError, match="exactly one"):
        C.bind_review(commit, "review.md", package["review"])


def test_sk1_declared_preflight_hash_must_match(package):
    commit = _commit_review(package, dict(package["payload"], preflight_plan_sha256="e" * 64))
    with pytest.raises(ValueError, match="current sources and both plans"):
        C.bind_review(commit, "review.md", package["review"])


def test_sk1_plans_of_another_cohort_are_rejected(package):
    root = package["root"]
    other = root / "other"
    other.mkdir()
    science = other / "science.json"
    shutil.copyfile(package["science"], science)
    pre = json.loads(package["preflight"].read_text())
    pre["science_plan"] = {"path": "other/science.json", "sha256": C.file_sha(science)}
    pre["cells"][0]["id"] = "other-audit"
    preflight = dump(other / "preflight.json", pre)
    package["git"]("add", "other")
    package["git"]("commit", "-qm", "another cohort with the same sources")
    code = package["git"]("rev-parse", "HEAD")
    payload = dict(package["payload"], reviewed_commit=code, science_plan_sha256=C.file_sha(science),
                   preflight_plan_sha256=C.file_sha(preflight))
    commit = _commit_review(package, payload)
    with pytest.raises(ValueError, match="another cohort"):
        C.bind_review(commit, "review.md", package["review"], science_plan=science, preflight_plan=preflight)
