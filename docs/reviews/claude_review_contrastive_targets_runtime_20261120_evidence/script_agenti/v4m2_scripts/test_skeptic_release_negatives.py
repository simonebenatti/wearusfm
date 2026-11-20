"""Skeptic probes: minimal negative tests that the delivered suite lacks (not part of the package)."""
import copy
import json

import pytest

import contrastive_release as C
from test_contrastive_release import package  # noqa: F401 - fixture reuse


def _rewrite_review(package, text):
    package["review"].write_text(text)
    binding = copy.deepcopy(package["binding"])
    binding.update(review=C.pin(package["review"], relative=True), original_review_sha256=C.file_sha(package["review"]))
    return binding


def test_two_blocks_rejected(package):
    block = "```contrastive-release-review\n" + json.dumps(package["payload"]) + "\n```\n"
    binding = _rewrite_review(package, block + "\n" + block)
    with pytest.raises(ValueError, match="exactly one"):
        C.validate_review_binding(binding)


def test_rejected_status_rejected(package):
    payload = dict(package["payload"], status="REJECTED")
    binding = _rewrite_review(package, "```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    binding["review_payload"] = payload
    with pytest.raises(ValueError, match="not an approved"):
        C.validate_review_binding(binding)


def test_extra_key_rejected(package):
    payload = dict(package["payload"], conditions="audit only")
    binding = _rewrite_review(package, "```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    binding["review_payload"] = payload
    with pytest.raises(ValueError, match="not an approved"):
        C.validate_review_binding(binding)


def test_tampered_binding_payload_rejected(package):
    binding = copy.deepcopy(package["binding"])
    binding["review_payload"] = dict(binding["review_payload"], note="edited")
    with pytest.raises(ValueError, match="authenticated original"):
        C.validate_review_binding(binding)
