"""r7: provenienza della review nel binding (solo lettura del codice a fbfde797)."""
import copy, hashlib, json, subprocess, sys
from pathlib import Path
import pytest
import contrastive_release as C


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


@pytest.fixture
def package(tmp_path, monkeypatch):
    # stessa fixture di tests/cpu/test_contrastive_release.py
    root = tmp_path / "repo"; root.mkdir()
    source = root / "scripts/contrastive_one.py"; source.parent.mkdir(); source.write_text("# reviewed execution\n")
    digest = hashlib.sha256(b"scripts/contrastive_one.py\0" + hashlib.sha256(source.read_bytes()).digest()).hexdigest()
    science = dump(root / "science.json", {"input_sha256": {"manifest": "1" * 64}})
    preflight = dump(root / "preflight.json", {"source_sha256": digest, "science_plan": {"path": "science.json", "sha256": C.file_sha(science)},
                     "cells": [{"id": "new-audit", "mode": "audit", "arm": None, "seconds": 7200}]})
    monkeypatch.setattr(C, "ROOT", root); monkeypatch.setattr(C, "PLAN", science); monkeypatch.setattr(C, "PREFLIGHT", preflight)
    monkeypatch.setattr(C, "source_paths", lambda: [source])
    monkeypatch.setattr(C, "source_digest", lambda: digest)
    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", *args], cwd=root, stderr=subprocess.PIPE).decode().strip()
    git("init", "-q"); git("add", "."); git("commit", "-qm", "Reviewed package")
    code = git("rev-parse", "HEAD")
    payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": code, "source_sha256": digest,
               "science_plan_sha256": C.file_sha(science), "preflight_plan_sha256": C.file_sha(preflight)}
    return {"root": root, "payload": payload, "git": git, "code": code}


def test_dispatcher_binding_accepts_nonexistent_review_commit(package):
    """validate_review_binding (dispatcher, runtime, make_job) non rilegge il commit della review."""
    root, payload = package["root"], package["payload"]
    selfmade = root / "docs/any_author_note.md"
    selfmade.parent.mkdir(parents=True, exist_ok=True)
    selfmade.write_text("```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    forged = {"schema": C.BINDING_SCHEMA, "review_commit": "f" * 40,  # commit inesistente
              "original_review_path": "nowhere.md", "original_review_sha256": C.file_sha(selfmade),
              "review": C.pin(selfmade, relative=True), "review_payload": payload,
              "science_plan": C.pin(C.PLAN, relative=True), "preflight_plan": C.pin(C.PREFLIGHT, relative=True)}
    assert C.validate_review_binding(forged)["status"] == "APPROVED"
    job = C.make_job(forged, "new-audit", {}, "/remote/out")
    assert job["status"] == "APPROVED_FOR_PREFLIGHT"


def test_bind_review_accepts_any_file_in_any_commit(package):
    """bind-review verifica solo i byte al commit indicato: nessun vincolo su autore, ramo o percorso della review."""
    root, payload, git = package["root"], package["payload"], package["git"]
    note = root / "docs/gpt_self_note.md"; note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# nota dell'autore del codice\n```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")
    git("add", "."); git("-c", "user.name=GPT", "commit", "-qm", "self note")
    binding = C.bind_review(git("rev-parse", "HEAD"), "docs/gpt_self_note.md", note)
    assert binding["review_payload"]["status"] == "APPROVED"


def test_two_blocks_in_review_fail_closed(package):
    """Una review che cita anche l'esempio del prompt (secondo blocco) e' rifiutata: fallimento sicuro."""
    payload = package["payload"]
    text = ("```contrastive-release-review\n" + json.dumps(payload) + "\n```\n"
            "````text\n```contrastive-release-review\n{\"schema\": \"x\"}\n```\n````\n")
    with pytest.raises(ValueError, match="exactly one"):
        C.review_payload(text)


def test_extra_key_in_block_is_rejected(package):
    """Il blocco non puo' contenere campi in piu' (es. ambito «solo audit» o revisore)."""
    payload = dict(package["payload"], scope="audit_only")
    with pytest.raises(ValueError, match="not an approved"):
        C.review_payload("```contrastive-release-review\n" + json.dumps(payload) + "\n```\n")


