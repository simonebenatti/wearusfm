"""Offline integrity and isolation checks for the external SEEDS downloader."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest

SPEC = importlib.util.spec_from_file_location("seeds_download", Path(__file__).parents[2] / "scripts/seeds_download.py")
seeds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seeds)


def entry(payload=b"mat-data", path="subj01/trial.mat", **extra):
    return {"path": path, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "md5": hashlib.md5(payload).hexdigest(), "url": "https://osf.io/download/abc12/", **extra}


def manifest(tmp_path, files):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "dataset": "SEEDS", "source_node": "wa3qk",
                                "pretraining_allowed": False, "files": files}))
    return path


class Response(io.BytesIO):
    def getcode(self):
        return 200

    def geturl(self):
        return "https://storage.example.org/object"


def invoke(tmp_path, files, **kwargs):
    return seeds.run(manifest(tmp_path, files), tmp_path / "external_benchmarks/seeds",
                     tmp_path / "receipt.json", max_bytes=kwargs.pop("max_bytes", 10000),
                     free_margin_bytes=0, **kwargs)


def test_verified_download_and_resume_existing_without_network(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: calls.append(1) or Response(b"mat-data"))
    report = invoke(tmp_path, [entry()])
    assert report["status"] == "complete" and report["verified_files"] == 1
    target = tmp_path / "external_benchmarks/seeds/subj01/trial.mat"
    assert target.read_bytes() == b"mat-data"
    assert not target.with_suffix(".mat.part").exists()
    assert json.loads((target.parents[1] / seeds.POLICY_NAME).read_text())["pretraining_allowed"] is False
    second = seeds.run(tmp_path / "manifest.json", target.parents[1], tmp_path / "receipt2.json",
                       max_bytes=10000, free_margin_bytes=0)
    assert len(calls) == 1
    assert second["files"][0]["status"] == "verified_existing"
    assert second["network_bytes"] == 0


@pytest.mark.parametrize("content,extra", [(b"bad-data", {}), (b"mat-data", {"md5": "0" * 32}), (b"too long data", {})])
def test_corruption_has_no_retry_and_never_publishes(tmp_path, monkeypatch, content, extra):
    calls = []
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: calls.append(1) or Response(content))
    report = invoke(tmp_path, [entry(**extra)])
    assert report["status"] == "failed" and len(calls) == 1
    assert not (tmp_path / "external_benchmarks/seeds/subj01/trial.mat").exists()
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "a//b", "a/./b", "a\\b", "a.part", seeds.POLICY_NAME])
def test_unsafe_manifest_paths_rejected_before_download(tmp_path, monkeypatch, path):
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network called"))
    with pytest.raises(ValueError):
        invoke(tmp_path, [entry(path=path)])
    assert not (tmp_path / "external_benchmarks").exists()


@pytest.mark.parametrize("paths", [["subj01/a", "subj01/a"], ["subj01/A", "subj01/a"],
                                   ["subj01", "subj01/file"], ["SUBJ01", "subj01/file"]])
def test_duplicate_or_directory_collision_rejected(tmp_path, paths):
    with pytest.raises(ValueError, match="duplicate|collision"):
        invoke(tmp_path, [entry(path=p) for p in paths])


@pytest.mark.parametrize("linked", ["subj01", "subj01/trial.mat", "subj01/trial.mat.part"])
def test_symlink_refused_and_external_file_untouched(tmp_path, monkeypatch, linked):
    root = tmp_path / "external_benchmarks/seeds"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "original"
    external.write_bytes(b"keep")
    link = root / linked
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside if linked == "subj01" else external)
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network called"))
    with pytest.raises(ValueError, match="symlink"):
        invoke(tmp_path, [entry()])
    assert external.read_bytes() == b"keep"


def test_bad_completed_file_preserved(tmp_path, monkeypatch):
    target = tmp_path / "external_benchmarks/seeds/subj01/trial.mat"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"damaged")
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network called"))
    with pytest.raises(seeds.IntegrityError):
        invoke(tmp_path, [entry()])
    assert target.read_bytes() == b"damaged"


def test_transport_retries_bounded_but_http_404_is_final(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(seeds.time, "sleep", lambda _: None)

    def transport(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError("network unavailable")

    monkeypatch.setattr(seeds.urllib.request, "urlopen", transport)
    report = invoke(tmp_path, [entry()])
    assert len(calls) == 3 and report["status"] == "failed"
    calls.clear()

    def missing(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://osf.io/download/abc12/", 404, "missing", {}, None)

    monkeypatch.setattr(seeds.urllib.request, "urlopen", missing)
    report = seeds.run(tmp_path / "manifest.json", tmp_path / "external_benchmarks/seeds",
                       tmp_path / "missing.json", max_bytes=1000, free_margin_bytes=0)
    assert len(calls) == 1 and report["status"] == "failed"


def test_subset_is_explicit_and_size_budget_is_before_network(tmp_path, monkeypatch):
    files = [entry(), entry(path="subj02/trial.mat"), entry(path="functions/read.py")]
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: Response(b"mat-data"))
    with pytest.raises(ValueError, match="max-bytes"):
        invoke(tmp_path, files, max_bytes=1)
    report = invoke(tmp_path, files, subjects={2})
    assert report["selected_files"] == 1 and report["whole_manifest_selected"] is False
    assert report["subjects_filter"] == [2]
    assert report["files"][0]["path"] == "subj02/trial.mat"


def test_pretraining_root_and_marker_changes_forbidden(tmp_path):
    path = manifest(tmp_path, [entry()])
    with pytest.raises(ValueError, match="external_benchmarks/seeds"):
        seeds.run(path, tmp_path / "data/processed/seeds", tmp_path / "r.json", max_bytes=1000)
    root = tmp_path / "external_benchmarks/seeds"
    root.mkdir(parents=True)
    (root / seeds.POLICY_NAME).write_text('{"pretraining_allowed":true}')
    with pytest.raises(ValueError, match="policy differs"):
        invoke(tmp_path, [entry()])


def test_insufficient_space_rejects_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(seeds.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network called"))
    with pytest.raises(ValueError, match="insufficient space"):
        invoke(tmp_path, [entry()])


def test_existing_part_restarts_and_existing_receipt_is_not_overwritten(tmp_path, monkeypatch):
    part = tmp_path / "external_benchmarks/seeds/subj01/trial.mat.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"old partial bytes")
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: Response(b"mat-data"))
    report = invoke(tmp_path, [entry()])
    assert report["status"] == "complete" and not part.exists()
    receipt = (tmp_path / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        invoke(tmp_path, [entry()])
    assert (tmp_path / "receipt.json").read_bytes() == receipt


def test_atomic_publication_never_overwrites_concurrent_target(tmp_path, monkeypatch):
    original = seeds.os.link

    def racing_link(source, destination):
        destination.write_bytes(b"other writer")
        return original(source, destination)

    monkeypatch.setattr(seeds.os, "link", racing_link)
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: Response(b"mat-data"))
    report = invoke(tmp_path, [entry()])
    assert report["status"] == "failed"
    assert (tmp_path / "external_benchmarks/seeds/subj01/trial.mat").read_bytes() == b"other writer"


def test_receipt_cannot_collide_with_payload(tmp_path):
    root = tmp_path / "external_benchmarks/seeds"
    with pytest.raises(ValueError, match="outside"):
        seeds.run(manifest(tmp_path, [entry()]), root, root / "subj01/trial.mat", max_bytes=1000)


@pytest.mark.parametrize("url", ["http://osf.io/download/abc12/", "https://other.example/download/abc12/",
                                 "https://osf.io@evil.test/download/abc12/", "https://osf.io/download/abc12/?extra=x"])
def test_manifest_requires_official_url(tmp_path, url):
    with pytest.raises(ValueError, match="official OSF"):
        invoke(tmp_path, [entry(url=url)])


def test_parallel_workers_record_all_completed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: Response(b"mat-data"))
    files = [entry(path=f"subj01/trial{i}.mat") for i in range(8)]
    report = invoke(tmp_path, files, workers=4)
    assert report["verified_files"] == 8 and report["status"] == "complete"
    assert len({x["path"] for x in report["files"]}) == 8
