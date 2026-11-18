"""Offline OSF fixtures exercise traversal across pages, folders and components."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("seeds_inventory", Path(__file__).parents[2] / "scripts/seeds_inventory.py")
seeds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seeds)
API = "https://api.osf.io/v2/"
LICENSE = API + "licenses/ccby/"


def first(url):
    return url + "?page%5Bsize%5D=100"


def page(rows, next_url=None):
    return {"data": rows, "links": {"next": next_url}}


def folder(name, node, identifier):
    return {"id": identifier, "attributes": {"name": name, "kind": "folder"},
            "relationships": {"files": {"links": {"related": {
                "href": API + f"nodes/{node}/files/osfstorage/{identifier}/"}}}}}


def file(name, identifier, url="https://osf.io/download/abc12/"):
    data = identifier.encode()
    return {"id": identifier,
            "attributes": {"name": name, "kind": "file", "size": len(data), "current_version": 1,
                           "extra": {"hashes": {"sha256": hashlib.sha256(data).hexdigest(),
                                                "md5": hashlib.md5(data).hexdigest()}}},
            "links": {"download": url}}


def fixture_pages():
    children2 = API + "nodes/wa3qk/children/?page=2"
    pages = {
        first(API + "nodes/wa3qk/children/"): page([{"id": "e7m5k"}], children2),
        children2: page([{"id": "yp4vq"}]),
        LICENSE: {"data": {"attributes": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}}},
    }
    for node, subject in [("wa3qk", "subj01"), ("e7m5k", "subj11"), ("yp4vq", "subj21")]:
        pages[API + f"nodes/{node}/"] = {
            "data": {"relationships": {"license": {"links": {"related": {"href": LICENSE}}}}}}
        root_url = API + f"nodes/{node}/files/osfstorage/"
        pages[first(API + f"nodes/{node}/files/")] = page([
            {"attributes": {"name": "osfstorage"},
             "relationships": {"files": {"links": {"related": {"href": root_url}}}}}])
        pages[first(root_url)] = page([folder(subject, node, subject)])
        subject_url = API + f"nodes/{node}/files/osfstorage/{subject}/"
        session_url = API + f"nodes/{node}/files/osfstorage/{subject}-session/"
        pages[first(subject_url)] = page([folder("session1", node, subject + "-session")])
        pages[first(session_url)] = page([file("trial1.mat", subject + "-trial1")])
    session_url = API + "nodes/wa3qk/files/osfstorage/subj01-session/"
    second_trials = session_url + "?page=2"
    pages[first(session_url)]["links"]["next"] = second_trials
    pages[second_trials] = page([file("trial2.mat", "subj01-trial2")])
    root = pages[first(API + "nodes/wa3qk/files/osfstorage/")]["data"]
    root.extend([folder("functions", "wa3qk", "helpers"), folder("figures", "wa3qk", "figures")])
    pages[first(API + "nodes/wa3qk/files/osfstorage/helpers/")] = page([file("load.py", "helper")])
    return pages


def mocked_api(monkeypatch, pages):
    seen = []

    def open_url(request, timeout):
        seen.append(request.full_url)
        assert request.full_url in pages, f"Unexpected request: {request.full_url}"
        return io.BytesIO(json.dumps(pages[request.full_url]).encode())

    monkeypatch.setattr(seeds.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(seeds.time, "sleep", lambda _: None)
    return seen


def test_inventory_preserves_all_components_pages_and_nested_paths(tmp_path, monkeypatch):
    pages = fixture_pages()
    seen = mocked_api(monkeypatch, pages)
    result = seeds.inventory(tmp_path / "api")
    assert set(result["nodes"]) == {"wa3qk", "e7m5k", "yp4vq"}
    expected = {"subj01/session1/trial1.mat", "subj01/session1/trial2.mat",
                "subj11/session1/trial1.mat", "subj21/session1/trial1.mat", "functions/load.py"}
    assert {f["path"] for f in result["files"]} == expected
    assert result["subjects"] == ["subj01", "subj11", "subj21"]
    assert {f["source_node"] for f in result["files"]} == {"wa3qk", "e7m5k", "yp4vq"}
    assert result["pretraining_allowed"] is False
    assert result["total_bytes"] == sum(f["size"] for f in result["files"])
    assert {item["name"] for item in result["excluded_non_signal_folders"]} == {"figures"}
    assert len(result["licenses"]) == 3
    assert any("children/?page=2" in u for u in seen)
    assert any("subj01-session/?page=2" in u for u in seen)
    assert not any("/figures/" in u for u in seen)
    assert len(list((tmp_path / "api").glob("*.json"))) == len(seen)


def test_cached_inventory_can_be_reconstructed_without_network(tmp_path, monkeypatch):
    pages = fixture_pages()
    mocked_api(monkeypatch, pages)
    result = seeds.inventory(tmp_path / "api")
    monkeypatch.setattr(seeds.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unexpected network"))
    replay = seeds.inventory(tmp_path / "api")
    assert replay["files"] == result["files"]
    assert replay["licenses"] == result["licenses"]


def test_duplicate_destination_from_multiple_components_fails(tmp_path, monkeypatch):
    pages = fixture_pages()
    pages[first(API + "nodes/e7m5k/files/osfstorage/")] = page([folder("subj01", "e7m5k", "subj11")])
    mocked_api(monkeypatch, pages)
    with pytest.raises(ValueError, match="Duplicate"):
        seeds.inventory(tmp_path / "api")


def test_unsafe_api_pagination_url_is_not_requested(tmp_path, monkeypatch):
    pages = fixture_pages()
    pages[first(API + "nodes/wa3qk/children/")]["links"]["next"] = "https://api.osf.io.evil.test/v2/page2"
    seen = mocked_api(monkeypatch, pages)
    with pytest.raises(ValueError, match="Unexpected API"):
        seeds.inventory(tmp_path / "api")
    assert len(seen) == 1


def test_pagination_cycle_stops(tmp_path, monkeypatch):
    url = first(API + "nodes/wa3qk/children/")
    pages = fixture_pages()
    pages[url]["links"]["next"] = url
    mocked_api(monkeypatch, pages)
    with pytest.raises(ValueError, match="Pagination loop"):
        seeds.inventory(tmp_path / "api")


def test_unsafe_nested_file_name_rejected(tmp_path, monkeypatch):
    pages = fixture_pages()
    pages[first(API + "nodes/wa3qk/files/osfstorage/subj01-session/")]["data"][0]["attributes"]["name"] = "../outside.mat"
    mocked_api(monkeypatch, pages)
    with pytest.raises(ValueError, match="Unsafe"):
        seeds.inventory(tmp_path / "api")


def test_missing_payload_checksum_rejected(tmp_path, monkeypatch):
    pages = fixture_pages()
    pages[first(API + "nodes/wa3qk/files/osfstorage/subj01-session/")]["data"][0]["attributes"]["extra"]["hashes"]["sha256"] = None
    mocked_api(monkeypatch, pages)
    with pytest.raises(ValueError, match="checksum"):
        seeds.inventory(tmp_path / "api")


def test_unsafe_payload_url_rejected_before_manifest_freeze(tmp_path, monkeypatch):
    pages = fixture_pages()
    pages[first(API + "nodes/wa3qk/files/osfstorage/subj01-session/")]["data"][0]["links"]["download"] = "https://other.example/payload"
    mocked_api(monkeypatch, pages)
    with pytest.raises(ValueError, match="official OSF"):
        seeds.inventory(tmp_path / "api")
