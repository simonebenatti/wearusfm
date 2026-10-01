"""scripts/dataverse_download.py contro un Dataverse finto servito in locale (http.server in un thread)."""

import hashlib
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("dataverse_download", Path(__file__).resolve().parents[2] / "scripts" / "dataverse_download.py")
DD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DD)

CONTENT = {1: b"A" * 1000, 2: b"B" * 2500, 3: b"video" * 10, 4: b"tab,originale\n1,2\n"}


class _Fake:
    def __init__(self):
        self.files = [
            {"dataFile": {"id": 1, "filename": "S010_ex1.mat", "filesize": 1000, "md5": hashlib.md5(CONTENT[1]).hexdigest()}},
            {"dataFile": {"id": 2, "filename": "S010_ex1_orig.mat", "filesize": 2500,
                          "checksum": {"type": "MD5", "value": hashlib.md5(CONTENT[2]).hexdigest()}}},
            {"dataFile": {"id": 3, "filename": "S010_ex1.mp4", "filesize": 50, "md5": hashlib.md5(CONTENT[3]).hexdigest()}},
            {"dataFile": {"id": 4, "filename": "Invalid Gaze.tab", "filesize": 18, "originalFileFormat": "text/csv",
                          "md5": hashlib.md5(CONTENT[4]).hexdigest()}},
        ]
        self.requests = []
        self.corrupt = set()


def _serve(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            fake.requests.append(self.path)
            if self.path.startswith("/api/datasets/:persistentId/"):
                body = json.dumps({"status": "OK", "data": {"latestVersion": {"files": fake.files}}}).encode()
            elif self.path.startswith("/api/access/datafile/"):
                fid = int(self.path.split("/")[-1].split("?")[0])
                body = CONTENT[fid] if fid not in fake.corrupt else b"X" * len(CONTENT[fid])
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def fake_server():
    fake = _Fake()
    srv, url = _serve(fake)
    yield fake, url
    srv.shutdown()


def test_downloads_only_requested_suffix_and_verifies(tmp_path, fake_server):
    fake, url = fake_server
    r = DD.run(url, [("doi:10.7910/DVN/X", "MDS1")], [".mat"], tmp_path, None, False)
    e = r["datasets"]["MDS1"]
    assert sorted(e["downloaded"]) == ["S010_ex1.mat", "S010_ex1_orig.mat"] and not e["failed"] and e["bytes"] == 3500
    assert (tmp_path / "MDS1" / "S010_ex1.mat").read_bytes() == CONTENT[1]
    assert not (tmp_path / "MDS1" / "S010_ex1.mp4").exists() and not list((tmp_path / "MDS1").glob("*.part"))
    assert not any("datafile/3" in p for p in fake.requests)


def test_resume_skips_complete_files_and_redownloads_wrong_ones(tmp_path, fake_server):
    fake, url = fake_server
    d = tmp_path / "MDS1"
    d.mkdir()
    (d / "S010_ex1.mat").write_bytes(CONTENT[1])
    (d / "S010_ex1_orig.mat").write_bytes(b"B" * 2499 + b"Z")  # dimensione giusta, MD5 sbagliato
    r = DD.run(url, [("doi:10.7910/DVN/X", "MDS1")], [".mat"], tmp_path, None, False)
    assert r["datasets"]["MDS1"]["skipped_complete"] == ["S010_ex1.mat"] and r["datasets"]["MDS1"]["downloaded"] == ["S010_ex1_orig.mat"]
    assert (d / "S010_ex1_orig.mat").read_bytes() == CONTENT[2]
    assert not any("datafile/1" in p for p in fake.requests)


def test_md5_mismatch_is_reported_and_file_not_kept(tmp_path, fake_server):
    fake, url = fake_server
    fake.corrupt.add(2)
    r = DD.run(url, [("doi:10.7910/DVN/X", "MDS1")], [".mat"], tmp_path, None, False)
    assert "MD5" in r["datasets"]["MDS1"]["failed"]["S010_ex1_orig.mat"]
    assert not (tmp_path / "MDS1" / "S010_ex1_orig.mat").exists()
    assert sum("datafile/2" in p for p in fake.requests) == DD.RETRIES
    rc = DD.main(["--server", url, "--dataset", "doi:10.7910/DVN/X=MDS1", "--suffix", ".mat", "--dest", str(tmp_path / "b")])
    assert rc == 1


def test_time_budget_stops_before_next_file_and_tabular_asks_original(tmp_path, fake_server):
    fake, url = fake_server
    r = DD.run(url, [("doi:10.7910/DVN/X", "MDS1")], [".mat"], tmp_path, -1.0, False)
    assert r["stopped"] == "time-budget" and not r["datasets"]["MDS1"]["downloaded"]
    DD.run(url, [("doi:10.7910/DVN/X", "T")], [".tab"], tmp_path, None, False)
    assert any(p.endswith("datafile/4?format=original") for p in fake.requests)
    assert (tmp_path / "T" / "Invalid Gaze.tab").read_bytes() == CONTENT[4]


def test_dry_run_lists_without_downloading(tmp_path, fake_server):
    fake, url = fake_server
    r = DD.run(url, [("doi:10.7910/DVN/X", "MDS1")], [".mat", ".MP4"], tmp_path, None, True)
    assert r["datasets"]["MDS1"]["n_files"] == 3 and not any("datafile" in p for p in fake.requests) and not (tmp_path / "MDS1").exists()


def test_dataset_argument_needs_doi_and_label():
    assert DD._dataset_arg("doi:10.7910/DVN/1Z3IOM=MDS1") == ("doi:10.7910/DVN/1Z3IOM", "MDS1")
    for bad in ("10.7910/DVN/1Z3IOM=MDS1", "doi:10.7910/DVN/1Z3IOM", "doi:x=a/b"):
        with pytest.raises(Exception):
            DD._dataset_arg(bad)
