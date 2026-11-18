#!/usr/bin/env python3
"""Freeze the official SEEDS OSF inventory; never download signal payloads."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from seeds_download import validate_manifest
except ModuleNotFoundError:
    from scripts.seeds_download import validate_manifest

ROOT = "wa3qk"
API = "https://api.osf.io/v2/"


def get_json(url: str, archive: Path) -> dict:
    if not url.startswith(API):
        raise ValueError(f"Unexpected API URL: {url}")
    key = hashlib.sha256(url.encode()).hexdigest()
    path = archive / (key + ".json")
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "WearUsFM-SEEDS-inventory/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            value = json.loads(raw)
            path.write_bytes(raw)
            (archive / (key + ".url")).write_text(url + "\n")
            time.sleep(0.65)
            return value
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(min(120, max(5, int(exc.headers.get("Retry-After", "10")))))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def listing(url: str, archive: Path):
    sep = "&" if "?" in url else "?"
    url = url + sep + "page%5Bsize%5D=100"
    seen = set()
    while url:
        if url in seen:
            raise ValueError("Pagination loop")
        seen.add(url)
        page = get_json(url, archive)
        yield from page["data"]
        url = page["links"].get("next")


def inventory(archive: Path) -> dict:
    archive.mkdir(parents=True, exist_ok=True)
    nodes = [ROOT]
    nodes += [x["id"] for x in listing(API + f"nodes/{ROOT}/children/", archive)]
    if len(nodes) != len(set(nodes)):
        raise ValueError("Duplicate source node")
    files, licenses, ignored = [], [], []
    for node in nodes:
        meta = get_json(API + f"nodes/{node}/", archive)["data"]
        license_url = meta["relationships"]["license"]["links"]["related"]["href"]
        license_meta = get_json(license_url, archive)["data"]
        licenses.append({"node": node, "name": license_meta["attributes"]["name"],
                         "url": license_meta["attributes"]["url"], "api": license_url})
        providers = list(listing(API + f"nodes/{node}/files/", archive))
        if [p["attributes"]["name"] for p in providers] != ["osfstorage"]:
            raise ValueError(f"Unexpected providers on {node}")
        roots = list(listing(providers[0]["relationships"]["files"]["links"]["related"]["href"], archive))
        stack = []
        for row in roots:
            name = row["attributes"]["name"]
            if re.fullmatch(r"subj\d{2}", name) or name == "functions" or name == "Demographic Data.pdf":
                stack.append((row, name))
            else:
                ignored.append({"node": node, "name": name, "id": row["id"]})
        visited_folders = set()
        while stack:
            row, path = stack.pop()
            attrs = row["attributes"]
            rel = PurePosixPath(path)
            if rel.is_absolute() or ".." in rel.parts or "\\" in path:
                raise ValueError(f"Unsafe path {path}")
            if attrs["kind"] == "folder":
                if row["id"] in visited_folders:
                    raise ValueError("Repeated folder or folder cycle")
                visited_folders.add(row["id"])
                url = row["relationships"]["files"]["links"]["related"]["href"]
                stack.extend((child, path + "/" + child["attributes"]["name"]) for child in listing(url, archive))
            else:
                hashes = attrs["extra"]["hashes"]
                if not hashes.get("sha256") or not hashes.get("md5"):
                    raise ValueError(f"Missing source checksum: {path}")
                files.append({"path": path, "size": attrs["size"], "sha256": hashes["sha256"],
                              "md5": hashes["md5"], "url": row["links"]["download"],
                              "source_node": node, "osf_id": row["id"],
                              "version": attrs["current_version"]})
        print(f"node={node} files={len(files)} bytes={sum(x['size'] for x in files)}", flush=True)
    paths = [x["path"] for x in files]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate destination path")
    files.sort(key=lambda x: x["path"])
    subjects = sorted({x["path"].split("/")[0] for x in files if re.match(r"subj\d{2}/", x["path"])})
    result = {"schema_version": 1, "dataset": "SEEDS", "source_node": ROOT,
            "source_doi": "10.17605/OSF.IO/WA3QK", "pretraining_allowed": False,
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "nodes": nodes, "licenses": licenses, "subjects": subjects,
            "files": files, "total_bytes": sum(x["size"] for x in files),
            "excluded_non_signal_folders": ignored}
    validate_manifest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("subjects", "total_bytes", "licenses")}))


if __name__ == "__main__":
    main()
