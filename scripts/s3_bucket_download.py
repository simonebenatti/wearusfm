#!/usr/bin/env python3
"""Scarica un prefisso da un bucket S3 pubblico via REST API (no aws-cli, no boto3 -
solo libreria standard). Scritto per physionet-open (GRABMyo, Hyser): il server web di
physionet.org da Leonardo era lentissimo (~150KB/s), il bucket S3 sottostante e'
veloce (stesso dato, host diverso). Vedi docs/decisioni.md, 22/09/2026.

Uso:
    python3 s3_bucket_download.py --bucket physionet-open --prefix grabmyo/1.1.0/ \
        --dest $WORK/data/raw/grabmyo --strip-prefix
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_objects(bucket: str, prefix: str) -> list[tuple[str, int]]:
    """Ritorna [(key, size), ...] per tutti gli oggetti sotto prefix, paginando."""
    base = f"https://{bucket}.s3.amazonaws.com/"
    objects: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = base + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as resp:
            root = ET.fromstring(resp.read())
        for contents in root.findall(f"{S3_NS}Contents"):
            key = contents.find(f"{S3_NS}Key").text
            size = int(contents.find(f"{S3_NS}Size").text)
            objects.append((key, size))
        truncated = root.find(f"{S3_NS}IsTruncated").text == "true"
        if not truncated:
            break
        token_el = root.find(f"{S3_NS}NextContinuationToken")
        token = token_el.text if token_el is not None else None
        if token is None:
            break
    return objects


def download_object(bucket: str, key: str, dest: Path, tries: int = 3) -> bool:
    url = f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  tentativo {attempt}/{tries} fallito per {key}: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--strip-prefix", action="store_true", help="rimuove il prefix dal path locale")
    p.add_argument("--workers", type=int, default=8, help="download paralleli (default 8, come nel resto del progetto)")
    args = p.parse_args()

    dest_root = Path(args.dest)
    print(f"elenco oggetti: s3://{args.bucket}/{args.prefix} ...")
    objects = list_objects(args.bucket, args.prefix)
    total_bytes = sum(size for _, size in objects)
    print(f"trovati {len(objects)} file, {total_bytes / 1e9:.2f} GB totali")

    to_download: list[tuple[str, int, Path]] = []
    skipped = 0
    skipped_bytes = 0
    for key, size in objects:
        rel = key[len(args.prefix) :] if args.strip_prefix else key
        local_path = dest_root / rel
        if local_path.exists() and local_path.stat().st_size == size:
            skipped += 1
            skipped_bytes += size
        else:
            to_download.append((key, size, local_path))
    print(f"da scaricare: {len(to_download)} file ({sum(s for _, s, _ in to_download) / 1e9:.2f} GB); "
          f"gia' presenti: {skipped} ({skipped_bytes / 1e9:.2f} GB)")

    ok, failed = 0, 0
    done_bytes = skipped_bytes
    lock = threading.Lock()
    t0 = time.time()

    def _task(item: tuple[str, int, Path]) -> tuple[str, int, bool]:
        key, size, local_path = item
        return key, size, download_object(args.bucket, key, local_path)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, item) for item in to_download]
        for i, fut in enumerate(as_completed(futures), 1):
            key, size, success = fut.result()
            with lock:
                if success:
                    ok += 1
                    done_bytes += size
                else:
                    failed += 1
                    print(f"FALLITO definitivamente: {key}", file=sys.stderr)
                if i % 200 == 0 or i == len(to_download):
                    elapsed = time.time() - t0
                    rate = (done_bytes - skipped_bytes) / elapsed / 1e6 if elapsed > 0 else 0
                    print(f"[{i}/{len(to_download)}] ok={ok} fail={failed} "
                          f"({done_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB, {rate:.1f} MB/s)")

    print(f"=== fine: ok={ok} skip={skipped} fail={failed} su {len(objects)} file ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
