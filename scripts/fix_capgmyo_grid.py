#!/usr/bin/env python3
"""Correzione della griglia di CapgMyo nei sidecar gia' ingeriti (Simone, 07/10/2026, opzione A; fatti 14 e 26): da 8 x 16 con `divmod(i, 16)` a
16 x 8 con `divmod(i, 8)` (`ingest.capgmyo.grid_position`). I dati (`data_int16.npy`) non cambiano: si riscrivono solo `grid_row` e `grid_col`
dei canali nel `metadata.json`. Poi:
- **manifest v1.1**: il manifest congelato con lo sha256 nuovo dei sidecar di CapgMyo e nient'altro di diverso (pesi, split, sessioni uguali);
- **scale**: il file delle scale senza le scale bipolari di CapgMyo (le coppie di vicini cambiano), da ricalcolare con
  `scripts/measure_loader.py --manifest <manifest v1.1> --scales-from <questo file> --bipolar-scales --skip-rate`.
I sidecar vecchi si copiano in `--out-dir/backup/` prima di riscriverli. Rilanciabile: un sidecar gia' corretto si lascia com'e'.

  python3 scripts/fix_capgmyo_grid.py --manifest M.json.gz --root $WORK/data/processed --root $SCRATCH/data/processed \\
      --scales session_scales.json --out-dir $WORK/wearusfm_runs/results/passo6/capgmyo_grid_fix [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.ingest.capgmyo import GRID_COLS, GRID_ROWS, N_CHANNELS, grid_position  # noqa: E402

DATASET = "capgmyo"


def fixed_meta(meta: dict) -> tuple[dict, bool]:
    """(sidecar con la griglia 16 x 8, cambiato?). Errore se il montaggio non e' quello atteso (un gruppo grid_2d da 128 canali in ordine)."""
    groups = meta["montage"]["groups"]
    if len(groups) != 1 or groups[0]["topology"] != "grid_2d" or len(groups[0]["channels"]) != N_CHANNELS:
        raise ValueError("montaggio di CapgMyo inatteso: serve un solo gruppo grid_2d da 128 canali")
    changed = False
    for i, ch in enumerate(groups[0]["channels"]):
        sc = ch["sensor_coords"]
        if int(sc["channel_index"]) != i:
            raise ValueError(f"canale {i}: channel_index {sc['channel_index']}, atteso {i}")
        r, c = grid_position(i)
        if (sc["grid_row"], sc["grid_col"]) != (r, c):
            sc["grid_row"], sc["grid_col"] = r, c
            changed = True
    return meta, changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--scales", type=Path, required=True, help="session_scales.json attuale (con le scale bipolari)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="controlla e conta, non scrive nulla")
    args = ap.parse_args(argv)
    raw_manifest = args.manifest.read_bytes()
    doc = json.loads(gzip.decompress(raw_manifest) if str(args.manifest).endswith(".gz") else raw_manifest)
    cols = doc["columns"]
    i_ds, i_sha = cols.index("dataset"), cols.index("sidecar_sha256")
    index = L.ManifestIndex.load(args.manifest, args.root)  # per `path_of`
    report = {"manifest_in": str(args.manifest), "manifest_in_sha256": hashlib.sha256(raw_manifest).hexdigest(),
              "grid": [GRID_ROWS, GRID_COLS], "sessions": []}
    new_sha: dict[tuple, str] = {}
    rows = [dict(zip(cols, r)) for r in doc["rows"] if r[i_ds] == DATASET]  # tutte le righe, anche i soggetti di test
    for row in rows:
        path = index.path_of(row) / "metadata.json"
        raw = path.read_bytes()
        old = hashlib.sha256(raw).hexdigest()
        meta, changed = fixed_meta(json.loads(raw))
        if changed and old != row["sidecar_sha256"]:
            raise ValueError(f"{path}: sidecar diverso da quello del manifest e non ancora corretto: fermarsi")
        out = (json.dumps(meta, indent=2)).encode() if changed else raw
        sha = hashlib.sha256(out).hexdigest()
        if changed and not args.dry_run:
            bak = args.out_dir / "backup" / row["subject"] / row["session"] / "metadata.json"
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bak)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_bytes(out)
            tmp.replace(path)
        new_sha[(row["subject"], row["session"])] = sha
        report["sessions"].append({"subject": row["subject"], "session": row["session"], "path": str(path), "changed": changed,
                                   "sha256_old": old, "sha256_new": sha})
    if not new_sha:
        raise ValueError("nessuna sessione di CapgMyo nel manifest")
    i_sub, i_ses = cols.index("subject"), cols.index("session")
    n_rows = 0
    for r in doc["rows"]:
        if r[i_ds] == DATASET and (r[i_sub], r[i_ses]) in new_sha:
            r[i_sha] = new_sha[(r[i_sub], r[i_ses])]
            n_rows += 1
    doc.setdefault("params", {})["revision"] = {
        "version": "v1.1", "from_sha256": report["manifest_in_sha256"],
        "change": "sha256 dei sidecar di CapgMyo dopo la correzione della griglia a 16x8 (fatti 14 e 26, Simone 07/10/2026); nient'altro"}
    scales = json.loads(args.scales.read_text())
    dropped = [k for k in scales["scales"] if k.startswith(f"{DATASET}/") and "|bip:" in k]
    for k in dropped:
        del scales["scales"][k]
    report.update({"rows_updated": n_rows, "bipolar_scales_dropped": len(dropped), "dry_run": args.dry_run})
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "manifest_v1_1.json.gz").write_bytes(gzip.compress(json.dumps(doc).encode()))
        (args.out_dir / "session_scales_without_capgmyo_bipolar.json").write_text(json.dumps(scales))
        report["manifest_out_sha256"] = hashlib.sha256((args.out_dir / "manifest_v1_1.json.gz").read_bytes()).hexdigest()
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"CapgMyo: {len(new_sha)} sessioni, {sum(s['changed'] for s in report['sessions'])} sidecar da correggere, {n_rows} righe del manifest, "
          f"{len(dropped)} scale bipolari tolte{' (prova, nulla scritto)' if args.dry_run else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
