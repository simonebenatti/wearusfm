# Sola lettura: statistiche dei metadati delle finestre nelle due cache pinnate (solo meta_json). Niente numpy.
import ast
import collections
import hashlib
import json
import math
import sys
import zipfile

W = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105"
S = "/leonardo_scratch/large/userexternal/sbenatti"
PATHS = {
    "cache_train": W + "/49046/pose/train/train.npz",
    "cache_val": W + "/49046/pose/val/val.npz",
    "target_reference_train": S + "/wearusfm_runs/results/pilot_posa/B19618_20261028/B/pose/train_subset.npz",
    "target_reference_val": S + "/wearusfm_runs/results/pilot_posa/B19618_20261028/B/pose/val.npz",
}


def text_member(archive, name):
    raw = archive.read(name)
    assert raw[:6] == b"\x93NUMPY"
    if raw[6] == 1:
        hlen, start = int.from_bytes(raw[8:10], "little"), 10
    else:
        hlen, start = int.from_bytes(raw[8:12], "little"), 12
    head = ast.literal_eval(raw[start:start + hlen].decode("latin1"))
    assert head["descr"].startswith("<U"), head
    return raw[start + hlen:].decode("utf-32-le").rstrip("\x00")


def canonical(value):  # pose_temponet.controls.canonical
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


out = {}
for key, path in PATHS.items():
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(text_member(archive, "meta_json.npy"))
    entry = {"windows": len(meta)}
    keysets = collections.Counter(tuple(sorted(m.keys())) for m in meta)
    entry["key_sets"] = [[list(k), v] for k, v in keysets.items()]
    types = collections.defaultdict(collections.Counter)
    nonfinite = 0
    for m in meta:
        for k, v in m.items():
            types[k][type(v).__name__] += 1
            if isinstance(v, float) and not math.isfinite(v):
                nonfinite += 1
    entry["types"] = {k: dict(v) for k, v in types.items()}
    entry["nonfinite_floats"] = nonfinite
    try:
        ids = [canonical(m) for m in meta]
        entry["canonical_ok"] = True
    except ValueError as error:
        ids = [json.dumps(m, sort_keys=True) for m in meta]
        entry["canonical_ok"] = False
        entry["canonical_error"] = str(error)
    counts = collections.Counter(ids)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    entry["unique_windows"] = len(counts)
    entry["duplicate_groups"] = len(duplicates)
    entry["duplicate_windows"] = sum(duplicates.values())
    entry["duplicate_examples"] = list(duplicates.items())[:3]
    entry["ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    entry["meta_canonical_sha256"] = hashlib.sha256(canonical(meta).encode()).hexdigest() if entry["canonical_ok"] else None
    entry["users"] = len({m["user"] for m in meta})
    entry["n_patches"] = dict(sorted(collections.Counter(m["n_patches"] for m in meta).items())[:5])
    entry["n_patches_min_max"] = [min(m["n_patches"] for m in meta), max(m["n_patches"] for m in meta)]
    per_user = collections.Counter(m["user"] for m in meta)
    entry["windows_per_user_min_max"] = [min(per_user.values()), max(per_user.values())]
    entry["first"] = meta[0]
    # the same (stem, start) twice would be the natural source of duplicates
    pairs = collections.Counter((m.get("stem"), m.get("start")) for m in meta)
    entry["duplicate_stem_start"] = sum(1 for v in pairs.values() if v > 1)
    out[key] = entry
for split in ("train", "val"):
    out["same_meta_cache_vs_reference_" + split] = (
        out["cache_" + split]["meta_canonical_sha256"] == out["target_reference_" + split]["meta_canonical_sha256"])
json.dump(out, sys.stdout)
