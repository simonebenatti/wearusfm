# Sola lettura: estrae gli utenti da meta_json delle due cache storiche, senza numpy.
import ast
import json
import sys
import zipfile

PATHS = {
    "train": "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105/49046/pose/train/train.npz",
    "val": sys.argv[1] if len(sys.argv) > 1 else "",
}


def read_npy_unicode(raw):
    assert raw[:6] == b"\x93NUMPY", "not npy"
    major = raw[6]
    if major == 1:
        hlen = int.from_bytes(raw[8:10], "little")
        start = 10
    else:
        hlen = int.from_bytes(raw[8:12], "little")
        start = 12
    header = ast.literal_eval(raw[start:start + hlen].decode("latin1"))
    descr = header["descr"]
    assert descr.startswith("<U") and header["shape"] == (), header
    body = raw[start + hlen:]
    return body.decode("utf-32-le").rstrip("\x00")


out = {}
for split, path in PATHS.items():
    if not path:
        continue
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("meta_json.npy")
    meta = json.loads(read_npy_unicode(raw))
    users = sorted({m["user"] for m in meta})
    out[split] = {"path": path, "windows": len(meta), "n_users": len(users), "users": users,
                  "first_meta_keys": sorted(meta[0].keys())}
print(json.dumps(out))
