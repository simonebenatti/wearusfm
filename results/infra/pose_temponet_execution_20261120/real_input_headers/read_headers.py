# Sola lettura: intestazioni .npy e membri JSON piccoli degli input pinnati del preflight TCN. Niente numpy.
import ast
import json
import sys
import zipfile

W = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105"
S = "/leonardo_scratch/large/userexternal/sbenatti"
PATHS = {
    "cache_train": W + "/49046/pose/train/train.npz",
    "cache_val": W + "/49046/pose/val/val.npz",
    "target_reference_train": S + "/wearusfm_runs/results/pilot_posa/B19618_20261028/B/pose/train_subset.npz",
    "target_reference_val": S + "/wearusfm_runs/results/pilot_posa/B19618_20261028/B/pose/val.npz",
    "post_ridge_coefficients": W + "/fits/pose/49046/post_ridge.coefficients.npz",
    "post_ridge_predictions": W + "/fits/pose/49046/post_ridge.predictions.npz",
    "trained_head_predictions": W + "/fits/pose/49046/trained_head.predictions.npz",
}


def header(stream):
    raw = stream.read(4096)
    assert raw[:6] == b"\x93NUMPY", "not npy"
    if raw[6] == 1:
        hlen, start = int.from_bytes(raw[8:10], "little"), 10
    else:
        hlen, start = int.from_bytes(raw[8:12], "little"), 12
    return ast.literal_eval(raw[start:start + hlen].decode("latin1")), start + hlen


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


out = {}
for key, path in PATHS.items():
    entry = {"path": path, "members": {}}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            with archive.open(info) as stream:
                head, _ = header(stream)
            entry["members"][info.filename] = {"descr": head["descr"], "shape": list(head["shape"]),
                                              "fortran": head["fortran_order"], "bytes": info.file_size,
                                              "compress": info.compress_type}
        names = set(archive.namelist())
        if "config_json.npy" in names:
            entry["config"] = json.loads(text_member(archive, "config_json.npy"))
        if "meta_json.npy" in names and key.startswith("target_reference"):
            meta = json.loads(text_member(archive, "meta_json.npy"))
            entry["meta"] = {"windows": len(meta), "users": sorted({m["user"] for m in meta}),
                             "keys": sorted(meta[0].keys())}
    out[key] = entry
try:
    with open(S + "/data/processed/emg2pose_pose/pose_stats.json") as stream:
        stats = json.load(stream)
    out["pose_stats"] = {"keys": sorted(stats.keys()) if isinstance(stats, dict) else str(type(stats))}
except Exception as error:  # noqa: BLE001
    out["pose_stats"] = {"error": repr(error)}
json.dump(out, sys.stdout)
