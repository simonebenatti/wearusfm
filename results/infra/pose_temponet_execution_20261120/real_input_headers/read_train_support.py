# Sola lettura: supporto centrale della posa nella cache train pinnata (mask.npy e meta_json). Niente numpy.
# Serve a sapere se la prima finestra dei primi 15 utenti (benchmark._report_measurement) ha una posa valida.
import ast
import json
import sys
import zipfile

PATH = ("/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105/"
        "49046/pose/train/train.npz")
CROP = 6


def split_npy(raw):
    assert raw[:6] == b"\x93NUMPY"
    if raw[6] == 1:
        hlen, start = int.from_bytes(raw[8:10], "little"), 10
    else:
        hlen, start = int.from_bytes(raw[8:12], "little"), 12
    return ast.literal_eval(raw[start:start + hlen].decode("latin1")), raw[start + hlen:]


with zipfile.ZipFile(PATH) as archive:
    head, body = split_npy(archive.read("mask.npy"))
    mhead, mbody = split_npy(archive.read("meta_json.npy"))
assert head["descr"] == "|b1" and not head["fortran_order"], head
n, t, j = head["shape"]
assert len(body) == n * t * j, (len(body), head)
meta = json.loads(mbody.decode("utf-32-le").rstrip("\x00"))
assert len(meta) == n and all(m["n_patches"] == t for m in meta)
row = t * j
rows = []
for w in range(n):
    block = body[w * row:(w + 1) * row]
    central = block[CROP * j:(t - CROP) * j]
    per_joint = [central[k::j].count(1) for k in range(j)]
    rows.append({"supported": sum(per_joint), "max_joint": max(per_joint), "joints_with_2": sum(c >= 2 for c in per_joint),
                 "full_mask": block.count(1)})
users = sorted({m["user"] for m in meta})
first = {}
for i, m in enumerate(meta):
    first.setdefault(m["user"], i)
per_user = {}
for i, m in enumerate(meta):
    entry = per_user.setdefault(m["user"], {"windows": 0, "supported_windows": 0, "usable_windows": 0})
    entry["windows"] += 1
    entry["supported_windows"] += rows[i]["supported"] > 0
    entry["usable_windows"] += rows[i]["joints_with_2"] > 0
out = {
    "path": PATH, "shape": [n, t, j], "central_patches": t - 2 * CROP,
    "windows_with_central_support": sum(r["supported"] > 0 for r in rows),
    "windows_without_central_support": sum(r["supported"] == 0 for r in rows),
    "windows_with_a_joint_of_two_centres": sum(r["joints_with_2"] > 0 for r in rows),
    "windows_with_empty_full_mask": sum(r["full_mask"] == 0 for r in rows),
    "first_15_sorted_users": [{"user": u, "first_index": first[u], **rows[first[u]],
                               "start": meta[first[u]]["start"], "stage": meta[first[u]]["stage"]} for u in users[:15]],
    "users_whose_first_window_has_no_central_support": [u for u in users if rows[first[u]]["supported"] == 0],
    "users_whose_first_window_has_no_joint_with_two_centres": [u for u in users if rows[first[u]]["joints_with_2"] == 0],
    "per_user_min": {k: min(v[k] for v in per_user.values()) for k in ("windows", "supported_windows", "usable_windows")},
    "users_with_unsupported_windows": sum(1 for v in per_user.values() if v["supported_windows"] < v["windows"]),
    "meta_in_user_order": [m["user"] for m in meta] == sorted(m["user"] for m in meta),
}
json.dump(out, sys.stdout)
