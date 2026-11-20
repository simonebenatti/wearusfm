# Read-only, standard library only (like real_input_headers/read_headers.py): do the 15 surrogate windows of
# benchmark._report_measurement have centred-moment support in the pinned TRAIN cache?  Reads mask.npy (6.1 MB) and meta_json.npy.
import ast, json, sys, zipfile
path = sys.argv[1] if len(sys.argv) > 1 else "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105/49046/pose/train/train.npz"
def member(archive, name):
    raw = archive.read(name)
    assert raw[:6] == b"\x93NUMPY"
    hlen, start = (int.from_bytes(raw[8:10], "little"), 10) if raw[6] == 1 else (int.from_bytes(raw[8:12], "little"), 12)
    return ast.literal_eval(raw[start:start+hlen].decode("latin1")), raw[start+hlen:]
with zipfile.ZipFile(path) as archive:
    head, body = member(archive, "meta_json.npy"); meta = json.loads(body.decode("utf-32-le").rstrip("\x00"))
    head, mask = member(archive, "mask.npy")
assert head["descr"] == "|b1" and head["shape"] == (len(meta), 160, 20) and not head["fortran_order"], head
empty = []
for user in sorted({m["user"] for m in meta})[:15]:
    i = next(k for k, m in enumerate(meta) if m["user"] == user)
    centre = mask[i*3200+6*20:i*3200+154*20]                      # patches 6..153, 20 joints each
    best = max(sum(centre[j::20]) for j in range(20))             # supported centres of the best joint
    print(user, "window", i, "supported centres (best joint):", best, "OK" if best >= 2 else "NO SUPPORT")
    if best < 2: empty.append(user)
print(json.dumps({"windows": len(meta), "surrogate_users_without_support": empty, "cpu_benchmark_report_surrogate_would_fail": bool(empty)}))
