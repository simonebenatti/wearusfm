"""Probe: does the file written by the test's _cache have the same members/dtypes as the real headers? (reviewer TESTS)"""
import json, sys, zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0, "tests/cpu")
import test_pose_temponet_preflight_real_layout as T
from pose_temponet import data as D

out = Path(sys.argv[1])
headers = json.loads(Path("results/infra/pose_temponet_execution_20261120/real_input_headers/headers.json").read_text())
print("headers.json top keys:", sorted(headers))
for split, windows in (("train", 2), ("val", 4)):
    users = T.USERS[split]["users"]
    data = T._cache(split, users, windows, 100)
    path = out/(split+".npz")
    if path.exists():
        path.unlink()
    D.historical.save_cache(path, data)
    got = {}
    with zipfile.ZipFile(path) as z:
        for item in z.infolist():
            with z.open(item) as f:
                version = np.lib.format.read_magic(f)
                shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                                         else np.lib.format.read_array_header_2_0(f))
                got[item.filename[:-4]] = {"descr": dtype.str, "shape": list(shape), "fortran": fortran, "version": version}
    real = T.LAYOUT[split]["members"]
    print(split, "member sets equal:", sorted(got) == sorted(real), "| synthetic members:", len(got), "real:", len(real))
    for name in sorted(real):
        g, r = got[name], real[name]
        same_dtype = g["descr"] == r["descr"] or (g["descr"].startswith("<U") and r["descr"].startswith("<U"))
        same_tail = g["shape"][1:] == r["shape"][1:] and (len(g["shape"]) == len(r["shape"]))
        print("  %-15s synthetic %-10s %-18s real %-10s %-20s dtype_ok=%s rank/tail_ok=%s" % (
            name, g["descr"], g["shape"], r["descr"], r["shape"], same_dtype, same_tail))
    pv = data["pose_var"]
    print("  pose_var NaN windows:", int(np.isnan(pv).sum()), "of", len(pv), "| windows with no valid pose:",
          int((~data["mask"].any(axis=(1, 2))).sum()), "| n_patches values:", sorted({m["n_patches"] for m in data["meta"]}))
    print("  dynamic (pv>=THRESHOLD):", int((np.isfinite(pv) & (pv >= T.M.THRESHOLD)).sum()),
          "static:", int((np.isfinite(pv) & (pv < T.M.THRESHOLD)).sum()))
    print("  meta[0]:", data["meta"][0])
    path.unlink()
