"""Candidate change for benchmark._report_measurement, applied in memory only (no file of any worktree is edited):
take, for each surrogate user, the first window that has centred-moment support, among users that have one."""
import inspect, json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path[:0] = ["scripts", "src"]
from pose_temponet import benchmark

OLD_USERS = '    selected_users = sorted({row["user"] for row in data["meta"]})[:15]\n'
NEW_USERS = ('    support = data_api.central_support(data["mask"], data_api.lengths_for(data), data.get("key_time_valid"))\n'
             '    usable = (support.sum(axis=1) >= 2).any(axis=1)  # a joint with two supported centres: enters the centred moments\n'
             '    selected_users = sorted({row["user"] for row, ok in zip(data["meta"], usable) if ok})[:15]\n')
OLD_IDS = '    ids = [next(i for i, row in enumerate(data["meta"]) if row["user"] == user) for user in selected_users]\n'
NEW_IDS = '    ids = [next(i for i, row in enumerate(data["meta"]) if row["user"] == user and usable[i]) for user in selected_users]\n'
source = inspect.getsource(benchmark._report_measurement)
assert source.count(OLD_USERS) == 1 and source.count(OLD_IDS) == 1
original = benchmark._report_measurement
namespace = benchmark.__dict__
exec(source.replace(OLD_USERS, NEW_USERS).replace(OLD_IDS, NEW_IDS), namespace)
fixed = namespace["_report_measurement"]

users_real = json.load(open("tests/cpu/data/pose_temponet_cache_users_20261120.json"))["train"]["users"]
def cache(empty_users=(), windows=3):
    rng = np.random.default_rng(71)
    n = len(users_real)*windows
    target = rng.normal(size=(n, 160, 20)).astype(np.float32)
    mask = np.ones_like(target, dtype=bool)
    meta = [{"user": users_real[i // windows], "session": "s", "stage": "st", "side": "left", "stem": "r%d" % (i // windows),
             "recording": i // windows, "start": 8000*(i % windows), "fs": 2000.0, "n_patches": 160} for i in range(n)]
    for user in empty_users:
        first = next(i for i, m in enumerate(meta) if m["user"] == user)
        mask[first] = False; target[first] = 0
    return {"config": {"split": "train", "which": "teacher", "checkpoint_step": 49046},
            "lat": rng.normal(size=(n, 160, 384)).astype(np.float16), "target": target, "mask": mask,
            "key_time_valid": np.ones((n, 160), bool), "std": np.ones((n, 20), np.float32), "pose_var": np.ones(n, np.float32), "meta": meta}
base = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf_rdk7q_work"
for label, empty in (("no empty first window", ()), ("first windows of sorted users 5 and 11 empty", (sorted(users_real)[4], sorted(users_real)[10]))):
    for name, function in (("current", original), ("candidate", fixed)):
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            try:
                result = function(cache(empty), output_dir=Path(tmp))
                print(f"{label:46s} {name:9s} OK   moment_windows {result['moment_windows']} bootstrap {result['bootstrap_replicates']}")
            except ValueError as error:
                print(f"{label:46s} {name:9s} FAIL {error}")
