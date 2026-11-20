"""Compare the real input headers (headers.json, read-only on Leonardo, 20/11) with what the reader requires."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
headers = json.loads((HERE/"headers.json").read_text())
plan = json.loads((HERE.parent/"plan.json").read_text())
users = json.loads((ROOT/"tests/cpu/data/pose_temponet_cache_users_20261120.json").read_text())
FIELDS = ("which", "split", "checkpoint_step", "checkpoint_sha256", "code_sha", "source_sha256", "recipe_sha256",
          "pose_stats_sha256", "checkpoint_pose_stats_sha256")  # pose_temponet.data.PRODUCER_FIELDS
checks = []


def check(name, condition):
    checks.append((name, bool(condition)))


for split, windows, people in (("train", 1907, 40), ("val", 18548, 15)):
    cache, reference = headers["cache_"+split], headers["target_reference_"+split]
    members, config, producer = cache["members"], cache["config"], plan["producers"][split]
    check(split+": path is the pinned cache", cache["path"] == plan["inputs"]["cache_"+split]["path"])
    check(split+": reference path is the pinned one", reference["path"] == plan["inputs"]["target_reference_"+split]["path"])
    for key, descr, shape in (("pre", "<f2", [windows, 160, 384]), ("lat", "<f2", [windows, 160, 384]),
                              ("target", "<f4", [windows, 160, 20]), ("pred", "<f4", [windows, 160, 20]),
                              ("mask", "|b1", [windows, 160, 20]), ("key_time_valid", "|b1", [windows, 160]),
                              ("std", "<f4", [windows, 20]), ("pose_var", "<f4", [windows])):
        member = members[key+".npy"]
        check(f"{split}: {key} {descr} {shape}", member["descr"] == descr and member["shape"] == shape and not member["fortran"])
    for key in FIELDS:
        check(f"{split}: config.{key} equals the pinned producer", config.get(key) == producer["metadata"][key])
    for key, value in {"n_windows": 10, "window_s": 4.0, "seed": 20261026, "per_side": False}.items():
        check(f"{split}: config.{key} == {value!r}", config.get(key) == value)
    check(split+": pinned window and user counts", plan["inputs"]["cache_"+split]["windows"] == windows and
          plan["inputs"]["cache_"+split]["users"] == people == len(users[split]["users"]))
    check(split+": reference metadata has the same users as the cache", reference["meta"]["users"] == users[split]["users"]
          and reference["meta"]["windows"] == windows)
    check(split+": producer users of the new plan are the cache IDs", producer["users"] == users[split]["users"])
    for key in ("target", "mask", "std", "pose_var"):
        check(f"{split}: reference {key} has the cache geometry", reference["members"][key+".npy"]["shape"] == members[key+".npy"]["shape"]
              and reference["members"][key+".npy"]["descr"] == members[key+".npy"]["descr"])
for key in ("post_ridge_predictions", "trained_head_predictions"):
    member = headers[key]["members"]
    check(key+": only float64 prediction 18548x148x20", list(member) == ["prediction.npy"] and
          member["prediction.npy"]["descr"] == "<f8" and member["prediction.npy"]["shape"] == [18548, 148, 20])
    check(key+": path is the pinned one", headers[key]["path"] == plan["inputs"][key]["path"])
coefficients = headers["post_ridge_coefficients"]["members"]
check("post_ridge_coefficients: weights 384x20 and bias 20, float64",
      coefficients["weights.npy"]["shape"] == [384, 20] and coefficients["bias.npy"]["shape"] == [20] and
      coefficients["weights.npy"]["descr"] == coefficients["bias.npy"]["descr"] == "<f8")
# Window metadata (meta_stats.json): the reader needs unique, canonical, finite metadata with only two fields read by name.
stats = json.loads((HERE/"meta_stats.json").read_text())
KEYS = ["fs", "n_patches", "recording", "session", "side", "stage", "start", "stem", "user"]
for split, windows, people in (("train", 1907, 40), ("val", 18548, 15)):
    for source in ("cache_", "target_reference_"):
        entry = stats[source+split]
        name = source+split
        check(name+": every window has unique canonical metadata", entry["canonical_ok"] and entry["windows"] == windows ==
              entry["unique_windows"] and entry["duplicate_groups"] == 0)
        check(name+": one key set, the nine real keys", entry["key_sets"] == [[KEYS, windows]])
        check(name+": user is str and n_patches is int everywhere", entry["types"]["user"] == {"str": windows} and
              entry["types"]["n_patches"] == {"int": windows})
        check(name+": n_patches inside (12, 160]", 12 < entry["n_patches_min_max"][0] <= entry["n_patches_min_max"][1] <= 160)
        check(name+": no non-finite float and the pinned user count", entry["nonfinite_floats"] == 0 and entry["users"] == people)
    check(split+": cache metadata identical to the target reference metadata", stats["same_meta_cache_vs_reference_"+split] is True)
    # One of the three "targets" digests that the preflight compares with the producer receipt depends on metadata only.
    check(split+": real metadata digest equals details.targets.metadata of the producer receipt",
          stats["cache_"+split]["meta_canonical_sha256"] == plan["producers"][split]["details"]["targets"]["metadata"])
# Pose support of the train cache (train_support.json). benchmark._report_measurement builds its cost surrogate from the
# FIRST window of each of the first fifteen sorted training users: those windows must carry valid pose.
support = json.loads((HERE/"train_support.json").read_text())
surrogate = support["first_15_sorted_users"]
check("train: mask geometry 1907x160x20 and metadata grouped by user", support["shape"] == [1907, 160, 20] and
      support["meta_in_user_order"] is True)
check("train: the fifteen surrogate users are the first fifteen sorted cache users",
      [row["user"] for row in surrogate] == users["train"]["users"][:15])
check("train: every surrogate first window has central support on all twenty joints",
      all(row["supported"] > 0 and row["joints_with_2"] == 20 for row in surrogate))
check("train: windows without central support are counted, not assumed absent",
      support["windows_with_central_support"] + support["windows_without_central_support"] == 1907)
failed = [name for name, passed in checks if not passed]
print(json.dumps({"checks": len(checks), "passed": len(checks)-len(failed), "failed": failed}, indent=1))
sys.exit(1 if failed else 0)
