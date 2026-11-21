"""Real-shaped inputs: same npz members/dtypes/metadata keys as headers.json, real user IDs, written by the historical
producer writer (consolidation_long_pose.save_cache) and reference writer (pose_eval.save_npz)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["scripts", "src"]
from pose_temponet import data as D
import consolidation_long_pose as H
from wearusfm.harness import pose_eval as PE

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
real = json.load(open("tests/cpu/data/pose_temponet_cache_users_20261120.json"))
SIZES = {"train": int(sys.argv[2]), "val": int(sys.argv[3])}
EMPTY = {"train": 53/1907, "val": 1890/18548}
producer_meta = {"which": "teacher", "checkpoint_step": 49046, "checkpoint_sha256": "bf"*32, "code_sha": "7d"*20,
                 "source_sha256": "1c"*32, "recipe_sha256": "b2"*32, "pose_stats_sha256": "75"*32,
                 "checkpoint_pose_stats_sha256": "75"*32}
plan = {"inputs": {}, "producers": {}}
stats = out/"pose_stats.json"; stats.write_text(json.dumps({"all": {}, "joints": []}))
plan["inputs"]["pose_stats"] = {"path": str(stats), "sha256": D.file_sha256(stats)}
rng = np.random.default_rng(20261120)
for split in ("train", "val"):
    users, n = real[split]["users"], SIZES[split]
    if n == 0:
        continue
    per_recording = int(__import__('os').environ.get('PER_RECORDING', 10))
    n_recordings = -(-n // per_recording)
    owner = np.sort(np.arange(n_recordings) % len(users))
    meta = [{"user": users[owner[i // per_recording]], "session": "2022-04-13-1649836800-9aadd-cv-emg-pose-train@2",
             "stage": "WristFlexionAbduction", "side": "left" if (i // per_recording) % 2 else "right",
             "stem": "2022-04-13-1649836800-9aadd-cv-emg-pose-train@2-recording-%d_left" % (i // per_recording),
             "recording": i // per_recording, "start": 12000*(i % per_recording) + 400, "fs": 2000.0, "n_patches": 160} for i in range(n)]
    rows = rng.random((n, 160)) > .08
    if split == "train":
        # REAL facts of the pinned train cache (train_support.json): 53 windows without any valid pose, none of them the first
        # window of the first 15 sorted users, the first window of 76bfc2d385 (19th) empty, two windows with ONE supported centre.
        ordered = sorted(users)
        first = {u: next(i for i, m in enumerate(meta) if m["user"] == u) for u in ordered}
        protected = {first[u] for u in ordered[:15]}
        empty = [first["76bfc2d385"]]
        candidates = [i for i in range(n) if i not in protected and i != empty[0]]
        empty += [int(i) for i in rng.choice(candidates, 52, replace=False)]
        rows[empty] = False
        single = [int(i) for i in rng.choice([i for i in candidates if i not in set(empty)], 2, replace=False)]
        for i in single:
            rows[i] = False; rows[i, 80] = True
        print("train: empty windows", int((~rows.any(axis=1)).sum()), "| one-centre windows", single,
              "| empty first window of", [u for u in ordered if not rows[first[u]].any()])
    else:
        rows[rng.random(n) < EMPTY[split]] = False
    mask = np.repeat(rows[..., None], 20, axis=2)
    lat = rng.standard_normal((n, 160, 384), dtype=np.float32).astype(np.float16)
    z = lat.astype(np.float32); z = (z-z.mean(-1, keepdims=True))/np.sqrt(z.var(-1, keepdims=True)+1e-5)
    readable = z @ (rng.standard_normal((384, 20), dtype=np.float32)/np.sqrt(np.float32(384)))
    smooth = np.cumsum(rng.standard_normal((n, 160, 20), dtype=np.float32), axis=1)/np.float32(8)
    target = (rng.normal(0, .5, size=(n, 1, 20)).astype(np.float32) + np.float32(.7)*readable + smooth
              + np.float32(.5)*rng.standard_normal((n, 160, 20), dtype=np.float32)).astype(np.float32)
    target[~mask] = 0
    pose_var = np.asarray([float(target[i][rows[i]].var(axis=0).mean()) if rows[i].sum() > 1 else float("nan")
                           for i in range(n)], dtype=np.float32)
    pre = rng.standard_normal((n, 160, 384), dtype=np.float32).astype(np.float16)
    pred = (target*.5 + rng.normal(size=target.shape).astype(np.float32)*.3).astype(np.float32)
    std = rng.uniform(.1, .6, size=(n, 20)).astype(np.float32)
    base_config = {"n_windows": 10, "window_s": 4.0, "seed": 20261026, "pose_stats_sha256": "75"*32, "per_side": False}
    reference = out/(split+"_reference.npz")
    PE.save_npz(reference, {"lat": pre, "pred": pred*0, "target": target, "mask": mask, "std": std, "pose_var": pose_var,
                            "has_head": True, "dim": 384, "meta": meta, "n_recordings": n_recordings,
                            "config": {**base_config, "which": "student", "split": split}})
    refpin = {"path": str(reference), "sha256": D.file_sha256(reference), "bytes": reference.stat().st_size,
              "n_windows": n, "n_users": len(users), "n_recordings": n_recordings}
    data = {"pre": pre, "lat": lat, "key_time_valid": np.ones((n, 160), bool), "pred": pred, "target": target, "mask": mask,
            "std": std, "pose_var": pose_var, "has_head": True, "dim": 384, "n_recordings": n_recordings, "meta": meta,
            "config": {**base_config, "checkpoint": "/x/checkpoint_postdecay.pt", **producer_meta, "split": split}}
    cache = out/(split+".npz")
    H.save_cache(cache, data)
    target_reference = H.R.read_targets(str(reference))
    details = H.validate_cache(H.read_cache(str(cache), target_reference, refpin), target_reference, refpin)
    plan["inputs"]["cache_"+split] = {"path": str(cache), "sha256": D.file_sha256(cache), "bytes": cache.stat().st_size,
                                      "users": len(users), "windows": n}
    plan["inputs"]["target_reference_"+split] = refpin
    plan["producers"][split] = {"metadata": {**producer_meta, "split": split}, "users": sorted(users),
                                "storage_directories": sorted("u"+u for u in users), "details": details}
(out/"plan.json").write_text(json.dumps(plan, indent=1))
print({k: v.get("bytes") for k, v in plan["inputs"].items()})

# --- development references (only when a val split was built): files as fit-pose wrote them, constants from the HISTORICAL summary
if SIZES["val"]:
    plan = json.loads((out/"plan.json").read_text())
    refpin = plan["inputs"]["target_reference_val"]
    target_reference = H.R.read_targets(refpin["path"])
    data = H.read_cache(plan["inputs"]["cache_val"]["path"], target_reference, refpin)
    view = H._target_view(data)
    rng = np.random.default_rng(7)
    n = len(data["meta"])
    policy = {}
    for name, prediction in (("post_ridge", rng.normal(size=(n, 148, 20))), ("trained_head", data["pred"][:, 6:-6].astype(np.float64))):
        path = out/(name+".predictions.npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, prediction=prediction)
        plan["inputs"][name+"_predictions"] = {"path": str(path), "sha256": D.file_sha256(path), "bytes": path.stat().st_size}
        dynamic = H.R.summarize(view, prediction)["groups"]["dynamic"]
        policy[name+"_E_dyn"] = dynamic["centered"]["centered_mse_ratio"]
        policy.update(dynamic_selected_windows=dynamic["n_selected_windows"], dynamic_centered_windows=dynamic["centered"]["n_windows"],
                      dynamic_centered_elements=dynamic["centered"]["n_valid_elements"],
                      dynamic_window_joint_pairs=dynamic["centered"]["n_window_joints"], dynamic_users=dynamic["centered"]["n_users"])
    coefficients = out/"post_ridge.coefficients.npz"
    np.savez_compressed(coefficients, weights=rng.normal(size=(384, 20)), bias=rng.normal(size=20))
    plan["inputs"]["post_ridge_coefficients"] = {"path": str(coefficients), "sha256": D.file_sha256(coefficients), "bytes": coefficients.stat().st_size}
    plan["historical_reference_fixture"] = policy
    (out/"plan.json").write_text(json.dumps(plan, indent=1))
    print("policy", policy)
print("built with numpy", np.__version__)
