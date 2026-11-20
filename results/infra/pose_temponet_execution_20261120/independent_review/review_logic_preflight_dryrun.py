"""Dry run of the REAL provenance.preflight / read_verified_inputs bodies on small real-format files.

Reviewer LOGIC, 20/11/2026, commit 5d6fb2dc. Run from a checkout of that commit:
  PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python <this file> <empty output dir>
Real: user IDs (fixture read from the two caches), producer receipts (only the artifact pin and the six details are
re-bound to the small files), cache config_json, metadata key layout, dtypes, file writer and readers.
Synthetic: array values and sizes; therefore only metrics.HISTORICAL_REFERENCE is rebound to the synthetic numbers.
Nothing is monkeypatched in provenance, data or the historical readers. Writes only under argv[1], then removes it.
"""
import copy, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
from pose_temponet import data as D, execution as E, metrics as M, provenance as P

OUT = Path(sys.argv[1]); shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
REPO = Path.cwd()
fix = json.loads((REPO/"tests/cpu/data/pose_temponet_cache_users_20261120.json").read_text())
headers = json.loads((REPO/"results/infra/pose_temponet_execution_20261120/real_input_headers/headers.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin = lambda p, **kw: {"path": str(p), "sha256": sha(p), "bytes": Path(p).stat().st_size, **kw}

def make(split, users, windows, seed):
    rng = np.random.default_rng(seed)
    n = len(users)*windows
    lengths = np.full(n, 160); lengths[3::7] = 150
    avail = np.arange(160)[None, :] < lengths[:, None]
    lat = rng.normal(size=(n, 160, 384)).astype(np.float16); pre = rng.normal(size=(n, 160, 384)).astype(np.float16)
    lat[~avail] = 0; pre[~avail] = 0
    scale = np.where(np.arange(n) % 3 == 0, .05, 1.)[:, None, None]          # some static windows
    target = (rng.normal(size=(n, 160, 20))*scale).astype(np.float32)
    mask = np.repeat(avail[..., None], 20, axis=2) & (rng.random((n, 160, 20)) > .05)
    pred = (.6*target + .3*scale*rng.normal(size=target.shape)).astype(np.float32)
    meta = [{"fs": 2000.0, "n_patches": int(lengths[i]), "recording": i//windows, "session": "2022-04-13-1649836800-9aadd",
             "side": "left" if i % 2 else "right", "stage": "cv-emg-pose-train@2", "start": 4000*i,
             "stem": "2022-04-13-1649836800-9aadd-cv-emg-pose-train@2-recording-%d_left" % (i//windows),
             "user": users[i//windows]} for i in range(n)]
    assert sorted(meta[0]) == headers["target_reference_"+split]["meta"]["keys"]        # the real key layout
    return {"pre": pre, "lat": lat, "key_time_valid": avail, "pred": pred, "target": target, "mask": mask,
            "std": np.full((n, 20), .3, np.float32), "pose_var": M.dynamic_variance(target, mask).astype(np.float32),
            "has_head": True, "dim": 384, "n_recordings": len(users), "meta": meta,
            "config": copy.deepcopy(headers["cache_"+split]["config"])}             # the real config_json

files, sizes = {}, {"train": 2, "val": 4}
repo = OUT/"repo"
for rel in ("src/example.py", "scripts/legacy.py", "scripts/pose_temponet/example.py"):
    (repo/rel).parent.mkdir(parents=True, exist_ok=True); (repo/rel).write_text("# immutable source\n")
proposal = json.loads((REPO/"results/infra/pose_temponet_proposal_20261117/proposal.json").read_text())
receipts = {}
for i, split in enumerate(("train", "val")):
    users = fix[split]["users"]
    data = make(split, users, sizes[split], 100+i)
    cache, reference = OUT/(split+".npz"), OUT/(split+"_reference.npz")
    D.historical.save_cache(cache, data)                                      # the real writer
    with reference.open("wb") as stream:                                     # reference: a pose cache of another model
        np.savez_compressed(stream, lat=data["lat"], pred=data["pred"], target=data["target"], mask=data["mask"], std=data["std"],
                            pose_var=data["pose_var"], has_head=True, dim=384, meta_json=np.asarray(json.dumps(data["meta"])),
                            n_recordings=data["n_recordings"], config_json=np.asarray(json.dumps({"which": "student"})))
    n = len(data["meta"])
    proposal["caches"][split] = pin(cache, users=len(users), windows=n)
    refpin = pin(reference, n_recordings=data["n_recordings"], n_users=len(users), n_windows=n)
    proposal["scratch_dependencies"]["pose_references"][split] = refpin
    receipt = json.loads((REPO/P.HISTORICAL_RECEIPTS[split]["local_copy"]).read_text())   # the REAL receipt (real directories)
    receipt["artifacts"]["cache"] = {**receipt["artifacts"]["cache"], "sha256": sha(cache), "bytes": cache.stat().st_size}
    target_reference = D.reference.read_targets(str(reference))
    # exactly how the historical runtime computed "details" (consolidation_long_pose_runtime.py:136)
    receipt["details"] = D.historical.validate_cache(D.historical.read_cache(str(cache), target_reference, refpin), target_reference, refpin)
    path = repo/"receipts"/(split+".json"); path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(receipt))
    receipts[split] = {"path": str(path.relative_to(repo)), "sha256": sha(path), "bytes": path.stat().st_size}
    files[split] = data
stats = OUT/"pose_stats.json"; stats.write_text("{}\n")
proposal["scratch_dependencies"]["pose_stats"] = {"path": str(stats), "sha256": sha(stats)}
val = files["val"]
np.savez_compressed(OUT/"post_ridge.predictions.npz", prediction=(.5*val["target"][:, 6:-6]).astype(np.float64))
np.savez_compressed(OUT/"trained_head.predictions.npz", prediction=val["pred"][:, 6:-6].astype(np.float64))
np.savez_compressed(OUT/"post_ridge.coefficients.npz", weights=np.zeros((384, 20)), bias=np.zeros(20))
for key, name in (("post_ridge_coefficients", "post_ridge.coefficients.npz"), ("post_ridge_predictions", "post_ridge.predictions.npz"),
                  ("trained_head_predictions", "trained_head.predictions.npz")):
    proposal["references"][key] = pin(OUT/name)
proposal_path = repo/"results/infra/pose_temponet_proposal_20261117/proposal.json"
proposal_path.parent.mkdir(parents=True); proposal_path.write_text(json.dumps(proposal))
plan = P.build_plan("results/infra/pose_temponet_proposal_20261117/proposal.json", receipts, source_root=repo, jobs=E.job_graph())
print("plan built by the real build_plan/validate_plan | train users == fixture:", plan["producers"]["train"]["users"] == fix["train"]["users"],
      "| val:", plan["producers"]["val"]["users"] == fix["val"]["users"])

# Rebind ONLY the parity constants to the synthetic numbers (the real ones belong to the real arrays).
probe = P.read_verified_inputs(plan, "val", stage="preflight", depth="post")
rows = {name: M.evaluate(D.read_predictions(plan["inputs"][name+"_predictions"]), probe)["groups"]["dynamic"] for name in ("post_ridge", "trained_head")}
c = rows["post_ridge"]["centered"]
M.HISTORICAL_REFERENCE = {"post_ridge_E_dyn": c["centered_mse_ratio"], "trained_head_E_dyn": rows["trained_head"]["centered"]["centered_mse_ratio"],
                          "dynamic_selected_windows": rows["post_ridge"]["n_selected_windows"], "dynamic_centered_windows": c["n_windows"],
                          "dynamic_centered_elements": c["n_valid_elements"], "dynamic_window_joint_pairs": c["n_window_joints"], "dynamic_users": c["n_users"]}
del probe

result = P.preflight(plan)                                                    # the REAL, unpatched body
print("preflight returned | integrity:", result["integrity"], "| R1 passed:", result["historical_R1"]["passed"],
      "| folds:", [len(f) for f in result["train"]["folds"]], "| supported train windows:", result["train"]["supported_windows"],
      "| inputs verified:", len(result["inputs"]))
receipt_like = {"details": {"allocation": {"slurm_job_id": "1"}, "preflight": copy.deepcopy(result), "elapsed_seconds": 1.0},
                "gates": {"integrity": True, "historical_R1": bool(result["historical_R1"].get("passed", False))}}
print("receipt payload serializes with provenance.canonical_bytes:", len(P.canonical_bytes(receipt_like)), "bytes")
bad_types = []
def walk(o, path="$"):
    if isinstance(o, dict):
        for k, v in o.items():
            if type(k) is not str: bad_types.append((path, type(k).__name__))
            walk(v, path+"."+str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): walk(v, path+"[%d]" % i)
    elif type(o) not in (str, int, float, bool, type(None)): bad_types.append((path, type(o).__name__))
walk(result)
print("non-builtin value types inside the preflight result:", bad_types[:5], "count", len(bad_types))

# The first delivery's plan content on the same files: users as directory names.
first = copy.deepcopy(plan)
for split in ("train", "val"):
    first["producers"][split]["users"] = list(first["producers"][split]["storage_directories"])
try: P.read_verified_inputs(first, "train", stage="preflight", depth="post"); print("ACCEPTED users := directory names")
except ValueError as e: print("REFUSED  users := directory names (the 64931573 condition) ->", e)
drift = copy.deepcopy(plan); drift["producers"]["val"]["storage_directories"][0] = "uffffffffff"
try: P.read_verified_inputs(drift, "val", stage="preflight"); print("ACCEPTED drifted storage_directories")
except ValueError as e: print("REFUSED  one storage directory replaced in the plan ->", e)
for key in ("targets", "key_time_valid_sha256", "dim", "n_recordings", "n_users", "n_windows"):
    t = copy.deepcopy(plan); t["producers"]["train"]["details"][key] = "x"
    try: P.read_verified_inputs(t, "train", stage="preflight"); print("ACCEPTED changed detail", key)
    except ValueError as e: print("REFUSED  detail", key, "->", e)
shutil.rmtree(OUT)
