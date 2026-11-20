"""R-1 parity at the REAL val size: historical summary path vs metrics.evaluate.

gen <dir>   builds the synthetic arrays once (real geometry: 18548 windows, 15 real val user IDs with 700..1540
            windows each, target float32 [N,160,20], per-time-step masks with empty / one-centre / edge-only windows,
            float32 pose_var with NaN, float64 ridge-like predictions [N,148,20], float32 head predictions [N,160,20]).
run <dir>   loads them and compares the two paths bit by bit, then varies batch size, window order and summation order.
"""
import json, resource, sys, time
from pathlib import Path
import numpy as np

sys.path[:0] = ["scripts", "src"]
mode, root = sys.argv[1], Path(sys.argv[2])
N, T, J, CROP = 18548, 160, 20, 6
THRESHOLD = 0.2574231028556824
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20)

if mode == "gen":
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026112001)
    users = sorted(json.load(open("tests/cpu/data/pose_temponet_cache_users_20261120.json"))["val"]["users"])
    assert len(users) == 15
    # 700..1540 windows per user, total 18548 (real minimum and maximum are both used)
    counts = np.array([700, 1540] + [0]*13)
    rest = N - 2240
    extra = rng.multinomial(rest - 13*900, np.ones(13)/13) + 900
    counts[2:] = extra
    assert counts.sum() == N and counts.min() >= 700 and counts.max() <= 1540, counts
    owner = np.repeat(np.arange(15), counts)
    meta = [{"user": users[owner[i]], "session": "s%d" % owner[i], "stage": "st%d" % (i % 7), "side": "left" if i % 2 else "right",
             "stem": "rec-%d" % (i // 10), "recording": i // 10, "start": 400 + 8000*(i % 10), "fs": 2000.0, "n_patches": T}
            for i in range(N)]
    rows = np.ones((N, T), bool)
    kind = rng.random(N)
    for i in np.flatnonzero(kind < 0.1005):            # no valid pose at all (real: 1864 of 18548)
        rows[i] = False
    for i in np.flatnonzero((kind >= 0.1005) & (kind < 0.40)):   # one valid segment
        a = rng.integers(0, T-2); b = rng.integers(a+1, T+1)
        rows[i] = False; rows[i, a:b] = True
    for i in np.flatnonzero((kind >= 0.40) & (kind < 0.50)):     # random gaps
        rows[i] = rng.random(T) > 0.2
    special = rng.choice(np.flatnonzero(kind >= 0.5), 40, replace=False)
    for i in special[:20]:                               # exactly one supported centre
        rows[i] = False; rows[i, rng.integers(CROP, T-CROP)] = True
    for i in special[20:]:                               # valid only in the cropped edges
        rows[i] = False; rows[i, :CROP] = True; rows[i, -CROP:] = True
    amp = np.exp(rng.normal(np.log(0.51), 0.55, size=(N, 1, J))).astype(np.float32)
    level = rng.normal(0, 0.9, size=(N, 1, J)).astype(np.float32)
    walk = np.cumsum(rng.standard_normal((N, T, J), dtype=np.float32), axis=1)
    walk -= walk.mean(axis=1, keepdims=True); walk /= walk.std(axis=1, keepdims=True)
    target = (level + amp*walk).astype(np.float32)
    mask = np.repeat(rows[..., None], J, axis=2)
    target[~mask] = 0
    pose_var = np.full(N, np.nan, np.float32)
    for i in range(N):
        if rows[i].sum() > 1:
            pose_var[i] = np.float32(target[i][rows[i]].var(axis=0, dtype=np.float32).mean(dtype=np.float32))
    std = rng.uniform(0.1, 0.6, size=(N, J)).astype(np.float32)
    centre = target[:, CROP:-CROP].astype(np.float64)
    ridge = 0.25*centre + 0.3*level + 0.55*amp*rng.standard_normal((N, T-2*CROP, J))
    head = (0.3*target + 0.25*level + 0.5*amp*rng.standard_normal((N, T, J), dtype=np.float32)).astype(np.float32)
    np.save(root/"target.npy", target); np.save(root/"rows.npy", rows); np.save(root/"pose_var.npy", pose_var)
    np.save(root/"std.npy", std); np.save(root/"ridge.npy", ridge); np.save(root/"head.npy", head)
    (root/"meta.json").write_text(json.dumps(meta))
    print("generated with numpy", np.__version__, "| dynamic share", float(np.mean(pose_var[np.isfinite(pose_var)] >= THRESHOLD)),
          "| windows per user", counts.tolist(), "| empty windows", int((~rows.any(axis=1)).sum()))
    sys.exit(0)

import pose_dynamics as HD
import tokenizer_pose_readout as HR
import consolidation_long_pose as HP
from pose_temponet import metrics as M, data as D

target = np.load(root/"target.npy"); rows = np.load(root/"rows.npy")
mask = np.repeat(rows[..., None], J, axis=2)
data = {"target": target, "mask": mask, "pose_var": np.load(root/"pose_var.npy"), "std": np.load(root/"std.npy"),
        "meta": json.loads((root/"meta.json").read_text()), "n_recordings": 1855, "key_time_valid": np.ones((N, T), bool),
        "pred": np.load(root/"head.npy")}
ridge = np.load(root/"ridge.npy")
head = data["pred"][:, CROP:-CROP].astype(np.float64)       # exactly what consolidation_long_pose_runtime._fit saved
print("numpy", np.__version__, "| python", sys.version.split()[0], "| arrays", target.dtype, mask.dtype, data["pose_var"].dtype, ridge.dtype, head.dtype)

def dyn(result):
    d = result["groups"]["dynamic"]; c = d["centered"]
    return c["centered_mse_ratio"], (d["n_selected_windows"], c["n_windows"], c["n_valid_elements"], c["n_window_joints"], c["n_users"])

# --- historical path: exactly the first statement of consolidation_long_pose.summarize (the E2 part does not enter the reference)
view = HP._target_view(data)
out = {}
for name, prediction in (("post_ridge", ridge), ("trained_head", head)):
    started = time.perf_counter()
    hist = HR.summarize(view, prediction)
    t_hist = time.perf_counter() - started
    started = time.perf_counter()
    new = M.evaluate(np.asarray(prediction, dtype=np.float64), data)
    t_new = time.perf_counter() - started
    e_h, c_h = dyn(hist); e_n, c_n = dyn(new)
    same_user_rows = hist["groups"]["dynamic"]["centered"]["per_user"] == new["groups"]["dynamic"]["centered"]["per_user"]
    all_groups = all(hist["groups"][g]["centered"] == new["groups"][g]["centered"] for g in ("all", "dynamic", "static"))
    print(f"{name}: historical E_dyn {e_h!r} | evaluate {e_n!r} | abs diff {abs(e_h-e_n):.3e} | counts equal {c_h == c_n} {c_n}"
          f" | per-user sums bitwise equal {same_user_rows} | all three groups' centered blocks equal {all_groups}"
          f" | support hash equal {hist['support_sha256'] == new['support_sha256']} | seconds hist {t_hist:.1f} new {t_new:.1f}", flush=True)
    out[name] = {"historical": e_h, "evaluate64": e_n, "counts": list(c_n)}
    # (1) batch size inside evaluate
    for batch in (1, 7, 32, 128, 1000, N):
        e_b, c_b = dyn(M.evaluate(prediction, data, batch_windows=batch))
        print(f"   batch_windows={batch:<6d} E_dyn {e_b!r} abs diff vs historical {abs(e_b-e_h):.3e} counts equal {c_b == c_h}", flush=True)
        out[name]["batch_%d" % batch] = e_b
    # (2) window order: the same windows, shuffled (users interleaved), batch 64
    order = np.random.default_rng(5).permutation(N)
    shuffled = {k: (v[order] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    shuffled["meta"] = [data["meta"][i] for i in order]
    e_s, c_s = dyn(M.evaluate(prediction[order], shuffled))
    print(f"   shuffled window order    E_dyn {e_s!r} abs diff vs historical {abs(e_s-e_h):.3e} counts equal {c_s == c_h}", flush=True)
    out[name]["shuffled"] = e_s
    # (3) summation order inside every window: time axis reversed (a different but legitimate order of the same 148 terms)
    flipped = dict(data); flipped["target"] = target[:, ::-1]; flipped["mask"] = mask[:, ::-1]; flipped["pred"] = data["pred"][:, ::-1]
    e_f, c_f = dyn(M.evaluate(np.ascontiguousarray(prediction[:, ::-1]), flipped))
    print(f"   time axis reversed       E_dyn {e_f!r} abs diff vs historical {abs(e_f-e_h):.3e} counts equal {c_f == c_h}", flush=True)
    out[name]["time_reversed"] = e_f
    # (4) the other historical formula (aggregate._metrics: numpy sum over sorted users, ratio of sums)
    per_user = hist["groups"]["dynamic"]["centered"]["per_user"]
    sums = {k: np.asarray([per_user[u]["sums"][k] for u in sorted(per_user)]) for k in ("centered", "vary")}
    e_a = float(sums["centered"].sum()/sums["vary"].sum())
    print(f"   aggregate-formula E_dyn  {e_a!r} abs diff vs historical {abs(e_a-e_h):.3e}", flush=True)
    out[name]["aggregate_formula"] = e_a
    # (5) float32 predictions handed over instead of float64 (what would happen if a cast were forgotten)
    if name == "trained_head":
        e_32, _ = dyn(M.evaluate(data["pred"][:, CROP:-CROP], data))
        print(f"   float32 input (cast by evaluate) E_dyn {e_32!r} abs diff {abs(e_32-e_h):.3e}", flush=True)

# --- the gate itself at the real size, with the constants of the historical path
policy = {"post_ridge_E_dyn": out["post_ridge"]["historical"], "trained_head_E_dyn": out["trained_head"]["historical"]}
policy.update(zip(("dynamic_selected_windows", "dynamic_centered_windows", "dynamic_centered_elements", "dynamic_window_joint_pairs", "dynamic_users"),
                  out["post_ridge"]["counts"]))
M.HISTORICAL_REFERENCE = policy
before = rss(); started = time.perf_counter()
gate = M.historical_reference_gate(data, ridge, head)
print("historical_reference_gate at real size: passed", gate["passed"], "| seconds", round(time.perf_counter()-started, 1),
      "| peak RSS GiB before/after", round(before, 2), round(rss(), 2), flush=True)
json.dumps(gate, allow_nan=False, sort_keys=True)
# a prediction shifted by one patch must be refused by the array comparison
try:
    M.historical_reference_gate(data, ridge, data["pred"][:, CROP+1:T-CROP+1].astype(np.float64))
    print("shifted head prediction: NOT refused")
except ValueError as error:
    print("shifted head prediction refused:", error)
(root/("result_numpy_%s.json" % np.__version__)).write_text(json.dumps(out, indent=1))
