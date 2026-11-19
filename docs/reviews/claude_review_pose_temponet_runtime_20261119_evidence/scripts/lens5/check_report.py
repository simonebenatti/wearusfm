"""Independent checks of the final readout chain (lens 5). Read-only on the repo."""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/simonebenatti/dev/wearusfm/.claude/worktrees/review-pose-heads-2f98211")
sys.path.insert(0, str(ROOT / "tests" / "cpu"))
from pose_temponet import metrics as M  # noqa: E402
from pose_temponet import report as R  # noqa: E402
from test_pose_temponet_data import fixture  # noqa: E402

USERS = tuple(f"u{i:02d}" for i in range(15))
data, _ = fixture(users=USERS, windows=2)
data.pop("lat"); data.pop("pre")
# first window of every user dynamic, second static (historical threshold)
data["pose_var"][:] = np.tile([M.THRESHOLD, M.THRESHOLD - 1e-6], 15).astype(np.float32)
y = data["target"].astype(np.float64)
rng = np.random.default_rng(5)
noise = rng.normal(size=y.shape)
user_scale = np.repeat(np.linspace(0.8, 1.2, 15), 2)[:, None, None]


def indep_user_sse(pred):
    """Own implementation: crop6 support, mask, n>=2 per window/joint, dynamic windows."""
    support = M.central_support(data["mask"], M.lengths_for(data), data["key_time_valid"])
    yt = y[:, 6:-6]
    p = pred[:, 6:-6] if pred.shape[1] == 160 else pred
    sse, tss = {}, {}
    for w, meta in enumerate(data["meta"]):
        if not (np.isfinite(data["pose_var"][w]) and data["pose_var"][w] >= 0.2574231028556824):
            continue
        for j in range(20):
            m = support[w, :, j]
            if m.sum() < 2:
                continue
            pc = p[w, m, j] - p[w, m, j].mean()
            yc = yt[w, m, j] - yt[w, m, j].mean()
            sse[meta["user"]] = sse.get(meta["user"], 0.0) + float(((pc - yc) ** 2).sum())
            tss[meta["user"]] = tss.get(meta["user"], 0.0) + float((yc ** 2).sum())
    return np.array([sse[u] for u in USERS]), np.array([tss[u] for u in USERS])


def build(factors, ref_factors, seed_jitter=0.0, alarms=None, tradeoff=None, heterogeneous=False):
    cells, preds = {}, {}
    for head, c in factors.items():
        cells[head], preds[head] = {}, {}
        for s in R.SEEDS:
            scale = c * (1 + seed_jitter * (s - 1))
            if heterogeneous:
                scale = scale * (user_scale if head == "TCN_post" else 1)
            p = y + scale * noise
            preds[head][s] = p
            cells[head][s] = {"complete": True, "analysis": M.evaluate(p, data, batch_windows=32, per_joint=True)}
    refs, rpreds = {}, {}
    for h, c in ref_factors.items():
        p = (y + c * noise)[:, 6:-6]
        rpreds[h] = p
        refs[h] = {"complete": True, "analysis": M.evaluate(p, data, batch_windows=32)}
    assessments = {h: {"optimizer_alarms": list((alarms or {}).get(h, [])),
                       "objective_tradeoff": {"flag": bool((tradeoff or {}).get(h, False))}} for h in R.HEADS}
    gates = {g: True for g in R.COMMON_GATES}
    return cells, refs, assessments, gates, preds, rpreds


def run(**kw):
    cells, refs, assessments, gates, preds, rpreds = build(**kw)
    return R.aggregate(cells, refs, assessments=assessments, common_gates=gates), preds, rpreds


print("== 1. homogeneous scaling: degenerate bootstrap, exact gains and labels")
base = {"L0": 1.0, "L13": 1.0, "MLP": 1.0, "TCN_pre": 1.0, "TCN_post": 1.0}
fac = dict(base, L0=1.0, MLP=np.sqrt(0.97 / 0.97), L13=np.sqrt(0.97 / 0.99), TCN_post=np.sqrt(0.97))
# TCN E = 0.97 x (L0); MLP = TCN -> gain 0; L13: E_L13 = 0.99*E_L0 -> gain 100(1-0.97/0.99)=2.0202
fac["MLP"] = np.sqrt(0.97)  # equal to TCN
fac["L13"] = np.sqrt(0.99)
reff = {"historical_post_ridge": np.sqrt(0.97 / 0.95), "historical_trained_head": np.sqrt(0.97 / 0.951)}
result, preds, rpreds = run(factors=fac, ref_factors=reff)
assert result["complete"], result
for i in "12345":
    c = result["contrasts"][i]
    print(i, c["comparator"], round(c["relative_E_dyn"]["point"], 6),
          [round(x, 6) for x in c["relative_E_dyn"]["ci"]], c["statistical_label"], c["al_bordo_MC"])
print("candidacy", result["candidacy"]["candidate"], result["candidacy"]["name"],
      result["candidacy"]["practical_veto_contrasts"], result["candidacy"]["mixed_better_comparators"])

print("== 2. independent E_dyn/seed-average/bootstrap reproduction with heterogeneity and seed jitter")
fac2 = {"L0": 1.0, "L13": 0.97, "MLP": 0.98, "TCN_pre": 1.05, "TCN_post": 0.96}
reff2 = {"historical_post_ridge": 1.01, "historical_trained_head": 0.995}
result2, preds2, rpreds2 = run(factors=fac2, ref_factors=reff2, seed_jitter=0.05, heterogeneous=True)
assert result2["complete"], result2
draws = np.random.default_rng(20261102).integers(0, 15, (2000, 15))
cand = np.mean([indep_user_sse(preds2["TCN_post"][s])[0] for s in R.SEEDS], axis=0)
_, tss = indep_user_sse(preds2["TCN_post"][0])
worst = 0.0
for i, comp in enumerate(R.COMPARATORS, 1):
    if comp in preds2:
        ctrl = np.mean([indep_user_sse(preds2[comp][s])[0] for s in R.SEEDS], axis=0)
    else:
        ctrl = indep_user_sse(rpreds2[comp])[0]
    point = 100 * (ctrl.sum() / tss.sum() - cand.sum() / tss.sum()) / (ctrl.sum() / tss.sum())
    reps = np.array([100 * (ctrl[d].sum() - cand[d].sum()) / ctrl[d].sum() for d in draws])
    ci = np.percentile(reps, [0.5, 99.5])
    eff = result2["contrasts"][str(i)]["relative_E_dyn"]
    worst = max(worst, abs(eff["point"] - point), *np.abs(np.array(eff["ci"]) - ci))
    print(i, comp, "code", round(eff["point"], 9), np.round(eff["ci"], 9), "indep", round(point, 9), np.round(ci, 9))
print("max abs diff vs independent", worst)
ed = result2["heads"]["TCN_post"]["dynamic"]["mean_accumulators_metrics"]["E_dyn"]
print("TCN_post E_dyn mean-acc code", ed, "indep", cand.sum() / tss.sum())

print("== 3. label/candidacy paths on hand-built contrasts")
def mk(points):
    contrasts = {}
    for i, (g, lo, hi) in enumerate(points, 1):
        lab = R.outcome(g, lo, hi)
        contrasts[str(i)] = {"statistical_label": lab, "relative_E_dyn": {"evaluable": True, "point": g, "ci": [lo, hi]}}
    return contrasts
ok = {h: {"optimization_unresolved": False, "objective_tradeoff": False} for h in R.HEADS}
g = {k: True for k in R.COMMON_GATES}
cases = {
    "all pass": [(3, 1, 5), (3, 1, 5), (3, 1, 5), (3, 1, 5), (3, 1, 5)],
    "MLP resolved better, gain -1.5 (mixed)": [(3, 1, 5), (-1.5, -2.5, -0.5), (3, 1, 5), (3, 1, 5), (3, 1, 5)],
    "MLP harm (-2, upper<0) veto": [(3, 1, 5), (-2, -3, -1), (3, 1, 5), (3, 1, 5), (3, 1, 5)],
    "contrast 5 only CI>0, gain 1.9": [(3, 1, 5), (3, 1, 5), (3, 1, 5), (3, 1, 5), (1.9, 0.5, 3)],
    "contrast 2 inconclusive, others pass": [(3, 1, 5), (1, -1, 3), (3, 1, 5), (3, 1, 5), (3, 1, 5)],
    "L13 gain -3 but upper 0.5 (not harm)": [(3, 1, 5), (3, 1, 5), (-3, -6, 0.5), (3, 1, 5), (3, 1, 5)],
}
for name, pts in cases.items():
    c = mk(pts)
    out = R.candidacy(c, ok, g)
    print(f"{name:45s}", [c[str(i)]["statistical_label"] for i in range(1, 6)], out["candidate"], out["name"])

print("== 4. directional readings")
for lab in (R.BENEFIT, R.HARM, R.UNSUPPORTED, R.INCONCLUSIVE):
    for ca in (False, True):
        for co in (False, True):
            d = R.directional_reading(lab, ca, co)
            print(f"{lab:30s} tcn_alarm={ca!s:5s} comp_alarm={co!s:5s} -> sci={d['scientific_label']} veto_absence_usable={d['may_use_absence_of_practical_veto']}")

print("== 5. MC border flags")
for ci in ([0.1, 3], [0.1000001, 3], [-0.1, 3], [1.9, 5], [2.1, 5], [2.1000001, 5], [0.2, 1.95]):
    print(ci, R.monte_carlo_border(ci))
