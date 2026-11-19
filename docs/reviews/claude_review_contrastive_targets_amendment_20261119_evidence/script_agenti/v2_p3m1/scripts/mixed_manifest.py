"""Skeptic 2 for P3-M1: independent reproduction on contrastive_compare.from_manifest.

Builds pinned manifests in a temp dir (argv[2]) using the clone's comparator (argv[1] = clone root).
Cases:
  A  mixed recipe annotations in pose cells (C0 25/50/v1, CN/CR 100/125/v3), plain v2 manifest
  B  same as A plus extra top-level keys (amendment, cohort pin, per-arm configs) -> tolerated? echoed?
  C  a pose cell whose internal arm is "T100-G" placed under key C0 -> rejected?
  D  audit rows labelled arm "T100-G" under C0 -> rejected?
"""
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np

root, work = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root / "scripts"))
import contrastive_compare as C

work.mkdir(parents=True, exist_ok=True)
sha = lambda b: hashlib.sha256(b).hexdigest()


def cells():
    cl, po, au = {}, {}, []
    for arm in C.ARMS:
        ck = sha(("ckpt-" + arm).encode())
        cl[arm] = {}
        for ds in C.DATASETS:
            y = np.tile(np.array([0, 1, 0, 1]), 4)
            pred = y.copy(); pred[[0, 4, 8, 12]] = 1
            cl[arm][ds] = {"row": np.arange(16), "y": y, "pred": pred,
                           "s": np.repeat(np.arange(4), 4), "fold": np.repeat(np.arange(4), 4)}
        users = {f"u{i}": {"n_windows": 2, "n_window_joints": 40, "n_valid_elements": 400,
                           "sums": {"vary": 10., "centered": 8.}} for i in range(4)}
        po[arm] = {"complete": True, "arm": arm, "checkpoint_step": 19618, "which": "teacher",
                   "checkpoint_sha256": ck, "fit": {"selection": {"boundary_unresolved": False}},
                   "analysis": {"support_sha256": "a" * 64, "targets_sha256": "b" * 64, "dynamic_threshold": .05,
                                "groups": {"dynamic": {"n_selected_windows": 8, "centered": {"per_user": users}}}}}
        for task, dss in C.TASKS.items():
            for ds in dss:
                for fold in range(4):
                    for phase, grid in (("candidate", C.C_GRID), ("refit", [1.])):
                        for c in grid:
                            au.append({"arm": arm, "task": task, "dataset": ds, "fold": fold, "checkpoint_sha256": ck,
                                       "fit_phase": phase, "C": c, "n_iter": 5, "max_iter": 12000, "exception": None,
                                       "convergence_warnings": 0, "converged_by_criterion": True})
    return cl, po, au


def write_manifest(d, cl, po, au, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    pin = lambda p: {"path": p.name, "sha256": sha(p.read_bytes())}
    m = {"protocol": C.PROTOCOL, "step": 19618, "which": "teacher", "pose": {}, "classification": {}}
    for arm in C.ARMS:
        p = d / f"{arm}.pose.json"; p.write_text(json.dumps(po[arm])); m["pose"][arm] = pin(p)
        m["classification"][arm] = {}
        for task, dss in C.TASKS.items():
            p = d / f"{arm}.{task}.npz"
            np.savez(p, **{f"{ds}_{k}": v for ds in dss for k, v in cl[arm][ds].items()})
            a = d / f"{arm}.{task}.audit.jsonl"
            a.write_text("\n".join(json.dumps(r) for r in au if r["arm"] == arm and r["task"] == task) + "\n")
            m["classification"][arm][task] = {"oof": pin(p), "audit": pin(a)}
    if extra:
        m.update(extra(d, pin))
    mp = d / "manifest.json"; mp.write_text(json.dumps(m))
    return mp


def run(label, mp):
    try:
        r = C.from_manifest(mp)
        flat = json.dumps(r)
        print(f"{label}: ACCEPTED protocol={r['protocol']} decision_valid={r['decision_valid']} reading={r['reading']}")
        print(f"   top-level keys mention recipe: {[k for k in r if any(t in k for t in ('amend','cohort','recipe','target','rms','fast'))]}")
        print(f"   input_pins keys: {sorted(r['input_pins'])}")
        print(f"   'contrastive-target-windows-20261118-v1' in output: {'contrastive-target-windows-20261118-v1' in flat}")
        return r
    except ValueError as e:
        print(f"{label}: REJECTED ValueError: {e}")


hist = {"rms_window_ms": 25.0, "ms_fast_window_ms": 50.0, "target_version": "step1_psd_power_shape_energy_v1"}
new = {"rms_window_ms": 100.0, "ms_fast_window_ms": 125.0, "target_version": "step1_psd_power_shape_energy_fast125_v3"}

# A: mixed annotations inside cells
cl, po, au = cells()
po["C0"]["recipe"] = dict(hist); po["CN"]["recipe"] = dict(new); po["CR"]["recipe"] = dict(new)
run("A mixed-recipe cells, plain v2 manifest", write_manifest(work / "A", cl, po, au))

# B: extra top-level keys carrying amendment + cohort/config pins
def extra(d, pin):
    coh = d / "cohort.json"; coh.write_text(json.dumps({"arms": list(C.ARMS), **new}))
    out = {"amendment": "contrastive-target-windows-20261118-v1", "cohort": pin(coh), "configs": {}}
    for arm in C.ARMS:
        cf = d / f"{arm}.config.json"; cf.write_text(json.dumps({"loader": hist if arm == "C0" else new}))
        out["configs"][arm] = pin(cf)
    return out
cl, po, au = cells()
rb = run("B extra keys amendment/cohort/configs (C0 config deliberately 25/50)", write_manifest(work / "B", cl, po, au, extra))
if rb:
    print("   echoed amendment:", rb["input_pins"].get("amendment"), "| cohort pin echoed:", "cohort" in rb["input_pins"])

# C: historical-style pose cell labelled T100-G under key C0
cl, po, au = cells()
po["C0"]["arm"] = "T100-G"
run("C pose cell arm=T100-G under key C0", write_manifest(work / "C", cl, po, au))

# D: audit rows labelled T100-G under key C0
cl, po, au = cells()
for r in au:
    if r["arm"] == "C0":
        r["arm"] = "T100-G"
mp = write_manifest(work / "D", cl, po, [r if r["arm"] != "T100-G" else r for r in au])
# write_manifest filters rows by r["arm"] == arm; write C0 files explicitly with relabelled rows
for task in C.TASKS:
    a = work / "D" / f"C0.{task}.audit.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in au if r["arm"] == "T100-G" and r["task"] == task) + "\n")
m = json.loads(mp.read_text())
for task in C.TASKS:
    a = work / "D" / f"C0.{task}.audit.jsonl"
    m["classification"]["C0"][task]["audit"] = {"path": a.name, "sha256": sha(a.read_bytes())}
mp.write_text(json.dumps(m))
run("D audit rows arm=T100-G under key C0", mp)

# E: same checkpoint SHA reused for C0 and CN (the comparator does not require distinct checkpoints)
cl, po, au = cells()
same = po["CN"]["checkpoint_sha256"]
po["C0"]["checkpoint_sha256"] = same
for r in au:
    if r["arm"] == "C0":
        r["checkpoint_sha256"] = same
run("E C0 and CN pinned to the same checkpoint sha", write_manifest(work / "E", cl, po, au))
