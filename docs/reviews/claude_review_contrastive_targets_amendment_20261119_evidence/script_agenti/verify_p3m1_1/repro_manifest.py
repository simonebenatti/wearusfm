"""Independent check of P3-M1 through from_manifest (not compare). Synthetic data only."""
import copy, hashlib, json, sys, tempfile
from pathlib import Path
import numpy as np
clone = Path(sys.argv[1]); work = Path(sys.argv[2])
sys.path.insert(0, str(clone / "tests/cpu")); sys.path.insert(0, str(clone / "scripts"))
import test_contrastive_compare as T
import contrastive_compare as C

T100G = json.loads((clone / "results/infra/tokenizer_execution_20261103/19618/T100-G/receipt.json").read_text())["checkpoint_sha256"]
HIST = {"rms_window_ms": 25.0, "ms_fast_window_ms": 50.0, "target_version": "step1_psd_power_shape_energy_v1"}
NEW = {"rms_window_ms": 100.0, "ms_fast_window_ms": 125.0, "target_version": "step1_psd_power_shape_energy_fast125_v3"}

def build(d, ckpt, cell_recipe, top_extra):
    d.mkdir(parents=True, exist_ok=True)
    cl, po = T.inputs(); recs = T.audits()
    for arm in C.ARMS:
        po[arm]["checkpoint_sha256"] = ckpt[arm]
        if cell_recipe: po[arm]["config"] = {"loader": {k: v for k, v in cell_recipe[arm].items() if k != "target_version"}, "target_version": cell_recipe[arm]["target_version"]}
    for r in recs: r["checkpoint_sha256"] = ckpt[r["arm"]]
    m = {"protocol": C.PROTOCOL, "step": 19618, "which": "teacher", "pose": {}, "classification": {}, **top_extra}
    pin = lambda p: {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for arm in C.ARMS:
        p = d / f"{arm}.pose.json"; p.write_text(json.dumps(po[arm])); m["pose"][arm] = pin(p)
        m["classification"][arm] = {}
        for task, dss in C.TASKS.items():
            p = d / f"{arm}.{task}.npz"; np.savez(p, **{f"{ds}_{k}": v for ds in dss for k, v in cl[arm][ds].items()})
            a = d / f"{arm}.{task}.audit.jsonl"
            a.write_text("\n".join(json.dumps(r) for r in recs if r["arm"] == arm and r["task"] == task) + "\n")
            m["classification"][arm][task] = {"oof": pin(p), "audit": pin(a)}
    mp = d / "manifest.json"; mp.write_text(json.dumps(m)); return mp

def run(name, mp):
    try:
        r = C.from_manifest(mp)
    except ValueError as e:
        print(f"[{name}] REJECTED: {e}"); return None
    def has(o, s):
        return s in json.dumps(o)
    print(f"[{name}] ACCEPTED decision_valid={r['decision_valid']} protocol={r['protocol']} reading={r['reading']}")
    print(f"   output top-level keys mentioning amend/recipe/cohort/target/config: {sorted(k for k in r if any(s in k for s in ('amend','recipe','cohort','target','config')))}")
    print(f"   amendment string anywhere in output: {has(r, 'contrastive-target-windows-20261118-v1')}; manifest echoed as input_pins: {'input_pins' in r}; input_manifest_sha256 present: {'input_manifest_sha256' in r}")
    print(f"   pose report per-arm config retained in output: {has(r['pose'], 'rms_window_ms')}")
    return r

ck_mixed = {"C0": T100G, "CN": hashlib.sha256(b"CN-new").hexdigest(), "CR": hashlib.sha256(b"CR-new").hexdigest()}
# 1: T100-G (historical 25/50/v1) checkpoint labelled C0, CN/CR 100/125/v3, manifest declares the amendment (false claim for C0)
run("mixed_T100G_as_C0_with_amendment_field", build(work / "case1", ck_mixed, {"C0": HIST, "CN": NEW, "CR": NEW}, {"amendment": "contrastive-target-windows-20261118-v1"}))
# 2: same mixture, manifest without any amendment/cohort field
run("mixed_T100G_as_C0_no_amendment", build(work / "case2", ck_mixed, {"C0": HIST, "CN": NEW, "CR": NEW}, {}))
# 3: homogeneous new cohort, no amendment: indistinguishable protocol field from case 2
run("homogeneous_new_no_amendment", build(work / "case3", {a: hashlib.sha256(a.encode()).hexdigest() for a in C.ARMS}, {a: NEW for a in C.ARMS}, {}))
# 4: sanity, the one identity check that exists: pose checkpoint differs from classification audit checkpoint
d = work / "case4"; mp = build(d, ck_mixed, None, {})
pose = json.loads((d / "C0.pose.json").read_text()); pose["checkpoint_sha256"] = "e" * 64
(d / "C0.pose.json").write_text(json.dumps(pose)); m = json.loads(mp.read_text())
m["pose"]["C0"]["sha256"] = hashlib.sha256((d / "C0.pose.json").read_bytes()).hexdigest(); mp.write_text(json.dumps(m))
run("pose_vs_audit_checkpoint_mismatch", mp)
print("T100-G 19618 checkpoint sha used:", T100G)
