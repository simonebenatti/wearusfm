"""Diagnosi del test distribuito fallito (CR, 4 processi): quale metrica differisce, di quanto, e se i gradienti coincidono.
Uso (dall'albero di db5ac41): PYTHONPATH=src:tests/cpu python diag_ddp.py <cartella di lavoro> [n_pose_rimosse]
"""
import sys, json
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
import test_contrastive_runtime as T
from test_contrastive_runtime import L, R, WearUsFM

def run(tmp, arm, world, drop=2, seed=3):
    tmp = Path(tmp) / f"{arm}_{world}_{drop}"; tmp.mkdir(parents=True, exist_ok=True)
    root, manifest, pose_root = T._pose_setup(tmp)
    cfg = T._config(pose_root, arm)
    cfg = replace(cfg, contrastive={**cfg.contrastive, "grad_every": 1})
    loader = L.PretrainLoader(R.load_index(manifest, [root], cfg.datasets, cfg.nested), cfg.loader, {}, None)
    batch = loader.batch(4, np.random.default_rng(seed))
    for i in range(drop):
        batch.pose[i] = None
    batch.rows = [{**row, "session": f"synthetic_session_{i}"} for i, row in enumerate(batch.rows)]
    torch.manual_seed(0)
    init = WearUsFM(cfg.model).double().state_dict()
    ref = T._global_update(cfg, init, batch, R.DistContext())
    out = tmp / "ddp"; out.mkdir(exist_ok=True)
    T._spawn_bounded(world, str(tmp / "store"), cfg, init, batch, str(out))
    vals = [torch.load(out / f"rank{r}.pt", weights_only=False) for r in range(world)]
    print(f"\n=== arm {arm}, world {world}, pose rimosse {drop}: finestre con posa {[p is not None for p in batch.pose]}")
    keys = sorted(set(ref["metrics"]) | set(vals[0]["metrics"]))
    for k in keys:
        a = ref["metrics"].get(k); b = [v["metrics"].get(k) for v in vals]
        if a is None or any(x is None for x in b):
            print(f"  {k:38s} ref {a} ddp {b}  <-- chiave mancante"); continue
        rel = max(abs(x - a) for x in b) / max(abs(a), 1e-300)
        flag = "  <--" if rel > 1e-9 and abs(max(b, key=lambda x: abs(x - a)) - a) > 1e-10 else ""
        print(f"  {k:38s} ref {a:.15g} ddp0 {b[0]:.15g} scarto rel max {rel:.2e}{flag}")
    worst = (0.0, None)
    for part in ("grads", "student", "teacher"):
        w = 0.0; wk = None
        for key, value in ref[part].items():
            for v in vals:
                d = float((v[part][key].double() - value.double()).abs().max()); s = float(value.double().abs().max())
                r = d / max(s, 1e-300)
                if d > 2e-9 + 2e-8 * s and r > w:
                    w, wk = r, key
        print(f"  {part}: peggior scarto relativo fuori tolleranza (rtol 2e-8, atol 2e-9): {w:.2e} {wk}")

if __name__ == "__main__":
    tmp = sys.argv[1]
    for arm, world in (("CR", 4), ("CR", 2), ("CN", 4), ("C0", 4)):
        run(tmp, arm, world)
    run(tmp, "CR", 4, drop=0)
