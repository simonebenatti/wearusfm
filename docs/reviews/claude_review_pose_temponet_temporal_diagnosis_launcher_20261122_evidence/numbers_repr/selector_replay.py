"""Riapplica il selettore congelato (selection.choose / assessment) alle curve archiviate e confronta con i report."""
import json
from pathlib import Path
from pose_temponet import selection as S
L = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher")
ok = True
for w in ("linear", "temporal"):
    for h in ("L0", "MLP", "TCN_post"):
        r = json.loads((L/f"selection_{w}_{h}.json").read_text())
        curves = [S.curve_from_dict(c) for c in r["curves"]]
        a = S.assessment(curves)
        same = (a["selected"] == r["selected"] and a["optimizer_alarms"] == r["optimizer_alarms"] and
                a["minimum_exposures"] == r["minimum_exposures"] and a["objective_tradeoff"] == r["objective_tradeoff"] and
                a["optimization_unresolved"] == r["optimization_unresolved"])
        ok &= same
        nxt = None
        try: nxt = S.next_learning_rate(curves)
        except Exception as e: nxt = "errore: "+str(e)
        ext = {c.lr: (S.needs_epoch_extension(c) if c.horizon == 150 else "gia' a 300") for c in curves}
        print(f"{w:8s} {h:8s} replay==report {same} | selected {a['selected']} | alarms {a['optimizer_alarms']} | LR attivi {sorted(c.lr for c in curves)} | prossimo LR {nxt} | estensione dovuta {ext}")
print("tutti e sei coerenti:", ok)
t = json.loads((L/"selection_temporal_TCN_post.json").read_text())
c3 = next(c for c in t["curves"] if c["lr"] == 0.003)
cls, orc = 0.7284790286546209, 0.6706589239838878
print("LR 0,003, punti alle esposizioni di lettura proposte e vicino al minimo (pooled, esclusi, bersaglio rumoroso):")
for p in c3["points"]:
    if p["exposure"] in (1.0, 10.0, 27.0, 28.0, 31.0, 50.0, 75.0, 100.0, 150.0):
        print(f"   esp {p['exposure']:5.0f} passi {p['fold_steps']} MSE {p['mse']:.6f} E_dyn {p['edyn']:.6f} recupero {100*(cls-p['edyn'])/(cls-orc):7.2f}%")
c1 = next(c for c in t["curves"] if c["lr"] == 0.0001)
for p in c1["points"]:
    if p["exposure"] in (31.0, 150.0, 181.0, 201.0, 300.0):
        print(f"   LR 1e-4 esp {p['exposure']:5.0f} MSE {p['mse']:.6f} E_dyn {p['edyn']:.6f} recupero {100*(cls-p['edyn'])/(cls-orc):7.2f}%")
