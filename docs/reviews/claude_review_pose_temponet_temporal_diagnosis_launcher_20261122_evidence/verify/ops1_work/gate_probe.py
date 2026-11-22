"""Riproduce il cancello di quota di A1 con il codice congelato e le ricevute archiviate (solo CPU, nessuna scrittura nell'albero)."""
import json, sys
from pathlib import Path
from pose_temponet import resources
from pose_temponet import budget_amendment as A1

ROOT = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1")
L = ROOT / "launcher"
cpu = json.load(open(L / "TCNH-cpu-benchmark-20261120.json"))["details"]["benchmark"]
gpu = json.load(open(L / "TCNH-gpu-benchmark-20261120.json"))["details"]["benchmark"]
plan = json.load(open(ROOT / "plan_controls.json"))
checks = plan["operator_checks"]
free, pending = checks["work_free_bytes"], checks["pending_write_bytes"]
own = gpu["persistent_output_bytes"]
caps = A1.effective_caps(plan)
GIB = 1024**3
P20 = 1907*148*20*4*20
P24 = 1907*148*20*4*24
out = {"free": free, "pending_others": pending, "floor": 12*GIB, "margin_over_floor": free - pending - 12*GIB,
       "a1_projected_own_output": own, "pred20": P20, "pred24": P24,
       "own_output_over_margin_ratio": own / (free - pending - 12*GIB)}

def run(label, *, free, pending, own):
    g = resources.gate_and_pack(cpu, gpu, work_free_bytes=free, pending_write_bytes=pending,
                                persistent_output_bytes=own, budget_caps=caps)
    out[label] = {"free": free, "pending": pending, "own_output": own, "all_passed": g["all_passed"],
                  "quota": g["gates"]["quota"], "persistent_output": g["gates"]["persistent_output"]}

# 1) A1 cosi' come e' stato rilasciato (deve coincidere con plan_controls.json["resource_gate"])
run("A1_as_released", free=free, pending=pending, own=own)
out["archived_gate"] = {k: plan["resource_gate"]["gates"][k] for k in ("quota", "persistent_output")}
# 2) uscite proprie aumentate delle predizioni diagnostiche (lettura del protocollo: stanno negli 8 GiB di uscite)
run("own_output_plus_pred20", free=free, pending=pending, own=own + P20)
run("own_output_plus_pred24", free=free, pending=pending, own=own + P24)
# 3) lettura dell'altro revisore: le predizioni come scritture pendenti in piu'
run("pred20_counted_as_pending", free=free, pending=pending + P20, own=own)
run("pred24_counted_as_pending", free=free, pending=pending + P24, own=own)
# 4) dopo la scrittura (spazio libero ridotto), cancello di A1 riletto con le stesse pendenti
run("after_write_pred20", free=free - P20, pending=pending, own=own)
run("after_write_pred24", free=free - P24, pending=pending, own=own)
# 5) riduzione all'assurdo: se le uscite proprie dovessero stare nel margine oltre i 12 GiB
out["reductio_own_output_fits_margin"] = own <= free - pending - 12*GIB
# 6) quanta parte degli 8 GiB di uscite resterebbe
out["own_cap_8gib"] = 8*GIB
out["cap_left_after_A1_full_plus_pred24"] = 8*GIB - own - P24
out["deficit_if_pred20_pending"] = 12*GIB + pending + P20 - free
out["deficit_if_pred24_pending"] = 12*GIB + pending + P24 - free
json.dump(out, sys.stdout, indent=1, sort_keys=True)
print()
