import json
from pathlib import Path
from pose_temponet import selection as S
L = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher")
norm = lambda v: json.loads(json.dumps(v))
allok = True
for w in ("linear", "temporal"):
    for h in ("L0", "MLP", "TCN_post"):
        r = json.loads((L/f"selection_{w}_{h}.json").read_text())
        a = norm(S.assessment([S.curve_from_dict(c) for c in r["curves"]]))
        diffs = [k for k in ("selected", "optimizer_alarms", "minimum_exposures", "objective_tradeoff", "optimization_unresolved", "lower_edge_relative_mse_gain") if a[k] != r[k]]
        allok &= not diffs
        print(w, h, "campi diversi:", diffs)
print("replay del selettore congelato == report archiviati per tutti e sei:", allok)
