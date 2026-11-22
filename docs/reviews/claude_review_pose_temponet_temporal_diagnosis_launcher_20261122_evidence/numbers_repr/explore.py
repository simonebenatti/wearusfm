import json, sys
from pathlib import Path
L = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher")
def show(obj, depth=0, maxdepth=4, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and depth < maxdepth:
                desc = f"dict[{len(v)}]" if isinstance(v, dict) else f"list[{len(v)}]"
                print("  "*depth + f"{k}: {desc}")
                show(v, depth+1, maxdepth)
            else:
                s = json.dumps(v)
                print("  "*depth + f"{k}: {s[:140]}")
    elif isinstance(obj, list):
        if obj:
            print("  "*depth + f"[0] of {len(obj)}:")
            if isinstance(obj[0], (dict, list)) and depth < maxdepth:
                show(obj[0], depth+1, maxdepth)
            else:
                print("  "*(depth+1) + json.dumps(obj[0])[:140])
name = sys.argv[1]; md = int(sys.argv[2]) if len(sys.argv) > 2 else 4
show(json.loads((L/name).read_text()), maxdepth=md)
