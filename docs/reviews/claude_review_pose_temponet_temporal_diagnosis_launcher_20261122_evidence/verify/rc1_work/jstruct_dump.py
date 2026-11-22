import json, sys
B = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/"
def show(o, depth=0, maxdepth=6, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                n = len(v)
                print("  "*depth + f"{k}: {type(v).__name__}[{n}]")
                if depth < maxdepth:
                    show(v, depth+1, maxdepth, path+"/"+k)
            else:
                s = repr(v)
                print("  "*depth + f"{k}: {s[:100]}")
    elif isinstance(o, list):
        if o and isinstance(o[0], (dict, list)):
            print("  "*depth + f"[0] of {len(o)}:")
            if depth < maxdepth:
                show(o[0], depth+1, maxdepth, path+"/0")
        else:
            print("  "*depth + repr(o)[:160])
name = sys.argv[1]
maxdepth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
d = json.load(open(B+name))
show(d, 0, maxdepth)
