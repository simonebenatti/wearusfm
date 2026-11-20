import json, sys
p = sys.argv[1]
d = json.load(open(p))
def show(o, depth=0, maxdepth=2, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)) and depth < maxdepth:
                print("  "*depth + f"{k}: ({type(v).__name__}, len={len(v)})")
                show(v, depth+1, maxdepth)
            else:
                s = json.dumps(v)
                print("  "*depth + f"{k}: {s[:200]}{'...' if len(s)>200 else ''}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]):
            s = json.dumps(v)
            print("  "*depth + f"[{i}]: {s[:200]}{'...' if len(s)>200 else ''}")
show(d, maxdepth=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
