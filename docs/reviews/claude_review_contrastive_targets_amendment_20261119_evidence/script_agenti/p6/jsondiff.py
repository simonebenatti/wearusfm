import json, subprocess, sys
repo, a_ref, b_ref = sys.argv[1:4]
def load(ref):
    c, p = ref.split(":", 1)
    return json.loads(subprocess.run(["git", "-C", repo, "show", f"{c}:{p}"], capture_output=True, check=True).stdout)
def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: print("ADDED  ", path + "/" + k, "=", json.dumps(b[k], ensure_ascii=False)[:300])
            elif k not in b: print("REMOVED", path + "/" + k, "=", json.dumps(a[k], ensure_ascii=False)[:300])
            else: walk(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]")
    elif a != b or type(a) != type(b):
        print("CHANGED", path, ":", json.dumps(a, ensure_ascii=False)[:300], "->", json.dumps(b, ensure_ascii=False)[:300])
walk(load(a_ref), load(b_ref))
