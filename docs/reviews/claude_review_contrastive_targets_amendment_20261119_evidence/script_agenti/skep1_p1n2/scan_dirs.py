import json, subprocess, sys, collections
C = sys.argv[1]
files = subprocess.run(["git", "-C", C, "ls-files", "*.json"], capture_output=True, text=True, check=True).stdout.split()
def walk(o):
    if isinstance(o, dict):
        if {"model", "jepa", "loader"} <= set(o) and isinstance(o.get("loader"), dict):
            yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)
agg = collections.Counter()
for f in files:
    if not ("contrastive" in f or "T100-G" in f): continue
    try: d = json.load(open(f"{C}/{f}"))
    except Exception: continue
    for cfg in walk(d):
        top = "/".join(f.split("/")[:4])
        agg[(top, cfg.get("target_version", "<MISSING>"), "rms_window_ms" in cfg["loader"])] += 1
for k, v in sorted(agg.items()): print(v, k)
