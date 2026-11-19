"""Scan every tracked JSON in the clone for run-config-like dicts (keys model+jepa+loader) and report target_version presence."""
import json, subprocess, sys
C = sys.argv[1]
files = subprocess.run(["git", "-C", C, "ls-files", "*.json"], capture_output=True, text=True, check=True).stdout.split()
def walk(obj, path):
    if isinstance(obj, dict):
        if {"model", "jepa", "loader"} <= set(obj) and isinstance(obj.get("loader"), dict):
            yield path, obj
        for k, v in obj.items():
            yield from walk(v, path + "/" + str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + f"[{i}]")
tot = 0; missing = []; by_tv = {}; line3 = {"tot": 0, "missing": 0, "tv": {}}
for f in files:
    try:
        data = json.load(open(f"{C}/{f}"))
    except Exception:
        continue
    for path, cfg in walk(data, ""):
        tot += 1
        tv = cfg.get("target_version", "<MISSING>")
        by_tv[tv] = by_tv.get(tv, 0) + 1
        if "contrastive" in f or "T100-G" in f:
            line3["tot"] += 1; line3["tv"][tv] = line3["tv"].get(tv, 0) + 1
            if tv == "<MISSING>": line3["missing"] += 1
        if tv == "<MISSING>":
            missing.append((f, path))
print("run-config-like dicts:", tot)
print("by target_version:", json.dumps(by_tv, indent=1))
print("line-3 (path contains 'contrastive' or 'T100-G'):", json.dumps(line3))
print("missing target_version (first 30):")
for m in missing[:30]: print("  ", m)
print("n missing:", len(missing))
