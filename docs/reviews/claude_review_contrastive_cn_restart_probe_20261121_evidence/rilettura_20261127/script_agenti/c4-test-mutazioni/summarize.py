import json, sys, collections
from pathlib import Path
W = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (W / "mutation_log.txt").read_text().splitlines() if l.startswith("{")]
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    g = r["id"][0]
    by[g][0] += 1
    by[g][1] += not r["killed"]
print("total", len(rows), "survived", sum(not r["killed"] for r in rows))
for g, (n, s) in sorted(by.items()):
    print(g, n, "survived", s)
cat = collections.Counter((r["category"], r["killed"]) for r in rows)
print(cat)
for r in sorted(rows, key=lambda r: r["id"]):
    if not r["killed"]:
        print("SURV", r["id"], r["category"], r["label"])
