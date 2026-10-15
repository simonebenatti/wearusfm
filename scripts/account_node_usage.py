"""Nodi GPU in uso dall'account IscrB_WearUsFM nel tempo, da un'uscita di sacct (letta il 15/10/2026 per il 14/10): massimo contemporaneo,
ore per numero di nodi in uso e, per i job di sbenatti con attese lunghe, quanti nodi usava l'account durante l'attesa."""
from collections import defaultdict
from datetime import datetime

import sys

# uso: python scripts/account_node_usage.py <sacct -A ... --allusers -X -P --format=JobID,User,Partition,QOS,Submit,Start,End,NNodes,AllocTRES,State> [fine ISO]
F = sys.argv[1]
END = datetime.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else datetime(2026, 10, 15, 6, 0)
rows = [l.rstrip("\n").split("|") for l in open(F)][1:]
ev = []
users = defaultdict(int)
for jid, user, part, qos, sub, start, end, nn, tres, state in rows:
    if not part.startswith("boost") or start in ("Unknown", "None", ""):
        continue
    s = datetime.fromisoformat(start)
    e = END if end in ("Unknown", "None", "") else datetime.fromisoformat(end)
    gpus = 0
    for kv in tres.split(","):
        if kv.startswith("gres/gpu="):
            gpus = int(kv.split("=")[1])
    ev.append((s, int(nn), gpus, user, jid))
    ev.append((e, -int(nn), -gpus, user, jid))
ev.sort(key=lambda x: (x[0], x[1]))
cur_n = cur_g = 0
best = (0, None)
peak_by_user = defaultdict(int)
cur_user = defaultdict(int)
timeline = []
for t, dn, dg, user, jid in ev:
    cur_n += dn
    cur_g += dg
    cur_user[user] += dn
    timeline.append((t, cur_n, cur_g, dict(cur_user)))
    if cur_n > best[0]:
        best = (cur_n, t, dict(cur_user), cur_g)
print("massimo di nodi in uso contemporaneamente:", best[0], "alle", best[1], "per utente", best[2], "GPU", best[3])
# distribuzione del tempo per numero di nodi in uso
dur = defaultdict(float)
for (t0, n, g, _), (t1, *_r) in zip(timeline, timeline[1:]):
    dur[n] += (t1 - t0).total_seconds() / 3600
print("ore per numero di nodi in uso:", {k: round(v, 1) for k, v in sorted(dur.items()) if v > 0.05})
# job di sbenatti in boost con attesa lunga (submit -> start)
for jid, user, part, qos, sub, start, end, nn, tres, state in rows:
    if user == "sbenatti" and part.startswith("boost") and start not in ("Unknown", "None", ""):
        w = (datetime.fromisoformat(start) - datetime.fromisoformat(sub)).total_seconds() / 60
        if w > 20:
            n_at = max((n for t, n, g, u in timeline if t <= datetime.fromisoformat(start)), default=0)
            busy = [n for t, n, g, u in timeline if datetime.fromisoformat(sub) <= t <= datetime.fromisoformat(start)]
            print(f"attesa {w:5.0f} min job {jid} {qos} nodi {nn} {tres.split(',')[2] if ',' in tres else tres}: nodi dell'account in uso durante l'attesa "
                  f"min {min(busy) if busy else '-'} max {max(busy) if busy else '-'}")
