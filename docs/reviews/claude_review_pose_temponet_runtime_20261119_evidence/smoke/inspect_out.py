import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for receipt in sorted((root/"receipts").glob("*.json")):
    r = json.loads(receipt.read_text())
    print(r["job_id"], r["stage"], r["status"], {k: v for k, v in r["gates"].items()},
          r["details"].get("error_type"), (r["details"].get("error") or "")[:200])
gate = root/"selection/L0/positive_gate.json"
if gate.exists():
    g = json.loads(gate.read_text())
    print("positive gate", g["status"], {h: v["status"] for h, v in g["linear_head_gates"].items()},
          g["temporal_head_gate"]["status"])
rep = root/"metrics/report.json"
if rep.exists():
    r = json.loads(rep.read_text())
    print("report complete", r.get("complete"), r.get("status"), r.get("reason"))
    if r.get("complete"):
        print("users", len(r["users"]), "contrasts", sorted(r["contrasts"]))
        for k, c in r["contrasts"].items():
            print(k, c["comparator"], c["relative_E_dyn"]["point"], c["relative_E_dyn"]["ci"], c["statistical_label"])
        print("candidacy", r["candidacy"]["candidate"], r["candidacy"]["name"])
inv = root/"collection/inventory.json"
if inv.exists():
    items = json.loads(inv.read_text())
    print("inventory files", len(items), "bytes", sum(i["bytes"] for i in items))
