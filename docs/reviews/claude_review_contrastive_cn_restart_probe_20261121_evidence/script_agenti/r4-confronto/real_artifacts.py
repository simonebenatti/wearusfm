import json, traceback
from pathlib import Path
import contrastive_cn_restart_compare as C
P = C.PACKAGE
CN = P/"benchmark_execution_20261121/CN"; A = P/"reloads_execution_20261121/CN_A"; B = P/"reloads_execution_20261121/CN_B"
def tryit(label, f):
    try:
        r = f(); print("OK", label, r if not isinstance(r, dict) or len(str(r))<400 else str(r)[:400])
    except Exception as e:
        print("EXC", label, type(e).__name__, e)
for r in range(4):
    tryit(f"branch_traces(A, CN, warm, {r})", lambda: C.branch_traces(A, CN, "warm", r)["update"])
    tryit(f"branch_traces(B, A, ordinary, {r})", lambda: C.branch_traces(B, A, "ordinary", r)["update"])
m = C.metrics_evidence(A, CN, "warm")
print("warm metrics diffs", len(m["differences"]), "first steps", sorted({d["step"] for d in m["differences"]})[:5], "fields at 545", [ (d["field"], d["actual"], d["reference"]) for d in m["differences"] if d["step"]==545])
tryit("metrics_evidence(B, A, ordinary)", lambda: len(C.metrics_evidence(B, A, "ordinary")["differences"]))
tryit("metrics_evidence(A, A?) not distinct", lambda: 0)
tryit("history(A, CN)", lambda: C.history(A, CN))
