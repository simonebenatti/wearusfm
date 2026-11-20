"""Positive control L3: real bind_review -> real make_job -> real build_command (audit cell)."""
import copy, json, os, sys, traceback
from pathlib import Path
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
import contrastive_release as C
import contrastive_job as J
root = C.ROOT
commit = sys.argv[1]
binding = C.bind_review(commit, "docs/reviews/sk1r57_review.md", root / "docs/reviews/sk1r57_review.md")
print("bind_review ok; review pin:", binding["review"])
fresh = json.loads(C.PREFLIGHT.read_text())["fresh_root"]
job = C.make_job(binding, "CONTRAST-AUDIT-targets-v3-20261118", {}, fresh + "/sk1r57_audit_probe")
print("make_job ok; keys:", sorted(job))
def attempt(label, value):
    try:
        J.build_command(value, "cpu", 7200)
        print(label, "-> command built")
    except Exception as exc:
        print(label, "->", type(exc).__name__, exc)
attempt("original make_job output", job)
mutated = copy.deepcopy(job); del mutated["review"]   # what L3_makejob_drop_review would emit
attempt("L3 output (no review)", mutated)
other = copy.deepcopy(job); other["arm"] = "C0"      # e.g. arm dropped/changed by a producer drift
attempt("arm drift", other)
