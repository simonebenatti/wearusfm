"""Third pass, R-1b: apply GPT's numeric_compatibility (f8cb53f) to real Leonardo contexts.

Usage: python claude_review_consolidation_long_20261105_rule_b.py <GPT checkout root>
Reads the pinned historical P1/P2 reports and GPT's login-node preflight from the checkout,
and my own import-only probes from the JSON next to this script. No job, no fit."""
import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import consolidation_long_classification as M
import consolidation_long_classification_runtime as R

base = ROOT / "results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/completed_aggregates/T100-G"
hist = {t: json.loads((base / f"{t}.json").read_text()) for t in ("p1", "p2")}
assert all(M.jhash(h["numeric_context"]) == h["numeric_sha256"] for h in hist.values())
assert hist["p1"]["numeric_sha256"] == hist["p2"]["numeric_sha256"]
old = hist["p1"]["numeric_context"]
reference = {"numeric_sha256": hist["p1"]["numeric_sha256"], "numeric_context": old, "reports": {t: [] for t in M.TASKS}}
print("historical numeric_sha256:", reference["numeric_sha256"], "| pools:", len(old["threadpools"]))

rows = []
gpt = json.loads((ROOT / "results/infra/consolidation_long_20261105/numeric_preflight_20261105.json").read_text())
rows += [("GPT " + p["probe"], p["hostname"], p["numeric_context"]) for p in gpt["probes"]]
mine = json.loads(Path(__file__).with_name("claude_review_consolidation_long_20261105_numeric_probe.json").read_text())
rows += [("Claude " + p["probe"], p["hostname"], p["numeric_context"]) for p in mine["probes"]]
hashes = set()
for name, host, ctx in rows:
    v = R._numeric_match(ctx, reference)   # same helper the benchmark, fit and compare stages call
    assert v == {**M.numeric_compatibility(ctx, old), "historical_reports": reference["reports"]}
    if not v["exact_match"]:
        hashes.add(v["numeric_sha256"])
    print(f"{name:40s} {host.split('.')[0]:8s} pools {len(ctx['threadpools'])} hash {v['numeric_sha256'][:12]} "
          f"compatible {v['compatible']} exact {v['exact_match']} extra {[p['prefix'] for p in v['extra_threadpools']]}")
print("distinct hashes among torch-loaded processes:", len(hashes))
