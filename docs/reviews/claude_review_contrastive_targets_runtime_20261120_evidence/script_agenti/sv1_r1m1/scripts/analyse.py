import json, sys
from pathlib import Path
base = Path(sys.argv[1])
pairs = [t.split(":") for t in sys.argv[2:]]  # tagA:tagO
for ta, to in pairs:
    A, O = base / ("run_" + ta), base / ("run_" + to)
    ra = [json.loads(l) for l in (A / "metrics.jsonl").read_text().splitlines() if l.strip()]
    ro = [json.loads(l) for l in (O / "metrics.jsonl").read_text().splitlines() if l.strip()]
    sa, so = json.loads((A / "summary.json").read_text()), json.loads((O / "summary.json").read_text())
    lo = json.loads((O / "sv1.rank0.json").read_text())
    la = json.loads((A / "sv1.rank0.json").read_text())
    calls = {c["update"]: c for c in lo["measure_calls"] if c["measured_point"]}
    print(f"== {ta} vs {to}: exceptions A={la.get('exception')} O={lo.get('exception')}; steps {sa['steps']}/{so['steps']} stopped {sa['stopped']}/{so['stopped']}")
    print("  calls to measure on rank0 (update, s, point):", [(c["update"], round(c["seconds"], 3), c["measured_point"]) for c in lo["measure_calls"]])
    for a, o in zip(ra, ro):
        m = calls.get(o["step"], {}).get("seconds", 0.)
        print(f"  step {o['step']}: t_step_s A {a['t_step_s']:.3f}  O {o['t_step_s']:.3f}  measure(rank0) {m:.3f}  O-measure {o['t_step_s']-m:.3f}")
    tot = sum(c["seconds"] for c in calls.values())
    print(f"  elapsed_s A {sa['elapsed_s']:.3f} O {so['elapsed_s']:.3f} diff {so['elapsed_s']-sa['elapsed_s']:.3f}; sum measure(rank0) {tot:.3f}; O-sum {so['elapsed_s']-tot:.3f}")
    par = json.loads((O / "zero_weight_parity.json").read_text())
    obs = [[r["observer_seconds_including_setup"] for r in p["ranks"]] for p in par["points"]]
    print("  observer_seconds_including_setup per point/rank:", [[round(x, 3) for x in row] for row in obs])
    print("  parity status:", par["status"], "gate errors:", [p.get("errors") for p in par["gate"]["points"]],
          "inconclusive:", [p.get("inconclusive") for p in par["gate"]["points"]])
