import json, sys
from pathlib import Path
base = Path(sys.argv[1])
for tag in sys.argv[2:]:
    run = base / f"run_{tag}"
    recs = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    s = json.loads((run / "summary.json").read_text())
    meas = [json.loads((base / f"timing_{tag}.rank{r}.json").read_text())["measure_seconds_by_update"] for r in range(4)]
    line = f"{tag}: steps {[r['step'] for r in recs]} t_step_s {[round(r['t_step_s'], 2) for r in recs]} diag {['contrast_global_grad_base' in r for r in recs]} elapsed_s {s['elapsed_s']:.2f} sum_t_step {sum(r['t_step_s'] for r in recs):.2f}"
    print(line)
    if any(meas):
        for u in ("2", "4"):
            print(f"   measure() wall per rank at update {u}: {[round(m[u], 2) for m in meas]} max {max(m[u] for m in meas):.2f}")
        rep = json.loads((run / "zero_weight_parity.json").read_text())
        for p in rep["points"]:
            print(f"   report update {p['update']}: observer_seconds_including_setup per rank {[round(r['observer_seconds_including_setup'], 2) for r in sorted(p['ranks'], key=lambda r: r['rank'])]}, cases per rank {[len(r['cases']) for r in p['ranks']]}, live_restored {[r['live_state_restored'] for r in p['ranks']]}, gate {p['gate']['status']}")
        print(f"   sum max observer seconds: {sum(max(r['observer_seconds_including_setup'] for r in p['ranks']) for p in rep['points']):.2f}")
