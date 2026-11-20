"""Show that read_preflight replaces compare_three_paths' evidence pins (traces/start.json)."""
import sys, json, tempfile
from pathlib import Path
from _pytest.monkeypatch import MonkeyPatch
import contrastive_benchmark_readout as B
import contrastive_target_resume
import test_contrastive_targets_readout as T

tmp = Path(tempfile.mkdtemp(dir=sys.argv[1]))
mp = MonkeyPatch()
try:
    kwargs, parity_path, parity, calls, pairs = T.readout_fixture(tmp, mp)
    def compare(paths, *a, **k):
        return {"status": "PASSED", "passed": True,
                "evidence_sha256": {phase: {"trace-0.jsonl": "T"*64, "start.json": "S"*64} for phase in paths}}
    mp.setattr(contrastive_target_resume, "compare_three_paths", compare)
    result = B.read_preflight(**kwargs)
    ev = result["arms"]["CN"]["resume"]["evidence_sha256"]
    print("status:", result["status"])
    print("resume evidence keys:", sorted(ev))
    print("per-phase keys:", {k: sorted(v) for k, v in ev.items()})
    print("trace pins retained:", any("trace-0.jsonl" in v for v in ev.values()))
finally:
    mp.undo()
