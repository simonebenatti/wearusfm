import os, sys, time, json, types
from pathlib import Path
import contrastive_cn_restart as g, contrastive_job
W = Path(sys.argv[1]); root = W/"simout"
g.require_binding = lambda: {}
g.commands = lambda plan: (root, [["a"],["b"]], ["c"], [])
contrastive_job.validate_allocation = lambda k: None
g.subprocess.check_output = lambda *a, **k: "0, GPU-x, 00, A100\n"
g.BINDING = g.PLAN
os.environ.update(TORCH_DIST_INIT_BARRIER="1", SLURM_JOB_ID="1", WEARUSFM_CODE_SHA="x")
for label, endoff in (("ample", 3600), ("tight", 300)):
    if root.exists():
        for f in root.iterdir(): f.unlink()
        root.rmdir()
    os.environ["SLURM_JOB_END_TIME"] = str(int(time.time()) + endoff)
    seen = []
    def fake(cmd, deadline):
        seen.append(deadline - time.monotonic())
        if len(seen) == 2: raise TimeoutError("simulated exhaustion")
    g.checked_command = fake
    try: g.run(False)
    except TimeoutError: pass
    print(label, "remaining at first cell ~%.0f s" % seen[0], json.loads((root/"execution_failure.json").read_text()))
