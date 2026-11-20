"""L1-2: inject a technical failure mid-search and check whether the initial-state hash
reached any persisted/logged output before the failure (production on_checkpoint=_progress)."""
import contextlib, io, sys
from pathlib import Path
import torch
sys.path.insert(0, "/Users/simonebenatti/dev/wearusfm/.claude/worktrees/review-pose-heads-2f98211/tests/cpu")
import test_pose_temponet_train as F          # reuse the package's own tiny fixture (factory, dataset, evaluator)
from pose_temponet import train as T, runner

fitting = [F.dataset(n=7, seed=10), F.dataset(n=7, seed=11)]
recipe = T._TestRecipe(initial_epochs=2, maximum_epochs=3)
good = F.evaluator_for(fitting)
calls = {"n": 0}
def failing(model, fold):
    calls["n"] += 1
    if calls["n"] > 6:                         # fail after a few checkpoints of the first LR
        raise FloatingPointError("injected technical failure mid-search")
    return good(model, fold)

buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        T._run_search(F.factory, fitting, failing, device="cpu", recipe=recipe, on_checkpoint=runner._progress)
except FloatingPointError as e:
    print("search raised:", e)
log = buf.getvalue()
expected = [T.Trajectory(F.factory, d, lr=1e-3, seed=0, fold=f, _recipe=recipe).initial_sha256 for f, d in enumerate(fitting)]
print("progress lines logged before failure:", len(log.splitlines()))
print("first line:", log.splitlines()[0] if log else None)
print("initial hash", expected[0][:16], "present in log:", any(h in log for h in expected))
print("'initial' substring in log:", "initial" in log)
