"""Clean repro: production on_checkpoint=runner._progress vs on_checkpoint=None, same tiny fixture."""
import contextlib, io, sys
sys.path.insert(0, "/Users/simonebenatti/dev/wearusfm/.claude/worktrees/review-pose-heads-2f98211/tests/cpu")
import test_pose_temponet_train as F
from pose_temponet import train as T, runner

fitting = [F.dataset(n=7, seed=10), F.dataset(n=7, seed=11)]
recipe = T._TestRecipe(initial_epochs=2, maximum_epochs=3)
ok = T._run_search(F.factory, fitting, F.evaluator_for(fitting), device="cpu", recipe=recipe, on_checkpoint=None)
print("on_checkpoint=None: completed, selected", ok["selection"]["selected"])
calls = {"eval": 0}
ev = F.evaluator_for(fitting)
def counting(model, fold):
    calls["eval"] += 1
    return ev(model, fold)
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        T._run_search(F.factory, fitting, counting, device="cpu", recipe=recipe, on_checkpoint=runner._progress)
    print("on_checkpoint=runner._progress: completed")
except ValueError as e:
    print("on_checkpoint=runner._progress: ValueError:", e, "| fold evaluations before failure:", calls["eval"],
          "| lines printed:", len(buf.getvalue().splitlines()))
