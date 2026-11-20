import sys, math
sys.path.insert(0, "tests/cpu")
import torch
from pose_temponet import train as T, selection as S
import test_pose_temponet_train as F

orig = T.Trajectory
class Tagged(orig):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.model._lr = self.lr
        self.model._traj = self
T.Trajectory = Tagged

def mse_for(lr, epoch):
    # 1e-4 best among initial (lower edge); 3e-5 better still, its minimum at the final initial epoch (3)
    base = {1e-4: 1.0, 3e-4: 1.2, 1e-3: 1.3, 3e-3: 1.4, 3e-5: 0.9}[lr]
    if lr == 3e-5:
        return base + abs(epoch - 3)*0.01 if epoch <= 3 else base - 0.05 + abs(epoch - 5)*0.001
    return base + abs(epoch - 1)*0.01

log = []
def evaluator(model, fold):
    t = model._traj
    exposure = t.updates / t.steps_per_epoch
    log.append((model._lr, fold, t.updates))
    v = mse_for(model._lr, exposure)
    return S.FoldScore(v, 1, v, 1.0)

fitting = [F.dataset(n=7, seed=10), F.dataset(n=7, seed=11)]
recipe = T._TestRecipe(initial_epochs=3, maximum_epochs=6, last_epochs=1)
res = T._run_search(F.factory, fitting, evaluator, device="cpu", recipe=recipe)
rep = res["selection"]
print("LR order:", [c["lr"] for c in rep["curves"]], "horizons:", [c["horizon"] for c in rep["curves"]])
print("selected:", rep["selected"], "alarms:", rep["optimizer_alarms"], "lower gain:", rep["lower_edge_relative_mse_gain"])
st = res["candidate_states"][3e-5]["current_states"]
print("3e-5 final updates per fold:", [s["updates"] for s in st], "steps/epoch:", [s["steps_per_epoch"] for s in st])
# continuation is not a restart: number of shuffle orders recorded equals epochs
print("epoch orders recorded:", [len(s["epoch_order_sha256"]) for s in st], "epoch losses:", [len(s["epoch_losses"]) for s in st])
# evaluation steps for 3e-5 fold 0 monotone and unique
steps = [u for lr, f, u in log if lr == 3e-5 and f == 0]
print("3e-5 fold0 eval steps:", steps)
