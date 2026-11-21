import sys, json, numpy as np, torch
from torch import nn
sys.path.insert(0, "scripts")
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_stat_work")
import toy
from pose_temponet import heads, metrics
torch.set_num_threads(1)
d = np.load(toy.WORK+"toy_data.npz"); refs = json.load(open(toy.WORK+"toy_refs.json")); a = refs["selected"]["a"]
fit = d["fit"][:16]
x = torch.from_numpy(d["x"][fit].astype(np.float32))
ystar = d["L"][fit].astype(np.float64)+a*d["T"][fit].astype(np.float64)
y = torch.from_numpy((ystar+1.3*d["eps"][fit]).astype(np.float32))[:, 6:-6]
sup = torch.ones(y.shape, dtype=torch.bool)
for mode in ("zero", "free"):
    m = heads.make_head("TCN_post", seed=0)
    toy.embed_truth(m, a, mode)
    m.eval()
    with torch.no_grad():
        err = (m(x)[:, 6:-6].double().numpy()-ystar[:, 6:-6])
    m.train()
    loss = metrics.per_window_mse(m(x)[:, 6:-6], y, sup); loss.backward()
    convs = [c for c in m.deep if isinstance(c, nn.Conv1d)]
    out = {"mode": mode, "embedding_rmse_vs_ystar": float(np.sqrt((err**2).mean())), "loss": float(loss)}
    for i, c in enumerate(convs):
        g = c.weight.grad
        if i < 5:
            out["conv%d_grad_extra_rows" % i] = float(g[8:].norm()); out["conv%d_grad_true_rows" % i] = float(g[:8].norm())
        if i > 0:
            out["conv%d_grad_extra_cols" % i] = float(g[:, 8:].norm())
    print(json.dumps(out))
