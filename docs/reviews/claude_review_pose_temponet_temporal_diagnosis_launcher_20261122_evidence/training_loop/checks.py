"""Verifiche meccaniche del ciclo VERO (train.Trajectory / PairedCandidate / heads) su dati sintetici piccoli.
Ricetta di produzione Recipe() dove serve la forma di produzione (N,160,384); nessun dato vero.
"""
import math, copy, json
import numpy as np
import torch
from torch import nn
from pose_temponet import heads, metrics, selection, train as T

torch.set_num_threads(1)
torch.backends.nnpack.set_flags(False)
g = torch.Generator().manual_seed(7)
N = 70
x = torch.randn((N, 160, 384), generator=g).to(torch.float16)
y = torch.randn((N, 160, 20), generator=g, dtype=torch.float64)*1.7
sup = torch.ones((N, 148, 20), dtype=torch.bool)
ids = ["w%03d" % i for i in range(N)]
out = {}

# (a) quali parametri partono a zero, uscita iniziale, gruppi dell'ottimizzatore
m = heads.make_head("TCN_post", 0)
zero = [n for n, p in m.named_parameters() if float(p.abs().max()) == 0.0]
out["zero_init_params"] = zero
out["initial_output_absmax"] = float(m.eval()(x[:4].float()).abs().max())
groups = heads.optimizer_groups(m, 0.01)
names = {id(p): n for n, p in m.named_parameters()}
out["decay_group"] = [names[id(p)] for p in groups[0]["params"]]
out["no_decay_group"] = [names[id(p)] for p in groups[1]["params"]]
out["params_decay_nodecay"] = [sum(p.numel() for p in groups[0]["params"]), sum(p.numel() for p in groups[1]["params"])]
out["params_by_branch"] = {"norm": sum(p.numel() for n, p in m.named_parameters() if n.startswith("norm")),
                           "affine": sum(p.numel() for n, p in m.named_parameters() if n.startswith("affine")),
                           "deep": sum(p.numel() for n, p in m.named_parameters() if n.startswith("deep"))}

# (b) flusso del gradiente ai primi passi con il ciclo vero (norme per ramo PRIMA del clip)
data = T.TensorTrainingSet(x, y, sup, ids)
traj = T.Trajectory(lambda s, f: heads.make_head("TCN_post", seed=s), data, lr=0.003, seed=0, fold=0)
rows = []
orig = torch.nn.utils.clip_grad_norm_
def spy(params, max_norm, **kw):
    r = {}
    for n, p in traj.model.named_parameters():
        key = "norm" if n.startswith("norm") else "affine" if n.startswith("affine") else "deep_out" if n.startswith("deep.12") else "deep_hidden"
        r[key] = r.get(key, 0.0) + float(p.grad.pow(2).sum())
    total = orig(params, max_norm, **kw)
    r = {k: math.sqrt(v) for k, v in r.items()}; r["total_preclip"] = float(total)
    r["total_postclip"] = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in traj.model.parameters()))
    rows.append(r)
    return total
torch.nn.utils.clip_grad_norm_ = spy
before = {n: p.detach().clone() for n, p in traj.model.named_parameters()}
traj.train_to_step(1)
after1 = {n: p.detach().clone() for n, p in traj.model.named_parameters()}
out["step1_max_abs_update_by_param"] = {n: float((after1[n]-before[n]).abs().max()) for n in before}
traj.train_to_step(3)
torch.nn.utils.clip_grad_norm_ = orig
out["grad_norms_first3_steps"] = rows
out["training_flag_after_train"] = traj.model.training

# (c) stato migliore: copia profonda, non alias di tensori poi modificati
st = traj.state()
snap = {k: v.clone() for k, v in st["model"].items()}
alias = any(v.data_ptr() == p.data_ptr() for (k, v), p in zip(st["model"].items(), traj.model.state_dict().values()))
traj.train_to_step(6)
out["state_aliases_live_tensors"] = bool(alias)
out["state_unchanged_after_more_training"] = all(torch.equal(st["model"][k], snap[k]) for k in snap)
out["live_model_changed"] = any(not torch.equal(traj.model.state_dict()[k], snap[k]) for k in snap)
opt_alias = False
live = traj.optimizer.state_dict()["state"]
for i, s in st["optimizer"]["state"].items():
    for k, v in s.items():
        if isinstance(v, torch.Tensor) and isinstance(live[i][k], torch.Tensor) and v.data_ptr() == live[i][k].data_ptr():
            opt_alias = True
out["optimizer_state_aliases_live"] = opt_alias

# (d) dropout: inattivo in evaluate (due valutazioni identiche), attivo in train; evaluate non consuma l'RNG del dropout
preds = []
def evaluator(model, fold):
    preds.append(model(x[:8].float()).clone()); preds.append(model(x[:8].float()).clone())
    out["model_training_inside_evaluate"] = model.training
    return {"mse_sum": 1.0, "window_count": 1, "dynamic_sse": 1.0, "dynamic_target_ss": 1.0}
rng_before = traj.dropout_cpu.clone()
traj.evaluate(evaluator)
out["eval_deterministic"] = bool(torch.equal(preds[0], preds[1]))
out["dropout_rng_untouched_by_evaluate"] = bool(torch.equal(rng_before, traj.dropout_cpu))
out["training_flag_restored_after_evaluate"] = traj.model.training
traj.model.train()
a1 = traj.model(x[:8].float()); a2 = traj.model(x[:8].float())
out["train_mode_stochastic"] = bool(not torch.equal(a1, a2))

# (e) restore del best state riproduce le predizioni; ripresa = corsa continua (bit a bit)
t2 = T.Trajectory(lambda s, f: heads.make_head("TCN_post", seed=s), data, lr=0.003, seed=0, fold=0)
t2.train_to_step(3); mid = t2.state(); t2.train_to_step(6)
t3 = T.Trajectory(lambda s, f: heads.make_head("TCN_post", seed=s), data, lr=0.003, seed=0, fold=0)
t3.restore(mid); t3.train_to_step(6)
out["resume_equals_continuous_bitwise"] = all(torch.equal(a, b) for a, b in zip(t2.model.state_dict().values(), t3.model.state_dict().values()))
out["continuous_equals_first_traj_bitwise"] = all(torch.equal(a, b) for a, b in zip(t2.model.state_dict().values(), traj.model.state_dict().values()))

# (f) allineamento bersaglio/maschera nel batch: batch() restituisce le righe giuste e il crop giusto
pos = torch.tensor([5, 0, 17])
bx, by, bm = data.batch(pos, torch.device("cpu"))
rows_ = data._rows[pos]
out["batch_alignment_ok"] = bool(torch.equal(bx, x[rows_].float()) and torch.equal(by, y[rows_][:, 6:154].float()) and bm.all())
out["steps_per_epoch_70"] = traj.steps_per_epoch
out["steps_per_epoch_920_934_1854"] = [math.ceil(n/32) for n in (920, 934, 1854)]
out["last_batch_sizes_920_934_1854"] = [n - 32*(math.ceil(n/32)-1) for n in (920, 934, 1854)]

# (g) per_window_mse: gradiente rispetto alla predizione = 2*(p-y)/(count_finestra*B); finestra senza supporto esclusa
p = torch.randn(4, 148, 20, requires_grad=True); yy = torch.randn(4, 148, 20); mm = torch.ones(4, 148, 20, dtype=torch.bool)
mm[1, :100] = False; mm[3] = False
loss = metrics.per_window_mse(p, yy, mm); loss.backward()
cnt = mm.sum(dim=(1, 2)).float()
expected = torch.zeros_like(p)
for i in range(4):
    if cnt[i] > 0:
        expected[i] = torch.where(mm[i], 2*(p[i]-yy[i]).detach()/(cnt[i]*3), torch.zeros(()))
out["per_window_mse_grad_matches_formula"] = bool(torch.allclose(p.grad, expected, atol=1e-8))
out["per_window_mse_masked_grad_zero"] = float(p.grad[~mm].abs().max())

# (h) decadimento complessivo dei pesi: fattore (1-lr*wd)^passi
out["decay_factor_lr0.003"] = {"exposure31_fold0_899steps": (1-0.003*0.01)**899, "exposure150_4350steps": (1-0.003*0.01)**4350}
print(json.dumps(out, indent=1))
