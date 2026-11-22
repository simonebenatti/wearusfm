"""Prova CPU su latenti sintetici, codice congelato vero (albero e8c98244, sola lettura).
Domande operative:
 A) un driver esterno che ricarica best_states/current_states e chiama workflow._score_model riproduce ESATTAMENTE i punti della curva?
 B) valutazioni aggiuntive (fitting, pulito, ripetute) fra i passi cambiano la traiettoria? (parita' di un'epoca e oltre)
 C) si puo' allenare sul bersaglio pulito y* usando solo funzioni congelate, senza toccarle?
Nessun dato reale. Ricetta di prova (_TestRecipe) tranne dove indicato."""
import dataclasses, json, sys, time
import numpy as np, torch
from pose_temponet import controls as C, data as D, metrics as M, train as T, workflow as W, heads, selection as S

torch.set_num_threads(1)
t0 = time.time()

def stats(mse, edyn, count=4, ss=40.):
    return {"mse_sum": mse*count, "window_count": count, "dynamic_sse": edyn*ss, "dynamic_target_ss": ss}

def cache(users=("u3", "u0", "u2", "u1", "u5", "u4"), windows=3):
    rng = np.random.default_rng(41)
    n = len(users) * windows
    x = rng.normal(size=(n, 160, 384)).astype(np.float16)
    target = x[..., :20].astype(np.float32) * np.linspace(.6, 1.5, n)[:, None, None]
    mask = np.ones(target.shape, dtype=bool)
    mask[0, 70:90, 4:] = False
    return {"lat": x, "pre": x * np.float16(.7), "target": target.astype(np.float32), "mask": mask,
            "std": np.ones((n, 20), np.float32), "pose_var": np.full(n, 999., dtype=np.float32),
            "key_time_valid": np.ones((n, 160), bool),
            "config": {"split": "train", "which": "teacher", "checkpoint_step": 49046},
            "meta": [{"user": user, "recording": i, "start": 0, "n_patches": 160}
                     for i, user in enumerate(np.repeat(users, windows))]}

c = cache()
real = W._prepare_real_folds(c, allowed_train_users=sorted({m["user"] for m in c["meta"]}),
                             source_identity={"cache_sha256": "a"*64, "protocol_sha256": "b"*64}, expected_user_count=6)
arts = []
for fold in (0, 1):
    y = real.cache["target"].astype(np.float64)
    comp = {"L": y*.5, "T": np.roll(y, 1, axis=1)*.25, "epsilon": np.random.default_rng(fold).normal(size=y.shape)*4}
    th = {}
    for world, a in (("linear", 0.), ("temporal", .3)):
        clean, _ = C.synthetic_target(comp, a, .8)
        th[world+"_threshold"] = M.training_threshold(clean[real.fitting[fold]], real.cache["mask"][real.fitting[fold]])
    arts.append({"outer_fold": fold,
                 "partition": C.validate_outer_partition(real.cache["meta"], real.fitting[fold], real.held[fold], real.manifest["users"]),
                 "components": comp,
                 "calibration": {"status": "calibrated", "selected": {"linear": {"a": 0., "sigma": .8}, "temporal": {"a": .3, "sigma": .8}}},
                 "outer_scored_after_inner_selection": True,
                 "outer_reference_statistics": {**th,
                     "linear": {"constant": stats(1., 1.), "ridge": stats(.2, .2), "oracle": stats(.18, .18), "clean_ls": stats(.18, .18)},
                     "temporal": {"constant": stats(2., 2.), "clean_ls": stats(1., 1.), "oracle": stats(.92, .92)}}})
cb = W._prepare_control_folds(real.cache, arts, "temporal", allowed_train_users=real.manifest["users"],
                              source_identity=real.manifest["source_identity"], expected_user_count=6)
rec = T._TestRecipe(initial_epochs=3, maximum_epochs=6, last_epochs=1, batch_size=4)
result = W._run_selection(cb, "TCN_post", recipe=rec)
report, states = result["selection"], result["states"]["candidate_states"]
out = {"lrs": [cv["lr"] for cv in report["curves"]], "selected": report["selected"]}

# A) inferenza esterna su tutti gli stati salvati
factory = lambda seed, fold: heads.make_head("TCN_post", seed=seed)
def pooled(models, exposure, steps, *, clean=False):
    obs, cl = [], []
    for fold, model in enumerate(models):
        o, k = W._score_model(model, cb, fold, cb.held[fold], "post", "cpu", include_clean=True)
        obs.append(o); cl.append(k)
    return S.pool_scores(cl if clean else obs, exposure, steps)
A = []
for curve in report["curves"]:
    lr = curve["lr"]; pair = states[lr]
    pts = {p["exposure"]: p for p in curve["points"]}
    for kind, exposure in (("best_states", pair["best_exposure"]), ("current_states", curve["points"][-1]["exposure"])):
        models = []
        for fold in (0, 1):
            m = heads.make_head("TCN_post", seed=0); m.load_state_dict(pair[kind][fold]["model"], strict=True); m.eval(); models.append(m)
        p = pooled(models, exposure, pts[exposure]["fold_steps"])
        A.append({"lr": lr, "kind": kind, "exposure": exposure, "mse_equal": p.mse == pts[exposure]["mse"], "edyn_equal": p.edyn == pts[exposure]["edyn"],
                  "updates": [pair[kind][f]["updates"] for f in (0, 1)]})
out["A_states_checked"] = len(A)
out["A_all_exact"] = all(r["mse_equal"] and r["edyn_equal"] for r in A)
out["A_rows"] = A

# B) parita' della traiettoria con valutazioni aggiuntive
lr = report["selected"]["lr"]
curve = next(cv for cv in report["curves"] if cv["lr"] == lr)
fitting = [W._training_set(cb, ix, "post", fold) for fold, ix in enumerate(cb.fitting)]
traj = [T.Trajectory(factory, fitting[f], lr=lr, seed=0, fold=f, device="cpu", _recipe=rec) for f in (0, 1)]
out["B_initial_sha256_equal"] = [t.initial_sha256 for t in traj] == report["initial_sha256_per_fold"]
held_eval = lambda model, fold: W._score_model(model, cb, fold, cb.held[fold], "post", "cpu")
extra = 0
B = []
for exposure, steps in S.checkpoint_schedule([t.steps_per_epoch for t in traj], curve["horizon"]):
    for t, s in zip(traj, steps):
        t.train_to_step(s)
    # valutazioni in piu', prima e dopo quella della curva: fitting e esclusi, con e senza pulito, ripetute
    for t in traj:
        t.evaluate(lambda model, fold: W._score_model(model, cb, fold, cb.fitting[fold], "post", "cpu", include_clean=True)[1]); extra += 1
        t.evaluate(lambda model, fold: W._score_model(model, cb, fold, cb.held[fold], "post", "cpu", include_clean=True)[1]); extra += 1
    point = S.pool_scores([t.evaluate(held_eval) for t in traj], exposure, steps)
    ref = next(p for p in curve["points"] if p["exposure"] == exposure)
    B.append({"exposure": exposure, "mse_equal": point.mse == ref["mse"], "edyn_equal": point.edyn == ref["edyn"], "fold_steps_equal": list(point.fold_steps) == ref["fold_steps"]})
out["B_points"] = len(B); out["B_extra_evaluations"] = extra
out["B_all_exact"] = all(r["mse_equal"] and r["edyn_equal"] and r["fold_steps_equal"] for r in B)
out["B_final_payload_sha256_equal"] = [t.state()["payload_sha256"] for t in traj] == [states[lr]["current_states"][f]["payload_sha256"] for f in (0, 1)]

# C) allenamento sul bersaglio pulito con sole funzioni congelate (nessuna modifica al pacchetto)
clean_bundle = dataclasses.replace(cb, targets=cb.clean_targets)
try:
    W._assert_bundle(clean_bundle); out["C_assert_bundle_accepts_swapped_targets"] = True
except ValueError as e:
    out["C_assert_bundle_accepts_swapped_targets"] = False; out["C_assert_bundle_error"] = str(e)
fit_clean = [W._training_set(clean_bundle, ix, "post", fold) for fold, ix in enumerate(clean_bundle.fitting)]
out["C_dataset_identity_differs_from_noisy"] = [a.identity != b.identity for a, b in zip(fit_clean, fitting)]
tc = [T.Trajectory(factory, fit_clean[f], lr=lr, seed=0, fold=f, device="cpu", _recipe=rec) for f in (0, 1)]
out["C_same_initialization_as_noisy_run"] = [t.initial_sha256 for t in tc] == report["initial_sha256_per_fold"]
rows = []
for exposure, steps in S.checkpoint_schedule([t.steps_per_epoch for t in tc], rec.initial_epochs):
    for t, s in zip(tc, steps):
        t.train_to_step(s)
    scores = [t.evaluate(lambda model, fold: W._score_model(model, cb, fold, cb.held[fold], "post", "cpu", include_clean=True)[1]) for t in tc]
    p = S.pool_scores(scores, exposure, steps)
    rows.append([exposure, p.mse, p.edyn])
out["C_clean_fit_curve_vs_clean_target"] = rows
# la Trajectory di produzione (Recipe()) accetta gli stessi insiemi? (solo costruzione + 1 passo)
tp = T.Trajectory(factory, fit_clean[0], lr=0.003, seed=0, fold=0, device="cpu")
tp.train_to_step(1)
out["C_production_recipe_trajectory_one_step_ok"] = tp.updates == 1
out["C_production_steps_per_epoch_toy"] = tp.steps_per_epoch
out["elapsed_seconds"] = round(time.time()-t0, 1)
print(json.dumps(out, indent=1, default=str))
