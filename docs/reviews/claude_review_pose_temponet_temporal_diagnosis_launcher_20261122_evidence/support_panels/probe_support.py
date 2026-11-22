"""Audit CPU del supporto con le funzioni vere del commit e8c98244 (dati sintetici, nessun dato vero).

Prova A: allineamento posizione-per-posizione di predizione, bersaglio e supporto
         - nel training (train.Trajectory vero -> metrics.per_window_mse)
         - nella selezione (workflow._score_model vero -> metrics._cropped / evaluate)
         contro un supporto ricalcolato in modo indipendente in coordinate 0..159.
Prova B: mondo di controllo sintetico con le funzioni vere (generate_components,
         evaluate_outer_references, _prepare_control_folds, _score_model):
         uguaglianza dei pannelli clean_ls / oracolo / testa, statistiche ricalcolate
         in modo indipendente, identita' SSE_y = SSE_y* + SSE_oracolo - 2<e,n>, pooled.

Unica alterazione in memoria (nessun file toccato): controls.combine_reference_folds
restituisce lo stato "diagnostic" forzato, perche' il mondo giocattolo non e' calibrato
e workflow.py:174 altrimenti rifiuta il bundle. Nessuna quantita' sotto prova ne dipende.
"""
import json
import sys

import numpy as np
import torch
from torch import nn

from pose_temponet import controls as C, data as D, heads as H, metrics as M, selection as S, train as T, workflow as W

torch.set_num_threads(1)
OUT = {}
FAIL = []


def check(name, condition, detail=""):
    OUT[name] = bool(condition)
    print(("OK   " if condition else "FAIL ") + name + ((" :: " + str(detail)) if detail else ""))
    if not condition:
        FAIL.append(name)


# ----------------------------------------------------------------------------
# cache sintetica con maschere "cattive"
# ----------------------------------------------------------------------------
USERS = ["u%d" % k for k in range(8)]
WINDOWS = 6
N = len(USERS) * WINDOWS
TIME, JOINTS = 160, 20


def make_cache(coded, seed=7):
    rng = np.random.default_rng(seed)
    meta = [{"user": USERS[i // WINDOWS], "recording": i, "start": 0, "n_patches": TIME} for i in range(N)]
    mask = np.ones((N, TIME, JOINTS), dtype=bool)
    # buchi casuali a blocchi
    for w in range(N):
        for _ in range(3):
            j = rng.integers(0, JOINTS)
            a = rng.integers(0, TIME - 12)
            mask[w, a:a + rng.integers(1, 12), j] = False
    for user_index in (0, 1):                      # un esemplare per ciascuna meta' degli utenti
        base = user_index * WINDOWS
        mask[base + 1] = False                       # nessuna posa valida
        mask[base + 2] = False                       # posa valida solo ai bordi tagliati
        mask[base + 2, :6] = True
        mask[base + 2, 154:] = True
        mask[base + 3, 70:90, 4:] = False            # buco temporale
        mask[base + 4, :, 7] = False                 # giunto 7 con un solo campione centrale
        mask[base + 4, 80, 7] = True
        mask[base + 5, :, 3] = False                 # giunto 3 valido solo ai due estremi del crop
        mask[base + 5, 6, 3] = True
        mask[base + 5, 153, 3] = True
    # una finestra corta per meta'
    for w in (2 * WINDOWS, 3 * WINDOWS):
        meta[w]["n_patches"] = 120
        mask[w, 120:] = False
    lengths = np.asarray([m["n_patches"] for m in meta])
    available = np.arange(TIME)[None, :] < lengths[:, None]
    if coded:
        lat = np.zeros((N, TIME, 384), dtype=np.float32)
        w_, t_, j_ = np.meshgrid(np.arange(N), np.arange(TIME), np.arange(JOINTS), indexing="ij")
        code = (w_ * 4096 + t_ * 20 + j_).astype(np.float32)
        lat[..., :JOINTS] = code
        offset = (((w_ * 7 + t_ * 3 + j_) % 5) - 2).astype(np.float32)
        target = code + offset
    else:
        # latenti lisci nel tempo + rumore, FP16 come la cache vera
        base = rng.normal(size=(N, TIME, 384))
        smooth = base.copy()
        for t in range(1, TIME):
            smooth[:, t] = .8 * smooth[:, t - 1] + .6 * base[:, t]
        lat = smooth.astype(np.float16)
        target = (smooth[..., :JOINTS] * np.linspace(.5, 1.5, N)[:, None, None] +
                  rng.normal(size=(N, TIME, JOINTS)) * .3).astype(np.float32)
    # contesto imbottito: valori finiti ma arbitrari (devono essere ignorati)
    lat[~available] = 3.0
    return {"lat": lat, "pre": lat.copy(), "target": target, "mask": mask,
            "std": np.ones((N, JOINTS), np.float32), "pose_var": np.full(N, 999., np.float32),
            "key_time_valid": available.copy(),
            "config": {"split": "train", "which": "teacher", "checkpoint_step": 49046},
            "meta": meta}


def independent_support(cache):
    """S[w,t,j] in coordinate 0..159: posa valida, 6 <= t <= 153, contesto reale t+6 <= n-1."""
    mask = cache["mask"]
    out = np.zeros_like(mask)
    for w, meta in enumerate(cache["meta"]):
        n = meta["n_patches"]
        for t in range(TIME):
            if 6 <= t <= 153 and t + 6 <= n - 1:
                out[w, t] = mask[w, t]
    return out


def independent_pose_var(clean, mask):
    """Varianza di popolazione per giunto sulla finestra INTERA (160), media sui giunti con >= 2 campioni."""
    pv = np.full(len(clean), np.nan)
    for w in range(len(clean)):
        values = []
        for j in range(clean.shape[2]):
            sel = mask[w, :, j]
            if sel.sum() >= 2:
                x = clean[w, sel, j].astype(np.float64)
                values.append(np.mean((x - x.mean()) ** 2))
        if values:
            pv[w] = float(np.mean(values))
    return pv


def independent_stats(pred160, target160, support160, rows, pose_var, threshold):
    mse_sum, count, sse, ss, n_dyn, cells, elements = 0., 0, 0., 0., 0, 0, 0
    for w in rows:
        s = support160[w]
        q = int(s.sum())
        if q > 0:
            d = (pred160[w].astype(np.float64) - target160[w].astype(np.float64))[s]
            mse_sum += float((d * d).sum() / q)
            count += 1
        if np.isfinite(pose_var[w]) and pose_var[w] >= threshold:
            n_dyn += 1
            for j in range(s.shape[1]):
                sel = s[:, j]
                if sel.sum() >= 2:
                    p = pred160[w, sel, j].astype(np.float64)
                    y = target160[w, sel, j].astype(np.float64)
                    sse += float((((p - p.mean()) - (y - y.mean())) ** 2).sum())
                    ss += float(((y - y.mean()) ** 2).sum())
                    cells += 1
                    elements += int(sel.sum())
    return {"mse_sum": mse_sum, "window_count": count, "dynamic_sse": sse, "dynamic_target_ss": ss,
            "n_dynamic_windows": n_dyn, "n_cells": cells, "n_elements": elements}


def close(a, b, rel=1e-9):
    return all(abs(a[k] - b[k]) <= rel * max(1., abs(a[k]), abs(b[k]))
               for k in ("mse_sum", "window_count", "dynamic_sse", "dynamic_target_ss"))


def to160(pred148):
    full = np.full((len(pred148), TIME, pred148.shape[2]), np.nan)
    full[:, 6:154] = pred148
    return full


# ----------------------------------------------------------------------------
# PROVA A: allineamento nel training e nella selezione (mondo "real" del workflow)
# ----------------------------------------------------------------------------
print("=== PROVA A: allineamento posizione-per-posizione (feature e bersagli codificati) ===")
cache_a = make_cache(coded=True)
S160 = independent_support(cache_a)
# confronto diretto con D.central_support
central = D.central_support(cache_a["mask"], D.lengths_for(cache_a), cache_a["key_time_valid"])
check("A1 central_support == S[:, 6:154] indipendente", np.array_equal(central, S160[:, 6:154]),
      "elementi supportati %d; finestre supportate %d su %d" % (int(central.sum()), int(central.any(axis=(1, 2)).sum()), N))
check("A1b nessun supporto fuori da 6..153", not S160[:, :6].any() and not S160[:, 154:].any())

bundle_a = W._prepare_real_folds(cache_a, allowed_train_users=USERS, source_identity={"pin": "toy"},
                                 expected_user_count=len(USERS))


class ProbeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.zero = nn.Parameter(torch.zeros(JOINTS))

    def forward(self, x):
        return x[..., :JOINTS].float() + self.zero


records = []
real_loss = M.per_window_mse


def recording_loss(prediction, target, support):
    records.append((prediction.detach().clone().numpy(), target.detach().clone().numpy(), support.clone().numpy()))
    return real_loss(prediction, target, support)


recipe = T._TestRecipe(initial_epochs=1, maximum_epochs=2, last_epochs=1, batch_size=4)
code = cache_a["lat"][..., :JOINTS].astype(np.float64)
for fold in (0, 1):
    fit = bundle_a.fitting[fold]
    ds = W._training_set(bundle_a, fit, "post", fold)
    expected_rows = sorted(int(w) for w in fit if S160[w].any())
    check("A2 fold%d: finestre di training = finestre di fitting con supporto centrale" % fold,
          ds.size == len(expected_rows) and ds.excluded_windows == len(fit) - len(expected_rows),
          "size %d, escluse %d (fitting %d)" % (ds.size, ds.excluded_windows, len(fit)))
    records.clear()
    M.per_window_mse = recording_loss
    try:
        trajectory = T.Trajectory(lambda seed, f: ProbeHead(), ds, lr=1e-3, seed=0, fold=fold, device="cpu", _recipe=recipe)
        trajectory.train_to_step(trajectory.steps_per_epoch)
    finally:
        M.per_window_mse = real_loss
    seen, aligned_pred, aligned_target, aligned_support, loss_ok = [], True, True, True, True
    for prediction, target, support in records:
        for b in range(len(prediction)):
            w = int(round(float(prediction[b, 0, 0]))) // 4096
            seen.append(w)
            real = np.arange(6, 154) < cache_a["meta"][w]["n_patches"]
            aligned_pred &= bool(np.array_equal(np.rint(prediction[b][real]), code[w, 6:154][real]))
            # oltre la lunghezza reale le feature sono azzerate da workflow._training_set: uscita = solo il parametro (~0)
            aligned_pred &= bool((np.abs(prediction[b][~real]) < .5).all())
            aligned_target &= bool(np.array_equal(target[b], cache_a["target"][w, 6:154]))
            aligned_support &= bool(np.array_equal(support[b], S160[w, 6:154]))
    check("A3 fold%d: ogni finestra supportata vista una volta per epoca, nessun'altra" % fold,
          sorted(seen) == expected_rows, "viste %d" % len(seen))
    check("A4 fold%d: la loss riceve predizione = patch 6..153 dell'uscita" % fold, aligned_pred)
    check("A5 fold%d: la loss riceve bersaglio = patch 6..153 del bersaglio" % fold, aligned_target)
    check("A6 fold%d: la loss riceve supporto = S indipendente" % fold, aligned_support)
    # valore della loss sull'intero fitting contro il ricalcolo indipendente
    model = ProbeHead()
    x, y, m = ds.batch(torch.arange(ds.size), torch.device("cpu"))
    loss = float(M.per_window_mse(model(x)[:, 6:154], y, m))
    pv_dummy = np.full(N, np.nan)
    ind = independent_stats(code, cache_a["target"], S160, fit, pv_dummy, 0.)
    check("A7 fold%d: loss di training (tutto il fitting) = MSE per finestra indipendente" % fold,
          abs(loss - ind["mse_sum"] / ind["window_count"]) <= 1e-6 * max(1., loss),
          "loss %.9f, indipendente %.9f, finestre %d" % (loss, ind["mse_sum"] / ind["window_count"], ind["window_count"]))

    # selezione: _score_model vero, con registrazione di cio' che entra nella metrica
    cropped_records = []
    real_cropped = M._cropped

    def recording_cropped(prediction, data):
        out = real_cropped(prediction, data)
        cropped_records.append((out[0].copy(), out[1].copy(), out[2].copy()))
        return out

    for name, rows in (("held", bundle_a.held[fold]), ("fitting", bundle_a.fitting[fold])):
        cropped_records.clear()
        M._cropped = recording_cropped
        try:
            stats = W._score_model(ProbeHead(), bundle_a, fold, rows, "post", "cpu")
        finally:
            M._cropped = real_cropped
        ok_p = ok_t = ok_s = True
        seen = []
        for prediction, target, support in cropped_records:
            for b in range(len(prediction)):
                # le finestre corte hanno feature azzerate oltre la lunghezza: la decodifica usa il bersaglio
                w = int(round(float(target[b, 0, 0]) + 2)) // 4096
                seen.append(w)
                n = cache_a["meta"][w]["n_patches"]
                real = np.arange(6, 154) < n
                ok_p &= bool(np.array_equal(np.rint(prediction[b][real]), code[w, 6:154][real]))
                ok_p &= bool((prediction[b][~real] == 0).all())
                ok_t &= bool(np.array_equal(target[b], cache_a["target"][w, 6:154].astype(np.float64)))
                ok_s &= bool(np.array_equal(support[b], S160[w, 6:154]))
        check("A8 fold%d %s: metrica riceve predizione/bersaglio/supporto allineati a 6..153" % (fold, name),
              ok_p and ok_t and ok_s and seen == [int(w) for w in rows])
        pv = independent_pose_var(cache_a["target"], cache_a["mask"])
        finite = pv[bundle_a.fitting[fold]]
        threshold = max(float(np.median(finite[np.isfinite(finite)])), 1e-12)
        check("A9 fold%d %s: soglia del bundle = mediana indipendente sul fitting (finestra intera)" % (fold, name),
              abs(threshold - bundle_a.thresholds[fold]) <= 1e-9 * threshold,
              "bundle %.12g indipendente %.12g" % (bundle_a.thresholds[fold], threshold))
        true_pv = bundle_a.pose_variances[fold]
        ties = int(np.sum(np.isfinite(true_pv[rows]) & (np.abs(true_pv[rows] - bundle_a.thresholds[fold]) <= 1e-9 * bundle_a.thresholds[fold])))
        check("A9b fold%d %s: pose_var del bundle = ricalcolo indipendente (rel 1e-9); finestre esattamente sulla soglia: %d" % (fold, name, ties),
              np.allclose(pv, true_pv, rtol=1e-9, atol=0, equal_nan=True))
        pred160 = code.copy()
        ind = independent_stats(pred160, cache_a["target"], S160, rows, true_pv, bundle_a.thresholds[fold])
        check("A10 fold%d %s: quattro accumulatori di _score_model = ricalcolo indipendente" % (fold, name),
              close(stats, ind), json.dumps({"vero": stats, "indipendente": {k: ind[k] for k in stats},
                                              "dinamiche": ind["n_dynamic_windows"]}))
    if fold == 0:
        # la loss sul fitting e la MSE di selezione sul fitting sono lo stesso numero
        stats_fit = W._score_model(ProbeHead(), bundle_a, fold, bundle_a.fitting[fold], "post", "cpu")
        check("A11 loss di training sull'intero fitting = mse_sum/window_count della selezione sul fitting",
              abs(loss - stats_fit["mse_sum"] / stats_fit["window_count"]) <= 1e-6 * max(1., loss) and
              stats_fit["window_count"] == ds.size,
              "loss %.9f selezione %.9f finestre %d/%d" % (loss, stats_fit["mse_sum"] / stats_fit["window_count"],
                                                           stats_fit["window_count"], ds.size))

# sensibilita': perturbare la predizione fuori da S non cambia nulla, dentro S cambia come previsto
fold = 0
rows = bundle_a.held[fold]
panel = W._metric_panel(bundle_a, fold, rows)
base_pred = code[rows][:, 6:154].copy()
base = M.selection_statistics(base_pred, panel, bundle_a.thresholds[fold])
rng = np.random.default_rng(3)
outside = base_pred.copy()
outside[~S160[rows][:, 6:154]] += rng.normal(size=int((~S160[rows][:, 6:154]).sum())) * 50
check("A12 perturbare la predizione fuori dal supporto non cambia i quattro accumulatori (uguaglianza esatta)",
      M.selection_statistics(outside, panel, bundle_a.thresholds[fold]) == base)
full_pred = code[rows].copy()                       # predizione lunga 160: il crop lo fa metrics._cropped
edges = full_pred.copy()
edges[:, :6] += 1000.
edges[:, 154:] -= 1000.
check("A13 predizione lunga 160: i 6+6 patch di bordo non entrano (uguaglianza esatta con il crop esplicito)",
      M.selection_statistics(edges, panel, bundle_a.thresholds[fold]) == base)
shifted = np.roll(cache_a["target"][rows].astype(np.float64), 1, axis=1)
exact = M.selection_statistics(cache_a["target"][rows].astype(np.float64), panel, bundle_a.thresholds[fold])
moved = M.selection_statistics(shifted, panel, bundle_a.thresholds[fold])
check("A14 predizione = bersaglio -> errore 0; bersaglio spostato di una patch -> errore > 0",
      exact["mse_sum"] == 0 and exact["dynamic_sse"] == 0 and moved["mse_sum"] > 0 and moved["dynamic_sse"] > 0,
      "spostato: mse_sum %.3f dynamic_sse %.3f" % (moved["mse_sum"], moved["dynamic_sse"]))

# ----------------------------------------------------------------------------
# PROVA B: mondo di controllo con le funzioni vere
# ----------------------------------------------------------------------------
print()
print("=== PROVA B: controllo sintetico, pannelli clean_ls / oracolo / testa ===")
cache_b = make_cache(coded=False, seed=11)
S160 = independent_support(cache_b)
real_b = W._prepare_real_folds(cache_b, allowed_train_users=USERS, source_identity={"pin": "toy"},
                               expected_user_count=len(USERS))
support_b = D.central_support(cache_b["mask"], D.lengths_for(cache_b), cache_b["key_time_valid"])
SELECTED = {"linear": {"a": 0., "sigma": 1.35, "alpha": 1.}, "temporal": {"a": .4, "sigma": 1.3}}
evaluate_calls = []
real_evaluate = M.evaluate


def recording_evaluate(prediction, data, dynamic_threshold=M.THRESHOLD, **kwargs):
    result = real_evaluate(prediction, data, dynamic_threshold, **kwargs)
    dyn = result["groups"]["dynamic"]
    evaluate_calls.append({"target": np.asarray(data["target"]).copy(), "mask": np.asarray(data["mask"]).copy(),
                           "pose_var": np.asarray(data["pose_var"]).copy(), "threshold": float(dynamic_threshold),
                           "ids": [C.canonical(m) for m in data["meta"]], "support_sha256": result["support_sha256"],
                           "n_dynamic_selected": dyn["n_selected_windows"],
                           "n_dynamic_centered": dyn["centered"]["n_windows"],
                           "n_cells": dyn["centered"]["n_window_joints"],
                           "n_elements": dyn["centered"]["n_valid_elements"],
                           "prediction": np.asarray(prediction).copy()})
    return result


artifacts, reference_calls = [], {}
for fold in (0, 1):
    fit, held = real_b.fitting[fold], real_b.held[fold]
    components = C.generate_components(cache_b["lat"], cache_b["meta"], support_b, fit, held, USERS,
                                       outer_fold=fold, n_joints=JOINTS, block_windows=4)
    check("B0 fold%d: L, T, epsilon definiti su 160 patch" % fold,
          all(np.asarray(components[k]).shape == (N, TIME, JOINTS) for k in ("L", "T", "epsilon")))
    evaluate_calls.clear()
    M.evaluate = recording_evaluate
    try:
        outer = C.evaluate_outer_references(cache_b, components, fit, held, SELECTED, block_windows=4,
                                            include_l13_diagnostics=False)
    finally:
        M.evaluate = real_evaluate
    # chiamate nell'ordine del codice: linear (constant, oracle, clean_ls, ridge), temporal (constant, oracle, clean_ls), poi diagnostici
    reference_calls[fold] = {"linear": dict(zip(("constant", "oracle", "clean_ls", "ridge"), evaluate_calls[:4])),
                             "temporal": dict(zip(("constant", "oracle", "clean_ls"), evaluate_calls[4:7]))}
    artifacts.append({"outer_fold": fold,
                      "partition": C.validate_outer_partition(cache_b["meta"], fit, held, USERS),
                      "components": components,
                      "calibration": {"status": "calibrated", "selected": SELECTED},
                      "outer_scored_after_inner_selection": True, "outer_reference_statistics": outer})

real_combine = C.combine_reference_folds


def forced_combine(values):
    out = dict(real_combine(values))
    out["toy_real_status"] = out["status"]
    out["status"] = "diagnostic"
    return out


C.combine_reference_folds = forced_combine
try:
    bundle_b = W._prepare_control_folds(cache_b, artifacts, "temporal", allowed_train_users=USERS,
                                        source_identity={"pin": "toy"}, expected_user_count=len(USERS))
finally:
    C.combine_reference_folds = real_combine
print("     (stato vero del cancello di riferimento sul mondo giocattolo: %s; forzato a 'diagnostic' solo per costruire il bundle)"
      % bundle_b.reference_gate["toy_real_status"])

# una "TCN" non addestrata ma con uscita non banale (qui si verifica il supporto, non l'apprendimento)
model = H.make_head("TCN_post", seed=0)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(5)
    for parameter in (model.affine.weight, model.deep[-1].weight):
        nn.init.normal_(parameter, std=.05)
model.eval()

head_stats, head_clean, identity_rows = {}, {}, {}
for fold in (0, 1):
    fit, held = bundle_b.fitting[fold], bundle_b.held[fold]
    ystar, y = bundle_b.clean_targets[fold], bundle_b.targets[fold]
    evaluate_calls.clear()
    M.evaluate = recording_evaluate
    try:
        observed, clean = W._score_model(model, bundle_b, fold, held, "post", "cpu", include_clean=True)
    finally:
        M.evaluate = real_evaluate
    head_stats[fold], head_clean[fold] = observed, clean
    call_y, call_clean = evaluate_calls[0], evaluate_calls[1]
    ref = reference_calls[fold]["temporal"]
    # --- uguaglianza dei pannelli (bit per bit) fra clean_ls, oracolo, costante e testa ---
    same = True
    for name in ("clean_ls", "oracle", "constant"):
        r = ref[name]
        same &= (np.array_equal(r["target"], call_y["target"]) and np.array_equal(r["mask"], call_y["mask"]) and
                 np.array_equal(r["pose_var"], call_y["pose_var"], equal_nan=True) and
                 r["threshold"] == call_y["threshold"] and r["ids"] == call_y["ids"] and
                 r["support_sha256"] == call_y["support_sha256"] and
                 r["n_dynamic_selected"] == call_y["n_dynamic_selected"] and r["n_cells"] == call_y["n_cells"] and
                 r["n_elements"] == call_y["n_elements"])
    check("B1 fold%d: pannello identico (bersaglio y, maschera, pose_var, soglia, finestre, supporto, celle dinamiche) per clean_ls/oracolo/costante/testa" % fold,
          same, "finestre %d, dinamiche %d, celle n>=2 %d, elementi %d, soglia %.12g" % (
              len(call_y["ids"]), call_y["n_dynamic_selected"], call_y["n_cells"], call_y["n_elements"], call_y["threshold"]))
    check("B2 fold%d: pannello pulito (y*) = stesso supporto, stessa pose_var, stessa soglia, stesse celle dinamiche del pannello y" % fold,
          np.array_equal(call_clean["mask"], call_y["mask"]) and
          np.array_equal(call_clean["pose_var"], call_y["pose_var"], equal_nan=True) and
          call_clean["threshold"] == call_y["threshold"] and call_clean["support_sha256"] == call_y["support_sha256"] and
          call_clean["n_dynamic_selected"] == call_y["n_dynamic_selected"] and call_clean["n_cells"] == call_y["n_cells"] and
          call_clean["n_elements"] == call_y["n_elements"] and
          np.array_equal(call_clean["target"], ystar[held]) and np.array_equal(call_y["target"], y[held]))
    # --- ricalcolo indipendente ---
    pv = independent_pose_var(ystar, cache_b["mask"])
    finite = pv[fit]
    threshold = max(float(np.median(finite[np.isfinite(finite)])), 1e-12)
    check("B3 fold%d: soglia = mediana sul fitting della varianza del bersaglio PULITO (finestra intera); pose_var dal pulito" % fold,
          abs(threshold - bundle_b.thresholds[fold]) <= 1e-9 * threshold and
          np.allclose(pv[held], call_y["pose_var"], rtol=1e-9, equal_nan=True) and
          bundle_b.thresholds[fold] == artifacts[fold]["outer_reference_statistics"]["temporal_threshold"],
          "soglia %.12g; pose_var dal rumoroso darebbe mediana %.6g" % (
              threshold, float(np.nanmedian(independent_pose_var(y, cache_b["mask"])[fit]))))
    pred_head = to160(call_y["prediction"])
    full = np.full((N, TIME, JOINTS), np.nan)
    full[held] = pred_head
    ind_head = independent_stats(full, y, S160, held, pv, threshold)
    check("B4 fold%d: testa contro y: quattro accumulatori = ricalcolo indipendente" % fold, close(observed, ind_head),
          json.dumps({"vero": observed, "indipendente": {k: ind_head[k] for k in observed}}))
    ind_head_clean = independent_stats(full, ystar, S160, held, pv, threshold)
    check("B5 fold%d: testa contro y*: quattro accumulatori = ricalcolo indipendente" % fold, close(clean, ind_head_clean))
    outer = artifacts[fold]["outer_reference_statistics"]["temporal"]
    ind_oracle = independent_stats(ystar, y, S160, held, pv, threshold)
    check("B6 fold%d: oracolo (y* contro y): accumulatori veri = ricalcolo indipendente" % fold, close(outer["oracle"], ind_oracle),
          json.dumps({"vero": outer["oracle"]}))
    full_ls = np.full((N, TIME, JOINTS), np.nan)
    full_ls[held] = to160(ref["clean_ls"]["prediction"])
    ind_ls = independent_stats(full_ls, y, S160, held, pv, threshold)
    check("B7 fold%d: clean_ls contro y: accumulatori veri = ricalcolo indipendente" % fold, close(outer["clean_ls"], ind_ls))
    check("B8 fold%d: denominatore dynamic_target_ss e window_count uguali fra clean_ls, oracolo, costante e testa" % fold,
          all(abs(outer[k]["dynamic_target_ss"] - observed["dynamic_target_ss"]) <= 1e-12 * observed["dynamic_target_ss"]
              for k in ("clean_ls", "oracle", "constant")) and
          len({outer[k]["window_count"] for k in ("clean_ls", "oracle", "constant")} | {observed["window_count"]}) == 1,
          "SS(y) %.9f, finestre %d" % (observed["dynamic_target_ss"], observed["window_count"]))
    # --- identita' SSE_y = SSE_y* + SSE_oracolo - 2<e,n> sullo stesso supporto dinamico ---
    cross = 0.
    cross_mse = 0.
    for w in held:
        s = S160[w]
        q = int(s.sum())
        if q > 0:
            e = (full[w] - ystar[w])[s]
            n = (y[w] - ystar[w])[s]
            cross_mse += float((e * n).sum() / q)
        if np.isfinite(pv[w]) and pv[w] >= threshold:
            for j in range(JOINTS):
                sel = s[:, j]
                if sel.sum() >= 2:
                    e = full[w, sel, j] - ystar[w, sel, j]
                    n = y[w, sel, j] - ystar[w, sel, j]
                    cross += float(((e - e.mean()) * (n - n.mean())).sum())
    lhs = observed["dynamic_sse"]
    rhs = clean["dynamic_sse"] + outer["oracle"]["dynamic_sse"] - 2 * cross
    lhs_mse = observed["mse_sum"]
    rhs_mse = clean["mse_sum"] + outer["oracle"]["mse_sum"] - 2 * cross_mse
    identity_rows[fold] = {"SSE_y": lhs, "SSE_ystar": clean["dynamic_sse"], "SSE_oracle": outer["oracle"]["dynamic_sse"],
                           "cross_e_n": cross, "rhs": rhs, "abs_residual": abs(lhs - rhs),
                           "mse_lhs": lhs_mse, "mse_rhs": rhs_mse, "mse_abs_residual": abs(lhs_mse - rhs_mse)}
    check("B9 fold%d: SSE_y = SSE_y* + SSE_oracolo - 2<e,n> (funzioni vere per i tre SSE, <e,n> indipendente)" % fold,
          abs(lhs - rhs) <= 1e-9 * abs(lhs), json.dumps(identity_rows[fold]))
    check("B10 fold%d: stessa identita' per la MSE per finestra" % fold, abs(lhs_mse - rhs_mse) <= 1e-9 * abs(lhs_mse))
    # clean_ls: stessa identita' con la funzione vera sul pannello pulito
    clean_panel = W._metric_panel(bundle_b, fold, held, clean=True)
    ls_clean = M.selection_statistics(ref["clean_ls"]["prediction"], clean_panel, bundle_b.thresholds[fold])
    cross_ls = 0.
    for w in held:
        s = S160[w]
        if np.isfinite(pv[w]) and pv[w] >= threshold:
            for j in range(JOINTS):
                sel = s[:, j]
                if sel.sum() >= 2:
                    e = full_ls[w, sel, j] - ystar[w, sel, j]
                    n = y[w, sel, j] - ystar[w, sel, j]
                    cross_ls += float(((e - e.mean()) * (n - n.mean())).sum())
    check("B11 fold%d: identita' anche per clean_ls" % fold,
          abs(outer["clean_ls"]["dynamic_sse"] - (ls_clean["dynamic_sse"] + outer["oracle"]["dynamic_sse"] - 2 * cross_ls))
          <= 1e-9 * outer["clean_ls"]["dynamic_sse"])
    # training set del controllo: bersaglio osservato del fold, supporto del bundle
    ds = W._training_set(bundle_b, fit, "post", fold)
    x, yb, mb = ds.batch(torch.arange(ds.size), torch.device("cpu"))
    order = [int(i) for i in ds._rows]
    ids = W._metadata_ids(cache_b)
    wins = [ids.index(ds.window_ids[i]) for i in order]
    check("B12 fold%d: training del controllo = y osservato del fold, crop 6..153, supporto S, sole finestre supportate" % fold,
          np.array_equal(yb.numpy(), y[wins][:, 6:154].astype(np.float32)) and
          np.array_equal(mb.numpy(), S160[wins][:, 6:154]) and
          sorted(wins) == sorted(int(w) for w in fit if S160[w].any()),
          "finestre %d su %d" % (ds.size, len(fit)))
    with torch.no_grad():
        loss = float(M.per_window_mse(model(x)[:, 6:154], yb, mb))
    fit_stats = W._score_model(model, bundle_b, fold, fit, "post", "cpu")
    check("B13 fold%d: loss di training sul fitting = MSE della metrica di selezione sul fitting" % fold,
          abs(loss - fit_stats["mse_sum"] / fit_stats["window_count"]) <= 2e-6 * loss and fit_stats["window_count"] == ds.size,
          "loss %.9f metrica %.9f" % (loss, fit_stats["mse_sum"] / fit_stats["window_count"]))

# ----------------------------------------------------------------------------
# pooled: somma degli accumulatori, nel cancello e nella selezione
# ----------------------------------------------------------------------------
print()
print("=== POOLED ===")
combined = real_combine(artifacts)
ref_sum = {name: {k: sum(a["outer_reference_statistics"]["temporal"][name][k] for a in artifacts)
                  for k in ("dynamic_sse", "dynamic_target_ss")} for name in ("clean_ls", "oracle")}
for name in ("clean_ls", "oracle"):
    ratio_of_sums = ref_sum[name]["dynamic_sse"] / ref_sum[name]["dynamic_target_ss"]
    mean_of_ratios = float(np.mean([a["outer_reference_statistics"]["temporal"][name]["dynamic_sse"] /
                                    a["outer_reference_statistics"]["temporal"][name]["dynamic_target_ss"] for a in artifacts]))
    check("P1 cancello: pooled %s = rapporto delle somme (non media dei rapporti)" % name,
          combined["temporal_records"]["pooled"][name] == ratio_of_sums and ratio_of_sums != mean_of_ratios,
          "somme %.12f, media dei rapporti %.12f" % (ratio_of_sums, mean_of_ratios))
pooled_head = C.pool_statistics([head_stats[0], head_stats[1]])
point = S.pool_scores([head_stats[0], head_stats[1]], 1.0, (1, 1))
check("P2 selezione: pool_scores = somme degli accumulatori = pool_statistics/statistic_values",
      abs(point.mse - C.statistic_values(pooled_head)["MSE"]) <= 1e-12 * point.mse and
      abs(point.edyn - C.statistic_values(pooled_head)["E_dyn"]) <= 1e-12 * point.edyn,
      "MSE %.12f E_dyn %.12f" % (point.mse, point.edyn))
gate = C.temporal_head_gate({k: v for k, v in combined["temporal_records"].items()},
                            {"fold0": C.statistic_values(head_stats[0])["E_dyn"],
                             "fold1": C.statistic_values(head_stats[1])["E_dyn"],
                             "pooled": C.statistic_values(pooled_head)["E_dyn"]}) \
    if C.temporal_reference_gate(combined["temporal_records"])["status"] == "diagnostic" else None
manual = {}
for split, e_head in (("fold0", C.statistic_values(head_stats[0])["E_dyn"]), ("fold1", C.statistic_values(head_stats[1])["E_dyn"]),
                      ("pooled", C.statistic_values(pooled_head)["E_dyn"])):
    row = combined["temporal_records"][split]
    manual[split] = C._recovery(row["clean_ls"], e_head, row["oracle"])
print("     R giocattolo (testa non addestrata, solo aritmetica):", manual,
      "| media dei due R di fold:", (manual["fold0"] + manual["fold1"]) / 2 if None not in manual.values() else None)

print()
print("ESITO: %d controlli, %d falliti%s" % (len(OUT), len(FAIL), (": " + ", ".join(FAIL)) if FAIL else ""))
json.dump({"checks": OUT, "failed": FAIL, "identity": {str(k): v for k, v in identity_rows.items()}},
          open(sys.argv[1], "w"), indent=1)
