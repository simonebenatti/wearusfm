"""verifica-rnc: test del PUNTO DI CHIAMATA vero (ContrastiveRuntime.attach) contro un oracolo numpy vettoriale e indipendente.
Serve da "uccisore" per i mutanti del punto di chiamata e da controllo a taglie di produzione (fino a 400 ancore, oltre il
blocco da 256). Encoder e micro-batch sono oggetti finti con i soli attributi letti da attach (v_common.fake_inputs)."""
import math

import numpy as np
import pytest
import torch

from v_common import make_windows, run_attach, PATCH_MS


def oracle(windows, *, temperature=0.1, stride=8, patch_ms=PATCH_MS):
    pts = [(w, p) for w, win in enumerate(windows) if win["y"] is not None
           for p in range(stride // 2, win["zbar"].shape[0], stride) if win["tv"][p] and win["m"][p].any()]
    n = len(pts)
    if n == 0:
        return 0.0, 0, {"anchors": 0, "points": 0, "dup": 0, "recordings": 0, "subjects": 0}
    U = np.stack([windows[w]["zbar"][p].astype(np.float64) for w, p in pts])
    U = U - U.mean(1, keepdims=True)
    U = U / np.sqrt((U * U).sum(1, keepdims=True))
    sim = U @ U.T / temperature
    Y = np.stack([windows[w]["y"][p].astype(np.float64) for w, p in pts])
    M = np.stack([windows[w]["m"][p] for w, p in pts])
    inst = []
    for w, p in pts:
        ds, sub, ses, trial, start, fs = windows[w]["ident"]
        inst.append((ds, sub, ses, trial, fs, start + int(math.ceil((p + 0.5) * patch_ms * fs / 1000.0 - 1e-9))))
    per_window, anchors = {}, 0
    for i in range(n):
        common = M & M[i][None]
        nc = common.sum(1)
        cand = nc > 0
        cand[i] = False
        cand &= np.array([inst[k] != inst[i] for k in range(n)])
        idx = np.nonzero(cand)[0]
        if len(idx) < 2:
            continue
        d = np.sqrt((((Y[idx] - Y[i][None]) ** 2) * common[idx]).sum(1) / nc[idx])
        s = sim[i, idx]
        tail = d[None, :] >= d[:, None]
        x = np.where(tail, s[None, :], -np.inf)
        mx = x.max(1)
        num = float((mx + np.log(np.exp(x - mx[:, None]).sum(1)) - s).sum())
        ref = float(np.log(tail.sum(1)).sum())
        per_window.setdefault(pts[i][0], []).append(num / ref)
        anchors += 1
    info = {"anchors": anchors, "points": n, "dup": n - len(set(inst)),
            "recordings": len({windows[w]["ident"][:3] for w, _ in pts}), "subjects": len({windows[w]["ident"][:2] for w, _ in pts})}
    if not per_window:
        return 0.0, 0, info
    return float(np.mean([np.mean(v) for v in per_window.values()])), len(per_window), info


def check(windows, arm="CR", fd=True, seed=0, cfg_over=None):
    terms, enc, engine = run_attach(windows, arm=arm, cfg_over=cfg_over)
    got = terms.num["contrast_rnc"]
    ref, n_win, info = oracle(windows, stride=engine.cfg.rnc_stride, temperature=engine.cfg.rnc_temperature)
    assert float(got) == pytest.approx(ref, rel=1e-10, abs=1e-12)
    logs = {k: float(v) for k, v in terms.log_num.items()}
    assert int(logs["contrast_rnc_windows"]) == n_win and int(logs["contrast_rnc_anchors"]) == info["anchors"]
    assert int(logs["contrast_rnc_duplicate_instants"]) == info["dup"]
    assert int(logs["contrast_rnc_recordings"]) == info["recordings"] and int(logs["contrast_rnc_subjects"]) == info["subjects"]
    assert all(float(terms.log_den[k]) == 1.0 for k in logs)
    assert int(terms.den["contrast_rnc"]) == 1 and terms.den["contrast_rnc"].dtype == torch.int64
    wn, wr = engine.cfg.nce_weight, engine.cfg.rnc_weight
    assert terms.weight["contrast_rnc"] == (wr if arm == "CR" else 0.0) and terms.weight["contrast_nce"] == (wn if arm == "CN" else 0.0)
    assert logs["contrast_rnc_weight"] == pytest.approx(terms.weight["contrast_rnc"]) and logs["contrast_nce_weight"] == pytest.approx(terms.weight["contrast_nce"])
    assert engine.pending_stats["rnc_active"] == (n_win > 0)
    if n_win and fd:
        g, = torch.autograd.grad(got, enc.z, allow_unused=True)
        assert g is not None and torch.isfinite(g).all()
        for s, w in enumerate(windows):                                   # nessun gradiente nel riempimento e fuori dalle ancore
            P = w["z"].shape[0]
            assert float(g[s, P:].abs().max()) == 0.0 if P < g.shape[1] else True
            used = np.zeros(P, bool)
            if w["y"] is not None:
                for p in range(engine.cfg.rnc_stride // 2, P, engine.cfg.rnc_stride):
                    used[p] = bool(w["tv"][p] and w["m"][p].any())
            assert float(g[s, :P][torch.from_numpy(~used)].abs().max()) == 0.0 if (~used).any() else True
        rng = np.random.default_rng(seed)
        dirs = [rng.standard_normal(w["z"].shape) for w in windows]
        eps = 1e-5
        vals = []
        for sign in (+1, -1):
            pert = [dict(w, z=w["z"] + sign * eps * d, zbar=(w["z"] + sign * eps * d).mean(1)) for w, d in zip(windows, dirs)]
            vals.append(oracle(pert, stride=engine.cfg.rnc_stride, temperature=engine.cfg.rnc_temperature)[0])
        fdv = (vals[0] - vals[1]) / (2 * eps)
        ad = sum(float((g[s, :w["z"].shape[0]].numpy() * d).sum()) for s, (w, d) in enumerate(zip(windows, dirs)))
        assert ad == pytest.approx(fdv, rel=1e-5, abs=1e-9)
    return float(got), n_win, info


def specs_for(seed):
    g = np.random.default_rng(5000 + seed)
    n_rec, n_win = int(g.integers(1, 4)), int(g.integers(2, 9))
    specs = []
    for _ in range(n_win):
        rec = int(g.integers(0, n_rec))
        specs.append((f"u{rec % 2}", f"s{rec}", int(g.integers(0, 30)) * 400, int(g.choice([40, 80, 120, 160])), bool(g.random() > 0.2)))
    if seed % 2 == 0:
        specs.append(specs[0])                                             # copia fisica (altro montaggio)
    if seed % 3 == 1:
        s0 = specs[0]
        specs.append((s0[0], s0[1], s0[2] + 400 * int(g.integers(1, 6)), s0[3], True))   # sovrapposta sulla griglia
    return specs


@pytest.mark.parametrize("seed", range(10))
def test_attach_value_logs_and_gradient_match_numpy_oracle(seed):
    wins = make_windows(seed, specs_for(seed), per_joint_masks=(seed % 4 == 1), invalid_block_prob=(0.25 if seed % 2 else 0.0))
    arm = ("CR", "CN", "C0")[seed % 3]
    check(wins, arm=arm, seed=seed)


def test_attach_at_production_population_beyond_one_distance_chunk():
    """20 finestre da 4 s di 3 registrazioni con sovrapposizioni: 400 ancore (> 256), con istanti duplicati."""
    g = np.random.default_rng(77)
    specs = [(f"u{r}", f"s{r}", int(g.integers(0, 12)) * 400, 160, True) for r in g.integers(0, 3, size=20)]
    wins = make_windows(77, specs, K=2, D=16)
    value, n_win, info = check(wins, arm="CR", seed=77)
    assert info["points"] == 400 and info["dup"] > 20 and n_win == 20


def test_attach_identity_fields_all_matter_for_the_same_instant_rule():
    """Stesso indice di campione ma dataset, soggetto, sessione, trial o frequenza diversi NON sono lo stesso istante."""
    base = make_windows(3, [("u0", "s0", 0, 40, True), ("u0", "s0", 0, 40, True), ("u0", "s0", 400, 40, True)])
    ref_value, _, info = check(base, seed=1)
    assert info["dup"] == 5 + 4                                             # copia intera (5 ancore) + sovrapposta (4 ancore, 3 copie -> ...)
    for field, value in ((0, "altro_dataset"), (1, "u9"), (2, "s9"), (3, 7), (5, 1000.0)):
        wins = [dict(w) for w in base]
        ident = list(wins[1]["ident"])
        ident[field] = value
        if field == 5:
            ident[4] = 112                                                 # fs 1000: 112 + ceil((p+0,5)*25) = 225 per p = 4, come la finestra 0
        wins[1]["ident"] = tuple(ident)
        v, _, inf = check(wins, seed=2)
        assert inf["dup"] < info["dup"]
        assert abs(v - ref_value) > 1e-6                                   # il valore cambia davvero quando la coppia rientra


def test_attach_without_any_pose_and_with_two_anchors_is_zero_with_graph():
    wins = make_windows(5, [("u0", "s0", 0, 40, False), ("u1", "s1", 0, 40, False)])
    terms, enc, engine = run_attach(wins, arm="CR")
    assert float(terms.num["contrast_rnc"]) == 0.0 and terms.num["contrast_rnc"].requires_grad
    assert engine.pending_stats["rnc_active"] is False and float(terms.log_num["contrast_rnc_windows"]) == 0
    wins = make_windows(6, [("u0", "s0", 0, 16, True), ("u1", "s1", 0, 40, False)])   # 2 ancore (patch 4 e 12)
    v, n_win, info = check(wins)
    assert v == 0.0 and n_win == 0 and info["points"] == 2 and info["anchors"] == 0
    wins = make_windows(6, [("u0", "s0", 0, 24, True), ("u1", "s1", 0, 40, False)])   # 3 ancore: attivo
    v, n_win, info = check(wins)
    assert n_win == 1 and info["anchors"] == 3 and v > 0
