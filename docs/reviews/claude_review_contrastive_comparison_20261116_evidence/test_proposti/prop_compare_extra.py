"""Controlli aggiuntivi indipendenti per scripts/contrastive_compare.py (review di Claude, 16/11/2026).

Motivo: nel file consegnato tutti i dati sintetici hanno soggetti, utenti e dataset identici fra loro, quindi
varianza bootstrap nulla e pesi mai esercitati. Qui: dati con varianza, dataset diversi, oracolo indipendente.
Solo dati sintetici. Da copiare in tests/cpu/ accanto a test_contrastive_compare.py.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_compare as C

PLAN = ROOT / "results/infra/contrastive_comparison_20261116/plan.json"
SHAPES = {"epn612": (12, 3, 30), "ninapro_db2": (9, 4, 24), "ninapro_db3": (5, 4, 24), "ninapro_db6": (6, 3, 30), "uci_emg": (7, 3, 18)}
BASE = {"epn612": .80, "ninapro_db2": .35, "ninapro_db3": .20, "ninapro_db6": .50, "uci_emg": .70}
Q = .05 / 18


def rich(seed=0, rare=()):
    """OOF e posa con soggetti/utenti/dataset diversi fra loro; errori appaiati ma non identici fra bracci."""
    rng = np.random.default_rng(seed)
    cl = {a: {} for a in C.ARMS}
    for ds, (ns, nc, per) in SHAPES.items():
        subj = np.repeat(np.arange(ns), per)
        n = len(subj)
        y = rng.integers(0, nc - (ds in rare), n)
        if ds in rare:
            y[:5] = nc - 1                      # classe presente in un solo soggetto
        skill = rng.normal(0, .08, ns)[subj]
        u = rng.random(n)
        wrong = (y + rng.integers(1, nc, n)) % nc
        for i, a in enumerate(C.ARMS):
            ua = np.where(rng.random(n) < .3, rng.random(n), u)
            ok = ua < np.clip(BASE[ds] + skill + .03 * i, .02, .98)
            cl[a][ds] = {"row": np.arange(n), "y": y.copy(), "pred": np.where(ok, y, wrong),
                         "s": np.array([f"s{k:02d}" for k in subj]), "fold": (subj % 4).astype(np.int64)}
    vary, ratio = np.exp(rng.normal(3, .8, 9)), .85 + rng.normal(0, .05, 9)
    po = {}
    for i, a in enumerate(C.ARMS):
        r = ratio * (1 - .02 * i) * np.exp(rng.normal(0, .01, 9))
        rows = {f"u{k}": {"n_windows": 5 + k, "n_window_joints": 100 + k, "n_valid_elements": 1000 + k,
                          "sums": {"vary": float(vary[k]), "centered": float(vary[k] * r[k])}} for k in range(9)}
        po[a] = cell(a, rows)
    return cl, po


def cell(arm, rows, step=19618):
    return {"complete": True, "arm": arm, "checkpoint_step": step, "which": "teacher",
            "checkpoint_sha256": hashlib.sha256(arm.encode()).hexdigest(), "fit": {"selection": {"boundary_unresolved": False}},
            "analysis": {"support_sha256": "s" * 64, "targets_sha256": "t" * 64, "dynamic_threshold": .05,
                         "groups": {"dynamic": {"n_selected_windows": 80, "centered": {"per_user": rows}}}}}


def flat(edyn=None, correct=None):
    """Soggetti e utenti identici (varianza nulla): serve solo alla logica della lettura."""
    edyn, correct = edyn or {}, correct or {}
    cl, po = {}, {}
    for a in C.ARMS:
        y = np.tile(np.repeat(np.array([0, 1]), 50), 4)
        k = correct.get(a, 40)
        pred = np.where(np.tile(np.arange(50), 8) < k, y, 1 - y)
        cl[a] = {ds: {"row": np.arange(400), "y": y.copy(), "pred": pred.copy(), "s": np.repeat(np.array(["a", "b", "c", "d"]), 100),
                      "fold": np.repeat(np.arange(4), 100)} for ds in C.DATASETS}
        rows = {f"u{i}": {"n_windows": 2, "n_window_joints": 40, "n_valid_elements": 400,
                          "sums": {"vary": 10., "centered": 10. * edyn.get(a, .8)}} for i in range(4)}
        po[a] = cell(a, rows)
    return cl, po


def audits():
    out = []
    for a in C.ARMS:
        for task, dss in C.TASKS.items():
            for ds in dss:
                for fold in range(4):
                    for phase, grid in (("candidate", C.C_GRID), ("refit", (1.,))):
                        for c in grid:
                            out.append({"arm": a, "task": task, "dataset": ds, "fold": fold, "fit_phase": phase, "C": c,
                                        "checkpoint_sha256": hashlib.sha256(a.encode()).hexdigest(), "n_iter": 9, "max_iter": 12000,
                                        "exception": None, "convergence_warnings": 0, "converged_by_criterion": True})
    return out


def oracle_classification(cl, n_boot, seed):
    rng = np.random.default_rng(seed)
    pts, smp, missing = {}, {}, {}
    for ds in sorted(SHAPES):
        y, s = cl["C0"][ds]["y"], cl["C0"][ds]["s"]
        classes, subjects = np.unique(y), np.unique(s)
        n = np.array([[np.sum((s == u) & (y == c)) for c in classes] for u in subjects], dtype=float)
        draws = rng.integers(0, len(subjects), size=(n_boot, len(subjects)))
        mult = np.stack([np.bincount(d, minlength=len(subjects)) for d in draws]).astype(float)
        den = mult @ n
        present = den > 0
        missing[ds] = int((present.sum(1) < len(classes)).sum())
        p, v = [], []
        for a in C.ARMS:
            ok = cl[a][ds]["pred"] == y
            h = np.array([[np.sum((s == u) & (y == c) & ok) for c in classes] for u in subjects], dtype=float)
            p.append(np.mean(h.sum(0) / n.sum(0)))
            v.append(np.where(present, (mult @ h) / np.where(present, den, 1.), 0.).sum(1) / present.sum(1))
        pts[ds], smp[ds] = np.array(p), np.stack(v, 1)
    w = {"p1": {"ninapro_db2": 8 / 13, "ninapro_db3": 3 / 13, "ninapro_db6": 2 / 13}, "p2": {"epn612": .5, "uci_emg": .5}}
    return ({t: sum(k * pts[d] for d, k in ww.items()) for t, ww in w.items()},
            {t: sum(k * smp[d] for d, k in ww.items()) for t, ww in w.items()}, pts, missing)


def test_constants_match_protocol_and_plan():
    assert (C.N_BOOT, C.SEED, C.FAMILY_SIZE, C.ALPHA) == (100_000, 20261116, 9, .05)
    assert C.PAIRS == (("CN", "C0"), ("CR", "C0"), ("CN", "CR")) and C.C_GRID == (.001, .01, .1, 1., 10., 100.)
    assert C.MARGINS == {"E_dyn": {"gain": 2., "ni": 2.}, "p1": {"gain": .5, "ni": 1.}, "p2": {"gain": .5, "ni": 1.}}
    assert dict(zip(C.TASKS["p1"], C.WEIGHTS["p1"])) == {"ninapro_db2": 8 / 13, "ninapro_db3": 3 / 13, "ninapro_db6": 2 / 13}
    assert dict(zip(C.TASKS["p2"], C.WEIGHTS["p2"])) == {"epn612": .5, "uci_emg": .5}
    if not PLAN.exists():
        pytest.skip("plan.json non presente in questo albero")
    ev = json.loads(PLAN.read_text())["evaluation"]
    assert ev["bootstrap"]["replicates"] == C.N_BOOT and ev["bootstrap"]["seed"] == C.SEED
    assert ev["bootstrap"]["family_size"] == C.FAMILY_SIZE and ev["bootstrap"]["family_alpha"] == C.ALPHA
    assert [tuple(p) for p in ev["pairs"]] == list(C.PAIRS) and ev["expected_classification_fits"] == 3 * 5 * 4 * 7
    for t in C.TASKS:
        assert ev["weights"][t] == pytest.approx(dict(zip(C.TASKS[t], C.WEIGHTS[t])), abs=1e-15)
    assert ev["margins"]["E_dyn"] == {"useful_reduction_percent": 2., "noninferiority_increase_percent": 2.}
    for t in ("p1", "p2"):
        assert ev["margins"][t] == {"useful_gain_pp": C.MARGINS[t]["gain"], "noninferiority_loss_pp": C.MARGINS[t]["ni"]}


def test_interval_levels_and_order():
    r = C.intervals(np.arange(100_001.))
    assert r["ci95"] == pytest.approx([2500., 97500.])
    assert r["ci_simultaneous"] == pytest.approx([Q * 100_000, (1 - Q) * 100_000])
    assert r["ci_simultaneous"][0] < r["ci95"][0] < r["ci95"][1] < r["ci_simultaneous"][1]
    with pytest.raises(ValueError):
        C.intervals(np.array([0., np.nan]))


def test_classification_matches_independent_oracle_with_weights_and_missing_classes():
    cl, _ = rich(1, rare=("uci_emg", "ninapro_db3"))
    tp, ts, rep = C.classification_statistics(cl, 300, 7)
    op, os_, dpts, missing = oracle_classification(cl, 300, 7)
    for t in C.TASKS:
        np.testing.assert_allclose(tp[t], op[t], rtol=0, atol=1e-12)
        np.testing.assert_allclose(ts[t], os_[t], rtol=0, atol=1e-12)
        assert np.ptp(ts[t][:, 0]) > 0                     # la prova ha davvero varianza
    for ds in C.DATASETS:
        assert rep[ds]["replicates_missing_classes"] == missing[ds]
        assert list(rep[ds]["balanced_accuracy"].values()) == pytest.approx(dpts[ds].tolist(), abs=1e-12)
    assert missing["uci_emg"] > 0 and missing["ninapro_db3"] > 0 and missing["epn612"] == 0
    assert not np.allclose(dpts["ninapro_db2"], dpts["ninapro_db3"])   # i pesi contano


def test_pose_is_ratio_of_sums_with_shared_user_draws():
    _, po = rich(2)
    point, values, rep = C.pose_statistics(po, 19618, 300, 11)
    users = sorted(po["C0"]["analysis"]["groups"]["dynamic"]["centered"]["per_user"])
    den = np.array([po["C0"]["analysis"]["groups"]["dynamic"]["centered"]["per_user"][u]["sums"]["vary"] for u in users])
    err = np.array([[po[a]["analysis"]["groups"]["dynamic"]["centered"]["per_user"][u]["sums"]["centered"] for u in users] for a in C.ARMS])
    draws = np.random.default_rng(11).integers(0, len(users), (300, len(users)))
    mult = np.stack([np.bincount(d, minlength=len(users)) for d in draws]).astype(float)
    np.testing.assert_allclose(point, err.sum(1) / den.sum(), rtol=0, atol=1e-14)
    np.testing.assert_allclose(values, (mult @ err.T) / (mult @ den)[:, None], rtol=0, atol=1e-12)
    assert not np.allclose(point, (err / den).mean(1))       # diverso dalla media dei rapporti per utente


def crafted(spec):
    """spec[endpoint] = (stima, minimo, massimo) del beneficio CN-C0; repliche su griglia uniforme: quantile(p)=min+p*(max-min)."""
    n, points, samples = 100_001, {}, {}
    for e in C.MARGINS:
        est, lo, hi = spec.get(e, (0., -.01, .01))
        grid = np.linspace(lo, hi, n)
        if e == "E_dyn":
            points[e], samples[e] = np.array([1., 1 - est / 100, 1.]), np.stack([np.ones(n), 1 - grid / 100, np.ones(n)], 1)
        else:
            points[e], samples[e] = np.array([0., est / 100, 0.]), np.stack([np.zeros(n), grid / 100, np.zeros(n)], 1)
    return points, samples


@pytest.mark.parametrize("endpoint,gain,ni", [("E_dyn", 2., 2.), ("p1", .5, 1.), ("p2", .5, 1.)])
def test_flags_use_simultaneous_lower_bound_and_practical_threshold(endpoint, gain, ni):
    def flags(est, lo, hi):
        r = C.summarize_pair("CN", "C0", *crafted({endpoint: (est, lo, hi)}))[endpoint]
        assert r["benefit"] == pytest.approx(est, abs=1e-9)
        assert r["ci_simultaneous"][0] == pytest.approx(lo + Q * (hi - lo), abs=1e-6)
        return r["useful_superiority"], r["noninferiority"]
    # stima sopra la soglia pratica, IC95 sopra zero, IC simultaneo che tocca lo zero: NON utile
    assert flags(gain + .3, -.1, 9.9) == (False, True)
    # IC simultaneo sopra zero ma stima sotto la soglia pratica: NON utile
    assert flags(gain - .1, .01, gain) == (False, True)
    # entrambe le condizioni: utile
    assert flags(gain + .3, .01, gain + 1) == (True, True)
    assert flags(gain, .01, gain + 1)[0] is True and flags(gain - 1e-6, .01, gain + 1)[0] is False
    # non inferiorita': limite inferiore simultaneo appena sotto / sopra il margine, con IC95 e stima sopra
    assert flags(0., -ni - .1, 9.9 - ni) == (False, False)
    assert flags(0., -ni + .01, ni)[1] is True


def test_cr_can_be_supported_and_direct_comparison_alone_is_not_enough():
    cl, po = flat(edyn={"CR": .76})
    assert C.compare(cl, po, audit_records=audits())["reading"] == "CR_supported_for_confirmation"
    cl, po = flat(edyn={"CR": .84})      # CN = C0, CR peggiore: CN batte CR ma non C0
    r = C.compare(cl, po, audit_records=audits())
    assert r["contrasts"]["CN_minus_CR"]["useful_without_material_loss"] is True
    assert r["contrasts"]["CN_minus_C0"]["useful_without_material_loss"] is False and r["reading"] == "unresolved_or_tradeoff"
    cl, po = flat(edyn={"CN": .84})      # CR = C0, CN peggiore: CR batte CN ma non C0
    r = C.compare(cl, po, audit_records=audits())
    assert r["reverse_head_to_head"]["useful_without_material_loss"] is True and r["reading"] == "unresolved_or_tradeoff"
    cl, po = flat(correct={"CN": 45})    # +10 punti su P1 e P2 per CN
    assert C.compare(cl, po, audit_records=audits())["reading"] == "CN_supported_for_confirmation"


def withheld(r):
    assert r["decision_valid"] is False and r["reading"] == "interpretation_withheld"
    for pair in [*r["contrasts"].values(), r["reverse_head_to_head"]]:
        assert pair["useful_without_material_loss"] is None
        assert all(pair[e]["useful_superiority"] is None and pair[e]["noninferiority"] is None for e in C.MARGINS)
        assert all(isinstance(pair[e]["benefit"], float) for e in C.MARGINS)       # le stime restano


@pytest.mark.parametrize("case", ["limit", "warning", "ridge_true", "ridge_missing", "seed", "no_audit"])
def test_verdict_is_withheld_at_full_bootstrap(case):
    cl, po = flat(edyn={"CN": .76})
    a, kw = audits(), {}
    if case == "limit":
        a[5].update(n_iter=12000, converged_by_criterion=False)
    elif case == "warning":
        a[5].update(convergence_warnings=2, converged_by_criterion=False)
    elif case == "ridge_true":
        po["CR"]["fit"]["selection"]["boundary_unresolved"] = True
    elif case == "ridge_missing":
        del po["C0"]["fit"]["selection"]["boundary_unresolved"]
    elif case == "seed":
        kw["seed"] = C.SEED + 1
    else:
        a = None
    r = C.compare(cl, po, audit_records=a, **kw)
    withheld(r)
    if case in ("limit", "warning"):
        assert r["fit_audit"]["converged"] is False and len(r["fit_audit"]["unresolved"]) == 1
    assert C.compare(*flat(edyn={"CN": .76}), audit_records=audits())["reading"] == "CN_supported_for_confirmation"


@pytest.mark.parametrize("case", ["extra", "grid", "refit_c", "exception", "max_iter", "n_iter_zero", "n_iter_str", "warnings_negative", "warning_flag"])
def test_audit_record_guards(case):
    a = audits()
    if case == "extra":
        a.append(dict(a[0], arm="ZZ"))
    elif case == "grid":
        a[0]["C"] = a[1]["C"]
    elif case == "refit_c":
        next(r for r in a if r["fit_phase"] == "refit")["C"] = 5.
    elif case == "exception":
        a[3]["exception"] = "RuntimeError: boom"
    elif case == "max_iter":
        a[3]["max_iter"] = 100
    elif case == "n_iter_zero":
        a[3]["n_iter"] = 0
    elif case == "n_iter_str":
        a[3]["n_iter"] = "9"
    elif case == "warnings_negative":
        a[3].update(convergence_warnings=-1, converged_by_criterion=False)
    else:
        a[3]["convergence_warnings"] = 1          # flag rimasto True
    with pytest.raises(ValueError):
        C.verify_audits(a)


@pytest.mark.parametrize("case", ["duplicate_rows", "fold_range", "subject_split", "negative_row", "extra_key", "one_subject"])
def test_oof_validation_when_all_arms_agree(case):
    cl, po = rich(3)
    for a in C.ARMS:
        d = cl[a]["epn612"]
        if case == "duplicate_rows":
            d["row"][1] = d["row"][0]
        elif case == "fold_range":
            d["fold"][d["s"] == "s00"] = 7
        elif case == "subject_split":
            d["fold"][0] = (d["fold"][0] + 1) % 4
        elif case == "negative_row":
            d["row"][0] = -1
        elif case == "extra_key":
            d["extra"] = d["row"].copy()
        else:
            keep = d["s"] == "s00"
            cl[a]["epn612"] = {k: v[keep] for k, v in d.items()}
    with pytest.raises(ValueError):
        C.classification_statistics(cl, 100, 0)


@pytest.mark.parametrize("case", ["y_swap", "fold_move", "row_id"])
def test_oof_pairing_detects_subtle_differences_in_one_arm(case):
    cl, _ = rich(4)
    d = cl["CN"]["ninapro_db6"]
    if case == "y_swap":                      # stesso soggetto, stessi conteggi per classe: cambia solo l'abbinamento
        idx = np.flatnonzero(d["s"] == "s01")
        i, j = idx[d["y"][idx] == 0][0], idx[d["y"][idx] == 1][0]
        d["y"][i], d["y"][j] = d["y"][j], d["y"][i]
    elif case == "fold_move":                 # un soggetto intero in un altro fold
        d["fold"][d["s"] == "s01"] = (d["fold"][d["s"] == "s01"] + 1) % 4
    else:                                     # identificativo di riga diverso, ordine invariato
        d["row"][-1] += 1000
    with pytest.raises(ValueError, match="pairing"):
        C.classification_statistics(cl, 100, 0)


def test_row_order_is_irrelevant():
    cl, _ = rich(5)
    ref = C.classification_statistics(cl, 100, 3)
    rng = np.random.default_rng(0)
    for a in C.ARMS:
        for ds in C.DATASETS:
            perm = rng.permutation(len(cl[a][ds]["row"]))
            cl[a][ds] = {k: v[perm] for k, v in cl[a][ds].items()}
    got = C.classification_statistics(cl, 100, 3)
    for t in C.TASKS:
        np.testing.assert_array_equal(ref[0][t], got[0][t])
        np.testing.assert_array_equal(ref[1][t], got[1][t])


@pytest.mark.parametrize("case", ["complete", "arm", "selected", "joints", "elements", "negative", "zero_error", "one_user", "extra_arm", "step5000", "pin_format"])
def test_pose_and_identity_guards(case):
    cl, po = rich(6)
    rows = po["CN"]["analysis"]["groups"]["dynamic"]["centered"]["per_user"]
    a, kw = audits(), {}
    if case == "complete":
        po["CN"]["complete"] = False
    elif case == "arm":
        po["CN"]["arm"] = "CR"
    elif case == "selected":
        po["CN"]["analysis"]["groups"]["dynamic"]["n_selected_windows"] += 1
    elif case == "joints":
        rows["u0"]["n_window_joints"] += 1
    elif case == "elements":
        rows["u0"]["n_valid_elements"] += 1
    elif case == "negative":
        rows["u0"]["sums"]["centered"] = -1e-9
    elif case == "zero_error":
        for r in rows.values():
            r["sums"]["centered"] = 0.
    elif case == "one_user":
        for arm in C.ARMS:
            per = po[arm]["analysis"]["groups"]["dynamic"]["centered"]["per_user"]
            for u in list(per)[1:]:
                del per[u]
    elif case == "extra_arm":
        po["CX"] = dict(copy.deepcopy(po["C0"]), arm="CX")
    elif case == "step5000":
        for arm in C.ARMS:
            po[arm]["checkpoint_step"] = 5000
        kw["step"] = 5000
    else:
        po["CN"]["checkpoint_sha256"] = "xyz"
        for r in a:
            if r["arm"] == "CN":
                r["checkpoint_sha256"] = "xyz"
    with pytest.raises(ValueError):
        C.compare(cl, po, audit_records=a, n_boot=100, **kw)


def test_too_few_replicates_are_refused():
    cl, po = rich(7)
    with pytest.raises(ValueError):
        C.compare(cl, po, n_boot=99)


def write_manifest(folder, cl, po, records, edit=None):
    manifest = {"protocol": C.PROTOCOL, "step": 19618, "which": "teacher", "pose": {}, "classification": {}}

    def pin(p):
        return {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for a in C.ARMS:
        p = folder / f"{a}.pose.json"
        p.write_text(json.dumps(po[a]))
        manifest["pose"][a] = pin(p)
        manifest["classification"][a] = {}
        for task, dss in C.TASKS.items():
            o = folder / f"{a}.{task}.npz"
            np.savez(o, **{f"{ds}_{k}": v for ds in dss for k, v in cl[a][ds].items()})
            j = folder / f"{a}.{task}.audit.jsonl"
            j.write_text("\n".join(json.dumps(r) for r in records if r["arm"] == a and r["task"] == task) + "\n")
            manifest["classification"][a][task] = {"oof": pin(o), "audit": pin(j)}
    if edit:
        edit(manifest)
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("case", ["protocol", "step", "which", "pin_extra", "cell_extra", "task_extra", "audit_task"])
def test_manifest_guards(tmp_path, case):
    cl, po = flat()
    records = audits()

    def edit(m):
        if case == "protocol":
            m["protocol"] = "altro"
        elif case == "step":
            m["step"] = 5000
        elif case == "which":
            m["which"] = "student"
        elif case == "pin_extra":
            m["pose"]["CN"]["note"] = "x"
        elif case == "cell_extra":
            m["classification"]["CN"]["p1"]["report"] = m["classification"]["CN"]["p1"]["audit"]
        elif case == "task_extra":
            m["classification"]["CN"]["p3"] = m["classification"]["CN"]["p1"]
    if case == "audit_task":                  # un record di P2 finito nel file di P1 dello stesso braccio
        moved = next(r for r in records if r["arm"] == "CN" and r["task"] == "p2")
        moved["task"] = "p1"
    (tmp_path / "ok").mkdir()
    (tmp_path / "bad").mkdir()
    assert C.from_manifest(write_manifest(tmp_path / "ok", cl, po, audits()))["decision_valid"] is True
    with pytest.raises(ValueError):
        C.from_manifest(write_manifest(tmp_path / "bad", cl, po, records, edit))


def test_cli_refuses_to_overwrite(tmp_path, monkeypatch):
    cl, po = flat()
    m = write_manifest(tmp_path, cl, po, audits())
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["contrastive_compare.py", "--manifest", str(m), "--out", str(out)])
    C.main()
    first = out.read_bytes()
    assert json.loads(first)["decision_valid"] is True
    with pytest.raises(ValueError, match="overwrite"):
        C.main()
    assert out.read_bytes() == first
