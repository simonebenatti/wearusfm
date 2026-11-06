"""Prove indipendenti di Claude sugli stadi finali dello scaling 30M/300M (review del 06/11/2026, codice aef665b).

Non sostituiscono i test di GPT: ne riusano le fixture (cache, piani e lettore della posa sintetici, codice reale) e
controllano cio' che quei test non esercitano:
- la disposizione REALE dei 14 fit pubblicati (piu' shard per cella e compito), letta da `submission_plan.json`;
- l'indipendenza di aggregati e confronto dalla disposizione e dall'ordine degli shard;
- il rifiuto di buchi, doppioni e shard estranei quando gli shard sono piu' d'uno;
- che il comparatore risalga davvero agli shard (aggregato che ne tace uno, shard con predizioni diverse);
- l'impronta `identity_sha256` ricostruita con la sola libreria standard (file `..._oof_identity.py`);
- punti, pesi, segni e intervalli ricalcolati per un'altra via, e i tempi sulle dimensioni reali degli OOF.
Nessun dato reale, nessun job, nessuno score reale.

Uso, dalla radice di un checkout del ramo di GPT:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/cpu \
  ~/.venvs/wearusfm-tok/bin/python -m pytest -q -s -p no:cacheprovider <questo file>
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

import test_scaling_compare as TC
from test_scaling_compare import scene  # noqa: F401  fixture di GPT: scena completa, uno shard per aggregato

M, CODE, pin, save_json = TC.M, TC.CODE, TC.pin, TC.save_json
C = M.C
HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("oof_identity", HERE.with_name(HERE.name.replace("_checks.py", "_oof_identity.py")))
OI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OI)
LAYOUT = json.loads((TC.ROOT / "results/infra/scaling_300m_20261104/cpu_fit_release_20261105/submission_plan.json").read_text())["jobs"][1:]
NAMES = ("T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched")
WEIGHTS = {"p1": {"ninapro_db2": 8/13, "ninapro_db3": 3/13, "ninapro_db6": 2/13}, "p2": {"epn612": .5, "uci_emg": .5}}
CONTRASTS = {"natural_300m_minus_30m": (1, 0), "matched_300m_minus_30m": (2, 0), "natural_300m_minus_matched_300m": (1, 2)}


@pytest.fixture(scope="module")
def fourteen(scene):
    root, probe = scene["root"], scene["probe"]
    identity = {"code_sha": CODE, "plan_sha256": scene["inputs"]["probe_plan"]["sha256"]}
    benchmarks = {cell: {ds: pin(root/(cell+"-"+ds+"-bench")/"benchmark.json") for ds in C.DATASETS} for cell in C.CELLS}
    shards = [{"cell": j["cell"], "task": j["task"], "datasets": j["datasets"], "folds": j["folds"],
               "cpus": 1, "memory_gib": 50, "walltime_seconds": 10000} for j in LAYOUT]
    resource = save_json(root/"claude14-resources.json", {"protocol": C.RESOURCE_PROTOCOL, "complete": True,
        "plan_content_sha256": C.jhash(probe), "code_sha": CODE, "caps": {"cpus": 4, "memory_gib": 60, "walltime_seconds": 11000},
        "benchmarks": benchmarks, "shards": shards})
    base = root/"claude14"
    base.mkdir()
    paths, groups, reports = {}, {}, {}
    for j in LAYOUT:
        C.run_shard(probe, j["cell"], j["task"], j["datasets"], j["folds"], base/j["id"], identity, resource)
        paths[j["id"]] = base/j["id"]/"shard.json"
        groups.setdefault((j["cell"], j["task"]), []).append(j["id"])
    for (cell, task), ids in groups.items():
        reports[(cell, task)] = base/(cell+"-"+task+".json")
        C.aggregate(probe, cell, task, [paths[i] for i in ids], reports[(cell, task)], identity)
    return {"identity": identity, "paths": paths, "groups": groups, "reports": reports, "base": base, "probe": probe}


@pytest.fixture(scope="module")
def results(scene, fourteen):
    single = M.compare(scene["inputs"], scene["root"]/"claude-single.json", CODE)
    inputs = copy.deepcopy(scene["inputs"])
    inputs["probe_reports"] = [pin(p) for p in fourteen["reports"].values()]
    multi = M.compare(inputs, scene["root"]/"claude-multi.json", CODE)
    return single, multi, inputs


def test_published_layout_covers_every_cell_dataset_fold_once():
    coverage = [(j["cell"], d, f) for j in LAYOUT for d in j["datasets"] for f in j["folds"]]
    assert len(LAYOUT) == 14 and len(coverage) == len(set(coverage)) == 60
    assert set(coverage) == {(c, d, f) for c in C.CELLS for d in C.DATASETS for f in range(4)}
    assert all(set(j["datasets"]) <= set(C.TASKS[j["task"]]) for j in LAYOUT)
    count = {}
    for j in LAYOUT:
        count[(j["cell"], j["task"])] = count.get((j["cell"], j["task"]), 0) + 1
    assert count == {("T100-G-30M-natural", "p1"): 2, ("T100-G-30M-natural", "p2"): 1, ("T100-G-300M-natural", "p1"): 4,
                     ("T100-G-300M-natural", "p2"): 4, ("T100-G-300M-matched", "p1"): 2, ("T100-G-300M-matched", "p2"): 1}


def test_aggregates_from_the_real_layout_equal_single_shard_aggregates(scene, fourteen):
    for (cell, task), target in fourteen["reports"].items():
        new = json.loads(target.read_text())
        old = json.loads((scene["root"]/(cell+task+"-aggregate.json")).read_text())
        assert [s["path"] for s in new["shards"]] == [str(fourteen["paths"][i]) for i in fourteen["groups"][(cell, task)]]
        assert all(s["sha256"] == C.sha(s["path"]) for s in new["shards"])
        assert new["audit_summary"]["fits"] == 28*len(C.TASKS[task]) == old["audit_summary"]["fits"]
        assert new["weights"] == old["weights"] and new["balanced_accuracy"] == old["balanced_accuracy"]
        assert new["reading_status"] == old["reading_status"] and new["numeric_sha256"] == old["numeric_sha256"]
        with np.load(new["oof"], allow_pickle=False) as a, np.load(old["oof"], allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            assert all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a.files)
        for ds in C.TASKS[task]:
            for key in ("balanced_accuracy", "n_oof_windows", "n_oof_subjects", "identity_sha256"):
                assert new["datasets"][ds][key] == old["datasets"][ds][key]
            for x, y in zip(new["datasets"][ds]["folds"], old["datasets"][ds]["folds"]):
                assert all(x[k] == y[k] for k in ("fold", "c", "val_bacc", "val_scores", "held_bacc", "n_train", "n_val",
                                                  "n_held", "roles_sha256", "scaler_sha256"))


def test_comparison_is_identical_with_the_real_fourteen_shard_layout(results):
    single, multi, _ = results
    assert multi["secondary"]["classification"] == single["secondary"]["classification"]
    for key in ("epn", "p1", "p2", "pose", "quality", "reading", "bootstrap", "scope"):
        assert multi[key] == single[key]
    assert multi["reading"]["continuation_authorized"] is False


def test_stdlib_reader_predicts_the_identity_written_by_aggregate(fourteen):
    value = {"shards": [[j["id"], j["cell"], j["task"], j["datasets"], j["folds"], str(fourteen["base"]/j["id"])] for j in LAYOUT],
             "subjects": {ds: v["subjects"] for ds, v in fourteen["probe"]["datasets"].items()}}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(OI.script(value), "remoto", "exec"), {"__name__": "remoto"})
    text = out.getvalue()
    assert "== problemi: nessuno" in text and "INCOERENTI" not in text and "NON CONCLUSO" not in text
    predicted = {tuple(line.split()[:2]): line.split()[-1] for line in text.splitlines() if "identity_sha256 atteso" in line}
    assert len(predicted) == 15
    for (cell, task), target in fourteen["reports"].items():
        report = json.loads(target.read_text())
        for ds in C.TASKS[task]:
            assert predicted[(cell, ds)] == report["datasets"][ds]["identity_sha256"]
    assert text.count("| identiche: True |") == 20 and "identiche: False" not in text
    assert text.count("identity_sha256 uguale fra le celle complete: True celle complete: 3") == 5


def refused(fourteen, cell, task, shards, name, match):
    target = fourteen["base"]/name
    with pytest.raises(ValueError, match=match):
        C.aggregate(fourteen["probe"], cell, task, shards, target, fourteen["identity"])
    assert not any(target.with_suffix(s).exists() for s in (".json", ".oof.npz", ".audit.jsonl"))


def test_aggregate_refuses_holes_duplicates_and_foreign_shards_in_the_real_layout(fourteen):
    p, g = fourteen["paths"], fourteen["groups"]
    natural = "T100-G-300M-natural"
    p1, p2 = [p[i] for i in g[(natural, "p1")]], [p[i] for i in g[(natural, "p2")]]
    refused(fourteen, natural, "p1", p1[:-1], "senza-fit14.json", "missing dataset/fold")
    refused(fourteen, natural, "p1", p1[1:], "senza-fit01.json", "missing dataset/fold")
    refused(fourteen, natural, "p2", p2[:2] + p2[3:], "senza-fold2.json", "missing dataset/fold")
    copia = fourteen["base"]/"copia-fit06"
    copia.mkdir()
    shutil.copy(p2[0], copia/"shard.json")
    refused(fourteen, natural, "p2", p2 + [copia/"shard.json"], "fold-doppio.json", "duplicate/invalid fold")
    refused(fourteen, natural, "p2", p2 + [p2[0]], "stesso-shard.json", "invalid aggregation")
    other = p[g[("T100-G-30M-natural", "p2")][0]]
    refused(fourteen, natural, "p2", p2[:3] + [other], "altra-cella.json", "shard identity mismatch")
    refused(fourteen, natural, "p1", p1 + [p2[0]], "altro-compito.json", "shard identity mismatch")
    refused(fourteen, natural, "p1", p1 + [p[g[("T100-G-300M-matched", "p1")][0]]], "altra-vista.json", "shard identity mismatch")


def test_shard_order_does_not_change_the_comparison(scene, fourteen, results):
    cell, task = "T100-G-300M-natural", "p1"
    target = fourteen["base"]/"ordine-inverso.json"
    C.aggregate(fourteen["probe"], cell, task, [fourteen["paths"][i] for i in fourteen["groups"][(cell, task)][::-1]], target, fourteen["identity"])
    inputs = copy.deepcopy(results[2])
    inputs["probe_reports"] = [pin(target) if q["path"] == str(fourteen["reports"][(cell, task)]) else q for q in inputs["probe_reports"]]
    assert sum(q["path"] == str(target) for q in inputs["probe_reports"]) == 1
    again = M.compare(inputs, scene["root"]/"claude-ordine-inverso.json", CODE)
    assert again["secondary"]["classification"] == results[1]["secondary"]["classification"]


def replaced(results, fourteen, key, report):
    inputs = copy.deepcopy(results[2])
    inputs["probe_reports"] = [pin(report) if q["path"] == str(fourteen["reports"][key]) else q for q in inputs["probe_reports"]]
    assert sum(q["path"] == str(report) for q in inputs["probe_reports"]) == 1
    return inputs


def test_comparator_rereads_every_source_shard(scene, fourteen, results):
    key = ("T100-G-300M-natural", "p2")
    original = json.loads(fourteen["reports"][key].read_text())
    # 1. Un aggregato coerente con se stesso che tace uno dei suoi shard.
    silent = copy.deepcopy(original)
    silent["shards"] = silent["shards"][:-1]
    save_json(fourteen["base"]/"tace-uno-shard.json", silent)
    with pytest.raises(ValueError, match="aggregate audit differs from source shards"):
        M.compare(replaced(results, fourteen, key, fourteen["base"]/"tace-uno-shard.json"), scene["root"]/"claude-tace.json", CODE)
    # 2. Uno shard con una predizione diversa, hash interni rifatti e pin dell'aggregato riallineato.
    source = Path(original["shards"][0]["path"]).parent
    fake = fourteen["base"]/"shard-alterato"
    shutil.copytree(source, fake)
    with np.load(fake/"shard.oof.npz", allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    values["epn612_pred"][0] = "fist" if values["epn612_pred"][0] != "fist" else "noGesture"
    (fake/"shard.oof.npz").unlink()
    np.savez(fake/"shard.oof.npz", **values)
    report = json.loads((fake/"shard.json").read_text())
    report.update(oof=str(fake/"shard.oof.npz"), audit=str(fake/"audit.jsonl"), oof_sha256=C.sha(fake/"shard.oof.npz"))
    (fake/"shard.json").unlink()
    save_json(fake/"shard.json", report)
    altered = copy.deepcopy(original)
    altered["shards"][0] = pin(fake/"shard.json")
    save_json(fourteen["base"]/"punta-a-shard-alterato.json", altered)
    with pytest.raises(ValueError, match="aggregate OOF differs from source shards"):
        M.compare(replaced(results, fourteen, key, fourteen["base"]/"punta-a-shard-alterato.json"), scene["root"]/"claude-alterato.json", CODE)
    # 3. Lo stesso shard alterato senza riallineare il pin: lo ferma l'hash registrato nell'aggregato.
    shutil.copy(fake/"shard.json", fourteen["base"]/"shard-sostituito.json")
    swapped = copy.deepcopy(original)
    swapped["shards"][0] = {"path": str(fourteen["base"]/"shard-sostituito.json"), "sha256": original["shards"][0]["sha256"]}
    save_json(fourteen["base"]/"pin-non-riallineato.json", swapped)
    with pytest.raises(ValueError):
        M.compare(replaced(results, fourteen, key, fourteen["base"]/"pin-non-riallineato.json"), scene["root"]/"claude-pin.json", CODE)
    assert not any((scene["root"]/n).exists() for n in ("claude-tace.json", "claude-alterato.json", "claude-pin.json"))


def independent(cells, n_boot=2000, seed=20261102, explicit=12):
    """Stesso stimatore per un'altra via: matrici di conteggio con add.at, matrice delle molteplicita' per replica."""
    rng = np.random.default_rng(seed)
    points, replicates, hashes = {}, {}, {}
    for ds in sorted(d for w in WEIGHTS.values() for d in w):
        base = cells[NAMES[0]][ds]
        first = np.argsort(base["row"])
        y, s = base["y"][first], base["s"][first].astype(str)
        users, ui = np.unique(s, return_inverse=True)
        classes, yi = np.unique(y, return_inverse=True)
        total = np.zeros((len(users), len(classes)))
        np.add.at(total, (ui, yi), 1)
        draw = rng.integers(0, len(users), size=(n_boot, len(users)))
        hashes[ds] = hashlib.sha256(json.dumps([draw.dtype.str, draw.shape]).encode() + draw.tobytes() + b"\0").hexdigest()
        mult = np.zeros((n_boot, len(users)))
        np.add.at(mult, (np.repeat(np.arange(n_boot), len(users)), draw.ravel()), 1)
        den = mult @ total
        pts, reps = [], []
        members = [np.flatnonzero(ui == u) for u in range(len(users))]
        for name in NAMES:
            a = cells[name][ds]
            order = np.argsort(a["row"])
            assert np.array_equal(a["row"][order], base["row"][first]) and np.array_equal(a["y"][order], y)
            assert np.array_equal(a["s"][order].astype(str), s) and np.array_equal(a["fold"][order], base["fold"][first])
            pred = a["pred"][order]
            hit = np.zeros_like(total)
            np.add.at(hit, (ui, yi), (pred == y).astype(float))
            pts.append(balanced_accuracy_score(y, pred))
            with np.errstate(invalid="ignore", divide="ignore"):
                recall = np.where(den > 0, (mult @ hit)/den, np.nan)
            reps.append(np.nanmean(recall, axis=1))
            for b in range(explicit):  # ricampionamento esplicito delle finestre degli utenti estratti
                ix = np.concatenate([members[u] for u in draw[b]])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    assert abs(balanced_accuracy_score(y[ix], pred[ix]) - reps[-1][b]) < 1e-12
        points[ds], replicates[ds] = np.asarray(pts), np.column_stack(reps)
    return points, replicates, hashes


def assert_same_statistics(out, points, replicates, hashes):
    def check(block, pts, reps):
        for i, name in enumerate(NAMES):
            assert abs(block["cells"][name]["balanced_accuracy"] - pts[i]) < 1e-12
            assert np.allclose(block["cells"][name]["ci95_percent"], 100*np.quantile(reps[:, i], [.025, .975]), rtol=0, atol=1e-9)
        for name, (plus, minus) in CONTRASTS.items():
            assert abs(block["contrasts"][name]["delta_pp"] - 100*(pts[plus]-pts[minus])) < 1e-9
            assert np.allclose(block["contrasts"][name]["ci95_pp"], 100*np.quantile(reps[:, plus]-reps[:, minus], [.025, .975]), rtol=0, atol=1e-9)
    assert out["n_bootstrap"] == 2000 and out["seed"] == 20261102 and set(out["datasets"]) == set(points)
    for ds in points:
        check(out["datasets"][ds], points[ds], replicates[ds])
        assert out["datasets"][ds]["draws_sha256"] == hashes[ds]
        assert out["datasets"][ds]["role"] == ("primary_classification" if ds == "epn612" else "secondary_dataset")
    for task, weights in WEIGHTS.items():
        check(out["tasks"][task], sum(w*points[d] for d, w in weights.items()), sum(w*replicates[d] for d, w in weights.items()))


def oof_cells(reports):
    cells = {name: {} for name in NAMES}
    for (cell, task), target in reports.items():
        report = json.loads(Path(target).read_text())
        with np.load(report["oof"], allow_pickle=False) as z:
            for ds in C.TASKS[task]:
                cells[cell][ds] = {k: z[f"{ds}_{k}"] for k in ("row", "y", "pred", "s", "fold")}
    return cells


def test_final_statistics_recomputed_independently_from_the_aggregates(fourteen, results):
    assert_same_statistics(results[1]["secondary"]["classification"], *independent(oof_cells(fourteen["reports"]), explicit=2000))


# Dimensioni reali lette su Leonardo il 06/11/2026: utenti, classi, finestre OOF, etichette testuali.
REAL = {"epn612": (314, 6, 47100, True), "ninapro_db2": (32, 49, 38544, False), "ninapro_db3": (8, 49, 9834, False),
        "ninapro_db6": (8, 7, 33961, False), "uci_emg": (19, 6, 364, False)}
GESTURES = np.asarray(["fist", "noGesture", "open", "pinch", "waveIn", "waveOut"])


def real_shaped(accuracy, seed=7):
    rng = np.random.default_rng(seed)
    cells = {name: {} for name in NAMES}
    for ds, (n_users, n_classes, n, text) in REAL.items():
        user = np.sort(rng.integers(0, n_users, n))
        user[:n_users] = np.arange(n_users)
        user = np.sort(user)
        code = rng.integers(0, n_classes, n)
        subjects = np.asarray([("user_%03d_%s" % (u, "x"*11)) if text else "s%02d" % u for u in user])
        skill = rng.normal(0, .08, n_users)[user]
        shared = rng.random(n)
        for name in NAMES:
            wrong = shared > np.clip(accuracy[name] + skill, .02, .98)
            pred = np.where(wrong, (code + 1 + rng.integers(0, n_classes - 1, n)) % n_classes, code)
            y, p = (GESTURES[code], GESTURES[pred]) if text else (code, pred)
            order = rng.permutation(n) if name == NAMES[2] else np.arange(n)  # una cella in ordine diverso: l'allineamento e' per riga
            cells[name][ds] = {"row": (3*np.arange(n) + 5)[order], "y": y[order], "pred": p[order], "s": subjects[order], "fold": (user % 4)[order]}
    return cells


def summary_for(scene, classification, converged=True):
    report = scene["pose_report"]
    probes = [{"audit_summary": {"converged_by_criterion": converged or i > 0}} for i in range(6)]
    return M.compose_summary(classification, report["paired"], probes, report["cells"])


def test_real_sizes_text_labels_signs_and_timing(scene):
    cells = real_shaped({NAMES[0]: .60, NAMES[1]: .70, NAMES[2]: .65})
    assert cells[NAMES[0]]["epn612"]["y"].dtype.str == "<U9" and cells[NAMES[0]]["epn612"]["s"].dtype.str == "<U20"
    began = time.perf_counter()
    out = C.compare_arrays(cells)
    elapsed = time.perf_counter() - began
    print("\ncompare_arrays sulle dimensioni reali (un thread, Mac): %.1f s" % elapsed)
    assert elapsed < 600
    assert_same_statistics(out, *independent(cells))
    for block in [out["datasets"][ds] for ds in REAL if ds != "uci_emg"] + list(out["tasks"].values()):
        natural, matched, between = (block["contrasts"][k] for k in CONTRASTS)
        assert natural["delta_pp"] > matched["delta_pp"] > 0 and between["delta_pp"] > 0 and natural["ci95_pp"][0] > 0
    good = summary_for(scene, out)
    epn = out["datasets"]["epn612"]["contrasts"]["natural_300m_minus_30m"]
    assert good["epn"] == {"delta_pp": epn["delta_pp"], "ci95": epn["ci95_pp"]}
    assert all(good[t] == {"delta_pp": out["tasks"][t]["contrasts"]["natural_300m_minus_30m"]["delta_pp"],
                           "ci95": out["tasks"][t]["contrasts"]["natural_300m_minus_30m"]["ci95_pp"]} for t in WEIGHTS)
    assert good["reading"]["reading_valid"] is True and good["reading"]["primary_reading"]["epn"] == "better"
    assert good["reading"]["classification_noninferiority"] == {"p1": "supported", "p2": "supported"}
    assert good["reading"]["continuation_authorized"] is False
    # Convergenza non risolta: i numeri restano, la lettura e la raccomandazione no.
    blocked = summary_for(scene, out, converged=False)
    assert blocked["quality"]["solver_converged"] is False and blocked["reading"]["reading_valid"] is False
    assert blocked["reading"]["recommendation"] == "repair_evaluation_no_scientific_reading"
    assert blocked["reading"]["failed_checks"] == ["solver_converged"] and "primary_reading" not in blocked["reading"]
    assert all(blocked[k] == good[k] for k in ("epn", "p1", "p2", "pose", "secondary"))
    # Segno: se il 300M naturale e' peggiore, il contrasto e' negativo e la lettura lo dice.
    worse = C.compare_arrays(real_shaped({NAMES[0]: .70, NAMES[1]: .60, NAMES[2]: .65}))
    reading = summary_for(scene, worse)["reading"]
    assert worse["datasets"]["epn612"]["contrasts"]["natural_300m_minus_30m"]["ci95_pp"][1] < 0
    assert reading["primary_reading"]["epn"] == "worse" and reading["recommendation"] == "pause_this_recipe_review_before_19618"
    assert reading["classification_noninferiority"] == {"p1": "loss_beyond_margin", "p2": "loss_beyond_margin"}
