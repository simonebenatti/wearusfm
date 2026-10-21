"""Confronto dei bracci dell'obiettivo (`scripts/arms_compare.py`) con le regole firmate il 16/10/2026: regola del 2A/2C (accoppiato per seme,
+1 punto, limite inferiore > 0, positivo in entrambi i semi), default scelto su S della P1 e della P2, P1/P2 ufficiali come lettura scientifica,
braccio collassato che non batte il riferimento, piu' vincitori lasciati a Simone, disaccordi registrati."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wearusfm.harness import downstream_ab as DA

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("arms_compare", REPO / "scripts" / "arms_compare.py")
AC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AC)

P1_DS = {"ninapro_db2": 8 / 13, "ninapro_db3": 3 / 13, "ninapro_db6": 2 / 13}
P2_DS = {"epn612": 0.5, "uci_emg": 0.5}
N_SUB = 24


def _preds(y, quality, rng):
    return np.where(rng.random(len(y)) < quality, y, (y % 4) + 1)


def _data():
    y = np.tile(np.arange(1, 5), 10 * N_SUB)
    s = np.repeat([f"s{i:02d}" for i in range(N_SUB)], 40)
    return y, s


def _s_report(d, name, task, quality, seed):
    """Rapporto di calibration_probe.py con predizioni fuori fold a accuratezza `quality`."""
    from wearusfm.data import pretraining_manifest  # noqa: F401 -- stesso ambiente dei rapporti veri

    rng = np.random.default_rng(seed)
    y, s = _data()
    weights = P1_DS if task == "p1" else P2_DS
    from wearusfm.harness.downstream_ab import fingerprint

    oof, rows = {}, np.arange(len(y), dtype=np.int64)
    for ds in weights:
        oof.update({f"{ds}_y": y, f"{ds}_pred": _preds(y, quality, rng), f"{ds}_s": s, f"{ds}_row": rows})
    path = d / f"{name}_s_{task}.oof.npz"
    np.savez(path, datasets=np.asarray(list(weights)), **oof)
    sel = AC._select()
    # campi di provenienza dei rapporti veri (review GPT 41ce569: accoppiamento controllato prima di ogni confronto)
    rep = {"protocol": sel.PROTOCOL, "task": task, "weights": weights, "oof": str(path), "oof_sha256": sel._sha256(path),
           "splits_sha256": "a" * 64 if task == "p1" else "b" * 64, "k_folds": 4, "fold_seed": 20261015, "val_frac": 0.1, "probe_seed": 0,
           "feature_set": "model", "window_s": 1.0 if task == "p1" else None, "extraction_recipe": {"task": task},
           "datasets": {ds: {"input_sha256": "i" * 64, "oof_rows_sha256": fingerprint(rows)} for ds in weights}}
    p = d / f"{name}_s_{task}.json"
    p.write_text(json.dumps(rep))
    return str(p)


def _counts_report(y, pred, s):
    c = DA.subject_counts(y, pred, s)
    _, n, hits = DA._arrays(c)
    return DA.bacc(n, hits), c


def _official(d, name, quality, seed):
    rng = np.random.default_rng(seed + 100)
    y, s = _data()
    p1 = {"datasets": {}}
    for ds in P1_DS:
        b, c = _counts_report(y, _preds(y, quality, rng), s)
        p1["datasets"][ds] = {"test_bacc": b, "paired_counts": c}
    p1["p1"] = float(sum(P1_DS[ds] * p1["datasets"][ds]["test_bacc"] for ds in P1_DS))
    p2 = {"datasets": {}}
    for ds in P2_DS:
        splits = []
        for k in (0, 1, 2):
            b, c = _counts_report(y, _preds(y, quality, rng), s)
            splits.append({"split_seed": k, "test_bacc": b, "paired_counts": c})
        p2["datasets"][ds] = {"splits": splits, "test_bacc": float(np.mean([v["test_bacc"] for v in splits]))}
    p2["p2"] = float(np.mean([p2["datasets"][ds]["test_bacc"] for ds in P2_DS]))
    out = {}
    for t, rep in (("p1", p1), ("p2", p2)):
        path = d / f"{name}_{t}.json"
        path.write_text(json.dumps(rep))
        out[t] = str(path)
    return out


def _run(d, name, stop=None):
    rd = d / name
    rd.mkdir()
    (rd / "summary.json").write_text(json.dumps({"stopped": "max_steps" if stop is None else "allarme", "steps": 10}))
    if stop is not None:
        (rd / "STOP").write_text(stop)
    return str(rd)


def _arm(d, name, s_quality=(0.6, 0.6), off_quality=(0.6, 0.6), stop=None):
    """Due semi di un braccio: qualita' di S e dei test ufficiali per seme."""
    out = {}
    for i, sd in enumerate(("0", "1")):
        entry = {"run_dir": _run(d, f"{name}_s{sd}", stop)}
        if stop is None:
            entry.update({"s_p1": _s_report(d, f"{name}{sd}", "p1", s_quality[i], 10 * i + sum(map(ord, name)) % 7),
                          "s_p2": _s_report(d, f"{name}{sd}", "p2", s_quality[i], 10 * i + 1 + sum(map(ord, name)) % 7),
                          **_official(d, f"{name}{sd}", off_quality[i], 10 * i + 2 + sum(map(ord, name)) % 7)})
        out[sd] = entry
    return out


def _spec(d, arms):
    runs = {"A": _arm(d, "A")}
    runs.update(arms)
    return {"reference": "A", "arms": list(arms), "runs": runs}


def test_an_arm_that_is_clearly_better_becomes_the_default(tmp_path):
    res = AC.compare(_spec(tmp_path, {"JEPA-only": _arm(tmp_path, "JEPA-only", (0.8, 0.8), (0.8, 0.8))}))
    arm = res["arms"]["JEPA-only"]
    assert arm["S"]["p1"]["beats"] and arm["S"]["p2"]["beats"] and arm["official"]["beats"]
    assert res["default"]["choice"] == "JEPA-only" and res["disagreements"] == []


def test_equal_or_one_seed_only_does_not_beat(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    res = AC.compare(_spec(a, {"ancore-only": _arm(a, "ancore-only")}))
    assert not res["arms"]["ancore-only"]["S"]["beats"] and res["default"]["choice"] == "A"
    res = AC.compare(_spec(b, {"ancore-only": _arm(b, "ancore-only", (0.9, 0.5), (0.9, 0.5))}))  # meglio in un seme, peggio nell'altro
    row = res["arms"]["ancore-only"]["S"]["p1"]
    assert row["delta_by_model_seed"][0] > 0 > row["delta_by_model_seed"][1] and not row["beats"]
    assert res["default"]["choice"] == "A"


def test_collapsed_arm_does_not_beat_and_failed_runs_are_not_evaluable(tmp_path):
    res = AC.compare(_spec(tmp_path, {"JEPA-only": _arm(tmp_path, "JEPA-only", stop="allarme: collasso da query")}))
    assert res["arms"]["JEPA-only"]["status"] == "collassato" and not res["arms"]["JEPA-only"]["S"]["beats"]
    assert res["default"]["choice"] == "A"
    x = tmp_path / "x"
    x.mkdir()
    with pytest.raises(ValueError, match="non valutabile"):
        AC.compare(_spec(x, {"ancore-only": _arm(x, "ancore-only", stop="errore numerico al passo 7")}))
    y = tmp_path / "y"
    y.mkdir()
    spec = _spec(y, {"ancore-only": _arm(y, "ancore-only")})
    spec["runs"]["A"] = _arm(y, "A2", stop="allarme: rango")
    with pytest.raises(ValueError, match="riferimento A non e' completo"):
        AC.compare(spec)


def test_two_winners_are_left_to_simone_and_disagreements_are_recorded(tmp_path):
    res = AC.compare(_spec(tmp_path, {"JEPA-only": _arm(tmp_path, "JEPA-only", (0.8, 0.8), (0.8, 0.8)),
                                      "ancore-only": _arm(tmp_path, "ancore-only", (0.8, 0.8), (0.6, 0.6))}))
    assert res["default"]["choice"] is None and sorted(res["default"]["candidates"]) == ["JEPA-only", "ancore-only"]
    assert res["disagreements"] == ["ancore-only"]  # vince su S, non sui test ufficiali: il default segue S, il disaccordo si registra
    assert not res["arms"]["ancore-only"]["official"]["beats"] and res["arms"]["ancore-only"]["official"]["not_harmful"]


def test_unpaired_predictions_are_refused(tmp_path):
    spec = _spec(tmp_path, {"JEPA-only": _arm(tmp_path, "JEPA-only")})
    rep = json.loads(Path(spec["runs"]["JEPA-only"]["0"]["s_p1"]).read_text())
    with np.load(rep["oof"]) as z:
        d = {k: z[k] for k in z.files}
    d["ninapro_db2_s"] = np.roll(d["ninapro_db2_s"], 1)
    np.savez(rep["oof"], **d)
    rep["oof_sha256"] = AC._select()._sha256(Path(rep["oof"]))
    Path(spec["runs"]["JEPA-only"]["0"]["s_p1"]).write_text(json.dumps(rep))
    with pytest.raises(ValueError, match="non e' accoppiato"):
        AC.compare(spec)


def test_cli_writes_the_decision(tmp_path):
    spec = _spec(tmp_path, {"JEPA-only": _arm(tmp_path, "JEPA-only")})
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    assert AC.main([str(p), "--out", str(tmp_path / "out" / "esito.json")]) == 0
    assert json.loads((tmp_path / "out" / "esito.json").read_text())["default"]["choice"] == "A"


def test_s_reports_with_different_provenance_are_refused(tmp_path):
    """Merge con la review GPT (41ce569): rapporti di S con fold, split o righe diversi non si confrontano, nemmeno con OOF uguali."""
    spec = _spec(tmp_path, {"JEPA-only": _arm(tmp_path, "JEPA-only")})
    path = Path(spec["runs"]["JEPA-only"]["0"]["s_p1"])
    rep = json.loads(path.read_text())
    path.write_text(json.dumps({**rep, "fold_seed": 1}))
    with pytest.raises(ValueError, match="provenienza non accoppiata"):
        AC.compare(spec)
