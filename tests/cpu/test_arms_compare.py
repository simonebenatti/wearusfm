"""Confronto dei bracci dell'obiettivo (`scripts/arms_compare.py`) con le regole firmate il 16/10/2026: regola del 2A/2C (accoppiato per seme,
+1 punto, limite inferiore > 0, positivo in entrambi i semi), default scelto su S della P1 e della P2, P1/P2 ufficiali come lettura scientifica,
braccio collassato che non batte il riferimento, piu' vincitori lasciati a Simone, disaccordi registrati. Identita' dei run e provenienza dei
rapporti verificate prima di ogni confronto (rilievo 1 del critico, 21/10): run e configurazioni veri del piano di `arms_plan.py`."""

import importlib.util
import json
import shlex
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wearusfm.harness import downstream_ab as DA

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AC, AP, W, TJ = _load("arms_compare"), _load("arms_plan"), _load("wsd_chain"), _load("train_jepa")

P1_DS = {"ninapro_db2": 8 / 13, "ninapro_db3": 3 / 13, "ninapro_db6": 2 / 13}
P2_DS = {"epn612": 0.5, "uci_emg": 0.5}
N_SUB = 24
SHA = "a" * 40
_TMP = Path(tempfile.mkdtemp())
WS = _TMP / "ws.json"
WS.write_text(json.dumps({"window_s": {}, "scales": {}}))
REPORT = {"fractions": {"1": {"windows_per_epoch": 784756.9792373349, "file_sha256": "6d6d" + "0" * 60, "file": "m_nested_1.json.gz"}},
          "code": {"source": "launcher (lib_code_dir.sh)"}}


def _camp(root):
    return {"campaign": "w1arm", "agent": "claude", "code_dir": "/C", "expected_sha": SHA, "manifest": "/M/m_nested_1.json.gz",
            "scales": str(WS), "window_seconds": str(WS), "run_root": str(root / "runs"), "slurm": {"qos": "normal", "time": "24:00:00"},
            "margin_min": 40, "startup_min": 15, "slack": 1.15, "spare_jobs": 0, "gpus": 4,
            "eval": {"features_root": "/F", "results_root": "/R", "raw_root": "/raw"}}


class Env:
    """Piano di training dei bracci (arms_plan) scritto in `tmp`, con la base della spec che arms_plan produce."""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        res = AP.plan(REPORT, 1e-4, _camp(self.tmp), plans_dir="results/infra/w1arm")
        self.plan = res["plans"]["objective"]
        path = self.tmp / "bracci_objective.json"
        path.write_text(AP.plan_text(self.plan))
        base = res["specs"]["objective"]
        self.base = {**{k: v for k, v in base.items() if k != "runs"}, "plan": str(path)}  # stesso testo: stesso SHA della spec
        self.steps = self.plan["runs"][0]["total_steps"]


def _config(plan, run_id):
    """`config.json` come la scrive il trainer: `train_jepa.main` con gli argomenti del job (`train_ddp.sbatch` e `wsd_chain.extra_args`)."""
    run = next(r for r in W.resolve(plan) if r["id"] == run_id)
    argv = ["--preset", run["preset"], "--manifest", "m.json.gz", "--root", "r", "--scales", str(WS), "--out-dir", str(_TMP / "out"),
            "--max-steps", str(run["total_steps"])] + shlex.split(W.extra_args(plan, run))
    captured = {}
    orig = TJ.R.train, TJ.R.check_input_files
    try:
        TJ.R.train = lambda cfg, *a, **k: captured.__setitem__("cfg", cfg) or {"steps": 0}
        TJ.R.check_input_files = lambda *a, **k: None
        TJ.main(argv)
    finally:
        TJ.R.train, TJ.R.check_input_files = orig
    return json.dumps(TJ.R.config_to_dict(captured["cfg"]))


def _preds(y, quality, rng):
    return np.where(rng.random(len(y)) < quality, y, (y % 4) + 1)


def _data():
    y = np.tile(np.arange(1, 5), 10 * N_SUB)
    s = np.repeat([f"s{i:02d}" for i in range(N_SUB)], 40)
    return y, s


def _s_report(d, name, task, quality, seed, ckpt):
    """Rapporto di calibration_probe.py con predizioni fuori fold a accuratezza `quality`, dal checkpoint `ckpt`."""
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
           "checkpoint": str(ckpt), "checkpoint_sha256": sel._sha256(ckpt),
           "datasets": {ds: {"input_sha256": "i" * 64, "oof_rows_sha256": fingerprint(rows)} for ds in weights}}
    p = d / f"{name}_s_{task}.json"
    p.write_text(json.dumps(rep))
    return str(p)


def _counts_report(y, pred, s):
    c = DA.subject_counts(y, pred, s)
    _, n, hits = DA._arrays(c)
    return DA.bacc(n, hits), c


def _official(d, name, quality, seed, ckpt):
    """Rapporti di probe_p1.py e probe_p2.py; `quality` = (P1, P2)."""
    rng = np.random.default_rng(seed + 100)
    y, s = _data()
    p1 = {"checkpoint": str(ckpt), "which": "teacher", "datasets": {}}
    for ds in P1_DS:
        b, c = _counts_report(y, _preds(y, quality[0], rng), s)
        p1["datasets"][ds] = {"test_bacc": b, "paired_counts": c}
    p1["p1"] = float(sum(P1_DS[ds] * p1["datasets"][ds]["test_bacc"] for ds in P1_DS))
    p2 = {"checkpoint": str(ckpt), "which": "teacher", "datasets": {}}
    for ds in P2_DS:
        splits = []
        for k in (0, 1, 2):
            b, c = _counts_report(y, _preds(y, quality[1], rng), s)
            splits.append({"split_seed": k, "test_bacc": b, "paired_counts": c})
        p2["datasets"][ds] = {"splits": splits, "test_bacc": float(np.mean([v["test_bacc"] for v in splits]))}
    p2["p2"] = float(np.mean([p2["datasets"][ds]["test_bacc"] for ds in P2_DS]))
    out = {}
    for t, rep in (("p1", p1), ("p2", p2)):
        path = d / f"{name}_{t}.json"
        path.write_text(json.dumps(rep))
        out[t] = str(path)
    return out


def _run(env, name, sd, stop=None, config_of=None):
    run_id = f"W1ARM-{name}-30M-s{sd}"
    rd = env.tmp / "runs" / run_id
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(_config(env.plan, config_of or run_id))
    (rd / "summary.json").write_text(json.dumps({"stopped": "max_steps" if stop is None else "allarme",
                                                 "steps": env.steps if stop is None else 10, "code": {"sha": SHA}}))
    if stop is not None:
        (rd / "STOP").write_text(stop)
    (rd / AC.CHECKPOINT).write_bytes(f"checkpoint di {run_id}".encode())
    return run_id, rd


def _arm(env, name, s_quality=(0.6, 0.6), off_quality=(0.6, 0.6), stop=None, p2_quality=None):
    """Due semi di un braccio: qualita' di S e dei test ufficiali per seme; `p2_quality` = (S, ufficiale) della P2 se diversa dalla P1."""
    out = {}
    reps = env.tmp / "reports"
    reps.mkdir(exist_ok=True)
    for i, sd in enumerate(("0", "1")):
        run_id, rd = _run(env, name, sd, stop)
        entry = {"id": run_id, "run_dir": str(rd)}
        if stop is None:
            ck = rd / AC.CHECKPOINT
            sq2, oq2 = (s_quality[i], off_quality[i]) if p2_quality is None else (p2_quality[0][i], p2_quality[1][i])
            k = sum(map(ord, name)) % 7
            entry.update({"s_p1": _s_report(reps, f"{name}{sd}", "p1", s_quality[i], 10 * i + k, ck),
                          "s_p2": _s_report(reps, f"{name}{sd}", "p2", sq2, 10 * i + 1 + k, ck),
                          **_official(reps, f"{name}{sd}", (off_quality[i], oq2), 10 * i + 2 + k, ck)})
        out[sd] = entry
    return out


ARMS = ("JEPA-only", "ancore-only")


def _spec(env, arms, ref=None):
    """Spec del gruppo dell'obiettivo: i bracci non dati sono neutri (stessa qualita' di A), perche' la spec deve coprirli tutti."""
    runs = {"A": ref or _arm(env, "A"), **{a: arms[a] if a in arms else _arm(env, a) for a in ARMS}}
    return {**env.base, "arms": list(ARMS), "runs": runs}


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def test_an_arm_that_is_clearly_better_becomes_the_default(env):
    res = AC.compare(_spec(env, {"JEPA-only": _arm(env, "JEPA-only", (0.8, 0.8), (0.8, 0.8))}))
    arm = res["arms"]["JEPA-only"]
    assert arm["S"]["p1"]["beats"] and arm["S"]["p2"]["beats"] and arm["official"]["beats"]
    assert res["default"]["choice"] == "JEPA-only" and res["disagreements"] == []
    # identita' verificata e registrata per ogni run: configurazione completa, codice, passi e checkpoint
    assert set(res["identity"]) == {f"{a}/{s}" for a in ("A", *ARMS) for s in "01"}
    assert all(v["config_checked"] and v["steps"] == env.steps and len(v["checkpoint_sha256"]) == 64 for v in res["identity"].values())
    assert res["plan"] == {"path": env.base["plan"], "sha256": env.base["plan_sha256"], "lr": 1e-4}


def test_equal_or_one_seed_only_does_not_beat(tmp_path):
    a, b = Env(tmp_path / "a"), Env(tmp_path / "b")
    res = AC.compare(_spec(a, {"ancore-only": _arm(a, "ancore-only")}))
    assert not res["arms"]["ancore-only"]["S"]["beats"] and res["default"]["choice"] == "A"
    res = AC.compare(_spec(b, {"ancore-only": _arm(b, "ancore-only", (0.9, 0.5), (0.9, 0.5))}))  # meglio in un seme, peggio nell'altro
    row = res["arms"]["ancore-only"]["S"]["p1"]
    assert row["delta_by_model_seed"][0] > 0 > row["delta_by_model_seed"][1] and not row["beats"]
    assert res["default"]["choice"] == "A"


def test_winning_p1_but_not_p2_does_not_beat(env):
    """«P1 E P2» (rilievo 3 del critico): meglio sulla P1, uguale sulla P2, sia su S sia sui test ufficiali."""
    arm = _arm(env, "JEPA-only", (0.8, 0.8), (0.8, 0.8), p2_quality=((0.6, 0.6), (0.6, 0.6)))
    res = AC.compare(_spec(env, {"JEPA-only": arm}))
    r = res["arms"]["JEPA-only"]
    assert r["S"]["p1"]["beats"] and not r["S"]["p2"]["beats"] and not r["S"]["beats"]
    assert r["official"]["tasks"]["p1"]["beats"] and not r["official"]["tasks"]["p2"]["beats"] and not r["official"]["beats"]
    assert res["default"]["choice"] == "A"


def test_verdict_needs_the_lower_ci_bound_above_zero():
    """Caso limite della regola (rilievo 3): +1 punto e positivo nei due semi non bastano se il limite inferiore e' <= 0."""
    ok = {"delta": 0.02, "ci95": [0.001, 0.04], "delta_by_model_seed": [0.02, 0.02]}
    assert AC.verdict(ok)["beats"]
    assert not AC.verdict({**ok, "ci95": [0.0, 0.04]})["beats"] and not AC.verdict({**ok, "ci95": [-0.001, 0.04]})["beats"]
    assert not AC.verdict({**ok, "delta": 0.0099})["beats"] and not AC.verdict({**ok, "delta_by_model_seed": [0.03, 0.0]})["beats"]
    assert AC.verdict({**ok, "delta": -0.01})["not_harmful"] and not AC.verdict({**ok, "delta": -0.0101})["not_harmful"]


def test_collapsed_arm_does_not_beat_and_failed_runs_are_not_evaluable(tmp_path):
    e = Env(tmp_path / "e")
    res = AC.compare(_spec(e, {"JEPA-only": _arm(e, "JEPA-only", stop="allarme: collasso da query")}))
    assert res["arms"]["JEPA-only"]["status"] == "collassato" and not res["arms"]["JEPA-only"]["S"]["beats"]
    assert res["default"]["choice"] == "A" and res["identity"]["JEPA-only/0"]["config_checked"]  # anche un collassato e' il braccio pianificato
    x = Env(tmp_path / "x")
    with pytest.raises(ValueError, match="non valutabile"):
        AC.compare(_spec(x, {"ancore-only": _arm(x, "ancore-only", stop="errore numerico al passo 7")}))
    y = Env(tmp_path / "y")
    spec = _spec(y, {"ancore-only": _arm(y, "ancore-only")}, ref=_arm(y, "A", stop="allarme: rango"))
    with pytest.raises(ValueError, match="riferimento A non e' completo"):
        AC.compare(spec)


def test_two_winners_are_left_to_simone_and_disagreements_are_recorded(env):
    res = AC.compare(_spec(env, {"JEPA-only": _arm(env, "JEPA-only", (0.8, 0.8), (0.8, 0.8)),
                                 "ancore-only": _arm(env, "ancore-only", (0.8, 0.8), (0.6, 0.6))}))
    assert res["default"]["choice"] is None and sorted(res["default"]["candidates"]) == ["JEPA-only", "ancore-only"]
    assert res["disagreements"] == ["ancore-only"]  # vince su S, non sui test ufficiali: il default segue S, il disaccordo si registra
    assert not res["arms"]["ancore-only"]["official"]["beats"] and res["arms"]["ancore-only"]["official"]["not_harmful"]


def test_unpaired_predictions_are_refused(env):
    spec = _spec(env, {"JEPA-only": _arm(env, "JEPA-only")})
    rep = json.loads(Path(spec["runs"]["JEPA-only"]["0"]["s_p1"]).read_text())
    with np.load(rep["oof"]) as z:
        d = {k: z[k] for k in z.files}
    d["ninapro_db2_s"] = np.roll(d["ninapro_db2_s"], 1)
    np.savez(rep["oof"], **d)
    rep["oof_sha256"] = AC._select()._sha256(Path(rep["oof"]))
    Path(spec["runs"]["JEPA-only"]["0"]["s_p1"]).write_text(json.dumps(rep))
    with pytest.raises(ValueError, match="non e' accoppiato"):
        AC.compare(spec)


def test_cli_writes_the_decision(env):
    spec = _spec(env, {"JEPA-only": _arm(env, "JEPA-only")})
    p = env.tmp / "spec.json"
    p.write_text(json.dumps(spec))
    assert AC.main([str(p), "--out", str(env.tmp / "out" / "esito.json")]) == 0
    assert json.loads((env.tmp / "out" / "esito.json").read_text())["default"]["choice"] == "A"


def test_s_reports_with_different_provenance_are_refused(env):
    """Merge con la review GPT (41ce569): rapporti di S con fold, split o righe diversi non si confrontano, nemmeno con OOF uguali."""
    spec = _spec(env, {"JEPA-only": _arm(env, "JEPA-only")})
    path = Path(spec["runs"]["JEPA-only"]["0"]["s_p1"])
    rep = json.loads(path.read_text())
    path.write_text(json.dumps({**rep, "fold_seed": 1}))
    with pytest.raises(ValueError, match="provenienza non accoppiata"):
        AC.compare(spec)


# ---- identita' dei run e provenienza dei rapporti (rilievo 1 del critico, 21/10) ----

def test_an_arm_launched_with_the_reference_weights_is_refused(env):
    """Un JEPA-only lanciato senza --anchor-weight 0 (configurazione di A): la configurazione effettiva non e' quella del piano."""
    good = _arm(env, "JEPA-only")
    rd = Path(good["1"]["run_dir"])
    (rd / "config.json").write_text(_config(env.plan, "W1ARM-A-30M-s1"))
    with pytest.raises(ValueError, match="configurazione effettiva diversa dal piano"):
        AC.compare(_spec(env, {"JEPA-only": good}))


def test_a_report_reused_for_two_seeds_is_refused(env):
    arm = _arm(env, "JEPA-only")
    arm["1"]["s_p1"] = arm["0"]["s_p1"]
    with pytest.raises(ValueError, match="stesso rapporto"):
        AC.compare(_spec(env, {"JEPA-only": arm}))


def test_a_report_of_another_checkpoint_is_refused(tmp_path):
    e = Env(tmp_path / "path")
    arm = _arm(e, "JEPA-only")
    other = _arm(e, "ancore-only")
    arm["0"]["p2"], other["0"]["p2"] = other["0"]["p2"], arm["0"]["p2"]  # i rapporti ufficiali della P2 scambiati fra due run
    with pytest.raises(ValueError, match="il rapporto p2 viene da"):
        AC.compare(_spec(e, {"JEPA-only": arm, "ancore-only": other}))
    e = Env(tmp_path / "sha")
    arm = _arm(e, "JEPA-only")
    ck = Path(arm["0"]["run_dir"]) / AC.CHECKPOINT
    ck.write_bytes(b"checkpoint riscritto dopo la sonda")  # stesso percorso, file diverso da quello sondato
    with pytest.raises(ValueError, match="SHA-256 di un altro checkpoint"):
        AC.compare(_spec(e, {"JEPA-only": arm}))


def test_identical_checkpoints_across_runs_are_refused(env):
    arm = _arm(env, "JEPA-only")
    ref = _arm(env, "A")
    data = (Path(ref["0"]["run_dir"]) / AC.CHECKPOINT).read_bytes()
    for sd in "01":  # lo stesso modello copiato al posto del braccio, rapporti rifatti su quel file
        rd = Path(arm[sd]["run_dir"])
        (rd / AC.CHECKPOINT).write_bytes(data)
        for t in ("s_p1", "s_p2"):
            p = Path(arm[sd][t])
            p.write_text(json.dumps({**json.loads(p.read_text()), "checkpoint_sha256": AC._select()._sha256(rd / AC.CHECKPOINT)}))
    with pytest.raises(ValueError, match="checkpoint ripetuti"):
        AC.compare(_spec(env, {"JEPA-only": arm}, ref=ref))


def test_wrong_steps_seed_encoder_plan_or_keep_calibration_are_refused(tmp_path):
    e = Env(tmp_path / "steps")
    arm = _arm(e, "JEPA-only")
    s = Path(arm["0"]["run_dir"]) / "summary.json"
    s.write_text(json.dumps({**json.loads(s.read_text()), "steps": e.steps - 1}))
    with pytest.raises(ValueError, match="passi, il piano ne prevede"):
        AC.compare(_spec(e, {"JEPA-only": arm}))
    e = Env(tmp_path / "seed")
    arm = _arm(e, "JEPA-only")
    arm["0"], arm["1"] = arm["1"], arm["0"]
    with pytest.raises(ValueError, match="seme"):
        AC.compare(_spec(e, {"JEPA-only": arm}))
    e = Env(tmp_path / "which")
    arm = _arm(e, "JEPA-only")
    p = Path(arm["1"]["p1"])
    p.write_text(json.dumps({**json.loads(p.read_text()), "which": "student"}))
    with pytest.raises(ValueError, match="non del teacher"):
        AC.compare(_spec(e, {"JEPA-only": arm}))
    e = Env(tmp_path / "plan")
    spec = _spec(e, {"JEPA-only": _arm(e, "JEPA-only")})
    with pytest.raises(ValueError, match="SHA-256 diverso"):
        AC.compare({**spec, "plan_sha256": "0" * 64})
    with pytest.raises(ValueError, match="spec senza 'plan'"):
        AC.compare({k: v for k, v in spec.items() if k != "plan"})
    with pytest.raises(ValueError, match="non e' quello dei bracci"):  # il piano ha --lr 0.0001: non e' quello dei bracci a 3e-4
        AC.compare({**spec, "lr": 3e-4})
    with pytest.raises(ValueError, match="calibrazione della keep"):
        AC.compare({**spec, "keep_calibration_sha256": "f" * 64})


def test_reference_and_arm_swapped_or_an_arm_omitted_are_refused(tmp_path):
    """Re-review del critico (21/10), rilievo 1: un'etichetta deve essere il run del suo braccio, e la spec deve coprire tutto il gruppo."""
    e = Env(tmp_path / "swap")
    spec = _spec(e, {"JEPA-only": _arm(e, "JEPA-only", (0.8, 0.8), (0.8, 0.8))})
    spec["runs"]["A"], spec["runs"]["JEPA-only"] = spec["runs"]["JEPA-only"], spec["runs"]["A"]  # ID, cartelle e rapporti coerenti fra loro
    with pytest.raises(ValueError, match="il braccio A al seme 0 e' 'W1ARM-A-30M-s0'"):
        AC.compare(spec)
    e = Env(tmp_path / "omit")
    spec = _spec(e, {})
    with pytest.raises(ValueError, match="coprire una volta sola"):
        AC.compare({**spec, "arms": ["JEPA-only"], "runs": {k: v for k, v in spec["runs"].items() if k != "ancore-only"}})
    with pytest.raises(ValueError, match="coprire una volta sola"):
        AC.compare({**spec, "arms": ["JEPA-only", "JEPA-only", "ancore-only"]})
    with pytest.raises(ValueError, match="riferimento 'JEPA-only'"):
        AC.compare({**spec, "reference": "JEPA-only", "arms": ["A", "ancore-only"]})
    with pytest.raises(ValueError, match="gruppo 'altro'"):
        AC.compare({**spec, "group": "altro"})


def test_a_plan_that_is_not_the_arms_plan_is_refused(env):
    """Il piano (anche con SHA coerente) deve essere esattamente quello di arms_plan.arm_runs per il gruppo e il LR della spec."""
    spec = _spec(env, {})
    plan = json.loads(Path(spec["plan"]).read_text())
    for r in plan["runs"]:
        if r["id"] == "W1ARM-JEPA-only-30M-s0":
            r["args"] = r["args"].replace("--anchor-weight 0", "--anchor-weight 0.1")
    other = env.tmp / "altro_piano.json"
    other.write_text(AP.plan_text(plan))
    with pytest.raises(ValueError, match="non e' quello dei bracci del gruppo objective"):
        AC.compare({**spec, "plan": str(other), "plan_sha256": AC._select()._sha256(other)})


def test_an_s_report_of_the_other_task_is_refused(env):
    arm = _arm(env, "JEPA-only")
    p = Path(arm["0"]["s_p1"])
    p.write_text(json.dumps({**json.loads(p.read_text()), "task": "p2"}))
    with pytest.raises(ValueError, match="del compito 'p2'"):
        AC.compare(_spec(env, {"JEPA-only": arm}))


LAUNCHED_PLAN = "results/infra/w1arm_20261021/bracci_objective.json"
LAUNCHED_PLAN_SHA256 = "dbe0f8feff7ebc3604dfc3c414646b51ddf3160bcfb699d8a5ed7021e6940b52"  # decisioni.md, lancio dei bracci (21/10)


def test_the_launched_plan_loads_from_its_relative_path_in_any_cwd(tmp_path, monkeypatch):
    """Re-review, rilievo 6: sul cluster il confronto legge il piano col percorso relativo al checkout, da una cwd qualunque; il piano
    lanciato e' quello registrato (SHA in decisioni.md) e quello dei bracci di arms_plan con LR 1e-4."""
    monkeypatch.chdir(tmp_path)
    spec = {"group": "objective", "reference": "A", "arms": list(ARMS), "plan": LAUNCHED_PLAN, "plan_sha256": LAUNCHED_PLAN_SHA256,
            "lr": 1e-4}
    plan = AC.training_plan(spec)
    ids = AC.expected_runs(spec, plan)
    assert len(plan["runs"]) == 6 and ids["JEPA-only", "1"] == "W1ARM-JEPA-only-30M-s1"
    assert plan["expected_sha"] == "72cf24abec5b8332011d83ce2b2fc7ae6bcc9cec"
