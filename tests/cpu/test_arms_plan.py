"""Pianificatore dei bracci dell'obiettivo (`scripts/arms_plan.py`): bracci e argomenti firmati, 30M a 4 epoche con WSD D14, semi accoppiati,
keep solo con la calibrazione sul corpus intero, valutazione P1/P2 ufficiale e S della P1 e della P2 validata dall'orchestratore, run fermati
senza nodi, sonde su GPU a richiesta; stesso stato iniziale fra bracci a pari seme (critica del punto 5, rilievo 5)."""

import importlib.util
import hashlib
import json
import math
import shlex
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AP, EC = _load("arms_plan"), _load("evaluation_chain")
PD = "results/infra/w1arm"
KEEP_SHA = "c" * 64
WPE = 784756.9792373349
REPORT = {"fractions": {"1": {"windows_per_epoch": WPE, "file_sha256": "6d6d" + "0" * 60, "file": "m_nested_1.json.gz"}},
          "code": {"source": "launcher (lib_code_dir.sh)"}}
CAMP = {"campaign": "w1arm", "agent": "claude", "code_dir": "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/w1arm", "expected_sha": "a" * 40,
        "manifest": "/M/m_nested_1.json.gz", "scales": "/M/s.json", "window_seconds": "/M/w.json", "run_root": "/leonardo_scratch/R",
        "slurm": {"qos": "normal", "time": "24:00:00"}, "margin_min": 40, "startup_min": 15, "slack": 1.15, "spare_jobs": 0, "gpus": 4,
        "eval": {"features_root": "/leonardo_scratch/F", "results_root": "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/w1arm",
                 "raw_root": "/leonardo_work/IscrB_WearUsFM/data/raw"}}


def test_objective_arms_follow_the_signed_design():
    r = AP.plan(REPORT, 3e-4, CAMP, plans_dir=PD)
    steps = math.floor(4 * WPE / 32)
    assert r["steps"] == steps == 98094 and r["warmup_steps"] == 1962 and set(r["plans"]) == {"objective"} and "keep" in r["blocked"]
    rows = r["runs"]
    assert [(x["arm"], x["seed"]) for x in rows] == [(a, s) for a in ("A", "JEPA-only", "ancore-only") for s in (0, 1)]
    assert all(x["steps"] == steps and x["decay_start"] == steps - round(0.2 * steps) for x in rows)
    args = {x["id"]: x["args"] for x in r["plans"]["objective"]["runs"]}
    assert args["W1ARM-A-30M-s0"] == "--lr 0.0003 --nested 1"
    assert args["W1ARM-JEPA-only-30M-s1"].endswith("--anchor-weight 0") and args["W1ARM-ancore-only-30M-s0"].endswith("--jepa-weight 0")
    W = _load("wsd_chain")
    jobs = W.plan_jobs(r["plans"]["objective"])  # i piani passano i controlli di D14 e degli argomenti liberi
    assert all("--manifest-sha256" in c and "--fixed-eval-every 500" in c for c in jobs["commands"])


def test_evaluation_plan_is_valid_and_compares_with_the_signed_rules():
    r = AP.plan(REPORT, 3e-4, CAMP, plans_dir=PD)
    e = r["eval_plans"]["objective"]
    order = EC.validate(e)
    kinds = [n["kind"] for n in order]
    assert kinds.count("guard") == 6 and kinds.count("extract") == 24 and kinds.count("probe") == 24 and kinds[-1] == "compare"
    cmp = order[-1]
    assert cmp["args"] == ["results/infra/w1arm/confronto_objective.json", f"{CAMP['eval']['results_root']}/confronto_objective.json"]
    assert len(cmp["after"]) == 24 and all(EC.node_labels(n) == {"lrd_all_serial", "normal"} for n in order if n["kind"] == "probe")
    s_p2 = next(n for n in order if n["id"] == "W1ARM-A-30M-s0-s_p2-ext")
    assert s_p2["args"][2:4] == ["--task", "p2"] and "--uci-root" in s_p2["args"]
    spec = r["specs"]["objective"]
    assert spec["reference"] == "A" and spec["arms"] == ["JEPA-only", "ancore-only"]
    assert set(spec["runs"]["A"]["0"]) == {"id", "run_dir", "p1", "p2", "s_p1", "s_p2"} and spec["runs"]["A"]["0"]["id"] == "W1ARM-A-30M-s0"
    # piano di training, suo SHA e LR nella spec: arms_compare verifica l'identita' dei run (rilievo 1 del critico, 21/10)
    assert spec["plan"] == "results/infra/w1arm/bracci_objective.json" and spec["lr"] == 3e-4
    assert spec["plan_sha256"] == hashlib.sha256(AP.plan_text(r["plans"]["objective"]).encode()).hexdigest()
    assert e["estimate"]["serial_probe_hours"] == pytest.approx(6 * 4.0, abs=0.1)  # sonde in fila sulla seriale: dichiarate


def test_keep_needs_its_calibration_and_failed_runs_have_no_nodes():
    r = AP.plan(REPORT, 3e-4, CAMP, keep_calibration="/K/keep_corpus.npz", keep_calibration_sha256=KEEP_SHA, plans_dir=PD)
    assert set(r["plans"]) == {"objective", "keep"} and r["blocked"] == {}
    keep = {x["id"]: x["args"] for x in r["plans"]["keep"]["runs"]}
    assert keep["W1ARM-B-30M-s0"].endswith("--keep-calibration /K/keep_corpus.npz --keep-weight 0.05")
    assert keep["W1ARM-Aprime-30M-s1"].endswith("--keep-weight 0") and r["specs"]["keep"]["reference"] == "Aprime"
    failed = "W1ARM-JEPA-only-30M-s1"
    r = AP.plan(REPORT, 3e-4, CAMP, failed_runs=frozenset({failed}), plans_dir=PD)
    order = EC.validate(r["eval_plans"]["objective"])
    assert not any(n["id"].startswith(failed) for n in order) and len(order[-1]["after"]) == 20
    assert r["specs"]["objective"]["runs"]["JEPA-only"]["1"] == {"id": failed, "run_dir": f"/leonardo_scratch/R/{failed}"}
    with pytest.raises(ValueError, match="riferimento A non e' arrivato"):
        AP.plan(REPORT, 3e-4, CAMP, failed_runs=frozenset({"W1ARM-A-30M-s0"}), plans_dir=PD)
    with pytest.raises(ValueError, match="non sono run dei bracci"):
        AP.plan(REPORT, 3e-4, CAMP, failed_runs=frozenset({"W1ARM-X-30M-s0"}), plans_dir=PD)


def test_probes_on_gpu_and_input_checks():
    r = AP.plan(REPORT, 3e-4, CAMP, probes_on_gpu=True, plans_dir=PD)
    probes = [n for n in EC.validate(r["eval_plans"]["objective"]) if n["kind"] == "probe"]
    assert all("boost_usr_prod" in EC.node_labels(n) and n["gpu_hours"] > 0 for n in probes)
    assert r["eval_plans"]["objective"]["estimate"]["serial_probe_hours"] == 0
    with pytest.raises(ValueError, match="costruzione dei derivati sul cluster"):
        AP.plan({**REPORT, "code": {"source": "git"}}, 3e-4, CAMP, plans_dir=PD)
    with pytest.raises(ValueError, match="il rapporto descrive"):
        AP.plan(REPORT, 3e-4, {**CAMP, "manifest": "/M/altro.json.gz"}, plans_dir=PD)


def test_arms_with_the_same_seed_start_from_the_same_state(monkeypatch):
    """Rilievo 5 della critica: a pari seme i tre bracci hanno la stessa architettura e lo stesso stato iniziale (cambiano solo i pesi della
    loss), con le configurazioni costruite da train_jepa.py dagli argomenti del piano."""
    import torch

    from wearusfm.model.fm import WearUsFM
    from wearusfm.training import run as R

    TJ = _load("train_jepa")
    r = AP.plan(REPORT, 3e-4, CAMP, plans_dir=PD)
    plan = r["plans"]["objective"]
    captured = {}
    # monkeypatch, non assegnazione: TJ.R e' il modulo condiviso training.run e va ripristinato per gli altri test
    monkeypatch.setattr(TJ.R, "train", lambda cfg, *a, **k: captured.__setitem__("cfg", cfg) or {"steps": 0})
    monkeypatch.setattr(TJ.R, "check_input_files", lambda *a, **k: None)
    states = {}
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws.json"
        ws.write_text(json.dumps({"window_s": {}, "scales": {}}))
        for run in plan["runs"]:
            if run["seed"] != 0:
                continue
            extra = f"{plan['common_args']} --window-seconds {ws} --batch-size 32 --accum-steps 1 --seed 0 --schedule wsd " \
                    f"--total-steps {run['total_steps']} --steps-at-d {run['steps_at_d']} {run['args']}"
            TJ.main(["--preset", "sanity", "--manifest", "m.json.gz", "--root", "r", "--scales", str(ws), "--out-dir", str(Path(tmp) / run["id"])]
                    + shlex.split(extra))
            cfg = captured["cfg"]
            torch.manual_seed(cfg.seed)  # come train(): stessa inizializzazione su tutti i processi
            states[run["id"]] = (R.config_to_dict(cfg)["model"], WearUsFM(cfg.model).state_dict(), cfg.jepa)
    (m0, s0, j0), *others = states.values()
    for m, s, j in others:
        assert m == m0 and s.keys() == s0.keys() and all(torch.equal(s[k], s0[k]) for k in s0)
    assert {j.anchor_weight for _, _, j in states.values()} == {0.2, 0.0} and {j.jepa_weight for _, _, j in states.values()} == {1.0, 0.0}


def test_extractions_fit_the_walltime_of_their_effective_qos():
    """R3 della review GPT 3272398: le estrazioni chiedono QoS normal esplicita (probe_p1.sbatch ha boost_qos_dbg, 30 min, nell'intestazione);
    QoS effettiva letta dall'intestazione del launcher allo SHA vero, e un nodo oltre il limite e' un errore del pianificatore."""
    import subprocess

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    result = AP.plan(REPORT, 3e-4, {**CAMP, "expected_sha": head}, plans_dir=PD)
    for name, ev in result["eval_plans"].items():
        for n in EC.validate(ev):
            if n["kind"] == "extract":
                assert "--qos=normal" in n["sbatch"] and "boost_qos_dbg" not in EC.node_labels(n), (name, n["id"])
    bad = {"id": "x", "kind": "extract", "queue": "boost_usr_prod", "code_dir": "/C", "expected_sha": head,
           "launcher": "scripts/slurm/probe_p1.sbatch", "sbatch": ["--time=00:45:00"]}
    assert "boost_qos_dbg" in EC.node_labels(bad)  # l'intestazione vera, senza la QoS esplicita
    with pytest.raises(ValueError, match="oltre il limite di boost_qos_dbg"):
        AP.check_qos_walltime([bad])
    AP.check_qos_walltime([{**bad, "sbatch": ["--time=00:25:00"]}])  # entro il limite: nessun errore


# ---- rilievi del critico del 21/10 (LR dalla regola firmata, cartella dei piani, calibrazione della keep) ----

DECISIONS = REPO / "results" / "infra" / "w1cal_decisioni.json"
REGISTRO_C2 = REPO / "results" / "infra" / "w1cal_c2_20261020" / "valutazione" / "registro_C2.json"


def _registro(s_lo, s_hi, tie, confirmed=(1e-4, 3e-4)):
    return {"phase": "C2", "decision": {"confirmed": list(confirmed)}, "rules": {"tie": "entro 1 SE"},
            "candidates": {"0.0001": {"S": s_lo}, "0.0003": {"S": s_hi}},
            "descriptive_comparisons": [{"a": 1e-4, "b": 3e-4, "delta": s_lo - s_hi, "se": 0.003, "tie": tie}]}


def test_the_arms_lr_follows_the_signed_rule_on_the_real_c2_record():
    lr, why = AP.signed_lr(json.loads(DECISIONS.read_text()), json.loads(REGISTRO_C2.read_text()))
    assert lr == 1e-4 and why["tie"] and why["confirmed"] == [1e-4, 3e-4]  # S 0,1801 contro 0,1795, entro 1 SE: il piu' basso
    assert why["delta_low_minus_high"] == pytest.approx(0.18010974754 - 0.17950757857, abs=1e-9)


def test_signed_lr_cases():
    dec = {"C2": {"confirmed": [1e-4, 3e-4]}}
    assert AP.signed_lr(dec, _registro(0.17, 0.19, tie=False))[0] == 3e-4  # S migliore al seme 1, fuori dalla parita'
    assert AP.signed_lr(dec, _registro(0.19, 0.17, tie=False))[0] == 1e-4
    assert AP.signed_lr(dec, _registro(0.17, 0.172, tie=True))[0] == 1e-4  # parita': il piu' basso anche con S piu' alta per 3e-4
    one = {"C2": {"confirmed": [3e-4]}}
    assert AP.signed_lr(one, {**_registro(0.1, 0.2, False, confirmed=(3e-4,)), "descriptive_comparisons": []})[0] == 3e-4
    with pytest.raises(ValueError, match="incoerente col file delle decisioni"):
        AP.signed_lr(one, _registro(0.17, 0.19, tie=False))
    bad = _registro(0.17, 0.19, tie=False)
    bad["descriptive_comparisons"][0]["delta"] = 0.02  # segno sbagliato rispetto alle S
    with pytest.raises(ValueError, match="incoerente con le S"):
        AP.signed_lr(dec, bad)


def test_plans_dir_must_be_relative_to_the_repo():
    for bad in ("/Users/x/w1arm", "../w1arm", "results/../../w1arm", ""):
        with pytest.raises(ValueError, match="relativa al repo"):
            AP.plan(REPORT, 3e-4, CAMP, plans_dir=bad)


def test_keep_needs_a_declared_full_corpus_calibration():
    with pytest.raises(ValueError, match="richiede --keep-calibration-sha256"):
        AP.plan(REPORT, 3e-4, CAMP, keep_calibration="/K/keep.npz", plans_dir=PD)
    with pytest.raises(ValueError, match="dello Step 1"):
        AP.plan(REPORT, 3e-4, CAMP, keep_calibration="/K/keep.npz", keep_calibration_sha256=AP.STEP1_KEEP_CALIBRATION_SHA256, plans_dir=PD)
    with pytest.raises(ValueError, match="senza --keep-calibration"):
        AP.plan(REPORT, 3e-4, CAMP, keep_calibration_sha256=KEEP_SHA, plans_dir=PD)
    r = AP.plan(REPORT, 3e-4, CAMP, keep_calibration="/K/keep.npz", keep_calibration_sha256=KEEP_SHA, plans_dir=PD)
    assert r["specs"]["keep"]["keep_calibration_sha256"] == KEEP_SHA and "keep_calibration_sha256" not in r["specs"]["objective"]


def test_cli_derives_the_lr_and_refuses_another(tmp_path, monkeypatch):
    rep, camp = tmp_path / "rep.json", tmp_path / "camp.json"
    rep.write_text(json.dumps(REPORT))
    camp.write_text(json.dumps(CAMP))
    monkeypatch.chdir(tmp_path)
    base = [str(rep), "--decisions", str(DECISIONS), "--registro-c2", str(REGISTRO_C2), "--campaign", str(camp), "--out", "piani"]
    assert AP.main(base) == 0
    summary = json.loads((tmp_path / "piani" / "bracci_riepilogo.json").read_text())
    assert summary["lr"] == 1e-4 and summary["lr_rule"]["tie"] and len(summary["lr_rule"]["registro_c2_sha256"]) == 64
    spec = json.loads((tmp_path / "piani" / "confronto_objective.json").read_text())
    assert spec["plan"] == "piani/bracci_objective.json"
    assert spec["plan_sha256"] == hashlib.sha256((tmp_path / "piani" / "bracci_objective.json").read_bytes()).hexdigest()
    assert AP.main(base[:-2] + ["--out", "piani2", "--lr", "1e-4"]) == 0
    with pytest.raises(SystemExit, match="regola firmata"):
        AP.main(base[:-2] + ["--out", "piani3", "--lr", "3e-4"])
