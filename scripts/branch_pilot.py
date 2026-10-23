#!/usr/bin/env python3
"""Pilot dell'obiettivo temporale dei rami spettrali (protocollo `rami_pilota_v1`, `docs/rami_pilota_obiettivo_v1.md`; punto 5, PSD-2).
Raw congelato dei teacher B (B_seed0, B_seed1), stessi dati, configurazione e passo 0 del 2C (`step2c_frozen_branches_v1`, run `step2c_1410`);
cambiano solo l'obiettivo dei rami (PWm), gli strati allenati (PWf, PWm: `output` e `combine` fissi al passo 0) o l'LR (PWl).

  train    --arm PWl|PWf|PWm --teacher T [--collaudo]   rami sul teacher T congelato (GPU) -> PILOT_RUNS/<braccio>/T (collaudo: collaudo/)
  hybrid   --teacher T                                  PWh = token di PW, `output` e `combine` di PW0 (CPU, minuti) -> PILOT_RUNS/PWh/T
  extract  --task p1|p2 --arm PWl|PWf|PWm|PWh --teacher T   uscite dei rami (CPU) -> EVAL/<task>/T/branches_<braccio>
  compare  --diagnostic-dir D --reports-dir P --out esito.json [--primary-only]   confronti e lettura congelata
  plan     ...                                          comandi dei training e piano della catena di valutazione

Le sonde sono quelle della diagnostica (`spectral_diagnostic.py probe --pilot-root EVAL --sets RPWm ...`): stessa validation interna, stesse
cache raw, stessi soggetti e fold; PW0 e PW (2C) sono i rapporti della diagnostica. Ogni passo controlla la provenienza di cio' che legge.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch  # prima di sklearn sul Mac

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import probe_p1  # noqa: E402
import step1_downstream as SD  # noqa: E402
import step2a as S2  # noqa: E402
import step2c as S2C  # noqa: E402
from spectral_step1 import output_path  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.spectral_calibration import sha256_file  # noqa: E402

PROTOCOL = "rami_pilota_v1"
TEACHERS = S2C.FAMILIES["B"]
# braccio -> LR, strati finali (`output`, `combine`) fissi al passo 0, obiettivo dei rami
ARMS = {"PWl": {"lr": 3e-4, "freeze_output": False, "objective": "readout"},
        "PWf": {"lr": S2C.LR, "freeze_output": True, "objective": "readout"},
        "PWm": {"lr": S2C.LR, "freeze_output": True, "objective": "masked"}}
MODELS = (*ARMS, "PWh")  # i modelli nuovi di cui si estraggono le uscite dei rami (PW0 e PW sono quelle del 2C)
# le sole differenze ammesse dalla configurazione del 2C (chiavi appiattite); `max_steps` solo nel controllo del passo 1 del collaudo
ALLOWED = {"PWl": {("lr",), ("branch_monitor",)}, "PWf": {("model", "branch_freeze_output")},
           "PWm": {("model", "branch_objective"), ("model", "branch_freeze_output")}}
STEP2C_COMMIT = "1c9485260aaf32d5936346b50225f5cff1d1b8ea"
STEP2C_RUNS = {  # results/passo6/step2c_runs_20261014.json
    "B_seed0": {"checkpoint_sha256": "7d45049efc1af79003ba4d34f8d2ab26aea4aeed3ce7acbab1861b5a2413bcb8",
                "init_checkpoint_sha256": "6d4f9e68fbd59be28ed2fd79eff6b9e833294e1344fe70c174a01e9de5dd6cc8"},
    "B_seed1": {"checkpoint_sha256": "1c4016ac79e5567b74fc73a8ee8f8396ffedb0a62b6feea9f254e76c1c575865",
                "init_checkpoint_sha256": "30d447364966d70d6fb548aa59939aaed69d93f94b5bd096dfbff8eb40380a0d"}}
OLD = ("spectral_branches.", "branch_heads.")  # rami e teste del 2C (i predittori dell'obiettivo nuovo non ci sono)
FIXED_LAYERS = (".output.", ".combine.")
TOKEN_LAYERS = (".project.", ".temporal.")
STEP1_RTOL = 1e-4  # perdita del passo 1 di PWl e PWf contro quella del 2C
LEARN_MAX, LEARN_WINDOW = 0.9, 100  # PWm «ha imparato»: perdita normalizzata media sugli ultimi 100 aggiornamenti <= 0,9
HARM_DELTA = -0.01  # il «non dannoso» del 2A/2C sulla stima puntuale
# collaudo su dbg (PWm, B_seed0; criteri del protocollo, congelati prima)
DBG_SECONDS = 1800
COLLAUDO = {"arm": "PWm", "teacher": "B_seed0", "coverage_min": 0.5, "margin": 1.25, "slack_s": 600, "max_run_h": 1.5, "skip": 50,
            "smooth": 50, "window": 100, "max_rise": 1.5}


@functools.lru_cache(maxsize=None)
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_pilot", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- configurazione e controlli del run (funzioni pure, provate nei test CPU) -------------------------------------------------------------

def build_config(arm: str, teacher: str, calibration, max_steps: int | None = None) -> R.RunConfig:
    """La configurazione del 2C per il teacher (`step2c.build_config`) col cambiamento del braccio. PWl accende i monitor dei rami (PWf e PWm
    li hanno sempre). Nessun pannello fisso: col PWm misurerebbe la loss di readout (il trainer lo rifiuta comunque)."""
    spec = ARMS[arm]
    cfg = S2C.build_config(teacher, calibration)
    cfg = replace(cfg, lr=spec["lr"], branch_monitor=arm == "PWl",
                  model=replace(cfg.model, branch_objective=spec["objective"], branch_freeze_output=spec["freeze_output"]))
    if max_steps is not None:
        cfg = replace(cfg, max_steps=max_steps)
    if cfg.fixed_eval_every:
        raise ValueError("pilot: pannello fisso spento")
    return cfg


def _flat(d: dict, prefix: tuple = ()) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def config_check(cfg: dict, ref: dict, arm: str, extra: tuple = ()) -> dict:
    """La configurazione del run contro quella del run del 2C dello stesso teacher, normalizzate coi default dei campi aggiunti dopo
    (`run._config_with_defaults`): diverse solo nelle chiavi del braccio (`ALLOWED`, piu' `extra`)."""
    a, b = _flat(R._config_with_defaults(cfg)), _flat(R._config_with_defaults(ref))
    missing = "<assente>"
    diff = sorted(k for k in set(a) | set(b) if a.get(k, missing) != b.get(k, missing))
    allowed = ALLOWED[arm] | set(extra)
    unexpected = [".".join(k) for k in diff if k not in allowed]
    return {"differences": {".".join(k): [b.get(k, missing), a.get(k, missing)] for k in diff}, "unexpected": unexpected,
            "ok": not unexpected}


def trainable_names(cfg: R.RunConfig) -> set[str]:
    """I parametri che il trainer allena con questa configurazione (raw congelato, `freeze_branch_layers`)."""
    from wearusfm.model import spectral_branches as SB
    from wearusfm.model.fm import WearUsFM

    m = WearUsFM(cfg.model)
    SB.freeze_raw(m)
    SB.freeze_branch_layers(m)
    return {n for n, p in m.named_parameters() if p.requires_grad}


def init_check(ref_init: dict, own_init: dict) -> dict:
    """Il passo 0 del run (`checkpoint_init.pt`, dopo il caricamento dei rami) uguale bit per bit al passo 0 del 2C su rami e teste, nello
    studente e nel teacher."""
    out = {}
    for part in ("student", "teacher"):
        keys = sorted(k for k in ref_init[part] if k.startswith(OLD))
        same = sorted(k for k in own_init[part] if k.startswith(OLD)) == keys
        out[part] = bool(keys and same and all(torch.equal(own_init[part][k], ref_init[part][k]) for k in keys))
    out["identical"] = out["student"] and out["teacher"]
    return out


def fixed_check(trainable: set[str], ref_init: dict, final: dict) -> dict:
    """A fine run, nello studente e nel teacher: ogni tensore di rami e teste che il braccio non allena (statistiche d'ingresso, frontend,
    strati fissi, teste fisse) uguale bit per bit al passo 0 del 2C; e i token dei rami cambiati (descrittivo)."""
    keys = sorted(k for k in ref_init["student"] if k.startswith(OLD) and k not in trainable)
    tokens = sorted(k for k in ref_init["student"] if k.startswith("spectral_branches.") and any(t in k for t in TOKEN_LAYERS))
    out = {"fixed_tensors": len(keys), "trainable_tensors": len(trainable)}
    for part in ("student", "teacher"):
        out[part] = bool(keys and all(torch.equal(final[part][k], ref_init[part][k]) for k in keys))
        out[part + "_tokens_changed"] = any(not torch.equal(final[part][k], ref_init[part][k]) for k in tokens)
    out["identical"] = out["student"] and out["teacher"]
    return out


def step1_check(metrics: list[dict], ref_metrics: list[dict]) -> dict:
    """PWl e PWf: la perdita del passo 1, calcolata prima del primo aggiornamento (indipendente da LR e strati fissi), uguale a quella del 2C
    entro 1e-4 relativo. Se no il run non e' valido (protocollo: STOP, nessuna ri-estrazione di ripiego)."""
    a = next((r.get("keep_branches") for r in metrics if r.get("step") == 1), None)
    b = next((r.get("keep_branches") for r in ref_metrics if r.get("step") == 1), None)
    ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b) and \
        abs(a - b) <= STEP1_RTOL * abs(b)
    return {"run": a, "step2c": b, "rtol": STEP1_RTOL, "ok": bool(ok)}


def token_distance(ref: dict, state: dict) -> dict:
    """||theta - theta0|| / ||theta0|| dei token dei rami (`project`, `temporal`), per ramo e in tutto (descrittivo)."""
    out = {}
    groups = {"all": "spectral_branches."} | {k: f"spectral_branches.branches.{k}." for k in S2C.KINDS}
    for name, prefix in groups.items():
        keys = [k for k in ref if k.startswith(prefix) and any(t in k for t in TOKEN_LAYERS)]
        num = math.sqrt(sum(float((state[k].double() - ref[k].double()).pow(2).sum()) for k in keys))
        den = math.sqrt(sum(float(ref[k].double().pow(2).sum()) for k in keys))
        out[name] = num / den if den > 0 else None
    return out


def hybrid_state(trained: dict, init: dict) -> dict:
    """PWh: il checkpoint di PW (2C) con `output` e `combine` di ogni ramo presi da PW0 (il passo 0 del 2C), nello studente e nel teacher."""
    out = dict(trained)
    for part in ("student", "teacher"):
        out[part] = {k: (init[part][k].clone() if k.startswith("spectral_branches.") and any(f in k for f in FIXED_LAYERS) else v)
                     for k, v in trained[part].items()}
    out.pop("optimizer", None)  # non serve all'estrazione, e i momenti sarebbero quelli di PW
    return out


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _undefined(r: dict) -> bool:
    """Misura non definita: la media (`masked_normalized` null) o una componente ramo x scala presente ma non finita (riferimento a varianza
    nulla). La media del trainer salta le componenti non finite (`branch_masking`): senza questo controllo resterebbe definita (critico, 23/10).
    Una componente assente (nessun bersaglio di quella scala nel batch) non e' una misura."""
    if not _finite(r.get("masked_normalized")):
        return True
    return any(k.startswith("masked_") and k.endswith("_normalized") and k != "masked_normalized" and not _finite(v) for k, v in r.items())


def learned(metrics: list[dict]) -> dict:
    """PWm ha imparato il suo compito: media della perdita normalizzata sugli ultimi 100 aggiornamenti <= 0,9. Una sola misura non definita
    fra le ultime 100 (`_undefined`; il trainer scrive null) rende il criterio non verificabile (review GPT, R1)."""
    window = metrics[-LEARN_WINDOW:]
    last = [r.get("masked_normalized") for r in window]
    undefined = sum(_undefined(r) for r in window)
    if len(last) < LEARN_WINDOW or undefined:
        return {"verifiable": False, "updates": len(last), "undefined": undefined, "mean": None, "learned": False}
    mean = float(np.mean(last))
    return {"verifiable": True, "updates": len(last), "undefined": 0, "mean": mean, "max": LEARN_MAX, "learned": mean <= LEARN_MAX}


def branch_keys(kinds=S2C.KINDS) -> list[str]:
    from wearusfm.model.spectral_branches import SCALES

    return [f"{k}_{s[0]}" for k in kinds for s in SCALES]


def collaudo_verdict(metrics: list[dict], summary: dict, integrity: dict) -> dict:
    """Verdetto congelato del collaudo (PWm, B_seed0, dbg): integrita' (nessuno STOP, raw e strati fissi uguali al passo 0 del 2C, passo 0
    caricato dal 2C, configurazione del 2C, controllo del passo 1 di PWf), apprendimento (`learned`), monitor (`collapse_monitor.verified`:
    almeno un controllo, nessun allarme, riferimenti definiti), copertura (ogni ramo e scala con almeno un bersaglio con contesto in almeno il
    50% degli aggiornamenti; conteggi per banda registrati), tempo ((avvio + 3.000 x mediana dei secondi per aggiornamento dal 51esimo) x 1,25
    + 10 minuti, al quarto d'ora: e' l'allocazione dei run, oltre 1 h 30 non passa). Stabilita' descrittiva (regola del collaudo del 2C sulla
    media mobile della perdita mascherata, non bloccante)."""
    c = COLLAUDO
    n = len(metrics)
    out = {"updates": n, "criteria": dict(c), "integrity": integrity, "integrity_ok": bool(integrity and all(integrity.values()))}
    out["learning"] = learned(metrics)
    monitor = (summary.get("spectral_branches") or {}).get("collapse_monitor") or {}
    out["monitor"] = {"verified": bool(monitor.get("verified")), "alarm": monitor.get("alarm"), "checks": monitor.get("checks"),
                      "reference_undefined": monitor.get("reference_undefined")}
    coverage, bands = {}, {}
    for key in branch_keys():
        hit = sum(1 for r in metrics if (r.get(f"masked_{key}_targets") or 0) > 0)
        coverage[key] = hit / n if n else 0.0
        per = [r.get(f"masked_{key}_context_per_band") for r in metrics if r.get(f"masked_{key}_context_per_band") is not None]
        bands[key] = np.sum(np.asarray(per, dtype=np.int64), 0).tolist() if per else None
    out["coverage"] = {"fraction": coverage, "context_per_band": bands, "ok": bool(n and all(v >= c["coverage_min"] for v in coverage.values()))}
    t = np.asarray([r["t_step_s"] for r in metrics], dtype=np.float64)
    if n:
        step_s = float(np.median(t[c["skip"]:] if n > 2 * c["skip"] else t))
        startup = float(summary["elapsed_s"] - t.sum())
        alloc_h = math.ceil(((startup + S2C.STEPS * step_s) * c["margin"] + c["slack_s"]) / 900) / 4
        out["time"] = {"median_step_s": step_s, "startup_s": startup, "run_allocation_h": alloc_h, "ok": alloc_h <= c["max_run_h"]}
    else:
        out["time"] = {"ok": False}
    loss = np.asarray([r.get("masked") if _finite(r.get("masked")) else np.nan for r in metrics], dtype=np.float64)
    if n >= 2 * c["window"] and np.isfinite(loss).all():
        ma = np.convolve(loss, np.ones(c["smooth"]) / c["smooth"], mode="valid")
        rise = ma[c["window"]:] / np.minimum.accumulate(ma)[c["window"]:]
        out["stability_descriptive"] = {"max_rise": float(rise.max()) if len(rise) else None,
                                        "stable": bool(not len(rise) or rise.max() <= c["max_rise"])}
    else:
        out["stability_descriptive"] = {"max_rise": None, "stable": None}
    out["passed"] = bool(out["integrity_ok"] and out["learning"]["learned"] and out["monitor"]["verified"] and out["coverage"]["ok"]
                         and out["time"]["ok"])
    return out


# --- training (GPU) ---------------------------------------------------------------------------------------------------------------------

def step2c_run(args, teacher: str) -> tuple[Path, dict]:
    """Il run del 2C del teacher (passo 0 e PW), con la provenienza e i due checkpoint congelati."""
    folder = args.step2c_runs / teacher
    prov = json.loads((folder / "run_provenance.json").read_text())
    expected = {"protocol": S2C.PROTOCOL, "source_commit": STEP2C_COMMIT, "arm": teacher, "collaudo": False, "complete": True,
                "encoder_identical": True, "source_checkpoint_sha256": SD.CHECKPOINTS[teacher], **STEP2C_RUNS[teacher]}
    bad = {k: (prov.get(k), v) for k, v in expected.items() if prov.get(k) != v}
    if bad:
        raise ValueError(f"{folder}: run del 2C diverso da quello congelato {bad}")
    for name, key in S2C.WHICH.values():
        if sha256_file(folder / name) != prov[key]:
            raise ValueError(f"{folder / name}: checkpoint del 2C cambiato")
    return folder, prov


def _run_dir(args, arm: str, teacher: str, collaudo: bool = False) -> Path:
    return args.pilot_runs / "collaudo" / f"{arm}_{teacher}" if collaudo else args.pilot_runs / arm / teacher


def _train_one(args, arm: str, teacher: str, out: Path, *, max_steps: int | None, time_limit_s: float, source: Path, run2c: Path,
               scales: dict | None, window_s: dict | None, extra: tuple = ()) -> tuple[dict, dict]:
    """Un run del braccio e i suoi controlli (provenienza scritta anche per un run fallito: nessuna estrazione afterok)."""
    cfg = build_config(arm, teacher, args.calibration, max_steps)
    ref_cfg = json.loads((run2c / "config.json").read_text())
    cfg_check = config_check(R.config_to_dict(cfg), ref_cfg, arm, extra)
    if not cfg_check["ok"]:
        raise ValueError(f"configurazione diversa dal 2C oltre il braccio: {cfg_check['unexpected']}")
    print(json.dumps({"protocol": PROTOCOL, "arm": arm, "teacher": teacher, "config_check": cfg_check, "config": R.config_to_dict(cfg)}),
          flush=True)
    summary = R.train(cfg, args.manifest, args.root, out, scales=scales, window_s=window_s, device=args.device, num_workers=args.num_workers,
                      time_limit_s=time_limit_s, log=lambda m: print(m, flush=True), init_from=source,
                      branch_state_from=run2c / "checkpoint_init.pt")
    prov = {"protocol": PROTOCOL, "source_commit": args.expected_commit, "arm": arm, "teacher": teacher, "collaudo": args.collaudo,
            "source_checkpoint_sha256": SD.CHECKPOINTS[teacher], "step2c": {"run": str(run2c), **STEP2C_RUNS[teacher]},
            "input_sha256": S2C.INPUT_SHA256, "time_limit_s": time_limit_s, "steps": summary["steps"], "stopped": summary["stopped"],
            "complete": False, "checks_passed": False, "config_check": cfg_check,
            **{key: sha256_file(out / name) if (out / name).exists() else None for name, key in S2C.WHICH.values()}}
    (out / "run_provenance.json").write_text(json.dumps(prov, indent=2))
    prov["complete"] = bool(summary["steps"] == cfg.max_steps and summary["stopped"] == "max_steps" and not (out / "STOP").exists())
    checks = {"config": cfg_check["ok"]}
    if prov["checkpoint_sha256"] is not None and prov["init_checkpoint_sha256"] is not None:
        state, own_init, ref_init = _load(out / "checkpoint.pt"), _load(out / "checkpoint_init.pt"), _load(run2c / "checkpoint_init.pt")
        source_teacher = _load(source)["teacher"]
        raw = {"final": S2C.raw_check(source_teacher, state), "init": S2C.raw_check(source_teacher, own_init)}
        spectral = json.loads((out / "spectral_branches.json").read_text())
        branch_state = spectral.get("branch_state") or {}
        try:
            S2C.SA._finite_state(state)
            finite = True
        except ValueError:
            finite = False
        # come il 2C (step2c.train): checkpoint del run, passo 0 al passo 0, provenienza dei rami col passo 0 registrato
        consistent = bool(state.get("config") == R.config_to_dict(cfg) and own_init.get("config") == state.get("config")
                          and own_init.get("step") == 0 and spectral.get("init_checkpoint_sha256") == prov["init_checkpoint_sha256"])
        prov["checks"] = {
            "raw": raw, "init": init_check(ref_init, own_init), "fixed": fixed_check(trainable_names(cfg), ref_init, state),
            "branch_state": {"sha256": branch_state.get("sha256"), "calibration_reproduced": branch_state.get("calibration_reproduced")},
            "token_distance": {p: token_distance(ref_init[p], state[p]) for p in ("student", "teacher")}}
        checks.update(finite=finite, consistent=consistent, raw=raw["final"]["encoder_identical"] and raw["init"]["encoder_identical"],
                      init=prov["checks"]["init"]["identical"], fixed=prov["checks"]["fixed"]["identical"],
                      branch_state=branch_state.get("sha256") == STEP2C_RUNS[teacher]["init_checkpoint_sha256"])
        if ARMS[arm]["objective"] == "readout":
            prov["checks"]["step1"] = step1_check(_records(out / "metrics.jsonl"), _records(run2c / "metrics.jsonl"))
            checks["step1"] = prov["checks"]["step1"]["ok"]
    else:
        checks["checkpoint"] = False
    prov["checks_ok"] = checks
    prov["checks_passed"] = bool(all(checks.values()))
    SD.check_source(args.expected_commit)
    if {k: sha256_file(getattr(args, k)) for k in S2C.INPUTS} != S2C.INPUT_SHA256 or sha256_file(source) != SD.CHECKPOINTS[teacher]:
        raise ValueError("ingressi o teacher cambiati durante il training")
    (out / "run_provenance.json").write_text(json.dumps(prov, indent=2))
    return prov, summary


def train(args) -> None:
    import time

    t0 = time.time()
    SD.check_source(args.expected_commit)
    arm, teacher = args.arm, args.teacher
    if {k: sha256_file(getattr(args, k)) for k in S2C.INPUTS} != S2C.INPUT_SHA256:
        raise ValueError("ingressi del training diversi da quelli dell'A/B e del 2C")
    source = S2C._source(args, teacher)
    run2c, _ = step2c_run(args, teacher)
    data = {"scales": json.loads(args.scales.read_text())["scales"], "window_s": json.loads(args.window_seconds.read_text())["window_s"]}
    if not args.collaudo:
        out = output_path(_run_dir(args, arm, teacher))
        prov, _ = _train_one(args, arm, teacher, out, max_steps=None, time_limit_s=args.time_limit_s, source=source, run2c=run2c, **data)
        if not (prov["complete"] and prov["checks_passed"]):
            raise ValueError(f"run incompleto o con controlli falliti ({prov['checks_ok']}): nessuna estrazione, nessuna ripresa")
        return
    if (arm, teacher) != (COLLAUDO["arm"], COLLAUDO["teacher"]) or not 0 < args.time_limit_s <= DBG_SECONDS:
        raise ValueError("collaudo: PWm su B_seed0, su dbg (al piu' 30 minuti)")
    # 1) PWf per un passo: configurazione, passo 0 e perdita del passo 1 contro il 2C (la verifica d'inizializzazione dei bracci col vecchio
    #    obiettivo, che il PWm non puo' dare)
    first = output_path(args.pilot_runs / "collaudo" / f"PWf_{teacher}_passo1")
    p1, _ = _train_one(args, "PWf", teacher, first, max_steps=1, time_limit_s=args.time_limit_s, source=source, run2c=run2c, **data,
                       extra=(("max_steps",),))
    # 2) PWm col tempo che resta
    out = output_path(_run_dir(args, arm, teacher, collaudo=True))
    left = args.time_limit_s - (time.time() - t0)
    if left < 300:
        raise ValueError("collaudo: tempo esaurito dal controllo del passo 1")
    prov, summary = _train_one(args, arm, teacher, out, max_steps=None, time_limit_s=left, source=source, run2c=run2c, **data)
    integrity = {"no_stop": not (out / "STOP").exists(), **{f"PWm_{k}": v for k, v in prov["checks_ok"].items()},
                 **{f"PWf_passo1_{k}": v for k, v in p1["checks_ok"].items()}}
    verdict = collaudo_verdict(_records(out / "metrics.jsonl"), summary, integrity)
    prov["collaudo_verdict"] = verdict
    (out / "run_provenance.json").write_text(json.dumps(prov, indent=2, allow_nan=False))
    print(json.dumps(verdict), flush=True)
    if not verdict["passed"]:
        raise ValueError("collaudo non passato: stop e proposta, nessun retry automatico")


# --- ibrido PWh (CPU) -------------------------------------------------------------------------------------------------------------------

def hybrid(args) -> None:
    SD.check_source(args.expected_commit)
    teacher = args.teacher
    run2c, prov2c = step2c_run(args, teacher)
    trained, init = _load(run2c / "checkpoint.pt"), _load(run2c / "checkpoint_init.pt")
    state = hybrid_state(trained, init)
    fixed = [k for k in state["teacher"] if k.startswith("spectral_branches.") and any(f in k for f in FIXED_LAYERS)]
    checks = {"fixed_from_PW0": bool(fixed) and all(torch.equal(state[p][k], init[p][k]) for p in ("student", "teacher") for k in fixed),
              "rest_from_PW": all(torch.equal(state[p][k], trained[p][k]) for p in ("student", "teacher") for k in trained[p] if k not in fixed)}
    out = output_path(_run_dir(args, "PWh", teacher))
    out.mkdir(parents=True)
    torch.save(state, out / "checkpoint.pt")
    prov = {"protocol": PROTOCOL, "source_commit": args.expected_commit, "arm": "PWh", "teacher": teacher, "collaudo": False,
            "source_checkpoint_sha256": SD.CHECKPOINTS[teacher], "step2c": {"run": str(run2c), **STEP2C_RUNS[teacher]},
            "complete": True, "checks_ok": checks, "checks_passed": all(checks.values()), "fixed_tensors": len(fixed),
            "checkpoint_sha256": sha256_file(out / "checkpoint.pt"),
            # la distanza dei token di PW (2C) dal passo 0: sono i token di PWh
            "checks": {"token_distance": {p: token_distance(init[p], trained[p]) for p in ("student", "teacher")}}}
    SD.check_source(args.expected_commit)
    (out / "run_provenance.json").write_text(json.dumps(prov, indent=2))
    if not prov["checks_passed"]:
        raise ValueError(f"ibrido non valido: {checks}")


# --- uscite dei rami (CPU) --------------------------------------------------------------------------------------------------------------

def pilot_run(args, arm: str, teacher: str) -> tuple[Path, dict]:
    """Un run del pilot completo e coi controlli passati, dallo stesso codice di questa estrazione, col checkpoint registrato."""
    folder = _run_dir(args, arm, teacher)
    prov = json.loads((folder / "run_provenance.json").read_text())
    expected = {"protocol": PROTOCOL, "source_commit": args.expected_commit, "arm": arm, "teacher": teacher, "collaudo": False,
                "complete": True, "checks_passed": True, "source_checkpoint_sha256": SD.CHECKPOINTS[teacher]}
    bad = {k: (prov.get(k), v) for k, v in expected.items() if prov.get(k) != v}
    if bad:
        raise ValueError(f"{folder}: run non valido per il pilot {bad}")
    if sha256_file(folder / "checkpoint.pt") != prov["checkpoint_sha256"]:
        raise ValueError(f"{folder}: checkpoint cambiato dopo il training")
    return folder, prov


EXTRACT_DEVICE = "cpu"  # come il 2C (fp32): su CUDA l'estrazione va in bf16 e le uscite non sarebbero confrontabili con PW0 e PW (critico)


def cache_identity(commit: str, task: str, teacher: str, arm: str) -> dict:
    """Identita' di una cache delle uscite dei rami del pilot; `arm` = teacher come nella diagnostica, `model` = il braccio."""
    return {"protocol": PROTOCOL, "source_commit": commit, "task": task, "arm": teacher, "model": arm, "variant": f"branches_{arm}",
            "which": "teacher", "probe_seed": 0, "split_sha256": SD.SPLIT_SHA, "source_checkpoint_sha256": SD.CHECKPOINTS[teacher],
            "device": EXTRACT_DEVICE}


def extract(args) -> None:
    SD.check_source(args.expected_commit)
    commit, task, arm, teacher = args.expected_commit, args.task, args.arm, args.teacher
    run, prov = pilot_run(args, arm, teacher)
    _, meta = S2._downstream(args, task, teacher)  # identita' e file della cache raw, prima di calcolare
    folder = output_path(args.eval_root / task / teacher / f"branches_{arm}")
    ckpt = run / "checkpoint.pt"
    if task == "p2":
        if not args.epn_root or not args.uci_root:
            raise ValueError("P2: servono le radici di EPN-612 e UCI-EMG")
        S2C._extract_p2(args, {arm: ckpt}, {arm: folder}, device=EXTRACT_DEVICE)
    else:
        argv = S2C._probe_argv(args, task, ckpt, folder, "extract", "branches") + ["--audit-inputs", "--batch", "64", "--device",
                                                                                   EXTRACT_DEVICE]
        if probe_p1.main(argv) != 0:
            raise ValueError("estrazione dei rami fallita")
    SD.check_source(commit)
    if sha256_file(ckpt) != prov["checkpoint_sha256"]:
        raise ValueError("checkpoint cambiato durante l'estrazione")
    inv = S2.inventory(folder, task, S2C.BRANCH_COLS)
    if any(v.get("feature_set") != "branches" or v.get("model_dim") != S2C.BRANCH_COLS for v in inv.values()):
        raise ValueError("cache dei rami non registrata come 'branches'")
    if not S2._paired(meta["cache"], inv):
        raise ValueError(f"{task}/{teacher}/{arm}: uscite dei rami non sulle stesse finestre, etichette e soggetti della cache raw")
    (folder / "provenance.json").write_text(json.dumps({"identity": cache_identity(commit, task, teacher, arm), "run": prov, "cache": inv},
                                                       indent=2, allow_nan=False))


# --- confronti e lettura (CPU) ----------------------------------------------------------------------------------------------------------

DIAG_SETS = ("PW0", "PW", "RPW0", "RPW")  # dalla diagnostica (rami del 2C)
PILOT_SETS = tuple(n for m in MODELS for n in (m, f"R{m}"))
PRIMARY = ("RPWf", "RPWm")
PRIMARY_ARMS = ("PWf", "PWm")
BESIDE = ("RPW0", "RPWm")
DESCRIPTIVE = {  # nome -> (da, a)
    "PWf_PWm_su_PW": ("PWf", "PWm"), "PW_PWm_su_PW": ("PW", "PWm"), "PW_PWm_su_RPW": ("RPW", "RPWm"),
    "PW0_PWm_su_PW": ("PW0", "PWm"), "PW_PWl_su_PW": ("PW", "PWl"), "PW_PWl_su_RPW": ("RPW", "RPWl"),
    "PW_PWf_su_PW": ("PW", "PWf"), "PW_PWf_su_RPW": ("RPW", "RPWf"), "PW0_PWh_su_PW": ("PW0", "PWh"),
    "PW0_PWh_su_RPW": ("RPW0", "RPWh"), "PWh_PW_su_PW": ("PWh", "PW"), "PWh_PW_su_RPW": ("RPWh", "RPW")}
PROBE_KEYS = ("splits_sha256", "k_folds", "fold_seed", "val_frac", "probe_seed", "window_s")


def _report(args, task: str, teacher: str, name: str) -> dict:
    SDG = _script("spectral_diagnostic")
    diag = name in DIAG_SETS
    path = (args.diagnostic_dir if diag else args.reports_dir) / task / teacher / f"{name}.json"
    r = json.loads(path.read_text())
    protocol = SDG.PROTOCOL if diag else PROTOCOL
    if (r.get("protocol"), r.get("task"), r.get("arm"), r.get("set")) != (protocol, task, teacher, name):
        raise ValueError(f"{path}: rapporto di un altro protocollo, compito, teacher o insieme")
    recipe = r.get("extraction_recipe") or {}
    if diag and recipe.get("branches") != SDG.STEP2C:
        raise ValueError(f"{path}: rami che non sono quelli del 2C")
    if not diag and (recipe.get("branches") or {}).get("protocol") != PROTOCOL:
        raise ValueError(f"{path}: rami che non sono quelli del pilot")
    if recipe.get("raw") != SDG.DOWNSTREAM or recipe.get("split_sha256") != SDG.SPLIT_SHA:
        raise ValueError(f"{path}: cache raw o split diversi")
    r["_path"] = str(path)
    r["_sha256"] = sha256_file(path)
    return r


def _normalized(r: dict) -> dict:
    """Le sonde della diagnostica e del pilot hanno ricetta e protocollo propri (le sorgenti dei rami sono diverse, controllate in `_report`):
    per l'accoppiamento di `arms_compare.s_comparison` contano lo split, i fold, la sonda e le righe, che restano quelli del rapporto."""
    return {**r, "protocol": "rami_pilota_o_diagnostica", "feature_set": "cache", "extraction_recipe": {"raw": r["extraction_recipe"]["raw"]}}


def _raw_files(r: dict) -> dict:
    return {d: (v.get("files") or {}).get("raw", {}).get("sha256") for d, v in r["datasets"].items()}


def paired(reps: dict, task: str, ref: str, var: str) -> dict:
    """Confronto accoppiato da `ref` a `var` coi due teacher (`arms_compare.s_comparison`), dopo i controlli dei rapporti: stessa sonda, e per
    gli insiemi col raw lo stesso file raw (SHA) per teacher."""
    AC, SDG = _script("arms_compare"), _script("spectral_diagnostic")
    a = [reps[task][t][ref] for t in TEACHERS]
    b = [reps[task][t][var] for t in TEACHERS]
    if len({json.dumps({k: r.get(k) for k in PROBE_KEYS}, sort_keys=True) for r in a + b}) != 1:
        raise ValueError(f"{task}: {ref} e {var} con split, fold o sonda diversi")
    for ra, rb in zip(a, b):
        fa, fb = _raw_files(ra), _raw_files(rb)
        if all(fa.values()) and all(fb.values()) and fa != fb:
            raise ValueError(f"{task}/{ra['arm']}: {ref} e {var} da file raw diversi")
    row = AC.s_comparison([_normalized(r) for r in a], [_normalized(r) for r in b])
    return {"from": ref, "to": var, **row, "reading": SDG.reading(row)}


def primary_reading(rows: dict) -> dict:
    """Positivo = «migliora» in almeno un compito e delta >= -0,01 sulla stima puntuale nell'altro."""
    tasks = ("p1", "p2")
    positive = any(rows[t]["reading"] == "migliora" and rows[o]["delta"] >= HARM_DELTA for t, o in (tasks, tasks[::-1]))
    return {"positive": bool(positive), "readings": {t: rows[t]["reading"] for t in tasks}, "deltas": {t: rows[t]["delta"] for t in tasks},
            "harm_delta": HARM_DELTA}


COLLAPSE_STOP = "allarme di collasso dei rami"  # l'inizio del motivo dello STOP scritto dal trainer (`branch_masking.CollapseAlarm`)


def run_state(args, arm: str, teacher: str) -> str:
    """«collassato» se il run e' stato fermato dall'allarme di collasso dei rami (collassato = non batte il riferimento, protocollo), se no
    «valido» per un run completo coi controlli passati; altrimenti errore (un run fallito per altro non ha esito)."""
    folder = _run_dir(args, arm, teacher)
    stop = folder / "STOP"
    if stop.exists() and stop.read_text().startswith(COLLAPSE_STOP):
        return "collassato"
    run_summary(args, arm, teacher)
    return "valido"


def run_summary(args, arm: str, teacher: str) -> dict:
    folder = _run_dir(args, arm, teacher)
    prov = json.loads((folder / "run_provenance.json").read_text())
    if (prov.get("protocol"), prov.get("arm"), prov.get("teacher"), prov.get("complete"), prov.get("checks_passed")) != \
            (PROTOCOL, arm, teacher, True, True) or (folder / "STOP").exists():
        raise ValueError(f"{folder}: run del pilot non valido")
    out = {"source_commit": prov["source_commit"], "checkpoint_sha256": prov["checkpoint_sha256"],
           "token_distance": (prov.get("checks") or {}).get("token_distance")}
    if arm != "PWh":
        summary = json.loads((folder / "summary.json").read_text())
        monitor = (summary.get("spectral_branches") or {}).get("collapse_monitor") or {}
        out["collapse_monitor_verified"] = bool(monitor.get("verified"))
        out["collapse_alarm"] = monitor.get("alarm")
        if arm == "PWm":
            out["learned"] = learned(_records(folder / "metrics.jsonl"))
    return out


NOTE = ("Validation interna (soggetti di pretraining P1, soggetti fuori dai test P2), nessun numero dei test ufficiali. Un seme per teacher, due "
        "teacher della famiglia B: incertezza condizionata a questi modelli; piu' confronti senza correzione (il primario decide, gli altri "
        "descrivono); asimmetria delle bande wavelet lente (poco contesto: i loro pesi ricevono poco gradiente nel PWm); le perdite dei due "
        "obiettivi non si confrontano; nessuna conclusione causale ne' adozione automatica.")


def _write(args, result: dict) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(json.dumps({k: result.get(k) for k in ("gate", "collapsed")} | {"primary": result["primary"]["reading"]}), flush=True)


def compare(args) -> None:
    SD.check_source(args.expected_commit)
    arms = ("PWf", "PWm") if args.primary_only else MODELS
    states = {a: {te: run_state(args, a, te) for te in TEACHERS} for a in arms}
    collapsed = sorted(f"{a}/{te}" for a, v in states.items() for te, s in v.items() if s == "collassato")
    base = {"protocol": PROTOCOL, "compare_commit": args.expected_commit, "states": states, "collapsed": collapsed, "note": NOTE}
    if any(c.split("/")[0] in PRIMARY_ARMS for c in collapsed):  # la catena si ferma: questo esito si calcola a mano (protocollo)
        why = f"braccio collassato ({', '.join(collapsed)}): collassato = non batte il riferimento"
        _write(args, {**base, "primary": {"from": PRIMARY[0], "to": PRIMARY[1], "readable": False,
                                          "reading": {"positive": False, "reason": why}}})
        return
    alive = [a for a in arms if all(s == "valido" for s in states[a].values())]
    sets = (("RPW0", "RPWf", "RPWm") if args.primary_only else
            DIAG_SETS + tuple(n for n in PILOT_SETS if (n[1:] if n.startswith("R") else n) in alive))
    reps = {t: {te: {n: _report(args, t, te, n) for n in sets} for te in TEACHERS} for t in ("p1", "p2")}
    runs = {a: {te: run_summary(args, a, te) for te in TEACHERS} for a in alive}
    commits = set()
    for t, v in reps.items():
        for te, w in v.items():
            for n, r in w.items():
                if n in DIAG_SETS:
                    continue
                b = r["extraction_recipe"]["branches"]
                model = n[1:] if n.startswith("R") else n
                if b.get("checkpoint_sha256") != runs[model][te]["checkpoint_sha256"] or b.get("device") != EXTRACT_DEVICE:
                    raise ValueError(f"{t}/{te}/{n}: uscite dei rami non dal checkpoint del run {model}/{te} o non in CPU")
                commits.add(b.get("source_commit"))
    if len(commits) != 1:
        raise ValueError(f"sonde del pilot da codici diversi: {sorted(map(str, commits))}")
    if {r["source_commit"] for a in runs.values() for r in a.values()} != commits:
        raise ValueError("run e uscite dei rami del pilot da codici diversi")
    gate = {"PWm_learned": all(runs["PWm"][te]["learned"]["learned"] for te in TEACHERS),
            "PWm_learned_verifiable": all(runs["PWm"][te]["learned"]["verifiable"] for te in TEACHERS),
            "PWf_not_collapsed": all(runs["PWf"][te]["collapse_monitor_verified"] for te in TEACHERS),
            "PWm_not_collapsed": all(runs["PWm"][te]["collapse_monitor_verified"] for te in TEACHERS)}
    rows = {t: paired(reps, t, *PRIMARY) for t in ("p1", "p2")}
    readable = gate["PWm_learned"] and gate["PWf_not_collapsed"] and gate["PWm_not_collapsed"]
    result = {**base, "source_commit": commits.pop(), "gate": gate,
              "primary": {"from": PRIMARY[0], "to": PRIMARY[1], "readable": readable,
                          "reading": primary_reading(rows) if readable else "non si legge: PWm non ha imparato o un braccio non e' verificato",
                          "paired": rows},
              "beside_PW0_PWm_su_RPW": {t: paired(reps, t, *BESIDE) for t in ("p1", "p2")}, "runs": runs}
    if not args.primary_only:
        have = set(sets)
        result["descriptive"] = {k: {t: paired(reps, t, a, b) for t in ("p1", "p2")} for k, (a, b) in DESCRIPTIVE.items()
                                 if a in have and b in have}
        result["descriptive_skipped"] = sorted(k for k, (a, b) in DESCRIPTIVE.items() if a not in have or b not in have)
    result["report_sha256"] = {f"{t}/{te}/{n}": r["_sha256"] for t, v in reps.items() for te, w in v.items() for n, r in w.items()}
    _write(args, result)


# --- piano (comandi dei training e catena di valutazione) -------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--stage", choices=["train", "hybrid", "extract", "compare", "plan"], required=True)
    ap.add_argument("--arm", choices=list(MODELS))
    ap.add_argument("--teacher", choices=list(TEACHERS))
    ap.add_argument("--task", choices=["p1", "p2"])
    ap.add_argument("--collaudo", action="store_true")
    ap.add_argument("--expected-commit")
    ap.add_argument("--runs-root", type=Path, help="runs dell'A/B dello Step 1 (teacher d'origine, sola lettura)")
    ap.add_argument("--step2c-runs", type=Path, help="runs del 2C (passo 0 e PW, sola lettura)")
    ap.add_argument("--pilot-runs", type=Path, help="runs di questo protocollo")
    ap.add_argument("--downstream-root", type=Path, help="tranche step1_downstream_4000_v1 (sola lettura)")
    ap.add_argument("--eval-root", type=Path, help="uscite dei rami del pilot")
    ap.add_argument("--diagnostic-dir", type=Path, help="compare: rapporti della diagnostica (PW0, PW, RPW0, RPW)")
    ap.add_argument("--reports-dir", type=Path, help="compare: rapporti delle sonde del pilot")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--primary-only", action="store_true", help="compare: solo il primario e il confronto accanto")
    ap.add_argument("--root", type=Path, action="append")
    ap.add_argument("--epn-root", type=Path, action="append")
    ap.add_argument("--uci-root", type=Path)
    for name in S2C.INPUTS:
        ap.add_argument("--" + name.replace("_", "-"), type=Path)
    ap.add_argument("--time-limit-s", type=float)
    ap.add_argument("--device", default="cuda", help="train")
    ap.add_argument("--num-workers", type=int, default=S2C.NUM_WORKERS)
    q = ap.add_argument_group("plan")
    q.add_argument("--code-dir", help="checkout di campagna (cluster)")
    q.add_argument("--expected-sha")
    q.add_argument("--pilot-dir", help="cartella del pilot sul cluster (runs/, cache/, sonde/)")
    q.add_argument("--campaign")
    q.add_argument("--commands", type=Path, help="file dei comandi del collaudo e dei training")
    q.add_argument("--run-time", default="01:30:00", help="allocazione dei run (dal verdetto del collaudo)")
    q.add_argument("--eval-queue", choices=["serial", "gpu"], default="serial")
    q.add_argument("--max-serial", type=int, default=2)
    q.add_argument("--max-gpu", type=int, default=8)
    args = ap.parse_args(argv)
    need = {"train": ("arm", "teacher", "expected_commit", "runs_root", "step2c_runs", "pilot_runs", "time_limit_s", "root", *S2C.INPUTS),
            "hybrid": ("teacher", "expected_commit", "step2c_runs", "pilot_runs"),
            "extract": ("arm", "teacher", "task", "expected_commit", "pilot_runs", "downstream_root", "eval_root"),
            "compare": ("expected_commit", "diagnostic_dir", "reports_dir", "pilot_runs", "out"),
            "plan": ("code_dir", "expected_sha", "pilot_dir", "campaign", "out", "commands")}[args.stage]
    missing = [k for k in need if not getattr(args, k)]
    if missing:
        ap.error(f"{args.stage}: mancano {['--' + k.replace('_', '-') for k in missing]}")
    if args.stage == "train" and args.arm not in ARMS:
        ap.error("train: --arm PWl, PWf o PWm (PWh si costruisce con hybrid)")
    {"train": train, "hybrid": hybrid, "extract": extract, "compare": compare, "plan": plan}[args.stage](args)
    return 0


NICE = 10000  # come la diagnostica: abbassa la priorita' fra i job in attesa della coda seriale
EXTRACT_TIME = {"p1": "01:30:00", "p2": "03:00:00"}
PROBE_TIME = {"p1": {"branches": "04:00:00", "raw": "03:00:00"}, "p2": {"branches": "03:00:00", "raw": "02:00:00"}}  # come SET_TIME
DIAG_COMPARE = "RAMI-rami-diag-20261021b-compare"  # il confronto della diagnostica: dopo tutte le sue sonde (PW0, PW, RPW0, RPW dei teacher B)


def _opt(node: dict, name: str) -> str | None:
    return next((o.split("=", 1)[1] for o in node["sbatch"] if o.startswith(name + "=")), None)


def _hours(t: str) -> float:
    h, m, s = (int(x) for x in t.split(":"))
    return round(h + m / 60 + s / 3600, 2)


def plan(args) -> None:
    """Comandi del collaudo e dei 6 training (submit_job.py) e piano della catena di valutazione (schema eval-plan-v0): prima il primario
    (uscite di PWf e PWm, sonde RPWf e RPWm, confronto del solo primario), poi ibridi, PWl e PWh, gli insiemi dei soli rami e il confronto
    completo. `--eval-queue gpu`: estrazioni e sonde su un nodo GPU (piu' job insieme; GPU-ora = tempo massimo), sempre in CPU (`EXTRACT_DEVICE`)
    come il 2C: la GPU resta inattiva, si compra solo il parallelismo. Sopra il residuo del budget del passo serve una decisione registrata."""
    pilot, c, code, sha = args.pilot_dir, args.campaign, args.code_dir, args.expected_sha
    if not pilot.startswith("/") or not code.startswith("/"):
        raise ValueError("--pilot-dir e --code-dir assolute (cluster)")
    if not c or not all(ch.isalnum() or ch in "-_" for ch in c):
        raise ValueError("--campaign: solo lettere, cifre, - e _")
    export = f"--export=ALL,CODE_DIR={code},EXPECTED_SHA={sha}"
    train_id = {(a, t): f"RAMIPILOT-{c}-{a}-{t}" for a in ARMS for t in TEACHERS}

    def submit(job_id, gpu_h, qos, time, *rest):
        return " ".join(["python3 scripts/submit_job.py --id", job_id, "--agente claude --campagna", c, "--gpu-ore", f"{gpu_h:g}", "--",
                         export, f"--qos={qos}", f"--time={time}", f"{code}/scripts/slurm/branch_pilot_train.sbatch", pilot, *rest])

    commands = {"collaudo": submit(f"RAMIPILOT-{c}-collaudo", 0.5, "boost_qos_dbg", "00:30:00", "PWm", "B_seed0", "collaudo"),
                "training": [submit(train_id[a, t], _hours(args.run_time), "boost_qos_lprod", args.run_time, a, t) for a in ARMS for t in TEACHERS]}
    gpu = args.eval_queue == "gpu"
    queue = "boost_usr_prod" if gpu else "lrd_all_serial"
    where = ["--partition=boost_usr_prod", "--qos=normal", "--gres=gpu:1"] if gpu else [f"--nice={NICE}"]
    nodes, by_id = [], {}

    def node(i, kind, after, launcher, argv, time, outputs, inputs=(), exports=None, cpus=None):
        n = {"id": i, "kind": kind, "after": list(after), "queue": queue, "gpu_hours": _hours(time) if gpu else 0, "code_dir": code,
             "expected_sha": sha, "launcher": launcher, "args": argv, "exports": exports or {},
             "sbatch": [f"--time={time}", *where, *([f"--cpus-per-task={cpus}"] if cpus else [])], "inputs": list(inputs),
             "outputs": list(outputs)}
        nodes.append(n)
        by_id[i] = n
        return n

    def ext(task, arm, t):
        i = f"RAMIPILOT-{c}-{arm}-{t}-{task}-ext"
        if i not in by_id:
            after = [f"RAMIPILOT-{c}-PWh-{t}"] if arm == "PWh" else [train_id[arm, t]]
            node(i, "extract", after, "scripts/slurm/branch_pilot_cpu.sbatch",
                 [pilot, "extract", "--task", task, "--arm", arm, "--teacher", t],
                 EXTRACT_TIME[task], [f"{pilot}/cache/{task}/{t}/branches_{arm}/provenance.json"],
                 inputs=[f"{pilot}/runs/PWh/{t}/run_provenance.json"] if arm == "PWh" else ())
        return i

    def probe(task, t, name):
        arm = name[1:] if name.startswith("R") else name
        e = ext(task, arm, t)
        i = f"RAMIPILOT-{c}-{t}-{task}-{name}"
        node(i, "probe", [e], "scripts/slurm/spectral_diagnostic.sbatch", [f"{pilot}/sonde", "probe", "--task", task, "--arm", t, "--sets", name],
             PROBE_TIME[task]["raw" if name.startswith("R") else "branches"], [f"{pilot}/sonde/{task}/{t}/{name}.json"],
             inputs=by_id[e]["outputs"], exports={"PILOT_ROOT": f"{pilot}/cache"})
        return i

    def compare_node(i, probes, primary_only):
        node(i, "compare", probes + [DIAG_COMPARE], "scripts/slurm/branch_pilot_cpu.sbatch",
             [pilot, "compare", *(["--primary-only"] if primary_only else [])], "00:30:00",
             [f"{pilot}/esito_primario.json" if primary_only else f"{pilot}/esito.json"],
             inputs=[o for p in probes for o in by_id[p]["outputs"]], cpus=2)

    primary = [probe(task, t, name) for name in PRIMARY for task in ("p1", "p2") for t in TEACHERS]
    compare_node(f"RAMIPILOT-{c}-compare-primario", primary, True)
    for t in TEACHERS:
        node(f"RAMIPILOT-{c}-PWh-{t}", "extract", [], "scripts/slurm/branch_pilot_cpu.sbatch", [pilot, "hybrid", "--teacher", t], "00:30:00",
             [f"{pilot}/runs/PWh/{t}/run_provenance.json"], cpus=2)
    rest = [probe(task, t, name) for name in PILOT_SETS if name not in PRIMARY for task in ("p1", "p2") for t in TEACHERS]
    compare_node(f"RAMIPILOT-{c}-compare", primary + rest, False)
    doc = {"schema": "eval-plan-v0", "campaign": c, "agent": "claude",
           "limits": {"boost_usr_prod": args.max_gpu} if gpu else {"lrd_all_serial": args.max_serial}, "nodes": nodes,
           "estimate": {"eval_queue": args.eval_queue, "gpu_hours_max": round(sum(n["gpu_hours"] for n in nodes), 2),
                        "cpu_core_hours_max": round(sum(_hours(_opt(n, "--time")) * int(_opt(n, "--cpus-per-task") or 8) for n in nodes), 2),
                        "jobs": len(nodes)}}
    _script("evaluation_chain").validate(doc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    args.commands.write_text(commands["collaudo"] + "\n\n" + "\n".join(commands["training"]) + "\n")
    print(f"{len(nodes)} nodi ({args.eval_queue}) -> {args.out}; comandi del collaudo e dei 6 training -> {args.commands}")


if __name__ == "__main__":
    sys.exit(main())
