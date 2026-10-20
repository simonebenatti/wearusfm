#!/usr/bin/env python3
"""Ammissibilita' e selezione della calibrazione C1-C4 della finestra 1, con le regole congelate in `docs/calibrazione_c1_c4_selezione.md`
prima di C1 (accordo GPT-Claude del 15/10/2026, sezione «Calibrazione»). Nessun numero di test P1/P2 e nessuna training loss scelgono un LR:
la selezione usa la sola metrica S della validation interna (`scripts/calibration_probe.py`); il pannello fisso e le diagnostiche del training
decidono solo l'ammissibilita'.

Ammissibilita' di un run (tutte):
  A1 completo: nessuno STOP, `summary.json` con `stopped` = `max_steps` e i passi pianificati, il rapporto della sonda viene dal suo
     `checkpoint_postdecay.pt`;
  A2 numerico: ogni valore numerico di `metrics.jsonl` e di `fixed_eval.jsonl` finito;
  A3 non-collasso (soglie firmate, decisione 16, `training.run.AlarmRule`): nessun allarme nel run e, all'ultima valutazione (passo finale),
     rapporto di collasso min(studente, teacher) >= 0,05 e rango effettivo dello studente >= 10% della dimensione;
  A4 stabilita' sul pannello fisso (`fixed_eval.jsonl`, ogni `PANEL_EVERY` passi dal passo 0, senza buchi): mai `PANEL_K` valutazioni
     consecutive con L_t > (1 + `PANEL_DELTA`) * min_{s<=t} L_s, e L all'ultima valutazione <= `PANEL_LEARN` * L_0 (L = loss totale).
Selezione (S fuori fold; confronto accoppiato sui soggetti: differenza entro 1 errore standard = parita', vince il LR piu' basso):
  C1 (1e-4, 3e-4, 1e-3, seme 0): se 3e-4 e' ammissibile, finalisti 3e-4 e la migliore alternativa ammissibile; se 3e-4 non lo e', le due
     alternative solo se entrambe ammissibili; altrimenti stop (meno di due candidati).
  C2 (i finalisti, seme 1): confermati = quelli ammissibili; se uno manca il pianificatore ferma C3 e C4.
  C3 (100M, i due confermati): se entrambi ammissibili, LR proposto per il 300M = il migliore; meno di due ammissibili -> stop (accordo del
     15/10: «Se mancano due candidati ammissibili [...] interrompere il passaggio alla fase successiva»; continuare con uno solo richiede una
     firma di Simone e un cambio qui).
  C4 (300M, il LR proposto): superata se ammissibile.
  collaudo (solo collaudi della catena, mai nel file delle decisioni): candidati con un'etichetta al posto del LR; la decisione elenca gli
     ammissibili, nessuna scelta.
Con due o piu' candidati il registro riporta anche i confronti accoppiati di tutte le coppie, descrittivi: non decidono nulla.

Identita' dei candidati (review di GPT e del critico del 16/10/2026): nelle fasi C1-C4 `--plan` (il piano di training di
`calibration_plan.py`, quello dato a `wsd_chain.py`) e' obbligatorio; per ogni candidato l'INTERA configurazione effettiva del run
(`config.json`) deve essere quella che `train_jepa.build_config` costruisce dagli argomenti del job pianificato, lo SHA del codice
(`summary.json`) quello del piano e il LR dichiarato quello del piano: altrimenti la selezione si ferma con un errore (il run non e' il
candidato pianificato). Un candidato il cui training si e' fermato prima della sonda (STOP, allarme, run incompleto, anche senza
`config.json`) si passa col rapporto «-»: e' inammissibile (A1-A4 sui file del run) e la fase procede con gli altri; un run ammissibile
senza sonda e' un errore.

  python3 scripts/calibration_select.py select --phase C1 --plan calibrazione_C1.json --out decisione_C1.json --record registro_C1.json \\
      --candidate 0.0001 RUN_DIR RAPPORTO_SONDA PASSI --candidate 0.0003 ... --candidate 0.001 RUN_DIR - PASSI   (sul cluster, CPU)
  python3 scripts/calibration_select.py merge --decisions decisioni.json --add decisione_C1.json          (sul Mac: unisce e controlla)
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import shlex
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROTOCOL = "w1-calibrazione-selezione-v1"
GRID_C1 = (1e-4, 3e-4, 1e-3)
REFERENCE_LR = 3e-4
PANEL_EVERY = 500
PANEL_DELTA = 0.10
PANEL_K = 3
PANEL_LEARN = 0.9
N_BOOT = 1000
BOOT_SEED = 0
CHECKPOINT = "checkpoint_postdecay.pt"
PHASES = ("C1", "C2", "C3", "C4")
P1_DATASETS = ("ninapro_db2", "ninapro_db3", "ninapro_db6")  # S della P1 (scripts/calibration_probe.py, compito p1)
COLLAUDO = "collaudo"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _records(path: Path) -> list[dict]:
    """Le righe JSON per passo (l'ultima vince: una ripresa puo' ripetere i passi dopo l'ultimo checkpoint), in ordine di passo."""
    by_step = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_step[int(r["step"])] = r
    return [by_step[k] for k in sorted(by_step)]


def _non_finite(records: list[dict]) -> list[int]:
    bad = []
    for r in records:
        for v in r.values():
            vals = v if isinstance(v, list) else [v]
            if any(isinstance(x, float) and not math.isfinite(x) for x in vals):
                bad.append(int(r["step"]))
                break
    return bad


def alarm_rule():
    from wearusfm.training.run import AlarmRule

    return AlarmRule()


def panel_check(panel: list[dict], total_steps: int) -> dict:
    """A4 sulla loss totale del pannello fisso."""
    steps = [int(r["step"]) for r in panel]
    expected = list(range(0, total_steps + 1, PANEL_EVERY))
    out = {"evaluations": len(steps), "problems": []}
    if steps != expected:
        out["problems"].append(f"pannello: valutazioni ai passi {steps[:3]}...{steps[-3:]} invece di 0, {PANEL_EVERY}, ..., {expected[-1]}")
        return out
    loss = np.array([float(r["total"]) for r in panel])
    run_min = np.minimum.accumulate(loss)
    above = loss > (1 + PANEL_DELTA) * run_min
    longest, cur, where = 0, 0, None
    for i, a in enumerate(above):
        cur = cur + 1 if a else 0
        if cur > longest:
            longest, where = cur, steps[i]
    out.update({"L0": float(loss[0]), "L_last": float(loss[-1]), "ratio_last_to_first": float(loss[-1] / loss[0]),
                "max_ratio_to_running_min": float((loss / run_min).max()), "longest_run_above": longest})
    if longest >= PANEL_K:
        out["problems"].append(f"pannello: {longest} valutazioni consecutive oltre {1 + PANEL_DELTA:g} volte il minimo (fino al passo {where})")
    if not loss[-1] <= PANEL_LEARN * loss[0]:
        out["problems"].append(f"pannello: L finale {loss[-1]:.4g} oltre {PANEL_LEARN:g} x L0 ({loss[0]:.4g}): nessun apprendimento sufficiente")
    return out


def same_file(a, b) -> bool:
    """Lo stesso file: percorsi uguali dopo la normalizzazione (barre doppie o finali, `.`), oppure, se esistono entrambi, lo stesso file su
    disco (link simbolici, `$WORK` contro `/leonardo_work/...`)."""
    if a is None or b is None:
        return False
    a, b = os.path.normpath(str(a)), os.path.normpath(str(b))
    if a == b:
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


@functools.lru_cache(maxsize=None)
def _script(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"{name}_for_selection", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expected_config(plan: dict, run_id: str) -> dict:
    """La configurazione COMPLETA (`training.run.config_to_dict`) che il run `run_id` del piano di training deve avere: gli argomenti che il
    job passa a `train_jepa.py` (`train_ddp.sbatch`: preset, `--max-steps`, EXTRA_ARGS da `wsd_chain.extra_args`), costruiti con
    `train_jepa.build_config`, la stessa funzione del training (review del critico del 16/10/2026: confronto dell'intera configurazione)."""
    from wearusfm.training import run as R

    W, TJ = _script("wsd_chain"), _script("train_jepa")
    runs = {r["id"]: r for r in W.resolve(plan)}
    if run_id not in runs:
        raise ValueError(f"run {run_id} assente dal piano di training (run del piano: {sorted(runs)})")
    r = runs[run_id]
    argv = ["--preset", r["preset"], "--manifest", "-", "--root", "-", "--out-dir", "-", "--max-steps", str(r["total_steps"])]
    args = TJ.build_parser().parse_args(argv + shlex.split(W.extra_args(plan, r)))
    return R.config_to_dict(TJ.build_config(args))


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def check_identity(run_dir: Path, plan: dict, lr: float, allow_missing: bool = False) -> dict:
    """Il run e' il candidato pianificato: configurazione effettiva (`config.json`, tutti i campi) e codice (`summary.json`) uguali al piano,
    LR dichiarato uguale a quello del piano. Errore altrimenti (review di GPT e del critico del 16/10/2026). `allow_missing`: un candidato
    fermato prima della sonda che non ha scritto `config.json` ha l'identita' «non verificabile» e decide A1 (inammissibile)."""
    run_dir = Path(run_dir)
    want = expected_config(plan, run_dir.name)
    if lr != want["lr"]:
        raise ValueError(f"{run_dir.name}: LR dichiarato {lr!r}, il piano dice {want['lr']!r}")
    if not (run_dir / "config.json").exists():
        if allow_missing:
            return {"config_checked": False, "note": "config.json assente: run fermato prima di scriverla, identita' non verificabile; decide A1"}
        raise ValueError(f"{run_dir}: config.json assente, configurazione effettiva non verificabile")
    have, exp = _flatten(json.loads((run_dir / "config.json").read_text())), _flatten(want)
    diff = [f"{k}: run {have.get(k, '<assente>')!r}, piano {exp.get(k, '<assente>')!r}" for k in sorted(set(have) | set(exp))
            if have.get(k, "<assente>") != exp.get(k, "<assente>")]
    if diff:
        raise ValueError(f"{run_dir.name}: configurazione effettiva diversa dal piano: " + "; ".join(diff))
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    sha = (summary.get("code") or {}).get("sha")
    if summary and sha != plan["expected_sha"]:
        raise ValueError(f"{run_dir.name}: codice {sha!r}, il piano dice {plan['expected_sha']!r}")
    return {"config_checked": True, "fields": len(exp), "code_sha": sha}


def admissibility(run_dir: Path, report: dict | None, total_steps: int, dim: int | None = None) -> dict:
    """A1-A4 di un run; `report` = rapporto di `calibration_probe.py`, oppure None per un run fermato prima della sonda; `dim` = dimensione
    del modello per la soglia del rango (default: quella del rapporto o di `config.json`)."""
    run_dir = Path(run_dir)
    problems, out = [], {"run_dir": str(run_dir)}
    if (run_dir / "STOP").exists():
        problems.append(f"STOP nel run: {(run_dir / 'STOP').read_text().strip()[:300]}")
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    out["summary"] = {k: summary.get(k) for k in ("stopped", "steps", "elapsed_s", "code")}
    if summary.get("stopped") != "max_steps" or summary.get("steps") != total_steps:
        problems.append(f"run non completo: stopped {summary.get('stopped')!r}, passi {summary.get('steps')} su {total_steps}")
    if report is not None and not same_file(report.get("checkpoint"), run_dir / CHECKPOINT):
        problems.append(f"la sonda viene da {report.get('checkpoint')}, non da {run_dir / CHECKPOINT}")
    ckpt = run_dir / CHECKPOINT
    # con la sonda, il checkpoint deve esserci ed essere quello sondato (review GPT 41ce569); senza sonda la sua assenza non decide nulla:
    # un run altrimenti ammissibile senza sonda resta un errore operativo («la valutazione si completa, non si salta»; critico, 20/10)
    if report is not None and (not ckpt.is_file() or ckpt.stat().st_size == 0):
        problems.append("checkpoint post-decay assente o vuoto")
    elif report is not None and report.get("checkpoint_sha256") != _sha256(ckpt):
        problems.append("SHA-256 del checkpoint diverso da quello della sonda")
    metrics = _records(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
    panel = _records(run_dir / "fixed_eval.jsonl") if (run_dir / "fixed_eval.jsonl").exists() else []
    for name, recs in (("metrics.jsonl", metrics), ("fixed_eval.jsonl", panel)):
        if not recs:
            problems.append(f"{name} assente o vuoto")
        elif bad := _non_finite(recs):
            problems.append(f"{name}: valori non finiti ai passi {bad[:5]}")
    if dim is None:
        if report is not None:
            dim = int(report["model_dim"])
        elif (run_dir / "config.json").exists():
            dim = int(json.loads((run_dir / "config.json").read_text())["model"]["dim"])
        else:
            dim = 0
            problems.append("config.json assente: dimensione del modello ignota")
    rule = alarm_rule()
    diags = [r for r in metrics if "student_collapse_ratio" in r]
    alarms = [r for r in metrics if r.get("alarm")]
    out["non_collapse"] = {"evaluations": len(diags), "alarms": [r["alarm"] for r in alarms], "ratio_min": rule.collapse_ratio_min,
                           "erank_min": rule.erank_fraction_min * dim}
    if alarms:
        problems.append(f"allarme di collasso: {alarms[0]['alarm']}")
    if not diags or int(diags[-1]["step"]) != total_steps:
        problems.append("manca la diagnostica di collasso al passo finale")
    else:
        last = diags[-1]
        ratio = min(last["student_collapse_ratio"], last["teacher_collapse_ratio"])
        out["non_collapse"].update({"last_ratio": ratio, "last_student_erank": last["student_erank"]})
        if not ratio >= rule.collapse_ratio_min:
            problems.append(f"rapporto di collasso finale {ratio:.3g} sotto {rule.collapse_ratio_min}")
        if not last["student_erank"] >= rule.erank_fraction_min * dim:
            problems.append(f"rango effettivo finale {last['student_erank']:.1f} sotto {rule.erank_fraction_min:.0%} di {dim}")
    if panel and not _non_finite(panel):
        out["panel"] = panel_check(panel, total_steps)
        problems += out["panel"]["problems"]
    out["problems"], out["admissible"] = problems, not problems
    return out


def load_oof(report: dict) -> dict:
    """{dataset: (y, pred, soggetti)} dalle predizioni fuori fold, verificate per SHA-256."""
    path = Path(report["oof"])
    if _sha256(path) != report["oof_sha256"]:
        raise ValueError(f"{path}: SHA-256 diverso da quello del rapporto")
    from wearusfm.harness.downstream_ab import fingerprint

    with np.load(path) as z:
        datasets = z["datasets"].tolist()
        if sorted(datasets) != sorted(report["datasets"]) or len(set(datasets)) != len(datasets):
            raise ValueError("dataset OOF diversi dal rapporto")
        out = {}
        for d in datasets:
            y, pred, subjects, rows = (z[f"{d}_{k}"] for k in ("y", "pred", "s", "row"))
            if (y.ndim != 1 or not (y.shape == pred.shape == subjects.shape == rows.shape) or not len(y)
                    or rows.dtype.kind not in "iu" or np.any(rows < 0) or len(np.unique(rows)) != len(rows)):
                raise ValueError(f"{d}: righe OOF non valide")
            if fingerprint(rows) != report["datasets"][d].get("oof_rows_sha256"):
                raise ValueError(f"{d}: identita' delle righe OOF diversa dal rapporto")
            if len(np.unique(y)) < 2:
                raise ValueError(f"{d}: meno di due classi OOF")
            out[d] = (y, pred, subjects)
        return out


def check_pairing(reports: list[dict]) -> None:
    """Valida la provenienza prima di QUALUNQUE decisione, anche senza bootstrap."""
    keys = ("protocol", "splits_sha256", "k_folds", "fold_seed", "val_frac", "probe_seed", "feature_set", "window_s", "extraction_recipe")
    reference = reports[0]
    for report in reports:
        if any(k not in report or report[k] != reference.get(k) for k in keys):
            raise ValueError("provenienza non accoppiata: split, fold o ricetta della sonda diversi/mancanti")
        if sorted(report["datasets"]) != sorted(reference["datasets"]):
            raise ValueError("dataset diversi fra i candidati")
        for ds, fields in report["datasets"].items():
            for key in ("input_sha256", "oof_rows_sha256"):
                value = fields.get(key)
                if not value or value != reference["datasets"][ds].get(key):
                    raise ValueError(f"{ds}: provenienza non accoppiata ({key})")


def paired_difference(a: dict, b: dict, weights: dict, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """S(a) - S(b) e il suo errore standard: bootstrap dei soggetti per dataset, lo stesso campione per i due candidati (stesse finestre, stesse
    etichette, stessi soggetti: controllato)."""
    from sklearn.metrics import balanced_accuracy_score

    if sorted(a) != sorted(b) or sorted(a) != sorted(weights):
        raise ValueError("dataset diversi fra i candidati")
    prepared = []
    for d in sorted(a):
        (ya, pa, sa), (yb, pb, sb) = a[d], b[d]
        if not (np.array_equal(ya, yb) and np.array_equal(sa, sb)):
            raise ValueError(f"{d}: finestre o soggetti diversi fra i candidati: il confronto non e' accoppiato")
        subs = np.unique(sa)
        prepared.append((weights[d], ya, pa, pb, subs, {k: np.flatnonzero(sa == k) for k in subs}))
    point = sum(w * (balanced_accuracy_score(y, pa) - balanced_accuracy_score(y, pb)) for w, y, pa, pb, _, _ in prepared)
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n_boot):
        diff, ok = 0.0, True
        for w, y, pa, pb, subs, idx_by in prepared:
            idx = np.concatenate([idx_by[k] for k in rng.choice(subs, size=len(subs), replace=True)])
            if len(np.unique(y[idx])) < 2:
                ok = False
                break
            diff += w * (balanced_accuracy_score(y[idx], pa[idx]) - balanced_accuracy_score(y[idx], pb[idx]))
        if ok:
            vals.append(diff)
    if len(vals) < 2:
        raise ValueError("bootstrap accoppiato senza abbastanza repliche valide: nessuna selezione")
    se = float(np.std(vals, ddof=1))
    if not math.isfinite(se) or not math.isfinite(point):
        raise ValueError("confronto accoppiato non finito: nessuna selezione")
    return {"delta": float(point), "se": se, "tie": not abs(point) > se}


def better(lr_a: float, lr_b: float, cands: dict) -> tuple[float, dict]:
    """Il migliore dei due LR per S, con la parita' (entro 1 SE accoppiato) al LR piu' basso."""
    cmp = paired_difference(cands[lr_a]["oof"], cands[lr_b]["oof"], cands[lr_a]["report"]["weights"])
    if cmp["tie"]:
        win = min(lr_a, lr_b)
    else:
        win = lr_a if cmp["delta"] > 0 else lr_b
    return win, {"a": lr_a, "b": lr_b, **cmp, "winner": win}


def decide(phase: str, cands: dict) -> tuple[dict, list[dict]]:
    """La decisione della fase (formato di `calibration_plan.py`) e i confronti fatti. `cands`: {lr: {"admissible", "report", "oof"}}."""
    ok = sorted(lr for lr, c in cands.items() if c["admissible"])
    comparisons = []
    if phase == "C1":
        if sorted(cands) != sorted(GRID_C1):
            raise ValueError(f"C1: candidati {sorted(cands)}, attesi {GRID_C1}")
        alts = [lr for lr in ok if lr != REFERENCE_LR]
        if REFERENCE_LR in ok and alts:
            best = alts[0]
            if len(alts) == 2:
                best, c = better(alts[0], alts[1], cands)
                comparisons.append(c)
            return {"finalists": sorted([REFERENCE_LR, best])}, comparisons
        if REFERENCE_LR not in ok and len(alts) == 2:
            return {"finalists": alts}, comparisons
        return {"stop": f"meno di due candidati ammissibili secondo le regole di C1 (ammissibili: {ok})"}, comparisons
    if phase == "C2":
        if len(cands) != 2:
            raise ValueError("C2: servono i due finalisti di C1")
        return {"confirmed": ok}, comparisons
    if phase == "C3":
        if len(cands) != 2:
            raise ValueError("C3: servono i due LR confermati")
        if len(ok) < 2:  # accordo del 15/10: meno di due candidati ammissibili ferma il passaggio alla fase successiva
            return {"stop": f"meno di due LR ammissibili al 100M (ammissibili: {ok})"}, comparisons
        best, c = better(ok[0], ok[1], cands)
        comparisons.append(c)
        return {"proposed_300m_lr": best}, comparisons
    if phase == "C4":
        if len(cands) != 1:
            raise ValueError("C4: un solo LR, quello proposto dopo C3")
        (lr,) = cands
        return {"lr": lr, "passed": bool(cands[lr]["admissible"])}, comparisons
    if phase == COLLAUDO:
        return {"admissible": ok}, comparisons
    raise ValueError(f"fase {phase!r}")


def select(phase: str, candidates: list[tuple[str, Path, Path | None, int]], plan: dict | None = None) -> tuple[dict, dict]:
    """`candidates`: (LR come testo, o un'etichetta nel collaudo; cartella del run; rapporto della sonda, None se il training si e' fermato
    prima; passi pianificati). `plan`: il piano di training, obbligatorio nelle fasi C1-C4 (identita' dei candidati)."""
    if phase != COLLAUDO and plan is None:
        raise ValueError(f"{phase}: serve il piano di training (--plan) per verificare la configurazione effettiva dei candidati")
    cands, record = {}, {"protocol": PROTOCOL, "phase": phase, "candidates": {}}
    for key, run_dir, report_path, steps in candidates:
        lr = str(key) if phase == COLLAUDO else float(key)
        if lr in cands:
            raise ValueError(f"LR {lr} ripetuto")
        identity = check_identity(run_dir, plan, lr, allow_missing=report_path is None) if plan is not None else None
        report = json.loads(Path(report_path).read_text()) if report_path is not None else None
        if report is not None and report.get("protocol") != PROTOCOL:
            raise ValueError(f"{report_path}: protocollo {report.get('protocol')!r}, atteso {PROTOCOL}")
        if report is not None and (report.get("task", "p1") != "p1" or sorted(report["weights"]) != sorted(P1_DATASETS)):
            raise ValueError(f"{report_path}: la selezione della calibrazione usa S della P1 (compito {report.get('task')!r}, "
                             f"dataset {sorted(report['weights'])})")
        adm = admissibility(run_dir, report, steps)
        if report is None and adm["admissible"]:
            raise ValueError(f"{Path(run_dir).name}: run ammissibile senza sonda: la valutazione va completata, non saltata")
        cands[lr] = {"admissible": adm["admissible"], "report": report, "oof": load_oof(report) if report is not None else None}
        entry = {"admissibility": adm, "identity": identity, "report": str(report_path) if report_path is not None else None}
        if report is not None:
            entry.update({"S": report["S"], "S_se": report["S_se"], "report_sha256": _sha256(Path(report_path)),
                          "checkpoint_sha256": report["checkpoint_sha256"], "per_dataset_S": {d: v["s"] for d, v in report["datasets"].items()}})
        record["candidates"][lr if isinstance(lr, str) else repr(lr)] = entry
    if not cands:
        raise ValueError("nessun candidato")
    probed = {k: c for k, c in cands.items() if c["report"] is not None}  # un candidato fermato prima della sonda non ha rapporto
    if len({json.dumps(c["report"]["weights"], sort_keys=True) for c in probed.values()}) > 1:
        raise ValueError("pesi dei dataset diversi fra i candidati")
    if probed:
        check_pairing([c["report"] for c in probed.values()])
        reference = next(iter(probed.values()))["oof"]
        for c in probed.values():
            for ds, (y, _, subjects) in c["oof"].items():
                if not np.array_equal(y, reference[ds][0]) or not np.array_equal(subjects, reference[ds][2]):
                    raise ValueError(f"{ds}: etichette o soggetti OOF non accoppiati")
    decision, comparisons = decide(phase, cands)
    keys = sorted(probed)
    descriptive = [{"a": a, "b": b, **paired_difference(cands[a]["oof"], cands[b]["oof"], cands[a]["report"]["weights"])}
                   for i, a in enumerate(keys) for b in keys[i + 1:]]
    record.update({"decision": decision, "comparisons": comparisons, "descriptive_comparisons": descriptive,
                   "rules": {"panel_every": PANEL_EVERY, "panel_delta": PANEL_DELTA, "panel_k": PANEL_K, "panel_learn": PANEL_LEARN,
                             "tie": "differenza accoppiata entro 1 SE: il LR piu' basso", "n_boot": N_BOOT, "boot_seed": BOOT_SEED}})
    return {phase: decision}, record


def merge(decisions: dict, add: dict) -> dict:
    """Aggiunge la decisione di una fase a quelle gia' prese, controllando la sequenza: nessuna fase riscritta, candidati coerenti."""
    import importlib.util

    (phase, dec), = add.items()
    if phase not in PHASES:
        raise ValueError(f"fase {phase!r}: nel file delle decisioni entrano solo {PHASES} (un collaudo no)")
    if phase in decisions:
        raise ValueError(f"{phase} gia' decisa: nessuna riscrittura")
    i = PHASES.index(phase)
    if any(p not in decisions for p in PHASES[:i]):
        raise ValueError(f"{phase}: mancano le decisioni precedenti")
    if any("stop" in decisions[p] for p in PHASES[:i]):
        raise ValueError(f"{phase}: una fase precedente ha deciso stop")
    if i >= 2 and sorted(decisions["C2"].get("confirmed", [])) != sorted(decisions["C1"].get("finalists", [])):
        raise ValueError(f"{phase}: conferma C2 incompleta")
    out = {**decisions, phase: dec}
    spec = importlib.util.spec_from_file_location("calibration_plan", ROOT / "scripts" / "calibration_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.resolve_phases(out)  # stessi controlli del pianificatore
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--phase", choices=PHASES + (COLLAUDO,), required=True)
    s.add_argument("--candidate", nargs=4, action="append", required=True, metavar=("LR", "RUN_DIR", "RAPPORTO", "PASSI"),
                   help="RAPPORTO '-' per un run fermato prima della sonda")
    s.add_argument("--plan", type=Path, default=None, help="piano di training della fase (obbligatorio in C1-C4)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--record", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--decisions", type=Path, required=True)
    m.add_argument("--add", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "select":
        cands = [(lr, Path(rd), None if rp == "-" else Path(rp), int(st)) for lr, rd, rp, st in args.candidate]
        plan = json.loads(args.plan.read_text()) if args.plan is not None else None
        decision, record = select(args.phase, cands, plan)
        if plan is not None:
            record["plan"] = {"path": str(args.plan), "sha256": _sha256(args.plan)}
        for path, obj in ((args.record, record), (args.out, decision)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=1))
        print(json.dumps(decision))
        return 0
    old = json.loads(args.decisions.read_text()) if args.decisions.exists() else {}
    new = merge(old, json.loads(args.add.read_text()))
    args.decisions.write_text(json.dumps(new, indent=1))
    print(json.dumps(new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
