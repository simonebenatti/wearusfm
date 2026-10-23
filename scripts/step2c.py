#!/usr/bin/env python3
"""Step 2C dello step 2 (protocollo `docs/step2c_frozen_branches.md`, ID `step2c_frozen_branches_v1`; Simone, 14/10/2026: «per questa mi
piace la tua proposta, direi di procedere»): rami PSD e wavelet allenati sopra l'encoder raw CONGELATO dei quattro teacher dell'A/B dello
Step 1 (A_seed0, A_seed1, B_seed0, B_seed1: «proverei entrambe le soluzioni»), poi le sonde P1/P2 ufficiali sull'embedding [raw, rami] e il
confronto accoppiato (`harness.downstream_ab.compare_reports`) coi criteri congelati.

Con il raw congelato la parte raw dell'embedding e' quella del teacher d'origine, gia' estratta e verificata nella tranche downstream
(`step1_downstream_4000_v1`, lette e mai riscritte): si estraggono sulla CPU solo le uscite dei rami (insieme di feature "branches", nessun
encode raw) del modello allenato e di quello al passo 0 (`checkpoint_init.pt`, il controllo «rami non allenati»), e si concatenano alle cache
downstream solo se gli ingressi letti (hash) e le righe coincidono, e solo per un run che a fine training ha l'encoder bit per bit uguale a quello
del teacher d'origine. Prima dei quattro training, un collaudo su dbg (B_seed0, al piu' 30 minuti) misura tempi e curva della loss e applica il
verdetto congelato (`collaudo_verdict`). Tutto fuori da Git, in cartelle nuove.

  train    --arm X [--collaudo]              rami sul teacher X congelato (GPU) -> BRANCH_RUNS/X (collaudo: BRANCH_RUNS/collaudo/X)
  extract  --task T --arm X                  uscite dei rami (CPU), allenato e passo 0 -> EVAL/T/X/branches_{trained,init}; varianti RP, RW,
                                             RPW, RPW0 -> EVAL/T/X/<V>
  probe    --task T --arm X --variant V...   sonde ufficiali -> EVAL/T/X/<V>.json
  compare                                    confronti accoppiati e criteri congelati -> EVAL/comparison.json
"""
import argparse
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
import probe_p2  # noqa: E402
import step1_ab as SA  # noqa: E402
import step1_downstream as SD  # noqa: E402
import step2a as S2  # noqa: E402
from spectral_step1 import output_path  # noqa: E402
from wearusfm.harness.downstream_ab import compare_reports  # noqa: E402
from wearusfm.harness.feature_sets import cache_record  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.spectral_calibration import sha256_file  # noqa: E402

PROTOCOL = "step2c_frozen_branches_v1"
ARMS = tuple(SD.CHECKPOINTS)  # A_seed0, A_seed1, B_seed0, B_seed1: i teacher finali dell'A/B dello Step 1
FAMILIES = {"A": ("A_seed0", "A_seed1"), "B": ("B_seed0", "B_seed1")}
KINDS = ("psd", "wavelet")
BRANCH_DIM = 128
BRANCH_COLS = BRANCH_DIM * len(KINDS)
# training dei soli rami (congelato prima di qualunque misura; il collaudo lo conferma o lo ferma, non lo ritocca)
STEPS, LR, WARMUP, WEIGHT_DECAY, GRAD_CLIP = 3000, 1e-3, 100, 0.05, 1.0
ALPHA, BRANCH_WEIGHT, CALIBRATION_BATCHES, NUM_WORKERS = 0.5, 1.0, 16, 6
INPUTS = ("manifest", "scales", "window_seconds", "calibration")
INPUT_SHA256 = {k: SA.INPUT_SHA256[k] for k in INPUTS}  # gli ingressi del training dell'A/B
# varianti: (modello dei rami, rami concatenati all'embedding raw R); R e' il report della tranche downstream
VARIANTS = {"RP": ("trained", ("psd",)), "RW": ("trained", ("wavelet",)), "RPW": ("trained", KINDS), "RPW0": ("init", KINDS)}
WHICH = {"trained": ("checkpoint.pt", "checkpoint_sha256"), "init": ("checkpoint_init.pt", "init_checkpoint_sha256")}
# collaudo su dbg (criteri congelati in docs/decisioni.md prima del collaudo)
DBG_SECONDS = 1800
COLLAUDO = {"min_updates": 600, "window": 100, "learn_ratio": 0.9, "smooth": 50, "max_rise": 1.5, "margin": 1.25, "slack_s": 600,
            "max_run_h": 2.0}


# --- training dei rami (GPU) ---------------------------------------------------------------------------------------------------------------

def _ab_config(arm: str, calibration) -> R.RunConfig:
    """La configurazione del braccio dell'A/B (modello, dati, seme del run = seme del teacher)."""
    family, seed = arm.split("_seed")
    return SA.build_config(family, int(seed), calibration)


def build_config(arm: str, calibration) -> R.RunConfig:
    """A/B del teacher, encoder congelato, solo il termine dei rami (keep raw spenta), ottimizzazione dei soli rami e delle loro teste."""
    cfg = _ab_config(arm, calibration)
    if cfg.schedule.kind != "constant":
        raise ValueError("step 2C: schedule costante dopo il warmup")
    cfg = replace(cfg, max_steps=STEPS, lr=LR, warmup_steps=WARMUP, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP, keep_grad_every=0,
                  pairing_audit=False, jepa=replace(cfg.jepa, keep_weight=0.0))
    cfg = R.with_spectral_branches(cfg, KINDS, weight=BRANCH_WEIGHT, alpha=ALPHA, freeze_encoder=True, calibration_batches=CALIBRATION_BATCHES)
    if cfg.model.branch_dim != BRANCH_DIM:
        raise ValueError("step 2C: rami da 128 colonne")
    return cfg


def _source(args, arm: str) -> Path:
    """Il teacher d'origine (checkpoint finale dell'A/B), con l'hash del file e la provenienza del suo training (come la tranche downstream)."""
    ckpt = args.runs_root / arm / "checkpoint.pt"
    prov = json.loads((ckpt.parent / "run_provenance.json").read_text())
    if prov.get("source_commit") != SD.TRAIN_COMMIT or prov.get("checkpoint_sha256") != SD.CHECKPOINTS[arm]:
        raise ValueError(f"{ckpt}: provenienza del training diversa da quella congelata")
    if sha256_file(ckpt) != SD.CHECKPOINTS[arm]:
        raise ValueError(f"{ckpt}: checkpoint diverso dal teacher congelato")
    return ckpt


def _run_dir(args, arm: str, collaudo: bool = False) -> Path:
    return args.branch_runs / "collaudo" / arm if collaudo else args.branch_runs / arm


def raw_check(source_teacher: dict, state: dict) -> dict:
    """L'encoder del run contro il teacher d'origine: stesse chiavi (a parte rami e loro teste) e ogni tensore dell'encoder (tutto tranne testa
    dello Step 1, rami e teste) uguale bit per bit nello studente e nel teacher del run. La testa dello Step 1 (congelata e ricalibrata con lo
    stesso file, non usata dai rami ne' dalle sonde) si registra soltanto."""
    from wearusfm.model.spectral_branches import BRANCH_PREFIXES

    encoder = [k for k in source_teacher if not k.startswith(("spectral_keep.",) + BRANCH_PREFIXES)]
    head = [k for k in source_teacher if k.startswith("spectral_keep.")]
    out = {"encoder_tensors": len(encoder)}
    for part in ("student", "teacher"):
        same_keys = {k for k in state[part] if not k.startswith(BRANCH_PREFIXES)} == set(source_teacher)
        out[part] = bool(same_keys and all(torch.equal(state[part][k], source_teacher[k]) for k in encoder))
        out[part + "_step1_head_identical"] = bool(same_keys and all(torch.equal(state[part][k], source_teacher[k]) for k in head))
    out["encoder_identical"] = out["student"] and out["teacher"]
    return out


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(x, np.ones(n) / n, mode="valid")


def collaudo_verdict(metrics: list, summary: dict, integrity: bool) -> dict:
    """Verdetto congelato del collaudo (solo loss di training, nessuna sonda): integrita' (nessuno STOP, encoder del teacher intatto, controllo
    al passo 0 salvato), almeno 600 aggiornamenti, apprendimento (media degli ultimi 100 aggiornamenti <= 0,9 x media dei primi 100), stabilita'
    (media mobile su 50 aggiornamenti, dopo i primi 100, mai oltre 1,5 volte il minimo raggiunto fin li'). Tempo dei run da 3.000
    aggiornamenti = (avvio + 3.000 x mediana dei secondi per aggiornamento) x 1,25 + 10 minuti, arrotondato al quarto d'ora; oltre 2 ore il
    collaudo non passa (serve una decisione)."""
    c = COLLAUDO
    loss = np.asarray([r["keep_branches"] for r in metrics], dtype=np.float64)
    t = np.asarray([r["t_step_s"] for r in metrics], dtype=np.float64)
    n = len(loss)
    out = {"updates": n, "criteria": dict(c)}
    if n == 0:
        return {**out, "passed": False, "reason": "nessun aggiornamento"}
    step_s = float(np.median(t[50:] if n > 100 else t))
    startup = float(summary["elapsed_s"] - t.sum())
    run_s = (startup + STEPS * step_s) * c["margin"] + c["slack_s"]
    alloc_h = math.ceil(run_s / 900) / 4
    out.update({"median_step_s": step_s, "median_data_s": float(np.median([r["t_data_s"] for r in metrics])), "startup_s": startup,
                "run_allocation_h": alloc_h, "time_ok": alloc_h <= c["max_run_h"], "integrity": bool(integrity),
                "enough_updates": n >= c["min_updates"]})
    if n >= 2 * c["window"]:
        first, last = float(loss[:c["window"]].mean()), float(loss[-c["window"]:].mean())
        ma = _moving_average(loss, c["smooth"])
        rise = ma[c["window"]:] / np.minimum.accumulate(ma)[c["window"]:]
        out.update({"loss_first": first, "loss_last": last, "learned": bool(last <= c["learn_ratio"] * first),
                    "max_rise": float(rise.max()) if len(rise) else None,
                    "stable": bool(np.isfinite(loss).all() and (not len(rise) or rise.max() <= c["max_rise"]))})
    else:
        out.update({"learned": False, "stable": False})
    out["passed"] = bool(out["integrity"] and out["enough_updates"] and out["learned"] and out["stable"] and out["time_ok"])
    return out


def train(args) -> None:
    SD.check_source(args.expected_commit)
    commit, arm = args.expected_commit, args.arm
    source = _source(args, arm)
    out = output_path(_run_dir(args, arm, args.collaudo))
    hashes = {k: sha256_file(getattr(args, k)) for k in INPUTS}
    if hashes != INPUT_SHA256:
        raise ValueError("ingressi del training diversi da quelli dell'A/B")
    if args.collaudo and not 0 < args.time_limit_s <= DBG_SECONDS:
        raise ValueError("collaudo: su dbg, al piu' 30 minuti")
    cfg = build_config(arm, args.calibration)
    scales = json.loads(args.scales.read_text())["scales"]
    window_s = json.loads(args.window_seconds.read_text())["window_s"]
    print(json.dumps({"protocol": PROTOCOL, "source_commit": commit, "arm": arm, "collaudo": args.collaudo, "input_sha256": hashes,
                      "config": R.config_to_dict(cfg)}), flush=True)
    summary = R.train(cfg, args.manifest, args.root, out, scales=scales, window_s=window_s, device=args.device, num_workers=args.num_workers,
                      time_limit_s=args.time_limit_s, log=lambda m: print(m, flush=True), init_from=source)
    prov = {"protocol": PROTOCOL, "source_commit": commit, "arm": arm, "collaudo": args.collaudo,
            "source_checkpoint_sha256": SD.CHECKPOINTS[arm], "input_sha256": hashes, "time_limit_s": args.time_limit_s,
            "steps": summary["steps"], "stopped": summary["stopped"], "complete": False, "encoder_identical": None,
            **{key: sha256_file(out / name) if (out / name).exists() else None for name, key in WHICH.values()}}
    (out / "run_provenance.json").write_text(json.dumps(prov, indent=2))  # anche per un run fallito: niente estrazione afterok
    stopped = (out / "STOP").exists()
    prov["complete"] = bool(summary["steps"] == STEPS and summary["stopped"] == "max_steps" and not stopped)
    if not args.collaudo and not prov["complete"]:
        raise ValueError("run incompleto o fermato: nessuna estrazione; nessuna ripresa automatica")
    if stopped or prov["checkpoint_sha256"] is None:
        raise ValueError("collaudo fermato o senza checkpoint")
    state = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    init = torch.load(out / "checkpoint_init.pt", map_location="cpu", weights_only=False)
    SA._finite_state(state)
    branches = json.loads((out / "spectral_branches.json").read_text())
    if (state["config"] != R.config_to_dict(cfg) or init["config"] != state["config"] or init["step"] != 0
            or branches["init_checkpoint_sha256"] != prov["init_checkpoint_sha256"] or branches["frozen_init"]["sha256"] != SD.CHECKPOINTS[arm]):
        raise ValueError("checkpoint finale o del passo 0 diversi dal run")
    source_teacher = torch.load(source, map_location="cpu", weights_only=False)["teacher"]
    prov["raw_check"] = {"final": raw_check(source_teacher, state), "init": raw_check(source_teacher, init)}
    prov["encoder_identical"] = prov["raw_check"]["final"]["encoder_identical"] and prov["raw_check"]["init"]["encoder_identical"]
    SD.check_source(commit)
    if {k: sha256_file(getattr(args, k)) for k in INPUTS} != hashes or sha256_file(source) != SD.CHECKPOINTS[arm]:
        raise ValueError("ingressi o teacher cambiati durante il training")
    if args.collaudo:
        metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
        prov["collaudo_verdict"] = collaudo_verdict(metrics, summary, integrity=prov["encoder_identical"])
        print(json.dumps(prov["collaudo_verdict"]), flush=True)
    (out / "run_provenance.json").write_text(json.dumps(prov, indent=2))
    if not prov["encoder_identical"]:
        raise ValueError("encoder del run diverso dal teacher d'origine: niente estrazione")


# --- uscite dei rami e varianti (CPU) --------------------------------------------------------------------------------------------------

def _run(args, arm: str) -> tuple[Path, dict]:
    """Un run completo del protocollo con l'encoder intatto, e i suoi due checkpoint con l'hash registrato."""
    folder = _run_dir(args, arm)
    prov = json.loads((folder / "run_provenance.json").read_text())
    expected = {"protocol": PROTOCOL, "source_commit": args.expected_commit, "arm": arm, "collaudo": False, "complete": True,
                "encoder_identical": True, "source_checkpoint_sha256": SD.CHECKPOINTS[arm], "input_sha256": INPUT_SHA256}
    if any(prov.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{folder}: run non valido per il protocollo")
    for name, key in WHICH.values():
        if sha256_file(folder / name) != prov[key]:
            raise ValueError(f"{folder / name}: checkpoint cambiato dopo il training")
    return folder, prov


def _identity(commit: str, task: str, arm: str, variant: str) -> dict:
    return {"protocol": PROTOCOL, "source_commit": commit, "task": task, "arm": arm, "variant": variant, "which": "teacher", "probe_seed": 0,
            "split_sha256": SD.SPLIT_SHA, "source_checkpoint_sha256": SD.CHECKPOINTS[arm]}


def _branch_folder(args, task: str, arm: str, which: str) -> Path:
    return args.eval_root / task / arm / f"branches_{which}"


def _folder(args, task: str, arm: str, variant: str) -> Path:
    return args.eval_root / task / arm / variant


def _report_path(args, task: str, arm: str, variant: str) -> Path:
    return args.eval_root / task / arm / f"{variant}.json"


def _width(variant: str) -> int:
    return S2.DOWN_DIM + BRANCH_DIM * len(VARIANTS[variant][1])


def _probe_argv(args, task: str, checkpoint: Path, folder: Path, stage: str, feature_set: str) -> list:
    argv = ["--checkpoint", str(checkpoint), "--features", str(folder), "--stage", stage, "--which", "teacher", "--seed", "0",
            "--feature-set", feature_set]
    if task == "p1":
        if not args.root:
            raise ValueError("P1: servono le radici dei dati")
        for r in args.root:
            argv += ["--root", str(r)]
        argv += ["--window-s", "1", "--val-frac", "0.1", "--weighting", "subjects"]
    elif stage == "extract":
        if not args.epn_root or not args.uci_root:
            raise ValueError("P2: servono le radici di EPN-612 e UCI-EMG")
        for r in args.epn_root:
            argv += ["--epn-root", str(r)]
        argv += ["--uci-root", str(args.uci_root)]
    return argv


def _extract_p2(args, checkpoints: dict, folders: dict, device: str = "cpu") -> None:
    """Come `probe_p2 --stage extract --audit-inputs --feature-set branches --device cpu --batch 64`, per piu' modelli leggendo le finestre una
    volta sola (la lettura di EPN-612 e' la parte lunga): stesse finestre, stesso hash degli ingressi, stesso formato della cache (uguaglianza
    con `probe_p2` nei test). `device`: cpu nel 2C; il pilot dei rami lo passa (valutazione eventuale su nodo GPU)."""
    from types import SimpleNamespace

    from wearusfm.harness import fm_features as FF
    from wearusfm.harness.downstream_ab import fingerprint

    models = {which: FF.load_model(str(path), device, "teacher") for which, path in checkpoints.items()}
    source = SimpleNamespace(epn_root=args.epn_root, uci_root=args.uci_root, max_users=None)
    for ds in probe_p2.DATASETS:
        w, labels, subjects, fs = probe_p2.load_windows(ds, source)
        print(f"{ds}: {len(w)} finestre, {len(set(subjects.tolist()))} soggetti", flush=True)
        input_hash = fingerprint(w, labels, subjects, fs, FF.myo8_montage(ds, fs))
        for which, model in models.items():
            x = FF.extract_features(model, w, fs, FF.myo8_montage(ds, fs), subjects, batch=64, device=device, feature_set="branches")
            folders[which].mkdir(parents=True, exist_ok=True)
            np.savez(folders[which] / f"{ds}.npz", x=x, y=labels, s=subjects, input_sha256=np.asarray(input_hash),
                     **cache_record("branches", FF.feature_width(model, "branches")))
        del w


def extract(args) -> None:
    SD.check_source(args.expected_commit)
    commit, task, arm = args.expected_commit, args.task, args.arm
    run, prov = _run(args, arm)
    down, meta = S2._downstream(args, task, arm)  # identita' completa e file della cache downstream, prima di calcolare
    folders = {w: output_path(_branch_folder(args, task, arm, w)) for w in WHICH}
    variant_folders = {v: output_path(_folder(args, task, arm, v)) for v in VARIANTS}
    prefixes = ("train_", "val_", "test_") if task == "p1" else ("",)
    if task == "p2":
        if not args.epn_root or not args.uci_root:
            raise ValueError("P2: servono le radici di EPN-612 e UCI-EMG")
        _extract_p2(args, {which: run / name for which, (name, _) in WHICH.items()}, folders)
    else:
        for which, (name, _) in WHICH.items():
            argv = _probe_argv(args, task, run / name, folders[which], "extract", "branches") + ["--audit-inputs", "--batch", "64",
                                                                                               "--device", "cpu"]
            if probe_p1.main(argv) != 0:
                raise ValueError("estrazione dei rami fallita")
    inv = {}
    for which, (name, key) in WHICH.items():
        SD.check_source(commit)
        if sha256_file(run / name) != prov[key]:
            raise ValueError("checkpoint cambiato durante l'estrazione")
        inv[which] = S2.inventory(folders[which], task, BRANCH_COLS)
        if any(v.get("feature_set") != "branches" or v.get("model_dim") != BRANCH_COLS for v in inv[which].values()):
            raise ValueError("cache dei rami non registrata come 'branches'")
        if not S2._paired(meta["cache"], inv[which]):
            raise ValueError(f"{task}/{arm}: uscite dei rami non sulle stesse finestre, etichette e soggetti degli embedding")
        (folders[which] / "provenance.json").write_text(json.dumps({
            "identity": _identity(commit, task, arm, f"branches_{which}"), "run": prov, "cache": inv[which]}, indent=2, allow_nan=False))
    hashes = {ds: inv["trained"][ds]["input_sha256"] for ds in inv["trained"]}
    for variant, (which, kinds) in VARIANTS.items():
        cols = np.concatenate([np.arange(i * BRANCH_DIM, (i + 1) * BRANCH_DIM) for i, k in enumerate(KINDS) if k in kinds])
        folder = variant_folders[variant]
        folder.mkdir(parents=True)
        for ds in SD.DATASETS[task]:
            with np.load(down / f"{ds}.npz", allow_pickle=False) as zm, np.load(folders[which] / f"{ds}.npz", allow_pickle=False) as zb:
                m, b = S2._arrays(zm, prefixes), S2._arrays(zb, prefixes)
            for p in prefixes:
                if not (np.array_equal(m[p + "y"], b[p + "y"]) and np.array_equal(m[p + "s"], b[p + "s"])):
                    raise ValueError(f"{task}/{arm}/{ds}: righe non allineate")
            arrays = {**m, **{p + "x": np.concatenate([m[p + "x"], b[p + "x"][:, cols]], 1) for p in prefixes}}
            np.savez(folder / f"{ds}.npz", **arrays, input_sha256=np.asarray(hashes[ds]), **cache_record("model", _width(variant)))
        (folder / "provenance.json").write_text(json.dumps({
            "identity": _identity(commit, task, arm, variant), "downstream": meta, "branches": inv[which], "run": prov,
            "columns": {"raw": S2.DOWN_DIM, "branches": list(kinds), "branch_model": which},
            "cache": S2.inventory(folder, task, _width(variant))}, indent=2, allow_nan=False))
    SD.check_source(commit)


def _variant_meta(args, task: str, arm: str, variant: str) -> tuple[Path, dict]:
    """Cartella e provenienza ricontrollata di una variante: identita', file, cache downstream e uscite dei rami d'origine, run."""
    folder = _folder(args, task, arm, variant)
    meta = json.loads((folder / "provenance.json").read_text())
    inv = S2.inventory(folder, task, _width(variant))
    if meta["identity"] != _identity(args.expected_commit, task, arm, variant) or inv != meta["cache"]:
        raise ValueError(f"{folder}: provenienza o cache cambiate")
    if any(v.get("feature_set") != "model" or v.get("model_dim") != _width(variant) for v in inv.values()):
        raise ValueError(f"{folder}: insieme di feature registrato diverso dall'embedding")
    if S2._downstream(args, task, arm)[1] != meta["downstream"]:
        raise ValueError(f"{folder}: la cache downstream d'origine e' cambiata")
    which = VARIANTS[variant][0]
    if S2.inventory(_branch_folder(args, task, arm, which), task, BRANCH_COLS) != meta["branches"] or _run(args, arm)[1] != meta["run"]:
        raise ValueError(f"{folder}: uscite dei rami o run d'origine cambiati")
    return folder, meta


def probe(args) -> None:
    SD.check_source(args.expected_commit)
    task, arm = args.task, args.arm
    runner = probe_p1 if task == "p1" else probe_p2
    for variant in args.variant:
        folder, meta = _variant_meta(args, task, arm, variant)
        report_path = output_path(_report_path(args, task, arm, variant))
        ckpt = _run_dir(args, arm) / WHICH[VARIANTS[variant][0]][0]  # scritto nel report; la fase probe non lo carica
        if runner.main(_probe_argv(args, task, ckpt, folder, "probe", "model") + ["--out", str(report_path)]) != 0:
            raise ValueError("sonda fallita")
        SD.check_source(args.expected_commit)
        if _variant_meta(args, task, arm, variant)[1]["cache"] != meta["cache"]:
            raise ValueError("cache cambiata durante la sonda")
        report = json.loads(report_path.read_text())
        report["provenance"] = meta
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    if args.compare_after:
        compare(args)


# --- confronto e criteri --------------------------------------------------------------------------------------------------------------

def decide(result: dict) -> dict:
    """La regola congelata del 2A su un confronto accoppiato (`compare_reports`: «A» = riferimento, «B» = variante). Per task: «utile» se
    delta >= +1 punto di accuratezza bilanciata, limite inferiore dell'intervallo > 0 e delta > 0 con tutti e due i teacher; «non dannoso» se
    delta >= -1 punto. Positivo: utile in almeno un task e non dannoso in tutti e due."""
    out = {}
    for task in ("p1", "p2"):
        s = result["tasks"][task]
        useful = bool(s["delta"] >= S2.USEFUL_DELTA and s["ci95"][0] > 0 and all(d > 0 for d in s["delta_by_model_seed"]))
        out[task] = {"reference": s["A"], "variant": s["B"], "delta": s["delta"], "ci95": s["ci95"], "delta_by_teacher": s["delta_by_model_seed"],
                     "useful": useful, "not_harmful": bool(s["delta"] >= S2.HARM_DELTA)}
    out["positive"] = any(out[t]["useful"] for t in ("p1", "p2")) and all(out[t]["not_harmful"] for t in ("p1", "p2"))
    return out


def better_family(result: dict) -> dict:
    """Quale famiglia di teacher con i rami (R+PW): «B» = la regola di «positivo» su delta = B - A, «A» = la stessa su -delta, altrimenti
    nessuna differenza oltre le soglie (simmetrica: nessuna famiglia favorita a parita')."""
    def mirror(s):
        return {**s, "A": s["B"], "B": s["A"], "delta": -s["delta"], "ci95": [-s["ci95"][1], -s["ci95"][0]],
                "delta_by_model_seed": [-d for d in s["delta_by_model_seed"]]}

    b = decide(result)
    a = decide({"tasks": {t: mirror(s) for t, s in result["tasks"].items()}})
    winner = "B" if b["positive"] and not a["positive"] else "A" if a["positive"] and not b["positive"] else None
    return {"B_vs_A": b, "A_vs_B": a, "better": winner or "nessuna differenza oltre le soglie"}


def _strip(result: dict) -> dict:
    """La guardia -2 punti appartiene all'A/B dello Step 1, non a questo protocollo."""
    result.pop("both_practical_guards_pass", None)
    result.pop("guard_note", None)
    for row in result["tasks"].values():
        row.pop("practical_guard_pass", None)
    return result


def _reference(args, task: str, arm: str) -> tuple[dict, dict, str]:
    """R: il report della tranche downstream (stessa cache dell'embedding raw), con provenienza e hash nel suo confronto."""
    path = args.downstream_root / task / f"{arm}.json"
    report = json.loads(path.read_text())
    _, meta = S2._downstream(args, task, arm)
    if report["provenance"]["identity"] != meta["identity"] or report["provenance"]["cache"] != meta["cache"]:
        raise ValueError(f"{path}: report diverso dalla sua cache")
    digest = sha256_file(path)
    if json.loads((args.downstream_root / "comparison.json").read_text())["report_sha256"][f"{task}/{arm}"] != digest:
        raise ValueError(f"{path}: report diverso da quello del confronto downstream")
    return report, meta, digest


def _step2a(args, task: str, arm: str, cache: dict) -> tuple[dict, str]:
    """R+S del test 2A (teacher B): il report MS col suo hash nel confronto del 2A, sulle stesse finestre di questo protocollo."""
    comparison = json.loads((args.step2a_root / "comparison.json").read_text())
    path = args.step2a_root / task / arm / "MS.json"
    digest = sha256_file(path)
    if comparison.get("protocol") != S2.PROTOCOL or comparison["report_sha256"].get(f"{task}/{arm}/MS") != digest:
        raise ValueError(f"{path}: report diverso da quello del confronto del 2A")
    report = json.loads(path.read_text())
    ident = report["provenance"]["identity"]
    if (ident["protocol"], ident["task"], ident["variant"], ident["arm"]) != (S2.PROTOCOL, task, "MS", arm):
        raise ValueError(f"{path}: identita' del report del 2A")
    if not S2._paired(report["provenance"]["cache"], cache):
        raise ValueError(f"{path}: il 2A non e' sulle stesse finestre, etichette e soggetti")
    return report, digest


def _training_summary(args, arm: str) -> dict:
    """Descrittivo: loss dei rami (prime e ultime 100 medie), frame validi della calibrazione per ramo e scala."""
    folder = _run_dir(args, arm)
    metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    keys = ("keep_branches", "keep_fused", "keep_psd", "keep_wavelet")
    calibration = json.loads((folder / "spectral_branches.json").read_text())["calibration"]
    return {"first_100": {k: float(np.mean([r[k] for r in metrics[:100]])) for k in keys},
            "last_100": {k: float(np.mean([r[k] for r in metrics[-100:]])) for k in keys},
            "calibration_valid_frames": calibration["valid_frames"]}


def compare(args) -> None:
    SD.check_source(args.expected_commit)
    out = output_path(args.eval_root / "comparison.json")
    reports, hashes = {}, {}
    for task in SD.DATASETS:
        reports[task], caches = {}, []
        for arm in ARMS:
            reports[task][("R", arm)], meta, hashes[f"{task}/{arm}/R"] = _reference(args, task, arm)
            caches.append(meta["cache"])
            for variant in VARIANTS:
                path = _report_path(args, task, arm, variant)
                r = json.loads(path.read_text())
                _, meta = _variant_meta(args, task, arm, variant)
                if r["provenance"]["identity"] != meta["identity"] or r["provenance"]["cache"] != meta["cache"]:
                    raise ValueError(f"{path}: provenienza del report diversa dalla cache")
                if r.get("feature_set") != "model":
                    raise ValueError(f"{path}: insieme di feature registrato {r.get('feature_set')!r}")
                reports[task][(variant, arm)] = r
                hashes[f"{task}/{arm}/{variant}"] = sha256_file(path)
                caches.append(meta["cache"])
        if not all(S2._paired(caches[0], c) for c in caches):
            raise ValueError(f"{task}: finestre, etichette o soggetti diversi fra teacher o varianti")

    def pair(ref, new, ref_family, new_family=None, ref_reports=None):
        new_family = new_family or ref_family
        out = {}
        for t in SD.DATASETS:
            out[t] = {}
            for k in (0, 1):
                a, b = f"{ref_family}_seed{k}", f"{new_family}_seed{k}"
                out[t][f"A_seed{k}"] = ref_reports[t][a] if ref_reports else reports[t][(ref, a)]
                out[t][f"B_seed{k}"] = reports[t][(new, b)]
        return out

    def paired(*a, **k):
        return _strip(compare_reports(pair(*a, **k)))

    families = {}
    for f in FAMILIES:
        primary = paired("R", "RPW", f)
        training = paired("RPW0", "RPW", f)
        families[f] = {"primary_RPW_vs_R": {"decision": decide(primary), "paired": primary},
                       "training_RPW_vs_RPW0": {"decision": decide(training), "paired": training},
                       "descriptive": {v: paired("R", v, f) for v in ("RP", "RW", "RPW0")}}
    which = paired("RPW", "RPW", "A", "B")
    stats = {"available": args.step2a_root is not None}
    if args.step2a_root is not None:
        ms, ms_hash = {}, {}
        for t in SD.DATASETS:
            ms[t] = {}
            for arm in FAMILIES["B"]:
                ms[t][arm], ms_hash[f"{t}/{arm}/MS"] = _step2a(args, t, arm, reports[t][("RPW", arm)]["provenance"]["cache"])
        result = paired(None, "RPW", "B", ref_reports=ms)
        stats.update(decision=decide(result), paired=result, step2a_report_sha256=ms_hash)
    raw_ab = paired("R", "R", "A", "B")  # contesto: l'A/B dello Step 1 sul solo raw (gia' nella tranche downstream)
    result = {
        "protocol": PROTOCOL, "source_commit": args.expected_commit,
        "decision": {f: {"primary_positive": families[f]["primary_RPW_vs_R"]["decision"]["positive"],
                         "training_matters": families[f]["training_RPW_vs_RPW0"]["decision"]["positive"]} for f in FAMILIES}
        | {"better_family_with_branches": better_family(which)["better"],
           "better_than_fixed_stats_B": stats["decision"]["positive"] if stats["available"] else "2A non disponibile: rimandato"},
        "families": families, "family_A_vs_B_with_branches": {"rule": better_family(which), "paired": which},
        "secondary_RPW_vs_fixed_stats_B": stats,
        "descriptive": {"raw_A_vs_B": raw_ab, "training": {arm: _training_summary(args, arm) for arm in ARMS}},
        "thresholds": {"useful_delta": S2.USEFUL_DELTA, "harm_delta": S2.HARM_DELTA},
        "report_sha256": hashes, "checkpoint_sha256": {arm: SD.CHECKPOINTS[arm] for arm in ARMS},
        "pairing": "compare_reports: braccio A = riferimento, B = variante; teacher accoppiati per seme (A_seed0 con B_seed0 come nell'A/B)",
        "note": ("Un esito positivo dice che le uscite dei rami aiutano questa sonda lineare, non che quell'informazione mancasse dal raw. "
                 "Incertezza condizionata ai due teacher per famiglia; piu' confronti senza correzione (il primario decide, gli altri "
                 "interpretano). Rami allenati solo su emg2qwerty, come i teacher. Nessuna adozione automatica."),
    }
    SD.check_source(args.expected_commit)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["decision"]), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "extract", "probe", "compare"], required=True)
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--task", choices=list(SD.DATASETS))
    ap.add_argument("--variant", nargs="+", choices=list(VARIANTS))
    ap.add_argument("--collaudo", action="store_true", help="train: collaudo su dbg (cartella collaudo/, verdetto congelato)")
    ap.add_argument("--expected-commit", required=True)
    ap.add_argument("--runs-root", type=Path, required=True, help="runs dell'A/B dello Step 1 (teacher d'origine, sola lettura)")
    ap.add_argument("--branch-runs", type=Path, required=True, help="runs dei rami di questo protocollo")
    ap.add_argument("--downstream-root", type=Path, help="tranche step1_downstream_4000_v1 (sola lettura)")
    ap.add_argument("--eval-root", type=Path, help="valutazione di questo protocollo")
    ap.add_argument("--step2a-root", type=Path, help="compare: test 2A concluso (R+S dei teacher B); senza, quel confronto e' rimandato")
    ap.add_argument("--root", type=Path, action="append")
    ap.add_argument("--epn-root", type=Path, action="append")
    ap.add_argument("--uci-root", type=Path)
    for name in INPUTS:
        ap.add_argument("--" + name.replace("_", "-"), type=Path)
    ap.add_argument("--time-limit-s", type=float)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--compare-after", action="store_true", help="probe: poi il confronto, nella stessa allocazione")
    args = ap.parse_args(argv)
    if args.stage == "train":
        if args.arm is None or args.time_limit_s is None or not args.root or any(getattr(args, k) is None for k in INPUTS):
            ap.error("train vuole --arm, --time-limit-s, --root e gli ingressi dell'A/B")
    else:
        if args.downstream_root is None or args.eval_root is None:
            ap.error("extract, probe e compare vogliono --downstream-root e --eval-root")
        if args.stage != "compare" and (args.task is None or args.arm is None):
            ap.error("extract e probe vogliono --task e --arm")
    if args.stage == "probe" and not args.variant:
        ap.error("probe vuole --variant")
    if args.compare_after and args.stage != "probe":
        ap.error("--compare-after vale con probe")
    {"train": train, "extract": extract, "probe": probe, "compare": compare}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
