#!/usr/bin/env python3
"""Test preliminare 2A dello step 2 (protocollo `docs/step2a_stats_probe.md`, ID `step2a_stats_v1`; Simone, 13/10/2026: «parti dal protocollo
del test 2A»): le 32 statistiche spettrali fisse dello Step 1 (e i loro indicatori) accanto all'embedding dei due teacher B dell'A/B dello Step 1,
con le sonde P1/P2 ufficiali (`probe_p1.py`, `probe_p2.py`) e il confronto accoppiato della tranche downstream (`harness.downstream_ab`).

**Nessuna GPU.** Gli embedding sono le cache gia' estratte e verificate della tranche `step1_downstream_4000_v1` (lette, mai riscritte); le
statistiche non dipendono dal modello e si calcolano una volta per task sulla CPU, con l'hash degli ingressi realmente letti: devono coincidere
riga per riga con le cache degli embedding (stesso `input_sha256`, stesse etichette e soggetti nello stesso ordine), o la catena si ferma.
Varianti: M (embedding, cache downstream: la sonda rifatta deve riprodurre il report downstream) e MS (M + statistiche) per ciascun teacher B;
S (statistiche) e K (soli indicatori di disponibilita') una volta per task. M + indicatori non c'e': le sonde sono per dataset e dentro un dataset
fs e banda sono le stesse per tutte le finestre, quindi gli indicatori sono costanti (la fase stats conta le maschere diverse per dataset e lo
registra). Tutto fuori da Git, in cartelle nuove.

  stats   --task T                      statistiche (CPU) -> EVAL/T/S, poi EVAL/T/K e, per B_seed0 e B_seed1, EVAL/T/<arm>/MS
  probe   --task T --variant V [--arm]  sonda ufficiale -> EVAL/T/<arm>/<V>.json oppure EVAL/T/<V>.json
  compare                               confronto accoppiato MS-M sui due teacher e criteri congelati -> EVAL/comparison.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  prima di sklearn sul Mac

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import probe_p1  # noqa: E402
import probe_p2  # noqa: E402
import step1_downstream as SD  # noqa: E402
from spectral_step1 import output_path  # noqa: E402
from wearusfm.harness.downstream_ab import compare_reports, fingerprint  # noqa: E402
from wearusfm.training.spectral_calibration import sha256_file  # noqa: E402

PROTOCOL = "step2a_stats_v1"
ARMS = ("B_seed0", "B_seed1")
STATS_ARM = "B_seed0"  # checkpoint caricato (sulla CPU) dalle sonde in estrazione; le statistiche non lo usano
DOWN_DIM = 768  # embedding dei teacher dell'A/B (2 x 384)
VARIANTS = {"M": "model", "MS": "model+stats", "S": "stats", "K": "mask"}
PER_ARM = ("M", "MS")
DIMS = {"S": 64, "K": 32}
# criteri congelati (docs/decisioni.md, prima di qualunque misura)
USEFUL_DELTA, HARM_DELTA = 0.01, -0.01


def inventory(folder: Path, task: str, dim: int) -> dict:
    """Come `step1_downstream.cache_inventory`, per una dimensione qualunque: hash fisico, hash degli ingressi, righe e soggetti per ruolo."""
    out = {}
    for ds in SD.DATASETS[task]:
        path = folder / f"{ds}.npz"
        with np.load(path, allow_pickle=False) as z:
            if "input_sha256" not in z or len(str(z["input_sha256"].item())) != 64:
                raise ValueError("cache senza hash verificato degli ingressi")
            rows, seen = {}, set()
            for prefix in (("train_", "val_", "test_") if task == "p1" else ("",)):
                x, y, s = (z[prefix + k] for k in "xys")
                if x.ndim != 2 or x.shape != (len(y), dim) or s.shape != y.shape or y.ndim != 1 or not len(y) or not np.isfinite(x).all():
                    raise ValueError(f"cache {path}: dimensioni o valori non validi (attese {dim} colonne)")
                names = set(s.tolist())
                if seen & names:
                    raise ValueError("soggetti in piu' ruoli")
                seen.update(names)
                rows[prefix] = {"n": len(y), "labels_subjects_sha256": fingerprint(y, s), "subjects": sorted(names)}
            out[ds] = {"sha256": sha256_file(path), "input_sha256": str(z["input_sha256"].item()), "rows": rows}
    return out


def _paired(a: dict, b: dict) -> bool:
    """La parte di due inventari che deve coincidere fra varianti: tutto tranne l'hash fisico del file."""
    return set(a) == set(b) and all({k: v for k, v in a[ds].items() if k != "sha256"} == {k: v for k, v in b[ds].items() if k != "sha256"}
                                    for ds in a)


def _identity(commit: str, task: str, variant: str, arm: str | None) -> dict:
    return {"protocol": PROTOCOL, "source_commit": commit, "task": task, "variant": variant, "arm": arm, "probe_seed": 0, "which": "teacher",
            "split_sha256": SD.SPLIT_SHA, "stats_checkpoint_sha256": SD.CHECKPOINTS[STATS_ARM]}


def _downstream(args, task: str, arm: str) -> tuple[Path, dict]:
    """La cache degli embedding della tranche downstream, con la sua provenienza ricontrollata (identita' e hash dei file)."""
    folder = args.downstream_root / task / arm
    meta = json.loads((folder / "provenance.json").read_text())
    ident = meta["identity"]
    if (ident.get("protocol"), ident.get("checkpoint_sha256"), ident.get("arm"), ident.get("task"), ident.get("split_sha256"),
            ident.get("which"), ident.get("probe_seed")) != (SD.PROTOCOL, SD.CHECKPOINTS[arm], arm, task, SD.SPLIT_SHA, "teacher", 0):
        raise ValueError(f"cache downstream {folder}: provenienza diversa da quella congelata")
    if inventory(folder, task, DOWN_DIM) != meta["cache"]:
        raise ValueError(f"cache downstream {folder}: file cambiati dopo l'estrazione")
    return folder, meta


def _folder(args, task: str, variant: str, arm: str | None) -> Path:
    return args.eval_root / task / arm / variant if arm else args.eval_root / task / variant


def _report_path(args, task: str, variant: str, arm: str | None) -> Path:
    return args.eval_root / task / arm / f"{variant}.json" if arm else args.eval_root / task / f"{variant}.json"


def _probe_argv(args, task: str, folder: Path, stage: str, variant: str) -> list[str]:
    argv = ["--checkpoint", str(args.runs_root / STATS_ARM / "checkpoint.pt"), "--features", str(folder), "--stage", stage,
            "--which", "teacher", "--seed", "0", "--feature-set", VARIANTS[variant]]
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


def _write_variant(folder: Path, parts: dict, input_hash: dict) -> None:
    """Una cache nel formato delle sonde (`probe_p1`: train_/val_/test_x,y,s; `probe_p2`: x,y,s), con l'hash degli ingressi."""
    folder.mkdir(parents=True)
    for ds, arrays in parts.items():
        np.savez(folder / f"{ds}.npz", **arrays, input_sha256=np.asarray(input_hash[ds]))


def stats(args) -> None:
    SD.check_source(args.expected_commit)
    commit = args.expected_commit
    task = args.task
    ckpt = args.runs_root / STATS_ARM / "checkpoint.pt"
    if sha256_file(ckpt) != SD.CHECKPOINTS[STATS_ARM]:
        raise ValueError("checkpoint diverso dal teacher congelato")
    sources = {arm: _downstream(args, task, arm) for arm in ARMS}  # prima di calcolare qualunque cosa
    s_folder = output_path(_folder(args, task, "S", None))
    k_folder = output_path(_folder(args, task, "K", None))
    ms_folders = {arm: output_path(_folder(args, task, "MS", arm)) for arm in ARMS}
    runner = probe_p1 if task == "p1" else probe_p2
    if runner.main(_probe_argv(args, task, s_folder, "extract", "S") + ["--audit-inputs", "--batch", "64", "--device", "cpu"]) != 0:
        raise ValueError("estrazione delle statistiche fallita")
    SD.check_source(commit)
    inv_s = inventory(s_folder, task, 64)
    prefixes = ("train_", "val_", "test_") if task == "p1" else ("",)
    stats_arrays = {}
    for ds in SD.DATASETS[task]:
        with np.load(s_folder / f"{ds}.npz", allow_pickle=False) as z:
            stats_arrays[ds] = {k: z[k] for k in z.files if k != "input_sha256"}
    # maschere di disponibilita' diverse per dataset (1 = costanti: M + indicatori sarebbe uguale a M)
    patterns = {ds: int(len(np.unique(np.concatenate([a[p + "x"][:, 32:] for p in prefixes]), axis=0))) for ds, a in stats_arrays.items()}
    (s_folder / "provenance.json").write_text(json.dumps({"identity": _identity(commit, task, "S", None), "cache": inv_s,
                                                         "mask_patterns": patterns}, indent=2, allow_nan=False))
    hashes = {ds: inv_s[ds]["input_sha256"] for ds in inv_s}
    _write_variant(k_folder, {ds: {**a, **{p + "x": a[p + "x"][:, 32:] for p in prefixes}} for ds, a in stats_arrays.items()}, hashes)
    (k_folder / "provenance.json").write_text(json.dumps({"identity": _identity(commit, task, "K", None), "source": inv_s,
                                                         "cache": inventory(k_folder, task, 32)}, indent=2, allow_nan=False))
    for arm, (down, meta) in sources.items():
        if not _paired(meta["cache"], inv_s):
            raise ValueError(f"{task}/{arm}: le statistiche non sono sulle stesse finestre, etichette e soggetti degli embedding")
        ms = {}
        for ds in SD.DATASETS[task]:
            with np.load(down / f"{ds}.npz", allow_pickle=False) as z:
                m = {k: z[k] for k in z.files if k != "input_sha256"}
            a = stats_arrays[ds]
            for p in prefixes:
                if not (np.array_equal(m[p + "y"], a[p + "y"]) and np.array_equal(m[p + "s"], a[p + "s"])):
                    raise ValueError(f"{task}/{arm}/{ds}: righe non allineate")
            ms[ds] = {**m, **{p + "x": np.concatenate([m[p + "x"], a[p + "x"]], 1) for p in prefixes}}
        _write_variant(ms_folders[arm], ms, hashes)
        (ms_folders[arm] / "provenance.json").write_text(json.dumps({
            "identity": _identity(commit, task, "MS", arm), "downstream": meta, "stats": inv_s,
            "cache": inventory(ms_folders[arm], task, DOWN_DIM + 64)}, indent=2, allow_nan=False))
    SD.check_source(commit)


def _variant_meta(args, task: str, variant: str, arm: str | None) -> tuple[Path, dict]:
    """Cartella e provenienza ricontrollata di una variante (M: la cache downstream)."""
    commit = args.expected_commit
    if (variant in PER_ARM) != (arm is not None) or (arm is not None and arm not in ARMS):
        raise ValueError(f"variante {variant}: braccio {arm} non previsto")
    if variant == "M":
        folder, meta = _downstream(args, task, arm)
        return folder, {"identity": _identity(commit, task, "M", arm), "downstream": meta, "cache": meta["cache"]}
    folder = _folder(args, task, variant, arm)
    meta = json.loads((folder / "provenance.json").read_text())
    if meta["identity"] != _identity(commit, task, variant, arm) or inventory(folder, task, DIMS.get(variant, DOWN_DIM + 64)) != meta["cache"]:
        raise ValueError(f"{folder}: provenienza o cache cambiate")
    if variant == "MS" and inventory(_downstream(args, task, arm)[0], task, DOWN_DIM) != meta["downstream"]["cache"]:
        raise ValueError(f"{folder}: la cache downstream d'origine e' cambiata")
    return folder, meta


def probe(args) -> None:
    SD.check_source(args.expected_commit)
    task, variant, arm = args.task, args.variant, args.arm
    folder, meta = _variant_meta(args, task, variant, arm)
    report_path = output_path(_report_path(args, task, variant, arm))
    runner = probe_p1 if task == "p1" else probe_p2
    if runner.main(_probe_argv(args, task, folder, "probe", variant) + ["--out", str(report_path)]) != 0:
        raise ValueError("sonda fallita")
    SD.check_source(args.expected_commit)
    if _variant_meta(args, task, variant, arm)[1]["cache"] != meta["cache"]:
        raise ValueError("cache cambiata durante la sonda")
    report = json.loads(report_path.read_text())
    report["provenance"] = meta
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    if args.compare_after:
        compare(args)


def decide(with_stats: dict) -> dict:
    """I criteri congelati sul confronto accoppiato (`compare_reports`, braccio «A» = M, «B» = MS). Per task: «utile» se il guadagno delle
    statistiche e' >= 1 punto di accuratezza bilanciata, col limite inferiore dell'intervallo > 0 e positivo con tutti e due i teacher; «non
    dannoso» se >= -1 punto. 2A positivo: utile in almeno un task e non dannoso in tutti e due."""
    out = {}
    for task in ("p1", "p2"):
        s = with_stats["tasks"][task]
        useful = bool(s["delta"] >= USEFUL_DELTA and s["ci95"][0] > 0 and all(d > 0 for d in s["delta_by_model_seed"]))
        out[task] = {"M": s["A"], "MS": s["B"], "delta_stats": s["delta"], "delta_stats_ci95": s["ci95"],
                     "delta_stats_by_teacher": s["delta_by_model_seed"], "useful": useful, "not_harmful": bool(s["delta"] >= HARM_DELTA)}
    out["positive"] = any(out[t]["useful"] for t in ("p1", "p2")) and all(out[t]["not_harmful"] for t in ("p1", "p2"))
    out["thresholds"] = {"useful_delta": USEFUL_DELTA, "harm_delta": HARM_DELTA}
    out["note"] = ("Esito informativo per decidere se allenare i rami spettrali (step 2B-3), non adozione automatica. Incertezza condizionata "
                   "ai due teacher B; due task, nessuna correzione per confronti multipli.")
    return out


def compare(args) -> None:
    SD.check_source(args.expected_commit)
    out = output_path(args.eval_root / "comparison.json")
    reports, hashes, metas = {}, {}, {}
    for task in SD.DATASETS:
        reports[task] = {}
        for variant in VARIANTS:
            for arm in (ARMS if variant in PER_ARM else (None,)):
                path = _report_path(args, task, variant, arm)
                r = json.loads(path.read_text())
                _, meta = _variant_meta(args, task, variant, arm)
                if r["provenance"]["identity"] != meta["identity"] or r["provenance"]["cache"] != meta["cache"]:
                    raise ValueError(f"{path}: provenienza del report diversa dalla cache")
                if r.get("feature_set") != VARIANTS[variant]:
                    raise ValueError(f"{path}: insieme di feature registrato {r.get('feature_set')!r}")
                reports[task][(variant, arm)] = r
                metas[(task, variant, arm)] = meta
                hashes[f"{task}/{arm or '-'}/{variant}"] = sha256_file(path)
        caches = [m["cache"] for (t, _, _), m in metas.items() if t == task]
        if not all(_paired(caches[0], c) for c in caches):
            raise ValueError(f"{task}: finestre, etichette o soggetti diversi fra le varianti")

    def paired(variant):
        return {t: {"A_seed0": reports[t][("M", "B_seed0")], "B_seed0": reports[t][(variant, "B_seed0")],
                    "A_seed1": reports[t][("M", "B_seed1")], "B_seed1": reports[t][(variant, "B_seed1")]} for t in SD.DATASETS}

    with_stats = compare_reports(paired("MS"))
    with_stats.pop("both_practical_guards_pass", None)  # la guardia -2 punti appartiene all'A/B dello Step 1, non a questo protocollo
    with_stats.pop("guard_note", None)
    for row in with_stats["tasks"].values():
        row.pop("practical_guard_pass", None)
    # controlli descrittivi: la sonda rifatta su M riproduce il report della tranche downstream; maschere diverse per dataset
    reproduced = {}
    for t in SD.DATASETS:
        for arm in ARMS:
            old = json.loads((args.downstream_root / t / f"{arm}.json").read_text())
            new = reports[t][("M", arm)]
            reproduced[f"{t}/{arm}"] = bool(abs(old[t] - new[t]) <= 1e-12 and all(
                abs(old["datasets"][d]["test_bacc"] - new["datasets"][d]["test_bacc"]) <= 1e-12 for d in SD.DATASETS[t]))
    result = {"protocol": PROTOCOL, "source_commit": args.expected_commit, "decision": decide(with_stats), "paired_M_vs_MS": with_stats,
              "descriptive": {"S_K": {t: {v: reports[t][(v, None)][t] for v in ("S", "K")} for t in SD.DATASETS},
                              "M_reproduces_downstream_report": reproduced,
                              "mask_patterns_per_dataset": {t: metas[(t, "S", None)].get("mask_patterns") for t in SD.DATASETS}},
              "report_sha256": hashes, "checkpoint_sha256": {a: SD.CHECKPOINTS[a] for a in ARMS},
              "pairing": "braccio A di compare_reports = M (embedding), B = MS (embedding + statistiche), teacher B_seed0 e B_seed1"}
    SD.check_source(args.expected_commit)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["decision"]), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["stats", "probe", "compare"], required=True)
    ap.add_argument("--task", choices=list(SD.DATASETS))
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--expected-commit", required=True)
    ap.add_argument("--runs-root", type=Path, required=True, help="runs dell'A/B dello Step 1 (checkpoint dei teacher)")
    ap.add_argument("--downstream-root", type=Path, required=True, help="cartella della tranche step1_downstream_4000_v1 (sola lettura)")
    ap.add_argument("--eval-root", type=Path, required=True, help="cartella NUOVA di questo protocollo")
    ap.add_argument("--root", type=Path, action="append")
    ap.add_argument("--epn-root", type=Path, action="append")
    ap.add_argument("--uci-root", type=Path)
    ap.add_argument("--compare-after", action="store_true", help="probe: poi il confronto, nella stessa allocazione")
    args = ap.parse_args(argv)
    if args.stage != "compare" and args.task is None:
        ap.error("stats e probe vogliono --task")
    if args.stage == "probe" and args.variant is None:
        ap.error("probe vuole --variant")
    if args.compare_after and args.stage != "probe":
        ap.error("--compare-after vale con probe")
    {"stats": stats, "probe": probe, "compare": compare}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
