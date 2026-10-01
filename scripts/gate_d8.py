#!/usr/bin/env python3
"""Gate di consistenza al ricampionamento (D8, passo 3), con le definizioni FIRMATE il 01/10/2026 (`docs/proposta_gate_d8.md`).

Dati: Kaifosh processato (2 kHz). Per ogni soggetto 20 finestre da 4 s (piu' 0,5 s di margine per lato), a caso; i buchi marcati si evitano. Front-end
all'inizializzazione con 3 semi. Solo CPU. Due processi (`--stage all`, il default, li lancia entrambi): `features` (torch: misura (i) e vettori per la
sonda, salvati in `--work`) e `probe` (scikit-learn: misura (ii) ed esito). Su macOS torch e scikit-learn nello stesso processo vanno in segfault.

  python scripts/gate_d8.py --processed-root $SCRATCH/data/processed/kaifosh --out $WORK/wearusfm_runs/results/step3/gate_<job>.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data.processed import discover_sessions, load_session  # noqa: E402
from wearusfm.gate import consistency as G  # noqa: E402
from wearusfm.tokenizer_checks.run_io import argv_without_option  # noqa: E402


def load_windows(root: Path, per_subject: int, seed: int, max_subjects: int | None) -> list:
    """Un soggetto alla volta: si caricano le sue sessioni, si estraggono le finestre e si libera la memoria (Kaifosh intero in float64 sono ~30 GB)."""
    by_subject: dict[str, list] = {}
    for subj, sess, path in discover_sessions(root, "kaifosh"):
        by_subject.setdefault(subj, []).append((sess, path))
    subjects = sorted(by_subject)[:max_subjects] if max_subjects else sorted(by_subject)
    rng = np.random.default_rng([seed, 8, 2])
    windows = []
    for subj in subjects:
        sessions = []
        for sess, path in by_subject[subj]:
            s = load_session(path, "kaifosh", subj, sess)
            if s.fs != G.FS_A:
                raise SystemExit(f"{path}: frequenza {s.fs}, attesa {G.FS_A}")
            for k, seg in enumerate(s.segments):
                avoid = [(st, st + n) for kk, _, st, n in s.constant_runs if kk == k]
                sessions.append((subj, seg[:, s.qc_valid], avoid))
        windows += G.draw_windows(sessions, per_subject, rng)
    return windows


def stage_features(args) -> None:
    import torch

    from wearusfm.model.frontend import ContinuousKernelFrontEnd

    torch.set_num_threads(args.threads)
    t0 = time.time()
    windows = load_windows(args.processed_root, args.per_subject, args.split_seed, args.max_subjects)
    print(f"{len(windows)} finestre da {len({w.subject for w in windows})} soggetti", flush=True)
    if len({w.subject for w in windows}) < 3:
        raise SystemExit(f"servono finestre da almeno 3 soggetti (split 60/20/20), trovate {len(windows)} in {args.processed_root}")
    args.work.mkdir(parents=True, exist_ok=True)
    errors = {}
    for case, (fc, dec) in G.CASES.items():
        for seed in args.seeds:
            r = G.features_case(ContinuousKernelFrontEnd(seed=seed), windows, fc, dec)
            key = f"{case}_seed{seed}"
            np.savez(args.work / f"{key}.npz", x=r["x"], y=r["y"], units=r["units"])
            errors[key] = r["rel_error"]
            print(f"{key}: errore totale {r['rel_error']['total']:.4f}", flush=True)
    (args.work / "errors.json").write_text(json.dumps({"errors": errors, "n_windows": len(windows), "elapsed_s": time.time() - t0}, indent=2))


def stage_probe(args) -> int:
    e = json.loads((args.work / "errors.json").read_text())
    res: dict = {"decision": "docs/decisioni.md, D8a - definizioni operative FIRMATE il 01/10/2026", "dataset": "kaifosh",
                 "per_subject": args.per_subject, "n_windows": e["n_windows"], "trial_only": bool(args.max_subjects),
                 "frontend": {"patch_ms": 25.0, "context_ms": 100.0, "families": ["fourier", "spline", "mlp"], "seeds": args.seeds}, "cases": {}}
    for key, rel in e["errors"].items():
        z = np.load(args.work / f"{key}.npz")
        probe = G.probe_case(z["x"], z["y"], z["units"], args.split_seed)
        res["cases"][key] = {"rel_error": rel, "probe": probe, **G.verdict(rel, probe)}
        p = probe
        print(f"{key:14s} errore totale {rel['total']:.4f} (fourier {rel['fourier']:.4f}, spline {rel['spline']:.4f}, mlp {rel['mlp']:.4f}) | sonda "
              f"{p['balanced_accuracy']:.3f} IC [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] {p['model']} | {'PASSA' if res['cases'][key]['passes'] else 'NON PASSA'}")
    res["passes"] = all(v["passes"] for v in res["cases"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"GATE: {'SUPERATO' if res['passes'] else 'NON SUPERATO'}{' (prova ridotta, NON il gate firmato)' if args.max_subjects else ''}\nrisultati in {args.out}")
    return 0 if res["passes"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=None, help="cartella per le feature intermedie (default: accanto a --out)")
    ap.add_argument("--per-subject", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=None, help="solo per le prove: NON e' il gate firmato")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stage", choices=("all", "features", "probe"), default="all")
    args = ap.parse_args(argv)
    args.work = args.work or args.out.with_suffix("").with_name(args.out.stem + "_features")
    if args.stage == "all":
        raw = list(argv) if argv is not None else sys.argv[1:]
        base = [sys.executable, str(Path(__file__).resolve())] + argv_without_option(raw, "--stage")
        rc = subprocess.run(base + ["--stage", "features"]).returncode
        return rc if rc else subprocess.run(base + ["--stage", "probe"]).returncode
    if args.stage == "features":
        stage_features(args)
        return 0
    return stage_probe(args)


if __name__ == "__main__":
    sys.exit(main())
