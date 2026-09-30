#!/usr/bin/env python3
"""ANALISI DESCRITTIVA, NON CONGELATA (D5a resta com'e'): feature continue del tokenizer come bersaglio alternativo ai codici.
(a) sonda dataset-ID sulle feature continue contro 5 bande e codici (stesso campione/split/sonda di V3);
(b) stabilita' delle feature sotto il rumore di V4, su tutti i dataset del run.
Logica in `wearusfm.tokenizer_checks.continuous_features`; qui solo I/O e modello. NON tocca nessun verdetto di D5a.

Due processi separati (lo lancia da solo `--stage all`, il default): `torch` (guardie, feature, stabilita' (b)) e `probe` (sonda (a), scikit-learn). Su macOS
scikit-learn (HistGradientBoosting) va in segfault se torch e' gia' stato importato nello stesso processo (conflitto di OpenMP): scoperto con la
prova su dati sintetici, per questo non si possono unire.

Gira in LOCALE, su CPU (i codici della CPU riproducono quelli dell'A100: results/step1bis/NOTA_TOKENIZER_LOCALE.md), con l'ambiente
`~/.venvs/wearusfm-tok`. Servono, tutti FUORI dal repo (mai dati wearable in un commit): gli array salvati dal run vero
(`arrays_<jobid>/`, sul cluster), il clone di NeuroRVQ (commit 926e770) e il checkpoint (sha256 verificato qui sotto).

  ~/.venvs/wearusfm-tok/bin/python scripts/step1bis_continuous_features.py \
      --arrays ~/wearusfm_local/arrays_59048369 --repo ~/wearusfm_local/neurorvq_repo \
      --checkpoint ~/wearusfm_local/models/NeuroRVQ_EMG_tokenizer_v1.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.tokenizer_checks import continuous_features as C  # noqa: E402
from wearusfm.tokenizer_checks import metrics as M  # noqa: E402
from wearusfm.tokenizer_checks.neurorvq import SPATIAL_INDEX_FIXED  # noqa: E402
from wearusfm.tokenizer_checks.run_io import CHECKPOINT_SHA256, argv_without_option, load_arrays, sha256_file  # noqa: E402

GUARD_WINDOWS = 16  # finestre su cui si ricalcolano i codici per confrontarli con quelli salvati (GPU)


def cache_path(cache_dir: Path, name: str, tokens: np.ndarray) -> Path:
    key = hashlib.sha256(np.ascontiguousarray(tokens).tobytes()).hexdigest()[:12]
    return cache_dir / f"{name}_{key}.npy"


def update_json(path: Path, patch: dict) -> dict:
    cur = json.loads(path.read_text()) if path.exists() else {}
    for k, v in patch.items():
        cur[k] = {**cur.get(k, {}), **v} if isinstance(v, dict) and isinstance(cur.get(k), dict) else v
    path.write_text(json.dumps(cur, indent=2))
    return cur


def stage_torch(args) -> None:
    """Guardie, feature per token (cache) e stabilita' (b). Importa torch: qui NON si importa scikit-learn."""
    sha = sha256_file(args.checkpoint)
    if sha != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha256 {sha} diverso da quello verificato ({CHECKPOINT_SHA256}): mi fermo")
    import torch

    from wearusfm.tokenizer_checks.neurorvq import NeuroRVQRunner

    torch.set_num_threads(args.threads)
    runner = NeuroRVQRunner(args.repo, args.checkpoint, device="cpu")
    fn = lambda w: runner.features(w[:, None, :], [SPATIAL_INDEX_FIXED])  # noqa: E731
    rep = json.loads(args.run_report.read_text())
    per = load_arrays(args.arrays)
    print(f"dataset: {sorted(per)}", flush=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    update_json(args.out, {
        "note": "ANALISI DESCRITTIVA NON CONGELATA (D5a invariata). CPU, checkpoint sha256 verificato.",
        "checkpoint_sha256": sha, "run_report": args.run_report.name, "seed": args.seed,
    })

    # 0. guardia: i codici ricalcolati qui coincidono con quelli salvati dal run su GPU (altrimenti le feature non sono quelle del run)
    guard = {}
    for name, d in per.items():
        n = GUARD_WINDOWS * C.N_TOKENS_PER_WINDOW
        got = runner.run(d["tokens"][:n].reshape(-1, C.WINDOW_SAMPLES)[:, None, :], [SPATIAL_INDEX_FIXED], want_recon=False)["codes"]
        same = float((got == d["codes"][:, :, :n]).mean())
        guard[name] = {"codes_identical_fraction": same, "n_tokens": n}
        print(f"guardia {name}: codici identici a quelli del run GPU = {same:.4f}", flush=True)
    update_json(args.out, {"guard": guard})

    # 1. feature continue per token, solo per (a) (cache: minuti di CPU per dataset, non si rifanno). (b) calcola da se' 200 finestre.
    if args.only in ("a", "both"):
        for name, d in per.items():
            cache = cache_path(args.cache, name, d["tokens"])
            if cache.exists():
                continue
            t0 = time.time()
            f = C.token_features(fn, d["tokens"], args.batch)
            np.save(cache, f)
            print(f"feature {name}: {f.shape} in {time.time() - t0:.0f} s", flush=True)

    # 2. (b) stabilita' delle feature, tutti i dataset
    if args.only in ("b", "both"):
        print("\n(b) coseno feature pulite-vs-rumorose (mediana per ramo | 5% piu' basso), contro coseno fra token diversi")
        for name, d in per.items():
            floor = M.noise_floor_rms(d["tokens"])
            ref_floor = rep.get("v4_per_dataset", {}).get(name, {}).get("noise_floor_rms")
            r = C.stability_continuous(fn, d["tokens"], floor, n_windows=args.stability_windows, seed=args.seed, batch=args.batch)
            r["noise_floor_matches_run_report"] = None if ref_floor is None else bool(abs(floor - ref_floor) <= 1e-4 * abs(ref_floor))
            update_json(args.out, {"stability": {name: r}})
            print(f"  {name}  (soglia di rumore {floor:.4f}, coincide col run: {r['noise_floor_matches_run_report']})")
            for lvl, v in r["noise"].items():
                print(f"    {lvl:11s} mediana {np.round(v['median_per_branch'], 3)} | q05 {np.round(v['q05_per_branch'], 3)}")
            b = r["between_different_tokens"]
            print(f"    token diversi mediana {np.round(b['median_per_branch'], 3)} | q95 {np.round(b['q95_per_branch'], 3)}", flush=True)


def stage_probe(args) -> None:
    """(a) sonda dataset-ID dalle feature in cache. NON importa torch (vedi la nota in testa)."""
    rep = json.loads(args.run_report.read_text())
    per = load_arrays(args.arrays)
    feats = {}
    for name, d in per.items():
        cache = cache_path(args.cache, name, d["tokens"])
        if not cache.exists():
            raise SystemExit(f"manca la cache delle feature di {name} ({cache.name}): lancia prima `--stage torch`")
        feats[name] = np.load(cache)
    res = C.dataset_id_probe_continuous(per, feats, seed=args.seed)
    pairs = {"bands": [res["frozen_bands"]["balanced_accuracy"], rep["v3"]["bands"]["balanced_accuracy"]],
             "codes": [res["frozen_codes"]["balanced_accuracy"], rep["v3"]["codes"]["balanced_accuracy"]]}
    pairs["ok"] = all(abs(a - b) < 0.005 for a, b in (pairs["bands"], pairs["codes"]))
    res["reproduces_frozen_v3"] = pairs
    update_json(args.out, {"dataset_id": res})
    print("\n(a) sonda dataset-ID (accuratezza bilanciata, test; caso = %.3f)" % (1 / len(res["datasets"])))
    for k, v in res.items():
        if isinstance(v, dict) and "balanced_accuracy" in v:
            print(f"  {k:34s} {v['balanced_accuracy']:.3f}  ci95=[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]  {v['model']}")
    print(f"  V3 congelato riprodotto (bande, codici) = {pairs['ok']}  {pairs}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True, help="clone di NeuroRVQ")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--run-report", type=Path, default=ROOT / "results/step1bis/step1bis_59048369.json")
    ap.add_argument("--cache", type=Path, default=Path.home() / "wearusfm_local/step1bis_features", help="feature per token (fuori dal repo)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/step1bis/continuous_features.json")
    ap.add_argument("--only", choices=("a", "b", "both"), default="both")
    ap.add_argument("--stage", choices=("all", "torch", "probe"), default="all", help="`all` lancia torch e probe in due processi")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stability-windows", type=int, default=200)
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "all":
        raw = list(argv) if argv is not None else sys.argv[1:]
        base = [sys.executable, str(Path(__file__).resolve())] + argv_without_option(raw, "--stage")
        for stage in ("torch", "probe"):
            if stage == "probe" and args.only == "b":
                continue
            rc = subprocess.run(base + ["--stage", stage]).returncode
            if rc != 0:
                print(f"stage {stage}: uscito con codice {rc}", file=sys.stderr)
                return rc
        print(f"\nrisultati in {args.out}")
        return 0
    (stage_torch if args.stage == "torch" else stage_probe)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
