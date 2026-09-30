#!/usr/bin/env python3
"""ANALISI DESCRITTIVA, NON CONGELATA (D5a resta com'e'): feature continue del tokenizer come bersaglio alternativo ai codici.
(a) sonda dataset-ID sulle feature continue contro 5 bande e codici (stesso campione/split/sonda di V3);
(b) stabilita' delle feature sotto il rumore di V4, su tutti i dataset del run.
Logica in `wearusfm.tokenizer_checks.continuous_features`; qui solo I/O e modello. NON tocca nessun verdetto di D5a.

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
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.tokenizer_checks import continuous_features as C  # noqa: E402
from wearusfm.tokenizer_checks import metrics as M  # noqa: E402
from wearusfm.tokenizer_checks.neurorvq import SPATIAL_INDEX_FIXED  # noqa: E402

CHECKPOINT_SHA256 = "0d255bcc9f1c75ccc374cba06eab476f15bf5fb2a87115d6dc8d2dc0adadce49"
GUARD_WINDOWS = 16  # finestre su cui si ricalcolano i codici per confrontarli con quelli salvati (GPU)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_arrays(arrays_dir: Path) -> dict:
    out = {}
    for p in sorted(arrays_dir.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        out[p.stem] = {k: z[k] for k in ("codes", "tokens", "group_subject")}
    if not out:
        raise SystemExit(f"nessun .npz in {arrays_dir}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True, help="clone di NeuroRVQ")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--run-report", type=Path, default=ROOT / "results/step1bis/step1bis_59048369.json")
    ap.add_argument("--cache", type=Path, default=Path.home() / "wearusfm_local/step1bis_features", help="feature per token (fuori dal repo)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/step1bis/continuous_features.json")
    ap.add_argument("--only", choices=("a", "b", "both"), default="both")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stability-windows", type=int, default=200)
    args = ap.parse_args(argv)

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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "note": "ANALISI DESCRITTIVA NON CONGELATA (D5a invariata). CPU, checkpoint sha256 verificato.",
        "checkpoint_sha256": sha, "run_report": args.run_report.name, "seed": args.seed, "guard": {}, "stability": {},
    }

    def save():
        args.out.write_text(json.dumps(result, indent=2))

    # 0. guardia: i codici ricalcolati qui coincidono con quelli salvati dal run su GPU (altrimenti le feature non sono quelle del run)
    for name, d in per.items():
        n = GUARD_WINDOWS * C.N_TOKENS_PER_WINDOW
        got = runner.run(d["tokens"][:n].reshape(-1, C.WINDOW_SAMPLES)[:, None, :], [SPATIAL_INDEX_FIXED], want_recon=False)["codes"]
        same = float((got == d["codes"][:, :, :n]).mean())
        result["guard"][name] = {"codes_identical_fraction": same, "n_tokens": n}
        print(f"guardia {name}: codici identici a quelli del run GPU = {same:.4f}", flush=True)
    save()

    # 1. feature continue per token, solo per (a) (cache: minuti di CPU per dataset, non si rifanno). (b) le calcola da se' su 200 finestre.
    feats = {}
    for name, d in per.items() if args.only in ("a", "both") else ():
        key = hashlib.sha256(np.ascontiguousarray(d["tokens"]).tobytes()).hexdigest()[:12]
        cache = args.cache / f"{name}_{key}.npy"
        if cache.exists():
            feats[name] = np.load(cache)
        else:
            t0 = time.time()
            feats[name] = C.token_features(fn, d["tokens"], args.batch)
            np.save(cache, feats[name])
            print(f"feature {name}: {feats[name].shape} in {time.time() - t0:.0f} s", flush=True)

    # 2. (a) sonda dataset-ID
    if args.only in ("a", "both"):
        res = C.dataset_id_probe_continuous(per, feats, seed=args.seed)
        res["reproduces_frozen_v3"] = {
            "bands": [res["frozen_bands"]["balanced_accuracy"], rep["v3"]["bands"]["balanced_accuracy"]],
            "codes": [res["frozen_codes"]["balanced_accuracy"], rep["v3"]["codes"]["balanced_accuracy"]],
        }
        res["reproduces_frozen_v3"]["ok"] = all(abs(a - b) < 0.005 for a, b in (res["reproduces_frozen_v3"]["bands"], res["reproduces_frozen_v3"]["codes"]))
        result["dataset_id"] = res
        save()
        print("\n(a) sonda dataset-ID (accuratezza bilanciata, test; caso = %.3f)" % (1 / len(res["datasets"])))
        for k, v in res.items():
            if isinstance(v, dict) and "balanced_accuracy" in v:
                print(f"  {k:34s} {v['balanced_accuracy']:.3f}  ci95=[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]  {v['model']}")
        print(f"  V3 congelato riprodotto (bande, codici) = {res['reproduces_frozen_v3']['ok']}  {res['reproduces_frozen_v3']}", flush=True)

    # 3. (b) stabilita' delle feature, tutti i dataset
    if args.only in ("b", "both"):
        print("\n(b) coseno feature pulite-vs-rumorose (mediana per ramo | 5% piu' basso), contro coseno fra token diversi")
        for name, d in per.items():
            floor = M.noise_floor_rms(d["tokens"])
            ref_floor = rep.get("v4_per_dataset", {}).get(name, {}).get("noise_floor_rms")
            r = C.stability_continuous(fn, d["tokens"], floor, n_windows=args.stability_windows, seed=args.seed, batch=args.batch)
            r["noise_floor_matches_run_report"] = None if ref_floor is None else bool(abs(floor - ref_floor) <= 1e-4 * abs(ref_floor))
            result["stability"][name] = r
            save()
            print(f"  {name}  (soglia di rumore {floor:.4f}, coincide col run: {r['noise_floor_matches_run_report']})")
            for lvl, v in r["noise"].items():
                print(f"    {lvl:11s} mediana {np.round(v['median_per_branch'], 3)} | q05 {np.round(v['q05_per_branch'], 3)}")
            b = r["between_different_tokens"]
            print(f"    token diversi mediana {np.round(b['median_per_branch'], 3)} | q95 {np.round(b['q95_per_branch'], 3)}", flush=True)
    print(f"\nrisultati in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
