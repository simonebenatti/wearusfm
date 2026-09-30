#!/usr/bin/env python3
"""Regola dei rami dell'ancora RVQ (D5b, firmata e corretta il 30/09/2026, `docs/proposta_ancora_rvq.md`): quali rami (livello 0) si predicono.

Due processi (su macOS scikit-learn va in segfault se torch e' gia' importato nello stesso processo; `--stage all`, il default, li lancia entrambi):
- `codebooks` (torch): verifica lo sha256 del checkpoint, estrae il codebook del livello 0 dei 4 rami (`quantize_b.layers.0.embedding.weight`) e
  controlla che corrisponda ai codici del run: ricalcola su CPU i codici del livello 0 di alcune finestre come fa il quantizzatore (vicino piu'
  prossimo delle feature normalizzate) e li confronta con quelli salvati dal run su GPU. Guardia tecnica, scritta prima di guardare: se meno del
  99% coincide in un ramo, ci si ferma (sul Mac il livello 0 coincideva al 99,9-100%, `results/step1bis/NOTA_TOKENIZER_LOCALE.md`).
- `probe` (scikit-learn): riproduce prima i due numeri congelati di V3 (5 bande 0,655, codici 0,964, sui 6 dataset) come controllo che gli array siano
  quelli giusti, poi applica la regola (`wearusfm.tokenizer_checks.branch_probe`).

Tutti gli ingressi stanno FUORI dal repo; l'uscita e' un JSON di risultati (default in `results/step1bis/`), che NON si sovrascrive senza --overwrite.

  ~/.venvs/wearusfm-tok/bin/python scripts/rvq_branch_probe.py --arrays ~/wearusfm_local/arrays_59048369 \\
      --repo ~/wearusfm_local/neurorvq_repo --checkpoint ~/wearusfm_local/models/NeuroRVQ_EMG_tokenizer_v1.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.tokenizer_checks import metrics as M  # noqa: E402
from wearusfm.tokenizer_checks.branch_probe import ENABLED_DEFAULT  # noqa: E402
from wearusfm.tokenizer_checks.run_io import CHECKPOINT_SHA256, argv_without_option, load_arrays, sha256_file  # noqa: E402

GUARD_WINDOWS = 8
GUARD_MIN_MATCH = 0.99
FROZEN_V3 = {"bands": 0.6547619047619048, "codes": 0.9642857142857143}


def stage_codebooks(args) -> None:
    sha = sha256_file(args.checkpoint)
    if sha != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha256 {sha} diverso da quello verificato: mi fermo")
    import torch

    from wearusfm.tokenizer_checks.neurorvq import SPATIAL_INDEX_FIXED, NeuroRVQRunner

    torch.set_num_threads(args.threads)
    state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    cb = np.stack([state[f"quantize_{b + 1}.layers.0.embedding.weight"].numpy() for b in range(4)]).astype(np.float32)
    if cb.shape[:2] != (4, M.N_CODE):
        raise SystemExit(f"codebook con forma {cb.shape}")
    runner = NeuroRVQRunner(args.repo, args.checkpoint, device="cpu")
    guard = {}
    for name, d in load_arrays(args.arrays).items():
        n = GUARD_WINDOWS * 16
        feats = runner.features(d["tokens"][:n].reshape(-1, 3200)[:, None, :], [SPATIAL_INDEX_FIXED])  # (4, n, 128)
        f = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        dist = (cb**2).sum(-1)[:, None, :] - 2 * np.einsum("bnd,bkd->bnk", f, cb)  # come il quantizzatore: argmin di |e|^2 - 2 z.e
        match = (dist.argmin(-1) == d["codes"][:, 0, :n]).mean(axis=1)
        guard[name] = [float(m) for m in match]
        print(f"guardia {name}: codici del livello 0 ricalcolati = salvati, per ramo {np.round(match, 4)}", flush=True)
        if match.min() < GUARD_MIN_MATCH:
            raise SystemExit(f"{name}: il codebook estratto non riproduce i codici del run (min {match.min():.4f} < {GUARD_MIN_MATCH}): mi fermo")
    args.codebooks.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.codebooks, codebooks=cb, checkpoint_sha256=sha, guard=json.dumps(guard))
    print(f"codebook salvato in {args.codebooks}", flush=True)


def stage_probe(args) -> None:
    from wearusfm.tokenizer_checks import branch_probe as B
    from wearusfm.tokenizer_checks.continuous_features import v3_setup

    z = np.load(args.codebooks)
    if str(z["checkpoint_sha256"]) != CHECKPOINT_SHA256:
        raise SystemExit("codebook estratto da un checkpoint diverso")
    per = load_arrays(args.arrays)
    names, codes_l, tokens_l, y, units, part = v3_setup(per, args.seed)
    frozen = {}
    for key, x in (("bands", M.band_power_features(tokens_l)), ("codes", M.code_histogram_features(codes_l))):
        r = M.dataset_id_probe(x, y, units, part, np.random.default_rng([args.seed, 3, 1]))
        frozen[key] = [r.balanced_accuracy, FROZEN_V3[key]]
    ok = all(abs(a - b) < 0.005 for a, b in frozen.values())
    print(f"V3 congelato riprodotto: {ok} {frozen}", flush=True)
    if not ok:
        raise SystemExit("gli array non riproducono V3: non sono quelli del run, mi fermo")
    res = B.branch_probe(per, z["codebooks"], enabled=args.enabled.split(","), seed=args.seed,
                         setup=(names, codes_l, tokens_l, y, units, part))
    res.update({"reproduces_frozen_v3": frozen, "guard_codebooks": json.loads(str(z["guard"])), "checkpoint_sha256": CHECKPOINT_SHA256,
                "arrays": str(args.arrays), "decision": "docs/decisioni.md, D5b - correzione della regola dei rami (30/09/2026)"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    b = res["bands"]["balanced_accuracy"]
    print(f"\n5 bande: {b:.3f} (caso {res['bands']['chance']:.3f}); soglia: ramo - bande <= {M.Y_POINTS:.2f}")
    for k, v in res["branches"].items():
        print(f"  ramo {k}: istogramma {v['histogram']['balanced_accuracy']:.3f}, vettore medio {v['codebook_mean']['balanced_accuracy']:.3f}"
              f" -> {v['branch_accuracy']:.3f} (diff {v['diff_minus_bands']:+.3f}) {'IDONEO' if v['eligible'] else 'escluso'}")
    print(f"rami idonei: {res['eligible_branches']}; l'ancora parte: {res['anchor_starts']}\nrisultati in {args.out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--repo", type=Path, help="clone di NeuroRVQ (serve allo stage codebooks)")
    ap.add_argument("--checkpoint", type=Path, help="serve allo stage codebooks")
    ap.add_argument("--codebooks", type=Path, default=Path.home() / "wearusfm_local/neurorvq_level0_codebooks.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "results/step1bis/rvq_branch_probe.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--enabled", default=",".join(ENABLED_DEFAULT))
    ap.add_argument("--stage", choices=("all", "codebooks", "probe"), default="all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.codebooks.suffix != ".npz":  # np.savez aggiungerebbe .npz e lo stage probe cercherebbe il nome senza
        ap.error(f"--codebooks deve finire in .npz, ricevuto {args.codebooks}")
    if args.stage in ("all", "probe") and args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} esiste gia': non lo sovrascrivo senza --overwrite")
    if args.stage == "all":
        if not (args.repo and args.checkpoint):
            ap.error("--stage all richiede --repo e --checkpoint")
        raw = list(argv) if argv is not None else sys.argv[1:]
        base = [sys.executable, str(Path(__file__).resolve())] + argv_without_option(raw, "--stage")
        for stage in ("codebooks", "probe"):
            rc = subprocess.run(base + ["--stage", stage]).returncode
            if rc:
                return rc
        return 0
    if args.stage == "codebooks" and not (args.repo and args.checkpoint):
        ap.error("lo stage codebooks richiede --repo e --checkpoint")
    (stage_codebooks if args.stage == "codebooks" else stage_probe)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
