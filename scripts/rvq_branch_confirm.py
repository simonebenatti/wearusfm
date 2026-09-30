#!/usr/bin/env python3
"""Conferma della regola dei rami sui dataset nuovi, opzione (b) (docs/decisioni.md, 30/09/2026): ogni dataset con V2 <= 2 nel run nuovo si prova DA
SOLO con classi = base di oggi + lui; entra se il livello 0 del ramo 0 resta idoneo. Solo numpy e scikit-learn (niente torch).

Prima di tutto controlla, dai due report, che il run nuovo sia confrontabile con quello di ieri (scritto prima del run, decisioni.md): stesso
checkpoint; fattore di scala uguale (entro lo 0,1%); errore mediano di emg2pose entro l'1%; rapporto di V1 entro l'1%. Se non torna, si ferma e nessun
dataset si accende.

  python3 scripts/rvq_branch_confirm.py --new-report ~/wearusfm_local/reports/step1bis_<id>.json --new-arrays ~/wearusfm_local/arrays_<id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.tokenizer_checks import branch_probe as B  # noqa: E402
from wearusfm.tokenizer_checks.run_io import CHECKPOINT_SHA256, load_arrays  # noqa: E402


def comparability(base: dict, new: dict) -> dict:
    """I controlli di confrontabilita' fra il run di ieri e quello nuovo. Ritorna {controllo: [ieri, nuovo, ok]}."""
    def rel(a, b):
        return abs(a - b) / abs(a)
    fb, fn = base["scale_calibration"]["factor"], new["scale_calibration"]["factor"]
    eb, en = base["v2"]["emg2pose"]["v2_median_nmse"], new["v2"]["emg2pose"]["v2_median_nmse"]
    vb, vn = base["v1"]["ratio_single_over_multi"], new["v1"]["ratio_single_over_multi"]
    sha_n = (new.get("provenance") or {}).get("checkpoint_sha256")
    return {
        "checkpoint_sha256": [CHECKPOINT_SHA256, sha_n, sha_n == CHECKPOINT_SHA256],
        "scale_factor": [fb, fn, rel(fb, fn) <= 0.001],
        "emg2pose_median_nmse": [eb, en, rel(eb, en) <= 0.01],
        "v1_ratio": [vb, vn, rel(vb, vn) <= 0.01],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-report", type=Path, required=True)
    ap.add_argument("--new-arrays", type=Path, required=True)
    ap.add_argument("--base-report", type=Path, default=ROOT / "results/step1bis/step1bis_59048369.json")
    ap.add_argument("--base-arrays", type=Path, default=Path.home() / "wearusfm_local/arrays_59048369")
    ap.add_argument("--codebooks", type=Path, default=Path.home() / "wearusfm_local/neurorvq_level0_codebooks.npz")
    ap.add_argument("--out", type=Path, default=None, help="default: results/step1bis/rvq_branch_confirm_<run>.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    base_rep, new_rep = json.loads(args.base_report.read_text()), json.loads(args.new_report.read_text())
    out = args.out or ROOT / "results/step1bis" / f"rvq_branch_confirm_{args.new_report.stem.replace('step1bis_', '')}.json"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} esiste gia': non lo sovrascrivo senza --overwrite")
    result: dict = {"decision": "docs/decisioni.md, D5b - conferma sui dataset nuovi, opzione (b) (30/09/2026)", "base_enabled": list(B.ENABLED_DEFAULT),
                    "new_report": args.new_report.name, "base_report": args.base_report.name}
    comp = comparability(base_rep, new_rep)
    result["comparability"] = comp
    for k, (a, b, ok) in comp.items():
        print(f"controllo {k}: ieri {a}  nuovo {b}  -> {'ok' if ok else 'NON TORNA'}")
    v2 = {n: {"ratio": d.get("v2_ratio_vs_emg2pose"), "passes": d.get("v2_passes")} for n, d in new_rep["v2"].items() if n != "emg2pose"}
    result["v2_new"] = v2
    if not all(ok for *_, ok in comp.values()):
        result["stopped"] = "il run nuovo non riproduce quello di ieri: i rapporti V2 non sono confrontabili, nessun dataset si accende"
        out.write_text(json.dumps(result, indent=2))
        print(result["stopped"])
        return 2

    z = np.load(args.codebooks)
    if str(z["checkpoint_sha256"]) != CHECKPOINT_SHA256:
        raise SystemExit("codebook estratto da un checkpoint diverso")
    base = load_arrays(args.base_arrays)
    new = load_arrays(args.new_arrays)
    candidates = sorted(n for n, d in v2.items() if d["passes"])
    print(f"\nV2 <= 2: {candidates}; V2 > 2 (restano spenti): {sorted(n for n, d in v2.items() if not d['passes'])}")
    result["confirm"] = {}
    for name in candidates:
        r = B.confirm_new_dataset(base, name, new[name], z["codebooks"], seed=args.seed)
        result["confirm"][name] = r
        b0 = r["branch0"]
        print(f"  {name}: 5 bande {r['bands']['balanced_accuracy']:.3f}, ramo 0 {b0['branch_accuracy']:.3f} (diff {b0['diff_minus_bands']:+.3f})"
              f" -> {'ENTRA' if r['enters'] else 'resta spento'}")
        out.write_text(json.dumps(result, indent=2))
    result["final_enabled"] = sorted([*B.ENABLED_DEFAULT, *(n for n, r in result["confirm"].items() if r["enters"])])
    out.write_text(json.dumps(result, indent=2))
    print(f"\nancora RVQ (livello 0, ramo 0) accesa su: {result['final_enabled']}\nrisultati in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
