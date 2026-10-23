#!/usr/bin/env python3
"""Checkpoint di una rete con pesi casuali con l'architettura di un run (baseline della rete casuale, 23/10/2026): stessa configurazione del
modello del `config.json` del run, inizializzazione dal seme dato (un thread, generatore della CPU, come `harness.random_encoder`), studente =
teacher, nessun aggiornamento. Si valuta con le stesse estrazioni e sonde dei checkpoint allenati (`FF.load_model`).
NON e' il passo 0 del run: i run della finestra 1 non registrano il loro passo 0 e l'uguaglianza non e' verificata (il trainer crea il modello
dopo `torch.manual_seed(cfg.seed)`, ma con codice e thread di allora). E' una rete casuale qualunque della stessa architettura.

  python3 scripts/random_checkpoint.py --config RUN/config.json --seed 0 --out casuale_s0/checkpoint.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.model.fm import FMConfig, WearUsFM  # noqa: E402
from wearusfm.training.pairing import state_sha256  # noqa: E402


def random_state(config: dict, seed: int) -> dict:
    """{student, teacher, config, step=0}: il modello della configurazione inizializzato dal seme, senza leggere pesi di nessun run."""
    mcfg = config["model"]
    unknown = set(mcfg) - {f.name for f in fields(FMConfig)}
    if unknown:
        raise ValueError(f"campi del modello sconosciuti: {sorted(unknown)}")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            model = WearUsFM(FMConfig(**mcfg))
    finally:
        torch.set_num_threads(threads)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    return {"student": sd, "teacher": {k: v.clone() for k, v in sd.items()}, "config": {**config, "seed": int(seed)}, "step": 0,
            "random_init": {"seed": int(seed), "config_seed": config.get("seed"), "optimization_updates": 0, "is_run_step0": False}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="config.json del run di cui si copia l'architettura")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    prov_path = args.out.with_name("random_provenance.json")
    if args.out.exists() or prov_path.exists():
        raise ValueError(f"{args.out}: checkpoint o provenienza gia' presenti")
    config = json.loads(args.config.read_text())
    state = random_state(config, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    prov = {"protocol": "rete_casuale_v1", "config_from": str(args.config), "is_run_step0": False,
            "note": "rete casuale della stessa architettura; non e' il passo 0 del run (non registrato, uguaglianza non verificata)",
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "seed": args.seed,
            "state_sha256": state_sha256(state["teacher"]), "checkpoint_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    prov_path.write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov))
    return 0


if __name__ == "__main__":
    sys.exit(main())
