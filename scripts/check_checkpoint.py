"""Guardia fra un training e la sua valutazione (catena di valutazione della finestra 1, accordo GPT-Claude del 15/10/2026): il checkpoint
atteso e' stato pubblicato da un run finito, senza STOP, e ha l'identita' dichiarata. Uscita 0 se tutto torna, 3 altrimenti (la valutazione
che dipende da questa guardia con afterok allora non parte mai: nessun rilancio alla cieca).

Verifiche: nella cartella del run nessun file STOP; `summary.json` con `stopped` = `max_steps` e almeno `--min-steps` passi; il checkpoint
esiste e non e' vuoto; se dato, il suo SHA-256 e' `--sha256`. Scrive un rapporto JSON (percorso, dimensione, SHA-256, passi, esito).

Uso: python scripts/check_checkpoint.py <cartella del run> <nome del checkpoint> --report <rapporto.json> [--min-steps N] [--sha256 HEX]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def check(run_dir: Path, name: str, min_steps: int | None = None, sha256: str | None = None) -> dict:
    out = {"run_dir": str(run_dir), "checkpoint": name, "ok": False, "problems": []}
    stop = run_dir / "STOP"
    if stop.exists():
        out["problems"].append(f"STOP nel run: {stop.read_text().strip()[:300]}")
    summary = run_dir / "summary.json"
    if not summary.exists():
        out["problems"].append("summary.json assente: il run non ha finito un job")
    else:
        s = json.loads(summary.read_text())
        out["steps"], out["stopped"] = s.get("steps"), s.get("stopped")
        if s.get("stopped") != "max_steps":
            out["problems"].append(f"run non completo: stopped = {s.get('stopped')!r}")
        if min_steps is not None and (s.get("steps") or 0) < min_steps:
            out["problems"].append(f"{s.get('steps')} passi, attesi almeno {min_steps}")
    ckpt = run_dir / name
    if not ckpt.exists() or ckpt.stat().st_size == 0:
        out["problems"].append(f"checkpoint {name} assente o vuoto")
    else:
        h = hashlib.sha256()
        with ckpt.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        out["size"], out["sha256"] = ckpt.stat().st_size, h.hexdigest()
        if sha256 is not None and out["sha256"] != sha256:
            out["problems"].append(f"SHA-256 {out['sha256']} diverso da quello atteso {sha256}")
    out["ok"] = not out["problems"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("checkpoint")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--min-steps", type=int, default=None)
    ap.add_argument("--sha256", default=None)
    args = ap.parse_args(argv)
    out = check(args.run_dir, args.checkpoint, args.min_steps, args.sha256)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1))
    print(("checkpoint verificato: " if out["ok"] else "GUARDIA FALLITA: ") + json.dumps(out))
    return 0 if out["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
