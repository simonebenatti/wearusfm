"""Scettico 2, R6-1: il readout attribuisce per contenuto (cfg != expected[arm] a riga 205)?

Uso: python -B probe_readout_content.py <clone> <dir temporanea>
Isola il primo controllo di _run_evidence: stessa cartella, config CN nello slot C0 -> ValueError
'incomplete/unmatched benchmark C0'; config uguale all'atteso -> supera il controllo e si ferma dopo
(ricevuta numeric_profile assente, errore diverso).
"""
import json
import sys
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()
TMP = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(CLONE / "src"))
sys.path.insert(0, str(CLONE / "scripts"))
import contrastive_benchmark_readout as B  # noqa: E402

folder = TMP / "slot"
folder.mkdir(parents=True, exist_ok=True)
summary = {"steps": 576, "world_size": 4, "workers_per_rank": 6, "batch_per_process": 8, "accum_steps": 1, "stopped": "max_steps"}
(folder / "summary.json").write_text(json.dumps(summary))
(folder / "metrics.jsonl").write_text("".join(json.dumps({"step": s}) + "\n" for s in range(1, 577)))
expected_c0 = {"max_steps": 576, "contrastive": {"arm": "C0"}}
for label, cfg in (("config CN nello slot C0", {"max_steps": 576, "contrastive": {"arm": "CN"}}),
                   ("config C0 nello slot C0", expected_c0)):
    (folder / "config.json").write_text(json.dumps(cfg))
    try:
        B._run_evidence(folder, "C0", {"source_sha256": "0" * 64}, expected_c0, {})
        print(label, "-> nessun errore")
    except Exception as error:  # noqa: BLE001
        print(label, "->", type(error).__name__ + ":", str(error)[:120])
