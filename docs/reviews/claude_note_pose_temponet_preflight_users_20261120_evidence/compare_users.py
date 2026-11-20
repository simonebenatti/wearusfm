# Confronta gli utenti letti dalla cache (cache_users.json) con quelli del piano rilasciato a c531e76c.
import json
import re
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
cache = json.loads((here / "cache_users.json").read_text())
plan = json.loads(subprocess.check_output(
    ["git", "show", "c531e76c:results/infra/pose_temponet_execution_20261119/plan_released_20261120.json"]))
SAFE = re.compile(r"[^0-9A-Za-z_.@-]")  # wearusfm.ingest.emg2pose._SAFE_RE
for split in ("train", "val"):
    cu, pu = cache[split]["users"], plan["producers"][split]["users"]
    print(split, "finestre", cache[split]["windows"], "| utenti cache", len(cu), "| utenti ricevuta", len(pu))
    print("  esempi:", cu[:2], pu[:2])
    print("  uguali come stringhe:", cu == pu)
    print("  uguali con la cartella del produttore u+ID:", sorted("u" + SAFE.sub("_", u) for u in cu) == pu)
    print("  ID tutti di 10 esadecimali:", all(re.fullmatch(r"[0-9a-f]{10}", u) for u in cu))
sys.exit(0)
