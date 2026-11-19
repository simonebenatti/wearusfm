import json, sys
from pathlib import Path
CL = Path(sys.argv[1])
import contrastive_run as CRN
for rel in sys.argv[2:]:
    p = CL / rel
    d = json.loads(p.read_text())
    tv = d.get("target_version"); w = (d["loader"].get("rms_window_ms"), d["loader"].get("ms_fast_window_ms"))
    try:
        CRN.validate_recipe(d, "calibration"); print(rel, tv, w, "ACCEPTED")
    except Exception as e:
        print(rel, tv, w, "REJECTED", e)
