import copy, hashlib, json, sys
from pathlib import Path
CL = Path(sys.argv[1])
import contrastive_run as CRN
import contrastive_prepare as CP
raw = json.loads((CL/"results/infra/contrastive_targets_20261118/review_configs/calibration.config.json").read_text())
expected = CP.build_review_configs(CP._read(CP.BASELINE))["calibration"]
canon = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
# 1) intero al posto del float: equivalente semantico, canonico diverso
d = copy.deepcopy(raw); d["loader"]["rms_window_ms"] = 100
cfg = CRN.validate_recipe(d, "calibration")
print("int 100: validate_recipe ACCEPTED; fix proposto _jsonable(raw)==expected ->", CP._jsonable(d) == expected,
      "; canonico == atteso ->", canon(d) == canon(expected))
# 2) fix proposto sul caso del rilievo
d = copy.deepcopy(raw); d.pop("target_version")
print("senza target_version: fix proposto ->", CP._jsonable(d) == expected, "; canonico == atteso ->", canon(d) == canon(expected))
# 3) la configurazione effettiva restituita (quella che va a R.train) e' identica all'attesa
print("cfg effettivo == atteso:", CP.R.config_to_dict(CRN.validate_recipe(d, "calibration")) == expected)
