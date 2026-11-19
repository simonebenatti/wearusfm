import copy, hashlib, json, sys
from pathlib import Path
CL = Path(sys.argv[1])
import contrastive_run as CRN
import contrastive_prepare as CP
from wearusfm.training import run as R
assert Path(CP.ROOT).resolve() == CL.resolve(), (CP.ROOT, CL)
raw = json.loads((CL/"results/infra/contrastive_targets_20261118/review_configs/calibration.config.json").read_text())
expected = CP.build_review_configs(CP._read(CP.BASELINE))["calibration"]
canon = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print("file == expected (raw):", raw == expected)
def attempt(label, d):
    try:
        cfg = CRN.validate_recipe(d, "calibration")
        print(f"{label}: ACCEPTED target_version={cfg.target_version} rms={cfg.loader.rms_window_ms} fast={cfg.loader.ms_fast_window_ms}"
              f" | audit/cuda canonical == expected: {canon(d) == canon(expected)}")
    except Exception as e:
        print(f"{label}: REJECTED {type(e).__name__}: {e}")
attempt("A file pubblicato", copy.deepcopy(raw))
d = copy.deepcopy(raw); d.pop("target_version"); attempt("B senza target_version", d)
d = copy.deepcopy(raw); d["loader"].pop("rms_window_ms"); d["loader"].pop("ms_fast_window_ms"); attempt("C senza finestre", d)
d = copy.deepcopy(raw); d["loader"].pop("rms_window_ms"); attempt("C1 senza rms_window_ms", d)
d = copy.deepcopy(raw); d["target_version"] = "step1_psd_power_shape_energy_v1"; attempt("D target_version v1", d)
d = copy.deepcopy(raw); d.pop("target_version"); d["loader"]["rms_window_ms"]=25.; d["loader"]["ms_fast_window_ms"]=50.; attempt("E 25/50 senza target_version", d)
# generalita': altri campi con default nel dataclass
for k in ("readout_version", "amp_bf16", "stream_segment", "accum_steps", "keep_calibration"):
    if k in raw:
        d = copy.deepcopy(raw); d.pop(k); attempt(f"F senza {k}", d)
# campo loader con default
for k in ("pose_unprompted",):
    if k in raw["loader"]:
        d = copy.deepcopy(raw); d["loader"].pop(k); attempt(f"G senza loader.{k}", d)
