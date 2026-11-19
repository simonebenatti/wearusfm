"""Riproduzione indipendente di P1-N2 (scettico 1). Solo lettura del clone; scrive solo sotto TMP."""
import copy, json, sys
from pathlib import Path
import torch

C, TMP = Path(sys.argv[1]), Path(sys.argv[2])
import contrastive_prepare as P
import contrastive_run as CRUN
from wearusfm.training import run as R
from wearusfm.model.spectral_targets import target_version_for_fast_window as tvf, LEGACY_TARGET_VERSION, TARGET_VERSION

def triple(cfg):
    return (cfg.loader.rms_window_ms, cfg.loader.ms_fast_window_ms, cfg.target_version)

def coherent(cfg):
    return cfg.target_version == tvf(cfg.loader.ms_fast_window_ms)

def attempt(label, fn):
    try:
        out = fn()
        print(f"{label}: ACCETTATO -> {out!r}"[:300])
    except Exception as e:
        print(f"{label}: RIFIUTATO {type(e).__name__}: {e}"[:300])

print("TARGET_VERSION default =", TARGET_VERSION, "| tvf(50) =", tvf(50.), "| tvf(125) =", tvf(125.))
base = json.loads((C / "results/infra/tokenizer_execution_20261103/19618/T100-G/config.json").read_text())
cb = P.config_from_dict(base)
print("T100-G storico:", triple(cb), "coerente:", coherent(cb))

A = copy.deepcopy(base); del A["target_version"]
print("with_legacy_target_windows(A) contiene target_version?", "target_version" in R.with_legacy_target_windows(A))
print("_config_with_defaults(A) contiene target_version?", "target_version" in R._config_with_defaults(A))
cA = P.config_from_dict(A)
print("A = T100-G senza target_version:", triple(cA), "coerente:", coherent(cA))

outA = TMP / "train_out_A"
attempt("R.train(A) guardia", lambda: R.train(cA, Path("/nonexistent/manifest"), [], outA))
print("  cartella del run creata?", outA.exists())
attempt("P.build_review_configs(A)", lambda: P.build_review_configs(A))

# config reale anteriore a 0e8ba050 (l'unico nel repository senza target_version)
pre = json.loads((C / "results/passo6/w1_collaudo_59624004.json").read_text())["config"]
print("pre-07/10 multiscale_anchor:", pre["loader"].get("multiscale_anchor"), "| target_version presente:", "target_version" in pre)
attempt("P.config_from_dict(pre-07/10)", lambda: triple(P.config_from_dict(pre)))
attempt("P.build_review_configs(pre-07/10)", lambda: P.build_review_configs(pre))

# nuova calibrazione senza finestre e senza target_version -> 25/50/v3; il dispatcher la rifiuta
cal = json.loads((C / "results/infra/contrastive_targets_20261118/review_configs/calibration.config.json").read_text())
calX = copy.deepcopy(cal)
for k in ("rms_window_ms", "ms_fast_window_ms"):
    del calX["loader"][k]
del calX["target_version"]
print("calibrazione nuova senza finestre e senza tv:", triple(P.config_from_dict(calX)))
attempt("validate_recipe(calX, calibration)", lambda: triple(CRUN.validate_recipe(calX, "calibration")))
attempt("validate_recipe(cal, calibration) [controllo positivo]", lambda: triple(CRUN.validate_recipe(cal, "calibration")))

# il trainer, dal 07/10, legge un target_version ASSENTE nel checkpoint come definizione PSD diversa (non v1)
outR = TMP / "train_out_resume"; outR.mkdir(parents=True, exist_ok=True)
prev = copy.deepcopy(base); del prev["target_version"]
torch.save({"config": prev}, outR / "checkpoint.pt")
attempt("R.train(T100-G v1, ripresa da checkpoint senza target_version)", lambda: R.train(cb, Path("/nonexistent/manifest"), [], outR))
print("config_to_dict scrive sempre target_version?", "target_version" in R.config_to_dict(R.sanity_config()))
