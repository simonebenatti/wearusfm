"""Scettico 1, P3-N1: coerenza fra dichiarazioni del piano e config generati (sola lettura)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root / "src"))
import contrastive_prepare as P  # noqa: E402

plan = json.loads(P.PLAN.read_text())
print("PLAN =", P.PLAN.relative_to(root))
configs = P.build_review_configs(json.loads(P.BASELINE.read_text()))


def triple(cfg):
    return (cfg["loader"]["rms_window_ms"], cfg["loader"]["ms_fast_window_ms"], cfg["target_version"])


def cli(args):
    assert len(args) == 4 and args[0] == "--rms-window-ms" and args[2] == "--ms-fast-window-ms", args
    return (float(args[1]), float(args[3]))


gen = {name: triple(cfg) for name, cfg in configs.items()}
print("generated:", gen)
common = plan["common"]
decl_common = (common["rms_window_ms"], common["ms_fast_window_ms"], common["target_version"])
print("plan.common:", decl_common, "cli", cli(common["target_cli_args"]))
new = plan["target_recipe_amendment"]["new_recipe"]
decl_new = (new["rms_window_ms"], new["ms_fast_window_ms"], new["target_version"])
print("plan.target_recipe_amendment.new_recipe:", decl_new)
ok = True
for name in ("C0", "CN", "CR", "calibration"):
    ok &= gen[name] == decl_common == decl_new
    ok &= gen[name][:2] == cli(common["target_cli_args"])
for arm in ("C0", "CN", "CR"):
    a = plan["arms"][arm]
    decl_arm = (a["rms_window_ms"], a["ms_fast_window_ms"])
    print(f"plan.arms.{arm}:", decl_arm, "cli", cli(a["target_cli_args"]), "target_version in arm:", "target_version" in a)
    ok &= decl_arm == gen[arm][:2] == cli(a["target_cli_args"])
print("plan declarations == generated configs:", ok)

hist = plan["target_recipe_amendment"]["historical_baseline"]
base = P.R.config_to_dict(P.config_from_dict(json.loads(P.BASELINE.read_text())))
print("historical_baseline declared:", (hist["rms_window_ms"], hist["ms_fast_window_ms"], hist["target_version"]),
      "baseline restored:", triple(base))

# Committed review configs vs builder output (same serialization as 'prepare').
rc = root / "results/infra/contrastive_targets_20261118/review_configs"
for name, cfg in configs.items():
    committed = (rc / (name + ".config.json")).read_text()
    print(f"review_configs/{name}.config.json byte-identical to builder:",
          committed == json.dumps(cfg, indent=2, allow_nan=False) + "\n")
