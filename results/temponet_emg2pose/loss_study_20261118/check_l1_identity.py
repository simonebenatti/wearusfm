"""Confronto del percorso L1: script a ee118a25 contro script nuovo (--loss l1 esplicito), stessi argomenti, CPU, dati sintetici dei test."""
import importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np
import torch

WT = Path(sys.argv[1]); OLD = Path(sys.argv[2])
sys.path.insert(0, str(WT / "src")); sys.path.insert(0, str(WT / "tests" / "cpu")); sys.path.insert(0, str(WT / "scripts"))
import test_temponet_emg2pose as TT  # helper dei test (dataset sintetico)
import temponet_emg2pose as NEW

spec = importlib.util.spec_from_file_location("temponet_emg2pose_old", OLD / "temponet_emg2pose_old.py")
OLDM = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLDM)

tmp = Path(tempfile.mkdtemp())
csv, proc, pose, manifest = TT._dataset(tmp)
common = ["--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose), "--targets-manifest", str(manifest), "--device", "cpu"]
args = ["--epochs", "2", "--batch", "16", "--chunk-steps", "3", "--max-steps-per-epoch", "7"]
assert OLDM.main(["train", *common, "--out", str(tmp / "old"), *args]) == 0
assert NEW.main(["train", *common, "--out", str(tmp / "new"), *args, "--loss", "l1"]) == 0
res = {}
for f in ["model_final.pt", "checkpoints/epoch_01.pt", "checkpoints/epoch_02.pt"]:
    a = torch.load(tmp / "old" / f); b = torch.load(tmp / "new" / f)
    res[f] = all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()
ho = json.loads((tmp / "old" / "history.json").read_text()); hn = json.loads((tmp / "new" / "history.json").read_text())
skip = {"train_seconds", "ms_per_step", "val_seconds", "gather_seconds_total"}
res["history_same_keys"] = [list(r) for r in ho] == [list(r) for r in hn]
res["history_same_values"] = all(ro[k] == rn[k] for ro, rn in zip(ho, hn) for k in ro if k not in skip)
for name in ("old", "new"):
    m = OLDM if name == "old" else NEW
    assert m.main(["eval", *common, "--out", str(tmp / f"ev_{name}"), "--model-dir", str(tmp / name), "--val-only"]) == 0
eo = json.loads((tmp / "ev_old" / "eval.json").read_text()); en = json.loads((tmp / "ev_new" / "eval.json").read_text())
res["eval_val_same"] = eo["val"] == en["val"] and eo["alpha"] == en["alpha"] and eo["gate"] == en["gate"]
res["eval_new_train_loss"] = en.get("train_loss")
print(json.dumps(res, indent=1))
sys.exit(0 if all(v is True for k, v in res.items() if k != "eval_new_train_loss") else 1)
