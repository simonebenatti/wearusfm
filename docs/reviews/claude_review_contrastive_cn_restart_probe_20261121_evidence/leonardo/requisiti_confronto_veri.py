import hashlib, json, sys
from pathlib import Path
import torch
R = Path("/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118")
def sha(p):
    with open(p, "rb") as s: return hashlib.file_digest(s, "sha256").hexdigest()
out = {}
for label, d in (("CN", R / "benchmark/CN"), ("CN_A", R / "reloads/CN_A")):
    names = [f"trace-{r}.jsonl" for r in range(4)] + [f"numeric_profile.rank{r}.json" for r in range(4)] + ["metrics.jsonl", "config.json", "summary.json"]
    names += [f"trace-{r}.ddp.jsonl" for r in range(4)]
    if label == "CN_A": names.append("start.json")
    out[label] = {n: (sha(d / n) if (d / n).is_file() else None) for n in names}
print(json.dumps(out, sort_keys=True))
a = torch.load(str(R / "reloads/CN_A/checkpoint.pt"), map_location="cpu", weights_only=False, mmap=True)
cfg = json.loads((R / "reloads/CN_A/config.json").read_text())
print("config del payload == config.json:", a["config"] == cfg, "| tipi", type(a["config"]).__name__, type(cfg).__name__)
if a["config"] != cfg:
    ka, kc = set(a["config"]), set(cfg)
    print("  chiavi solo payload:", sorted(ka - kc)[:20], "| solo json:", sorted(kc - ka)[:20])
    print("  chiavi con valore diverso:", [k for k in ka & kc if a["config"][k] != cfg[k]][:20])
s = json.loads((R / "reloads/CN_A/summary.json").read_text())
print("summary: steps", s.get("steps"), "stopped", s.get("stopped"), "| STOP presente:", (R / "reloads/CN_A/STOP").exists())
st = json.loads((R / "reloads/CN_A/start.json").read_text())
print("start.json:", json.dumps(st)[:700])
print("step", a["step"], "world_size", a["world_size"])
print("fork presente accanto al continuo:", (R / "benchmark/CN/checkpoint_step0000544.pt").is_file(), "| uscita della prova gia' presente:", (R / "cn_restart_probe_20261121").exists())
import shutil
u = shutil.disk_usage(str(R)); print("disk_usage (non e' la quota):", round(u.free / 2**30, 1), "GiB liberi")
