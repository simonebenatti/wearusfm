"""Review Claude del prototipo Rank-N-Contrast (16/11/2026): riconteggio mio dei numeri principali sui 19.618 passi veri.

Legge due file FUORI dal repository, sul Mac (nessun segnale, solo indici e conteggi):
- ~/wearusfm_local/logs/review6_remote/control30m/metrics.jsonl: log del controllo 30M a 19.618 passi con `origin`
  ([sessione, trial, inizio] per finestra, 32 finestre per passo nell'ordine dei rank: training/run.py righe 514-515 a e114a78);
- ~/wearusfm_local/manifests_derived_20261015/derived_final/manifest_v1_1_nested_1.json.gz: righe `pretraining` nell'ordine del manifest.
Provenienza del log: i primi 500 passi coincidono con results/infra/scaling_300m_20261104/production_memory_63527446 su origin/main
(controllo dell'agente verificatore, non rifatto da me). Micro-batch = 8 finestre consecutive (4 processi x 8, accumulo 1).
"""
import gzip, json, sys
from pathlib import Path
from collections import Counter
man = Path.home() / "wearusfm_local/manifests_derived_20261015/derived_final/manifest_v1_1_nested_1.json.gz"
doc = json.loads(gzip.open(man, "rb").read())
col = doc["columns"]; i_split, i_ds = col.index("split"), col.index("dataset")
pose = [r[i_ds] == "emg2pose" for r in doc["rows"] if r[i_split] == "pretraining"]
log = Path.home() / "wearusfm_local/logs/review6_remote/control30m/metrics.jsonl"
mb = Counter(); steps = 0; step_null = 0; step_one = 0; tot_pose = 0; lone = 0; glob_le1 = 0; pw_le1 = 0
rec2 = rec3 = 0; nmb = 0; glob = Counter()
for line in open(log):
    r = json.loads(line); o = r["origin"]; assert len(o) == 32
    steps += 1; act = 0; g = 0
    for k in range(4):
        w = [s for s, _, _ in o[8 * k:8 * k + 8] if pose[s]]
        m = len(w); mb[min(m, 3)] += 1; nmb += 1; tot_pose += m; g += m
        lone += (m == 1); act += (m >= 2)
        rec2 += (len(set(w)) >= 2); rec3 += (len(set(w)) >= 3)
    step_null += (act == 0); step_one += (act == 1)
    glob[min(g, 3)] += 1
    pw_le1 += (r["pose_windows"] <= 1)
print(f"passi {steps}, micro-batch {nmb}, finestre emg2pose {tot_pose} ({tot_pose/(32*steps):.4f}), media per micro-batch {tot_pose/nmb:.3f}")
print("P(m=0,1,2,>=3) per micro-batch:", [round(mb[k]/nmb, 4) for k in range(4)], "| P(m<=1) =", round((mb[0]+mb[1])/nmb, 4))
print(f"registrazioni di posa distinte per micro-batch: >=2 {rec2/nmb:.4f}, >=3 {rec3/nmb:.4f}")
print(f"finestre di posa sole nel micro-batch: {lone/tot_pose:.4f}")
print(f"passi con cross nullo su tutti e 4 i processi: {step_null} ({step_null/steps:.4f}); attivo su uno solo: {step_one/steps:.4f}")
print("finestre emg2pose nel batch globale (0,1,2,>=3):", [round(glob[k]/steps, 4) for k in range(4)])
print(f"passi con pose_windows (posa valida, dal log) <= 1: {pw_le1} ({pw_le1/steps:.4f})")

# quota dei positivi dentro la propria finestra con candidati GLOBALI e positivi «all» (validita' ideale, 20 ancore per finestra)
import statistics
share = []; one = 0; n_g = 0
for line in open(log):
    o = json.loads(line)["origin"]
    g = sum(pose[s] for s, _, _ in o)
    if g >= 1:
        share.append(19 / (20 * g - 1)); n_g += 1; one += (g == 1)
print(f"candidati globali, all: quota media dei positivi nella propria finestra {statistics.mean(share):.4f} (mediana {statistics.median(share):.4f}); passi con una sola finestra emg2pose {one/steps:.4f}")
# stessa quota coi candidati del micro-batch (prototipo, all)
share = []
for line in open(log):
    o = json.loads(line)["origin"]
    for k in range(4):
        m = sum(pose[s] for s, _, _ in o[8 * k:8 * k + 8])
        share += [19 / (20 * m - 1)] * m
print(f"candidati del micro-batch, all: quota media per finestra {statistics.mean(share):.4f}")
