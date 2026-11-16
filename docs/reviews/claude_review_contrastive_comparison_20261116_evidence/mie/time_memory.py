"""Review Claude del pacchetto contrastivo db5ac41: costo CPU per update della memoria InfoNCE alle taglie vere.

Classi VERE della consegna (CandidateQueue, admissible_mask, WindowIdentity, physical_pair_mask); identita' ricostruite dal flusso
vero (log del controllo 30M e manifest, fuori dal repository: solo indici). Chiavi casuali a 128 dimensioni: si misura il tempo, non
la perdita. Per ogni update: prepare/snapshot (una volta per rank), admissible_mask per le 8 query locali di un rank, commit.
Uso: PYTHONPATH=<albero db5ac41>/src python time_memory.py [update da cronometrare]
"""
import gzip, json, sys, time
from pathlib import Path
import torch
from wearusfm.training.contrastive_memory import CandidateQueue, WindowIdentity, admissible_mask
from wearusfm.training.contrastive_objectives import physical_pair_mask

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
man = Path.home() / "wearusfm_local/manifests_derived_20261015/derived_final/manifest_v1_1_nested_1.json.gz"
doc = json.loads(gzip.open(man, "rb").read()); col = {c: i for i, c in enumerate(doc["columns"])}
rows = [r for r in doc["rows"] if r[col["split"]] == "pretraining"]
log = Path.home() / "wearusfm_local/logs/review6_remote/control30m/metrics.jsonl"
queue = CandidateQueue(1024, 256)
torch.manual_seed(0)
t_snap = t_adm = t_commit = t_pair = 0.0; sizes = []; n_timed = 0
with open(log) as f:
    for step, line in enumerate(f):
        if step >= 300 + N:
            break
        o = json.loads(line)["origin"]
        ids = []
        for s, tr, st in o:
            r = rows[s]; fs = float(r[col["fs_hz"]])
            ids.append(WindowIdentity(r[col["dataset"]], str(r[col["subject"]]), str(r[col["session"]]), tr, int(st), int(st + 4 * fs), fs))
        keys = torch.randn(32, 128)
        timed = step >= 300
        a = time.perf_counter()
        snap = queue.snapshot(step, keys, ids, None)
        b = time.perf_counter()
        mask = admissible_mask(ids[:8], snap.identities, gap_seconds=4.0)      # le 8 query di UN rank
        c = time.perf_counter()
        queue.commit(keys, ids, None, step=step)
        d = time.perf_counter()
        # maschera delle coppie RnC: ancore delle finestre emg2pose del batch globale (20 per finestra, stride 8, fase 4)
        pose = [i for i, x in enumerate(ids) if x.dataset == "emg2pose"]
        si = torch.tensor([i for i in pose for _ in range(20)]); pi = torch.tensor([4 + 8 * k for _ in pose for k in range(20)])
        e = time.perf_counter()
        if len(pose):
            physical_pair_mask(ids, si, pi, 25.0, "cpu")
        g = time.perf_counter()
        if timed:
            t_snap += b - a; t_adm += c - b; t_commit += d - c; t_pair += g - e; sizes.append((len(snap.identities), len(pose) * 20)); n_timed += 1
m = sum(s[0] for s in sizes) / n_timed; a_ = sum(s[1] for s in sizes) / n_timed
print(f"update cronometrati {n_timed} (dopo 300 di riempimento); candidati in memoria + batch: media {m:.0f}; ancore RnC globali: media {a_:.0f}, massimo {max(s[1] for s in sizes)}")
print(f"per update e per rank, secondi: snapshot {t_snap / n_timed:.4f} | admissible_mask (8 query) {t_adm / n_timed:.4f} | commit {t_commit / n_timed:.4f} | physical_pair_mask {t_pair / n_timed:.4f}")
tot = (t_snap + t_adm + t_commit + t_pair) / n_timed
print(f"totale per update e per rank {tot:.4f} s; passo storico di T100-G 0,777 s (15.250 s / 19.618 update): +{100 * tot / 0.777:.1f}% se questo lavoro e' seriale col passo")
