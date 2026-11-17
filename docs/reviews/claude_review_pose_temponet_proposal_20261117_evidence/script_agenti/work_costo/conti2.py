"""Conti rifatti nel secondo tentativo (dimensione 5, costo e fattibilita').

Solo aritmetica sui numeri della proposta c15aac9 e sulle uscite dei micro-benchmark SINTETICI
del Mac (bench_*_r2.json, prodotti in questo tentativo da bench_heads_dev.py).
Uso: python conti2.py <dir con i json>
"""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
GiB = 2**30

print("== a) tabella del par. 9 (righe 151-158 del protocollo) ==")
# nome, GPU allocate, CPU allocate, limite cumulativo in ore
phases = [
    ("checkout, manifest/hash, supporti", 0, 2, F(15, 60)),
    ("benchmark CPU train-only", 0, 8, F(15, 60)),
    ("benchmark GPU train-only", 1, 8, F(15, 60)),
    ("selezione + rifit", 1, 8, F(3)),
    ("inferenza val", 1, 8, F(30, 60)),
    ("riferimenti, metriche, bootstrap", 0, 8, F(1)),
    ("raccolta e conservazione", 0, 2, F(15, 60)),
]
reserve_gpuh, reserve_cpu_only_coreh = F(1, 4), F(1)
gpuh = sum(g * h for _, g, c, h in phases) + reserve_gpuh
cpu_only = sum(c * h for _, g, c, h in phases if g == 0) + reserve_cpu_only_coreh
cpu_in_gpu_jobs = sum(c * h for _, g, c, h in phases if g > 0) + 8 * reserve_gpuh
for n, g, c, h in phases:
    kind = "solo CPU" if g == 0 else "job GPU"
    loc = (8 * g * h) if g else (c * h)
    print(f"  {n:34s} {kind:8s} GPUh={float(g*h):5.2f} coreh={float(c*h):5.2f} ore locali={float(loc):5.2f}")
print(f"  riserva: GPUh {float(reserve_gpuh)} (coreh dei job GPU {float(8*reserve_gpuh)}), coreh solo CPU {float(reserve_cpu_only_coreh)}")
print(f"  GPUh totali = {gpuh} ; coreh solo CPU = {cpu_only} ; coreh dentro i job GPU = {cpu_in_gpu_jobs} ; coreh allocate totali = {cpu_only + cpu_in_gpu_jobs}")
local = 8 * gpuh + cpu_only  # billing 8/h per 1 GPU+8 CPU; billing = n. CPU sulla seriale (AllocTRES)
print(f"  ore locali (billing 8 per ora di job 1 GPU+8 CPU; billing = CPU per i job seriali) = {local}")
print(f"  ore locali dei soli job GPU = {8*gpuh} ; dei soli job CPU = {cpu_only}")
print(f"  somma dei limiti di tempo delle fasi = {float(sum(h for *_, h in phases))} h (riserva esclusa)")
res0911 = 691863
print(f"  44 ore locali sul residuo committato del 09/11 ({res0911}): {float(local/res0911*100):.5f} %")
# TIMEOUT: un job oltre il limite viene chiuso ~3 min dopo (63593508: limite 1:30:00, durata 1:33:04)
over = F(3 * 60 + 4, 3600)
print(f"  sforamento osservato oltre il limite (63593508): {float(over*3600):.0f} s = {float(over):.4f} h per job;"
      f" su un job da 1 GPU {float(over):.4f} GPUh = {float(8*over):.3f} ore locali; la riserva GPU ne copre {float(reserve_gpuh/over):.1f}")

print("== c) byte e memoria ==")
NV, NT, P, Dm, J = 18548, 1907, 160, 384, 20
pred = NV * P * J * 4
print(f"  una predizione val FP32 = {NV}*{P}*{J}*4 = {pred} byte ; 15 = {15*pred} byte = {15*pred/GiB:.6f} GiB")
print(f"  stessa a 148 centri (crop6): {NV*148*J*4} byte ; 15 = {15*NV*148*J*4/GiB:.4f} GiB")
tr, va = 474412856, 4593977338
refs = {"post_ridge.coefficients": 59367, "post_ridge.predictions": 418440754, "trained_head.predictions": 234048726}
print(f"  cache train+val = {tr+va} byte = {(tr+va)/GiB:.4f} GiB ; riferimenti = {sum(refs.values())} byte = {sum(refs.values())/GiB:.4f} GiB ;"
      f" input pinnati in tutto = {tr+va+sum(refs.values())} byte = {(tr+va+sum(refs.values()))/GiB:.4f} GiB")
def members(n):
    view = n * P * Dm * 2
    return {"pre FP16": view, "lat FP16": view, "key_time_valid bool": n * P, "pred FP32": n * P * J * 4,
            "target FP32": n * P * J * 4, "mask bool": n * P * J, "std FP32": n * J * 4, "pose_var FP32": n * 4}
for name, n, disk in (("val", NV, va), ("train", NT, tr)):
    m = members(n)
    tot = sum(m.values())
    print(f"  {name}: membri numerici non compressi (senza meta_json/config_json) = {tot} byte = {tot/GiB:.4f} GiB ;"
          f" su disco {disk} byte ; rapporto disco/non compresso = {disk/tot:.4f}")
    print(f"      una vista FP16 = {m['lat FP16']} byte = {m['lat FP16']/GiB:.4f} GiB ; due viste FP16 = {2*m['lat FP16']/GiB:.4f} GiB ;"
          f" una vista intera in FP32 = {n*P*Dm*4/GiB:.4f} GiB ; due viste FP32 = {2*n*P*Dm*4/GiB:.4f} GiB")
# riferimenti storici: prediction float64 su 148 centri (consolidation_long_pose_runtime.py: held['pred'][:, 6:-6].astype(np.float64))
ref_raw = NV * 148 * J * 8
print(f"  riferimento storico non compresso (float64, 148 centri) = {ref_raw} byte = {ref_raw/GiB:.4f} GiB ;"
      f" rapporto post_ridge {418440754/ref_raw:.4f}, trained_head {234048726/ref_raw:.4f}")
print(f"  24 GiB = {24*1024} MiB, limite seriale 30800 MB: {'entro' if 24*1024 <= 30800 else 'oltre'}")
params = {"L0": 8468, "MLP": 65300, "L13": 100628, "TCN_post": 65204, "TCN_pre": 65204}
w = sum(params.values()) * 3 * 4
print(f"  pesi FP32 dei 15 rifit = {w} byte = {w/2**20:.2f} MiB")
held_all = 20 * 50 * (NT // 2) * P * J * 4
print(f"  (se si salvassero le predizioni held di ogni epoca dei 20 fit di selezione: ~{held_all/GiB:.1f} GiB, oltre il tetto di 8)")
print(f"  quota 16/11 (dato storico): 1024 - 933,1 = {1024-933.1:.1f} GiB liberi; dopo 8 GiB = {1024-933.1-8:.1f}")

print("== b) matrice dei fit ==")
fold = NT / 2
steps_fold = -(-int(fold) // 32)
steps_full = -(-NT // 32)
sel_steps = 20 * 50 * steps_fold
ref_steps = 15 * 50 * steps_full
print(f"  passi per epoca: fold ~{steps_fold} (~{fold:.0f} finestre), full {steps_full}; a 50 epoche: selezione {sel_steps} + rifit {ref_steps} = {sel_steps+ref_steps}")
print(f"  per testa: epoche-equivalenti di train sulle 1.907 finestre = {4*50*0.5 + 3*50:.0f}; valutazioni held equivalenti = {4*50*0.5:.0f}")
print(f"  tetto 3 GPUh: {3*3600/(sel_steps+ref_steps)*1000:.0f} ms per passo; col fattore 2: {1.5*3600/(sel_steps+ref_steps)*1000:.0f} ms per passo")

def project(path, label):
    d = json.load(open(path))
    tot, rows = 0.0, []
    for arm, kind in (("L0", "L0"), ("MLP", "MLP"), ("L13", "L13"), ("TCN-post", "TCN"), ("TCN-pre", "TCN")):
        h = d["heads"][kind]
        tr_max = max(h["train_epoch_seconds"][1:])   # prima epoca esclusa (riscaldamento)
        ev_max = max(h["eval_epoch_seconds"][1:])
        sec = 250 * tr_max + 100 * ev_max
        tot += sec
        rows.append((arm, tr_max, ev_max, sec))
    inf = sum(3 * (NV / NT) * max(d["heads"][k]["eval_epoch_seconds"][1:]) for k in ("L0", "MLP", "L13", "TCN", "TCN"))
    print(f"  [{label}] device={d['device']} threads={d['threads']} eval_batch={d['eval_batch']}")
    for arm, a, e, sec in rows:
        print(f"      {arm:8s} epoca train (max) {a:.3f} s ; epoca eval (max) {e:.3f} s ; matrice 250 train + 100 eval = {sec:7.1f} s")
    print(f"      matrice dei fit, solo calcolo: {tot:.0f} s = {tot/3600:.3f} h ; raddoppiata {2*tot/3600:.3f} h ; margine sul tetto di 3 h: {3*3600/(2*tot):.1f}x")
    print(f"      inferenza 15 modelli x 18.548 finestre, solo calcolo: {inf:.0f} s = {inf/3600:.4f} h ; raddoppiata {2*inf/3600:.4f} h ; margine su 0,5 h: {0.5*3600/(2*inf):.1f}x")
    return tot, inf

for f, label in (("bench_mps_r2.json", "MPS M3 Pro, sintetico, NON A100, macchina carica"),
                 ("bench_cpu8_r2.json", "CPU Mac 8 thread, sintetico, macchina carica")):
    p = D / f
    if p.exists():
        project(p, label)
