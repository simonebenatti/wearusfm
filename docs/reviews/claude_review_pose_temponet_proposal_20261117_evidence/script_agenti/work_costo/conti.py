"""Conti rifatti per la dimensione 5 (costo e fattibilita'). Solo aritmetica."""
import json
from fractions import Fraction as F
GiB = 2**30
print("== a) tabella del par. 9 ==")
phases = [  # nome, gpu, cpu, ore limite
    ("checkout/manifest/hash/supporti", 0, 2, F(15, 60)),
    ("benchmark CPU", 0, 8, F(15, 60)),
    ("benchmark GPU", 1, 8, F(15, 60)),
    ("selezione + rifit", 1, 8, F(3)),
    ("inferenza val", 1, 8, F(30, 60)),
    ("riferimenti/metriche/bootstrap", 0, 8, F(1)),
    ("raccolta/conservazione", 0, 2, F(15, 60)),
]
reserve_gpu, reserve_cpu = F(1, 4), F(1)
gpuh = sum(g * h for _, g, c, h in phases) + reserve_gpu
cpu_only = sum(c * h for _, g, c, h in phases if g == 0) + reserve_cpu
cpu_of_gpu = sum(c * h for _, g, c, h in phases if g > 0) + 8 * reserve_gpu
for n, g, c, h in phases:
    print(f"  {n:34s} GPUh={float(g*h):5.2f} coreh={float(c*h):5.2f} ({'CPU-only' if g == 0 else 'CPU del job GPU'})")
print("  riserva: GPUh", float(reserve_gpu), "coreh CPU-only", float(reserve_cpu), "coreh dei job GPU", float(8 * reserve_gpu))
print("  GPUh totali =", float(gpuh), "| coreh CPU-only =", float(cpu_only), "| coreh dei job GPU =", float(cpu_of_gpu),
      "| coreh allocate totali =", float(cpu_only + cpu_of_gpu))
print("  ore locali con billing=cpu (8 per job 1 GPU+8 CPU; cpu per job seriale):", float(8 * gpuh + cpu_only))
print("  somma dei limiti di tempo delle fasi (ore):", float(sum(h for *_, h in phases)))
print("  quota sul residuo del 09/11 (691.863 ore locali):", f"{float((8*gpuh+cpu_only)/691863*100):.5f}%")

print("== c) byte ==")
N, P, D, J = 18548, 160, 384, 20
pred = N * P * J * 4
print("  una predizione val FP32:", pred, "byte; 15:", 15 * pred, "byte =", 15 * pred / GiB, "GiB")
print("  a 148 centri (crop6):", N * 148 * J * 4, "byte; 15:", 15 * N * 148 * J * 4 / GiB, "GiB")
tr, va = 474412856, 4593977338
refs = 59367 + 418440754 + 234048726
print("  cache train+val:", tr + va, "byte =", (tr + va) / GiB, "GiB; riferimenti:", refs, "byte =", refs / GiB,
      "GiB; tutti gli input pinnati:", tr + va + refs, "byte =", (tr + va + refs) / GiB, "GiB")
def raw(n):
    view = n * P * D * 2
    parts = {"pre FP16": view, "lat FP16": view, "key_time_valid": n * P, "pred FP32": n * P * J * 4,
             "target FP32": n * P * J * 4, "mask bool": n * P * J, "std FP32": n * J * 4, "pose_var FP32": n * 4}
    return parts
for name, n, disk in (("val", 18548, va), ("train", 1907, tr)):
    parts = raw(n)
    tot = sum(parts.values())
    print(f"  {name}: non compresso (senza meta/config) = {tot} byte = {tot/GiB:.4f} GiB; su disco {disk} byte; rapporto {disk/tot:.4f}")
    for k, v in parts.items():
        print(f"      {k:16s} {v:>13d} byte = {v/GiB:.4f} GiB")
    print(f"      una vista in FP32: {n*P*D*4} byte = {n*P*D*4/GiB:.4f} GiB")
print("  24 GiB in MiB:", 24 * 1024, "<= 30800:", 24 * 1024 <= 30800)
weights = (8468 + 65300 + 100628 + 65204 + 65204) * 4 * 3
print("  pesi dei 15 rifit FP32:", weights, "byte")
print("  quota 16/11: 1 TiB - 933,1 GiB =", 1024 - 933.1, "GiB liberi; dopo 8 GiB:", 1024 - 933.1 - 8)

print("== b) matrice dei fit ==")
n_train = 1907
steps_fold, steps_full = -(-(n_train // 2) // 32), -(-n_train // 32)
print("  passi per epoca: fold ~", steps_fold, "(circa meta' finestre), full", steps_full)
print("  passi totali a 50 epoche: selezione", 20 * 50 * steps_fold, "+ rifit", 15 * 50 * steps_full, "=", 20 * 50 * steps_fold + 15 * 50 * steps_full)
print("  per testa: epoche-equivalenti di train (1.907 finestre) =", 4 * 50 * 0.5 + 3 * 50, "; di valutazione interna =", 4 * 50 * 0.5)
print("  budget per passo se i 3 GPUh fossero tutti passi:", 3 * 3600 / (20 * 50 * steps_fold + 15 * 50 * steps_full) * 1000, "ms; col fattore 2:", 1.5 * 3600 / 75000 * 1000, "ms")
def project(path, label):
    d = json.load(open(path))
    tot = 0.0
    out = []
    for arm, kind in (("L0", "L0"), ("MLP", "MLP"), ("L13", "L13"), ("TCN-post", "TCN"), ("TCN-pre", "TCN")):
        h = d["heads"][kind]
        tr_s = min(h["train_epoch_seconds"][1:])
        tr_max = max(h["train_epoch_seconds"][1:])
        ev_s = min(h["eval_epoch_seconds"][1:])
        ev_max = max(h["eval_epoch_seconds"][1:])
        sec = 250 * tr_max + 100 * ev_max
        tot += sec
        out.append((arm, tr_s, tr_max, ev_s, ev_max, sec))
    print(f"  [{label}] device={d['device']} threads={d['threads']} eval_batch={d.get('eval_batch', 256)}")
    for arm, a, b, c, e, sec in out:
        print(f"      {arm:8s} epoca train {a:.3f}-{b:.3f} s; epoca eval {c:.3f}-{e:.3f} s; matrice (250 train + 100 eval, valori massimi) {sec:8.1f} s")
    inf = sum(3 * (18548 / 1907) * max(d["heads"][k]["eval_epoch_seconds"][1:]) for k in ("L0", "MLP", "L13", "TCN", "TCN"))
    print(f"      totale fit {tot:.0f} s = {tot/3600:.3f} h; raddoppiato {2*tot/3600:.3f} h; tetto 3 h / proiezione = {3*3600/tot:.1f}x; / raddoppiata = {3*3600/(2*tot):.1f}x")
    print(f"      inferenza 15 modelli x 18.548 finestre (solo calcolo) {inf:.0f} s = {inf/3600:.4f} h; raddoppiata {2*inf/3600:.4f} h; tetto 0,5 h / raddoppiata = {0.5*3600/(2*inf):.1f}x")
project("bench_mps.json", "MPS Apple M3 Pro, sintetico, NON A100")
project("bench_cpu8.json", "CPU Mac 8 thread, sintetico, senza oneDNN")
d1 = json.load(open("bench_1thread.json"))
tot = 0
for arm, kind in (("L0", "L0"), ("MLP", "MLP"), ("L13", "L13"), ("TCN-post", "TCN"), ("TCN-pre", "TCN")):
    h = d1["heads"][kind]
    tot += 250 * max(h["train_epoch_seconds"]) + 100 * max(h["eval_epoch_seconds"])
print(f"  [CPU Mac 1 thread, sintetico, eval batch 256] totale fit {tot:.0f} s = {tot/3600:.2f} h")
print("== storico (ricevuta LONG-FIT-G49046-pose, 8 CPU, 30 GiB) ==")
r = json.load(open("LONG-FIT-G49046-pose.json"))
print("  elapsed", r["measured_resources"]["elapsed_seconds"], "s; peak RSS", r["measured_resources"]["peak_rss_bytes"], "byte =",
      r["measured_resources"]["peak_rss_bytes"] / GiB, "GiB; coreh =", 8 * r["measured_resources"]["elapsed_seconds"] / 3600)
