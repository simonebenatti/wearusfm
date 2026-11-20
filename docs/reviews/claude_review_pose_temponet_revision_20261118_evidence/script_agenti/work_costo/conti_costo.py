# Conti di controllo del punto 6 (costo). Solo aritmetica, nessun dato del progetto letto.
import json, math, sys
from fractions import Fraction as F
from datetime import date, timedelta

snap = sys.argv[1]
prop = json.load(open(snap + "/results/infra/pose_temponet_proposal_20261117/proposal.json"))
ac = json.load(open(snap + "/results/infra/pose_temponet_proposal_20261117/analytic_checks.json"))
rp = prop["resource_proposal"]; fit = prop["fit"]

print("== 1. Traiettorie")
heads_real, ctrl, heads_ctrl, folds, seeds = 5, 2, 3, 2, 3
combos = heads_real + ctrl * heads_ctrl
for L, lab in ((4, "iniziale"), (5, "max raggiungibile (una sola estensione LR)"), (6, "max dichiarato")):
    cv_real, cv_ctrl = heads_real * L * folds, ctrl * heads_ctrl * L * folds
    print(f"  L={L} {lab}: CV reale {cv_real}, controlli {cv_ctrl}, rifit {heads_real*seeds}, totale {cv_real+cv_ctrl+heads_real*seeds}")
print("  combinazioni testa/bersaglio =", combos)
print("  JSON:", fit["new_cv_fits_initial"], fit["new_cv_fits_max"], fit["control_cv_fits_initial"], fit["control_cv_fits_max"],
      fit["new_refits"], fit["new_total_trajectories_initial"], fit["new_total_trajectories_max"])

print("== 2. Aggiornamenti: 11*L*E*(S0+S1) + 15*E*Sfull")
N, B = 1907, 32
Sfull = math.ceil(N / B); S01max = Sfull + 1
print(f"  ceil(1907/32) = {Sfull}; ceil(a/32)+ceil(b/32) <= ceil((a+b)/32)+1 = {S01max}")
# verifica esaustiva del limite su tutte le divisioni a+b=1907
worst = max(math.ceil(a / B) + math.ceil((N - a) / B) for a in range(0, N + 1))
print("  massimo esaustivo di S0+S1 su a+b=1907:", worst)
def upd(L, E, S01, Sf): return combos * L * E * S01 + 15 * E * Sf
for L, E, S01, Sf, lab in ((4,150,60,60,"nominale iniziale"),(6,300,60,60,"nominale max"),(4,150,61,60,"limite iniziale"),
                          (6,300,61,60,"limite max"),(5,300,60,60,"Claude 1,26M = 5 LR"),(5,300,61,60,"limite max raggiungibile 5 LR")):
    print(f"  {lab}: {upd(L,E,S01,Sf):,}")
n_valid = 1854  # finestre con posa valida nel riferimento pilota (stessa firma target/maschera della cache 49046)
print(f"  con <=1854 finestre supportate: Sfull <= {math.ceil(n_valid/B)}, S0+S1 <= {math.ceil(n_valid/B)+1}; limite max {upd(6,300,math.ceil(n_valid/B)+1,math.ceil(n_valid/B)):,}")
print(f"  fold <=1000 finestre (100 registrazioni x <=10): S_fold <= {math.ceil(1000/B)}, >= {math.ceil((N-1000)/B)}")

print("== 3. Valutazioni held")
for L, lab in ((6, "dichiarato"), (5, "raggiungibile")):
    cv = combos * L * folds
    ev = cv * (300 + 3 + 1)
    print(f"  {lab}: {cv} traiettorie CV x 304 = {ev:,}; + diagnostici {combos*2}+15 = {combos*2+15}")

print("== 4. GPU-ore a priori (solo update)")
for n in (537600, 1477800):
    for t in (0.0041, 0.0185):
        print(f"  {n:,} x {t*1000:.1f} ms = {n*t/3600:.4f} h; x2 = {2*n*t/3600:.4f} h")

print("== 5. Quota delle valutazioni held nel gate GPU (batch eval = 32)")
def evb(L): return combos * L * 304 * S01max
fitfw = combos * S01max + 15 * Sfull
for L in (6, 5):
    tr = upd(L, 300, S01max, Sfull); ev = evb(L) + fitfw
    print(f"  L={L}: batch train {tr:,}; batch eval held {evb(L):,} + fitting {fitfw:,} = {ev:,}; rapporto eval/train {ev/tr:.3f}; quota eval (pari tempo) {ev/(tr+ev):.3f}")
    for t in (0.0041, 0.0185):
        for r, lab in ((1, "eval=passo"), (F(1,2), "eval=1/2"), (F(1,3), "eval=1/3")):
            g = 2 * (tr + float(r) * ev) * t / 3600
            print(f"    t={t*1000:.1f} ms, {lab}: gate = {g:.2f} GPUh, margine vs 20 = {20-g:+.2f}")
    for r, lab in ((1, "eval=passo"), (F(1,3), "eval=1/3")):
        for cap, labc in ((20, "20 h"), (18, "18 h (40 job x 180 s)")):
            tstar = cap * 3600 / (2 * (tr + float(r) * ev))
            print(f"    pareggio {lab}, tetto {labc}: {tstar*1000:.2f} ms per batch")

print("== 6. Tabella delle fasi")
ph = rp["phases"]
g = sum(p["gpu_hours_cap"] for p in ph) + rp["reserve_gpu_hours"]
c = sum(p["cpu_only_core_hours_cap"] for p in ph) + rp["reserve_cpu_core_hours"]
a = sum(p["allocated_core_hours_cap"] for p in ph) + rp["reserve_gpu_hours"] * 8 + rp["reserve_cpu_core_hours"]
for p in ph:
    alloc = p["cpus"] * p["cumulative_wall_hours_including_180s_perjob"]
    gpuh = p["gpus"] * p["cumulative_wall_hours_including_180s_perjob"]
    ok = (abs(alloc - p["allocated_core_hours_cap"]) < 1e-9 and abs(gpuh - p["gpu_hours_cap"]) < 1e-9
          and abs(p["local_hours_if_stated_billing"] - p["allocated_core_hours_cap"]) < 1e-9)
    print(f"  {p['phase']}: cpu {p['cpus']} x {p['cumulative_wall_hours_including_180s_perjob']} h = {alloc}; GPUh {gpuh}; coerente {ok}")
print(f"  totali: GPUh {g}, CPU-only coreh {c}, allocate {a}; JSON {rp['gpu_hours_cap']}, {rp['cpu_only_core_hours_cap']}, {rp['total_allocated_core_hours_cap']}")
print(f"  24*8 + 80 = {24*8+80}; riserva locale 2,5*8+3 = {2.5*8+3}")

print("== 7. Job seriali")
w = 3*3600 + 57*60 + 180
print(f"  03:57:00 + 180 s = {w/3600:.4f} h; 2 job x 8 CPU = {2*8*w/3600} coreh")
print(f"  limite memoria seriale 8 x 3850 MB = {8*3850} MB; 24 GiB = {24*1024} MiB")
print(f"  boost: 8 x 15400 MB = {8*15400} MB default; 32 GiB = {32*1024} MiB")
print(f"  40 job GPU x 180 s = {40*180/3600} h dentro il tetto di 20")

print("== 8. Saldo")
print(f"  2.000.000 - 1.308.137 = {2000000-1308137}; 272/691863 = {272/691863*100:.5f}%; residuo {691863-272}")

print("== 9. Memoria e byte")
R = 4993**2 * 8
print(f"  R/Gram 4993^2 float64 = {R:,} B = {R/2**20:.1f} MiB; x20 = {20*R/2**30:.3f} GiB")
print(f"  L13 disegno per fold ~120.000 righe x 4993 x 8 B = {120000*4993*8/2**30:.2f} GiB")
pred = 18548*160*20*4
print(f"  predizione per testa {pred:,} B; x15 = {15*pred/2**30:.5f} GiB")
lat_val = 18548*160*384*2; lat_tr = 1907*160*384*2
print(f"  vista val FP16 {lat_val/2**30:.3f} GiB; FP32 {2*lat_val/2**30:.3f} GiB; vista train FP16 {lat_tr/2**30:.3f} GiB")
c2 = 474412856 + 4593977338; r3 = 59367 + 418440754 + 234048726
print(f"  cache {c2:,} B = {c2/2**30:.4f} GiB; + riferimenti {r3:,} = {(c2+r3):,} B = {(c2+r3)/2**30:.4f} GiB")
s = 257675076 + 2485295825
print(f"  riferimenti SCRATCH {s:,} B = {s/1e9:.3f} GB = {s/2**30:.3f} GiB")

print("== 10. Calendario")
d = date(2026, 11, 12) + timedelta(days=40); print("  12/11 + 40 giorni =", d)
d2 = date(2026, 12, 3) + timedelta(days=40); print("  03/12 + 40 giorni =", d2)
