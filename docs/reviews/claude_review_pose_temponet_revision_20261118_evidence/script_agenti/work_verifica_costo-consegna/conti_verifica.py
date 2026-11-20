# Conti del verificatore avversario (costo). Solo aritmetica, nessuna misura.
import math, itertools
B=32; N=1907
# 1) massimo di S0+S1 su tutte le divisioni
mx=max(math.ceil(a/B)+math.ceil((N-a)/B) for a in range(0,N+1))
print("max S0+S1 su tutte le divisioni:",mx,"; ceil(1907/32)=",math.ceil(N/B))
# combinazioni
combos={"L0":3,"L13":1,"MLP":3,"TCN":4}   # 5 reali + 2 controlli x (L0,MLP,TCN-post)
refits={"L0":3,"L13":3,"MLP":3,"TCN":6}   # 5 teste x 3 semi
assert sum(combos.values())==11 and sum(refits.values())==15
def matrix(L,E,S01=61,Sf=60):
    cv={h:c*L*E*S01 for h,c in combos.items()}            # passi CV (coppie di fold)
    rf={h:r*E*Sf for h,r in refits.items()}
    ev={h:c*L*(E+4)*S01 for h,c in combos.items()}         # batch held a batch 32 (E epoche + 3 quarti + passo0)
    return cv,rf,ev
for L,E,S01,Sf,lab in [(4,150,60,60,"nominale iniziale"),(6,300,60,60,"nominale max"),(4,150,61,60,"limite iniziale"),(6,300,61,60,"limite max 6LR"),(5,300,61,60,"limite max 5LR (raggiungibile)"),(5,300,60,60,"5LR a 30 passi (conto Claude)")]:
    cv,rf,ev=matrix(L,E,S01,Sf)
    print(f"{lab}: passi={sum(cv.values())+sum(rf.values()):,} ; traiettorie CV={11*L*2} ; eval held={11*L*2*(E+4):,} ; batch held={sum(ev.values()):,}")
fit37 = 22*61/2 + 15*60
print("batch dei 37 forward sul fitting ~", fit37)
# 2) gate a tempo uniforme (come costo-1)
def gate(t_step, frac_eval, L=6, t_by_head=None):
    cv,rf,ev=matrix(L,300,61,60)
    tot=0.0
    for h in combos:
        ts = t_by_head[h] if t_by_head else t_step
        tot += (cv[h]+rf[h])*ts + (ev[h])*ts*frac_eval
    # 37 forward sul fitting: attribuiti con lo stesso tempo medio pesato (piccoli)
    ts_mean = (t_by_head["TCN"] if t_by_head else t_step)
    tot += fit37*ts_mean*frac_eval
    return 2*tot/3600
print("\nGate raddoppiato, prima di I/O/avvii/salvataggi (GPUh)")
for L in (6,5):
  for fe in (1,0.5,1/3):
    print(f" uniforme 18,5 ms, {L} LR, eval={fe:.2f} passo: {gate(0.0185,fe,L):.2f}")
for L in (6,5):
  for fe in (1,0.5,1/3):
    th={"TCN":0.0185,"L0":0.0041,"L13":0.0041,"MLP":0.0041}
    print(f" misto: TCN 18,5 ms, altre 4,1 ms, {L} LR, eval={fe:.2f}: {gate(None,fe,L,th):.2f}")
for L in (6,5):
  for fe in (1,0.5,1/3):
    th={"TCN":0.0185,"L0":0.0185,"L13":0.0185,"MLP":0.0185}
# quota TCN dei passi
cv,rf,ev=matrix(6,300,61,60)
tcn=(cv["TCN"]+rf["TCN"]); tot=sum(cv.values())+sum(rf.values())
print(f"\nquota TCN dei passi (6LR300): {tcn:,}/{tot:,} = {tcn/tot:.3f}; batch held TCN {ev['TCN']:,}/{sum(ev.values()):,} = {ev['TCN']/sum(ev.values()):.3f}")
# pareggio con 18 h utili (40 job x 180 s)
print("40 job x 180 s =",40*180/3600,"h")
# 3) stime a priori del protocollo
for n in (537600,1477800):
    for t in (0.0041,0.0185):
        print(f" {n:,} x {t*1000:.1f} ms = {n*t/3600:.4f} h; x2 = {2*n*t/3600:.4f}")
# 4) conto di Claude nel testo per GPT
print("Claude 1,26M:",11*5*300*60+15*300*60, "; 531.000:",11*4*150*60+15*150*60)
for n in (531000,1260000):
    print(f"  {n:,}: x2 a 4,1 ms {2*n*0.0041/3600:.2f} h, a 18,5 ms {2*n*0.0185/3600:.2f} h")
# 5) tabella fasi
ph=[("checkout_preflight",0,2,0.25),("cpu_benchmark",0,8,0.5),("wls_calibration",0,8,8),("gpu_benchmark",1,8,0.5),("controls_selection_refits",1,8,20),("locked_inference",1,8,1),("metrics",0,8,1),("collection",0,2,0.25)]
g=sum(gp*h for _,gp,c,h in ph); cpu=sum(c*h for _,gp,c,h in ph if gp==0); alloc=sum(c*h for _,gp,c,h in ph)
print(f"\nfasi: GPUh={g}, CPU-only={cpu}, allocate={alloc}; +riserva GPU 2,5 e CPU 3 -> GPUh {g+2.5}, CPU {cpu+3}, allocate {alloc+2.5*8+3}")
print("saldo:",2000000-1308137, " 272/saldo=%.5f%%"%(272/(2000000-1308137)*100), " residuo", 2000000-1308137-272)
print("WLS: (3:57:00+180 s) =",(3*3600+57*60+180)/3600,"h ; 2 job x 8 CPU =",2*8*(3*3600+57*60+180)/3600)
print("R 4993^2 float64 =",4993**2*8,"B =",4993**2*8/2**20,"MiB ; x20 =",20*4993**2*8/2**30,"GiB")
print("15 pred FP32 =",18548*160*20*4*15,"B =",18548*160*20*4*15/2**30,"GiB")
print("cache =",474412856+4593977338,"B =",(474412856+4593977338)/2**30,"GiB; con riferimenti",474412856+4593977338+59367+418440754+234048726,(474412856+4593977338+59367+418440754+234048726)/2**30)
print("0,98 x 0,7786289843513977 =",0.98*0.7786289843513977, "; sotto ridge:", 1-0.98*0.7786289843513977/0.7859020004638595)
print("12/11 + 40 giorni:")
import datetime; print(datetime.date(2026,11,12)+datetime.timedelta(days=40))
