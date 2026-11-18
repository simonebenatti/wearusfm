# Ricalcoli indipendenti dei numeri dei passi riscritti (solo aritmetica, nessun dato).
from fractions import Fraction
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2; grid=[1,1.5,2,2.5,3]
def lin_inner(f,s,Kd): N=s*s*c_n; return (N+s*s*p*Kd/n_in)/(f+N)
print("== BL-1 controesempio, due coppie di K")
for Kd,Kt in ((2.5,5.0),(3.5,7.0)):
    for f,Er in ((0.5,0.92),(0.4,0.78),(0.4,0.80)):
        c=[(abs(lin_inner(f,s,Kd)-Er),s) for s in grid if abs(lin_inner(f,s,Kd)-Er)<=0.05]
        if not c: print(f"  K=({Kd},{Kt}) f={f} E_real={Er}: non calibrato"); continue
        d,s=min(c); N=s*s*c_n; Eor=N/(f+N); Ew_in=lin_inner(f,s,Kd); Ew_out=(N+s*s*p*Kd/n_out)/(f+N)
        Rd=(1-Ew_out)/(1-Eor); Rm=1-s*s*p*Kt/n_out
        print(f"  K=({Kd},{Kt}) f={f} E_real={Er}: sigma={s} WLSin={Ew_in:.4f} oracolo={Eor:.4f} WLSout={Ew_out:.4f} R_dyn={Rd:.4f} R_MSE={Rm:.4f}")
print("== griglia v_scan: primo E_real con F (WLS esatta sotto 90%) e ultimo con P")
for Kd,Kt in ((2.5,5.0),(3.5,7.0)):
    for f in (0.3,0.4,0.5,0.6):
        row=[]
        for i in range(12):
            Er=round(0.78+0.02*i,2)
            c=[(abs(lin_inner(f,s,Kd)-Er),s) for s in grid if abs(lin_inner(f,s,Kd)-Er)<=0.05]
            if not c: row.append((Er,'N')); continue
            d,s=min(c); Rd=1-s*s*p*Kd/n_out/f; Rm=1-s*s*p*Kt/n_out
            row.append((Er,'P' if (Rd>=0.9 and Rm>=0.9) else 'F'))
        firstF=next((e for e,o in row if o=='F'),None); lastP=max((e for e,o in row if o=='P'),default=None)
        print(f"  K=({Kd},{Kt}) f={f}: {''.join(o for e,o in row)} ultimo P={lastP} primo F={firstF}")
print("== BL-1(b): ridge in due parametrizzazioni (modello isotropo di conti_rilettura.py, sigma 2,5 e 3, K_tot 5)")
def risk(sk,v,c): sh=sk/(sk+c*v); return sk*(1-sh)**2+v*sh**2
spec=[1/384]*384
for sig in (2.5,3.0):
    v=sig*sig*5.0/n_out
    ols=len(spec)*v
    opt=sum(risk(sk,v,1) for sk in spec)   # alpha ottimo per 20 utenti
    dbl=sum(risk(sk,v,2) for sk in spec)   # alpha doppio: loss media, alpha degli inner (n/2) non riscalato
    print(f"  sigma={sig}: WLS {1-ols:.4f}; ridge alpha ottimo {1-opt:.4f} (= alpha inner con penalita' sulla SSE, alpha* indip. da n); ridge alpha x2 {1-dbl:.4f} (= alpha inner con loss media, alpha*~1/n)")
print("== BL-3")
print(f"  (0,7269-0,7268)/0,7269 = {100*(0.7269-0.7268)/0.7269:.4f}%")
print("== ricevuta: alpha 0,01 vs 1e5")
print(f"  0,7697254997148791/0,6561153689170307-1 = {100*(0.7697254997148791/0.6561153689170307-1):.3f}%")
print("== OB-3 e OB-4")
g=lambda r,t: 100*(r-t)/r
print(f"  gain3 L13 0,7619 TCN 0,750: {g(0.7619,0.750):.3f}; L13 0,735: {g(0.735,0.750):.3f}")
def lab(gain,lo,up):
    if gain>=2 and lo>0: return 1
    if gain<=-2 and up<0: return 2
    if up<2: return 3
    return 4
print("  OB-3 contrasto 2 (-1,5; [-1,9; -0,8]) ->", lab(-1.5,-1.9,-0.8))
print("  OB-4 contrasto 1 (+5; [+3; +7]) ->", lab(5,3,7))
print("== OB-2 regret L13: 3,66% -> ", round(3.66,1), "; 11,9% ->", round(11.9))
print("== OB-5 soglia")
t=0.98*0.7786289843513977; print(f"  {t!r}  sotto ridge {100*(0.7859020004638595-t)/0.7859020004638595:.4f}%")
print("== ricetta-7: (1-1e-2*0.01)^9000 =", round((1-1e-4)**9000,4))
print("== costo-2/ricetta-6")
print("  5LR nominale", 11*5*300*60+15*300*60, " limite", 11*5*300*61+15*300*60, " 6LR nominale", 11*6*300*60+15*300*60, " limite", 11*6*300*61+15*300*60)
print("  traiettorie 5LR", 2*5*5+2*3*2*5+15, " eval held 5LR", (2*5*5+2*3*2*5)*304, " 6LR", (2*5*6+2*3*2*6)*304)
print("== disposizioni: 45+18 =", 45+18, "; 8+37 =", 8+37, "; 5+13 =", 5+13)
print("== 2,0e-16 vs 1e-12 ok;  12/11+40 giorni:")
import datetime; print("  ", datetime.date(2026,11,12)+datetime.timedelta(days=40))
