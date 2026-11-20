# Modello analitico (forma chiusa, nessun dato e nessuna simulazione) della calibrazione e dei cancelli dei controlli positivi.
# Ipotesi dichiarate:
#  - per giunto, per elemento delle finestre dinamiche: segnale dinamico S, rumore dinamico N = sigma^2 * c_n (AR(1) rho=e^-1/4);
#  - eccesso della WLS non regolarizzata dovuto al rumore: X = sigma^2 * p * K / n (formula classica p/n con inflazione K per
#    autocorrelazione del rumore e spostamento fra utenti); K_dyn per la parte dinamica, K_tot per la MSE totale;
#  - n = elementi validi per giunto del fitting: outer ~118.000 (953 finestre x ~124), inner = meta';
#  - L_n e T_n normalizzati a varianza totale 1 sul fitting outer; covarianza L-T trascurata.
import itertools
c_n = 0.9372          # 1 - Var(media) per ~124 punti validi
p = 384
n_out = 118000.0
n_in = n_out/2
sig_grid = [1, 1.5, 2, 2.5, 3]
a_grid = [0.2, 0.4, 0.6, 0.8, 1.0]

def lin_inner(f, sig, Kd):
    N = sig**2*c_n; X = sig**2*p*Kd/n_in
    return (N+X)/(f+N)

def lin_outer(f, sig, Kd, Kt):
    N = sig**2*c_n; Xd = sig**2*p*Kd/n_out; Xt = sig**2*p*Kt/n_out
    Eor = N/(f+N); Els = (N+Xd)/(f+N)
    Rdyn = (1-Els)/(1-Eor)          # = 1 - Xd/f
    Rmse = 1 - Xt/1.0               # margine MSE ~ varianza di L_n = 1
    return Eor, Els, Rdyn, Rmse

print("=== CONTROLLO LINEARE: E_LS,inner(sigma) sulla griglia, K_dyn=3 ===")
for f in (0.2, 0.3, 0.5, 0.8, 1.0):
    vals = [lin_inner(f, s, 3.0) for s in sig_grid]
    gaps = [vals[i+1]-vals[i] for i in range(4)]
    holes = []
    lo = vals[0]-0.05
    # intervalli di E_real non ammissibili fra min-0.05 e max+0.05 (e fuori)
    cover = sorted([(v-0.05, v+0.05) for v in vals])
    cur = cover[0][1]
    for a,b in cover[1:]:
        if a > cur: holes.append((round(cur,3), round(a,3)))
        cur = max(cur,b)
    print(f"f'={f}: " + " ".join(f"s{ s}:{v:.3f}" for s,v in zip(sig_grid, vals)) +
          f" | salti {[round(g,3) for g in gaps]} | E_real non calibrabile: <{cover[0][0]:.3f}, buchi {holes}, >{cur:.3f}")

print()
print("=== CONTROLLO LINEARE: scelta di sigma e recupero della WLS esatta sugli held outer ===")
print("(K_dyn,K_tot) = (2.5,5) 'favorevole', (3.5,7) 'meno favorevole'; soglia del protocollo 0,90 su MSE ed E_dyn")
for (Kd, Kt) in ((2.5,5.0),(3.5,7.0)):
    print(f"-- K_dyn={Kd}, K_tot={Kt}")
    for Er in (0.80, 0.85, 0.90, 0.95, 1.00, 1.05):
        row=[]
        for f in (0.3, 0.5, 0.8, 1.0):
            cand=[(abs(lin_inner(f,s,Kd)-Er), s) for s in sig_grid if abs(lin_inner(f,s,Kd)-Er)<=0.05]
            if not cand:
                row.append(f"f'{f}: NON CALIBRATO"); continue
            d,s = min(cand)
            Eor,Els,Rd,Rm = lin_outer(f,s,Kd,Kt)
            ok = "passa" if (Rd>=0.9 and Rm>=0.9) else "FALLISCE"
            row.append(f"f'{f}: s={s} Eor={Eor:.3f} Rdyn={Rd:.3f} Rmse={Rm:.3f} {ok}")
        print(f"E_real={Er:.2f} | " + " | ".join(row))

print()
print("=== CONTROLLO TEMPORALE: spostamento del guadagno da inner (10 utenti) a outer (20 utenti) ===")
def temp(fL, fT, r, a, sig, Kd, n):
    N = sig**2*c_n; X = sig**2*p*Kd/n; RT = a**2*r*fT; S = fL + a**2*fT
    Els = (N+RT+X)/(S+N); Eor = N/(S+N); g = (Els-Eor)/Els
    return Els, Eor, g
res = {"tot":0, "noncal":0, "outer_fuori":0, "outer_ok":0}
rows=[]
for fL, fT, r, Kd, Er in itertools.product((0.3,0.5,0.8),(0.3,0.6,0.9),(0.2,0.4,0.6),(2.0,3.0,5.0),(0.80,0.90,1.00)):
    res["tot"]+=1
    cand=[]
    for a in a_grid:
        for s in sig_grid:
            Els,Eor,g = temp(fL,fT,r,a,s,Kd,n_in)
            if abs(Els-Er)<=0.05 and 0.06<=g<=0.10:
                cand.append((abs(g-0.08), abs(Els-Er), a, s, g))
    if not cand:
        res["noncal"]+=1; continue
    cand.sort()
    _,_,a,s,gin = cand[0]
    Els,Eor,gout = temp(fL,fT,r,a,s,Kd,n_out)
    if 0.06<=gout<=0.10: res["outer_ok"]+=1
    else: res["outer_fuori"]+=1
    rows.append((Kd, gin, gout))
print(res)
for Kd in (2.0,3.0,5.0):
    sel=[(gi,go) for k,gi,go in rows if k==Kd]
    if sel:
        sh=[go-gi for gi,go in sel]
        print(f"K_dyn={Kd}: coppie calibrate {len(sel)}; g_in medio {sum(g for g,_ in sel)/len(sel):.4f}; g_out medio {sum(g for _,g in sel)/len(sel):.4f};"
              f" spostamento medio {sum(sh)/len(sh):+.4f}; g_out<0,06 in {sum(1 for _,g in sel if g<0.06)} su {len(sel)}")
# spostamento con g_in = 0,08 esatto in funzione di K
print()
for Kd in (2,3,4,6,8,10):
    x_in = p*Kd/(n_in*c_n); x_out = x_in/2
    t = 0.08/(1-0.08) - x_in     # rho_T = a^2 r fT / N
    if t < 0: print(f"K={Kd}: guadagno inner 0,08 tutto da stima"); continue
    gout = (t + x_out)/(1 + t + x_out)
    print(f"K_dyn={Kd}: x_in={x_in:.4f}, quota del guadagno inner dovuta alla stima {(x_in)/(t+x_in):.2f}, g_out atteso={gout:.4f}")
