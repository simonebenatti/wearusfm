"""BL-1 (c) e tabella E_cal/E_LS*: calibrabilita' del controllo temporale nel modello dei revisori.
Stesso modello e stessa griglia di ipotesi di work_controlli/conti_temporale_dettaglio.py (f_L, f_T, r, K, E_real).
V0 = protocollo attuale: ancora e banda con la WLS su y (eccesso X a n_in), a <= 1, sigma {1..3}.
V1 = bozza: ancora E_cal = ridge su y (eccesso X_r = X f/(f+X), f = parte dinamica leggibile da L0), banda con E_LS* = LS su y*
     (eccesso da rumore nullo), a fino a 3 (opzione 1 di (c)), sigma fitta (passo 0,05 fra 0,5 e 3, e fra 0,5 e 5).
V1b = come V1 ma con la banda ancora sulla LS fittata su y (per isolare l'effetto di E_LS*).
Solo aritmetica, nessun dato."""
import itertools
from collections import Counter
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/tenuta/bl1_temporale_tabella.out.txt"
L=[]
a0=[0.2,0.4,0.6,0.8,1.0]; a3=a0+[1.5,2.0,2.5,3.0]
s0=[1,1.5,2,2.5,3]
s_f3=[round(0.5+0.05*i,2) for i in range(51)]
s_f5=[round(0.5+0.05*i,2) for i in range(91)]
def quant(fL,fT,r,a,s,Kd,n,anchor,band):
    N=s*s*c_n; X=s*s*p*Kd/n; RT=a*a*r*fT; S=fL+a*a*fT; flin=S-RT
    Xr=X*flin/(flin+X)
    Xa={'wls':X,'ridge':Xr}[anchor]
    Xb={'y':X,'ystar':0.0}[band]
    Ecal=(N+RT+Xa)/(S+N); Els=(N+RT+Xb)/(S+N); Eor=N/(S+N)
    return Ecal,(Els-Eor)/Els
def calibrabile(fL,fT,r,Kd,Er,ag,sg,anchor,band):
    for a in ag:
        for s in sg:
            Ecal,g=quant(fL,fT,r,a,s,Kd,n_in,anchor,band)
            if abs(Ecal-Er)<=0.05 and 0.06<=g<=0.10: return True
    return False
variants=[('V0 attuale: WLS/WLS su y, a<=1, sigma {1..3}',a0,s0,'wls','y'),
          ('V1b: ridge/LS su y, a<=3, sigma fitta 0,5-3',a3,s_f3,'ridge','y'),
          ('V1 bozza: ridge su y / LS su y*, a<=3, sigma fitta 0,5-3',a3,s_f3,'ridge','ystar'),
          ('V1 bozza: ridge su y / LS su y*, a<=3, sigma fitta 0,5-5',a3,s_f5,'ridge','ystar'),
          ('V1 bozza con a<=1: ridge su y / LS su y*, sigma fitta 0,5-3',a0,s_f3,'ridge','ystar')]
grid=list(itertools.product((0.3,0.5,0.8),(0.3,0.6,0.9),(0.2,0.4,0.6),(2.0,3.0,5.0)))
for name,ag,sg,anc,band in variants:
    L.append(name)
    for Er in (0.80,0.85,0.90,0.94):
        nc=sum(1 for (fL,fT,r,Kd) in grid if not calibrabile(fL,fT,r,Kd,Er,ag,sg,anc,band))
        ncr=Counter(r for (fL,fT,r,Kd) in grid if not calibrabile(fL,fT,r,Kd,Er,ag,sg,anc,band))
        L.append(f"  E_real={Er:.2f}: non calibrati {nc} su {len(grid)}; per r: "+", ".join(f"r={k}: {ncr[k]}/27" for k in (0.2,0.4,0.6)))
L.append("")
L.append("Soglia necessaria su r con la banda su y* (eccesso da rumore nullo nella banda): R_T/S >= E'/((1-E')/g+E'),")
L.append("E' = E_real - eccesso della ridge nell'ancora; con S = f_L + a^2 f_T: r >= c (1 + f_L/(a^2 f_T)).")
for E in (0.80,0.85,0.90,0.94):
    for g in (0.06,):
        c=E/((1-E)/g+E)
        L.append(f"  E={E:.2f}, g={g}: c={c:.3f}; f_L=f_T: r>={2*c:.3f} con a=1, r>={c*(1+1/9):.3f} con a=3; nessun a basta se r<{c:.3f}")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
