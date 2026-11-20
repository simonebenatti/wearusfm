"""(c) «a anche oltre 1 (per esempio fino a 3)» e (d) «griglia di sigma fitta o continua»: con L e T normalizzati a varianza 1
(P:86), alzare a alza il segnale, quindi per restare entro 0,05 da E_real serve un sigma circa proporzionale ad a. Se l'intervallo
di sigma resta quello di P:90 (1-3), le a oltre ~1,5 non sono usabili. Modello di conti_temporale_dettaglio.py, ancora E_cal =
ridge su y, banda E_LS* = LS su y* (tabella della bozza). Solo aritmetica."""
import itertools
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/tenuta/bl1_intervallo_sigma.out.txt"
a3=[0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0]
def pairs(fL,fT,r,Kd,Er,smax,band='ystar'):
    out=[]
    for a in a3:
        s=0.5
        while s<=smax+1e-9:
            N=s*s*c_n; X=s*s*p*Kd/n_in; RT=a*a*r*fT; S=fL+a*a*fT; flin=S-RT
            Xa=X*flin/(flin+X); Xb = X if band=='y' else 0.0
            Ecal=(N+RT+Xa)/(S+N); Els=(N+RT+Xb)/(S+N); Eor=N/(S+N); g=(Els-Eor)/Els
            if abs(Ecal-Er)<=0.05 and 0.06<=g<=0.10: out.append((a,round(s,2)))
            s+=0.05
    return out
L=[]
L.append("Esempio f_L 0,5, f_T 0,6, K 3 (quello del revisore): valori di a che hanno almeno una coppia ammissibile")
for r in (0.25,0.4):
  for Er in (0.85,0.90):
    for smax in (3.0,10.0):
        pr=pairs(0.5,0.6,r,3.0,Er,smax)
        aset=sorted(set(a for a,s in pr))
        smaxused=max([s for a,s in pr],default=None)
        L.append(f"  r={r} E_real={Er:.2f} sigma<= {smax:4.1f}: a usabili {aset}; sigma massimo usato {smaxused}")
grid=list(itertools.product((0.3,0.5,0.8),(0.3,0.6,0.9),(0.2,0.4,0.6),(2.0,3.0,5.0)))
L.append("Non calibrati su 81 (griglia di ipotesi del revisore), a fino a 3, tabella della bozza, al variare del sigma massimo")
for smax in (3.0,5.0,10.0):
    row=[]
    for Er in (0.80,0.85,0.90,0.94):
        nc=sum(1 for (fL,fT,r,Kd) in grid if not pairs(fL,fT,r,Kd,Er,smax))
        row.append(f"E_real {Er:.2f}: {nc}")
    L.append(f"  sigma <= {smax:4.1f} | "+" | ".join(row))
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
