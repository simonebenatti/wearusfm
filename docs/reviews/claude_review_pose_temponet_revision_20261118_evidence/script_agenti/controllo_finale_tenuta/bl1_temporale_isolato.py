"""Isola l'effetto della banda su y* (tabella della bozza) sulla calibrabilita' del temporale: a parita' di ancora, griglia di a e
griglia di sigma cambia solo il bersaglio della LS nella banda (y oppure y*). Stesso modello e griglia di ipotesi di
work_controlli/conti_temporale_dettaglio.py. Solo aritmetica."""
import itertools
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/tenuta/bl1_temporale_isolato.out.txt"
a0=[0.2,0.4,0.6,0.8,1.0]; a3=a0+[1.5,2.0,2.5,3.0]
s_f3=[round(0.5+0.05*i,2) for i in range(51)]
grid=list(itertools.product((0.3,0.5,0.8),(0.3,0.6,0.9),(0.2,0.4,0.6),(2.0,3.0,5.0)))
def cal(fL,fT,r,Kd,Er,ag,anchor,band):
    for a in ag:
        for s in s_f3:
            N=s*s*c_n; X=s*s*p*Kd/n_in; RT=a*a*r*fT; S=fL+a*a*fT; flin=S-RT
            Xa = X if anchor=='wls' else X*flin/(flin+X)
            Xb = X if band=='y' else 0.0
            Ecal=(N+RT+Xa)/(S+N); Els=(N+RT+Xb)/(S+N); Eor=N/(S+N); g=(Els-Eor)/Els
            if abs(Ecal-Er)<=0.05 and 0.06<=g<=0.10: return True
    return False
L=["Non calibrati su 81 (griglia di ipotesi del revisore), sigma fitta 0,5-3 a passi 0,05"]
for anchor in ('wls','ridge'):
  for ag,an in ((a0,'a<=1'),(a3,'a<=3')):
    for band in ('y','ystar'):
        row=[]
        for Er in (0.80,0.85,0.90,0.94):
            nc=sum(1 for (fL,fT,r,Kd) in grid if not cal(fL,fT,r,Kd,Er,ag,anchor,band))
            row.append(f"E_real {Er:.2f}: {nc}")
        L.append(f"ancora {anchor:5s} {an} banda su {band:5s} | "+" | ".join(row))
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
