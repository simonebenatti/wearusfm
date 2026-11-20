# Dettaglio della calibrazione temporale nel modello analitico di conti_calibrazione.py (stesse ipotesi).
import itertools
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
sig_grid=[1,1.5,2,2.5,3]; a_grid=[0.2,0.4,0.6,0.8,1.0]
def temp(fL,fT,r,a,sig,Kd,n):
    N=sig**2*c_n; X=sig**2*p*Kd/n; RT=a**2*r*fT; S=fL+a**2*fT
    Els=(N+RT+X)/(S+N); Eor=N/(S+N); return Els,Eor,(Els-Eor)/Els
from collections import Counter
tot=Counter(); nc=Counter(); why=Counter()
for fL,fT,r,Kd,Er in itertools.product((0.3,0.5,0.8),(0.3,0.6,0.9),(0.2,0.4,0.6),(2.0,3.0,5.0),(0.80,0.90,1.00)):
    tot[Er]+=1
    okE=okg=okboth=False
    for a in a_grid:
        for s in sig_grid:
            Els,Eor,g=temp(fL,fT,r,a,s,Kd,n_in)
            e=abs(Els-Er)<=0.05; gg=0.06<=g<=0.10
            okE|=e; okg|=gg; okboth|=(e and gg)
    if not okboth:
        nc[Er]+=1
        why[(Er, 'nessun E' if not okE else ('nessun g' if not okg else 'E e g mai insieme'))]+=1
for Er in sorted(tot): print(f"E_real={Er}: non calibrati {nc[Er]} su {tot[Er]}")
for k,v in sorted(why.items()): print(k,v)
# esempio puntuale
print()
for Er in (0.80,0.90,1.00):
    fL,fT,r,Kd=0.5,0.6,0.4,3.0
    best=[]
    for a in a_grid:
        for s in sig_grid:
            Els,Eor,g=temp(fL,fT,r,a,s,Kd,n_in)
            best.append((round(Els,3),round(g,3),a,s))
    adm=[b for b in best if abs(b[0]-Er)<=0.05 and 0.06<=b[1]<=0.10]
    print(f"esempio fL=0.5 fT=0.6 r=0.4 K=3, E_real={Er}: ammissibili {adm}")
    print("   griglia (E_LS,g,a,sigma) con g in banda:", [b for b in best if 0.06<=b[1]<=0.10])
