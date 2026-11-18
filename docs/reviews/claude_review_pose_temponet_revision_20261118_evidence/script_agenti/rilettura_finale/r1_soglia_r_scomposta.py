"""Rilettura controllo3: scompone l'aumento della quota minima r (0,27 -> 0,37, a=1, E_real 0,85) fra ancora ridge e banda su y*.
Stesso modello e stesse formule di controllo2/tenuta/bl1_soglia_r.py (copiate senza modifiche). Solo aritmetica."""
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo3/rilettura/r1_soglia_r_scomposta.out.txt"
sg=[0.5+0.01*i for i in range(451)]
def ok(fL,fT,r,Kd,Er,ag,anchor,band):
    for a in ag:
        for s in sg:
            N=s*s*c_n; X=s*s*p*Kd/n_in; RT=a*a*r*fT; S=fL+a*a*fT; flin=S-RT
            Xa = X if anchor=='wls' else X*flin/(flin+X)
            Xb = X if band=='y' else 0.0
            Ecal=(N+RT+Xa)/(S+N); Els=(N+RT+Xb)/(S+N); Eor=N/(S+N); g=(Els-Eor)/Els
            if abs(Ecal-Er)<=0.05 and 0.06<=g<=0.10: return True
    return False
def rmin(fL,fT,Kd,Er,ag,anchor,band):
    for k in range(1,101):
        r=k/100
        if ok(fL,fT,r,Kd,Er,ag,anchor,band): return r
    return None
L=[]
for Er in (0.79,0.85,0.90):
  for fL,fT in ((0.5,0.5),(0.3,0.3),(0.8,0.8)):
    ag=[1.0]
    row={}
    for an in ('wls','ridge'):
      for b in ('y','ystar'):
        row[(an,b)]=rmin(fL,fT,3.0,Er,ag,an,b)
    L.append(f"E_real={Er:.2f} f_L=f_T={fL} K=3 a=1: WLS/y {row[('wls','y')]} | WLS/y* {row[('wls','ystar')]} | ridge/y {row[('ridge','y')]} | ridge/y* {row[('ridge','ystar')]}")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
