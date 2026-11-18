"""Quota minima r di T non leggibile da L0 perche' il controllo temporale sia calibrabile, nel modello dei revisori
(work_controlli/conti_temporale_dettaglio.py). Confronta la regola attuale (ancora e banda con la WLS su y) con la tabella della
bozza (ancora E_cal = ridge su y, banda con E_LS* = LS su y*), a parita' di tutto il resto, con sigma fitta (0,5-5 a passi 0,01).
Tolleranza dell'ancora 0,05 e banda [0,06; 0,10] di P:92. Solo aritmetica."""
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/tenuta/bl1_soglia_r.out.txt"
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
    for ag,an in (([1.0],'a=1'),([1.0,1.5,2.0,2.5,3.0],'a fino a 3')):
        r0=rmin(fL,fT,3.0,Er,ag,'wls','y'); r1=rmin(fL,fT,3.0,Er,ag,'ridge','ystar')
        L.append(f"E_real={Er:.2f} f_L=f_T={fL} K=3 {an:10s}: r minimo attuale (WLS/WLS su y) {r0} | bozza (ridge su y / LS su y*) {r1}")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
