"""BL-1: che cosa fanno (a)+(b) quando E_real non scende con la ridge (E_real dominata dallo spostamento fra utenti).
Modello dei revisori (work_verifica_architettura-controlli/v_scan_lineare.py; rilettura_fedelta/conti_rilettura.py):
E_dyn inner della WLS = (N + X_in)/(f + N), N = s^2 c_n, X_in = s^2 p Kd / n_in; recupero outer WLS su E_dyn = 1 - X_out/f.
Ridge isotropa all'alpha ottimo (penalita' sulla somma dei quadrati, alpha degli inner = ottimo outer):
eccesso X_r = X f/(f + X) su E_dyn (segnale dinamico f), X_m/(1 + X_m) su MSE (segnale 1).
Solo aritmetica sulle formule degli agenti, nessun dato."""
import math
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/tenuta/bl1_ab_spostamento.out.txt"
L=[]
def wls_inner(f,s2,Kd):
    N=s2*c_n; X=s2*p*Kd/n_in; return (N+X)/(f+N)
def ridge_inner(f,s2,Kd):
    N=s2*c_n; X=s2*p*Kd/n_in; Xr=X*f/(f+X); return (N+Xr)/(f+N)
def solve(fun,f,Er,Kd):
    lo,hi=1e-6,100.0
    for _ in range(200):
        mid=(lo+hi)/2
        if fun(f,mid,Kd)<Er: lo=mid
        else: hi=mid
    return (lo+hi)/2
def rec_wls(f,s2,Kd,Kt):
    Xd=s2*p*Kd/n_out; Xm=s2*p*Kt/n_out; return 1-Xd/f, 1-Xm
def rec_ridge(f,s2,Kd,Kt):
    Xd=s2*p*Kd/n_out; Xm=s2*p*Kt/n_out; return 1-Xd/(f+Xd), 1-Xm/(1+Xm)
for Kd,Kt in ((2.5,5.0),(3.5,7.0)):
  for f in (0.5,):
    for Er in (0.90,0.92,0.93,0.94):
        s2w=solve(wls_inner,f,Er,Kd); s2r=solve(ridge_inner,f,Er,Kd)
        rw=rec_wls(f,s2w,Kd,Kt); rrw=rec_ridge(f,s2w,Kd,Kt)
        rw2=rec_wls(f,s2r,Kd,Kt); rr=rec_ridge(f,s2r,Kd,Kt)
        L.append(f"K=({Kd};{Kt}) f'={f} E_real={Er:.2f} | ancora WLS: sigma {math.sqrt(s2w):.3f}, WLS outer dyn/MSE {rw[0]:.4f}/{rw[1]:.4f}, ridge {rrw[0]:.4f}/{rrw[1]:.4f}"
                 f" | ancora ridge (a) con E_real invariata: sigma {math.sqrt(s2r):.3f}, WLS {rw2[0]:.4f}/{rw2[1]:.4f}, ridge (b) {rr[0]:.4f}/{rr[1]:.4f} -> (b) {'passa' if min(rr)>=0.9 else 'NON DIAGNOSTICO'}")
L.append("Lettura: se la ridge non abbassa E_real (spostamento fra utenti), l'ancora (a) sui target sintetici chiede piu' rumore della WLS,")
L.append("perche' la ridge sintetica ha meno errore di stima: sigma sale e il margine si stringe ancora; (b) decide fra passa e non diagnostico.")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
