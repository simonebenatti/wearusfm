# Conti in forma chiusa (nessun dato, nessuna simulazione) per verificare controlli-1, S-A, S-C.
# Stesso modello del revisore (conti_calibrazione.py): N=sigma^2*c_n, X=sigma^2*p*K/n, n_out=118000, n_in=59000.
import math
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
grid=[1,1.5,2,2.5,3]
print("== (1) Sensibilita' di E_real alla legge di scala (ricevuta storica: alpha 0,01 -> 0,76973; alpha 1e5 -> 0,65612)")
mse_ols20=0.7697254997148791; vary_dyn=0.5253446914952473
for Mstar in (0.62,0.6561):
    ex20=mse_ols20-Mstar
    # (a) formula classica p/(n-p-1) come il revisore; (b) asintotica: eccesso proporzionale a 1/n (raddoppia a 10 utenti)
    ratio=mse_ols20/Mstar-1; neff20=p/ratio+p+1; ex10a=Mstar*p/(neff20/2-p-1); ex10b=2*ex20
    for phi in (0.1,0.2,0.3):
        print(f"  M*={Mstar} phi={phi}: E_real classica {0.786+ex10a*phi/vary_dyn:.3f} | asintotica {0.786+ex10b*phi/vary_dyn:.3f}")
print()
print("== (2) Controllo lineare: soglia di E_real oltre la quale la WLS esatta non passa il 90% (K_dyn,K_tot)=(2.5,5) e (3.5,7)")
def lin_inner(f,s,Kd): N=s*s*c_n; return (N+s*s*p*Kd/n_in)/(f+N)
def lin_outer(f,s,Kd,Kt):
    Xd=s*s*p*Kd/n_out; Xt=s*s*p*Kt/n_out; return 1-Xd/f, 1-Xt
for Kd,Kt in ((2.5,5.0),(3.5,7.0)):
    for f in (0.3,0.4,0.5):
        first=None
        for k in range(780,1060,2):
            Er=k/1000
            cand=[(abs(lin_inner(f,s,Kd)-Er),s) for s in grid if abs(lin_inner(f,s,Kd)-Er)<=0.05]
            if not cand: st='NON CALIBRATO'
            else:
                d,s=min(cand); Rd,Rm=lin_outer(f,s,Kd,Kt); st='passa' if (Rd>=0.9 and Rm>=0.9) else f'FALLISCE(s={s})'
            if st!='passa' and first is None: first=(Er,st)
        print(f"  K=({Kd},{Kt}) f'={f}: primo E_real non 'passa' = {first}")
print()
print("== (3) Una L0 regolarizzata puo' passare dove la WLS non regolarizzata fallisce? (ridge ottima, spettro ipotetico)")
# Eccesso atteso per direzione k: OLS v=sigma^2*K/n; ridge ottima s_k*v/(s_k+v) con s_k = quota di segnale in quella direzione.
def excess(spec,sig,K):
    v=sig*sig*K/n_out; ols=len(spec)*v; ridge=sum(s*v/(s+v) for s in spec); return ols,ridge
iso=[1/384]*384
anis=[0.9/50]*50+[0.1/334]*334
for name,spec in (('isotropo',iso),('anisotropo 50 dir. al 90%',anis)):
    for sig in (2.5,3.0):
        o,r=excess(spec,sig,5.0)
        print(f"  {name}, sigma={sig}, K_tot=5: R_MSE OLS={1-o:.3f}, ridge ottima={1-r:.3f}")
print()
print("== (4) S-A: con E_real ~0,78 quanto margine usa la WLS esatta (modello del revisore)")
for Kd,Kt in ((2.5,5.0),(3.5,7.0)):
    for f in (0.3,0.5):
        cand=[(abs(lin_inner(f,s,Kd)-0.78),s) for s in grid if abs(lin_inner(f,s,Kd)-0.78)<=0.05]
        d,s=min(cand); N=s*s*c_n; Eor=N/(f+N); Rd,Rm=lin_outer(f,s,Kd,Kt); marg=1-Eor
        print(f"  K=({Kd},{Kt}) f'={f}: sigma={s}, margine E_dyn={marg:.3f}, tolleranza 10%={0.1*marg:.3f}, eccesso WLS={marg*(1-Rd):.4f} ({(1-Rd)/0.1:.0%} della tolleranza); R_MSE={Rm:.3f}")
# variabilita' campionaria del recupero (ordine di grandezza): SD relativa dell'eccesso ~ sqrt(2/p) (chi-quadro) e ~2/sqrt(p) (termine incrociato)
print(f"  SD relativa dell'eccesso: chi2 {math.sqrt(2/p):.3f}, termine incrociato ~{2/math.sqrt(p):.3f}")
print()
print("== (5) S-C: quota del margine outer LS->oracolo dovuta all'errore di stima della LS (g_inner=0,08)")
for K in (2,3,5,6,8):
    x_in=p*K/(n_in*c_n); x_out=x_in/2; t=0.08/0.92-x_in
    if t<=0: print(f"  K={K}: guadagno inner tutto da stima"); continue
    print(f"  K={K}: quota stima outer = {x_out/(t+x_out):.2f}; recupero 'gratuito' se una testa istantanea regolarizzata togliesse 25%/50% di quell'eccesso: {0.25*x_out/(t+x_out):.2f}/{0.5*x_out/(t+x_out):.2f}")
print()
print("== (6) Controllo temporale: quota minima r del segnale dinamico di T non leggibile da L0 istantanea")
# r_min (S = a^2 f_T, senza X) = E/((1-E)/g+E); con a=1 e f_L=f_T serve r >= 2*r_min (ignorando X)
for E in (0.79,0.85,0.90,0.95):
    for g in (0.06,0.08):
        rmin=E/((1-E)/g+E); print(f"  E={E} g={g}: R_T/S >= {rmin:.3f}; con a=1 e f_L=f_T serve r >= {2*rmin:.2f} (senza il contributo di X)")
