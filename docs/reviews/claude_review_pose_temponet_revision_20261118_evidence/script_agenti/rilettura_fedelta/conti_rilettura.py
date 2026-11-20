# Conti di controllo della rilettura (solo aritmetica, nessun dato).
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2
def lin_inner(f,s,Kd): N=s*s*c_n; return (N+s*s*p*Kd/n_in)/(f+N)
print("== BL-1 controesempio (modello del revisore, f'=0,5, K=(2,5;5), E_real=0,92)")
f=0.5; Kd=2.5; Kt=5.0; Er=0.92
for s in (1,1.5,2,2.5,3):
    e=lin_inner(f,s,Kd); print(f"  sigma={s}: E_LS,inner={e:.4f} dist={abs(e-Er):.4f} {'amm.' if abs(e-Er)<=0.05 else ''}")
s=2.5; N=s*s*c_n; Eor=N/(f+N); Xd=s*s*p*Kd/n_out; Ewls=(N+Xd)/(f+N)
print(f"  outer: E_or={Eor:.4f} E_WLS={Ewls:.4f} R_dyn={1-Xd/f:.4f} R_MSE={1-s*s*p*Kt/n_out:.4f}")
print("== BL-1(b): ridge con alpha portato da 10 a 20 utenti senza riscalare (parametrizzazione a loss media: alpha* ~ 1/n)")
# per direzione: segnale s_k, varianza OLS v; ridge con lambda=c*v: shrink s/(s+c v); rischio = s*(c v/(s+c v))^2 + v*(s/(s+c v))^2
def risk(sk,v,c): sh=sk/(sk+c*v); return sk*(1-sh)**2 + v*sh**2
for name,spec in (('isotropo',[1/384]*384),('anisotropo',[0.9/50]*50+[0.1/334]*334)):
    for sig in (2.5,3.0):
        v=sig*sig*5.0/n_out
        ols=len(spec)*v; opt=sum(risk(sk,v,1) for sk in spec); dbl=sum(risk(sk,v,2) for sk in spec)
        print(f"  {name} sigma={sig}: R_MSE OLS={1-ols:.3f} ridge ottima={1-opt:.3f} ridge alpha x2={1-dbl:.3f}")
print("== OB-3 e V-ricetta-criteri-1")
g=lambda ref,tcn: 100*(ref-tcn)/ref
print(f"  gain3 L13=0,7619 TCN=0,750: {g(0.7619,0.750):.3f}; L13=0,735: {g(0.735,0.750):.3f}; 0,735*1,0366={0.735*1.0366:.4f}")
print(f"  gain2 MLP=0,740 TCN=0,750: {g(0.740,0.750):.3f}")
print("== Soglia R-1 e OB-5")
r=0.7859020004638595; h=0.7786289843513977; t=0.98*h
print(f"  0,98*testa={t!r}; sotto la ridge={100*(r-t)/r:.4f}%")
print("== BL-3: 0,7268 contro 0,7269")
print(f"  differenza relativa={100*(0.7269-0.7268)/0.7269:.4f}%")
print("== OB-4, controesempio della lacuna (etichette del §5)")
def lab(gain,lo,up):
    if gain>=2 and lo>0: return 1
    if gain<=-2 and up<0: return 2
    if up<2: return 3
    return 4
print("  contrasto 2 gain -3,0 IC99 [-4,5; -1,5] ->", lab(-3.0,-4.5,-1.5), "(deterioramento_pratico)")
print("  con allarme TCN e regola direzionale la lettura negativa e' bloccata: se 'bloccata' = senza etichetta, il veto (etichetta 2 in 2 o 3) non scatta")
