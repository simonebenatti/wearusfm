# Scansione in forma chiusa (modello del revisore, nessun dato): esito del controllo lineare per E_real e f'.
c_n=0.9372; p=384; n_out=118000.0; n_in=n_out/2; grid=[1,1.5,2,2.5,3]
def lin_inner(f,s,Kd): N=s*s*c_n; return (N+s*s*p*Kd/n_in)/(f+N)
def outcome(Er,f,Kd,Kt):
    cand=[(abs(lin_inner(f,s,Kd)-Er),s) for s in grid if abs(lin_inner(f,s,Kd)-Er)<=0.05]
    if not cand: return 'N'
    d,s=min(cand); Rd=1-s*s*p*Kd/n_out/f; Rm=1-s*s*p*Kt/n_out
    return 'P' if (Rd>=0.9 and Rm>=0.9) else 'F'
for Kd,Kt in ((2.5,5.0),(3.5,7.0)):
    print(f"K=({Kd},{Kt})  E_real da 0,78 a 1,00 a passi di 0,02 (P passa, F WLS esatta sotto 90%, N non calibrato)")
    for f in (0.3,0.4,0.5,0.6):
        row=''.join(outcome(0.78+0.02*i,f,Kd,Kt) for i in range(12))
        print(f"  f'={f}: {row}")
