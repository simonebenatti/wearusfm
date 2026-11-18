# Rifà i numeri del controesempio di BL-1 con il modello di
# work_verifica_architettura-controlli/v_scan_lineare.py (formule ricopiate, nessun dato).
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2; grid = [1, 1.5, 2, 2.5, 3]
def lin_inner(f, s, Kd):
    N = s * s * c_n
    return (N + s * s * p * Kd / n_in) / (f + N)
def oracle(f, s):
    N = s * s * c_n
    return N / (f + N)
def wls_outer(f, s, Kd):
    N = s * s * c_n
    return (N + s * s * p * Kd / n_out) / (f + N)
def choose(Er, f, Kd):
    cand = [(abs(lin_inner(f, s, Kd) - Er), s) for s in grid if abs(lin_inner(f, s, Kd) - Er) <= 0.05]
    return min(cand)[1] if cand else None
def rec(f, s, Kd, Kt):
    Rd = 1 - s * s * p * Kd / n_out / f
    Rm = 1 - s * s * p * Kt / n_out
    return Rd, Rm
print('--- Controesempio della bozza: f=0,5, E_real=0,92 ---')
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    f = 0.5; Er = 0.92
    print(f'K=({Kd},{Kt}): E_LS,inner per sigma:', {s: round(lin_inner(f, s, Kd), 5) for s in grid})
    s = choose(Er, f, Kd)
    Rd, Rm = rec(f, s, Kd, Kt)
    print(f'   sigma scelto {s}; E_LS,inner {lin_inner(f,s,Kd):.5f}; oracolo outer {oracle(f,s):.5f}; WLS outer {wls_outer(f,s,Kd):.5f}')
    print(f'   recupero E_dyn (1-E_wls)/(1-E_or) = {(1-wls_outer(f,s,Kd))/(1-oracle(f,s)):.5f}; Rd formula {Rd:.5f}; Rm {Rm:.5f}')
print('--- Soglia continua di E_real oltre la quale la WLS esatta non passa ---')
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for f in (0.3, 0.4, 0.5, 0.6):
        last_pass = None; first_fail = None
        for i in range(0, 30001):
            Er = 0.70 + i * 1e-5
            s = choose(Er, f, Kd)
            if s is None:
                continue
            Rd, Rm = rec(f, s, Kd, Kt)
            ok = Rd >= 0.9 and Rm >= 0.9
            if ok: last_pass = Er
            elif first_fail is None and last_pass is not None and Er > last_pass:
                first_fail = Er
        print(f'  K=({Kd},{Kt}) f={f}: primo fallimento a E_real ~ {first_fail:.4f}')
print('--- Buco della griglia per f=0,4, K_dyn=2,5 ---')
f = 0.4; Kd = 2.5
vals = [lin_inner(f, s, Kd) for s in grid]
print('  E_LS,inner:', [round(v, 4) for v in vals])
print('  buco fra', round(vals[0] + 0.05, 4), 'e', round(vals[1] - 0.05, 4))
print('  E_real 0,78 e 0,80 ->', choose(0.78, f, Kd), choose(0.80, f, Kd), ' 0,786 ->', choose(0.786, f, Kd))
