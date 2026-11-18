"""Verifica avversaria di T-01 (solo aritmetica sulle formule degli agenti, nessun dato).
Modello del controllo lineare di work_verifica_architettura-controlli/v_scan_lineare.py e ridge isotropa di
controllo/verifica-avvocato-gpt/verifica_conti.py (eccesso ridge X f/(f+X) su E_dyn, X_m/(1+X_m) su MSE).
Domanda: con l'ancora (a) (ridge) invece della WLS, sigma sale o scende? Dipende da quanto la ridge abbassa E_real sui target
veri (Delta_real) rispetto a quanto abbassa E_cal sintetica a pari sigma (Delta_syn). Scrive solo in questa cartella."""
import math
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/verifica-tenuta/v_t01_ancora.out.txt"
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2
L = []

def wls_in(f, s2, Kd):
    N = s2 * c_n; X = s2 * p * Kd / n_in
    return (N + X) / (f + N)

def ridge_in(f, s2, Kd):
    N = s2 * c_n; X = s2 * p * Kd / n_in; Xr = X * f / (f + X)
    return (N + Xr) / (f + N)

def solve(fun, f, E, Kd):
    lo, hi = 1e-6, 100.0
    for _ in range(200):
        m = (lo + hi) / 2
        if fun(f, m, Kd) < E: lo = m
        else: hi = m
    return (lo + hi) / 2

def rec(f, s2, Kd, Kt, ridge):
    Xd = s2 * p * Kd / n_out; Xm = s2 * p * Kt / n_out
    if ridge:
        return min(1 - Xd / (f + Xd), 1 - Xm / (1 + Xm))
    return min(1 - Xd / f, 1 - Xm)

f = 0.5
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for Ew in (0.92, 0.94):
        s2w = solve(wls_in, f, Ew, Kd)
        dsyn = wls_in(f, s2w, Kd) - ridge_in(f, s2w, Kd)
        L.append("K=(%.1f;%.1f) f'=0,5 E_real(WLS)=%.2f: sigma con ancora WLS %.3f; Delta_syn a pari sigma %.4f; ridge (b) %.4f, WLS %.4f"
                 % (Kd, Kt, Ew, math.sqrt(s2w), dsyn, rec(f, s2w, Kd, Kt, True), rec(f, s2w, Kd, Kt, False)))
        for dreal in (0.0, 0.002, dsyn, 0.005, 0.01, 0.02, 0.04):
            Er = Ew - dreal
            s2r = solve(ridge_in, f, Er, Kd)
            rr = rec(f, s2r, Kd, Kt, True)
            L.append("   Delta_real %.4f -> E_real(ridge) %.4f: sigma con ancora ridge %.3f (%s); ridge (b) %.4f -> %s"
                     % (dreal, Er, math.sqrt(s2r), "sale" if s2r > s2w + 1e-12 else ("uguale" if abs(s2r - s2w) < 1e-9 else "scende"),
                        rr, "passa" if rr >= 0.9 else "non diagnostico"))
L.append("")
L.append("Lettura: (a) alza sigma solo se Delta_real < Delta_syn (circa 0,003 di E_dyn nel modello); con Delta_real di 0,005 o piu'")
L.append("sigma scende. Con E_real invariata la ridge di (b) perde 0,36-0,39 punti (K 2,5/5) e 0,4-0,7 (K 3,5/7).")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
