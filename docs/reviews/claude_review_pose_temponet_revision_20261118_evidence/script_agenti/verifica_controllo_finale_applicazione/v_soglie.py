"""Verifica avversaria di P-02 (lente applicazione, controllo 2). Modello in forma chiusa di v_scan_lineare.py, nessun dato.
Per ogni (K, f'): soglia con sigma continua; esito con la griglia attuale {1,1.5,2,2.5,3} a passi fini di E_real;
intervalli in cui griglia e continua danno esiti diversi; soglia effettiva della griglia attuale."""
import math
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/verifica-applicazione/v_soglie.out.txt"
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2
grid = [1, 1.5, 2, 2.5, 3]
def lin_inner(f, s, Kd):
    N = s * s * c_n; return (N + s * s * p * Kd / n_in) / (f + N)
def rec(f, s2, Kd, Kt):
    return 1 - s2 * p * Kd / n_out / f, 1 - s2 * p * Kt / n_out
def s2_cont(f, Er, Kd):
    # lin_inner = E  ->  s^2 = E f / (c_n (1-E) + p Kd / n_in)
    return Er * f / (c_n * (1 - Er) + p * Kd / n_in)
def esito_griglia(Er, f, Kd, Kt):
    cand = [(abs(lin_inner(f, s, Kd) - Er), s) for s in grid if abs(lin_inner(f, s, Kd) - Er) <= 0.05]
    if not cand: return 'N', None, None
    d, s = min(cand); Rd, Rm = rec(f, s * s, Kd, Kt)
    return ('P' if min(Rd, Rm) >= 0.9 else 'F'), s, min(Rd, Rm)
def esito_cont(Er, f, Kd, Kt):
    Rd, Rm = rec(f, s2_cont(f, Er, Kd), Kd, Kt)
    return ('P' if min(Rd, Rm) >= 0.9 else 'F'), min(Rd, Rm)
L = []
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for f in (0.3, 0.4, 0.5, 0.6):
        # soglia continua esatta per bisezione
        lo, hi = 0.80, 0.999
        for _ in range(60):
            mid = (lo + hi) / 2
            if esito_cont(mid, f, Kd, Kt)[0] == 'P': lo = mid
            else: hi = mid
        thr = lo
        # scansione fine
        step = 0.0001; Er = 0.78; prev = None; trans = []; diff = []; cur = None
        while Er <= 1.0:
            g, s, R = esito_griglia(Er, f, Kd, Kt); c, Rc = esito_cont(Er, f, Kd, Kt)
            if g != prev: trans.append((round(Er, 4), g, s)); prev = g
            key = (g, c)
            if g != c and g != 'N':
                if cur is None or cur[2] != key: 
                    if cur: diff.append(cur)
                    cur = [round(Er, 4), round(Er, 4), key]
                else: cur[1] = round(Er, 4)
            else:
                if cur: diff.append(cur); cur = None
            Er += step
        if cur: diff.append(cur)
        L.append("K=(%.1f,%.1f) f'=%.1f  soglia continua %.4f" % (Kd, Kt, f, thr))
        L.append("   transizioni griglia attuale (E_real da 0,78): " + ", ".join("%.4f->%s(s=%s)" % t for t in trans))
        L.append("   griglia != continua (escluso N): " + ("; ".join("[%.4f, %.4f] griglia %s / continua %s" % (a, b, k[0], k[1]) for a, b, k in diff) if diff else "nessuno"))
# casi puntuali citati da P-02
L.append("")
for Kd, Kt, f, Er in ((3.5, 7.0, 0.5, 0.92), (2.5, 5.0, 0.5, 0.92), (3.5, 7.0, 0.3, 0.915), (3.5, 7.0, 0.3, 0.92), (3.5, 7.0, 0.6, 0.895), (3.5, 7.0, 0.6, 0.905), (2.5, 5.0, 0.5, 0.94), (3.5, 7.0, 0.5, 0.94)):
    g, s, R = esito_griglia(Er, f, Kd, Kt); c, Rc = esito_cont(Er, f, Kd, Kt)
    L.append("caso K=(%.1f,%.1f) f'=%.1f E_real=%.3f: griglia s=%s inner=%.4f R=%.4f %s | continua s=%.3f R=%.4f %s" % (
        Kd, Kt, f, Er, s, lin_inner(f, s, Kd), R, g, math.sqrt(s2_cont(f, Er, Kd)), Rc, c))
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
